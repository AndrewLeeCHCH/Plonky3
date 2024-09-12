use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;
use std::marker::PhantomData;
use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::columns::{num_cols, FullRound, Poseidon2Cols, SBox};
use crate::{LinearLayer, Permutation, PermutationLinearLayer};

/// Generate trace
#[instrument(name = "generate Poseidon2 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField,
    L: PermutationLinearLayer,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    beginning_full_rounds_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    partial_round_constants: [F; PARTIAL_ROUNDS],
    ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    internal_matrix_diagonal: [F; WIDTH],
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );
    println!("generate rows");
    let ncols = num_cols::<L, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
    let mut trace = RowMajorMatrix::new(vec![F::zero(); n * ncols], ncols);
    let (prefix, rows, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            F,
            L,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows.iter_mut().zip(inputs).for_each(|(row, input)| {
        generate_trace_rows_for_perm(row, input, beginning_full_rounds_constants, partial_round_constants, ending_full_round_constants, internal_matrix_diagonal);
    });

    trace
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
fn generate_trace_rows_for_perm<
    F: PrimeField,
    L: PermutationLinearLayer,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,

>(
    row: &mut Poseidon2Cols<
        F,
        L,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    input: [F; WIDTH],
    beginning_full_rounds_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    partial_round_constants: [F; PARTIAL_ROUNDS],
    ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    internal_matrix_diagonal: [F; WIDTH],
) {
    row.inputs = input;
    // let mut current_state = input.clone();

    matmul_external(WIDTH, &mut row.inputs);

    for r in 0..HALF_FULL_ROUNDS {
        add_full_round_constants(&mut row.inputs, &beginning_full_rounds_constants[r]);

        let mut sbox_generated: [SBox<F, SBOX_DEGREE, SBOX_REGISTERS>; WIDTH] = SBox::default();
        for i in 0..WIDTH {
            let input = row.inputs[i].clone();
            sbox_generated[i] = generateSBox(input);
        }

        row.beginning_full_rounds[r].sbox = sbox_generated;
        sbox(SBOX_DEGREE, &mut row.inputs);
        matmul_external(WIDTH, &mut row.inputs);
    }

    for r in 0..PARTIAL_ROUNDS {
        add_partial_round_constant(&mut row.inputs, &partial_round_constants[r]);
        let input0 = row.inputs[0].clone();
        row.partial_rounds[r].sbox = generateSBox(input0);
        sbox_p(SBOX_DEGREE, &mut row.inputs[0]);
        matmul_internal(WIDTH, &mut row.inputs, &internal_matrix_diagonal);
    }

    for r in 0..HALF_FULL_ROUNDS {
        add_full_round_constants(&mut row.inputs, &ending_full_round_constants[r]);

        let mut sbox_generated: [SBox<F, SBOX_DEGREE, SBOX_REGISTERS>; WIDTH] = SBox::default();
        for i in 0..WIDTH {
            let input = row.inputs[i].clone();
            sbox_generated[i] = generateSBox(input);
        }
        row.ending_full_rounds[r].sbox = sbox_generated;

        sbox(SBOX_DEGREE, &mut row.inputs);
        matmul_external(WIDTH, &mut row.inputs);
    }
}

fn add_full_round_constants<F: PrimeField>(
    state: &mut [F],
    round_constants: &[F],
) {
    for (a, b) in state.iter_mut().zip(round_constants.iter()) {
        a.add_assign(b.clone());
    }
}

fn add_partial_round_constant<F: PrimeField>(state: &mut [F], constant: &F) {
    state[0].add_assign(constant.clone());
}

fn matmul_external<
    F: PrimeField,
>(width: usize, input: &mut[F]) {
    match width {
        2 => {
            let mut sum = input[0].clone();
            sum.add_assign(input[1].clone());
            input[0].add_assign(sum.clone());
            input[1].add_assign(sum);
        }
        3 => {
            // Matrix circ(2, 1, 1)
            let mut sum = input[0];
            sum.add_assign(input[1]);
            sum.add_assign(input[2]);
            input[0].add_assign(sum.clone());
            input[1].add_assign(sum.clone());
            input[2].add_assign(sum);
        }
        4 => {

        }
        8 | 12 | 16 | 20 | 24 => {
            // Applying cheap 4x4 MDS matrix to each 4-element part of the state
            matmul_m4(width, input);
            //
            // Applying second cheap matrix for t > 4
            let t4 = width / 4;
            let mut stored = [F::zero(); 4];
            for l in 0..4 {
                stored[l] = input[l];
                for j in 1..t4 {
                    stored[l].add_assign(input[4 * j + l].clone());
                }
            }
            for i in 0..input.len() {
                input[i].add_assign(stored[i % 4].clone());
            }
        }
        _ => {
            panic!("unsupported width: {width}")
        }
    }
}

fn matmul_m4<
    F: PrimeField,
>(width: usize, state: &mut[F]) {
    let t4 = width / 4;
    for i in 0..t4 {
        let start_index = i * 4;
        let mut t_0 = state[start_index].clone();
        t_0.add_assign(state[start_index + 1].clone());
        let mut t_1 = state[start_index + 2].clone();
        t_1.add_assign(state[start_index + 3].clone());
        let mut t_2 = state[start_index + 1].clone();
        t_2.add_assign(t_2.clone());
        t_2.add_assign(t_1.clone());
        let mut t_3 = state[start_index + 3].clone();
        t_3.add_assign(t_3.clone());
        t_3.add_assign(t_0.clone());
        let mut t_4 = t_1.clone();
        t_4.add_assign(t_4.clone());
        t_4.add_assign(t_4.clone());
        t_4.add_assign(t_3.clone());
        let mut t_5 = t_0.clone();
        t_5.add_assign(t_5.clone());
        t_5.add_assign(t_5.clone());
        t_5.add_assign(t_2.clone());
        let mut t_6 = t_3.clone();
        t_6.add_assign(t_5.clone());
        let mut t_7 = t_2.clone();
        t_7.add_assign(t_4.clone());
        state[start_index] = t_6.clone();
        state[start_index + 1] = t_5.clone();
        state[start_index + 2] = t_7.clone();
        state[start_index + 3] = t_4.clone();
    }
}

fn matmul_internal<
    F: PrimeField,
>(width: usize, input: &mut[F], internal_matrix_diagonal: &[F]) {
    match width {
        2 => {
            let mut sum = input[0];
            sum.add_assign(input[1].clone());
            input[0].add_assign(sum.clone());
            input[1].add_assign(input[1].clone());
            input[1].add_assign(sum);
        }
        3 => {
            // [2, 1, 1]
            // [1, 2, 1]
            // [1, 1, 3]
            let mut sum = input[0];
            sum.add_assign(input[1].clone());
            sum.add_assign(input[2].clone());
            input[0].add_assign(sum.clone());
            input[1].add_assign(sum.clone());
            input[2].add_assign(input[2].clone());
            input[2].add_assign(sum);
        }
        4 | 8 | 12 | 16 | 20 | 24 => {
            // Compute input sum
            let mut sum = input[0].clone();
            input
                .iter()
                .skip(1)
                .take(width-1)
                .for_each(|el| sum.add_assign(el.clone()));
            // Add sum + diag entry * element to each element
            for i in 0..input.len() {
                input[i].mul_assign(internal_matrix_diagonal[i].clone());
                input[i].add_assign(sum.clone());
            }
        }
        _ => {
            panic!()
        }
    }
}

fn sbox<F: PrimeField>(degree: usize, input: &mut [F]) {
    input.iter_mut().for_each(|t|  sbox_p(degree, t));
}

fn sbox_p<F: PrimeField>(degree: usize, input: &mut F) {
    // let mut input2 = *input;
    // input2.square_in_place();

    let mut input2 = input.clone();
    input2.mul_assign(input.clone());
    // x.mul_assign(x2);

    match degree {
        3 => {
            input.mul_assign(input2);
        }
        5 => {
            let mut input4 = input2;
            input4.mul_assign(input2.clone());
            input.mul_assign(input4);
        }
        7 => {
            let mut input6 = input2;
            input6.mul_assign(input2.clone());
            input6.mul_assign(input2.clone());
            input.mul_assign(input6)
        }
        _ => {
            panic!()
        }
    }
}

fn generateSBox<
    F: PrimeField,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize, >(input: &F) -> SBox<F, SBOX_DEGREE, SBOX_REGISTERS> {

    // let mut sbox = SBox<F, SBOX_DEGREE, SBOX_REGISTERS>{};
    let sbox = SBox::default();
    let mut x = input.clone();
    let mut x2 = input.clone();
    x2.mul_assign(input.clone());
    let mut x3 = x2.clone();
    x3.mul_assign(input.clone());

    sbox[0] = x3;

    // if DEGREE == 11 {
    //     (1..REGISTERS - 1).for_each(|j| crate::air::load_product(sbox, j, &[0, 0, j - 1], builder));
    // } else {
    //     (1..REGISTERS - 1).for_each(|j| crate::air::load_product(sbox, j, &[0, j - 1], builder));
    // }
    // crate::air::load_last_product(sbox, x.clone(), x2, x3, builder);
    // *x = sbox.0[REGISTERS - 1].into();

    for i in 1..SBOX_REGISTERS - 1 {
        // if (SBOX_DEGREE == 11) {
        //
        // } else {
        //
        // }

    }

    // let mut finalValue = [x3, x, x2][SBOX_DEGREE % 3].clone();
    // let s: F = sbox[SBOX_REGISTERS - 2];
    // sbox[SBOX_REGISTERS - 1] =  finalValue.mul_assign(s);

    sbox
}


