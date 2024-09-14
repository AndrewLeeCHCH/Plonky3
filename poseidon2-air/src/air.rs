use alloc::vec::Vec;
use core::borrow::Borrow;
use core::marker::PhantomData;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::columns::{num_cols, Poseidon2Cols, FullRound, PartialRound, SBox};
use crate::PermutationLinearLayer;

/// Poseidon2 Air
#[derive(Debug)]
pub struct Poseidon2Air<
    F: Field,
    L: PermutationLinearLayer,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    /// Beginning full round constants
    pub beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    /// Partial round constants
    pub partial_round_constants: [F; PARTIAL_ROUNDS],
    /// Ending full round constants
    pub ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    /// Internal matrix diagonal
    pub internal_matrix_diagonal: [F; WIDTH],
    _marker: PhantomData<L>,
}

impl<
        F: Field,
        L: PermutationLinearLayer,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Poseidon2Air<F, L, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    /// Generate Poseidon2
    pub fn new_from_rng<R: Rng>(
        rng: &mut R,
        beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
        partial_round_constants: [F; PARTIAL_ROUNDS],
        ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
        internal_matrix_diagonal: [F; WIDTH],
    ) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        // let beginning_full_round_constants = rng
        //     .sample_iter(Standard)
        //     .take(HALF_FULL_ROUNDS)
        //     .collect::<Vec<[F; WIDTH]>>()
        //     .try_into()
        //     .unwrap();
        // // beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
        // // let beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS] = [
        // //     [F::from_wrapped_u32(1); WIDTH]; HALF_FULL_ROUNDS
        // // ];
        // let partial_round_constants = rng
        //     .sample_iter(Standard)
        //     .take(PARTIAL_ROUNDS)
        //     .collect::<Vec<F>>()
        //     .try_into()
        //     .unwrap();
        // // let partial_round_constants: [F; PARTIAL_ROUNDS] = [
        // //     F::zero(); PARTIAL_ROUNDS
        // // ];
        // let ending_full_round_constants = rng
        //     .sample_iter(Standard)
        //     .take(HALF_FULL_ROUNDS)
        //     .collect::<Vec<[F; WIDTH]>>()
        //     .try_into()
        //     .unwrap();
        // // let ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS] = [
        // //     [F::zero(); WIDTH]; HALF_FULL_ROUNDS
        // // ];
        // let internal_matrix_diagonal = rng
        //     .sample_iter(Standard)
        //     .take(WIDTH)
        //     .collect::<Vec<F>>()
        //     .try_into()
        //     .unwrap();
        // // let internal_matrix_diagonal: [F; WIDTH]  = [
        // //     F::one(); WIDTH
        // // ];
        Self {
            beginning_full_round_constants,
            partial_round_constants,
            ending_full_round_constants,
            internal_matrix_diagonal,
            _marker: PhantomData
        }
    }
}

impl<
        F: Field,
        L:PermutationLinearLayer + core::marker::Sync,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > BaseAir<F>
    for Poseidon2Air<F, L, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    where L:PermutationLinearLayer,
{
    fn width(&self) -> usize {
        num_cols::<L, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }
}

impl<
        AB: AirBuilder,
        L,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > Air<AB>
    for Poseidon2Air<AB::F, L, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
    where L: PermutationLinearLayer + core::marker::Sync
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2Cols<
            AB::Var,
            L,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = (*local).borrow();

        let mut state: [AB::Expr; WIDTH] = local.inputs.map(|x| x.into());

        assert_eq!(
            L::WIDTH,
            WIDTH,
            "The WIDTH for this STARK does not match the Linear Layer WIDTH."
        );

        // state.clone().into_iter().for_each(|input| {
        //     builder.assert_zero(input)
        // });
        L::matmul_external(&mut state);
        for round in 0..HALF_FULL_ROUNDS {
            println!("eval bg full {round}");
            eval_full_round(
                &mut state,
                &local.beginning_full_rounds[round],
                &self.beginning_full_round_constants[round],
                builder,
            );
        }

        for round in 0..PARTIAL_ROUNDS {
            println!("eval partial {round}");
            eval_partial_round(
                &mut state,
                &local.partial_rounds[round],
                &self.partial_round_constants[round],
                &self.internal_matrix_diagonal,
                builder,
            );
        }
        //
        for round in 0..HALF_FULL_ROUNDS {
            println!("eval end full {round}");
            eval_full_round(
                &mut state,
                &local.ending_full_rounds[round],
                &self.ending_full_round_constants[round],
                builder,
            );
        }
    }
}

#[inline]
fn eval_full_round<
    AB: AirBuilder,
    L,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    full_round: &FullRound<AB::Var, L, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[AB::F; WIDTH],
    builder: &mut AB,
) where L: PermutationLinearLayer {
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s = s.clone() + *r;
        eval_sbox(&full_round.sbox[i], s, builder);
    }

    L::matmul_external(state);
}

#[inline]
fn eval_partial_round<
    AB: AirBuilder,
    L,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var, L, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: &AB::F,
    internal_matrix_diagonal: &[AB::F; WIDTH],
    builder: &mut AB,
) where L: PermutationLinearLayer  {
    state[0] = state[0].clone() + *round_constant;
    eval_sbox(&partial_round.sbox, &mut state[0], builder);
    let internal_matrix_diagonal_expr = internal_matrix_diagonal.iter().map(|imd| {
        AB::Expr::from(*imd)
    }).collect::<Vec<_>>();
    L::matmul_internal(state, internal_matrix_diagonal_expr.as_slice());
}

/// Evaluates the S-BOX over a degree-`1` expression `x`.
///
/// # Panics
///
/// This method panics if the number of `REGISTERS` is not chosen optimally for the given
/// `DEGREE` or if the `DEGREE` is not supported by the S-BOX. The supported degrees are
/// `3`, `5`, `7`, and `11`.
///
/// # Efficiency Note
///
/// This method computes the S-BOX by computing the cube of `x` and then successively
/// multiplying the running sum by the cube of `x` until the last multiplication where we use
/// the appropriate power to reach the final product:
///
/// ```text
/// (x^3) * (x^3) * ... * (x^k) where k = d mod 3
/// ```
///
/// The intermediate powers are stored in the auxiliary column registers. To maximize the
/// efficiency of the registers we try to do three multiplications per round. This algorithm
/// only multiplies the cube of `x` but a more optimal product would be to find the base-3
/// decomposition of the `DEGREE` and use that to generate the addition chain. Even this is not
/// the optimal number of multiplications for all possible degrees, but for the S-BOX powers we
/// are interested in for Poseidon2 (namely `3`, `5`, `7`, and `11`), we get the optimal number
/// with this algorithm. We use the following register table:
///
/// | `DEGREE` | `REGISTERS` |
/// |:--------:|:-----------:|
/// | `3`      | `1`         |
/// | `5`      | `2`         |
/// | `7`      | `3`         |
/// | `11`     | `3`         |
///
/// We record this table in [`Self::OPTIMAL_REGISTER_COUNT`] and this choice of registers is
/// enforced by this method.
#[inline]
fn eval_sbox<AB, const DEGREE: usize, const REGISTERS: usize>(
    sbox: &SBox<AB::Var, DEGREE, REGISTERS>,
    x: &mut AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    assert_ne!(REGISTERS, 0, "The number of REGISTERS must be positive.");
    assert!(DEGREE <= 11, "The DEGREE must be less than or equal to 11.");
    // assert_eq!(
    //     REGISTERS,
    //     Self::OPTIMAL_REGISTER_COUNT[DEGREE],
    //     "The number of REGISTERS must be optimal for the given DEGREE."
    // );

    let x2 = x.square();
    let x3 = x2.clone() * x.clone();
    load(sbox, 0, x3.clone(), builder);
    if REGISTERS == 1 {
        *x = sbox.0[0].into();
        return;
    }
    if DEGREE == 11 {
        (1..REGISTERS - 1).for_each(|j| load_product(sbox, j, &[0, 0, j - 1], builder));
    } else {
        (1..REGISTERS - 1).for_each(|j| load_product(sbox, j, &[0, j - 1], builder));
    }
    load_last_product(sbox, x.clone(), x2, x3, builder);
    *x = sbox.0[REGISTERS - 1].into();
}

/// Loads `value` into the `i`-th S-BOX register.
#[inline]
fn load<AB, const SBOX_DEGREE: usize, const SBOX_REGISTERS: usize>(
    sbox: &SBox<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    i: usize,
    value: AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    builder.assert_eq(sbox.0[i].into(), value);
}

/// Loads the product over all `product` indices the into the `i`-th S-BOX register.
#[inline]
fn load_product<AB, const SBOX_DEGREE: usize, const SBOX_REGISTERS: usize>(
    sbox: &SBox<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    i: usize,
    product: &[usize],
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    // assert!(
    //     product.len() <= 3,
    //     "Product is too big. We can only compute at most degree-3 constraints."
    // );
    load(
        sbox,
        i,
        product.iter().map(|j| sbox.0[*j].into()).product(),
        builder,
    );
}

/// Loads the final product into the last S-BOX register. The final term in the product is
/// `pow(x, DEGREE % 3)`.
#[inline]
fn load_last_product<AB, const SBOX_DEGREE: usize, const SBOX_REGISTERS: usize>(
    sbox: &SBox<AB::Var, SBOX_DEGREE, SBOX_REGISTERS>,
    x: AB::Expr,
    x2: AB::Expr,
    x3: AB::Expr,
    builder: &mut AB,
) where
    AB: AirBuilder,
{
    load(
        sbox,
        SBOX_REGISTERS - 1,
        [x3, x, x2][SBOX_DEGREE % 3].clone() * sbox.0[SBOX_REGISTERS - 2].into(),
        builder,
    );
}