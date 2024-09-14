use std::fmt::Debug;

use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks, HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS, HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS, MATRIX_DIAG_8_GOLDILOCKS_U64};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::{random, thread_rng};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};
use p3_poseidon2_air::air::Poseidon2Air;
use p3_poseidon2_air::generation::generate_trace_rows;
use p3_poseidon2_air::{LinearLayer};
use p3_field::{
    AbstractField,
};

const WIDTH: usize = 8;
const SBOX_DEGREE: usize = 7;
const SBOX_REGISTERS: usize = 3;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 22;

const NUM_HASHES: usize = 1<<0;

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
    const POSEIDON_D: u64 = 7;

    type Val = Goldilocks;
    // type Challenge = BinomialExtensionField<Val, 2>;
    //
    type L = LinearLayer<WIDTH>;
    //
    //
    // type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, WIDTH, POSEIDON_D>;
    // let perm = Perm::new_from_rng_128(
    //     Poseidon2ExternalMatrixGeneral,
    //     DiffusionMatrixGoldilocks::default(),
    //     &mut thread_rng(),
    // );
    //
    // type MyHash = PaddingFreeSponge<Perm, WIDTH, 8, 8>;
    // let hash = MyHash::new(perm.clone());
    //
    // type MyCompress = TruncatedPermutation<Perm, 2, 8, WIDTH>;
    // let compress = MyCompress::new(perm.clone());
    //
    // type ValMmcs = FieldMerkleTreeMmcs<
    //     <Val as Field>::Packing,
    //     <Val as Field>::Packing,
    //     MyHash,
    //     MyCompress,
    //     8,
    // >;
    // let val_mmcs = ValMmcs::new(hash, compress);

    // type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    // let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // type Dft = Radix2DitParallel;
    // let dft = Dft {};

    // type Challenger = DuplexChallenger<Val, Perm, WIDTH, 8>;
    //         beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    let mut beginning: [[Val;8]; HALF_FULL_ROUNDS] = [[Val::zero(); 8]; HALF_FULL_ROUNDS];
    let mut end: [[Val;8]; HALF_FULL_ROUNDS] = [[Val::zero(); 8]; HALF_FULL_ROUNDS];

    let full_round_constants = HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS.iter().map(|arr| {
        arr.map(|c| Val::from_canonical_u64(c))
    }).collect::<Vec<_>>();

    for i in 0..4 {
        beginning[i] = full_round_constants[i];
    }

    for i in 0..4 {
        end[i] = full_round_constants[i+4];
    }

    // let beginning_full: [[F; WIDTH]; HALF_FULL_ROUNDS] =

    let  partial_constants = HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS.into_iter().map(|value| {
         Val::from_canonical_u64(value)
    }).collect::<Vec<_>>();
    let mut partial: [Val; PARTIAL_ROUNDS] = [Val::zero(); PARTIAL_ROUNDS];
    for i in 0..PARTIAL_ROUNDS {
        partial[i] = partial_constants[i];
    }

    let internal_constants = MATRIX_DIAG_8_GOLDILOCKS_U64.into_iter().map(|value| {
        Val::from_canonical_u64(value)
    }).collect::<Vec<_>>();
    let mut internal: [Val; WIDTH] = [Val::zero(); WIDTH];
    for i in 0..WIDTH {
        internal[i] = internal_constants[i];
    }
    let air: Poseidon2Air<
        Val,
        L,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    > = Poseidon2Air::new_from_rng(
        &mut thread_rng(),
        beginning,
        partial,
        end,
        internal,
    );


    Val::from_wrapped_u64(0);
    // Vec<[F; WIDTH]>

    let mut input: [Val; 8] = [
        5116996373749832116,
        8931548647907683339,
        17132360229780760684,
        11280040044015983889,
        11957737519043010992,
        15695650327991256125,
        17604752143022812942,
        543194415197607509,
    ].map(Val::from_wrapped_u64);
    let inputs =  (0..NUM_HASHES).map(|_| input ).collect::<Vec<_>>();

    let trace = generate_trace_rows::<
        Val,
        L,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(inputs,
      air.beginning_full_round_constants,
      air.partial_round_constants,
      air.ending_full_round_constants,
      air.internal_matrix_diagonal,
    );


    // let fri_config = FriConfig {
    //     log_blowup: 1,
    //     num_queries: 100,
    //     proof_of_work_bits: 16,
    //     mmcs: challenge_mmcs,
    // };
    // type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    // let pcs = Pcs::new(dft, val_mmcs, fri_config);
    //
    // type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    // let config = MyConfig::new(pcs);
    //
    //
    // let mut challenger = Challenger::new(perm.clone());
    // let proof = prove(&config, &air, &mut challenger, trace, &vec![]);
    //
    // let mut challenger = Challenger::new(perm);
    // verify(&config, &air, &mut challenger, &proof, &vec![])
}
