use std::fmt::Debug;

use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
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

const NUM_HASHES: usize = 1<<1;

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
    
    
    
    let air: Poseidon2Air<
        Val,
        L,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    > = Poseidon2Air::new_from_rng(&mut thread_rng());


    Val::from_wrapped_u64(0);
    // Vec<[F; WIDTH]>
    let inputs = (0..NUM_HASHES).map(|_| [Goldilocks::zero(); WIDTH] ).collect::<Vec<_>>();

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
