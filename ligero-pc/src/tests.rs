// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::LigeroCommit;

use ff::Field;
use ft::*;
use itertools::iterate;
use merlin::Transcript;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha3::Sha3_256;
use std::iter::repeat_with;

mod ft {
    use ff::PrimeField;
    use lcpc2d::FieldHash;
    use serde::Serialize;

    #[derive(PrimeField, Serialize)]
    #[PrimeFieldModulus = "70386805592835581672624750593"]
    #[PrimeFieldGenerator = "17"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft([u64; 2]);

    impl FieldHash for Ft {
        type HashRepr = <Ft as PrimeField>::Repr;

        fn to_hash_repr(&self) -> Self::HashRepr {
            PrimeField::to_repr(self)
        }
    }
}

#[test]
fn end_to_end() {
    // commit to a random polynomial at a random rate
    let (coeffs, rho) = random_coeffs_rho();
    let n_degree_tests = 2;
    let n_col_opens = 128usize;
    let comm = LigeroCommit::<Sha3_256, _>::commit(&coeffs, rho).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root().unwrap();

    // evaluate the random polynomial we just generated at a random point x
    let x = Ft::random(&mut rand::thread_rng());

    // compute the outer and inner tensors for powers of x
    // NOTE: we treat coeffs as a univariate polynomial, but it doesn't
    // really matter --- the only difference from a multilinear is the
    // way we compute outer_tensor and inner_tensor from the eval point
    let inner_tensor: Vec<Ft> = iterate(Ft::one(), |&v| v * x)
        .take(comm.get_n_per_row())
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft::one(), |&v| v * xr)
            .take(comm.get_n_rows())
            .collect()
    };

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let pf = comm
        .prove(&outer_tensor[..], n_degree_tests, n_col_opens, &mut tr1)
        .unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    pf.verify(
        &root,
        &outer_tensor[..],
        &inner_tensor[..],
        rho,
        n_degree_tests,
        n_col_opens,
        &mut tr2,
    )
    .unwrap();
}

#[test]
fn end_to_end_two_proofs() {
    // commit to a random polynomial at a random rate
    let (coeffs, rho) = random_coeffs_rho();
    let n_degree_tests = 1;
    let n_col_opens = 128usize;
    let comm = LigeroCommit::<Sha3_256, _>::commit(&coeffs, rho).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root().unwrap();

    // evaluate the random polynomial we just generated at a random point x
    let x = Ft::random(&mut rand::thread_rng());

    // compute the outer and inner tensors for powers of x
    // NOTE: we treat coeffs as a univariate polynomial, but it doesn't
    // really matter --- the only difference from a multilinear is the
    // way we compute outer_tensor and inner_tensor from the eval point
    let inner_tensor: Vec<Ft> = iterate(Ft::one(), |&v| v * x)
        .take(comm.get_n_per_row())
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft::one(), |&v| v * xr)
            .take(comm.get_n_rows())
            .collect()
    };

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let pf = comm
        .prove(&outer_tensor[..], n_degree_tests, n_col_opens, &mut tr1)
        .unwrap();

    let challenge_after_first_proof_prover = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr1.challenge_bytes(b"ligero-pc//challenge", &mut key);
        let mut deg_test_rng = ChaCha20Rng::from_seed(key);
        Ft::random(&mut deg_test_rng)
    };

    // produce a second proof with the same transcript
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let pf2 = comm
        .prove(&outer_tensor[..], n_degree_tests, n_col_opens, &mut tr1)
        .unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let res = pf
        .verify(
            &root,
            &outer_tensor[..],
            &inner_tensor[..],
            rho,
            n_degree_tests,
            n_col_opens,
            &mut tr2,
        )
        .unwrap();

    let challenge_after_first_proof_verifier = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr2.challenge_bytes(b"ligero-pc//challenge", &mut key);
        let mut deg_test_rng = ChaCha20Rng::from_seed(key);
        Ft::random(&mut deg_test_rng)
    };
    assert_eq!(
        challenge_after_first_proof_prover,
        challenge_after_first_proof_verifier
    );

    // second proof verification with the same transcript
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let res2 = pf2
        .verify(
            &root,
            &outer_tensor[..],
            &inner_tensor[..],
            rho,
            n_degree_tests,
            n_col_opens,
            &mut tr2,
        )
        .unwrap();

    assert_eq!(res, res2);
}

fn random_coeffs_rho() -> (Vec<Ft>, f64) {
    let mut rng = rand::thread_rng();

    let lgl = 8 + rng.gen::<usize>() % 8;
    let len_base = 1 << (lgl - 1);
    let len = len_base + (rng.gen::<usize>() % len_base);

    (
        repeat_with(|| Ft::random(&mut rng)).take(len).collect(),
        rng.gen_range(0.1f64..0.9f64),
    )
}
