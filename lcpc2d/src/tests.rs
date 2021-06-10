// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of ligero-pc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::{LigeroCommit, LigeroEncoding};

use digest::Output;
use err_derive::Error;
use ff::Field;
use fffft::FieldFFT;
use ft::*;
use itertools::iterate;
use merlin::Transcript;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha3::Sha3_256;
use std::iter::repeat_with;

#[derive(Debug, Error)]
enum DummyError {}

mod ft {
    use crate::FieldHash;
    use ff::PrimeField;
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
fn log2() {
    use super::log2;

    for idx in 0..31 {
        assert_eq!(log2(1usize << idx), idx);
    }
}

#[test]
fn get_dims() {
    use super::get_dims;

    let mut rng = rand::thread_rng();

    for _ in 0..128 {
        let lgl = 8 + rng.gen::<usize>() % 8;
        for _ in 0..128 {
            let len_base = 1 << (lgl - 1);
            let len = len_base + (rng.gen::<usize>() % len_base);
            let rho = rng.gen_range(0.001f64..1f64);
            let (n_rows, n_per_row, n_cols) = get_dims::<DummyError>(len, rho).unwrap();
            assert!(n_rows * n_per_row >= len);
            assert!((n_rows - 1) * n_per_row < len);
            assert!(n_per_row as f64 / rho <= n_cols as f64);
        }
    }
}

#[test]
fn merkleize() {
    use super::{merkleize, merkleize_ser};

    let mut test_comm = random_comm();
    let mut test_comm_2 = test_comm.clone();

    merkleize(&mut test_comm).unwrap();
    merkleize_ser(&mut test_comm_2).unwrap();

    assert_eq!(&test_comm.comm, &test_comm_2.comm);
    assert_eq!(&test_comm.coeffs, &test_comm_2.coeffs);
    assert_eq!(&test_comm.hashes, &test_comm_2.hashes);
}

#[test]
fn eval_outer() {
    use super::{eval_outer, eval_outer_ser};

    let test_comm = random_comm();
    let mut rng = rand::thread_rng();
    let tensor: Vec<Ft> = repeat_with(|| Ft::random(&mut rng))
        .take(test_comm.n_rows)
        .collect();

    let res1 = eval_outer(&test_comm, &tensor[..]).unwrap();
    let res2 = eval_outer_ser(&test_comm, &tensor[..]).unwrap();

    assert_eq!(&res1[..], &res2[..]);
}

#[test]
fn open_column() {
    use super::{merkleize, open_column, verify_column};

    let mut rng = rand::thread_rng();

    let test_comm = {
        let mut tmp = random_comm();
        merkleize(&mut tmp).unwrap();
        tmp
    };

    let root = test_comm.get_root().unwrap();
    for _ in 0..64 {
        let col_num = rng.gen::<usize>() % test_comm.n_cols;
        let column = open_column(&test_comm, col_num).unwrap();
        assert!(verify_column::<Sha3_256, _>(
            &column,
            col_num,
            &root,
            &[],
            &Ft::zero(),
        ));
    }
}

#[test]
fn commit() {
    use super::{commit, eval_outer, eval_outer_fft};

    let (coeffs, rho) = random_coeffs_rho();
    let comm = commit::<Sha3_256, LigeroEncoding<_>>(&coeffs, rho, 1usize, 128usize).unwrap();

    let x = Ft::random(&mut rand::thread_rng());

    let eval = comm
        .coeffs
        .iter()
        .zip(iterate(Ft::one(), |&v| v * x).take(coeffs.len()))
        .fold(Ft::zero(), |acc, (c, r)| acc + *c * r);

    let roots_lo: Vec<Ft> = iterate(Ft::one(), |&v| v * x)
        .take(comm.n_per_row)
        .collect();
    let roots_hi: Vec<Ft> = {
        let xr = x * roots_lo.last().unwrap();
        iterate(Ft::one(), |&v| v * xr).take(comm.n_rows).collect()
    };
    let coeffs_flattened = eval_outer(&comm, &roots_hi[..]).unwrap();
    let eval2 = coeffs_flattened
        .iter()
        .zip(roots_lo.iter())
        .fold(Ft::zero(), |acc, (c, r)| acc + *c * r);
    assert_eq!(eval, eval2);

    let mut poly_fft = eval_outer_fft(&comm, &roots_hi[..]).unwrap();
    <Ft as FieldFFT>::ifft_oi(&mut poly_fft).unwrap();
    assert!(poly_fft
        .iter()
        .skip(comm.n_per_row)
        .all(|&v| v == Ft::zero()));
    let eval3 = poly_fft
        .iter()
        .zip(roots_lo.iter())
        .fold(Ft::zero(), |acc, (c, r)| acc + *c * r);
    assert_eq!(eval2, eval3);
}

#[test]
fn end_to_end() {
    use super::{commit, prove, verify};

    // commit to a random polynomial at a random rate
    let (coeffs, rho) = random_coeffs_rho();
    let n_degree_tests = 2;
    let n_col_opens = 128usize;
    let comm =
        commit::<Sha3_256, LigeroEncoding<_>>(&coeffs, rho, n_degree_tests, n_col_opens).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root().unwrap();

    // evaluate the random polynomial we just generated at a random point x
    let x = Ft::random(&mut rand::thread_rng());
    let eval = comm
        .coeffs
        .iter()
        .zip(iterate(Ft::one(), |&v| v * x).take(coeffs.len()))
        .fold(Ft::zero(), |acc, (c, r)| acc + *c * r);

    // compute the outer and inner tensors for powers of x
    // NOTE: we treat coeffs as a univariate polynomial, but it doesn't
    // really matter --- the only difference from a multilinear is the
    // way we compute outer_tensor and inner_tensor from the eval point
    let inner_tensor: Vec<Ft> = iterate(Ft::one(), |&v| v * x)
        .take(comm.n_per_row)
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft::one(), |&v| v * xr).take(comm.n_rows).collect()
    };

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let pf = prove::<Sha3_256, _>(&comm, &outer_tensor[..], &mut tr1).unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let res = verify(
        &root,
        &outer_tensor[..],
        &inner_tensor[..],
        &pf,
        rho,
        n_degree_tests,
        n_col_opens,
        &mut tr2,
    )
    .unwrap();

    assert_eq!(res, eval);
}

#[test]
fn end_to_end_two_proofs() {
    use super::{commit, prove, verify};

    // commit to a random polynomial at a random rate
    let (coeffs, rho) = random_coeffs_rho();
    let n_degree_tests = 1;
    let n_col_opens = 128usize;
    let comm =
        commit::<Sha3_256, LigeroEncoding<_>>(&coeffs, rho, n_degree_tests, n_col_opens).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root().unwrap();

    // evaluate the random polynomial we just generated at a random point x
    let x = Ft::random(&mut rand::thread_rng());
    let eval = comm
        .coeffs
        .iter()
        .zip(iterate(Ft::one(), |&v| v * x).take(coeffs.len()))
        .fold(Ft::zero(), |acc, (c, r)| acc + *c * r);

    // compute the outer and inner tensors for powers of x
    // NOTE: we treat coeffs as a univariate polynomial, but it doesn't
    // really matter --- the only difference from a multilinear is the
    // way we compute outer_tensor and inner_tensor from the eval point
    let inner_tensor: Vec<Ft> = iterate(Ft::one(), |&v| v * x)
        .take(comm.n_per_row)
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft::one(), |&v| v * xr).take(comm.n_rows).collect()
    };

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let pf = prove::<Sha3_256, _>(&comm, &outer_tensor[..], &mut tr1).unwrap();

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
    let pf2 = prove::<Sha3_256, _>(&comm, &outer_tensor[..], &mut tr1).unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(n_col_opens as u64).to_be_bytes()[..]);
    let res = verify(
        &root,
        &outer_tensor[..],
        &inner_tensor[..],
        &pf,
        rho,
        n_degree_tests,
        n_col_opens,
        &mut tr2,
    )
    .unwrap();

    assert_eq!(res, eval);
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
    let res2 = verify(
        &root,
        &outer_tensor[..],
        &inner_tensor[..],
        &pf2,
        rho,
        n_degree_tests,
        n_col_opens,
        &mut tr2,
    )
    .unwrap();

    assert_eq!(res2, eval);
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

fn random_comm() -> LigeroCommit<Sha3_256, Ft> {
    use super::get_dims;

    let mut rng = rand::thread_rng();

    let lgl = 8 + rng.gen::<usize>() % 8;
    let len_base = 1 << (lgl - 1);
    let len = len_base + (rng.gen::<usize>() % len_base);
    let rho = rng.gen_range(0.1f64..0.9f64);
    let (n_rows, n_per_row, n_cols) = get_dims::<DummyError>(len, rho).unwrap();

    let coeffs_len = (n_per_row - 1) * n_rows + 1 + (rng.gen::<usize>() % n_rows);
    let coeffs = {
        let mut tmp = repeat_with(|| Ft::random(&mut rng))
            .take(coeffs_len)
            .collect::<Vec<Ft>>();
        tmp.resize(n_per_row * n_rows, Ft::zero());
        tmp
    };

    let comm_len = n_rows * n_cols;
    let comm: Vec<Ft> = repeat_with(|| Ft::random(&mut rng))
        .take(comm_len)
        .collect();

    LigeroCommit::<Sha3_256, Ft> {
        comm,
        coeffs,
        rho,
        n_degree_tests: 1usize,
        n_col_opens: 128usize,
        n_rows,
        n_cols,
        n_per_row,
        hashes: vec![<Output<Sha3_256> as Default>::default(); 2 * n_cols - 1],
    }
}
