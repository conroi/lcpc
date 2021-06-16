// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::tests::ft::*;
use super::{LigeroCommit, LigeroEncoding};

use blake2::Blake2b;
use blake3::Hasher;
use digest::Digest;
use ff::Field;
use itertools::iterate;
use merlin::Transcript;
use sha3::Sha3_256;
use std::iter::repeat_with;
use test::{black_box, Bencher};

const N_DEGREE_TESTS: usize = 2;
const N_COL_OPENS: usize = 128;

fn commit_bench<D: Digest>(b: &mut Bencher, log_len: usize) {
    let coeffs = random_coeffs(log_len);
    let enc = LigeroEncoding::new(coeffs.len(), 0.25);

    b.iter(|| {
        black_box(LigeroCommit::<D, Ft>::commit(&coeffs, &enc).unwrap());
    });
}

#[bench]
fn commit_sha3_24(b: &mut Bencher) {
    commit_bench::<Sha3_256>(b, 24);
}

#[bench]
fn commit_sha3_16(b: &mut Bencher) {
    commit_bench::<Sha3_256>(b, 16);
}

#[bench]
fn commit_sha3_20(b: &mut Bencher) {
    commit_bench::<Sha3_256>(b, 20);
}

#[bench]
fn commit_blake2b_24(b: &mut Bencher) {
    commit_bench::<Blake2b>(b, 24);
}

#[bench]
fn commit_blake2b_16(b: &mut Bencher) {
    commit_bench::<Blake2b>(b, 16);
}

#[bench]
fn commit_blake2b_20(b: &mut Bencher) {
    commit_bench::<Blake2b>(b, 20);
}

#[bench]
fn commit_blake3_24(b: &mut Bencher) {
    commit_bench::<Hasher>(b, 24);
}

#[bench]
fn commit_blake3_16(b: &mut Bencher) {
    commit_bench::<Hasher>(b, 16);
}

#[bench]
fn commit_blake3_20(b: &mut Bencher) {
    commit_bench::<Hasher>(b, 20);
}

fn prove_bench<D: Digest>(b: &mut Bencher, log_len: usize) {
    let coeffs = random_coeffs(log_len);
    let enc = LigeroEncoding::new(coeffs.len(), 0.25);
    let comm = LigeroCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

    // random point to eval at
    let x = Ft::random(&mut rand::thread_rng());
    let inner_tensor: Vec<Ft> = iterate(Ft::one(), |&v| v * x)
        .take(comm.get_n_per_row())
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft::one(), |&v| v * xr)
            .take(comm.get_n_rows())
            .collect()
    };

    b.iter(|| {
        let mut tr = Transcript::new(b"bench transcript");
        tr.append_message(b"polycommit", comm.get_root().as_ref());
        tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
        tr.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
        tr.append_message(b"ndegs", &(N_DEGREE_TESTS as u64).to_be_bytes()[..]);
        black_box(
            comm.prove(
                &outer_tensor[..],
                &enc,
                N_DEGREE_TESTS,
                N_COL_OPENS,
                &mut tr,
            )
            .unwrap(),
        );
    });
}

#[bench]
fn prove_sha3_24(b: &mut Bencher) {
    prove_bench::<Sha3_256>(b, 24);
}

#[bench]
fn prove_sha3_16(b: &mut Bencher) {
    prove_bench::<Sha3_256>(b, 16);
}

#[bench]
fn prove_sha3_20(b: &mut Bencher) {
    prove_bench::<Sha3_256>(b, 20);
}

#[bench]
fn prove_blake2b_24(b: &mut Bencher) {
    prove_bench::<Blake2b>(b, 24);
}

#[bench]
fn prove_blake2b_16(b: &mut Bencher) {
    prove_bench::<Blake2b>(b, 16);
}

#[bench]
fn prove_blake2b_20(b: &mut Bencher) {
    prove_bench::<Blake2b>(b, 20);
}

#[bench]
fn prove_blake3_24(b: &mut Bencher) {
    prove_bench::<Hasher>(b, 24);
}

#[bench]
fn prove_blake3_16(b: &mut Bencher) {
    prove_bench::<Hasher>(b, 16);
}

#[bench]
fn prove_blake3_20(b: &mut Bencher) {
    prove_bench::<Hasher>(b, 20);
}

fn verify_bench<D: Digest>(b: &mut Bencher, log_len: usize) {
    let coeffs = random_coeffs(log_len);
    let enc = LigeroEncoding::new(coeffs.len(), 0.25);
    let comm = LigeroCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

    // random point to eval at
    let x = Ft::random(&mut rand::thread_rng());
    let inner_tensor: Vec<Ft> = iterate(Ft::one(), |&v| v * x)
        .take(comm.get_n_per_row())
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft::one(), |&v| v * xr)
            .take(comm.get_n_rows())
            .collect()
    };

    let mut tr = Transcript::new(b"bench transcript");
    tr.append_message(b"polycommit", comm.get_root().as_ref());
    tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
    tr.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    tr.append_message(b"ndegs", &(N_DEGREE_TESTS as u64).to_be_bytes()[..]);
    let pf = comm
        .prove(
            &outer_tensor[..],
            &enc,
            N_DEGREE_TESTS,
            N_COL_OPENS,
            &mut tr,
        )
        .unwrap();
    let root = comm.get_root();

    b.iter(|| {
        let mut tr = Transcript::new(b"bench transcript");
        tr.append_message(b"polycommit", comm.get_root().as_ref());
        tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
        tr.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
        tr.append_message(b"ndegs", &(N_DEGREE_TESTS as u64).to_be_bytes()[..]);
        black_box(
            pf.verify(
                root.as_ref(),
                &outer_tensor[..],
                &inner_tensor[..],
                &enc,
                N_DEGREE_TESTS,
                N_COL_OPENS,
                &mut tr,
            )
            .unwrap(),
        );
    });
}

#[bench]
fn verify_sha3_24(b: &mut Bencher) {
    verify_bench::<Sha3_256>(b, 24);
}

#[bench]
fn verify_sha3_16(b: &mut Bencher) {
    verify_bench::<Sha3_256>(b, 16);
}

#[bench]
fn verify_sha3_20(b: &mut Bencher) {
    verify_bench::<Sha3_256>(b, 20);
}

#[bench]
fn verify_blake2b_24(b: &mut Bencher) {
    verify_bench::<Blake2b>(b, 24);
}

#[bench]
fn verify_blake2b_16(b: &mut Bencher) {
    verify_bench::<Blake2b>(b, 16);
}

#[bench]
fn verify_blake2b_20(b: &mut Bencher) {
    verify_bench::<Blake2b>(b, 20);
}

#[bench]
fn verify_blake3_24(b: &mut Bencher) {
    verify_bench::<Hasher>(b, 24);
}

#[bench]
fn verify_blake3_16(b: &mut Bencher) {
    verify_bench::<Hasher>(b, 16);
}

#[bench]
fn verify_blake3_20(b: &mut Bencher) {
    verify_bench::<Hasher>(b, 20);
}

fn random_coeffs(log_len: usize) -> Vec<Ft> {
    let mut rng = rand::thread_rng();
    repeat_with(|| Ft::random(&mut rng))
        .take(1 << log_len)
        .collect()
}
