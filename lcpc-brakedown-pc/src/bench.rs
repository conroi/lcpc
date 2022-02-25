// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-brakedown-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::{BrakedownCommit, SdigEncoding};

use blake3::{Hasher as Blake3, traits::digest::Digest};
use ff::{Field, PrimeField};
use itertools::iterate;
use lcpc_2d::{FieldHash, LcEncoding};
use merlin::Transcript;
use num_traits::Num;
use sprs::MulAcc;
use test::{black_box, Bencher};
use lcpc_test_fields::{def_bench, ft127::*, ft255::*, random_coeffs};

#[bench]
fn matgen_bench(b: &mut Bencher) {
    use super::codespec::SdigCode3 as TestCode;
    use super::matgen::generate;

    b.iter(|| {
        generate::<Ft127, TestCode>(1048576, 0u64);
    })
}

fn commit_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: Field + FieldHash + MulAcc + Num + PrimeField,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new(coeffs.len(), 0);

    b.iter(|| {
        black_box(BrakedownCommit::<D, Ft>::commit(&coeffs, &enc).unwrap());
    });
}

fn prove_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: Field + FieldHash + MulAcc + Num + PrimeField,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new(coeffs.len(), 0);
    let comm = BrakedownCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

    // random point to eval at
    let x = Ft::random(&mut rand::thread_rng());
    let inner_tensor: Vec<Ft> = iterate(<Ft as Field>::one(), |&v| v * x)
        .take(comm.get_n_per_row())
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(<Ft as Field>::one(), |&v| v * xr)
            .take(comm.get_n_rows())
            .collect()
    };

    b.iter(|| {
        let mut tr = Transcript::new(b"bench transcript");
        tr.append_message(b"polycommit", comm.get_root().as_ref());
        tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
        tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
        tr.append_message(
            b"ndegs",
            &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
        );
        black_box(comm.prove(&outer_tensor[..], &enc, &mut tr).unwrap());
    });
}

fn verify_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: Field + FieldHash + MulAcc + Num + PrimeField,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new(coeffs.len(), 0);
    let comm = BrakedownCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

    // random point to eval at
    let x = Ft::random(&mut rand::thread_rng());
    let inner_tensor: Vec<Ft> = iterate(<Ft as Field>::one(), |&v| v * x)
        .take(comm.get_n_per_row())
        .collect();
    let outer_tensor: Vec<Ft> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(<Ft as Field>::one(), |&v| v * xr)
            .take(comm.get_n_rows())
            .collect()
    };

    let mut tr = Transcript::new(b"bench transcript");
    tr.append_message(b"polycommit", comm.get_root().as_ref());
    tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
    tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    tr.append_message(
        b"ndegs",
        &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
    );
    let pf = comm.prove(&outer_tensor[..], &enc, &mut tr).unwrap();
    let root = comm.get_root();

    b.iter(|| {
        let mut tr = Transcript::new(b"bench transcript");
        tr.append_message(b"polycommit", comm.get_root().as_ref());
        tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
        tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
        tr.append_message(
            b"ndegs",
            &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
        );
        black_box(
            pf.verify(
                root.as_ref(),
                &outer_tensor[..],
                &inner_tensor[..],
                &enc,
                &mut tr,
            )
            .unwrap(),
        );
    });
}

def_bench!(commit, Ft127, Blake3, 16);
def_bench!(commit, Ft127, Blake3, 20);
def_bench!(commit, Ft127, Blake3, 24);

def_bench!(prove, Ft127, Blake3, 16);
def_bench!(prove, Ft127, Blake3, 20);
def_bench!(prove, Ft127, Blake3, 24);

def_bench!(verify, Ft127, Blake3, 16);
def_bench!(verify, Ft127, Blake3, 20);
def_bench!(verify, Ft127, Blake3, 24);

def_bench!(commit, Ft255, Blake3, 16);
def_bench!(commit, Ft255, Blake3, 20);
def_bench!(commit, Ft255, Blake3, 24);

def_bench!(prove, Ft255, Blake3, 16);
def_bench!(prove, Ft255, Blake3, 20);
def_bench!(prove, Ft255, Blake3, 24);

def_bench!(verify, Ft255, Blake3, 16);
def_bench!(verify, Ft255, Blake3, 20);
def_bench!(verify, Ft255, Blake3, 24);
