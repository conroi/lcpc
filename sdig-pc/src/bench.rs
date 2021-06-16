// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of sdig-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::{SdigCommit, SdigEncoding, SdigFFTCommit, SdigFFTEncoding};

use blake2::{Blake2b, Digest};
use ff::Field;
use fffft::FieldFFT;
use itertools::iterate;
use lcpc2d::FieldHash;
use merlin::Transcript;
use num_traits::Num;
use sprs::MulAcc;
use std::iter::repeat_with;
use test::{black_box, Bencher};
use test_fields::{def_bench, ft127::*, ft255::*};

const N_DEGREE_TESTS: usize = 2;
const N_COL_OPENS: usize = 128;

fn random_coeffs<Ft: Field>(log_len: usize) -> Vec<Ft> {
    let mut rng = rand::thread_rng();
    repeat_with(|| Ft::random(&mut rng))
        .take(1 << log_len)
        .collect()
}

fn commit_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: Field + FieldHash + MulAcc + Num,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new(coeffs.len(), 0);

    b.iter(|| {
        black_box(SdigCommit::<D, Ft>::commit(&coeffs, &enc).unwrap());
    });
}

fn prove_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: Field + FieldHash + MulAcc + Num,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new(coeffs.len(), 0);
    let comm = SdigCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

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

fn verify_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: Field + FieldHash + MulAcc + Num,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new(coeffs.len(), 0);
    let comm = SdigCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

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

fn commit_fft_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + MulAcc + Num,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigFFTEncoding::new(coeffs.len(), 0);

    b.iter(|| {
        black_box(SdigFFTCommit::<D, Ft>::commit(&coeffs, &enc).unwrap());
    });
}

fn prove_fft_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + MulAcc + Num,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigFFTEncoding::new(coeffs.len(), 0);
    let comm = SdigFFTCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

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

fn verify_fft_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + MulAcc + Num,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigFFTEncoding::new(coeffs.len(), 0);
    let comm = SdigFFTCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

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

def_bench!(commit, Ft127, Blake2b, 16);
def_bench!(commit, Ft127, Blake2b, 20);
def_bench!(commit, Ft127, Blake2b, 24);

def_bench!(prove, Ft127, Blake2b, 16);
def_bench!(prove, Ft127, Blake2b, 20);
def_bench!(prove, Ft127, Blake2b, 24);

def_bench!(verify, Ft127, Blake2b, 16);
def_bench!(verify, Ft127, Blake2b, 20);
def_bench!(verify, Ft127, Blake2b, 24);

def_bench!(commit_fft, Ft127, Blake2b, 16);
def_bench!(commit_fft, Ft127, Blake2b, 20);
def_bench!(commit_fft, Ft127, Blake2b, 24);

def_bench!(prove_fft, Ft127, Blake2b, 16);
def_bench!(prove_fft, Ft127, Blake2b, 20);
def_bench!(prove_fft, Ft127, Blake2b, 24);

def_bench!(verify_fft, Ft127, Blake2b, 16);
def_bench!(verify_fft, Ft127, Blake2b, 20);
def_bench!(verify_fft, Ft127, Blake2b, 24);

def_bench!(commit, Ft255, Blake2b, 16);
def_bench!(commit, Ft255, Blake2b, 20);
def_bench!(commit, Ft255, Blake2b, 24);

def_bench!(prove, Ft255, Blake2b, 16);
def_bench!(prove, Ft255, Blake2b, 20);
def_bench!(prove, Ft255, Blake2b, 24);

def_bench!(verify, Ft255, Blake2b, 16);
def_bench!(verify, Ft255, Blake2b, 20);
def_bench!(verify, Ft255, Blake2b, 24);

def_bench!(commit_fft, Ft255, Blake2b, 16);
def_bench!(commit_fft, Ft255, Blake2b, 20);
def_bench!(commit_fft, Ft255, Blake2b, 24);

def_bench!(prove_fft, Ft255, Blake2b, 16);
def_bench!(prove_fft, Ft255, Blake2b, 20);
def_bench!(prove_fft, Ft255, Blake2b, 24);

def_bench!(verify_fft, Ft255, Blake2b, 16);
def_bench!(verify_fft, Ft255, Blake2b, 20);
def_bench!(verify_fft, Ft255, Blake2b, 24);
