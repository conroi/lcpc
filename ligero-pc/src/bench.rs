// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::LigeroEncodingRho;

use blake2::{Blake2b, Digest};
use fffft::FieldFFT;
use itertools::iterate;
use lcpc2d::{FieldHash, LcCommit, LcEncoding, SizedField};
use merlin::Transcript;
use test::{black_box, Bencher};
use test_fields::{def_bench, ft127::*, ft255::*, random_coeffs};
use typenum::{Unsigned, U39 as TLo};

type THi = <TLo as std::ops::Add<typenum::U1>>::Output;

fn _commit_bench<D, Ft, Rn, Rd>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
    Rn: Unsigned + std::fmt::Debug + std::marker::Sync,
    Rd: Unsigned + std::fmt::Debug + std::marker::Sync,
{
    let coeffs = random_coeffs(log_len);
    let enc = LigeroEncodingRho::<Ft, Rn, Rd>::new(coeffs.len());

    b.iter(|| {
        black_box(LcCommit::<D, _>::commit(&coeffs, &enc).unwrap());
    });
}

fn commit_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
{
    _commit_bench::<D, Ft, typenum::U1, typenum::U4>(b, log_len);
}

fn commit_isz_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
{
    _commit_bench::<D, Ft, TLo, THi>(b, log_len);
}

fn _prove_bench<D, Ft, Rn, Rd>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
    Rn: Unsigned + std::fmt::Debug + std::marker::Sync,
    Rd: Unsigned + std::fmt::Debug + std::marker::Sync,
{
    let coeffs = random_coeffs(log_len);
    let enc = LigeroEncodingRho::<Ft, Rn, Rd>::new(coeffs.len());
    let comm = LcCommit::<D, _>::commit(&coeffs, &enc).unwrap();

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
        tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
        tr.append_message(
            b"ndegs",
            &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
        );
        black_box(comm.prove(&outer_tensor[..], &enc, &mut tr).unwrap());
    });
}

fn prove_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
{
    _prove_bench::<D, Ft, typenum::U1, typenum::U4>(b, log_len);
}

fn prove_isz_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
{
    _prove_bench::<D, Ft, TLo, THi>(b, log_len);
}

fn _verify_bench<D, Ft, Rn, Rd>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
    Rn: Unsigned + std::fmt::Debug + std::marker::Sync,
    Rd: Unsigned + std::fmt::Debug + std::marker::Sync,
{
    let coeffs = random_coeffs(log_len);
    let enc = LigeroEncodingRho::<Ft, Rn, Rd>::new(coeffs.len());
    let comm = LcCommit::<D, _>::commit(&coeffs, &enc).unwrap();

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

fn verify_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
{
    _verify_bench::<D, Ft, typenum::U1, typenum::U4>(b, log_len);
}

fn verify_isz_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest,
    Ft: FieldFFT + FieldHash + SizedField,
{
    _verify_bench::<D, Ft, TLo, THi>(b, log_len);
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

def_bench!(commit, Ft255, Blake2b, 16);
def_bench!(commit, Ft255, Blake2b, 20);
def_bench!(commit, Ft255, Blake2b, 24);

def_bench!(prove, Ft255, Blake2b, 16);
def_bench!(prove, Ft255, Blake2b, 20);
def_bench!(prove, Ft255, Blake2b, 24);

def_bench!(verify, Ft255, Blake2b, 16);
def_bench!(verify, Ft255, Blake2b, 20);
def_bench!(verify, Ft255, Blake2b, 24);

def_bench!(commit_isz, Ft255, Blake2b, 16);
def_bench!(commit_isz, Ft255, Blake2b, 20);
def_bench!(commit_isz, Ft255, Blake2b, 24);

def_bench!(prove_isz, Ft255, Blake2b, 16);
def_bench!(prove_isz, Ft255, Blake2b, 20);
def_bench!(prove_isz, Ft255, Blake2b, 24);

def_bench!(verify_isz, Ft255, Blake2b, 16);
def_bench!(verify_isz, Ft255, Blake2b, 20);
def_bench!(verify_isz, Ft255, Blake2b, 24);
