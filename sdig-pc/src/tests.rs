// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use ff::Field;
use fffft::FieldFFT;
use ft::*;
use ndarray::{linalg::Dot, Array};
use rand::{thread_rng, Rng};
use sprs::{CsMat, TriMat};

mod ft {
    use ff::PrimeField;
    use ff_derive_num::Num;

    #[derive(PrimeField, Num)]
    #[PrimeFieldModulus = "70386805592835581672624750593"]
    #[PrimeFieldGenerator = "17"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft([u64; 2]);
}

#[test]
fn sprs_playground() {
    let mut rng = thread_rng();
    let n_rows = 65537;
    let n_cols = 32749;

    let m: CsMat<_> = {
        let mut tmp = TriMat::new((n_rows, n_cols));
        for i in 0..n_rows {
            let col1 = rng.gen_range(0..n_cols);
            let col2 = {
                let mut tmp = rng.gen_range(0..n_cols);
                while tmp == col1 {
                    tmp = rng.gen_range(0..n_cols);
                }
                tmp
            };
            tmp.add_triplet(i, col1, Ft::random(&mut rng));
            tmp.add_triplet(i, col2, Ft::random(&mut rng));
        }
        // to_csr appears to be considerably faster than to_csc
        // (note that because of the transpose, we end up with csc in the end)
        tmp.to_csr().transpose_into()
    };

    let v = {
        let mut tmp = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            tmp.push(Ft::random(&mut rng));
        }
        Array::from(tmp)
    };

    let mut t = Ft::zero();
    for i in 0..10 {
        let mv = m.dot(&v);
        t += mv[i % n_cols];
    }
    println!("{:?}", t);
}

#[test]
fn test_matgen_check_seed() {
    let mut rng = thread_rng();
    use super::matgen::check_seed;

    let n = 256usize + (rng.gen::<usize>() % 4096);
    for seed in 0..1024u64 {
        if check_seed::<Ft>(n, 128, seed) {
            println!("Seed {} was good for n={}", seed, n);
            return;
        }
    }
    panic!("did not find a good seed");
}

#[test]
fn test_matgen_encode() {
    let mut rng = thread_rng();
    use super::encode::{encode, reed_solomon, reed_solomon_fft};
    use super::matgen::generate;

    let baselen = 128;
    let n = 256usize + (rng.gen::<usize>() % 4096);
    let (precodes, postcodes) = generate(n, baselen, 0u64);

    let xi_len = precodes[0].cols() + postcodes[0].rows();
    let mut xi = Vec::with_capacity(xi_len);
    for _ in 0..xi_len {
        xi.push(Ft::random(&mut rng));
    }
    encode(&mut xi, baselen, &precodes, &postcodes, reed_solomon).unwrap();

    let pc = <Ft as FieldFFT>::precomp_fft(baselen).unwrap();
    encode(&mut xi, baselen, &precodes, &postcodes, |x, l| {
        reed_solomon_fft(x, l, &pc)
    })
    .unwrap();
}
