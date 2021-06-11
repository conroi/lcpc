// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.
#![deny(missing_docs)]

use ndarray::{linalg::Dot, Array};
use rand::{thread_rng, Rng};
use sprs::{CsMat, TriMat};

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
            tmp.add_triplet(i, col1, rng.gen::<u64>() % 65536u64);
            tmp.add_triplet(i, col2, rng.gen::<u64>() % 65536u64);
        }
        // csr appears to be considerably faster than csc
        tmp.to_csr().transpose_into()
    };

    let v = {
        let mut tmp = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            tmp.push(rng.gen::<u64>() % 65536u64);
        }
        Array::from(tmp)
    };

    let mut t = 0;
    for i in 0..1000 {
        let mv = m.dot(&v);
        t += mv[i % n_cols];
    }
    println!("{:?}", t);
}
