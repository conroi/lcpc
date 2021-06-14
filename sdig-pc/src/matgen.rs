// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/*! generate a random code for a given n */

use ff::Field;
use itertools::iterate;
use num_traits::Num;
use rand::{distributions::Uniform, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use sprs::CsMat;
use std::collections::HashSet;

// minimum dimension, at which point we just switch to R-S
const LOG_MIN_DIM: usize = 7;
const LOG_RS_RATE_INV: usize = 0;
pub(super) const RS_LEN: usize = 1 << (LOG_MIN_DIM + LOG_RS_RATE_INV);
// alpha = 0.32
const ALPHA_NUM: usize = 8;
const ALPHA_DEN: usize = 25;
/* // XXX(rsw) these don't appear to be used in the original code...
// beta = 0.16
const BETA_NUM: usize = 4;
const BETA_DEN: usize = 25;
// gamma = 0.45
const GAMMA_NUM: usize = 9;
const GAMMA_DEN: usize = 20;
*/
// k = 1.1
const K_NUM: usize = 11;
const K_DEN: usize = 10;
// row density of precodes
const D1: usize = 7;
// row density of postcodes
const D2: usize = 10;

const fn ceil_mul(n: usize, num: usize, den: usize) -> usize {
    (n * num + den - 1) / den
}

/// Generate a random code from a given seed
// XXX(rsw) we can't possibly need a cryptographically strong seed, can we???
pub fn generate<F>(n: usize, seed: u64) -> (Vec<CsMat<F>>, Vec<CsMat<F>>)
where
    F: Field + Num,
{
    let (pre_dims, post_dims) = get_dims(n);
    if pre_dims.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut precodes = Vec::with_capacity(pre_dims.len());
    let mut postcodes = Vec::with_capacity(pre_dims.len());
    pre_dims[..]
        .par_iter()
        .zip(&post_dims[..])
        .enumerate()
        .map(|(i, (&(ni, mi), &(nip, mip)))| {
            let (precode, mut rng) = precode_and_rng(seed, i, ni, mi);
            let postcode = gen_code(nip, mip, D2, &mut rng);
            (precode, postcode)
        })
        .unzip_into_vecs(&mut precodes, &mut postcodes);

    (precodes, postcodes)
}

// this is used in both generate() and check_seed(),
// ensuring that they generate exactly the same precodes
fn precode_and_rng<F>(seed: u64, i: usize, ni: usize, mi: usize) -> (CsMat<F>, ChaCha20Rng)
where
    F: Field + Num,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    rng.set_stream(i as u64);
    let precode = gen_code(ni, mi, D1, &mut rng);
    (precode, rng)
}

// compute dimensions for all of the matrices used by this code
#[allow(clippy::type_complexity)]
fn get_dims(n: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    const MIN_DIM: usize = ((1 << LOG_MIN_DIM) * K_DEN) / K_NUM;

    // if n is small enough, there are no matrices
    if n <= MIN_DIM {
        return (Vec::new(), Vec::new());
    }

    // figure out dimensions for the precode and postcode matrices
    let pre_dims = {
        let mut tmp: Vec<_> = iterate(n, |&ni| ceil_mul(ni, ALPHA_NUM, ALPHA_DEN))
            .take_while(|&ni| ni > MIN_DIM)
            .collect();
        if let Some(&ni) = tmp.last() {
            let last = ceil_mul(ni, ALPHA_NUM, ALPHA_DEN);
            assert!(last <= MIN_DIM);
            assert!(ceil_mul(last, K_NUM, K_DEN) <= (1 << LOG_MIN_DIM));
            tmp.push(last);
        }
        assert!(tmp.len() > 1);
        tmp[..]
            .windows(2)
            .map(|nm| (nm[0], nm[1]))
            .collect::<Vec<(usize, usize)>>()
    };

    let post_dims = pre_dims
        .iter()
        .map(|&(ni, mi)| {
            // for the last postcode matrix, adjust #rows to accommodate
            // (1<<LOG_MIN_DIM)-length R-S codeword
            let niprime = ni
                + if mi <= MIN_DIM {
                    RS_LEN
                } else {
                    ceil_mul(mi, K_NUM, K_DEN)
                };
            let miprime = ceil_mul(ni, K_NUM, K_DEN);
            (niprime, miprime)
        })
        .collect::<Vec<(usize, usize)>>();
    assert_eq!(pre_dims.len(), post_dims.len());

    (pre_dims, post_dims)
}

/// check that a given seed will generate a reasonable code for this dimension and field
pub fn check_seed<F>(n: usize, seed: u64) -> bool
where
    F: Field + Num,
{
    let (pre_dims, _) = get_dims(n);

    pre_dims[..].iter().enumerate().all(|(i, &(ni, mi))| {
        let (precode, _) = precode_and_rng(seed, i, ni, mi);
        check_precode::<F>(precode)
    })
}

// returns true if pc_cand is a good precode, false otherwise
fn check_precode<F>(pc_cand: CsMat<F>) -> bool
where
    F: Field + Num,
{
    // cols because pc_cand is transposed
    let mut row_indices = Vec::with_capacity(pc_cand.cols());

    // go through each row (column, really) checking and collecting nonzero indices
    for row_vec in pc_cand.outer_iterator() {
        let row_ind_set = row_vec.indices().iter().cloned().collect::<HashSet<_>>();
        // requirement 1: each row needs at least 2 nonzero columns
        if row_ind_set.len() < 2 {
            return false;
        }
        row_indices.push(row_ind_set);
    }

    // requirement 2: each pair of rows has at least three distinct non-zero columns
    for (row, row_ind1) in row_indices.iter().enumerate() {
        // if this row has at least three distinct nonzero elements, all related pairs do too
        if row_ind1.len() >= 3 {
            continue;
        }
        for row_ind2 in row_indices[row..].iter() {
            if row_ind1.union(row_ind2).count() < 3 {
                return false;
            }
        }
    }

    // if we got to here, we passed both requirements
    true
}

// generate a code matrix of a given size with specified row density
fn gen_code<F, R>(n: usize, m: usize, d: usize, mut rng: R) -> CsMat<F>
where
    F: Field + Num,
    R: Rng,
{
    let dist = Uniform::new(0, m);
    let mut data = Vec::<F>::with_capacity(d * n);
    let mut idxs = Vec::<usize>::with_capacity(d * n);
    let mut ptrs = Vec::<usize>::with_capacity(1 + n);
    ptrs.push(0); // ptrs always starts with 0

    for _ in 0..n {
        // for each row, generate D2 random nonzero columns (with replacement)
        let cols = {
            let mut tmp = (&mut rng).sample_iter(&dist).take(d).collect::<Vec<_>>();
            tmp.sort_unstable();
            tmp
        };

        // sample random elements for each column
        let mut last = m + 1;
        for &col in &cols[..] {
            // detect and skip repeats
            if col == last {
                continue;
            }
            last = col;

            let val = {
                let mut tmp = F::random(&mut rng);
                while <F as Field>::is_zero(&tmp) {
                    tmp = F::random(&mut rng);
                }
                tmp
            };
            idxs.push(col);
            data.push(val);
        }
        ptrs.push(data.len());
    }

    CsMat::new_csc((m, n), ptrs, idxs, data)
}
