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
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use sprs::{CsMat, TriMat};
use std::collections::HashSet;

// minimum dimension, at which point we just switch to R-S
const MIN_DIM: usize = 128;
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

fn ceil_div(n: usize, num: usize, den: usize) -> usize {
    (n * num + den - 1) / den
}

/// Generate a random code from a given seed
// XXX(rsw) we can't possibly need a cryptographically strong seed, can we???
pub fn generate<F>(n: usize, seed: u64) -> Vec<(CsMat<F>, CsMat<F>)>
where
    F: Field + Num,
{
    let (pre_dims, post_dims) = get_dims(n);
    if pre_dims.is_empty() {
        return Vec::new();
    }

    let mut ret = Vec::with_capacity(pre_dims.len());
    pre_dims[..]
        .par_iter()
        .zip(&post_dims[..])
        .enumerate()
        .map(|(i, (&(ni, mi), &(nip, mip)))| {
            let (precode, mut rng) = precode_and_rng(seed, i, ni, mi);
            let postcode = gen_postcode::<F, _>(nip, mip, &mut rng);
            (precode, postcode)
        })
        .collect_into_vec(&mut ret);

    ret
}

// this is used in both generate() and check_seed(),
// ensuring that they generate exactly the same precodes
fn precode_and_rng<F>(seed: u64, i: usize, ni: usize, mi: usize) -> (CsMat<F>, ChaCha20Rng)
where
    F: Field + Num,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    rng.set_stream(i as u64);
    let precode = gen_precode::<F, _>(ni, mi, &mut rng);
    assert!(precode.is_csc()); // because of transpose_into
    assert!(precode.transpose_view().is_csr());
    (precode, rng)
}

// compute dimensions for all of the matrices used by this code
#[allow(clippy::type_complexity)]
fn get_dims(n: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    // if n is small enough, there are no matrices
    if n <= MIN_DIM {
        return (Vec::new(), Vec::new());
    }

    // figure out dimensions for the precode and postcode matrices
    let pre_dims = {
        let mut tmp: Vec<_> = iterate(n, |&ni| ceil_div(ni, ALPHA_NUM, ALPHA_DEN))
            .take_while(|&ni| ni > MIN_DIM)
            .collect();
        if let Some(&ni) = tmp.last() {
            let last = ceil_div(ni, ALPHA_NUM, ALPHA_DEN);
            assert!(last <= MIN_DIM);
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
            let niprime = ni + ceil_div(mi, K_NUM, K_DEN);
            let miprime = ceil_div(ni, K_NUM, K_DEN);
            (niprime, miprime)
        })
        .collect::<Vec<(usize, usize)>>();
    assert_eq!(pre_dims.len(), post_dims.len());

    (pre_dims, post_dims)
}

// generate a postcode of a given size
fn gen_postcode<F, R>(n: usize, m: usize, rng: R) -> CsMat<F>
where
    F: Field + Num,
    R: Rng,
{
    gen_code(n, m, D2, rng).transpose_into()
}

// generate a postcode of a given size WITHOUT CHECKING
// NOTE: must use check_seed to make sure precode is OK!
fn gen_precode<F, R>(n: usize, m: usize, rng: R) -> CsMat<F>
where
    F: Field + Num,
    R: Rng,
{
    gen_code(n, m, D1, rng).transpose_into()
}

/// check that a given seed will generate a reasonable code for this dimension and field
pub fn check_seed<F>(n: usize, seed: u64) -> bool
where
    F: Field + Num,
{
    let (pre_dims, _) = get_dims(n);

    pre_dims[..].iter().enumerate().all(|(i, &(ni, mi))| {
        let (precode, _) = precode_and_rng(seed, i, ni, mi);
        check_precode::<F>(precode.transpose_into())
    })
}

// returns true if pc_cand is a good precode, false otherwise
fn check_precode<F>(pc_cand: CsMat<F>) -> bool
where
    F: Field + Num,
{
    assert!(pc_cand.is_csr());
    let mut row_indices = Vec::with_capacity(pc_cand.rows());

    // go through each row, checking and collecting nonzero indices
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
    let mut tmat = TriMat::new((n, m));
    for i in 0..n {
        // for each row, generate D2 random nonzero columns (with replacement)
        for _ in 0..d {
            let col = rng.gen_range(0..m);
            let val = {
                let mut tmp = F::random(&mut rng);
                // almost certainly will never happen...
                while <F as Field>::is_zero(&tmp) {
                    tmp = F::random(&mut rng);
                }
                tmp
            };
            tmat.add_triplet(i, col, val);
        }
    }
    tmat.to_csr()
}
