// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of sdig-pc, which is part of lcpc.
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

// alpha = 0.178
const ALPHA_NUM: usize = 89;
const ALPHA_DEN: usize = 500;
// beta = 0.0785
const BETA_NUM: usize = 157;
const BETA_DEN: usize = 2000;
// r = 1.57
const R_NUM: usize = 157;
const R_DEN: usize = 100;

// constants for computing cn (H is bin entropy fn)
// H(beta) + alpha * H(1.2 * beta / alpha)
const H_BETA_P_ALPHA_H_1P2BOA: f64 = 0.574433098504832;
// beta * math.log(alpha / (1.2 * beta), 2)
const BETA_LOG_AO1P2B: f64 = 0.0720691446849167;

// constants for computing dn
// mu = r - 1 - r * alpha
// nu = beta + alpha * beta + 0.03
// r * alpha * H(beta / r) + mu * H(nu / mu)
const R_ALPHA_H_BOR_P_MU_H_NUOMU: f64 = 0.365393858883888;
// alpha * beta * math.log(mu / nu, 2)
const ALPHA_BETA_LOG_MON: f64 = 0.0174141735715980;

const fn ceil_mul(n: usize, num: usize, den: usize) -> usize {
    (n * num + den - 1) / den
}

/// Generate a random code from a given seed with a base codeword length of `baselen`
pub fn generate<F>(
    n: usize,
    baselen: usize,
    seed: u64,
    log2p: f64,
) -> (Vec<CsMat<F>>, Vec<CsMat<F>>)
where
    F: Field + Num,
{
    let (pre_dims, post_dims) = get_dims(n, baselen, log2p);
    assert!(!pre_dims.is_empty());

    let mut precodes = Vec::with_capacity(pre_dims.len());
    let mut postcodes = Vec::with_capacity(pre_dims.len());
    pre_dims[..]
        .par_iter()
        .zip(&post_dims[..])
        .enumerate()
        .map(|(i, (&(ni, mi, cn), &(nip, mip, dn)))| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            rng.set_stream(i as u64);
            let precode = gen_code(ni, mi, cn, &mut rng);
            let postcode = gen_code(nip, mip, dn, &mut rng);
            (precode, postcode)
        })
        .unzip_into_vecs(&mut precodes, &mut postcodes);

    (precodes, postcodes)
}

// compute dimensions for all of the matrices used by this code
#[allow(clippy::type_complexity)]
fn get_dims(
    n: usize,
    baselen: usize,
    log2p: f64,
) -> (Vec<(usize, usize, usize)>, Vec<(usize, usize, usize)>) {
    use std::cmp::{max, min};
    assert!(n > baselen);

    // figure out dimensions for the precode and postcode matrices
    let pre_dims = {
        let mut tmp: Vec<_> = iterate(n, |&ni| ceil_mul(ni, ALPHA_NUM, ALPHA_DEN))
            .take_while(|&ni| ni > baselen)
            .collect();
        if let Some(&ni) = tmp.last() {
            let last = ceil_mul(ni, ALPHA_NUM, ALPHA_DEN);
            assert!(last <= baselen);
            tmp.push(last);
        }
        assert!(tmp.len() > 1);
        tmp[..]
            .windows(2)
            .map(|nm| {
                let ni = nm[0];
                let mi = nm[1];
                let cn = min(
                    max(
                        ceil_mul(ni, 6 * BETA_NUM, 5 * BETA_DEN),
                        3 + ceil_mul(ni, BETA_NUM, BETA_DEN),
                    ),
                    ((110f64 / (ni as f64) + H_BETA_P_ALPHA_H_1P2BOA) / BETA_LOG_AO1P2B).ceil()
                        as usize,
                );
                (ni, mi, cn)
            })
            .collect::<Vec<_>>()
    };

    let post_dims = pre_dims
        .iter()
        .map(|&(ni, mi, _)| {
            let niprime = ceil_mul(mi, R_NUM, R_DEN);
            let miprime = ceil_mul(ni, R_NUM, R_DEN) - ni - niprime;
            let tmp1 = ceil_mul(ni, 2 * BETA_NUM, BETA_DEN); // 2 * beta * ni
            let tmp2 = ceil_mul(ni, R_NUM, R_DEN) - ni + 110; // ni * (r - 1 + 110/ni)
            let dn = min(
                tmp1 + (tmp2 as f64 / log2p).ceil() as usize,
                ((110f64 / (ni as f64) + R_ALPHA_H_BOR_P_MU_H_NUOMU) / ALPHA_BETA_LOG_MON).ceil()
                    as usize,
            );
            (niprime, miprime, dn)
        })
        .collect::<Vec<_>>();

    assert_eq!(pre_dims.len(), post_dims.len());
    (pre_dims, post_dims)
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
        // for each row, sample d random nonzero columns (without replacement)
        let cols = {
            /*
            let mut nub = HashSet::new();
            let mut tmp = (&mut rng)
                .sample_iter(&dist)
                .filter(|&x| {
                    if nub.contains(&x) {
                        false
                    } else {
                        nub.insert(x);
                        true
                    }
                })
                .take(d)
                .collect::<Vec<_>>();
            */
            // for small d, the quadratic approach is almost certainly faster
            let mut tmp = Vec::with_capacity(d);
            assert_eq!(
                d,
                (&mut rng)
                    .sample_iter(&dist)
                    .filter(|&x| {
                        if tmp.contains(&x) {
                            false
                        } else {
                            tmp.push(x);
                            true
                        }
                    })
                    .take(d)
                    .count()
            );
            tmp.sort_unstable(); // need to sort to supply to new_csc below
            tmp
        };
        assert_eq!(d, cols.len());

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
