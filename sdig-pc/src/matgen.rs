// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of sdig-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/*! generate a random code for a given n */

use crate::codespec::SdigSpecification;

use ff::Field;
use itertools::iterate;
use lcpc2d::SizedField;
use num_traits::Num;
use rand::{distributions::Uniform, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use sprs::CsMat;

const fn ceil_muldiv(n: usize, num: usize, den: usize) -> usize {
    (n * num + den - 1) / den
}

/// Generate a random code from a given seed
pub fn generate<F, S>(n: usize, seed: u64) -> (Vec<CsMat<F>>, Vec<CsMat<F>>)
where
    F: Field + Num + SizedField,
    S: SdigSpecification,
{
    let (pre_dims, post_dims) = get_dims::<S>(n, F::FLOG2 as f64);
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
fn get_dims<S: SdigSpecification>(
    n: usize,
    log2p: f64,
) -> (Vec<(usize, usize, usize)>, Vec<(usize, usize, usize)>) {
    use std::cmp::{max, min};
    let baselen = S::baselen();
    assert!(n > baselen);

    // figure out dimensions for the precode and postcode matrices
    let pre_dims = {
        let mut tmp: Vec<_> = iterate(n, |&ni| ceil_muldiv(ni, S::alpha_num(), S::alpha_den()))
            .take_while(|&ni| ni > baselen)
            .collect();
        if let Some(&ni) = tmp.last() {
            let last = ceil_muldiv(ni, S::alpha_num(), S::alpha_den());
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
                        ceil_muldiv(ni, 32 * S::beta_num(), 25 * S::beta_den()),
                        4 + ceil_muldiv(ni, S::beta_num(), S::beta_den()),
                    ),
                    ((110f64 / (ni as f64) + S::cnst_cn_1()) / S::cnst_cn_2()).ceil() as usize,
                );
                let cn = min(cn, mi); // can't generate more nonzero entries than there are columns
                (ni, mi, cn)
            })
            .collect::<Vec<_>>()
    };

    let post_dims = pre_dims
        .iter()
        .map(|&(ni, mi, _)| {
            let niprime = ceil_muldiv(mi, S::r_num(), S::r_den());
            let miprime = ceil_muldiv(ni, S::r_num(), S::r_den()) - ni - niprime;
            let tmp1 = ceil_muldiv(ni, 2 * S::beta_num(), S::beta_den()); // 2 * beta * ni
            let tmp2 = ceil_muldiv(ni, S::r_num(), S::r_den()) - ni + 110; // ni * (r - 1 + 110/ni)
            let dn = min(
                tmp1 + (tmp2 as f64 / log2p).ceil() as usize,
                ((110f64 / (ni as f64) + S::cnst_dn_1()) / S::cnst_dn_2()).ceil() as usize,
            );
            let dn = min(dn, miprime); // can't generate more nonzero entries than there are columns
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
