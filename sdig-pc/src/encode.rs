// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of sdig-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/*! encode a vector of length n using an expander code */

use ff::Field;
use ndarray::{linalg::Dot, ArrayView};
use num_traits::Num;
use sprs::{CsMat, MulAcc};

// given a set of precodes and postcodes, output length of codeword
pub(super) fn codeword_length<F>(precodes: &[CsMat<F>], postcodes: &[CsMat<F>]) -> usize
where
    F: Field + Num,
{
    assert!(!precodes.is_empty());
    assert_eq!(precodes.len(), postcodes.len());

    // input
    precodes[0].cols()
    // R-S result
        + postcodes.last().unwrap().cols()
    // precode outputs (except input to R-S, which is not part of codeword)
        + precodes.iter().take(precodes.len() - 1).map(|pc| pc.rows()).sum::<usize>()
    // postcode outputs
        + postcodes.iter().map(|pc| pc.rows()).sum::<usize>()
}

/// encode a vector given a code of corresponding length
pub fn encode<F, T>(mut xi: T, precodes: &[CsMat<F>], postcodes: &[CsMat<F>])
where
    F: Field + Num + MulAcc,
    T: AsMut<[F]>,
{
    // check sizes
    assert_eq!(xi.as_mut().len(), codeword_length(precodes, postcodes));

    // compute precodes all the way down
    let mut in_start = 0usize;
    for precode in precodes.iter().take(precodes.len() - 1) {
        // compute matrix-vector product
        let in_end = in_start + precode.cols();
        let (in_arr, out_arr) = xi.as_mut().split_at_mut(in_end);
        out_arr[..precode.rows()].copy_from_slice(
            precode
                .dot(&ArrayView::from(&in_arr[in_start..]))
                .as_slice()
                .unwrap(),
        );

        in_start = in_end;
    }

    // base-case code: Reed-Solomon
    let (mut in_start, mut out_start) = {
        // first, evaluate last precode into temporary storage
        let precode = precodes.last().unwrap();
        let in_end = in_start + precode.cols();
        let in_arr = precode
            .dot(&ArrayView::from(&xi.as_mut()[in_start..in_end]))
            .into_raw_vec();

        // now evaluate Reed-Solomon code on the result
        let out_end = in_end + postcodes.last().unwrap().cols();
        reed_solomon(in_arr.as_ref(), &mut xi.as_mut()[in_end..out_end]);

        (in_end + precode.rows(), out_end)
    };

    for (precode, postcode) in precodes.iter().rev().zip(postcodes.iter().rev()) {
        // move input pointer backward
        in_start -= precode.rows();

        // compute matrix-vector product
        let (in_arr, out_arr) = xi.as_mut().split_at_mut(out_start);
        out_arr[..postcode.rows()].copy_from_slice(
            postcode
                .dot(&ArrayView::from(&in_arr[in_start..]))
                .as_slice()
                .unwrap(),
        );

        out_start += postcode.rows();
    }

    assert_eq!(in_start, precodes[0].cols());
    assert_eq!(out_start, xi.as_mut().len());
}

// Compute Reed-Solomon encoding using Vandermonde matrix
fn reed_solomon<F>(xi: &[F], xo: &mut [F])
where
    F: Field,
{
    let mut x = <F as Field>::one();
    for r in xo.as_mut().iter_mut() {
        *r = <F as Field>::zero();
        for j in (0..xi.len()).rev() {
            *r *= x;
            *r += xi[j];
        }
        x += <F as Field>::one();
    }
}
