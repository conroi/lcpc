// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/*! encode a vector of length n using an expander code */

use super::matgen::RS_LEN;

use ff::Field;
use fffft::{FFTError, FFTPrecomp, FieldFFT};
use ndarray::{linalg::Dot, ArrayView};
use num_traits::Num;
use sprs::{CsMat, MulAcc};

/// encode a vector given a code of corresponding length
pub fn encode<F, T, R, E>(
    mut xi: T,
    precodes: &[CsMat<F>],
    postcodes: &[CsMat<F>],
    mut rs: R,
) -> Result<(), E>
where
    F: Field + Num + MulAcc,
    T: AsMut<[F]>,
    R: FnMut(Vec<F>, usize) -> Result<Vec<F>, E>,
    E: std::error::Error,
{
    // check sizes
    assert!(!precodes.is_empty());
    assert_eq!(precodes.len(), postcodes.len());
    assert_eq!(xi.as_mut().len(), precodes[0].cols() + postcodes[0].rows());

    // intermediate values generated during the computation
    let mut intermeds: Vec<Vec<F>> = Vec::with_capacity(precodes.len() + 1);
    intermeds.push(Vec::from(&xi.as_mut()[..precodes[0].cols()]));

    // compute precodes all the way down
    for i in 0..precodes.len() {
        assert_eq!(precodes[i].cols(), intermeds[i].len());
        let res = precodes[i]
            .dot(&ArrayView::from(&intermeds[i][..]))
            .into_raw_vec();
        if i < precodes.len() - 1 {
            // recursive case: applying precode[i]
            intermeds.push(res);
        } else {
            // base case: reed-solomon
            assert!(res.len() < RS_LEN);
            let mut res = rs(res, RS_LEN)?;
            assert_eq!(res.len(), RS_LEN);
            intermeds[i].append(&mut res);
        }
    }
    assert_eq!(intermeds.len(), postcodes.len());

    for i in (0..postcodes.len()).rev() {
        assert_eq!(postcodes[i].cols(), intermeds[i].len());
        let mut res = postcodes[i]
            .dot(&ArrayView::from(&intermeds[i][..]))
            .into_raw_vec();
        if i > 0 {
            intermeds[i - 1].append(&mut res);
        } else {
            let outp = &mut xi.as_mut()[precodes[0].cols()..];
            outp.copy_from_slice(&res[..]);
        }
    }

    Ok(())
}

pub fn reed_solomon<F>(xi: Vec<F>, len: usize) -> Result<Vec<F>, std::io::Error>
where
    F: Field,
{
    let mut res = vec![<F as Field>::zero(); len];
    let mut x = <F as Field>::one();
    for r in res.iter_mut() {
        for j in (0..xi.len()).rev() {
            *r *= x;
            *r += xi[j];
        }
        x += <F as Field>::one();
    }
    Ok(res)
}

pub fn reed_solomon_fft<F>(
    mut xi: Vec<F>,
    len: usize,
    pc: &FFTPrecomp<F>,
) -> Result<Vec<F>, FFTError>
where
    F: FieldFFT,
{
    assert!(xi.len() < len);
    xi.resize(len, <F as Field>::zero());
    <F as FieldFFT>::fft_io_pc(&mut xi, pc)?;
    Ok(xi)
}
