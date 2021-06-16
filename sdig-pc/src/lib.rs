// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of sdig-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.
#![deny(missing_docs)]

/*!
sdig-pc is a polynomial commitment scheme from the SDIG expander code
*/

#![feature(test)]
#[cfg(feature = "bench")]
extern crate test;

use encode::{encode, reed_solomon, reed_solomon_fft};
use matgen::generate;

use ff::Field;
use fffft::{FFTError, FFTPrecomp, FieldFFT};
use lcpc2d::{def_labels, FieldHash, LcCommit, LcEncoding, LcEvalProof};
use num_traits::Num;
use serde::Serialize;
use sprs::{CsMat, MulAcc};

pub mod encode;
pub mod matgen;

#[cfg(all(test, feature = "bench"))]
mod bench;
#[cfg(any(test, feature = "bench"))]
mod tests;

/// Encoding definition for SDIG expander-based polycommit
#[derive(Clone, Debug)]
pub struct SdigEncoding<Ft> {
    n_per_row: usize, // number of inputs to the encoding
    n_cols: usize,    // number of outputs from the encoding
    precodes: Vec<CsMat<Ft>>,
    postcodes: Vec<CsMat<Ft>>,
}

const COL_ROW_RATIO_NOFFT: usize = 1;
const SDIG_BASELEN_NOFFT: usize = 20;
impl<Ft> SdigEncoding<Ft>
where
    Ft: Field + Num,
{
    /// Create a new SdigEncoding for a polynomial with `len` coefficients
    /// using a random expander code generated with seed `seed`.
    ///
    /// Note: you should use matgen::check_seed to make sure that `seed`
    /// gives a valid code. This function does not check the seed!
    pub fn new(len: usize, seed: u64) -> Self {
        let n_per_row = (len as f64).sqrt().ceil() as usize * COL_ROW_RATIO_NOFFT;
        let (precodes, postcodes) = generate(n_per_row, SDIG_BASELEN_NOFFT, seed);
        assert_eq!(n_per_row, precodes[0].cols());
        let n_cols = n_per_row + postcodes[0].rows();
        Self {
            n_per_row,
            n_cols,
            precodes,
            postcodes,
        }
    }

    /// Create a new SdigEncoding for a commitment with dimensions `n_per_row` and `n_cols`
    pub fn new_from_dims(n_per_row: usize, n_cols: usize, seed: u64) -> Self {
        let (precodes, postcodes) = generate(n_per_row, SDIG_BASELEN_NOFFT, seed);
        assert_eq!(n_per_row, precodes[0].cols());
        assert_eq!(n_cols, n_per_row + postcodes[0].rows());
        Self {
            n_per_row,
            n_cols,
            precodes,
            postcodes,
        }
    }
}

impl<Ft> LcEncoding for SdigEncoding<Ft>
where
    Ft: Field + FieldHash + MulAcc + Num + Serialize,
{
    type F = Ft;
    type Err = std::io::Error;

    def_labels!(sdig_nofft_pc);

    fn encode<T: AsMut<[Ft]>>(&self, inp: T) -> Result<(), Self::Err> {
        encode(
            inp,
            SDIG_BASELEN_NOFFT,
            &self.precodes,
            &self.postcodes,
            reed_solomon,
        )
    }

    fn get_dims(&self, len: usize) -> (usize, usize, usize) {
        let n_rows = (len + self.n_per_row - 1) / self.n_per_row;
        (n_rows, self.n_per_row, self.n_cols)
    }

    fn dims_ok(&self, n_per_row: usize, n_cols: usize) -> bool {
        let ok = n_per_row < n_cols;
        let np1 = n_per_row == self.n_per_row;
        let np2 = n_per_row == self.precodes[0].cols();
        let nc1 = n_cols == self.n_cols;
        let nc2 = n_cols == n_per_row + self.postcodes[0].rows();
        ok && np1 && np2 && nc1 && nc2
    }
}

/// SDIG expander-based polynomial commitment
pub type SdigCommit<D, F> = LcCommit<D, SdigEncoding<F>>;

/// An evaluation proof for SDIG expander-based polynomial commitment
pub type SdigEvalProof<D, F> = LcEvalProof<D, SdigEncoding<F>>;

/// Encoding definition for SDIG expander-based polycommit with FFT-based R-S base case
#[derive(Clone, Debug)]
pub struct SdigFFTEncoding<Ft> {
    n_per_row: usize,
    n_cols: usize,
    precodes: Vec<CsMat<Ft>>,
    postcodes: Vec<CsMat<Ft>>,
    pc: FFTPrecomp<Ft>,
}

const COL_ROW_RATIO_FFT: usize = 1;
const SDIG_BASELEN_FFT: usize = 128;
impl<Ft> SdigFFTEncoding<Ft>
where
    Ft: FieldFFT + Num,
{
    /// Create a new SdigFFTEncoding for a polynomial with `len` coefficients
    /// using a random expander code (with FFT-based R-S basecase) generated
    /// using seed `seed`.
    ///
    /// Note: you should use matgen::check_seed to make sure that `seed`
    /// gives a valid code. This function does not check the seed!
    pub fn new(len: usize, seed: u64) -> Self {
        let n_per_row = (len as f64).sqrt().ceil() as usize * COL_ROW_RATIO_FFT;
        let (precodes, postcodes) = generate(n_per_row, SDIG_BASELEN_FFT, seed);
        let pc = <Ft as FieldFFT>::precomp_fft(SDIG_BASELEN_FFT).unwrap();
        assert_eq!(n_per_row, precodes[0].cols());
        assert_eq!(SDIG_BASELEN_FFT, 1 << pc.get_log_len());
        let n_cols = n_per_row + postcodes[0].rows();
        Self {
            n_per_row,
            n_cols,
            precodes,
            postcodes,
            pc,
        }
    }

    /// Create new SdigFFTEncoding for a commitment with dimensions `n_per_row` and `n_cols`
    pub fn new_from_dims(n_per_row: usize, n_cols: usize, seed: u64) -> Self {
        let (precodes, postcodes) = generate(n_per_row, SDIG_BASELEN_FFT, seed);
        let pc = <Ft as FieldFFT>::precomp_fft(SDIG_BASELEN_FFT).unwrap();
        assert_eq!(SDIG_BASELEN_FFT, 1 << pc.get_log_len());
        assert_eq!(n_per_row, precodes[0].cols());
        assert_eq!(n_cols, n_per_row + postcodes[0].rows());
        Self {
            n_per_row,
            n_cols,
            precodes,
            postcodes,
            pc,
        }
    }
}

impl<Ft> LcEncoding for SdigFFTEncoding<Ft>
where
    Ft: FieldFFT + FieldHash + MulAcc + Num + Serialize,
{
    type F = Ft;
    type Err = FFTError;

    def_labels!(sdig_fft_pc);

    fn encode<T: AsMut<[Ft]>>(&self, inp: T) -> Result<(), Self::Err> {
        encode(
            inp,
            SDIG_BASELEN_FFT,
            &self.precodes,
            &self.postcodes,
            |x, l| reed_solomon_fft(x, l, &self.pc),
        )
    }

    fn get_dims(&self, len: usize) -> (usize, usize, usize) {
        let n_rows = (len + self.n_per_row - 1) / self.n_per_row;
        (n_rows, self.n_per_row, self.n_cols)
    }

    fn dims_ok(&self, n_per_row: usize, n_cols: usize) -> bool {
        let ok = n_per_row < n_cols;
        let np1 = n_per_row == self.n_per_row;
        let np2 = n_per_row == self.precodes[0].cols();
        let nc1 = n_cols == self.n_cols;
        let nc2 = n_cols == n_per_row + self.postcodes[0].rows();
        let pc = SDIG_BASELEN_FFT == (1 << self.pc.get_log_len());
        ok && np1 && np2 && nc1 && nc2 && pc
    }
}

/// SDIG expander-based polynomial commitment
pub type SdigFFTCommit<D, F> = LcCommit<D, SdigFFTEncoding<F>>;

/// An evaluation proof for SDIG expander-based polynomial commitment
pub type SdigFFTEvalProof<D, F> = LcEvalProof<D, SdigFFTEncoding<F>>;
