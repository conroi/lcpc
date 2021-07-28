// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of ligero-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.
#![deny(missing_docs)]

/*!
ligero-pc is a polynomial commitment based on R-S codes, from Ligero
*/

#![feature(test)]
#[cfg(feature = "bench")]
extern crate test;

use fffft::{FFTError, FFTPrecomp, FieldFFT};
use lcpc2d::{
    def_labels, n_degree_tests, FieldHash, LcCommit, LcEncoding, LcEvalProof, SizedField,
};
use typenum::{Unsigned, U1, U4};

#[cfg(all(test, feature = "bench"))]
mod bench;
#[cfg(any(test, feature = "bench"))]
mod tests;

/// Encoding definition for Ligero-based polycommit
#[derive(Clone, Debug)]
pub struct LigeroEncodingRho<Ft, Rn, Rd> {
    n_per_row: usize, // number of inputs to the encoding
    n_cols: usize,    // number of outputs from the encoding
    pc: FFTPrecomp<Ft>,
    _p: std::marker::PhantomData<(Rn, Rd)>,
}

impl<Ft, Rn, Rd> LigeroEncodingRho<Ft, Rn, Rd>
where
    Ft: FieldFFT + SizedField,
    Rn: Unsigned + std::fmt::Debug + std::marker::Sync,
    Rd: Unsigned + std::fmt::Debug + std::marker::Sync,
{
    const LAMBDA: usize = 128;

    fn _rho_num() -> usize {
        Rn::to_usize()
    }

    fn _rho_den() -> usize {
        Rd::to_usize()
    }

    fn _rho() -> f64 {
        assert!(Self::_rho_num() < Self::_rho_den());
        Self::_rho_num() as f64 / Self::_rho_den() as f64
    }

    // number of column openings required for soundness
    fn _n_col_opens() -> usize {
        let den = ((1f64 + Self::_rho()) / 2f64).log2();
        (-(Self::LAMBDA as f64) / den).ceil() as usize
    }

    fn _n_degree_tests(n_cols: usize) -> usize {
        n_degree_tests(Self::LAMBDA, n_cols, Ft::FLOG2 as usize)
    }

    fn _get_dims(len: usize) -> Option<(usize, usize, usize)> {
        // compute #cols, which must be a power of 2 because of FFT
        let n_col_opens = Self::_n_col_opens();
        let lncf = (n_col_opens * len) as f64;
        // approximation of num_degree_tests
        let ndt = Self::_n_degree_tests((lncf.sqrt() / Self::_rho()).ceil() as usize) as f64;
        let nc1 = (((lncf / ndt).sqrt() / Self::_rho()).ceil() as usize)
            .checked_next_power_of_two()
            .and_then(|nc| {
                if nc > (1 << <Ft as FieldFFT>::S) {
                    None
                } else {
                    Some(nc)
                }
            })?;

        // minimize nr subject to #cols and RHO
        let np1 = nc1 * Self::_rho_num() / Self::_rho_den();
        let nr1 = (len + np1 - 1) / np1;
        let nd1 = Self::_n_degree_tests(nc1);
        assert!(np1.is_power_of_two());
        assert!(np1 * nr1 >= len);
        assert!(np1 * (nr1 - 1) < len);

        let nc2 = nc1 / 2;
        let np2 = np1 / 2;
        let nr2 = (len + np2 - 1) / np2;
        let nd2 = Self::_n_degree_tests(nc2);
        assert!(np2.is_power_of_two());
        assert!(nc2.is_power_of_two());
        assert!(np2 * nr2 >= len);
        assert!(np2 * (nr2 - 1) < len);

        let sz1 = n_col_opens * nr1 + (1 + nd1) * np1;
        let sz2 = n_col_opens * nr2 + (1 + nd2) * np2;
        let (nr, np, nc) = if sz1 < sz2 {
            (nr1, np1, nc1)
        } else {
            (nr2, np2, nc2)
        };

        Some((nr, np, nc))
    }

    fn _dims_ok(n_per_row: usize, n_cols: usize) -> bool {
        let sz = n_per_row < n_cols;
        let pow = n_cols.is_power_of_two();
        sz && pow
    }

    /// Create a new LigeroEncoding for a univariate polynomial with `len` coefficients
    pub fn new(len: usize) -> Self {
        let (_, n_per_row, n_cols) = Self::_get_dims(len).unwrap();
        Self::new_from_dims(n_per_row, n_cols)
    }

    /// Create a new LigeroEncoding for a multilinear polynomial with `n_vars` variables
    /// (i.e., 2^`n_vars` monomials).
    pub fn new_ml(n_vars: usize) -> Self {
        let n_monomials = 1 << n_vars;
        let (n_rows, n_per_row, n_cols) = Self::_get_dims(n_monomials).unwrap();
        assert!(n_rows.is_power_of_two());
        assert!(n_per_row.is_power_of_two());
        assert_eq!(n_rows * n_per_row, n_monomials);
        Self::new_from_dims(n_per_row, n_cols)
    }

    /// Create a new LigeroEncoding for a commitment with dimensions `n_per_row` and `n_cols`
    pub fn new_from_dims(n_per_row: usize, n_cols: usize) -> Self {
        assert!(Self::_dims_ok(n_per_row, n_cols));
        let pc = <Ft as FieldFFT>::precomp_fft(n_cols).unwrap();
        assert_eq!(n_cols, 1 << pc.get_log_len());
        Self {
            n_per_row,
            n_cols,
            pc,
            _p: std::marker::PhantomData::default(),
        }
    }
}

impl<Ft, Rn, Rd> LcEncoding for LigeroEncodingRho<Ft, Rn, Rd>
where
    Ft: FieldFFT + FieldHash + SizedField,
    Rn: Unsigned + std::fmt::Debug + std::marker::Sync,
    Rd: Unsigned + std::fmt::Debug + std::marker::Sync,
{
    type F = Ft;
    type Err = FFTError;

    def_labels!(ligero_pc);

    fn encode<T: AsMut<[Ft]>>(&self, inp: T) -> Result<(), FFTError> {
        <Ft as FieldFFT>::fft_io_pc(inp, &self.pc)
    }

    fn get_dims(&self, len: usize) -> (usize, usize, usize) {
        let n_rows = (len + self.n_per_row - 1) / self.n_per_row;
        (n_rows, self.n_per_row, self.n_cols)
    }

    fn dims_ok(&self, n_per_row: usize, n_cols: usize) -> bool {
        let ok = Self::_dims_ok(n_per_row, n_cols);
        let pc = n_cols == (1 << self.pc.get_log_len());
        let np = n_per_row == self.n_per_row;
        let nc = n_cols == self.n_cols;
        ok && pc && np && nc
    }

    fn get_n_col_opens(&self) -> usize {
        Self::_n_col_opens()
    }

    fn get_n_degree_tests(&self) -> usize {
        Self::_n_degree_tests(self.n_cols)
    }
}

/// Ligero-based polynomial commitment, fixing Rho and Lambda
pub type LigeroEncoding<F> = LigeroEncodingRho<F, U1, U4>;

/// Ligero-based polynomial commitment
pub type LigeroCommit<D, F> = LcCommit<D, LigeroEncoding<F>>;

/// An evaluation proof for Ligero-based polynomial commitment
pub type LigeroEvalProof<D, F> = LcEvalProof<D, LigeroEncoding<F>>;
