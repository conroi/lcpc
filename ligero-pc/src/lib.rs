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

use fffft::{FFTError, FFTPrecomp, FieldFFT};
use lcpc2d::{def_labels, FieldHash, LcCommit, LcEncoding, LcEvalProof, ProverError, ProverResult};
use serde::Serialize;

#[cfg(test)]
mod tests;

/// Encoding definition for Ligero-based polycommit
#[derive(Clone, Debug)]
pub struct LigeroEncoding<Ft> {
    rho: f64,
    pc: FFTPrecomp<Ft>,
}

impl<Ft> LigeroEncoding<Ft>
where
    Ft: FieldFFT,
{
    fn _get_dims(len: usize, rho: f64) -> Option<(usize, usize, usize)> {
        if rho <= 0f64 || rho >= 1f64 {
            return None;
        }

        // compute #cols, which must be a power of 2 because of FFT
        let nc = (((len as f64).sqrt() / rho).ceil() as usize)
            .checked_next_power_of_two()
            .and_then(|nc| {
                if nc > (1 << <Ft as FieldFFT>::S) {
                    None
                } else {
                    Some(nc)
                }
            })?;

        // minimize nr subject to #cols and rho
        let np = ((nc as f64) * rho).floor() as usize;
        let nr = len / np + (len % np != 0) as usize;
        assert!(np * nr >= len);
        assert!(np * (nr - 1) < len);

        Some((nr, np, nc))
    }

    fn _dims_ok(n_per_row: usize, n_cols: usize, rho: f64) -> bool {
        let rate = n_cols as f64 * rho >= n_per_row as f64;
        let pow = n_cols.is_power_of_two();

        rate && pow
    }

    /// Create a new LigeroEncoding for a polynomial with `len` coefficients
    /// using R-S code with rate `rho`
    pub fn new(len: usize, rho: f64) -> Self {
        let (_, n_per_row, n_cols) = Self::_get_dims(len, rho).unwrap();
        let pc = <Ft as FieldFFT>::precomp_fft(n_cols).unwrap();
        assert!(Self::_dims_ok(n_per_row, n_cols, rho));
        Self { rho, pc }
    }

    /// Create a new LigeroEncoding for a commitment with dimensions `n_per_row` and `n_cols`
    pub fn new_from_dims(n_per_row: usize, n_cols: usize) -> Self {
        assert!(n_per_row < n_cols);
        // very approximate rate - make sure it will pass dims_ok
        let rho = (n_per_row + 1) as f64 / n_cols as f64;
        let pc = <Ft as FieldFFT>::precomp_fft(n_cols).unwrap();
        assert!(Self::_dims_ok(n_per_row, n_cols, rho));
        Self { rho, pc }
    }
}

impl<Ft> LcEncoding for LigeroEncoding<Ft>
where
    Ft: FieldFFT + FieldHash + Serialize,
{
    type F = Ft;
    type Err = FFTError;

    def_labels!(ligero_pc);

    fn encode<T: AsMut<[Ft]>>(&self, inp: T) -> Result<(), FFTError> {
        <Ft as FieldFFT>::fft_io_pc(inp, &self.pc)
    }

    fn get_dims(&self, len: usize) -> ProverResult<(usize, usize, usize), Self::Err> {
        Self::_get_dims(len, self.rho).ok_or(ProverError::TooBig)
    }

    fn dims_ok(&self, n_per_row: usize, n_cols: usize) -> bool {
        let ok = Self::_dims_ok(n_per_row, n_cols, self.rho);
        let pc = n_cols == (1 << self.pc.get_log_len());

        ok && pc
    }
}

/// Ligero-based polynomial commitment
pub type LigeroCommit<D, F> = LcCommit<D, LigeroEncoding<F>>;

/// An evaluation proof for Ligero-based polynomial commitment
pub type LigeroEvalProof<D, F> = LcEvalProof<D, LigeroEncoding<F>>;
