// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/*!
ligero-pc prover functionality
*/

use err_derive::Error;
use fffft::{FFTError, FieldFFT};
use num_integer::Roots;
use rayon::prelude::*;

#[cfg(test)]
mod tests;

/// Err variant for prover operations
#[derive(Debug, Error)]
pub enum ProverError {
    /// bad rho value
    #[error(display = "bad rho value --- must be between 0 and 1")]
    BadRho,
    /// size too big
    #[error(display = "size is too big --- ncols overflowed. increase rho?")]
    TooBig,
    /// error computing FFT
    #[error(display = "fft error: {:?}", _0)]
    FFT(#[source] FFTError),
}

/// Commit to the polynomial whose coefficients are `coeffs` using Reed-Solomon rate `0 < rho < 1`.
pub fn commit<F: FieldFFT, T: AsRef<[F]>>(coeffs: T, rho: f64) -> Result<Vec<F>, ProverError> {
    let coeffs = coeffs.as_ref();
    let (nrows, n_per_row, ncols) = get_dims(coeffs.len(), rho)?;

    // return value
    let mut res = vec![F::zero(); nrows * ncols];
    let res_mut: &mut [F] = res.as_mut();
    res_mut.par_chunks_mut(ncols)
        .zip(coeffs.par_chunks(n_per_row))
        .try_for_each(|(r, c)| {
            r[..c.len()].copy_from_slice(c);
            <F as FieldFFT>::fft_io(r)
        })?;

    Ok(res)
}

fn get_dims(len: usize, rho: f64) -> Result<(usize, usize, usize), ProverError> {
    if rho <= 0f64 || rho >= 1f64 {
        return Err(ProverError::BadRho);
    }

    // compute #cols, which must be a power of 2
    let nr = len.sqrt(); // initial estimate of #entries per row
    let nc = (((nr as f64) / rho).ceil() as usize)
        .checked_next_power_of_two()
        .ok_or(ProverError::TooBig)?;

    // now minimize #rows subject to #cols and rho
    let np = ((nc as f64) * rho).floor() as usize;
    let nr = len / np + (len % np != 0) as usize;
    assert!(np * nr >= len);
    assert!(np * (nr - 1) < len);

    Ok((nr, np, nc))
}
