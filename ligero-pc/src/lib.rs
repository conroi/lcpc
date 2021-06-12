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

use fffft::{FFTError, FieldFFT};
use lcpc2d::{FieldHash, LcCommit, LcEncoding, LcEvalProof};
use serde::Serialize;

#[cfg(test)]
mod tests;

/// Encoding definition for Ligero-based polycommit
#[derive(Clone, Debug, Serialize)]
pub struct LigeroEncoding<Ft> {
    _p: std::marker::PhantomData<Ft>,
}

impl<Ft> LcEncoding for LigeroEncoding<Ft>
where
    Ft: FieldFFT + FieldHash + serde::Serialize,
{
    type F = Ft;
    type Err = FFTError;

    const LABEL_DT: &'static [u8] = b"ligero-pc//DT";
    const LABEL_PR: &'static [u8] = b"ligero-pc//PR";
    const LABEL_PE: &'static [u8] = b"ligero-pc//PE";
    const LABEL_CO: &'static [u8] = b"ligero-pc//CO";

    fn encode<T: AsMut<[Ft]>>(inp: T) -> Result<(), FFTError> {
        <Ft as FieldFFT>::fft_io(inp)
    }
}

/// Ligero-based polynomial commitment
pub type LigeroCommit<D, F> = LcCommit<D, LigeroEncoding<F>>;

/// An evaluation proof for Ligero-based polynomial commitment
pub type LigeroEvalProof<D, F> = LcEvalProof<D, LigeroEncoding<F>>;
