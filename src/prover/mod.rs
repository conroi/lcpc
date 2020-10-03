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

use digest::{Digest, Output};
use err_derive::Error;
use fffft::{FFTError, FieldFFT};
use num_integer::Roots;
use rayon::prelude::*;
use std::marker::PhantomData;

#[cfg(test)]
mod tests;

/// Err variant for prover operations
#[derive(Debug, Error)]
pub enum ProverError {
    /// bad rho value
    #[error(display = "bad rho value --- must be between 0 and 1")]
    BadRho,
    /// size too big
    #[error(display = "size is too big (n_cols overflowed). increase rho?")]
    TooBig,
    /// error computing FFT
    #[error(display = "fft error: {:?}", _0)]
    FFT(#[source] FFTError),
    /// inconsistent LigeroCommit fields
    #[error(display = "inconsistent commitment fields")]
    BadCommit,
}

// XXX(hack): need a Send+Sync phantom type so that LigeroCommit is Send+Sync
struct MyPhantom<D> {
    _ghost: PhantomData<*const D>,
}

unsafe impl<D> Sync for MyPhantom<D> where D: Digest {}

/// a commitment
pub struct LigeroCommit<D, F>
where
    D: Digest,
    F: FieldFFT + AsRef<[u8]>,
{
    comm: Vec<F>,
    coeffs: Vec<F>,
    rho: f64,
    n_rows: usize,
    n_cols: usize,
    n_per_row: usize,
    hashes: Vec<Output<D>>,
    _ghost: MyPhantom<D>,
}

/// result of a prover operation
pub type ProverResult<T> = Result<T, ProverError>;

/// Commit to the polynomial whose coefficients are `coeffs` using Reed-Solomon rate `0 < rho < 1`.
pub fn commit<D, F>(coeffs: Vec<F>, rho: f64) -> ProverResult<LigeroCommit<D, F>>
where
    D: Digest,
    F: FieldFFT + AsRef<[u8]>,
{
    let (n_rows, n_per_row, n_cols) = get_dims(coeffs.len(), rho)?;

    // matrix (encoded as a vector)
    let mut comm = vec![F::zero(); n_rows * n_cols];
    // compute the FFT of each row
    comm.par_chunks_mut(n_cols)
        .zip(coeffs.par_chunks(n_per_row))
        .try_for_each(|(r, c)| {
            r[..c.len()].copy_from_slice(c);
            <F as FieldFFT>::fft_io(r)
        })?;

    // compute Merkle tree
    let mut ret = LigeroCommit {
        comm,
        coeffs,
        rho,
        n_rows,
        n_cols,
        n_per_row,
        hashes: vec![<Output<D> as Default>::default(); 2 * n_cols - 1],
        _ghost: MyPhantom {
            _ghost: PhantomData,
        },
    };
    merkleize::<D, F>(&mut ret)?;

    Ok(ret)
}

fn get_dims(len: usize, rho: f64) -> ProverResult<(usize, usize, usize)> {
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

/// Merkleize the output of [commit]
pub fn merkleize<D, F>(comm: &mut LigeroCommit<D, F>) -> ProverResult<()>
where
    D: Digest,
    F: FieldFFT + AsRef<[u8]>,
{
    // make sure commitment is self consistent
    check_comm(comm)?;

    // step 1: hash each column of the commitment (we always reveal a full column)
    let hashes = &mut comm.hashes[..comm.n_cols];
    hash_columns::<D, F>(&comm.comm, hashes, comm.n_rows, comm.n_cols, 0);

    // step 2: compute rest of Merkle tree
    let (hin, hout) = comm.hashes.split_at_mut(comm.n_cols);
    merkle_tree::<D>(hin, hout);

    Ok(())
}

pub(crate) fn check_comm<D, F>(comm: &LigeroCommit<D, F>) -> ProverResult<()>
where
    D: Digest,
    F: FieldFFT + AsRef<[u8]>,
{
    let comm_sz = comm.comm.len() != comm.n_rows * comm.n_cols;
    let coeff_sz_big = comm.coeffs.len() > comm.n_rows * comm.n_per_row;
    let coeff_sz_sm = comm.coeffs.len() < (comm.n_rows - 1) * comm.n_per_row;
    let rate = comm.n_cols as f64 * comm.rho < comm.n_per_row as f64;
    let pow = !comm.n_cols.is_power_of_two();
    let hashlen = comm.hashes.len() != 2 * comm.n_cols - 1;

    if comm_sz || coeff_sz_big || coeff_sz_sm || rate || pow || hashlen {
        Err(ProverError::BadCommit)
    } else {
        Ok(())
    }
}

fn hash_columns<D, F>(
    comm: &[F],
    hashes: &mut [Output<D>],
    n_rows: usize,
    n_cols: usize,
    offset: usize,
) where
    D: Digest,
    F: FieldFFT + AsRef<[u8]>,
{
    const LOG_MIN_NCOLS: usize = 6;
    if hashes.len() <= (1usize << LOG_MIN_NCOLS) {
        // base case: run the computation
        // 1. prepare the digests for each column
        let mut digests = Vec::with_capacity(hashes.len());
        for _ in 0..hashes.len() {
            // column hashes start with a block of 0's
            let mut dig = D::new();
            dig.update(<Output<D> as Default>::default());
            digests.push(dig);
        }
        // 2. for each row, update the digests for each column
        for row in 0..n_rows {
            for (col, digest) in digests.iter_mut().enumerate() {
                digest.update(comm[row * n_cols + offset + col]);
            }
        }
        // 3. finalize each digest and write the results back
        for (col, digest) in digests.into_iter().enumerate() {
            hashes[col] = digest.finalize();
        }
    } else {
        // recursive case: split and execute in parallel
        let half_cols = hashes.len() / 2;
        let (lo, hi) = hashes.split_at_mut(half_cols);
        rayon::join(
            || hash_columns::<D, F>(comm, lo, n_rows, n_cols, offset),
            || hash_columns::<D, F>(comm, hi, n_rows, n_cols, offset + half_cols),
        );
    }
}

fn merkle_tree<D>(ins: &[Output<D>], outs: &mut [Output<D>])
where
    D: Digest,
{
    // array should always be of length 2^k - 1
    assert_eq!(ins.len(), outs.len() + 1);

    let (outs, rems) = outs.split_at_mut((outs.len() + 1) / 2);
    merkle_layer::<D>(ins, outs);

    if !rems.is_empty() {
        return merkle_tree::<D>(outs, rems);
    }
}

fn merkle_layer<D>(ins: &[Output<D>], outs: &mut [Output<D>])
where
    D: Digest,
{
    const LOG_MIN_NCOLS: usize = 4;
    assert_eq!(ins.len(), 2 * outs.len());

    if ins.len() <= (1usize << LOG_MIN_NCOLS) {
        // base case: just compute all of the hashes
        for idx in 0..outs.len() {
            let mut digest = D::new();
            digest.update(ins[2 * idx].as_ref());
            digest.update(ins[2 * idx + 1].as_ref());
            outs[idx] = digest.finalize();
        }
    } else {
        // recursive case: split and compute
        let (inl, inr) = ins.split_at(ins.len() / 2);
        let (outl, outr) = outs.split_at_mut(outs.len() / 2);
        rayon::join(
            || merkle_layer::<D>(inl, outl),
            || merkle_layer::<D>(inr, outr),
        );
    }
}
