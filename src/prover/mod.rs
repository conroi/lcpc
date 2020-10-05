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

/* TODO
 *   - convert fns to methods on LigeroCommit
 */

use crate::FieldHash;

use digest::{Digest, Output};
use err_derive::Error;
use fffft::{FFTError, FieldFFT};
use rayon::prelude::*;
use std::marker::PhantomData;

#[cfg(test)]
mod tests;

/// Err variant for prover operations
#[derive(Debug, Error)]
pub enum ProverError {
    /// bad rho value
    #[error(display = "bad rho value --- must be between 0 and 1")]
    Rho,
    /// size too big
    #[error(display = "size is too big (n_cols overflowed). increase rho?")]
    TooBig,
    /// error computing FFT
    #[error(display = "fft error: {:?}", _0)]
    FFT(#[source] FFTError),
    /// inconsistent LigeroCommit fields
    #[error(display = "inconsistent commitment fields")]
    Commit,
    /// bad column number
    #[error(display = "bad column number")]
    ColumnNumber,
    /// bad outer tensor
    #[error(display = "outer tensor: wrong size")]
    OuterTensor,
}

// need a Send+Sync phantom type so that LigeroCommit is Send+Sync
// could just declare LigerCommit to be Send, but spplying the Sync
// only to PhantomData and keeping automatic Sync derivation for
// LigeroCommit could help catch mistakes if LigerCommit changes
#[derive(Clone)]
struct SyncPhantom<D> {
    _ghost: PhantomData<*const D>,
}
unsafe impl<D> Sync for SyncPhantom<D> where D: Digest {}
impl<D> SyncPhantom<D> {
    fn new() -> Self {
        Self {
            _ghost: PhantomData,
        }
    }
}

/// a commitment
#[derive(Clone)]
pub struct LigeroCommit<D, F>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    comm: Vec<F>,
    coeffs: Vec<F>,
    rho: f64,
    n_rows: usize,
    n_cols: usize,
    n_per_row: usize,
    hashes: Vec<Output<D>>,
    _ghost: SyncPhantom<D>,
}

/// result of a prover operation
pub type ProverResult<T> = Result<T, ProverError>;

// parallelization limit when working on columns
const LOG_MIN_NCOLS: usize = 5;

/// Commit to a univariate polynomial whose coefficients are `coeffs` using Reed-Solomon rate `0 < rho < 1`.
pub fn commit_uni<D, F>(coeffs: Vec<F>, rho: f64) -> ProverResult<LigeroCommit<D, F>>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    let (n_rows, n_per_row, n_cols) = get_dims_uni(coeffs.len(), rho)?;
    commit_with_dims(coeffs, rho, n_rows, n_per_row, n_cols)
}

/// Commit to a polynomial whose coeffs are `coeffs` using the given rate and dimensions.
pub fn commit_with_dims<D, F>(
    coeffs: Vec<F>,
    rho: f64,
    n_rows: usize,
    n_per_row: usize,
    n_cols: usize,
) -> ProverResult<LigeroCommit<D, F>>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    // check that parameters are ok
    assert!(n_rows * n_per_row >= coeffs.len());
    assert!(n_rows * (n_per_row - 1) < coeffs.len());
    assert!(n_cols.is_power_of_two());
    assert!(n_cols as f64 * rho >= n_per_row as f64);

    // matrix (encoded as a vector)
    let mut comm = vec![F::zero(); n_rows * n_cols];

    // arrange coeffs in column-major order, then compute FFT
    comm.par_chunks_mut(n_cols)
        .enumerate()
        .try_for_each(|(ridx, row)| {
            for (val, ent) in coeffs.iter().skip(ridx).step_by(n_rows).zip(row.iter_mut()) {
                *ent = *val;
            }
            <F as FieldFFT>::fft_io(row)
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
        _ghost: SyncPhantom::new(),
    };
    merkleize(&mut ret)?;

    Ok(ret)
}

fn get_dims_uni(len: usize, rho: f64) -> ProverResult<(usize, usize, usize)> {
    if rho <= 0f64 || rho >= 1f64 {
        return Err(ProverError::Rho);
    }

    // compute #cols, which must be a power of 2 because of FFT
    let nc = (((len as f64).sqrt() / rho).ceil() as usize)
        .checked_next_power_of_two()
        .ok_or(ProverError::TooBig)?;

    /* constraints:
        np < nc * rho

        np * nr >= len
        ->     nr >= len / np
        ->     nr >= len / (nc * rho)

        (np - 1) * nr < len
        ->     nr < len / (np - 1)
    */

    // minimize nr subject to requirements on np
    let np_max = ((nc as f64) * rho).floor() as usize;
    let nr = len / np_max + (len % np_max != 0) as usize;
    let np = len / nr + (len % nr != 0) as usize;
    assert!(nr * np >= len);
    assert!(nr * (np - 1) < len);

    Ok((nr, np, nc))
}

fn merkleize<D, F>(comm: &mut LigeroCommit<D, F>) -> ProverResult<()>
where
    D: Digest,
    F: FieldFFT + FieldHash,
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

#[cfg(test)]
fn merkleize_ser<D, F>(comm: &mut LigeroCommit<D, F>) -> ProverResult<()>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    check_comm(comm)?;

    let hashes = &mut comm.hashes;

    // hash each column
    for (col, hash) in hashes.iter_mut().enumerate().take(comm.n_cols) {
        let mut digest = D::new();
        digest.update(<Output<D> as Default>::default());
        for row in 0..comm.n_rows {
            comm.comm[row * comm.n_cols + col].digest_update(&mut digest);
        }
        *hash = digest.finalize();
    }

    // compute rest of Merkle tree
    let (mut ins, mut outs) = hashes.split_at_mut(comm.n_cols);
    while !outs.is_empty() {
        for idx in 0..ins.len() / 2 {
            let mut digest = D::new();
            digest.update(ins[2 * idx].as_ref());
            digest.update(ins[2 * idx + 1].as_ref());
            outs[idx] = digest.finalize();
        }
        let (new_ins, new_outs) = outs.split_at_mut((outs.len() + 1) / 2);
        ins = new_ins;
        outs = new_outs;
    }

    Ok(())
}

pub(crate) fn check_comm<D, F>(comm: &LigeroCommit<D, F>) -> ProverResult<()>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    let comm_sz = comm.comm.len() != comm.n_rows * comm.n_cols;
    let coeff_sz_lg = comm.coeffs.len() > comm.n_rows * comm.n_per_row;
    let coeff_sz_sm = comm.coeffs.len() <= comm.n_rows * (comm.n_per_row - 1);
    let rate = comm.n_cols as f64 * comm.rho < comm.n_per_row as f64;
    let pow = !comm.n_cols.is_power_of_two();
    let hashlen = comm.hashes.len() != 2 * comm.n_cols - 1;

    if comm_sz || coeff_sz_lg || coeff_sz_sm || rate || pow || hashlen {
        Err(ProverError::Commit)
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
    F: FieldFFT + FieldHash,
{
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
                comm[row * n_cols + offset + col].digest_update(digest);
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
    assert_eq!(ins.len(), 2 * outs.len());

    if ins.len() <= (1usize << LOG_MIN_NCOLS) {
        // base case: just compute all of the hashes
        let mut digest = D::new();
        for idx in 0..outs.len() {
            digest.update(ins[2 * idx].as_ref());
            digest.update(ins[2 * idx + 1].as_ref());
            outs[idx] = digest.finalize_reset();
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

/// Open the commitment to one column
pub fn open_column<D, F>(
    comm: &LigeroCommit<D, F>,
    column: usize,
) -> ProverResult<(Vec<F>, Vec<Output<D>>)>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    // make sure arguments are well formed
    check_comm(comm)?;
    if column >= comm.n_cols {
        return Err(ProverError::ColumnNumber);
    }

    // column of values
    let col = comm
        .comm
        .iter()
        .skip(column)
        .step_by(comm.n_cols)
        .cloned()
        .collect();

    // Merkle path
    let mask: usize = -2isize as usize;
    let mut column = column;
    let mut hashes = &comm.hashes[..];
    let path_len = log2(comm.n_cols);
    let mut path = Vec::with_capacity(path_len);
    for _ in 0..path_len {
        let other = (column & mask) | (!column & 1);
        assert_eq!(other ^ column, 1);
        path.push(hashes[other].clone());
        let (_, hashes_new) = hashes.split_at((hashes.len() + 1) / 2);
        hashes = hashes_new;
        column >>= 1;
    }
    assert_eq!(column, 0);

    Ok((col, path))
}

fn log2(v: usize) -> usize {
    (63 - (v.next_power_of_two() as u64).leading_zeros()) as usize
}

/// Evaluate the committed polynomial using the "outer" tensor
pub fn eval_outer<D, F>(comm: &LigeroCommit<D, F>, tensor: &[F]) -> ProverResult<Vec<F>>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    // make sure arguments are well formed
    check_comm(comm)?;
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // allocate result and compute
    let mut poly = vec![F::zero(); comm.n_per_row];
    comm.coeffs
        .par_chunks(comm.n_rows)
        .zip(poly.par_iter_mut())
        .for_each(|(col, polyval)| {
            for (c, t) in col.iter().zip(tensor) {
                *polyval += *c * t;
            }
        });

    Ok(poly)
}

#[cfg(test)]
fn eval_outer_ser<D, F>(comm: &LigeroCommit<D, F>, tensor: &[F]) -> ProverResult<Vec<F>>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    check_comm(comm)?;
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    let mut poly = vec![F::zero(); comm.n_per_row];
    for (coeffs, polyval) in comm.coeffs.chunks(comm.n_rows).zip(poly.iter_mut()) {
        for (c, t) in coeffs.iter().zip(tensor) {
            *polyval += *c * t;
        }
    }

    Ok(poly)
}

#[cfg(test)]
fn eval_outer_fft<D, F>(comm: &LigeroCommit<D, F>, tensor: &[F]) -> ProverResult<Vec<F>>
where
    D: Digest,
    F: FieldFFT + FieldHash,
{
    check_comm(comm)?;
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    let mut poly_fft = vec![F::zero(); comm.n_cols];
    for (coeffs, tensorval) in comm.comm.chunks(comm.n_cols).zip(tensor.iter()) {
        for (coeff, polyval) in coeffs.iter().zip(poly_fft.iter_mut()) {
            *polyval += *coeff * tensorval;
        }
    }

    Ok(poly_fft)
}
