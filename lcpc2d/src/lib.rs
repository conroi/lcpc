// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.
#![deny(missing_docs)]

/*!
lcpc2d is a polynomial commitment scheme based on linear codes
*/

use digest::{Digest, Output};
use err_derive::Error;
use ff::Field;
use fffft::{FFTError, FieldFFT};
use merlin::Transcript;
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::iter::repeat_with;

/// A type to wrap Output<D>
#[derive(Debug, Clone, Serialize)]
pub struct WrappedOutput {
    /// wrapped output
    pub bytes: Vec<u8>,
}

#[cfg(test)]
mod tests;

/// Trait for a field element that can be hashed via [digest::Digest]
pub trait FieldHash {
    /// A representation of `Self` that can be converted to a slice of `u8`.
    type HashRepr: AsRef<[u8]>;

    /// Convert `Self` into a `HashRepr` for hashing
    fn to_hash_repr(&self) -> Self::HashRepr;

    /// Update the digest `d` with the `HashRepr` of `Self`
    fn digest_update<D: Digest>(&self, d: &mut D) {
        d.update(self.to_hash_repr())
    }

    /// Update the [merlin::Transcript] `t` with the `HashRepr` of `Self` with label `l`
    fn transcript_update(&self, t: &mut Transcript, l: &'static [u8]) {
        t.append_message(l, self.to_hash_repr().as_ref())
    }
}

/// Trait for a linear encoding used by the polycommit
pub trait LcEncoding: Clone + std::fmt::Debug + serde::Serialize {
    /// Field over which coefficients are defined
    type F: Field + FieldHash + std::fmt::Debug + Clone + serde::Serialize;

    /// Error type for encoding
    type Err: std::fmt::Debug + std::error::Error + Send;

    /// Encoding function
    fn encode<T: AsMut<[Self::F]>>(inp: T) -> Result<(), Self::Err>;
}

// local accessors for enclosed types
type FldT<E> = <E as LcEncoding>::F;
type ErrT<E> = <E as LcEncoding>::Err;

/// Ligero encoding type
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

    fn encode<T: AsMut<[Ft]>>(inp: T) -> Result<(), FFTError> {
        <Ft as FieldFFT>::fft_io(inp)
    }
}

/// Err variant for prover operations
#[derive(Debug, Error)]
pub enum ProverError<ErrT>
where
    ErrT: std::fmt::Debug + std::error::Error + 'static,
{
    /// bad rho value
    #[error(display = "bad rho value --- must be between 0 and 1")]
    Rho,
    /// size too big
    #[error(display = "size is too big (n_cols overflowed). increase rho?")]
    TooBig,
    /// error encoding a vector
    #[error(display = "encoding error: {:?}", _0)]
    Encode(#[source] ErrT),
    /// inconsistent LcCommit fields
    #[error(display = "inconsistent commitment fields")]
    Commit,
    /// bad column number
    #[error(display = "bad column number")]
    ColumnNumber,
    /// bad outer tensor
    #[error(display = "outer tensor: wrong size")]
    OuterTensor,
}

/// result of a prover operation
pub type ProverResult<T, ErrT> = Result<T, ProverError<ErrT>>;

/// Err variant for verifier operations
#[derive(Debug, Error)]
pub enum VerifierError<ErrT>
where
    ErrT: std::fmt::Debug + std::error::Error + 'static,
{
    /// bad rho value
    #[error(display = "bad rho value --- must be between 0 and 1")]
    Rho,
    /// wrong number of column openings in proof
    #[error(display = "wrong number of column openings in proof")]
    NumColOpens,
    /// failed to verify column merkle path
    #[error(display = "column verification: merkle path failed")]
    ColumnPath,
    /// failed to verify column dot product for poly eval
    #[error(display = "column verification: eval dot product failed")]
    ColumnEval,
    /// failed to verify column dot product for degree test
    #[error(display = "column verification: degree test dot product failed")]
    ColumnDegree,
    /// bad outer tensor
    #[error(display = "outer tensor: wrong size")]
    OuterTensor,
    /// bad inner tensor
    #[error(display = "inner tensor: wrong size")]
    InnerTensor,
    /// error encoding a vector
    #[error(display = "encoding error: {:?}", _0)]
    Encode(#[source] ErrT),
}

/// result of a verifier operation
pub type VerifierResult<T, ErrT> = Result<T, VerifierError<ErrT>>;

/// a commitment
#[derive(Debug, Clone)]
pub struct LcCommit<D, E>
where
    D: Digest,
    E: LcEncoding,
{
    comm: Vec<FldT<E>>,
    coeffs: Vec<FldT<E>>,
    rho: f64,
    n_col_opens: usize,
    n_degree_tests: usize,
    n_rows: usize,
    n_cols: usize,
    n_per_row: usize,
    hashes: Vec<Output<D>>,
}

impl<D, E> LcCommit<D, E>
where
    D: Digest,
    E: LcEncoding,
{
    /// returns the Merkle root of this polynomial commitment (which is the commitment itself)
    pub fn get_root(&self) -> Option<Output<D>> {
        self.hashes.last().cloned()
    }
}

/// LigeroCommit type -- specialization of LcCommit
pub type LigeroCommit<D, F> = LcCommit<D, LigeroEncoding<F>>;

/// A column opening and the corresponding Merkle path.
#[derive(Debug, Clone)]
pub struct LcColumn<D, E>
where
    D: Digest,
    E: LcEncoding,
{
    col: Vec<FldT<E>>,
    path: Vec<Output<D>>,
}

/// A column opening and the corresponding Merkle path.
#[derive(Debug, Clone, Serialize)]
pub struct WrappedLcColumn<E>
where
    E: LcEncoding,
{
    col: Vec<FldT<E>>,
    path: Vec<WrappedOutput>,
}

impl<D, E> LcColumn<D, E>
where
    D: Digest,
    E: LcEncoding,
{
    // used locally to hash columns into the transcript
    fn wrapped(&self) -> WrappedLcColumn<E> {
        let path_wrapped = (0..self.path.len())
            .map(|i| WrappedOutput {
                bytes: self.path[i].to_vec(),
            })
            .collect();

        WrappedLcColumn {
            col: self.col.clone(),
            path: path_wrapped,
        }
    }

    /// unwrap WrappedLcColumn
    pub fn unwrapped(inp: &WrappedLcColumn<E>) -> LcColumn<D, E> {
        let path_unwrapped = (0..inp.path.len())
            .map(|_i| <Output<D> as Default>::default())
            .collect();

        LcColumn {
            col: inp.col.clone(),
            path: path_unwrapped,
        }
    }
}

/// LigeroColumn type --- specialization of LcColumn
pub type LigeroColumn<D, F> = LcColumn<D, LigeroEncoding<F>>;

/// WrappedLigeroColumn type --- specialization of WrappedLcColumn
pub type WrappedLigeroColumn<F> = WrappedLcColumn<LigeroEncoding<F>>;

/// An evaluation and proof of its correctness and of the low-degreeness of the commitment.
#[derive(Debug, Clone)]
pub struct LcEvalProof<D, E>
where
    D: Digest,
    E: LcEncoding,
{
    n_cols: usize,
    p_eval: Vec<FldT<E>>,
    p_random_vec: Vec<Vec<FldT<E>>>,
    columns: Vec<LcColumn<D, E>>,
}

/// An evaluation and proof of its correctness and of the low-degreeness of the commitment.
#[derive(Debug, Clone, Serialize)]
pub struct WrappedLcEvalProof<E>
where
    E: LcEncoding,
{
    n_cols: usize,
    p_eval: Vec<FldT<E>>,
    p_random_vec: Vec<Vec<FldT<E>>>,
    columns: Vec<WrappedLcColumn<E>>,
}

// used locally to hash columns into the transcript
impl<D, E> LcEvalProof<D, E>
where
    D: Digest,
    E: LcEncoding,
{
    /// wrapped method
    pub fn wrapped(&self) -> WrappedLcEvalProof<E> {
        let columns_wrapped = (0..self.columns.len())
            .map(|i| self.columns[i].wrapped())
            .collect();

        WrappedLcEvalProof {
            n_cols: self.n_cols,
            p_eval: self.p_eval.clone(),
            p_random_vec: self.p_random_vec.clone(),
            columns: columns_wrapped,
        }
    }

    /// unwrapped method
    pub fn unwrapped(inp: &WrappedLcEvalProof<E>) -> LcEvalProof<D, E> {
        let columns_unwrapped = (0..inp.columns.len())
            .map(|i| LcColumn::unwrapped(&inp.columns[i]))
            .collect();

        LcEvalProof {
            n_cols: inp.n_cols,
            p_eval: inp.p_eval.clone(),
            p_random_vec: inp.p_random_vec.clone(),
            columns: columns_unwrapped,
        }
    }
}

/// LigerEvalProof type --- specialization of LcEvalProof type
pub type LigeroEvalProof<D, F> = LcEvalProof<D, LigeroEncoding<F>>;

/// WrappedLigerEvalProof type --- specialization of WrappedLcEvalProof type
pub type WrappedLigeroEvalProof<F> = WrappedLcEvalProof<LigeroEncoding<F>>;

// parallelization limit when working on columns
const LOG_MIN_NCOLS: usize = 5;

/// Commit to a univariate polynomial whose coefficients are `coeffs` using Reed-Solomon rate `0 < rho < 1`.
pub fn commit<D, E>(
    coeffs: &[FldT<E>],
    rho: f64,
    n_degree_tests: usize,
    n_col_opens: usize,
) -> ProverResult<LcCommit<D, E>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    let (n_rows, n_per_row, n_cols) = get_dims(coeffs.len(), rho)?;
    commit_with_dims(
        coeffs,
        rho,
        n_degree_tests,
        n_col_opens,
        n_rows,
        n_per_row,
        n_cols,
    )
}

/// Commit to a polynomial whose coeffs are `coeffs_in` using the given rate and dimensions.
pub fn commit_with_dims<D, E>(
    coeffs_in: &[FldT<E>],
    rho: f64,
    n_degree_tests: usize,
    n_col_opens: usize,
    n_rows: usize,
    n_per_row: usize,
    n_cols: usize,
) -> ProverResult<LcCommit<D, E>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // check that parameters are ok
    assert!(n_rows * n_per_row >= coeffs_in.len());
    assert!((n_rows - 1) * n_per_row < coeffs_in.len());
    assert!(n_cols.is_power_of_two());
    assert!(n_cols as f64 * rho >= n_per_row as f64);

    // matrix (encoded as a vector)
    // XXX(zk) pad coeffs
    let mut coeffs = vec![FldT::<E>::zero(); n_rows * n_per_row];
    let mut comm = vec![FldT::<E>::zero(); n_rows * n_cols];

    // local copy of coeffs with padding
    coeffs
        .par_chunks_mut(n_per_row)
        .zip(coeffs_in.par_chunks(n_per_row))
        .for_each(|(c, c_in)| {
            c[..c_in.len()].copy_from_slice(c_in);
        });

    // now compute FFTs
    comm.par_chunks_mut(n_cols)
        .zip(coeffs.par_chunks(n_per_row))
        .try_for_each(|(r, c)| {
            r[..c.len()].copy_from_slice(c);
            <E as LcEncoding>::encode(r)
        })?;

    // compute Merkle tree
    let mut ret = LcCommit {
        comm,
        coeffs,
        rho,
        n_degree_tests,
        n_col_opens,
        n_rows,
        n_cols,
        n_per_row,
        hashes: vec![<Output<D> as Default>::default(); 2 * n_cols - 1],
    };
    merkleize(&mut ret)?;

    Ok(ret)
}

fn get_dims<ErrT>(len: usize, rho: f64) -> ProverResult<(usize, usize, usize), ErrT>
where
    ErrT: std::error::Error,
{
    if rho <= 0f64 || rho >= 1f64 {
        return Err(ProverError::<ErrT>::Rho);
    }

    // compute #cols, which must be a power of 2 because of FFT
    let nc = (((len as f64).sqrt() / rho).ceil() as usize)
        .checked_next_power_of_two()
        .ok_or(ProverError::<ErrT>::TooBig)?;

    // minimize nr subject to #cols and rho
    let np = ((nc as f64) * rho).floor() as usize;
    let nr = len / np + (len % np != 0) as usize;
    assert!(np * nr >= len);
    assert!(np * (nr - 1) < len);

    Ok((nr, np, nc))
}

fn merkleize<D, E>(comm: &mut LcCommit<D, E>) -> ProverResult<(), ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // make sure commitment is self consistent
    check_comm(comm)?;

    // step 1: hash each column of the commitment (we always reveal a full column)
    let hashes = &mut comm.hashes[..comm.n_cols];
    hash_columns::<D, E>(&comm.comm, hashes, comm.n_rows, comm.n_cols, 0);

    // step 2: compute rest of Merkle tree
    let (hin, hout) = comm.hashes.split_at_mut(comm.n_cols);
    merkle_tree::<D>(hin, hout);

    Ok(())
}

#[cfg(test)]
fn merkleize_ser<D, E>(comm: &mut LcCommit<D, E>) -> ProverResult<(), ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
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

fn check_comm<D, E>(comm: &LcCommit<D, E>) -> ProverResult<(), ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    let comm_sz = comm.comm.len() != comm.n_rows * comm.n_cols;
    let coeff_sz = comm.coeffs.len() != comm.n_rows * comm.n_per_row;
    let rate = comm.n_cols as f64 * comm.rho < comm.n_per_row as f64;
    let pow = !comm.n_cols.is_power_of_two();
    let hashlen = comm.hashes.len() != 2 * comm.n_cols - 1;

    if comm_sz || coeff_sz || rate || pow || hashlen {
        Err(ProverError::Commit)
    } else {
        Ok(())
    }
}

fn hash_columns<D, E>(
    comm: &[FldT<E>],
    hashes: &mut [Output<D>],
    n_rows: usize,
    n_cols: usize,
    offset: usize,
) where
    D: Digest,
    E: LcEncoding,
{
    if hashes.len() <= (1 << LOG_MIN_NCOLS) {
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
            || hash_columns::<D, E>(comm, lo, n_rows, n_cols, offset),
            || hash_columns::<D, E>(comm, hi, n_rows, n_cols, offset + half_cols),
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
        merkle_tree::<D>(outs, rems)
    }
}

fn merkle_layer<D>(ins: &[Output<D>], outs: &mut [Output<D>])
where
    D: Digest,
{
    assert_eq!(ins.len(), 2 * outs.len());

    if ins.len() <= (1 << LOG_MIN_NCOLS) {
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

// Open the commitment to one column
fn open_column<D, E>(
    comm: &LcCommit<D, E>,
    mut column: usize,
) -> ProverResult<LcColumn<D, E>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // make sure arguments are well formed
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
    let mut hashes = &comm.hashes[..];
    let path_len = log2(comm.n_cols);
    let mut path = Vec::with_capacity(path_len);
    for _ in 0..path_len {
        let other = (column & !1) | (!column & 1);
        assert_eq!(other ^ column, 1);
        path.push(hashes[other].clone());
        let (_, hashes_new) = hashes.split_at((hashes.len() + 1) / 2);
        hashes = hashes_new;
        column >>= 1;
    }
    assert_eq!(column, 0);

    Ok(LcColumn { col, path })
}

fn log2(v: usize) -> usize {
    (63 - (v.next_power_of_two() as u64).leading_zeros()) as usize
}

/// Verify the evaluation of a committed polynomial and return the result
#[allow(clippy::too_many_arguments)]
pub fn verify<D, E>(
    root: &Output<D>,
    outer_tensor: &[FldT<E>],
    inner_tensor: &[FldT<E>],
    proof: &LcEvalProof<D, E>,
    rho: f64,
    n_degree_tests: usize,
    n_col_opens: usize,
    tr: &mut Transcript,
) -> VerifierResult<FldT<E>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // make sure arguments are well formed
    if n_col_opens != proof.columns.len() || n_col_opens == 0 {
        return Err(VerifierError::NumColOpens);
    }
    let n_rows = proof.columns[0].col.len();
    let n_cols = proof.n_cols;
    let n_per_row = proof.p_eval.len();
    if inner_tensor.len() != n_per_row {
        return Err(VerifierError::InnerTensor);
    }
    if outer_tensor.len() != n_rows {
        return Err(VerifierError::OuterTensor);
    }
    if rho <= 0f64 || rho >= 1f64 || n_cols as f64 * rho < n_per_row as f64 {
        return Err(VerifierError::Rho);
    }

    // step 1: random tensor for degree test and random columns to test
    // step 1a: extract random tensor from transcript
    // we run multiple instances of this to boost soundness
    let mut rand_tensor_vec: Vec<Vec<FldT<E>>> = Vec::new();
    let mut p_random_fft: Vec<Vec<FldT<E>>> = Vec::new();
    for i in 0..n_degree_tests {
        let rand_tensor: Vec<FldT<E>> = {
            let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
            tr.challenge_bytes(b"ligero-pc//eval//degree_test", &mut key);
            let mut deg_test_rng = ChaCha20Rng::from_seed(key);
            // XXX(optimization) could expand seed in parallel instead of in series
            repeat_with(|| FldT::<E>::random(&mut deg_test_rng))
                .take(n_rows)
                .collect()
        };

        rand_tensor_vec.push(rand_tensor);

        // step 1b: eval encoding of p_random
        {
            let mut tmp = Vec::with_capacity(n_cols);
            tmp.extend_from_slice(&proof.p_random_vec[i][..]);
            tmp.resize(n_cols, FldT::<E>::zero());
            <E as LcEncoding>::encode(&mut tmp)?;
            p_random_fft.push(tmp);
        };

        // step 1c: push p_random and p_eval into transcript
        proof.p_random_vec[i]
            .iter()
            .for_each(|coeff| coeff.transcript_update(tr, b"ligero-pc//eval//p_random"));
    }

    proof
        .p_eval
        .iter()
        .for_each(|coeff| coeff.transcript_update(tr, b"ligero-pc//eval//p_eval"));

    // step 1d: extract columns to open
    let cols_to_open: Vec<usize> = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr.challenge_bytes(b"ligero-pc//eval//cols_to_open", &mut key);
        let mut cols_rng = ChaCha20Rng::from_seed(key);
        // XXX(optimization) could expand seed in parallel instead of in series
        let col_range = Uniform::new(0usize, n_cols);
        repeat_with(|| col_range.sample(&mut cols_rng))
            .take(n_col_opens)
            .collect()
    };

    // step 2: p_eval fft for column checks
    let p_eval_fft = {
        let mut tmp = Vec::with_capacity(n_cols);
        tmp.extend_from_slice(&proof.p_eval[..]);
        tmp.resize(n_cols, FldT::<E>::zero());
        <E as LcEncoding>::encode(&mut tmp)?;
        tmp
    };

    // step 3: check p_random, p_eval, and col paths
    cols_to_open
        .par_iter()
        .zip(&proof.columns[..])
        .try_for_each(|(&col_num, column)| {
            let rand = {
                let mut rand = true;
                for i in 0..n_degree_tests {
                    rand &=
                        verify_column_value(column, &rand_tensor_vec[i], &p_random_fft[i][col_num]);
                }
                rand
            };

            let eval = verify_column_value(column, outer_tensor, &p_eval_fft[col_num]);
            let path = verify_column_path(column, col_num, root);
            match (rand, eval, path) {
                (false, _, _) => Err(VerifierError::ColumnDegree),
                (_, false, _) => Err(VerifierError::ColumnEval),
                (_, _, false) => Err(VerifierError::ColumnPath),
                _ => Ok(()),
            }
        })?;

    // step 4: evaluate and return
    Ok(inner_tensor
        .par_iter()
        .zip(&proof.p_eval[..])
        .fold(FldT::<E>::zero, |a, (t, e)| a + *t * e)
        .reduce(FldT::<E>::zero, |a, v| a + v))
}

// Check a column opening
fn verify_column_path<D, E>(column: &LcColumn<D, E>, col_num: usize, root: &Output<D>) -> bool
where
    D: Digest,
    E: LcEncoding,
{
    let mut digest = D::new();
    digest.update(<Output<D> as Default>::default());
    for e in &column.col[..] {
        e.digest_update(&mut digest);
    }

    // check Merkle path
    let mut hash = digest.finalize_reset();
    let mut col = col_num;
    for p in &column.path[..] {
        if col % 2 == 0 {
            digest.update(&hash);
            digest.update(p);
        } else {
            digest.update(p);
            digest.update(&hash);
        }
        hash = digest.finalize_reset();
        col >>= 1;
    }

    &hash == root
}

// check column value
fn verify_column_value<D, E>(
    column: &LcColumn<D, E>,
    tensor: &[FldT<E>],
    poly_eval: &FldT<E>,
) -> bool
where
    D: Digest,
    E: LcEncoding,
{
    let tensor_eval = tensor
        .iter()
        .zip(&column.col[..])
        .fold(FldT::<E>::zero(), |a, (t, e)| a + *t * e);

    poly_eval == &tensor_eval
}

#[cfg(test)]
// Check a column opening
fn verify_column<D, E>(
    column: &LcColumn<D, E>,
    col_num: usize,
    root: &Output<D>,
    tensor: &[FldT<E>],
    poly_eval: &FldT<E>,
) -> bool
where
    D: Digest,
    E: LcEncoding,
{
    verify_column_path(column, col_num, root) && verify_column_value(column, tensor, poly_eval)
}

/// Evaluate the committed polynomial using the supplied "outer" tensor
/// and generate a proof of (1) low-degreeness and (2) correct evaluation.
pub fn prove<D, E>(
    comm: &LcCommit<D, E>,
    outer_tensor: &[FldT<E>],
    tr: &mut Transcript,
) -> ProverResult<LcEvalProof<D, E>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // make sure arguments are well formed
    check_comm(comm)?;
    if outer_tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // first, evaluate the polynomial on a random tensor (low-degree test)
    // we repeat this to boost soundness
    let mut p_random_vec: Vec<Vec<FldT<E>>> = Vec::new();
    for _i in 0..comm.n_degree_tests {
        let p_random = {
            let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
            tr.challenge_bytes(b"ligero-pc//eval//degree_test", &mut key);
            let mut deg_test_rng = ChaCha20Rng::from_seed(key);
            // XXX(optimization) could expand seed in parallel instead of in series
            let rand_tensor: Vec<FldT<E>> = repeat_with(|| FldT::<E>::random(&mut deg_test_rng))
                .take(comm.n_rows)
                .collect();
            let mut tmp = vec![FldT::<E>::zero(); comm.n_per_row];
            collapse_columns::<E>(
                &comm.coeffs,
                &rand_tensor,
                &mut tmp,
                comm.n_rows,
                comm.n_per_row,
                0,
            );
            tmp
        };
        // add p_random to the transcript
        p_random
            .iter()
            .for_each(|coeff| coeff.transcript_update(tr, b"ligero-pc//eval//p_random"));

        p_random_vec.push(p_random);
    }

    // next, evaluate the polynomial using the supplied tensor
    let p_eval = {
        let mut tmp = vec![FldT::<E>::zero(); comm.n_per_row];
        collapse_columns::<E>(
            &comm.coeffs,
            outer_tensor,
            &mut tmp,
            comm.n_rows,
            comm.n_per_row,
            0,
        );
        tmp
    };
    // add p_eval to the transcript
    p_eval
        .iter()
        .for_each(|coeff| coeff.transcript_update(tr, b"ligero-pc//eval//p_eval"));

    // now extract the column numbers to open
    // XXX(F-S) should we do this column-by-column, updating the transcript for each???
    //          It doesn't seem necessary to me...
    let columns: Vec<LcColumn<D, E>> = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr.challenge_bytes(b"ligero-pc//eval//cols_to_open", &mut key);
        let mut cols_rng = ChaCha20Rng::from_seed(key);
        // XXX(optimization) could expand seed in parallel instead of in series
        let col_range = Uniform::new(0usize, comm.n_cols);
        let cols_to_open: Vec<usize> = repeat_with(|| col_range.sample(&mut cols_rng))
            .take(comm.n_col_opens)
            .collect();
        cols_to_open
            .par_iter()
            .map(|&col| open_column(comm, col))
            .collect::<ProverResult<Vec<LcColumn<D, E>>, ErrT<E>>>()?
    };

    Ok(LcEvalProof {
        n_cols: comm.n_cols,
        p_eval,
        p_random_vec,
        columns,
    })
}

// Evaluate the committed polynomial using the "outer" tensor
#[cfg(test)]
fn eval_outer<D, E>(
    comm: &LcCommit<D, E>,
    tensor: &[FldT<E>],
) -> ProverResult<Vec<FldT<E>>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // make sure arguments are well formed
    check_comm(comm)?;
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // allocate result and compute
    let mut poly = vec![FldT::<E>::zero(); comm.n_per_row];
    collapse_columns::<E>(
        &comm.coeffs,
        tensor,
        &mut poly,
        comm.n_rows,
        comm.n_per_row,
        0,
    );

    Ok(poly)
}

fn collapse_columns<E>(
    coeffs: &[FldT<E>],
    tensor: &[FldT<E>],
    poly: &mut [FldT<E>],
    n_rows: usize,
    n_per_row: usize,
    offset: usize,
) where
    E: LcEncoding,
{
    if poly.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: run the computation
        // row-by-row, compute elements of dot product
        for (row, tensor_val) in tensor.iter().enumerate() {
            for (col, val) in poly.iter_mut().enumerate() {
                let entry = row * n_per_row + offset + col;
                *val += coeffs[entry] * tensor_val;
            }
        }
    } else {
        // recursive case: split and execute in parallel
        let half_cols = poly.len() / 2;
        let (lo, hi) = poly.split_at_mut(half_cols);
        rayon::join(
            || collapse_columns::<E>(coeffs, tensor, lo, n_rows, n_per_row, offset),
            || collapse_columns::<E>(coeffs, tensor, hi, n_rows, n_per_row, offset + half_cols),
        );
    }
}

#[cfg(test)]
fn eval_outer_ser<D, E>(
    comm: &LcCommit<D, E>,
    tensor: &[FldT<E>],
) -> ProverResult<Vec<FldT<E>>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    check_comm(comm)?;
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    let mut poly = vec![FldT::<E>::zero(); comm.n_per_row];
    for (row, tensor_val) in tensor.iter().enumerate() {
        for (col, val) in poly.iter_mut().enumerate() {
            let entry = row * comm.n_per_row + col;
            *val += comm.coeffs[entry] * tensor_val;
        }
    }

    Ok(poly)
}

#[cfg(test)]
fn eval_outer_fft<D, E>(
    comm: &LcCommit<D, E>,
    tensor: &[FldT<E>],
) -> ProverResult<Vec<FldT<E>>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    check_comm(comm)?;
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    let mut poly_fft = vec![FldT::<E>::zero(); comm.n_cols];
    for (coeffs, tensorval) in comm.comm.chunks(comm.n_cols).zip(tensor.iter()) {
        for (coeff, polyval) in coeffs.iter().zip(poly_fft.iter_mut()) {
            *polyval += *coeff * tensorval;
        }
    }

    Ok(poly_fft)
}
