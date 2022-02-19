// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::{def_labels, FieldHash, LcCommit, LcEncoding, LcEvalProof, LcRoot};

use blake3::Hasher as Blake3;
use digest::Output;
use ff::Field;
use fffft::{FFTError, FFTPrecomp, FieldFFT};
use itertools::iterate;
use merlin::Transcript;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::iter::repeat_with;
use lcpc_test_fields::ft63::*;

#[derive(Clone, Debug)]
struct LigeroEncoding<Ft> {
    n_per_row: usize, // number of inputs to the encoding
    n_cols: usize,    // number of outputs from the encoding
    pc: FFTPrecomp<Ft>,
}

const N_COL_OPENS: usize = 128usize; // arbitrary, not secure
impl<Ft> LigeroEncoding<Ft>
where
    Ft: FieldFFT,
{
    fn _get_dims(len: usize, rho: f64) -> Option<(usize, usize, usize)> {
        assert!(rho > 0f64);
        assert!(rho < 1f64);

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
        let nr = (len + np - 1) / np;
        assert!(np * nr >= len);
        assert!(np * (nr - 1) < len);

        Some((nr, np, nc))
    }

    fn _dims_ok(n_per_row: usize, n_cols: usize) -> bool {
        let sz = n_per_row < n_cols;
        let pow = n_cols.is_power_of_two();
        sz && pow
    }

    pub fn new(len: usize, rho: f64) -> Self {
        let (_, n_per_row, n_cols) = Self::_get_dims(len, rho).unwrap();
        assert!(Self::_dims_ok(n_per_row, n_cols));
        let pc = <Ft as FieldFFT>::precomp_fft(n_cols).unwrap();
        Self {
            n_per_row,
            n_cols,
            pc,
        }
    }

    pub fn new_from_dims(n_per_row: usize, n_cols: usize) -> Self {
        assert!(Self::_dims_ok(n_per_row, n_cols));
        let pc = <Ft as FieldFFT>::precomp_fft(n_cols).unwrap();
        assert_eq!(n_cols, 1 << pc.get_log_len());
        Self {
            n_per_row,
            n_cols,
            pc,
        }
    }
}

impl<Ft> LcEncoding for LigeroEncoding<Ft>
where
    Ft: FieldFFT + FieldHash,
{
    type F = Ft;
    type Err = FFTError;

    def_labels!(lcpc2d_test);

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
        N_COL_OPENS
    }

    fn get_n_degree_tests(&self) -> usize {
        2
    }
}

type LigeroCommit<D, F> = LcCommit<D, LigeroEncoding<F>>;

type LigeroEvalProof<D, F> = LcEvalProof<D, LigeroEncoding<F>>;

#[test]
fn log2() {
    use super::log2;

    for idx in 0..31 {
        assert_eq!(log2(1usize << idx), idx);
    }
}

#[test]
fn merkleize() {
    use super::{merkleize, merkleize_ser};

    let mut test_comm = random_comm();
    let mut test_comm_2 = test_comm.clone();

    merkleize(&mut test_comm);
    merkleize_ser(&mut test_comm_2);

    assert_eq!(&test_comm.comm, &test_comm_2.comm);
    assert_eq!(&test_comm.coeffs, &test_comm_2.coeffs);
    assert_eq!(&test_comm.hashes, &test_comm_2.hashes);
}

#[test]
fn eval_outer() {
    use super::{eval_outer, eval_outer_ser};

    let test_comm = random_comm();
    let mut rng = rand::thread_rng();
    let tensor: Vec<Ft63> = repeat_with(|| Ft63::random(&mut rng))
        .take(test_comm.n_rows)
        .collect();

    let res1 = eval_outer(&test_comm, &tensor[..]).unwrap();
    let res2 = eval_outer_ser(&test_comm, &tensor[..]).unwrap();

    assert_eq!(&res1[..], &res2[..]);
}

#[test]
fn open_column() {
    use super::{merkleize, open_column, verify_column};

    let mut rng = rand::thread_rng();

    let test_comm = {
        let mut tmp = random_comm();
        merkleize(&mut tmp);
        tmp
    };

    let root = test_comm.get_root();
    for _ in 0..64 {
        let col_num = rng.gen::<usize>() % test_comm.n_cols;
        let column = open_column(&test_comm, col_num).unwrap();
        assert!(verify_column::<Blake3, _>(
            &column,
            col_num,
            root.as_ref(),
            &[],
            &Ft63::zero(),
        ));
    }
}

#[test]
fn commit() {
    use super::{commit, eval_outer, eval_outer_fft};

    let (coeffs, rho) = random_coeffs_rho();
    let enc = LigeroEncoding::<Ft63>::new(coeffs.len(), rho);
    let comm = commit::<Blake3, LigeroEncoding<_>>(&coeffs, &enc).unwrap();

    let x = Ft63::random(&mut rand::thread_rng());

    let eval = comm
        .coeffs
        .iter()
        .zip(iterate(Ft63::one(), |&v| v * x).take(coeffs.len()))
        .fold(Ft63::zero(), |acc, (c, r)| acc + *c * r);

    let roots_lo: Vec<Ft63> = iterate(Ft63::one(), |&v| v * x)
        .take(comm.n_per_row)
        .collect();
    let roots_hi: Vec<Ft63> = {
        let xr = x * roots_lo.last().unwrap();
        iterate(Ft63::one(), |&v| v * xr)
            .take(comm.n_rows)
            .collect()
    };
    let coeffs_flattened = eval_outer(&comm, &roots_hi[..]).unwrap();
    let eval2 = coeffs_flattened
        .iter()
        .zip(roots_lo.iter())
        .fold(Ft63::zero(), |acc, (c, r)| acc + *c * r);
    assert_eq!(eval, eval2);

    let mut poly_fft = eval_outer_fft(&comm, &roots_hi[..]).unwrap();
    <Ft63 as FieldFFT>::ifft_oi(&mut poly_fft).unwrap();
    assert!(poly_fft
        .iter()
        .skip(comm.n_per_row)
        .all(|&v| v == Ft63::zero()));
    let eval3 = poly_fft
        .iter()
        .zip(roots_lo.iter())
        .fold(Ft63::zero(), |acc, (c, r)| acc + *c * r);
    assert_eq!(eval2, eval3);
}

#[test]
fn end_to_end() {
    use super::{commit, prove, verify};

    // commit to a random polynomial at a random rate
    let (coeffs, rho) = random_coeffs_rho();
    let enc = LigeroEncoding::<Ft63>::new(coeffs.len(), rho);
    let comm = commit::<Blake3, LigeroEncoding<_>>(&coeffs, &enc).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root();

    // evaluate the random polynomial we just generated at a random point x
    let x = Ft63::random(&mut rand::thread_rng());
    let eval = comm
        .coeffs
        .iter()
        .zip(iterate(Ft63::one(), |&v| v * x).take(coeffs.len()))
        .fold(Ft63::zero(), |acc, (c, r)| acc + *c * r);

    // compute the outer and inner tensors for powers of x
    // NOTE: we treat coeffs as a univariate polynomial, but it doesn't
    // really matter --- the only difference from a multilinear is the
    // way we compute outer_tensor and inner_tensor from the eval point
    let inner_tensor: Vec<Ft63> = iterate(Ft63::one(), |&v| v * x)
        .take(comm.n_per_row)
        .collect();
    let outer_tensor: Vec<Ft63> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft63::one(), |&v| v * xr)
            .take(comm.n_rows)
            .collect()
    };

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    let pf: LigeroEvalProof<Blake3, Ft63> =
        prove(&comm, &outer_tensor[..], &enc, &mut tr1).unwrap();
    let encoded: Vec<u8> = bincode::serialize(&pf).unwrap();
    let encroot: Vec<u8> = bincode::serialize(&LcRoot::<Blake3, LigeroEncoding<Ft63>> {
        root: *root.as_ref(),
        _p: Default::default(),
    })
    .unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    let enc2 = LigeroEncoding::<Ft63>::new_from_dims(pf.get_n_per_row(), pf.get_n_cols());
    let res = verify(
        root.as_ref(),
        &outer_tensor[..],
        &inner_tensor[..],
        &pf,
        &enc2,
        &mut tr2,
    )
    .unwrap();

    let root2 =
        bincode::deserialize::<LcRoot<Blake3, LigeroEncoding<Ft63>>>(&encroot[..]).unwrap();
    let pf2: LigeroEvalProof<Blake3, Ft63> = bincode::deserialize(&encoded[..]).unwrap();
    let mut tr3 = Transcript::new(b"test transcript");
    tr3.append_message(b"polycommit", root.as_ref());
    tr3.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr3.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    let enc3 = LigeroEncoding::<Ft63>::new_from_dims(pf2.get_n_per_row(), pf2.get_n_cols());
    let res2 = verify(
        root2.as_ref(),
        &outer_tensor[..],
        &inner_tensor[..],
        &pf2,
        &enc3,
        &mut tr3,
    )
    .unwrap();

    assert_eq!(res, eval);
    assert_eq!(res, res2);
}

#[test]
fn end_to_end_two_proofs() {
    use super::{commit, prove, verify};

    // commit to a random polynomial at a random rate
    let (coeffs, rho) = random_coeffs_rho();
    let enc = LigeroEncoding::<Ft63>::new(coeffs.len(), rho);
    let comm = commit::<Blake3, LigeroEncoding<_>>(&coeffs, &enc).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root();

    // evaluate the random polynomial we just generated at a random point x
    let x = Ft63::random(&mut rand::thread_rng());
    let eval = comm
        .coeffs
        .iter()
        .zip(iterate(Ft63::one(), |&v| v * x).take(coeffs.len()))
        .fold(Ft63::zero(), |acc, (c, r)| acc + *c * r);

    // compute the outer and inner tensors for powers of x
    // NOTE: we treat coeffs as a univariate polynomial, but it doesn't
    // really matter --- the only difference from a multilinear is the
    // way we compute outer_tensor and inner_tensor from the eval point
    let inner_tensor: Vec<Ft63> = iterate(Ft63::one(), |&v| v * x)
        .take(comm.n_per_row)
        .collect();
    let outer_tensor: Vec<Ft63> = {
        let xr = x * inner_tensor.last().unwrap();
        iterate(Ft63::one(), |&v| v * xr)
            .take(comm.n_rows)
            .collect()
    };

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    let pf = prove::<Blake3, _>(&comm, &outer_tensor[..], &enc, &mut tr1).unwrap();

    let challenge_after_first_proof_prover = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr1.challenge_bytes(b"ligero-pc//challenge", &mut key);
        let mut deg_test_rng = ChaCha20Rng::from_seed(key);
        Ft63::random(&mut deg_test_rng)
    };

    // produce a second proof with the same transcript
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr1.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    let pf2 = prove::<Blake3, _>(&comm, &outer_tensor[..], &enc, &mut tr1).unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    let enc2 = LigeroEncoding::<Ft63>::new_from_dims(pf.get_n_per_row(), pf.get_n_cols());
    let res = verify(
        root.as_ref(),
        &outer_tensor[..],
        &inner_tensor[..],
        &pf,
        &enc2,
        &mut tr2,
    )
    .unwrap();

    assert_eq!(res, eval);
    let challenge_after_first_proof_verifier = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr2.challenge_bytes(b"ligero-pc//challenge", &mut key);
        let mut deg_test_rng = ChaCha20Rng::from_seed(key);
        Ft63::random(&mut deg_test_rng)
    };
    assert_eq!(
        challenge_after_first_proof_prover,
        challenge_after_first_proof_verifier
    );

    // second proof verification with the same transcript
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"rate", &rho.to_be_bytes()[..]);
    tr2.append_message(b"ncols", &(N_COL_OPENS as u64).to_be_bytes()[..]);
    let enc3 = LigeroEncoding::<Ft63>::new_from_dims(pf2.get_n_per_row(), pf2.get_n_cols());
    let res2 = verify(
        root.as_ref(),
        &outer_tensor[..],
        &inner_tensor[..],
        &pf2,
        &enc3,
        &mut tr2,
    )
    .unwrap();

    assert_eq!(res2, eval);
}

fn random_coeffs_rho() -> (Vec<Ft63>, f64) {
    let mut rng = rand::thread_rng();

    let lgl = 8 + rng.gen::<usize>() % 8;
    let len_base = 1 << (lgl - 1);
    let len = len_base + (rng.gen::<usize>() % len_base);

    (
        repeat_with(|| Ft63::random(&mut rng)).take(len).collect(),
        rng.gen_range(0.1f64..0.9f64),
    )
}

fn random_comm() -> LigeroCommit<Blake3, Ft63> {
    let mut rng = rand::thread_rng();

    let lgl = 8 + rng.gen::<usize>() % 8;
    let len_base = 1 << (lgl - 1);
    let len = len_base + (rng.gen::<usize>() % len_base);
    let rho = rng.gen_range(0.1f64..0.9f64);
    let (n_rows, n_per_row, n_cols) = LigeroEncoding::<Ft63>::_get_dims(len, rho).unwrap();

    let coeffs_len = (n_per_row - 1) * n_rows + 1 + (rng.gen::<usize>() % n_rows);
    let coeffs = {
        let mut tmp = repeat_with(|| Ft63::random(&mut rng))
            .take(coeffs_len)
            .collect::<Vec<Ft63>>();
        tmp.resize(n_per_row * n_rows, Ft63::zero());
        tmp
    };

    let comm_len = n_rows * n_cols;
    let comm: Vec<Ft63> = repeat_with(|| Ft63::random(&mut rng))
        .take(comm_len)
        .collect();

    LigeroCommit::<Blake3, Ft63> {
        comm,
        coeffs,
        n_rows,
        n_cols,
        n_per_row,
        hashes: vec![<Output<Blake3> as Default>::default(); 2 * n_cols - 1],
    }
}
