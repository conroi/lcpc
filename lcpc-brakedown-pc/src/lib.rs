// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-brakedown-pc, which is part of lcpc.
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

use codespec::{SdigCode3 as SdigCodeDflt, SdigSpecification};
use encode::{codeword_length, encode};
use matgen::generate;

use ff::Field;
use lcpc_2d::{
    def_labels, n_degree_tests, FieldHash, LcCommit, LcEncoding, LcEvalProof, SizedField,
};
use num_traits::Num;
use sprs::{CsMat, MulAcc};

pub mod codespec;
pub mod encode;
pub mod matgen;

#[cfg(all(test, feature = "bench"))]
mod bench;
#[cfg(any(test, feature = "bench"))]
mod tests;

/// Encoding definition for SDIG expander-based polycommit
#[derive(Clone, Debug)]
pub struct SdigEncodingS<Ft, S> {
    n_per_row: usize, // number of inputs to the encoding
    n_cols: usize,    // number of outputs from the encoding
    precodes: Vec<CsMat<Ft>>,
    postcodes: Vec<CsMat<Ft>>,
    _p: std::marker::PhantomData<S>,
}

impl<Ft, S> SdigEncodingS<Ft, S>
where
    Ft: Field + Num + SizedField,
    S: SdigSpecification,
{
    const LAMBDA: usize = 128;

    // number of column openings required for soundness
    fn _n_col_opens() -> usize {
        let dist_ov_3 = S::dist() / 3f64;
        let den = (1f64 - dist_ov_3).log2();
        (-(Self::LAMBDA as f64) / den).ceil() as usize
    }

    // number of degree tests required for soundness
    fn _n_degree_tests(n_cols: usize) -> usize {
        n_degree_tests(Self::LAMBDA, n_cols, Ft::FLOG2 as usize)
    }

    // shared between new and new_ml
    fn _new_from_np1(len: usize, np1: usize, seed: u64) -> Self {
        // n_per_row can't be greater than length!
        let np1 = if np1 > len { len } else { np1 };

        let n_col_opens = Self::_n_col_opens();
        let nr1 = (len + np1 - 1) / np1;
        let nd1 = Self::_n_degree_tests(np1 * 2); // approximately
        assert!(np1 * nr1 >= len);
        assert!(np1 * (nr1 - 1) < len);

        let np2 = np1 / 2;
        let nr2 = (len + np2 - 1) / np2;
        let nd2 = Self::_n_degree_tests(np2 * 2); // approximately
        assert!(np2 * nr2 >= len);
        assert!(np2 * (nr2 - 1) < len);

        let sz1 = n_col_opens * nr1 + (1 + nd1) * np1;
        let sz2 = n_col_opens * nr2 + (1 + nd2) * np2;
        let n_per_row = if sz1 < sz2 { np1 } else { np2 };

        let (precodes, postcodes) = generate::<Ft, S>(n_per_row, seed);
        assert_eq!(n_per_row, precodes[0].cols());
        let n_cols = codeword_length(&precodes, &postcodes);
        Self {
            n_per_row,
            n_cols,
            precodes,
            postcodes,
            _p: std::marker::PhantomData::default(),
        }
    }

    /// Create a new SdigEncoding for a univariate polynomial with `len` coefficients
    /// using a random expander code generated with seed `seed`.
    pub fn new(len: usize, seed: u64) -> Self {
        // compute #cols, optimizing the communication cost
        let lncf = (Self::_n_col_opens() * len) as f64;
        // approximation of num_degree_tests
        let ndt = Self::_n_degree_tests(lncf.sqrt().ceil() as usize * 2) as f64;
        let np1 = (lncf / ndt).sqrt().ceil() as usize;
        Self::_new_from_np1(len, np1, seed)
    }

    /// Create a new SdigEncoding for a multilinear polynomial with `n_vars` variables
    /// (i.e., 2^`n_vars` monomials) using a random expander code generated with seed `seed`.
    pub fn new_ml(n_vars: usize, seed: u64) -> Self {
        let n_monomials = 1 << n_vars;
        let lncf = (Self::_n_col_opens() * n_monomials) as f64;
        // approximation of num_degree_tests
        let ndt = Self::_n_degree_tests(lncf.sqrt().ceil() as usize * 2) as f64;
        let np1 = ((lncf / ndt).sqrt().ceil() as usize)
            .checked_next_power_of_two()
            .unwrap();
        Self::_new_from_np1(n_monomials, np1, seed)
    }

    /// Create a new SdigEncoding for a commitment with dimensions `n_per_row` and `n_cols`
    pub fn new_from_dims(n_per_row: usize, n_cols: usize, seed: u64) -> Self {
        let (precodes, postcodes) = generate::<Ft, S>(n_per_row, seed);
        assert_eq!(n_per_row, precodes[0].cols());
        assert_eq!(n_cols, codeword_length(&precodes, &postcodes));
        Self {
            n_per_row,
            n_cols,
            precodes,
            postcodes,
            _p: std::marker::PhantomData::default(),
        }
    }
}

impl<Ft, S> LcEncoding for SdigEncodingS<Ft, S>
where
    Ft: Field + FieldHash + MulAcc + Num + SizedField,
    S: SdigSpecification + std::fmt::Debug + std::clone::Clone + std::marker::Sync,
{
    type F = Ft;
    type Err = std::io::Error;

    def_labels!(sdig_pc);

    fn encode<T: AsMut<[Ft]>>(&self, inp: T) -> Result<(), Self::Err> {
        encode(inp, &self.precodes, &self.postcodes);
        Ok(())
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
        let nc2 = n_cols == codeword_length(&self.precodes, &self.postcodes);
        ok && np1 && np2 && nc1 && nc2
    }

    fn get_n_col_opens(&self) -> usize {
        Self::_n_col_opens()
    }

    fn get_n_degree_tests(&self) -> usize {
        Self::_n_degree_tests(self.n_cols)
    }
}

/// default encoding
pub type SdigEncoding<F> = SdigEncodingS<F, SdigCodeDflt>;

/// SDIG expander-based polynomial commitment
pub type BrakedownCommit<D, F> = LcCommit<D, SdigEncoding<F>>;

/// An evaluation proof for SDIG expander-based polynomial commitment
pub type BrakedownEvalProof<D, F> = LcEvalProof<D, SdigEncoding<F>>;
