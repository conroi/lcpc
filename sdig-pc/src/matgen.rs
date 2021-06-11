// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/*
   From Sasha's python impl:

# The code is given by two lists of sparse matrices, precodes and postcodes.
# Let n0 = n, n1 = alpha n0, n2 = alpha n1,...., nt = alpha n_{t-1},
# where nt is the first value <= T.
#
# Each list of matrices will have t+1 matrices.
# The i-th matrix in precodes has dimensions ni x (alpha * ni), where alpha is fixed above.
# The i-th matrix in postcodes has dimensions (ni + k * alpha * ni) x (k * ni), where alpha and k are parameters fixed above.
#
# Each matrix in *postcodes* is just a random matrix sampled as follows:
# In each row of the matrix we first sample d2 random indices with replacement, and ignore repetitions.
# Then each of the sampled positions of the matrix gets a random non-zero element of the field.
# Since all the matrices are sparse, we store them as lists of non-zero elements.
#
# Matrices in *precodes* are generated in the same way as postcodes (with d1 non-zeros per row instead of d2),
# but if a precode doesn't pass check(), we resample it.

   Here is check() referenced above:

# This is the time-consuming part of the generation procedure. Most likely it will always return True, but we have to
# have it to guarantee a low probability (2^{-128}) of generating a bad code.
# Given an nxm matrix, we check that every row has at least two non-zero entries,
# and that every pair of rows has at least three distinct non-zeros.
def check(matrix):
    for i1 in range(matrix.n):
        if len(matrix.a[i1]) < 2:
            return False
        for i2 in range(i1 + 1, matrix.n):
            indices = set()
            for (j, val) in matrix.a[i1]+matrix.a[i2]:
                indices.add(j)
            if len(indices) < 3:
                return False
    return True
*/

/*
// minimum dimension, at which point we just switch to R-S
const MIN_DIM: usize = 128;
// alpha = 0.32
const ALPHA_NUM: usize = 8;
const ALPHA_DEN: usize = 25;
// beta = 0.16
const BETA_NUM: usize = 4;
const BETA_DEN: usize = 25;
// gamma = 0.45
const GAMMA_NUM: usize = 9;
const GAMMA_DEN: usize = 20;
// k = 1.1
const K_NUM: usize = 11;
const K_DEN: usize = 10;
// row density of precodes
const D1: usize = 7;
// row density of postcodes
const D2: usize = 10;
*/
