# encoding.py
# (C) 2021 Sasha Golovnev
# linear-time encoding with constant relative distance

import math
import random

# Auxiliary class to store sparse matrices as lists of non-zero elements
# (I couldn't figure out a simple way to use scipy's sparse matrices for our goals.)
class SparseMatrix:

    # A matrix of size n x m,
    # a[i] is a list of pairs (j, val) corresponding to matrix with A[i, j] = val
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.a = []
        for i in range(n):
            self.a.append([])

    # Add a non-zero element to the matrix
    def add(self, i, j, val):
        if val == 0:
            return
        assert 0 <= i < self.n
        assert 0<= j < self.m
        self.a[i].append((j, val))


    # Multiply a vector x of length self.n by this matrix
    def multiply(self, x):
        assert len(x) == self.n
        y = [0] * self.m
        for i in range(self.n):
            for (j, val) in self.a[i]:
                y[j] += x[i] * val
        return y

    # Generates a matrix of size n x m, where each row has d random positions *with* replacement
    # filled with random non-zero elements of the field
    @staticmethod
    def generate_random(n, m, d, p):
        matrix = SparseMatrix(n, m)
        for i in range(n):
            indices = set(random.choices(range(m), k=d))
            for j in indices:
                matrix.add(i, j, random.randint(1, p-1))
        return matrix




# field size, prime p ~ 2^{256}
p = 90589243044481683682024195529420329427646084220979961316176257384569345097147
n = 2**13

# the following parameters are taken from Table 1 of the write-up (and will be updated later)
T = 128
alpha = 0.32
beta = 0.16
gamma = 0.45
k = 1.1
d1 = 7
d2 = 10


# multiply two field elements
def field_mult(a, b):
    return (a % p) * (b % p) % p


# sum up two field elements
def field_add(a, b):
    return (a + b) % p


# I don't implement an efficient encoding by Reed-Solomon, because I assume we already have it.
# Instead I just generate a Vandermonde matrix and multiply it by the input vector x.
# This procedure takes a vector x of length l = len(x), and outputs a vector of length m.
def reed_solomon(x, m):
    l = len(x)
    # Reed-Solomon requires the field size to be at least the length of the output
    assert p > m
    y = [0] * m
    for i in range(1, m+1):
        a = 1
        for j in range(l):
            y[i-1] = field_add(y[i-1], field_mult(a, x[j]))
            a = field_mult(a, i)
    return y


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
def generate(n):
    precodes = []
    postcodes = []
    i = 0
    ni = n
    while ni > T:
        # the current precode matrix has dimensions ni x mi
        mi = math.ceil(ni * alpha)
        # the current postcode matrix has dimensions niprime x miprime
        niprime = ni + math.ceil(k * mi)
        miprime = math.ceil(k * ni)

        precode = SparseMatrix.generate_random(ni, mi, d1, p)
        while not check(precode):
            precode = SparseMatrix.generate_random(ni, mi, d1, p)

        precodes.append(precode)

        postcode = SparseMatrix.generate_random(niprime, miprime, d2, p)
        postcodes.append(postcode)
        i += 1
        ni = math.ceil(ni * alpha)
    return precodes, postcodes


# The recursive part of the encoding.
# If the length of x is less than T, then we just encode the vector by Reed-Solomon.
# Otherwise we take the next pair of matrices from precodes and postcodes
# y = x * precode
# z = recursive encoding of y
# the resulting encoding is (x, z) * postcode
def encode_recursive(x, code, shift = 0):
    if len(x) <= T:
        return reed_solomon(x, math.ceil(k*len(x)))
    precodes, postcodes = code
    assert precodes[shift].n == len(x)
    y = precodes[shift].multiply(x)
    z = encode_recursive(y, code, shift+1)
    assert postcodes[shift].n == len(x) + len(z)
    # here '+' is the list concatenation operator.
    return postcodes[shift].multiply(x+z)


# x is an input vector, code is a code generated by the generate function above.
def encode(x, code):
    non_systematic_part = encode_recursive(x, code)
    # here '+' is the list concatenation operator. That is, our code is systematic: it output the input vector x
    # followed by the vector encode_recursive(...)
    return x + non_systematic_part


# example
code = generate(n)
# generate a random vector x of length n without using range(p):
x = []
for _ in range(n):
    x.append(random.randint(0, p-1))

encoded = encode(x, code)
