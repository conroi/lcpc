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

    # Generate a matrix of size n x m, where each row has d *distinct* random positions
    # filled with random non-zero elements of the field
    @staticmethod
    def generate_random(n, m, d, p):
        matrix = SparseMatrix(n, m)
        for i in range(n):
            indices = random.sample(range(m), d)
            for j in indices:
                matrix.add(i, j, random.randint(1, p-1))
        return matrix




# field size, prime p ~ 2^{256}
p = 90589243044481683682024195529420329427646084220979961316176257384569345097147
n = 2**13

# the following parameters are taken from Table 1 of the write-up.
alpha = 0.178
beta = 0.0785
# r is the rate of the code
r = 1.57

# delta is the distance of the code, not really needed for the encoding procedure, used for testing.
delta = beta / r

# multiply two field elements
def field_mult(a, b):
    return (a % p) * (b % p) % p


# sum up two field elements
def field_add(a, b):
    return (a + b) % p


# the binary entropy function
def H(x):
    assert 0 < x < 1
    return -x*math.log(x,2)-(1-x)*math.log(1-x,2)


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


# The code is given by two lists of sparse matrices, precodes and postcodes.
# Let n0 = n, n1 = alpha n0, n2 = alpha n1,...., nt = alpha n_{t-1},
# where nt is the first value <= 20.
#
# Each list of matrices will have t matrices.
# The i-th matrix in precodes has dimensions ni x (alpha * ni), where alpha is the parameter fixed above.
# The i-th matrix in postcodes has dimensions (alpha * r * ni) x ((r - 1 - alpha * r) * n_i),
# where alpha and r are the parameters fixed above.
#
# Each matrix in *precodes* is just a random sparse matrix sampled as follows:
# In each row of the matrix we pick cn distinct random indices.
# Then each of the sampled positions of the matrix gets a random non-zero element of the field.
# Since all the matrices are sparse, we store them as lists of non-zero elements.
#
# Matrices in *postcode* are sampled in the same way as in precodes with dn non-zeros per row instead of cn.
def generate(n):
    precodes = []
    postcodes = []
    i = 0
    ni = n
    while ni > 20:
        # the current precode matrix has dimensions ni x mi
        mi = math.ceil(ni * alpha)
        # the current postcode matrix has dimensions niprime x miprime
        niprime = math.ceil(r * mi)
        miprime = math.ceil(r * ni) - ni - niprime

        # the sparsity of the precode matrix is cn
        cn = math.ceil(min(
            max(1.2 * beta * ni, beta * ni +3),
            (110/ni + H(beta) + alpha * H(1.2 * beta / alpha)) / (beta * math.log(alpha / (1.2 * beta), 2))
        ))
        precode = SparseMatrix.generate_random(ni, mi, cn, p)
        precodes.append(precode)

        # the sparsity of the postcode matrix is dn
        mu = r - 1 - r * alpha
        nu = beta + alpha * beta + 0.03
        dn = math.ceil(min(
            ni * (2 * beta + (r - 1 + 110/ni)/math.log(p, 2) ),
            (r * alpha * H(beta / r) + mu * H(nu / mu) + 110/ni) / (alpha * beta * math.log(mu / nu, 2))
        ))
        postcode = SparseMatrix.generate_random(niprime, miprime, dn, p)
        postcodes.append(postcode)

        i += 1
        ni = math.ceil(ni * alpha)
    return precodes, postcodes


# The encoding procedure.
# If the length of x is at most 20, then we just encode the vector by Reed-Solomon.
# Otherwise we take the next pair of matrices from precodes and postcodes
# y = x * precode
# z = recursive encoding of y
# v = z * postcode
# the resulting encoding is (x, z, v)
def encode(x, code, shift = 0):
    if len(x) <= 20:
        return reed_solomon(x, math.ceil(r * len(x)))
    precodes, postcodes = code
    assert precodes[shift].n == len(x)
    y = precodes[shift].multiply(x)
    z = encode(y, code, shift+1)
    assert postcodes[shift].n == len(z)
    v = postcodes[shift].multiply(z)
    # here '+' denotes the list concatenation operator.
    return x + z + v


# example
code = generate(n)

for _ in range(10):
    x = []
    # generate a random vector x of length n without using range(p):
    for _ in range(n):
        x.append(random.randint(0, p-1))
    encoded = encode(x, code)
    hammingWeight = sum(1 for element in encoded if element != 0)
    print(hammingWeight/len(encoded) >= delta)
