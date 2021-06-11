// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use ff::Field;
use ft::*;
use ndarray::{linalg::Dot, Array};
use rand::{thread_rng, Rng};
use sprs::{CsMat, TriMat};

mod ft {
    use ff::{Field, PrimeField};

    #[derive(PrimeField)]
    #[PrimeFieldModulus = "70386805592835581672624750593"]
    #[PrimeFieldGenerator = "17"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft([u64; 2]);

    impl num_traits::Num for Ft {
        type FromStrRadixErr = std::num::ParseIntError;

        fn from_str_radix(s: &str, r: u32) -> Result<Self, Self::FromStrRadixErr> {
            use std::ops::{AddAssign, MulAssign};

            if s.is_empty() {
                return Err(u32::from_str_radix(s, r).err().unwrap());
            }

            if s == "0" {
                return Ok(<Self as Field>::zero());
            }

            let mut res = Self::zero();
            let radix = Self::from(u64::from(r));
            let mut first_digit = true;
            for c in s.chars() {
                match c.to_digit(r) {
                    Some(c) => {
                        if first_digit {
                            if c == 0 {
                                return Err(u32::from_str_radix("3", 2).err().unwrap());
                            }
                            first_digit = false;
                        }

                        res.mul_assign(&radix);
                        res.add_assign(Self::from(u64::from(c)));
                    }
                    None => {
                        return Err(u32::from_str_radix("3",2).err().unwrap());
                    }
                }
            }
            Ok(res)
        }
    }

    impl num_traits::Zero for Ft {
        fn zero() -> Self {
            <Self as Field>::zero()
        }

        fn is_zero(&self) -> bool {
            <Self as Field>::is_zero(self)
        }
    }

    impl num_traits::One for Ft {
        fn one() -> Self {
            <Self as Field>::one()
        }

        fn is_one(&self) -> bool {
            self == &<Self as Field>::one()
        }
    }

    impl std::ops::Div for Ft {
        type Output = Ft;

        #[must_use]
        fn div(self, rhs: Self) -> Ft {
            self * rhs.invert().unwrap()
        }
    }

    impl std::ops::Rem for Ft {
        type Output = Ft;

        #[must_use]
        fn rem(self, rhs: Self) -> Ft {
            if rhs.is_zero() {
                panic!("divide by zero");
            }

            Self::zero()
        }
    }

    impl num_traits::ops::mul_add::MulAdd for Ft {
        type Output = Ft;

        fn mul_add(self, a: Self, b: Self) -> Ft {
            let mut res = self;
            res *= &a;
            res += &b;
            res
        }
    }
}

#[test]
fn sprs_playground() {
    let mut rng = thread_rng();
    let n_rows = 65537;
    let n_cols = 32749;

    let m: CsMat<_> = {
        let mut tmp = TriMat::new((n_rows, n_cols));
        for i in 0..n_rows {
            let col1 = rng.gen_range(0..n_cols);
            let col2 = {
                let mut tmp = rng.gen_range(0..n_cols);
                while tmp == col1 {
                    tmp = rng.gen_range(0..n_cols);
                }
                tmp
            };
            tmp.add_triplet(i, col1, Ft::random(&mut rng));
            tmp.add_triplet(i, col2, Ft::random(&mut rng));
        }
        // csr appears to be considerably faster than csc
        tmp.to_csr().transpose_into()
    };

    let v = {
        let mut tmp = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            tmp.push(Ft::random(&mut rng));
        }
        Array::from(tmp)
    };

    let mut t = Ft::zero();
    for i in 0..10 {
        let mv = m.dot(&v);
        t += mv[i % n_cols];
    }
    println!("{:?}", t);
}
