// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-brakedown-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/*! specify code parameters */

use std::fmt::Debug;
use std::marker::{PhantomData, Sync};
use std::ops::Add;
use typenum::Unsigned;

fn ent(z: f64) -> f64 {
    assert!(0f64 < z && z < 1f64);
    let mzp1 = 1f64 - z;
    -z * z.log2() - mzp1 * mzp1.log2()
}

/// Specify an SDIG code
pub trait SdigSpecification {
    /// numerator of alpha
    type An: Unsigned;
    /// denominator of alpha
    type Ad: Unsigned;

    /// numerator of beta
    type Bn: Unsigned;
    /// denominator of beta
    type Bd: Unsigned;

    /// numerator of R
    type Rn: Unsigned;
    /// denominator of R
    type Rd: Unsigned;

    /// base-case code length
    type Blen: Unsigned;

    /// distance as f64 {
    fn dist() -> f64 {
        (Self::Bn::to_usize() * Self::Rd::to_usize()) as f64
            / (Self::Bd::to_usize() * Self::Rn::to_usize()) as f64
    }

    /// alpha num as usize
    fn alpha_num() -> usize {
        Self::An::to_usize()
    }

    /// alpha den as usize
    fn alpha_den() -> usize {
        Self::Ad::to_usize()
    }

    /// beta num as usize
    fn beta_num() -> usize {
        Self::Bn::to_usize()
    }

    /// beta den as usize
    fn beta_den() -> usize {
        Self::Bd::to_usize()
    }

    /// r num as usize
    fn r_num() -> usize {
        Self::Rn::to_usize()
    }

    /// r den as usize
    fn r_den() -> usize {
        Self::Rd::to_usize()
    }

    /// baselen
    fn baselen() -> usize {
        Self::Blen::to_usize()
    }

    /// alpha as f64
    fn alpha() -> f64 {
        Self::An::to_usize() as f64 / Self::Ad::to_usize() as f64
    }

    /// beta as f64
    fn beta() -> f64 {
        Self::Bn::to_usize() as f64 / Self::Bd::to_usize() as f64
    }

    /// r as f64
    fn r() -> f64 {
        Self::Rn::to_usize() as f64 / Self::Rd::to_usize() as f64
    }

    /// mu = r - 1 - r * alpha
    fn mu() -> f64 {
        Self::r() - 1f64 - Self::r() * Self::alpha()
    }

    /// nu = beta + alpha * beta + 0.03
    fn nu() -> f64 {
        Self::beta() + Self::alpha() * Self::beta() + 0.03f64
    }

    /// constant for cn calculation
    fn cnst_cn_1() -> f64 {
        ent(Self::beta()) + Self::alpha() * ent(1.28f64 * Self::beta() / Self::alpha())
    }

    /// constant for cn calculation
    fn cnst_cn_2() -> f64 {
        Self::beta() * (Self::alpha() / (1.28f64 * Self::beta())).log2()
    }

    /// constant for dn calculation
    fn cnst_dn_1() -> f64 {
        Self::r() * Self::alpha() * ent(Self::beta() / Self::r())
            + Self::mu() * ent(Self::nu() / Self::mu())
    }

    /// constant for dn calculation
    fn cnst_dn_2() -> f64 {
        Self::alpha() * Self::beta() * (Self::mu() / Self::nu()).log2()
    }
}

/// A concrete SdigSpecification object
#[derive(Clone, Debug)]
pub struct SdigSpec<An, Ad, Bn, Bd, Rn, Rd, Blen>
where
    An: Unsigned + Debug + Sync,
    Ad: Unsigned + Debug + Sync,
    Bn: Unsigned + Debug + Sync,
    Bd: Unsigned + Debug + Sync,
    Rn: Unsigned + Debug + Sync,
    Rd: Unsigned + Debug + Sync,
    Blen: Unsigned + Debug + Sync,
{
    _p: PhantomData<(An, Ad, Bn, Bd, Rn, Rd, Blen)>,
}

impl<An, Ad, Bn, Bd, Rn, Rd, Blen> SdigSpecification for SdigSpec<An, Ad, Bn, Bd, Rn, Rd, Blen>
where
    An: Unsigned + Debug + Sync,
    Ad: Unsigned + Debug + Sync,
    Bn: Unsigned + Debug + Sync,
    Bd: Unsigned + Debug + Sync,
    Rn: Unsigned + Debug + Sync,
    Rd: Unsigned + Debug + Sync,
    Blen: Unsigned + Debug + Sync,
{
    type An = An;
    type Ad = Ad;
    type Bn = Bn;
    type Bd = Bd;
    type Rn = Rn;
    type Rd = Rd;
    type Blen = Blen;
}

type U1521 = <typenum::U1021 as Add<typenum::U500>>::Output;
type U2000 = <typenum::U1000 as Add<typenum::U1000>>::Output;
type U2500 = <typenum::U500 as Add<U2000>>::Output;
/// line 1 from table
pub type SdigCode1 = SdigSpec<
    typenum::U239, // alpha = 0.1195
    U2000,
    typenum::U71, // beta = 0.0284
    U2500,
    typenum::U71, // r = 1.42
    typenum::U50,
    typenum::U20, // baselen = 20
>;

/// line 2 from table
pub type SdigCode2 = SdigSpec<
    typenum::U69, // alpha = 0.138
    typenum::U500,
    typenum::U111, // beta = 0.0444
    U2500,
    typenum::U147, // r = 1.47
    typenum::U100,
    typenum::U20, // baselen = 20
>;

/// line 3 from table
pub type SdigCode3 = SdigSpec<
    typenum::U89, // alpha = 0.178
    typenum::U500,
    typenum::U61, // beta = 0.061
    typenum::U1000,
    U1521, // r = 1.521
    typenum::U1000,
    typenum::U20, // baselen = 20
>;

/// line 4 from table
pub type SdigCode4 = SdigSpec<
    typenum::U1, // alpha = 0.2
    typenum::U5,
    typenum::U41, // beta = 0.082
    typenum::U500,
    typenum::U41, // r = 1.64
    typenum::U25,
    typenum::U20, // baselen = 20
>;

/// line 5 from table
pub type SdigCode5 = SdigSpec<
    typenum::U211, // alpha = 0.211
    typenum::U1000,
    typenum::U97, // beta = 0.097
    typenum::U1000,
    typenum::U202, // r = 1.616
    typenum::U125,
    typenum::U20, // baselen = 20
>;

/// line 6 from table
pub type SdigCode6 = SdigSpec<
    typenum::U119, // alpha = 0.238
    typenum::U500,
    typenum::U241, // beta = 0.1205
    U2000,
    typenum::U43, // r = 1.72
    typenum::U25,
    typenum::U20, // baselen = 20
>;
