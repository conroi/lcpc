// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

/// Define domain separation labels for an LcEncoding trait implementation
///
/// Use this to conveniently define the LABEL_xx values for domain separation,
/// e.g.:
///
/// ```ignore
/// impl LcEncoding for ... {
///     ...
///
///     def_labels!(my_encoding_name);
///
///     ...
/// }
///
/// ```
///
/// Note that the argument may only contain alphanumerics and underscores,
/// and cannot be just an underscore (same rules as Rust identifiers).
#[macro_export]
macro_rules! def_labels {
    ($l:ident) => {
        const LABEL_DT: &'static [u8] = b"$l//DT";
        const LABEL_PR: &'static [u8] = b"$l//PR";
        const LABEL_PE: &'static [u8] = b"$l//PE";
        const LABEL_CO: &'static [u8] = b"$l//CO";
    };
}
