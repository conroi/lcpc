// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn get_dims() {
    use super::get_dims;
    use rand::Rng;

    let mut rng = rand::thread_rng();

    for _ in 0..16 {
        let lgl = 8 + rng.gen::<usize>() % 8;
        for _ in 0..16 {
            let len_base = 1 << (lgl - 1);
            let len = len_base + (rng.gen::<usize>() % len_base);
            let rho = rng.gen_range(0.001f64, 1f64);
            let (nrows, n_per_row, ncols) = get_dims(len, rho).unwrap();
            assert!(nrows * n_per_row >= len);
            assert!((nrows - 1) * n_per_row < len);
            assert!(n_per_row as f64 / rho <= ncols as f64);
        }
    }
}
