// Criterion Benchmarks for encode, commit, prove, verify.
// Use parameters that achieve distance 1/20 and rate 3/5.
use blake3::Hasher as Blake3;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::Field;
use itertools::iterate;
use lcpc_2d::LcEncoding;
use lcpc_test_fields::{ft127::*, random_coeffs};
use merlin::Transcript;
use rand::thread_rng;
use sha2::Sha256;
use std::hint::black_box;

use lcpc_brakedown_pc::{BrakedownCommit, SdigEncoding};

fn encode_benchmark(c: &mut Criterion) {
    use lcpc_brakedown_pc::codespec::SdigCode4 as TestCode;
    let mut rng = thread_rng();
    use lcpc_brakedown_pc::encode::{codeword_length, encode};
    use lcpc_brakedown_pc::matgen::generate;

    let mut group = c.benchmark_group("encode");

    for lgl in [10, 12, 14, 16, 18, 20].iter() {
        let (precodes, postcodes) = generate::<Ft127, TestCode>(1 << lgl, 0u64);
        let xi_len = codeword_length(&precodes, &postcodes);
        println!("lgl {} xi_len {}", lgl, xi_len);
        let mut xi = Vec::with_capacity(xi_len);
        for _ in 0..xi_len {
            xi.push(Ft127::random(&mut rng));
        }

        group.bench_with_input(BenchmarkId::from_parameter(lgl), lgl, |b, &_size| {
            b.iter(|| encode(&mut xi, &precodes, &postcodes));
        });
    }
    group.finish();
}

fn commit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_127");

    for lgl in [10usize, 12, 14, 16, 18].iter() {
        let coeffs = random_coeffs(*lgl);
        let enc = SdigEncoding::new(coeffs.len(), 0);

        group.bench_with_input(BenchmarkId::new("blake_127", *lgl), lgl, |b, _lgl| {
            b.iter(|| black_box(BrakedownCommit::<Blake3, Ft127>::commit(&coeffs, &enc).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("sha256_127", *lgl), lgl, |b, _lgl| {
            b.iter(|| black_box(BrakedownCommit::<Sha256, Ft127>::commit(&coeffs, &enc).unwrap()));
        });
    }

    group.finish();
}

fn matgen_benchmark(c: &mut Criterion) {
    use lcpc_brakedown_pc::codespec::SdigCode4 as TestCode;
    use lcpc_brakedown_pc::matgen::generate;
    let mut group = c.benchmark_group("matgen");

    for lgl in [16, 18, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(lgl), lgl, |b, &_size| {
            b.iter(|| generate::<Ft127, TestCode>(1 << lgl, 0u64));
        });
    }
}

fn prove_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove");

    for lgl in [14usize, 16, 18].iter() {
        let coeffs = random_coeffs(*lgl);
        let enc = SdigEncoding::new(coeffs.len(), 0);
        let comm = BrakedownCommit::<Blake3, Ft127>::commit(&coeffs, &enc).unwrap();

        // random point to eval at
        let x = Ft127::random(&mut rand::thread_rng());
        let inner_tensor: Vec<Ft127> = iterate(<Ft127 as Field>::one(), |&v| v * x)
            .take(comm.get_n_per_row())
            .collect();
        let outer_tensor: Vec<Ft127> = {
            let xr = x * inner_tensor.last().unwrap();
            iterate(<Ft127 as Field>::one(), |&v| v * xr)
                .take(comm.get_n_rows())
                .collect()
        };

        group.bench_with_input(BenchmarkId::new("Blake3_127", *lgl), lgl, |b, _lgl| {
            b.iter(|| {
                let mut tr = Transcript::new(b"bench transcript");
                tr.append_message(b"polycommit", comm.get_root().as_ref());
                tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
                tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
                tr.append_message(
                    b"ndegs",
                    &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
                );
                black_box(comm.prove(&outer_tensor[..], &enc, &mut tr).unwrap());
            });
        });
    }
}

fn verify_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");

    for lgl in [14usize, 16, 18].iter() {
        let coeffs = random_coeffs(*lgl);
        let enc = SdigEncoding::new(coeffs.len(), 0);
        let comm = BrakedownCommit::<Blake3, Ft127>::commit(&coeffs, &enc).unwrap();

        // random point to eval at
        let x = Ft127::random(&mut rand::thread_rng());
        let inner_tensor: Vec<Ft127> = iterate(<Ft127 as Field>::one(), |&v| v * x)
            .take(comm.get_n_per_row())
            .collect();
        let outer_tensor: Vec<Ft127> = {
            let xr = x * inner_tensor.last().unwrap();
            iterate(<Ft127 as Field>::one(), |&v| v * xr)
                .take(comm.get_n_rows())
                .collect()
        };

        let mut tr = Transcript::new(b"bench transcript");
        tr.append_message(b"polycommit", comm.get_root().as_ref());
        tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
        tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
        tr.append_message(
            b"ndegs",
            &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
        );
        let pf = comm.prove(&outer_tensor[..], &enc, &mut tr).unwrap();
        let root = comm.get_root();

        group.bench_with_input(BenchmarkId::new("Blake3_127", *lgl), lgl, |b, _lgl| {
            b.iter(|| {
                let mut tr = Transcript::new(b"bench transcript");
                tr.append_message(b"polycommit", comm.get_root().as_ref());
                tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
                tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
                tr.append_message(
                    b"ndegs",
                    &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
                );
                black_box(
                    pf.verify(
                        root.as_ref(),
                        &outer_tensor[..],
                        &inner_tensor[..],
                        &enc,
                        &mut tr,
                    )
                    .unwrap(),
                );
            });
        });
    }
}

criterion_group!(
    benches,
    commit_benchmark,
    encode_benchmark,
    matgen_benchmark,
    prove_benchmark,
    verify_benchmark
);

criterion_main!(benches);
