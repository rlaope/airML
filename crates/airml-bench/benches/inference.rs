//! Criterion benchmark for airML inference.
//!
//! Set `AIRML_BENCH_MODEL` to the path of an ONNX model before running:
//!
//! ```sh
//! AIRML_BENCH_MODEL=models/mobilenetv3.onnx cargo bench -p airml-bench
//! ```

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use criterion::measurement::WallTime;

use airml_bench::{BenchHarness, BenchProvider};

/// Read `AIRML_BENCH_MODEL` from the environment.
///
/// Returns `None` (and prints a notice) when the variable is unset or the path
/// does not exist, so the benchmark binary exits cleanly rather than panicking.
fn model_path() -> Option<PathBuf> {
    match std::env::var("AIRML_BENCH_MODEL") {
        Ok(s) => {
            let p = PathBuf::from(&s);
            if p.exists() {
                Some(p)
            } else {
                eprintln!(
                    "[airml-bench] AIRML_BENCH_MODEL={s:?} does not exist — skipping benchmarks."
                );
                None
            }
        }
        Err(_) => {
            eprintln!(
                "[airml-bench] AIRML_BENCH_MODEL is not set — skipping benchmarks.\n\
                 Set it to the path of an ONNX model, e.g.:\n\
                 \n  AIRML_BENCH_MODEL=models/mobilenetv3.onnx cargo bench -p airml-bench\n"
            );
            None
        }
    }
}

/// Benchmark a single provider inside a [`BenchmarkGroup`].
fn bench_provider(
    group: &mut BenchmarkGroup<WallTime>,
    model: &std::path::Path,
    provider: BenchProvider,
    label: &str,
) {
    let mut harness = match BenchHarness::load(model, provider) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[airml-bench] Could not load model with provider {label}: {e}");
            return;
        }
    };

    // Warm up outside the measured region.
    for _ in 0..3 {
        if let Err(e) = harness.run_one() {
            eprintln!("[airml-bench] Warmup error ({label}): {e}");
            return;
        }
    }

    group.throughput(Throughput::Elements(1));
    group.bench_function(BenchmarkId::new("inference", label), |b| {
        b.iter(|| {
            harness.run_one().expect("inference failed during benchmark")
        });
    });
}

fn benchmark_inference(c: &mut Criterion) {
    let Some(model) = model_path() else {
        return;
    };

    let mut group = c.benchmark_group("airml_inference");

    // CPU is always benchmarked.
    bench_provider(&mut group, &model, BenchProvider::Cpu, "cpu");

    // CoreML is only available on macOS.
    #[cfg(feature = "coreml")]
    bench_provider(&mut group, &model, BenchProvider::CoreML, "coreml");

    group.finish();
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
