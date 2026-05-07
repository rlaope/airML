//! airML Benchmarking Harness
//!
//! Provides a [`BenchHarness`] for loading models, generating realistic
//! synthetic inputs, measuring cold-start and steady-state latency, and
//! computing percentile statistics.
//!
//! JSON serialisation of results is in the [`report`] module; re-exported
//! at crate root as [`report_to_json`] and [`save_report_to`].

pub mod report;

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};

use airml_core::{InferenceEngine, SessionConfig};
use airml_providers::{CpuProvider, ExecutionProviderDispatch};

// ── Execution provider selector ──────────────────────────────────────────────

/// Which execution provider to use when loading a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchProvider {
    /// Automatically select the best available provider (CoreML > CPU).
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Force CoreML (macOS only; compile with `--features airml-providers/coreml`).
    #[cfg(feature = "coreml")]
    CoreML,
    /// Force the Neural Engine via CoreML ANE hint.
    #[cfg(feature = "coreml")]
    NeuralEngine,
}

fn providers_for(bench_provider: BenchProvider) -> Vec<ExecutionProviderDispatch> {
    match bench_provider {
        BenchProvider::Auto => airml_providers::auto_select_providers(),
        BenchProvider::Cpu => vec![CpuProvider::default().into_dispatch()],
        #[cfg(feature = "coreml")]
        BenchProvider::CoreML => {
            use airml_providers::CoreMLProvider;
            vec![CoreMLProvider::default().into_dispatch()]
        }
        #[cfg(feature = "coreml")]
        BenchProvider::NeuralEngine => {
            use airml_providers::{ComputeUnits, CoreMLConfig, CoreMLProvider};
            let config = CoreMLConfig {
                compute_units: ComputeUnits::CpuAndNeuralEngine,
                ..CoreMLConfig::default()
            };
            vec![CoreMLProvider::with_config(config).into_dispatch()]
        }
    }
}

// ── Input distribution ────────────────────────────────────────────────────────

/// Describes the statistical distribution used to generate realistic inputs.
#[derive(Debug, Clone, Copy)]
pub enum InputDistribution {
    /// Gaussian (normal) distribution.
    Gaussian {
        /// Mean of the distribution.
        mean: f32,
        /// Standard deviation.
        stddev: f32,
    },
    /// Uniform distribution over `[lo, hi)`.
    Uniform {
        /// Lower bound (inclusive).
        lo: f32,
        /// Upper bound (exclusive).
        hi: f32,
    },
    /// ImageNet-style normalised image inputs.
    ///
    /// Each channel is sampled from a Gaussian whose mean and stddev follow the
    /// standard ImageNet per-channel statistics (R/G/B order, NCHW layout
    /// assumed). For other layouts the values are still statistically valid
    /// because they are drawn from the same per-channel distributions.
    ImagenetNormalized,
    /// Integer token IDs cast to `f32`.
    ///
    /// Samples are drawn uniformly from `[0, vocab_size)`.
    TokenIds {
        /// Vocabulary size (exclusive upper bound).
        vocab_size: u32,
    },
}

/// Minimal Xorshift64 PRNG — no-alloc, no dependencies.
struct Rng {
    state: u64,
    /// Spare value from the previous Box–Muller pair.
    spare: Option<f32>,
}

impl Rng {
    fn new() -> Self {
        // Seed from subsecond clock bits XOR a compile-time constant.
        let seed = (Instant::now().elapsed().subsec_nanos() as u64) ^ 0xDEAD_BEEF_CAFE_1234;
        Self {
            state: seed | 1, // must be non-zero
            spare: None,
        }
    }

    /// Return the next pseudo-random `u32`.
    fn next_u32(&mut self) -> u32 {
        // Xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 32) as u32
    }

    /// Return the next pseudo-random `f32` in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }

    /// Return a standard-normal sample via Box–Muller transform.
    fn next_normal(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        loop {
            let u = self.next_f32() * 2.0 - 1.0;
            let v = self.next_f32() * 2.0 - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let mul = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * mul);
                return u * mul;
            }
        }
    }
}

/// Generate a realistic synthetic input tensor.
///
/// Returns an [`ArrayD<f32>`] of the requested shape filled according to
/// `distribution`. All sampling is done with a fast Xorshift PRNG seeded
/// from the system clock — no external crate is required.
pub fn realistic_input(shape: &[usize], distribution: InputDistribution) -> ArrayD<f32> {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    let mut rng = Rng::new();

    match distribution {
        InputDistribution::Gaussian { mean, stddev } => {
            for _ in 0..n {
                data.push(mean + stddev * rng.next_normal());
            }
        }
        InputDistribution::Uniform { lo, hi } => {
            for _ in 0..n {
                data.push(lo + rng.next_f32() * (hi - lo));
            }
        }
        InputDistribution::ImagenetNormalized => {
            // ImageNet per-channel (mean, std): R=(0.485,0.229), G=(0.456,0.224), B=(0.406,0.225)
            // We cycle through channels regardless of actual layout.
            const MEANS: [f32; 3] = [0.485, 0.456, 0.406];
            const STDS: [f32; 3] = [0.229, 0.224, 0.225];
            for i in 0..n {
                let ch = i % 3;
                data.push(MEANS[ch] + STDS[ch] * rng.next_normal());
            }
        }
        InputDistribution::TokenIds { vocab_size } => {
            for _ in 0..n {
                let id = rng.next_u32() % vocab_size;
                data.push(id as f32);
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(shape), data)
        .expect("shape and data length are consistent by construction")
}

// ── BenchHarness ─────────────────────────────────────────────────────────────

/// A loaded model paired with a pre-generated sample input.
///
/// Construct via [`BenchHarness::load`], then call [`BenchHarness::run_one`]
/// to measure a single inference pass. The sample input is cloned inside the
/// harness before each run so allocations do not pollute timing.
pub struct BenchHarness {
    engine: InferenceEngine,
    /// Pre-allocated sample input (cloned once per call to `run_one`).
    sample_input: ArrayD<f32>,
}

impl BenchHarness {
    /// Load a model from `path` using the given provider selection strategy.
    ///
    /// The first input tensor's shape is inspected to build a realistic sample
    /// input. Dynamic dimensions (`-1`) are resolved to sensible defaults
    /// (batch = 1, spatial = 224, sequence = 128).
    pub fn load(path: &Path, provider: BenchProvider) -> Result<Self> {
        let providers = providers_for(provider);
        let config = SessionConfig::new().with_providers(providers);

        let engine = InferenceEngine::from_file_with_config(path, config)
            .with_context(|| format!("loading model from {}", path.display()))?;

        let shape = resolve_shape(engine.inputs());
        let distribution = infer_distribution(engine.inputs());
        let sample_input = realistic_input(&shape, distribution);

        Ok(Self {
            engine,
            sample_input,
        })
    }

    /// Run a single inference pass and return the wall-clock duration.
    ///
    /// The sample input is cloned *before* timing begins so that allocation
    /// cost is excluded from the measured latency.
    pub fn run_one(&mut self) -> Result<Duration> {
        let input = self.sample_input.clone();
        let t0 = Instant::now();
        self.engine.run(input)?;
        Ok(t0.elapsed())
    }

    /// Access the underlying engine (e.g. to inspect metadata).
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }
}

// ── BenchReport ──────────────────────────────────────────────────────────────

/// Full statistical benchmark report for a model.
#[derive(Debug, Clone)]
pub struct BenchReport {
    /// 50th-percentile latency in milliseconds.
    pub p50: f64,
    /// 90th-percentile latency in milliseconds.
    pub p90: f64,
    /// 95th-percentile latency in milliseconds.
    pub p95: f64,
    /// 99th-percentile latency in milliseconds.
    pub p99: f64,
    /// Arithmetic mean latency in milliseconds.
    pub mean: f64,
    /// Sample standard deviation in milliseconds.
    pub stddev: f64,
    /// Minimum observed latency in milliseconds.
    pub min: f64,
    /// Maximum observed latency in milliseconds.
    pub max: f64,
    /// Steady-state throughput in inferences per second.
    pub throughput: f64,
    /// Cold-start latency: time from engine-loaded to first-run-complete (ms).
    pub cold_start_ms: f64,
    /// Warm-start latency: average of the first `warmup` runs (ms).
    pub warm_start_ms: f64,
    /// Model file size in megabytes.
    pub model_size_mb: f64,
}

impl BenchReport {
    /// Emit a single markdown table row for this report.
    ///
    /// The row is designed to be appended to a table whose header is:
    /// `| Label | p50 | p90 | p95 | p99 | mean | stddev | min | max | inf/s | cold ms | warm ms | MB |`
    pub fn to_markdown(&self, label: &str) -> String {
        format!(
            "| {label} | {p50:.2} | {p90:.2} | {p95:.2} | {p99:.2} \
             | {mean:.2} | {stddev:.2} | {min:.2} | {max:.2} \
             | {thr:.1} | {cold:.2} | {warm:.2} | {mb:.2} |",
            label = label,
            p50 = self.p50,
            p90 = self.p90,
            p95 = self.p95,
            p99 = self.p99,
            mean = self.mean,
            stddev = self.stddev,
            min = self.min,
            max = self.max,
            thr = self.throughput,
            cold = self.cold_start_ms,
            warm = self.warm_start_ms,
            mb = self.model_size_mb,
        )
    }
}

// ── run_full_benchmark ────────────────────────────────────────────────────────

/// Run a complete benchmark, returning a [`BenchReport`].
///
/// # Measurement phases
///
/// 1. **Cold start** — time from "engine loaded" to "first inference complete".
/// 2. **Warm-up** — `warmup` additional runs; their average is `warm_start_ms`.
/// 3. **Steady state** — `iterations` runs; percentiles are computed over these.
///
/// The model file at `path` is used to determine `model_size_mb`.
pub fn run_full_benchmark(
    path: &Path,
    provider: BenchProvider,
    warmup: usize,
    iterations: usize,
) -> Result<BenchReport> {
    // File size before loading (the file must exist for load to succeed anyway).
    let model_size_mb = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len() as f64
        / (1024.0 * 1024.0);

    let providers = providers_for(provider);
    let config = SessionConfig::new().with_providers(providers);

    let engine = InferenceEngine::from_file_with_config(path, config)
        .with_context(|| format!("loading model from {}", path.display()))?;

    let shape = resolve_shape(engine.inputs());
    let distribution = infer_distribution(engine.inputs());
    let sample_input = realistic_input(&shape, distribution);

    let mut harness = BenchHarness {
        engine,
        sample_input,
    };

    // ── Cold start ──────────────────────────────────────────────────────────
    let cold_start_ms = harness.run_one()?.as_secs_f64() * 1000.0;

    // ── Warm-up phase ───────────────────────────────────────────────────────
    let warm_start_ms = if warmup == 0 {
        cold_start_ms
    } else {
        let mut total = Duration::ZERO;
        for _ in 0..warmup {
            total += harness.run_one()?;
        }
        total.as_secs_f64() * 1000.0 / warmup as f64
    };

    // ── Steady-state measurements ───────────────────────────────────────────
    let mut times_ms: Vec<f64> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        times_ms.push(harness.run_one()?.as_secs_f64() * 1000.0);
    }

    let report = compute_report(
        &times_ms,
        cold_start_ms,
        warm_start_ms,
        model_size_mb,
    );
    Ok(report)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Resolve dynamic tensor dimensions to concrete defaults.
///
/// Rules: batch dim = 1, spatial dims (≥224 sentinel or unknown) = 224,
/// sequence dims = 128.
fn resolve_shape(inputs: &[airml_core::TensorInfo]) -> Vec<usize> {
    if inputs.is_empty() {
        return vec![1, 3, 224, 224];
    }
    let info = &inputs[0];
    if info.shape.is_empty() {
        return vec![1, 3, 224, 224];
    }

    info.shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d > 0 {
                d as usize
            } else if i == 0 {
                // batch dimension
                1
            } else if info.shape.len() == 4 {
                // NCHW: dim 1 = channels, 2/3 = spatial
                if i == 1 { 3 } else { 224 }
            } else {
                // Sequence / other: use 128 as a safe default
                128
            }
        })
        .collect()
}

/// Heuristically choose an [`InputDistribution`] based on tensor metadata.
fn infer_distribution(inputs: &[airml_core::TensorInfo]) -> InputDistribution {
    if inputs.is_empty() {
        return InputDistribution::ImagenetNormalized;
    }
    let info = &inputs[0];

    // Token-ID heuristic: the name contains "input_ids", "token", or the
    // dtype is integer-like (ort exposes dtype as Debug string).
    let name_lower = info.name.to_lowercase();
    let is_int_dtype = info.dtype.to_lowercase().contains("int");

    if name_lower.contains("input_id")
        || name_lower.contains("token")
        || name_lower.contains("ids")
        || is_int_dtype
    {
        return InputDistribution::TokenIds { vocab_size: 30_522 }; // BERT default
    }

    // 4-D tensor → image model
    if info.shape.len() == 4 {
        return InputDistribution::ImagenetNormalized;
    }

    // Default: zero-mean unit-variance Gaussian
    InputDistribution::Gaussian {
        mean: 0.0,
        stddev: 1.0,
    }
}

/// Compute a [`BenchReport`] from a slice of millisecond latencies.
fn compute_report(
    times_ms: &[f64],
    cold_start_ms: f64,
    warm_start_ms: f64,
    model_size_mb: f64,
) -> BenchReport {
    assert!(!times_ms.is_empty(), "need at least one measurement");

    let mut sorted = times_ms.to_vec(); // sort a copy, never mutate the original
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let percentile = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
        sorted[idx.min(n - 1)]
    };

    let mean = times_ms.iter().sum::<f64>() / n as f64;
    let variance = times_ms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    let throughput = if mean > 0.0 { 1000.0 / mean } else { 0.0 };

    BenchReport {
        p50: percentile(50.0),
        p90: percentile(90.0),
        p95: percentile(95.0),
        p99: percentile(99.0),
        mean,
        stddev,
        min: sorted[0],
        max: sorted[n - 1],
        throughput,
        cold_start_ms,
        warm_start_ms,
        model_size_mb,
    }
}

// ── Report re-exports ─────────────────────────────────────────────────────────

pub use report::{report_to_json, report_to_json_full, save_report_to};

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_edge_cases() {
        // Single element: all percentiles must equal that element.
        let report = compute_report(&[42.0], 0.0, 0.0, 0.0);
        assert_eq!(report.p50, 42.0);
        assert_eq!(report.p90, 42.0);
        assert_eq!(report.p95, 42.0);
        assert_eq!(report.p99, 42.0);
        assert_eq!(report.min, 42.0);
        assert_eq!(report.max, 42.0);

        // Two elements: p50 = first, p99 = last.
        let report = compute_report(&[1.0, 3.0], 0.0, 0.0, 0.0);
        assert_eq!(report.min, 1.0);
        assert_eq!(report.max, 3.0);

        // Verify that the original slice is not mutated (sorted internally).
        let times = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let _ = compute_report(&times, 0.0, 0.0, 0.0);
        assert_eq!(times, vec![5.0, 1.0, 3.0, 2.0, 4.0], "original not mutated");
    }

    #[test]
    fn test_realistic_input_shape_correctness() {
        let shape = [2, 3, 8, 8];
        let arr = realistic_input(&shape, InputDistribution::Uniform { lo: 0.0, hi: 1.0 });
        assert_eq!(arr.shape(), shape.as_slice());
        assert_eq!(arr.len(), 2 * 3 * 8 * 8);

        let arr2 = realistic_input(&[1, 128], InputDistribution::TokenIds { vocab_size: 30_522 });
        assert_eq!(arr2.shape(), [1, 128]);
        // All token IDs must be in [0, 30522).
        for &v in arr2.iter() {
            assert!((0.0..30_522.0).contains(&v), "token id out of range: {v}");
        }

        let arr3 = realistic_input(
            &[1, 3, 224, 224],
            InputDistribution::ImagenetNormalized,
        );
        assert_eq!(arr3.len(), 3 * 224 * 224);
    }

    #[test]
    fn test_bench_report_markdown_output() {
        let report = BenchReport {
            p50: 12.345,
            p90: 20.0,
            p95: 22.5,
            p99: 30.0,
            mean: 13.5,
            stddev: 2.1,
            min: 10.0,
            max: 35.0,
            throughput: 74.1,
            cold_start_ms: 250.0,
            warm_start_ms: 15.0,
            model_size_mb: 4.5,
        };

        let row = report.to_markdown("mobilenetv3 / CPU");

        // Must start and end with pipes (valid markdown row).
        assert!(row.starts_with('|'), "row must start with |");
        assert!(row.ends_with('|'), "row must end with |");

        // Label must appear in the row.
        assert!(row.contains("mobilenetv3 / CPU"));

        // Key numeric values must appear.
        assert!(row.contains("12.35"), "p50 formatted");
        assert!(row.contains("74.1"), "throughput formatted");
        assert!(row.contains("4.50"), "model MB formatted");

        // Count columns: split on | gives N+2 parts (leading/trailing empty).
        let cols: Vec<&str> = row.split('|').collect();
        // Expected: "" | label | p50 | p90 | p95 | p99 | mean | stddev | min | max | inf/s | cold | warm | mb | ""
        assert_eq!(
            cols.len(),
            15,
            "expected 13 data columns + 2 boundary empties, got row: {row}"
        );
    }
}
