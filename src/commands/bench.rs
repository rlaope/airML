//! Bench command implementation
//!
//! Benchmarks model inference performance.
//!
//! Note: this is a quick-look harness. The real benchmark suite with
//! criterion, warmup calibration, and HTML reports lives in the
//! `airml-bench` crate.

use std::time::{Duration, Instant};

use airml_core::{ndarray, InferenceEngine, SessionConfig};
use airml_providers::auto_select_providers;
use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};

use crate::cli::BenchArgs;

/// Execute the bench command
pub fn execute(args: &BenchArgs) -> Result<()> {
    println!("Benchmark: {}", args.model.display());
    println!("{:=<60}", "");

    // Configure session — probe with CPU first to get metadata for BackendOracle.
    let providers = select_bench_providers(args)?;

    let config = SessionConfig::new().with_providers(providers);

    // Load model
    println!("Loading model...");
    let load_start = Instant::now();
    let engine = InferenceEngine::from_file_with_config(&args.model, config)
        .context("Failed to load model")?;
    let load_time = load_start.elapsed();
    println!("Model loaded in {:.3} ms", load_time.as_secs_f64() * 1000.0);

    // Determine input shape
    let shape = if let Some(shape_str) = &args.shape {
        parse_shape(shape_str)?
    } else {
        get_model_input_shape(&engine)?
    };

    println!("Input shape: {:?}", shape);
    println!();

    // Create realistic pseudo-Gaussian input.
    let input = realistic_input(&shape);

    // Warmup
    let mut engine = engine;
    println!("Warming up ({} iterations)...", args.warmup);
    for _ in 0..args.warmup {
        let _ = engine.run(input.clone());
    }

    // Benchmark
    println!("Running benchmark ({} iterations)...", args.iterations);
    let mut times = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        let start = Instant::now();
        let _ = engine.run(input.clone()).context("Inference failed")?;
        times.push(start.elapsed());

        // Progress indicator
        if (i + 1) % 10 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!();
    println!();

    // Calculate statistics
    let stats = calculate_stats(&times);

    // Print results
    println!("Results:");
    println!("{:-<60}", "");
    println!("  Total iterations: {}", args.iterations);
    println!("  Total time:       {:.3} ms", stats.total.as_secs_f64() * 1000.0);
    println!();
    println!("  Mean latency:     {:.3} ms", stats.mean.as_secs_f64() * 1000.0);
    println!("  Median latency:   {:.3} ms", stats.median.as_secs_f64() * 1000.0);
    println!("  Min latency:      {:.3} ms", stats.min.as_secs_f64() * 1000.0);
    println!("  Max latency:      {:.3} ms", stats.max.as_secs_f64() * 1000.0);
    println!("  Std deviation:    {:.3} ms", stats.std_dev * 1000.0);
    println!();
    println!("  Throughput:       {:.2} inferences/sec", stats.throughput);
    println!();
    println!("  P50:              {:.3} ms", stats.p50.as_secs_f64() * 1000.0);
    println!("  P90:              {:.3} ms", stats.p90.as_secs_f64() * 1000.0);
    println!("  P95:              {:.3} ms", stats.p95.as_secs_f64() * 1000.0);
    println!("  P99:              {:.3} ms", stats.p99.as_secs_f64() * 1000.0);

    Ok(())
}

fn parse_shape(shape_str: &str) -> Result<Vec<usize>> {
    shape_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .context("Invalid shape dimension")
        })
        .collect()
}

fn get_model_input_shape(engine: &InferenceEngine) -> Result<Vec<usize>> {
    let input = engine
        .inputs()
        .first()
        .context("Model has no inputs")?;

    let shape: Vec<usize> = input
        .shape
        .iter()
        .map(|&d| {
            if d <= 0 {
                1 // Replace dynamic dimensions with 1
            } else {
                d as usize
            }
        })
        .collect();

    Ok(shape)
}

/// Build a provider list for benchmarking, consulting BackendOracle when `--provider auto`.
fn select_bench_providers(
    args: &BenchArgs,
) -> Result<Vec<airml_providers::ExecutionProviderDispatch>> {
    match args.provider.as_str() {
        "auto" => {
            #[cfg(feature = "coreml")]
            {
                let oracle = airml_tune::BackendOracle::new();

                // Prefer graph-based classification for higher accuracy.
                // Fall back to metadata heuristics if graph parsing fails
                // (e.g. corrupt file, unsupported opset).
                let rec = match oracle.recommend_for_path(&args.model) {
                    Ok(graph_rec) => {
                        // Patch with session metadata for family + dynamic shapes.
                        let probe_config =
                            SessionConfig::new().with_providers(auto_select_providers());
                        if let Ok(probe) =
                            InferenceEngine::from_file_with_config(&args.model, probe_config)
                        {
                            let mut profile =
                                oracle.profile_from_metadata(probe.metadata());
                            if let Ok(hist) = airml_tune::histogram_from_path(&args.model) {
                                profile.dominant_op_class = hist.dominant_class();
                            }
                            oracle.recommend(&profile)
                        } else {
                            graph_rec
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "[airml-tune] graph parse failed, falling back to metadata heuristics: {err}"
                        );
                        let probe_config =
                            SessionConfig::new().with_providers(auto_select_providers());
                        if let Ok(probe) =
                            InferenceEngine::from_file_with_config(&args.model, probe_config)
                        {
                            oracle.recommend_for_metadata(probe.metadata())
                        } else {
                            return Ok(auto_select_providers());
                        }
                    }
                };

                let providers = airml_tune::dispatch::recommendation_to_providers(&rec);
                eprintln!(
                    "[airml-tune] selected backend: {:?} (model: {})",
                    rec,
                    args.model.file_name().unwrap_or_default().to_string_lossy()
                );
                return Ok(providers);
            }
            #[allow(unreachable_code)]
            Ok(auto_select_providers())
        }
        "cpu" => Ok(vec![airml_providers::CpuProvider::default().into_dispatch()]),
        #[cfg(feature = "coreml")]
        "coreml" => Ok(vec![airml_providers::CoreMLProvider::default().into_dispatch()]),
        #[cfg(feature = "coreml")]
        "neural-engine" => Ok(vec![airml_providers::CoreMLProvider::default()
            .neural_engine_only()
            .into_dispatch()]),
        _ => Ok(auto_select_providers()),
    }
}

/// Generate pseudo-Gaussian data via the central limit theorem (sum of 12 uniforms − 6).
///
/// This produces more realistic activations than a ramp or uniform distribution,
/// reducing the chance that synthetic inputs trigger degenerate fast-paths in
/// quantised or sparse kernels.
fn realistic_input(shape: &[usize]) -> ArrayD<f32> {
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    // Splitmix64-style LCG — deterministic, no external deps.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..total {
        let mut sum = 0.0_f32;
        for _ in 0..12 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32 as f32 / u32::MAX as f32;
            sum += u;
        }
        data.push(sum - 6.0);
    }
    ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
}

struct BenchStats {
    total: Duration,
    mean: Duration,
    median: Duration,
    min: Duration,
    max: Duration,
    std_dev: f64,
    throughput: f64,
    p50: Duration,
    p90: Duration,
    p95: Duration,
    p99: Duration,
}

fn calculate_stats(times: &[Duration]) -> BenchStats {
    let mut sorted = times.to_vec();
    sorted.sort();

    let total: Duration = times.iter().sum();
    let count = times.len() as f64;

    let mean_nanos = total.as_nanos() as f64 / count;
    let mean = Duration::from_nanos(mean_nanos as u64);

    let min = *sorted.first().unwrap();
    let max = *sorted.last().unwrap();
    let median = sorted[sorted.len() / 2];

    // Standard deviation
    let variance: f64 = times
        .iter()
        .map(|t| {
            let diff = t.as_secs_f64() - mean.as_secs_f64();
            diff * diff
        })
        .sum::<f64>()
        / count;
    let std_dev = variance.sqrt();

    // Throughput
    let throughput = count / total.as_secs_f64();

    // Percentiles
    let p50 = percentile(&sorted, 50);
    let p90 = percentile(&sorted, 90);
    let p95 = percentile(&sorted, 95);
    let p99 = percentile(&sorted, 99);

    BenchStats {
        total,
        mean,
        median,
        min,
        max,
        std_dev,
        throughput,
        p50,
        p90,
        p95,
        p99,
    }
}

fn percentile(sorted: &[Duration], p: usize) -> Duration {
    let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
    sorted[idx]
}
