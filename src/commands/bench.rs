//! Bench command implementation
//!
//! Benchmarks model inference performance.

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

    // Configure session
    let providers = match args.provider.as_str() {
        "auto" => auto_select_providers(),
        "cpu" => vec![airml_providers::CpuProvider::default().into_dispatch()],
        #[cfg(feature = "coreml")]
        "coreml" => vec![airml_providers::CoreMLProvider::default().into_dispatch()],
        _ => auto_select_providers(),
    };

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

    // Create random input
    let input = create_random_input(&shape);

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

fn create_random_input(shape: &[usize]) -> ArrayD<f32> {
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.001) % 1.0).collect();
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
