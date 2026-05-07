//! tract-onnx baseline benchmark for airML cross-runtime comparison.
//!
//! Reads the model path from the `MODEL_PATH` environment variable, runs
//! warmup passes, then timed passes, and prints a single JSON report to
//! stdout matching the airML benchmark schema v1 (`bench/results/schema.json`).
//!
//! tract runs on CPU only (no hardware accelerators), making it a useful
//! apples-to-apples CPU comparison baseline for airML's CPU provider.
//!
//! # Usage
//!
//! ```bash
//! MODEL_PATH=path/to/model.onnx cargo run --release [-- --iterations 100 --warmup 10 --label "M2-Pro/CPU"]
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    let args = parse_args();

    let model_path: PathBuf = env::var("MODEL_PATH")
        .context("MODEL_PATH env var must be set to the ONNX model path")?
        .into();

    let model_size_mb = std::fs::metadata(&model_path)
        .with_context(|| format!("stat {}", model_path.display()))?
        .len() as f64
        / (1024.0 * 1024.0);

    // ── Load and optimise model ──────────────────────────────────────────────
    let model = tract_onnx::onnx()
        .model_for_path(&model_path)
        .with_context(|| format!("loading model from {}", model_path.display()))?
        .into_optimized()?
        .into_runnable()?;

    // ── Build synthetic input ────────────────────────────────────────────────
    let input = make_input(&model)?;

    // ── Cold start ───────────────────────────────────────────────────────────
    let cold_start_t0 = Instant::now();
    model.run(tvec!(input.clone().into()))?;
    let cold_start_ms = cold_start_t0.elapsed().as_secs_f64() * 1000.0;

    // ── Warmup ───────────────────────────────────────────────────────────────
    let mut warm_total_ms = 0.0_f64;
    for _ in 0..args.warmup {
        let t = Instant::now();
        model.run(tvec!(input.clone().into()))?;
        warm_total_ms += t.elapsed().as_secs_f64() * 1000.0;
    }
    let warm_start_ms = if args.warmup > 0 {
        warm_total_ms / args.warmup as f64
    } else {
        cold_start_ms
    };

    // ── Timed iterations ─────────────────────────────────────────────────────
    let mut times_ms: Vec<f64> = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let t = Instant::now();
        model.run(tvec!(input.clone().into()))?;
        times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let model_name = model_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    let report = build_report(
        &times_ms,
        cold_start_ms,
        warm_start_ms,
        model_size_mb,
        &args.label,
        &model_name,
    );

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_input(model: &SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>) -> Result<Tensor> {
    // Inspect first input fact for shape.
    let fact = model.model().input_fact(0)?;
    let shape: Vec<usize> = fact
        .shape
        .as_concrete()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![1, 3, 224, 224]);

    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) % 1.0).collect();
    Ok(tract_ndarray::Array::from_shape_vec(
        tract_ndarray::IxDyn(&shape),
        data,
    )?.into())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn build_report(
    times_ms: &[f64],
    cold_start_ms: f64,
    warm_start_ms: f64,
    model_size_mb: f64,
    label: &str,
    model: &str,
) -> serde_json::Value {
    let mut sorted = times_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len() as f64;
    let mean = times_ms.iter().sum::<f64>() / n;
    let variance = times_ms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();
    let throughput = if mean > 0.0 { 1000.0 / mean } else { 0.0 };

    serde_json::json!({
        "schema_version": "1",
        "runtime": "tract",
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "label": label,
        "model": model,
        "model_size_mb": model_size_mb,
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "cpu_brand": cpu_brand(),
            "ram_gb": ram_gb(),
        },
        "metrics": {
            "p50_ms": percentile(&sorted, 50.0),
            "p90_ms": percentile(&sorted, 90.0),
            "p95_ms": percentile(&sorted, 95.0),
            "p99_ms": percentile(&sorted, 99.0),
            "mean_ms": mean,
            "stddev_ms": stddev,
            "min_ms": sorted.first().copied().unwrap_or(0.0),
            "max_ms": sorted.last().copied().unwrap_or(0.0),
            "throughput_inf_per_sec": throughput,
            "cold_start_ms": cold_start_ms,
            "warm_start_ms": warm_start_ms,
        }
    })
}

fn cpu_brand() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            return String::from_utf8_lossy(&out.stdout).trim().to_owned();
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("model name") {
                    if let Some(v) = line.splitn(2, ':').nth(1) {
                        return v.trim().to_owned();
                    }
                }
            }
        }
    }
    "unknown".to_owned()
}

fn ram_gb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes / (1024 * 1024 * 1024);
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(mi) = std::fs::read_to_string("/proc/meminfo") {
            for line in mi.lines() {
                if line.starts_with("MemTotal:") {
                    let kb: u64 = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                    return kb / (1024 * 1024);
                }
            }
        }
    }
    0
}

struct Args {
    iterations: usize,
    warmup: usize,
    label: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut iterations = 100usize;
    let mut warmup = 10usize;
    let mut label = String::from("unknown");

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--iterations" | "-n" => {
                if let Some(v) = argv.get(i + 1) {
                    iterations = v.parse().unwrap_or(100);
                    i += 1;
                }
            }
            "--warmup" | "-w" => {
                if let Some(v) = argv.get(i + 1) {
                    warmup = v.parse().unwrap_or(10);
                    i += 1;
                }
            }
            "--label" | "-l" => {
                if let Some(v) = argv.get(i + 1) {
                    label = v.clone();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: MODEL_PATH=<path> airml-bench-tract [--iterations N] [--warmup N] [--label TEXT]"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    Args { iterations, warmup, label }
}
