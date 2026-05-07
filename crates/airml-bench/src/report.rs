//! JSON report serialisation for airML benchmarks.
//!
//! This module converts a [`BenchReport`] into a structured JSON value that
//! matches the canonical airML benchmark schema (schema_version = "1").
//! The schema file is at `bench/results/schema.json`.
//!
//! # Quick start
//!
//! ```no_run
//! use std::path::Path;
//! use airml_bench::{BenchReport, report_to_json, save_report_to};
//!
//! let report = BenchReport {
//!     p50: 12.3, p90: 18.1, p95: 20.0, p99: 30.0,
//!     mean: 13.5, stddev: 2.1, min: 10.0, max: 35.0,
//!     throughput: 74.0, cold_start_ms: 250.0, warm_start_ms: 14.0,
//!     model_size_mb: 133.0,
//! };
//!
//! let json = report_to_json("M2-Pro/CoreML-ANE", &report);
//! let path = save_report_to(
//!     Path::new("bench/results"),
//!     &report,
//!     "airml",
//!     "bge-small-en",
//!     "M2-Pro/CoreML-ANE",
//! ).unwrap();
//! println!("Saved to {}", path.display());
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use sysinfo::System;

use crate::BenchReport;

// ── Host info ─────────────────────────────────────────────────────────────────

/// Collect host information using `sysinfo`.
///
/// Returns a JSON object with `os`, `arch`, `cpu_brand`, and `ram_gb`.
fn host_info() -> Value {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.refresh_memory();

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());

    let ram_gb = sys.total_memory() / (1024 * 1024 * 1024);

    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu_brand": cpu_brand,
        "ram_gb": ram_gb,
    })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Serialise a [`BenchReport`] into the canonical airML benchmark JSON schema.
///
/// # Schema (version "1")
///
/// ```json
/// {
///   "schema_version": "1",
///   "runtime": "airml",
///   "runtime_version": "0.2.0",
///   "label": "M2-Pro/CoreML-ANE",
///   "model": "bge-small-en",
///   "model_size_mb": 133.0,
///   "host": { "os": "macos", "arch": "aarch64", "cpu_brand": "...", "ram_gb": 32 },
///   "metrics": {
///     "p50_ms": 12.3, "p90_ms": 18.1, "p95_ms": 20.0, "p99_ms": 30.0,
///     "mean_ms": 13.5, "stddev_ms": 2.1, "min_ms": 10.0, "max_ms": 35.0,
///     "throughput_inf_per_sec": 74.0,
///     "cold_start_ms": 250.0,
///     "warm_start_ms": 14.0
///   }
/// }
/// ```
pub fn report_to_json(label: &str, report: &BenchReport) -> Value {
    report_to_json_full(label, report, "airml", "unknown")
}

/// Like [`report_to_json`] but lets the caller specify `runtime` and `model`.
pub fn report_to_json_full(
    label: &str,
    report: &BenchReport,
    runtime: &str,
    model: &str,
) -> Value {
    json!({
        "schema_version": "1",
        "runtime": runtime,
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "label": label,
        "model": model,
        "model_size_mb": report.model_size_mb,
        "host": host_info(),
        "metrics": {
            "p50_ms": report.p50,
            "p90_ms": report.p90,
            "p95_ms": report.p95,
            "p99_ms": report.p99,
            "mean_ms": report.mean,
            "stddev_ms": report.stddev,
            "min_ms": report.min,
            "max_ms": report.max,
            "throughput_inf_per_sec": report.throughput,
            "cold_start_ms": report.cold_start_ms,
            "warm_start_ms": report.warm_start_ms,
        }
    })
}

/// Write a benchmark report as JSON to `dir/<runtime>__<model>__<label>.json`.
///
/// The `runtime`, `model`, and `label` strings are sanitised: spaces and `/`
/// are replaced with `-`, other special chars are stripped, so the filename is
/// always safe on every OS.
///
/// Returns the path of the written file.
pub fn save_report_to(
    dir: &Path,
    report: &BenchReport,
    runtime: &str,
    model: &str,
    label: &str,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating output directory {}", dir.display()))?;

    let sanitise = |s: &str| -> String {
        s.chars()
            .map(|c| match c {
                ' ' | '/' | '\\' => '-',
                c if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' => c,
                _ => '_',
            })
            .collect()
    };

    let filename = format!(
        "{}__{}__{}.json",
        sanitise(runtime),
        sanitise(model),
        sanitise(label)
    );
    let path = dir.join(&filename);

    let value = report_to_json_full(label, report, runtime, model);
    let json_str = serde_json::to_string_pretty(&value)
        .context("serialising benchmark report to JSON")?;

    std::fs::write(&path, json_str)
        .with_context(|| format!("writing report to {}", path.display()))?;

    Ok(path)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchReport;

    fn sample_report() -> BenchReport {
        BenchReport {
            p50: 12.3,
            p90: 18.1,
            p95: 20.5,
            p99: 30.0,
            mean: 13.5,
            stddev: 2.1,
            min: 10.0,
            max: 35.0,
            throughput: 74.07,
            cold_start_ms: 250.0,
            warm_start_ms: 14.0,
            model_size_mb: 133.0,
        }
    }

    #[test]
    fn schema_version_is_one() {
        let v = report_to_json("test-label", &sample_report());
        assert_eq!(v["schema_version"], "1");
    }

    #[test]
    fn required_top_level_fields_present() {
        let v = report_to_json_full("M2-label", &sample_report(), "airml", "bge-small-en");
        assert_eq!(v["runtime"], "airml");
        assert_eq!(v["model"], "bge-small-en");
        assert_eq!(v["label"], "M2-label");
        assert!(v["runtime_version"].is_string());
        assert!(v["model_size_mb"].is_number());
    }

    #[test]
    fn metrics_values_match_report() {
        let report = sample_report();
        let v = report_to_json("lbl", &report);
        let m = &v["metrics"];
        assert_eq!(m["p50_ms"].as_f64().unwrap(), report.p50);
        assert_eq!(m["p90_ms"].as_f64().unwrap(), report.p90);
        assert_eq!(m["p95_ms"].as_f64().unwrap(), report.p95);
        assert_eq!(m["p99_ms"].as_f64().unwrap(), report.p99);
        assert_eq!(m["mean_ms"].as_f64().unwrap(), report.mean);
        assert_eq!(m["stddev_ms"].as_f64().unwrap(), report.stddev);
        assert_eq!(m["min_ms"].as_f64().unwrap(), report.min);
        assert_eq!(m["max_ms"].as_f64().unwrap(), report.max);
        assert_eq!(
            m["throughput_inf_per_sec"].as_f64().unwrap(),
            report.throughput
        );
        assert_eq!(m["cold_start_ms"].as_f64().unwrap(), report.cold_start_ms);
        assert_eq!(m["warm_start_ms"].as_f64().unwrap(), report.warm_start_ms);
    }

    #[test]
    fn host_block_has_required_keys() {
        let v = report_to_json("lbl", &sample_report());
        let h = &v["host"];
        assert!(h["os"].is_string(), "host.os must be a string");
        assert!(h["arch"].is_string(), "host.arch must be a string");
        assert!(h["cpu_brand"].is_string(), "host.cpu_brand must be a string");
        assert!(h["ram_gb"].is_number(), "host.ram_gb must be a number");
        // os must be non-empty
        assert!(!h["os"].as_str().unwrap().is_empty());
    }

    #[test]
    fn save_report_creates_file_with_correct_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let report = sample_report();
        let path = save_report_to(dir.path(), &report, "airml", "bge-small-en", "M2-Pro/ANE")
            .expect("save_report_to");

        // File must exist
        assert!(path.exists(), "file should exist at {}", path.display());

        // Filename convention: runtime__model__label.json (slashes replaced)
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("airml__bge-small-en__"));
        assert!(name.ends_with(".json"));
        // Slash in label should be replaced with '-'
        assert!(!name.contains('/'), "filename must not contain slashes");

        // File must be valid JSON matching the schema
        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents)
            .expect("saved file is valid JSON");
        assert_eq!(parsed["schema_version"], "1");
        assert_eq!(parsed["runtime"], "airml");
        assert_eq!(parsed["model"], "bge-small-en");
    }
}
