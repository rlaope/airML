//! Prometheus metrics for airML.
//!
//! Call [`init`] once at startup (in `main`) before dispatching subcommands.
//! After initialisation, use the statics directly:
//!
//! ```rust,no_run
//! if let Some(c) = crate::metrics::INFERENCE_COUNT.get() { c.inc(); }
//! if let Some(h) = crate::metrics::INFERENCE_LATENCY.get() {
//!     h.observe(elapsed.as_secs_f64());
//! }
//! ```
//!
//! The rendered Prometheus text output is available via [`render`].

use prometheus::{Encoder, Histogram, HistogramOpts, IntCounter, Registry, TextEncoder};
use std::sync::OnceLock;

pub static REGISTRY: OnceLock<Registry> = OnceLock::new();
pub static INFERENCE_COUNT: OnceLock<IntCounter> = OnceLock::new();
pub static INFERENCE_LATENCY: OnceLock<Histogram> = OnceLock::new();
pub static MODEL_LOAD_LATENCY: OnceLock<Histogram> = OnceLock::new();

/// Initialise the Prometheus registry and register all airML metrics.
///
/// Safe to call multiple times — subsequent calls are no-ops because
/// [`OnceLock`] ignores the second `set`.
///
/// # Errors
///
/// Returns a [`prometheus::Error`] if metric registration fails (should not
/// happen in practice unless the function is called concurrently during tests).
pub fn init() -> Result<(), prometheus::Error> {
    let r = Registry::new();

    let inferences = IntCounter::new("airml_inference_total", "total inference calls")?;
    let latency = Histogram::with_opts(
        HistogramOpts::new("airml_inference_latency_seconds", "inference latency")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
    )?;
    let load = Histogram::with_opts(
        HistogramOpts::new("airml_model_load_seconds", "model load time")
            .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0]),
    )?;

    r.register(Box::new(inferences.clone()))?;
    r.register(Box::new(latency.clone()))?;
    r.register(Box::new(load.clone()))?;

    // OnceLock::set returns Err if already set — that's fine, ignore it.
    REGISTRY.set(r).ok();
    INFERENCE_COUNT.set(inferences).ok();
    INFERENCE_LATENCY.set(latency).ok();
    MODEL_LOAD_LATENCY.set(load).ok();

    Ok(())
}

/// Render the current metrics as Prometheus text format.
///
/// # Errors
///
/// Returns an error if metrics have not been initialised via [`init`], or if
/// the text encoder fails (extremely unlikely in practice).
pub fn render() -> Result<String, prometheus::Error> {
    let r = REGISTRY
        .get()
        .ok_or_else(|| prometheus::Error::Msg("metrics not initialized".into()))?;
    let mfs = r.gather();
    let mut buf = Vec::new();
    TextEncoder::new().encode(&mfs, &mut buf)?;
    Ok(String::from_utf8(buf).unwrap_or_default())
}
