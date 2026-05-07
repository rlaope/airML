//! airML Tune — Automatic backend dispatcher and performance oracle.
//!
//! Given ONNX model metadata, `airml-tune` recommends the optimal CoreML
//! compute units (or CPU fallback) without requiring a benchmark run.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use airml_tune::oracle::{BackendOracle, BackendRecommendation};
//! use airml_core::ModelMetadata;
//!
//! let oracle = BackendOracle::new();
//! let rec = oracle.recommend_for_metadata(&metadata);
//! println!("{:?}", rec);
//! ```
//!
//! For higher accuracy when the model file is available, use the graph-parsing
//! path which reads the actual ONNX op histogram instead of relying on
//! input-name heuristics:
//!
//! ```rust,no_run
//! use airml_tune::oracle::BackendOracle;
//! use std::path::Path;
//!
//! let oracle = BackendOracle::new();
//! if let Ok(rec) = oracle.recommend_for_path(Path::new("model.onnx")) {
//!     println!("{:?}", rec);
//! }
//! ```

pub mod graph_parser;
pub mod oracle;
pub mod profile_cache;

#[cfg(feature = "coreml")]
pub mod dispatch;

pub use graph_parser::{histogram_from_bytes, histogram_from_path, GraphParseError, OpHistogram};
pub use oracle::{BackendOracle, BackendRecommendation, ModelFamily, ModelProfile, OpClass};
pub use profile_cache::ProfileCache;
