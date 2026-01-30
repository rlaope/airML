//! airML Core - Inference Engine
//!
//! Core module for running ONNX model inference.

mod engine;
mod error;
mod session;

pub use engine::{InferenceEngine, ModelMetadata};
pub use error::{AirMLError, Result};
pub use session::SessionConfig;

pub use ndarray;
pub use ort;
