//! Error types for airML
//!
//! Defines all error types used across the airML crate.

use thiserror::Error;

/// Main error type for airML operations
#[derive(Error, Debug)]
pub enum AirMLError {
    /// Model file not found or inaccessible
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Failed to load ONNX model
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// Invalid model format
    #[error("Invalid model format: {0}")]
    InvalidModelFormat(String),

    /// Input shape mismatch
    #[error("Input shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<i64>,
        actual: Vec<usize>,
    },

    /// Input type mismatch
    #[error("Input type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    /// Inference execution error
    #[error("Inference failed: {0}")]
    InferenceError(String),

    /// Execution provider not available
    #[error("Execution provider not available: {0}")]
    ProviderNotAvailable(String),

    /// Image processing error
    #[error("Image processing error: {0}")]
    ImageError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// ORT (ONNX Runtime) error
    #[error("ONNX Runtime error: {0}")]
    OrtError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl From<ort::Error> for AirMLError {
    fn from(err: ort::Error) -> Self {
        AirMLError::OrtError(err.to_string())
    }
}

/// Result type alias for airML operations
pub type Result<T> = std::result::Result<T, AirMLError>;
