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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_messages() {
        let e = AirMLError::ModelNotFound("path/to/model.onnx".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(msg.contains("path/to/model.onnx"));

        let e = AirMLError::ModelLoadError("bad bytes".to_string());
        let msg = e.to_string();
        assert!(msg.contains("bad bytes"));

        let e = AirMLError::InvalidModelFormat("not onnx".to_string());
        let msg = e.to_string();
        assert!(msg.contains("not onnx"));

        let e = AirMLError::ShapeMismatch {
            expected: vec![1, 3, 224, 224],
            actual: vec![1, 3, 256, 256],
        };
        let msg = e.to_string();
        assert!(msg.contains("224"));
        assert!(msg.contains("256"));

        let e = AirMLError::TypeMismatch {
            expected: "f32".to_string(),
            actual: "i64".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("f32"));
        assert!(msg.contains("i64"));

        let e = AirMLError::InferenceError("kernel panic".to_string());
        let msg = e.to_string();
        assert!(msg.contains("kernel panic"));

        let e = AirMLError::ProviderNotAvailable("CoreML".to_string());
        let msg = e.to_string();
        assert!(msg.contains("CoreML"));

        let e = AirMLError::ImageError("corrupt jpeg".to_string());
        let msg = e.to_string();
        assert!(msg.contains("corrupt jpeg"));

        let e = AirMLError::OrtError("ort internal".to_string());
        let msg = e.to_string();
        assert!(msg.contains("ort internal"));

        let e = AirMLError::ConfigError("missing field".to_string());
        let msg = e.to_string();
        assert!(msg.contains("missing field"));
    }

    #[test]
    fn test_error_from_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let air_err: AirMLError = io_err.into();
        let msg = air_err.to_string();
        assert!(!msg.is_empty());
        assert!(msg.contains("file missing"));
    }
}
