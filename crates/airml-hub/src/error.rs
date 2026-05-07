//! Error types for `airml-hub`.
//!
//! All hub operations return [`Result<T>`] which is an alias for
//! `std::result::Result<T, HubError>`.

use thiserror::Error;

/// All errors that can occur during model hub operations.
#[derive(Error, Debug)]
pub enum HubError {
    /// An HTTP request failed with a status code or transport error.
    #[error("HTTP error: {0}")]
    HttpError(String),

    /// A filesystem I/O operation failed.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// The downloaded bytes did not match the expected SHA-256 digest.
    #[error("SHA-256 mismatch: expected {expected}, got {actual}")]
    Sha256Mismatch { expected: String, actual: String },

    /// The requested model ID was not found in the built-in registry.
    #[error("Unknown model: {0}")]
    UnknownModel(String),

    /// A cache operation could not be completed (e.g. corrupt index, bad path).
    #[error("Cache error: {0}")]
    CacheError(String),

    /// The supplied URI string could not be parsed.
    #[error("Malformed URI: {0}")]
    MalformedUri(String),
}

/// Convenience alias used throughout this crate.
pub type Result<T> = std::result::Result<T, HubError>;
