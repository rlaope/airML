//! airML Embed - Model embedding support
//!
//! Provides utilities for embedding ONNX models directly into Rust binaries
//! using `include_bytes!`.
//!
//! # Example
//!
//! ```ignore
//! use airml_embed::EmbeddedModel;
//!
//! // Embed model at compile time
//! static MODEL_BYTES: &[u8] = include_bytes!("../models/resnet50.onnx");
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = EmbeddedModel::new(MODEL_BYTES)?;
//!     let engine = model.into_engine()?;
//!     // Use engine for inference...
//!     Ok(())
//! }
//! ```

use airml_core::{InferenceEngine, Result, SessionConfig};

/// An embedded model loaded from bytes
pub struct EmbeddedModel<'a> {
    bytes: &'a [u8],
    config: SessionConfig,
}

impl<'a> EmbeddedModel<'a> {
    /// Create a new embedded model from bytes
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            config: SessionConfig::default(),
        }
    }

    /// Create with custom session configuration
    pub fn with_config(bytes: &'a [u8], config: SessionConfig) -> Self {
        Self { bytes, config }
    }

    /// Set session configuration
    pub fn config(mut self, config: SessionConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the raw model bytes
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    /// Get the model size in bytes
    pub fn size(&self) -> usize {
        self.bytes.len()
    }

    /// Convert to InferenceEngine
    pub fn into_engine(self) -> Result<InferenceEngine> {
        InferenceEngine::from_bytes_with_config(self.bytes, self.config)
    }
}

/// Macro for embedding a model at compile time
///
/// # Example
///
/// ```ignore
/// use airml_embed::embed_model;
///
/// // Creates a static EmbeddedModel
/// embed_model!(RESNET, "../models/resnet50.onnx");
///
/// fn main() {
///     let engine = RESNET.clone().into_engine().unwrap();
/// }
/// ```
#[macro_export]
macro_rules! embed_model {
    ($name:ident, $path:literal) => {
        static $name: $crate::EmbeddedModel<'static> =
            $crate::EmbeddedModel::new(include_bytes!($path));
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_model_size() {
        let bytes = &[0u8; 100];
        let model = EmbeddedModel::new(bytes);
        assert_eq!(model.size(), 100);
    }
}
