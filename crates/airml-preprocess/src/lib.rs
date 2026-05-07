//! airML Preprocess - Input preprocessing module
//!
//! Provides preprocessing utilities for various input types including images and text.

mod image;

#[cfg(feature = "nlp")]
mod text;

#[cfg(feature = "nlp")]
mod embedding;

pub use image::{ImagePreprocessor, ResizeMode};

#[cfg(feature = "nlp")]
pub use text::{TextPreprocessor, TextPreprocessError, TokenizedInput};

#[cfg(feature = "nlp")]
pub use embedding::embed_texts;

pub use ndarray;
