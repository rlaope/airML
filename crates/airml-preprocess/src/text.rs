//! Text preprocessing for NLP models
//!
//! Provides tokenization and text processing utilities.
//! This module requires the "nlp" feature.

use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer;

/// Errors that can occur during text preprocessing
#[derive(Error, Debug)]
pub enum TextPreprocessError {
    #[error("Failed to load tokenizer: {0}")]
    LoadError(String),

    #[error("Failed to encode text: {0}")]
    EncodeError(String),

    #[error("Text too long: {actual} tokens (max: {max})")]
    TextTooLong { actual: usize, max: usize },
}

type Result<T> = std::result::Result<T, TextPreprocessError>;

/// Text preprocessor with tokenization support
pub struct TextPreprocessor {
    tokenizer: Tokenizer,
    max_length: usize,
    padding: bool,
    truncation: bool,
}

impl TextPreprocessor {
    /// Load a tokenizer from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| TextPreprocessError::LoadError(e.to_string()))?;

        Ok(Self {
            tokenizer,
            max_length: 512,
            padding: true,
            truncation: true,
        })
    }

    /// Load a tokenizer from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|e| TextPreprocessError::LoadError(e.to_string()))?;

        Ok(Self {
            tokenizer,
            max_length: 512,
            padding: true,
            truncation: true,
        })
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Enable/disable padding
    pub fn with_padding(mut self, padding: bool) -> Self {
        self.padding = padding;
        self
    }

    /// Enable/disable truncation
    pub fn with_truncation(mut self, truncation: bool) -> Self {
        self.truncation = truncation;
        self
    }

    /// Tokenize a single text
    pub fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| TextPreprocessError::EncodeError(e.to_string()))?;

        let ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        if ids.len() > self.max_length && !self.truncation {
            return Err(TextPreprocessError::TextTooLong {
                actual: ids.len(),
                max: self.max_length,
            });
        }

        let (ids, attention_mask) = if self.truncation && ids.len() > self.max_length {
            (
                ids[..self.max_length].to_vec(),
                attention_mask[..self.max_length].to_vec(),
            )
        } else if self.padding && ids.len() < self.max_length {
            let mut padded_ids = ids;
            let mut padded_mask = attention_mask;
            padded_ids.resize(self.max_length, 0);
            padded_mask.resize(self.max_length, 0);
            (padded_ids, padded_mask)
        } else {
            (ids, attention_mask)
        };

        Ok(TokenizedInput {
            input_ids: ids,
            attention_mask,
        })
    }

    /// Tokenize a batch of texts
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<TokenizedInput>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

/// Tokenized input ready for model inference
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    /// Token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,
}

impl TokenizedInput {
    /// Convert to ndarray for model input
    pub fn to_array(&self) -> (ndarray::Array2<i64>, ndarray::Array2<i64>) {
        let input_ids: Vec<i64> = self.input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = self.attention_mask.iter().map(|&x| x as i64).collect();

        let len = input_ids.len();
        (
            ndarray::Array2::from_shape_vec((1, len), input_ids).unwrap(),
            ndarray::Array2::from_shape_vec((1, len), attention_mask).unwrap(),
        )
    }
}
