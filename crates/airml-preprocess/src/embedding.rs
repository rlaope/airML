//! Embedding helper — wraps tokenizer + engine into a single call.
//!
//! Requires the `nlp` feature.

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

use airml_core::InferenceEngine;

/// Tokenize `texts`, run inference through `engine`, apply mean-pooling over
/// real (non-padding) tokens, and optionally L2-normalize each vector.
///
/// Returns one `Vec<f32>` per input text.
pub fn embed_texts(
    engine: &mut InferenceEngine,
    tokenizer: &Tokenizer,
    texts: &[String],
    max_length: usize,
    l2_normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    let mut results = Vec::with_capacity(texts.len());

    for text in texts {
        // Tokenize with truncation/padding to max_length
        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

        let ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .take(max_length)
            .map(|&x| x as i64)
            .collect();

        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .take(max_length)
            .map(|&x| x as i64)
            .collect();

        let seq_len = ids.len();

        let input_ids =
            ndarray::Array2::from_shape_vec((1, seq_len), ids)
                .map_err(|e| anyhow!("Array build failed: {e}"))?;

        let attention_mask =
            ndarray::Array2::from_shape_vec((1, seq_len), mask.clone())
                .map_err(|e| anyhow!("Array build failed: {e}"))?;

        let n_inputs = engine.inputs().len();
        let outputs = if n_inputs >= 2 {
            engine
                .run_multiple(vec![
                    input_ids.into_dyn().mapv(|x| x as f32),
                    attention_mask.into_dyn().mapv(|x| x as f32),
                ])
                .map_err(|e| anyhow!("Inference failed: {e}"))?
        } else {
            engine
                .run(input_ids.into_dyn().mapv(|x| x as f32))
                .map_err(|e| anyhow!("Inference failed: {e}"))?
        };

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Model produced no output"))?;

        let shape = output.shape().to_vec();
        let embedding: Vec<f32> = match shape.len() {
            2 => {
                // [batch=1, hidden] — direct embedding
                output.iter().copied().collect()
            }
            3 => {
                // [batch=1, seq_len, hidden] — mean-pool over real tokens
                let hidden = shape[2];
                let mut pooled = vec![0.0f32; hidden];
                let mut real_tokens = 0.0f32;

                for i in 0..shape[1] {
                    let m = if i < mask.len() { mask[i] as f32 } else { 0.0 };
                    if m > 0.0 {
                        real_tokens += 1.0;
                        for j in 0..hidden {
                            pooled[j] += output[[0, i, j]];
                        }
                    }
                }

                if real_tokens > 0.0 {
                    for v in &mut pooled {
                        *v /= real_tokens;
                    }
                }
                pooled
            }
            _ => output.iter().copied().collect(),
        };

        let final_vec = if l2_normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                embedding.iter().map(|x| x / norm).collect()
            } else {
                embedding
            }
        } else {
            embedding
        };

        results.push(final_vec);
    }

    Ok(results)
}
