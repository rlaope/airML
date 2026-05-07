//! Example: Text embedding inference with a BERT-style ONNX model.
//!
//! This example demonstrates the engine call shape using **synthetic** token
//! inputs.  Real tokenisation (BPE / WordPiece) requires a tokenizer file and
//! the `nlp` feature — that is not shown here.
//!
//! Usage:
//!   cargo run --example 02_text_embed -- model.onnx [seq_len]
//!
//! `seq_len` defaults to 8.
//!
//! Note on int64 inputs
//! --------------------
//! BERT models expect int64 `input_ids` and `attention_mask`.  The current
//! `InferenceEngine::run` API accepts f32 only.  This example casts the
//! synthetic int64 tokens to f32 as a demonstration shim.
//!
//! TODO: switch to `engine.run_int64(...)` when that variant is available in
//! airml-core.

use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};

use airml_core::{InferenceEngine, SessionConfig};
use airml_providers::auto_select_providers;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx> [seq_len]", args[0]);
        std::process::exit(2);
    }

    let model_path = &args[1];
    let seq_len: usize = args
        .get(2)
        .map(|s| s.parse::<usize>().context("seq_len must be a positive integer"))
        .transpose()?
        .unwrap_or(8);

    // Load model.
    let config = SessionConfig::new().with_providers(auto_select_providers());
    let mut engine = InferenceEngine::from_file_with_config(model_path, config)
        .with_context(|| format!("Failed to load model from '{model_path}'"))?;

    println!("Model loaded.");
    println!("  Inputs:  {:?}", engine.inputs().iter().map(|i| &i.name).collect::<Vec<_>>());
    println!("  Outputs: {:?}", engine.outputs().iter().map(|o| &o.name).collect::<Vec<_>>());
    println!("  seq_len: {seq_len}");

    // ---- Build synthetic token tensors [1, seq_len] -------------------------
    // BERT convention: CLS=101 at position 0, SEP=102 at last position,
    // filler token 1000 in between.  Attention mask is all-ones.
    let mut ids = vec![1000_f32; seq_len];
    ids[0] = 101.0;
    if seq_len > 1 {
        ids[seq_len - 1] = 102.0;
    }
    let mask = vec![1.0_f32; seq_len];

    let input_ids = ArrayD::from_shape_vec(IxDyn(&[1, seq_len]), ids)
        .context("Failed to create input_ids tensor")?;
    let attention_mask = ArrayD::from_shape_vec(IxDyn(&[1, seq_len]), mask)
        .context("Failed to create attention_mask tensor")?;

    // Feed both inputs by name (order must match model expectations).
    let outputs = engine
        .run_multiple(vec![input_ids, attention_mask])
        .context("Inference failed")?;

    // ---- Print first 8 embedding dimensions ---------------------------------
    let embedding = outputs.first().context("Model returned no outputs")?;
    let flat: Vec<f32> = embedding.iter().copied().collect();
    let display_dims = flat.len().min(8);

    println!("\nFirst {display_dims} embedding dimensions:");
    for (i, val) in flat.iter().take(display_dims).enumerate() {
        println!("  [{i}] {val:.6}");
    }
    println!("  (total output elements: {})", flat.len());

    Ok(())
}
