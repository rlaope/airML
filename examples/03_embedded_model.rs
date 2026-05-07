//! Example: Loading a model from in-memory bytes with `airml-embed`.
//!
//! This example reads bytes from a file path at runtime to simulate what
//! `include_bytes!` does at compile time.  To switch to true compile-time
//! embedding, replace the file-reading block with:
//!
//! ```ignore
//! static MODEL_BYTES: &[u8] = include_bytes!("../models/my_model.onnx");
//! let model = EmbeddedModel::new(MODEL_BYTES);
//! ```
//!
//! Usage:
//!   cargo run --example 03_embedded_model -- model.onnx

use anyhow::{Context, Result};

use airml_embed::EmbeddedModel;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx>", args[0]);
        std::process::exit(2);
    }

    let model_path = &args[1];

    // Read bytes from disk (stand-in for include_bytes! at compile time).
    let bytes = std::fs::read(model_path)
        .with_context(|| format!("Failed to read '{model_path}'"))?;

    // Wrap in EmbeddedModel.
    let model = EmbeddedModel::new(&bytes);
    println!("Model size: {} bytes ({:.2} MB)", model.size(), model.size() as f64 / 1_048_576.0);

    // Convert to InferenceEngine for inference.
    let engine = model.into_engine()
        .context("Failed to create inference engine from embedded bytes")?;

    // Print metadata.
    let meta = engine.metadata();
    println!("\nModel metadata:");
    println!("  name:     {:?}", meta.name);
    println!("  producer: {:?}", meta.producer);
    println!("  version:  {:?}", meta.version);

    println!("\nInputs ({}):", meta.inputs.len());
    for input in &meta.inputs {
        println!("  '{}' — shape {:?}  dtype {}", input.name, input.shape, input.dtype);
    }

    println!("\nOutputs ({}):", meta.outputs.len());
    for output in &meta.outputs {
        println!("  '{}' — shape {:?}  dtype {}", output.name, output.shape, output.dtype);
    }

    Ok(())
}
