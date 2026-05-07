//! Example: compile-time model embedding with `embed_model!`.
//!
//! This is the canonical pattern for shipping a fixed ONNX model inside a
//! Rust binary.  The `embed_model!` macro expands to a `static` backed by
//! `include_bytes!`, so the file is baked in at compile time — no filesystem
//! access is needed at runtime.
//!
//! The fixture used here is a minimal synthetic ONNX Identity model
//! (ir_version 7, opset 13) that maps input `x` (shape [1,4], float32) to
//! output `y` unchanged.  It lives at `models/synthetic-identity.onnx` in
//! the repository root and is 97 bytes.
//!
//! Usage:
//!   cargo run --example 04_embedded_compiletime

use airml_embed::{embed_model, EmbeddedModel};

// Embed the synthetic identity model at compile time.
// Path is relative to this source file: examples/ → ../models/.
embed_model!(IDENTITY, "../models/synthetic-identity.onnx");

fn main() -> anyhow::Result<()> {
    // IDENTITY is a `static EmbeddedModel<'static>` — no heap allocation,
    // no file I/O, no ONNX Runtime until we call into_engine().
    println!(
        "Embedded model size: {} bytes ({:.2} KB)",
        IDENTITY.size(),
        IDENTITY.size() as f64 / 1024.0,
    );

    // Convert the embedded bytes into a live InferenceEngine.
    let engine = EmbeddedModel::new(IDENTITY.bytes())
        .into_engine()
        .map_err(|e| anyhow::anyhow!("Failed to create engine: {e}"))?;

    // Print metadata so the demo is concrete.
    let meta = engine.metadata();
    println!("\nModel metadata:");
    println!("  name:     {:?}", meta.name);
    println!("  producer: {:?}", meta.producer);
    println!("  version:  {:?}", meta.version);

    println!("\nInputs ({}):", meta.inputs.len());
    for inp in &meta.inputs {
        println!("  '{}' — shape {:?}  dtype {}", inp.name, inp.shape, inp.dtype);
    }

    println!("\nOutputs ({}):", meta.outputs.len());
    for out in &meta.outputs {
        println!("  '{}' — shape {:?}  dtype {}", out.name, out.shape, out.dtype);
    }

    Ok(())
}
