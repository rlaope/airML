# airml-core

The central inference crate. Wraps ONNX Runtime sessions behind a clean Rust API: load a model, run it, get tensors back. All other airML crates depend on this one.

**docs.rs:** `https://docs.rs/airml-core` (published with v1.0)

## Minimal example

```rust
use airml_core::{InferenceEngine, SessionConfig};
use airml_preprocess::ImagePreprocessor;

fn main() -> anyhow::Result<()> {
    let engine = InferenceEngine::from_file("model.onnx")?;
    let input = ImagePreprocessor::imagenet().load_and_process("image.jpg")?;
    let outputs = engine.run(input.into_dyn())?;
    println!("{:?}", outputs[0].shape());
    Ok(())
}
```

## Key types

### `InferenceEngine`

The main entry point. Load from file or bytes, run inference.

```rust
// From file
let engine = InferenceEngine::from_file("model.onnx")?;

// From bytes (embedded models)
let engine = InferenceEngine::from_bytes(model_bytes)?;

// With custom config
let config = SessionConfig::new().with_intra_threads(4);
let engine = InferenceEngine::from_file_with_config("model.onnx", config)?;

// Inference
let outputs: Vec<ArrayD<f32>> = engine.run(input)?;
let outputs = engine.run_multiple(vec![input_a, input_b])?;
let outputs = engine.run_named(vec![("input_ids", ids), ("attention_mask", mask)])?;
```

### `SessionConfig`

Builder for ONNX Runtime session options.

```rust
let config = SessionConfig::new()
    .with_intra_threads(4)
    .with_inter_threads(2)
    .with_optimization_level(level)
    .with_providers(providers);
```

### `ModelMetadata`

Returned by `engine.metadata()` — model name, description, version, producer, input/output tensor info.

### `AirMLError`

All errors are variants of `AirMLError`: `ModelNotFound`, `ModelLoadError`, `InferenceError`, `PreprocessError`, `ConfigError`, `OrtError`.

## Stability

Public API is tracked via `cargo public-api`. Breaking changes are gated until v1.0. See [Stability](../operations/stability.md).
