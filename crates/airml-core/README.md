# airml-core

Core ONNX inference engine powering the airML runtime.

`airml-core` wraps [ONNX Runtime](https://onnxruntime.ai/) (via the `ort` crate) and exposes a
clean, ergonomic Rust API for loading models, building sessions, and running inference with
`ndarray` tensors. It is the foundation that every other airML crate builds on.

## Quickstart

```rust
use airml_core::{InferenceEngine, SessionConfig};
use ndarray::Array4;

fn main() -> airml_core::Result<()> {
    // Load a model from disk
    let engine = InferenceEngine::from_file("model.onnx")?;

    // Inspect model I/O
    for input in engine.inputs() {
        println!("input: {} shape={:?}", input.name, input.shape);
    }

    // Build a batch-1 NCHW image tensor and run inference
    let img: ndarray::ArrayD<f32> = Array4::<f32>::zeros((1, 3, 224, 224)).into_dyn();
    let outputs = engine.run(img)?;
    println!("logits shape: {:?}", outputs[0].shape());

    Ok(())
}
```

With a custom session (e.g. setting thread counts or execution providers):

```rust
use airml_core::{InferenceEngine, SessionConfig};

let config = SessionConfig::new()
    .with_intra_threads(4)
    .with_inter_threads(1);

let engine = InferenceEngine::from_file_with_config("model.onnx", config)?;
```

## Feature flags

This crate has no Cargo features of its own. Execution-provider features are
controlled by `airml-providers`.

## Key types

| Type | Description |
|------|-------------|
| `InferenceEngine` | Loaded model session; call `.run()` to infer |
| `SessionConfig` | Builder for thread counts, providers, etc. |
| `ModelMetadata` | Name, description, custom metadata from the ONNX graph |
| `TensorInfo` | Name, dtype, shape for each model input/output |
| `AirMLError` | Unified error type (implements `std::error::Error`) |

## Links

- [Main project README](../../README.md)
- [airML on GitHub](https://github.com/airml/airml)
- [API docs on docs.rs](https://docs.rs/airml-core)
