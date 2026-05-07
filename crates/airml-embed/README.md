# airml-embed

Compile-time ONNX model embedding into Rust binaries via `include_bytes!`.

`airml-embed` lets you ship a self-contained binary that carries the model weights inside the
executable — no separate file needed at runtime. The `embed_model!` macro creates a
`LazyLock`-backed static that loads the model bytes exactly once on first access.

## Quickstart

```rust
use airml_embed::{embed_model, EmbeddedModel};

// Embed the model at compile time (path is relative to this source file)
embed_model!(CLASSIFIER, "../models/resnet50.onnx");

fn main() -> airml_core::Result<()> {
    // First access initialises the LazyLock; subsequent calls are free
    let engine = CLASSIFIER.clone().into_engine()?;

    // Run inference as usual
    let input = ndarray::Array4::<f32>::zeros((1, 3, 224, 224)).into_dyn();
    let _outputs = engine.run(input)?;
    Ok(())
}
```

You can also construct an `EmbeddedModel` at runtime from any byte slice:

```rust
use airml_embed::EmbeddedModel;
use airml_core::SessionConfig;

let bytes: &[u8] = include_bytes!("../models/resnet50.onnx");
let model = EmbeddedModel::with_config(bytes, SessionConfig::new().with_intra_threads(4));
let engine = model.into_engine()?;
```

## Feature flags

This crate has no optional Cargo features.

## Minimum Rust version

`embed_model!` uses `std::sync::LazyLock`, which requires **Rust 1.80**.

## Links

- [Main project README](../../README.md)
- [airML on GitHub](https://github.com/airml/airml)
- [API docs on docs.rs](https://docs.rs/airml-embed)
