# airml-embed

Utilities for embedding ONNX models directly into Rust binaries at compile time. The model bytes are baked into the binary via `include_bytes!`, so no external file is needed at runtime.

## Usage

### Manual embedding

```rust
use airml_embed::EmbeddedModel;

static MODEL: &[u8] = include_bytes!("../models/resnet50.onnx");

fn main() -> anyhow::Result<()> {
    let engine = EmbeddedModel::new(MODEL).into_engine()?;
    // use engine...
    Ok(())
}
```

### With custom config

```rust
use airml_embed::EmbeddedModel;
use airml_core::SessionConfig;

let config = SessionConfig::new().with_intra_threads(2);
let engine = EmbeddedModel::with_config(MODEL, config).into_engine()?;
```

### `embed_model!` macro

```rust
use airml_embed::embed_model;

// Declares a static EmbeddedModel
embed_model!(RESNET, "../models/resnet50.onnx");

fn main() -> anyhow::Result<()> {
    let engine = RESNET.clone().into_engine()?;
    Ok(())
}
```

The macro uses `LazyLock` internally for non-const initialization.

## Trade-offs

| | Embedded | Filesystem |
|--|---------|-----------|
| Deploy | Single binary | Binary + model file |
| Binary size | +model size | Unchanged |
| Update model | Recompile | Replace file |
| Memory | Baked into BSS | Mapped from disk |

Embedded models make sense for small models (< 20 MB). For larger models, use [`airml-hub`](airml-hub.md) for on-demand download.

## See also

- [examples/03_embedded_model.rs](https://github.com/rlaope/airML/blob/master/examples/03_embedded_model.rs)
- [examples/04_embedded_compiletime.rs](https://github.com/rlaope/airML/blob/master/examples/04_embedded_compiletime.rs)
