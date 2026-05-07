# airml-providers

Execution provider configuration and auto-selection for airML (CPU, CoreML).

`airml-providers` makes it easy to pick the right ONNX Runtime execution provider for the
current hardware. On Apple Silicon it can automatically select CoreML; on all other platforms
it falls back to CPU. Provider configuration objects are passed to `airml-core`'s
`SessionConfig::with_providers()`.

## Quickstart

```rust
use airml_providers::{auto_select_providers, is_apple_silicon, system_info};
use airml_core::{InferenceEngine, SessionConfig};

fn main() -> anyhow::Result<()> {
    // Print hardware summary
    let info = system_info();
    println!("os={} arch={} apple_silicon={}", info.os, info.arch, info.is_apple_silicon);

    // Let the library pick the best provider automatically
    let providers = auto_select_providers();
    let config = SessionConfig::new().with_providers(providers);
    let engine = InferenceEngine::from_file_with_config("model.onnx", config)?;
    Ok(())
}
```

Explicit CoreML configuration (requires `--features coreml`):

```rust
#[cfg(feature = "coreml")]
{
    use airml_providers::{ComputeUnits, CoreMLConfig, CoreMLProvider};

    let config = CoreMLConfig {
        compute_units: ComputeUnits::CpuAndNeuralEngine,
        ..CoreMLConfig::default()
    };
    let provider = CoreMLProvider::with_config(config);
}
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `cpu` | yes | Always-available CPU execution provider |
| `coreml` | no | CoreML execution provider (macOS only) |

## Links

- [Main project README](../../README.md)
- [airML on GitHub](https://github.com/airml/airml)
- [API docs on docs.rs](https://docs.rs/airml-providers)
