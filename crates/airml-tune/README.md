# airml-tune

Automatic backend dispatcher and CoreML compute-unit oracle for airML.

`airml-tune` analyses an ONNX model's operator histogram — either from metadata heuristics or
by parsing the actual protobuf graph — and recommends the optimal CoreML compute units
(`All`, `CpuOnly`, `CpuAndGpu`, `CpuAndNeuralEngine`) without requiring a benchmark run first.
A persistent `ProfileCache` stores past decisions so repeated launches are instant.

## Quickstart

```rust
use airml_tune::oracle::{BackendOracle, BackendRecommendation};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let oracle = BackendOracle::new();

    // Fast path: parse the ONNX graph for an accurate op histogram
    let rec = oracle.recommend_for_path(Path::new("model.onnx"))?;
    println!("recommended: {:?}", rec);

    Ok(())
}
```

Dispatching to CoreML automatically (requires `--features coreml`):

```rust
#[cfg(feature = "coreml")]
{
    use airml_tune::dispatch;
    use airml_core::SessionConfig;

    let config = dispatch::build_config_for_path(Path::new("model.onnx"))?;
    let engine = airml_core::InferenceEngine::from_file_with_config("model.onnx", config)?;
}
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `coreml` | no | Enables `dispatch` module; applies CoreML compute units recommended by the oracle |

## Key types

| Type | Description |
|------|-------------|
| `BackendOracle` | Recommends compute units from model op histogram |
| `BackendRecommendation` | Enum of `CpuOnly`, `CpuAndGpu`, `CpuAndNeuralEngine`, `All` |
| `OpHistogram` | Per-op-type counts parsed from an ONNX graph |
| `ProfileCache` | Disk-backed cache of past recommendations |

## Links

- [Main project README](../../README.md)
- [airML on GitHub](https://github.com/airml/airml)
- [API docs on docs.rs](https://docs.rs/airml-tune)
