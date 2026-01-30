# airML

A lightweight ML runtime that runs ONNX models without Python. Fast, portable, and efficient.

## Features

- **Single Binary**: Deploy ML models with a single ~50MB binary
- **Fast Cold Start**: 0.01-0.05s startup time (100x faster than Python)
- **Apple Silicon Acceleration**: Native Metal/CoreML support for M-series chips
- **ONNX Support**: Run models exported from PyTorch, TensorFlow, and more
- **Zero Dependencies**: No Python, no virtual environments, no package managers

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/airml/airml.git
cd airml

# Build release binary
cargo build --release

# Optional: Install to PATH
cargo install --path .
```

### Pre-built Binaries

Download from [Releases](https://github.com/airml/airml/releases).

## Quick Start

```bash
# Run image classification
airml run --model resnet50.onnx --input cat.jpg --labels imagenet_labels.txt

# Display model information
airml info --model resnet50.onnx

# Benchmark inference performance
airml bench --model resnet50.onnx -n 100

# Check system capabilities
airml system
```

## CLI Reference

### `airml run`

Run inference on an input.

```bash
airml run --model <MODEL> --input <INPUT> [OPTIONS]

Options:
  -m, --model <MODEL>       Path to ONNX model file
  -i, --input <INPUT>       Path to input file (image)
  -l, --labels <LABELS>     Path to labels file (one label per line)
  -k, --top-k <N>           Number of top predictions to show [default: 5]
  -p, --provider <PROVIDER> Execution provider (auto, cpu, coreml) [default: auto]
      --preprocess <PRESET> Preprocessing preset (imagenet, clip, yolo, none) [default: imagenet]
      --raw                 Output raw tensor values
```

### `airml info`

Display model information.

```bash
airml info --model <MODEL> [OPTIONS]

Options:
  -m, --model <MODEL>  Path to ONNX model file
  -v, --verbose        Show detailed information
```

### `airml bench`

Benchmark inference performance.

```bash
airml bench --model <MODEL> [OPTIONS]

Options:
  -m, --model <MODEL>      Path to ONNX model file
  -n, --iterations <N>     Number of iterations [default: 100]
  -w, --warmup <N>         Warmup iterations [default: 10]
  -p, --provider <PROVIDER> Execution provider [default: auto]
      --shape <SHAPE>      Input shape (e.g., "1,3,224,224")
```

### `airml system`

Display system information and available providers.

```bash
airml system
```

## Execution Providers

| Provider | Platform | Hardware |
|----------|----------|----------|
| CPU | All | Any CPU |
| CoreML | macOS | Apple Silicon (M1/M2/M3) |

Enable providers with feature flags:

```bash
# CPU only (default)
cargo build --release

# With CoreML support
cargo build --release --features coreml
```

## Embedding Models

Embed ONNX models directly in your binary:

```rust
use airml_embed::EmbeddedModel;

// Embed at compile time
static MODEL_BYTES: &[u8] = include_bytes!("../models/resnet50.onnx");

fn main() -> anyhow::Result<()> {
    let model = EmbeddedModel::new(MODEL_BYTES);
    let engine = model.into_engine()?;

    // Run inference...
    Ok(())
}
```

## Performance

| Metric | airML | Python (PyTorch) |
|--------|-------|------------------|
| Binary Size | ~50MB | ~2GB |
| Cold Start | 0.01-0.05s | 2-5s |
| Memory Usage | ~100MB | ~500MB+ |

## Project Structure

```
airML/
├── crates/
│   ├── airml-core/        # Inference engine
│   ├── airml-preprocess/  # Image/text preprocessing
│   ├── airml-providers/   # Execution providers (CPU, CoreML)
│   └── airml-embed/       # Model embedding utilities
├── src/                   # CLI binary
└── models/                # Test models (gitignored)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.
