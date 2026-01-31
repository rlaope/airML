# airML

A lightweight ML runtime that runs ONNX models without Python. Fast, portable, and efficient.

## Features

- **Single Binary**: Deploy ML models with a single ~50MB binary
- **Fast Cold Start**: 0.01-0.05s startup time (100x faster than Python)
- **Apple Silicon Acceleration**: Native CoreML/Metal/Neural Engine support
- **ONNX Support**: Run models exported from PyTorch, TensorFlow, and more
- **Zero Dependencies**: No Python, no virtual environments, no package managers
- **NLP Support**: Text tokenization and embedding generation

## Installation

### macOS (Apple Silicon) - Recommended

```bash
# 1. Download airml
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-aarch64.tar.gz | tar xz
sudo mv airml /usr/local/bin/

# 2. Download ONNX Runtime (required)
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-osx-arm64-1.23.1.tgz | tar xz -C /usr/local/lib

# 3. Set environment variable (add to ~/.zshrc for persistence)
export ORT_DYLIB_PATH=/usr/local/lib/onnxruntime-osx-arm64-1.23.1/lib/libonnxruntime.dylib
```

### macOS (Intel)

```bash
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-x86_64.tar.gz | tar xz
sudo mv airml /usr/local/bin/

curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-osx-x86_64-1.23.1.tgz | tar xz -C /usr/local/lib
export ORT_DYLIB_PATH=/usr/local/lib/onnxruntime-osx-x86_64-1.23.1/lib/libonnxruntime.dylib
```

### Linux (x86_64)

```bash
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-linux-x86_64.tar.gz | tar xz
sudo mv airml /usr/local/bin/

curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz | tar xz -C /usr/local/lib
export ORT_DYLIB_PATH=/usr/local/lib/onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so
```

### From Source

```bash
git clone https://github.com/rlaope/airML.git
cd airML
cargo build --release --features coreml,nlp
```

### Verify Installation

```bash
airml system
```

## Quick Start

### Image Classification

```bash
# Run classification on an image
airml run -m resnet50.onnx -i cat.jpg -l imagenet_labels.txt

# Output:
# Top 5 predictions:
# --------------------------------------------------
#  281  95.23% ======================================== tabby
#  282   3.12% === tiger cat
#  285   0.89% = Egyptian cat
```

### Text Embeddings

```bash
# Generate text embeddings
airml embed -m sentence-transformer.onnx -t tokenizer.json --text "Hello world"

# Output:
# {
#   "text": "Hello world",
#   "dimension": 384,
#   "embedding": [0.123456, 0.234567, ...]
# }
```

### Benchmarking

```bash
# Benchmark inference performance
airml bench -m model.onnx -n 100 -p neural-engine

# Output:
# Mean latency:     12.34 ms
# Throughput:       81.00 inferences/sec
```

### System Info

```bash
# Check available providers
airml system

# Output:
# OS: macos
# Architecture: aarch64
# Apple Silicon: true
# Available providers: cpu, coreml
```

## CLI Reference

### `airml run`

Run inference on an input image.

```bash
airml run --model <MODEL> --input <INPUT> [OPTIONS]

Options:
  -m, --model <MODEL>       Path to ONNX model file
  -i, --input <INPUT>       Path to input file (image)
  -l, --labels <LABELS>     Path to labels file
  -k, --top-k <N>           Top predictions to show [default: 5]
  -p, --provider <PROVIDER> Execution provider (auto, cpu, coreml, neural-engine)
      --preprocess <PRESET> Preprocessing (imagenet, clip, yolo, none)
      --raw                 Output raw tensor values
```

### `airml embed`

Generate text embeddings (requires `nlp` feature).

```bash
airml embed --model <MODEL> --tokenizer <TOKENIZER> --text <TEXT> [OPTIONS]

Options:
  -m, --model <MODEL>          ONNX embedding model
  -t, --tokenizer <TOKENIZER>  tokenizer.json file
      --text <TEXT>            Text to embed
      --max-length <N>         Max sequence length [default: 512]
  -p, --provider <PROVIDER>    Execution provider
      --output <FORMAT>        Output format (json, raw)
      --normalize              L2 normalize embeddings
```

### `airml info`

Display model information.

```bash
airml info --model <MODEL> [-v]
```

### `airml bench`

Benchmark inference performance.

```bash
airml bench --model <MODEL> [OPTIONS]

Options:
  -n, --iterations <N>     Iterations [default: 100]
  -w, --warmup <N>         Warmup iterations [default: 10]
  -p, --provider <PROVIDER> Execution provider
      --shape <SHAPE>      Input shape (e.g., "1,3,224,224")
```

### `airml system`

Display system capabilities.

## Execution Providers

| Provider | Platform | Hardware | Flag |
|----------|----------|----------|------|
| CPU | All | Any CPU | (default) |
| CoreML | macOS | Apple Silicon | `--features coreml` |
| Neural Engine | macOS | M1/M2/M3 ANE | `--features coreml` |

```bash
# Build with specific providers
cargo build --release                      # CPU only
cargo build --release --features coreml    # + CoreML
cargo build --release --features nlp       # + NLP
cargo build --release --features coreml,nlp # All features
```

## Performance

Benchmarked on Apple M2 with ResNet50:

| Provider | Latency | Throughput |
|----------|---------|------------|
| CPU | ~50ms | ~20 inf/s |
| CoreML (All) | ~15ms | ~65 inf/s |
| Neural Engine | ~8ms | ~125 inf/s |

| Metric | airML | Python (PyTorch) |
|--------|-------|------------------|
| Binary Size | ~50MB | ~2GB |
| Cold Start | 0.01-0.05s | 2-5s |
| Memory Usage | ~100MB | ~500MB+ |

## Using as a Library

```rust
use airml_core::{InferenceEngine, SessionConfig};
use airml_preprocess::ImagePreprocessor;
use airml_providers::CoreMLProvider;

fn main() -> anyhow::Result<()> {
    // Configure with CoreML
    let providers = vec![CoreMLProvider::default().neural_engine_only().into_dispatch()];
    let config = SessionConfig::new().with_providers(providers);

    // Load model
    let mut engine = InferenceEngine::from_file_with_config("model.onnx", config)?;

    // Preprocess and run
    let input = ImagePreprocessor::imagenet().load_and_process("image.jpg")?;
    let outputs = engine.run(input.into_dyn())?;

    Ok(())
}
```

## Embedding Models in Binary

```rust
use airml_embed::EmbeddedModel;

static MODEL: &[u8] = include_bytes!("model.onnx");

fn main() -> anyhow::Result<()> {
    let engine = EmbeddedModel::new(MODEL).into_engine()?;
    // Use engine...
    Ok(())
}
```

## Project Structure

```
airML/
├── crates/
│   ├── airml-core/        # Inference engine (ONNX Runtime wrapper)
│   ├── airml-preprocess/  # Image/text preprocessing
│   ├── airml-providers/   # Execution providers (CPU, CoreML)
│   └── airml-embed/       # Model embedding utilities
├── src/                   # CLI binary
│   ├── main.rs
│   ├── cli.rs             # Argument parsing
│   └── commands/          # Command implementations
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # Internal architecture
│   ├── TUTORIAL.md        # Step-by-step tutorials
│   └── API.md             # API reference
└── models/                # Test models (gitignored)
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Internal design and data flow
- [Tutorial](docs/TUTORIAL.md) - Step-by-step guides
- [API Reference](docs/API.md) - Complete API documentation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Maintainer

- [@rlaope](https://github.com/rlaope) - piyrw9754@gmail.com

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.
