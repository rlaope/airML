# airML Tutorial

This tutorial walks you through using airML for common ML tasks.

## Prerequisites

- Rust toolchain (1.70+)
- macOS (for CoreML support) or Linux/Windows (CPU only)

## Installation

```bash
# Clone and build
git clone https://github.com/airml/airml.git
cd airml

# Build with all features
cargo build --release --features coreml,nlp

# Add to PATH (optional)
export PATH="$PWD/target/release:$PATH"
```

## Tutorial 1: Image Classification

### Step 1: Get a Model

Download a pre-trained ResNet50 model:

```bash
# From ONNX Model Zoo
curl -L -o resnet50.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
```

### Step 2: Get Labels

```bash
curl -L -o imagenet_labels.txt \
  "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
```

### Step 3: Run Inference

```bash
# Basic usage
airml run -m resnet50.onnx -i cat.jpg -l imagenet_labels.txt

# Output:
# Top 5 predictions:
# --------------------------------------------------
#  281  95.23% ======================================== tabby
#  282   3.12% === tiger cat
#  285   0.89% = Egyptian cat
#  287   0.34%  lynx
#  283   0.21%  Persian cat
```

### Step 4: Try Different Providers

```bash
# CPU only
airml run -m resnet50.onnx -i cat.jpg -l imagenet_labels.txt -p cpu

# CoreML (macOS)
airml run -m resnet50.onnx -i cat.jpg -l imagenet_labels.txt -p coreml

# Neural Engine optimized (Apple Silicon)
airml run -m resnet50.onnx -i cat.jpg -l imagenet_labels.txt -p neural-engine
```

### Step 5: Benchmark Performance

```bash
# Compare CPU vs CoreML
airml bench -m resnet50.onnx -p cpu -n 100
airml bench -m resnet50.onnx -p coreml -n 100
airml bench -m resnet50.onnx -p neural-engine -n 100
```

## Tutorial 2: Text Embeddings

### Step 1: Get an Embedding Model

Download a sentence transformer model (e.g., all-MiniLM-L6-v2):

```bash
# Using Hugging Face optimum
pip install optimum[exporters]
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./minilm/
```

This creates:
- `minilm/model.onnx` - The model
- `minilm/tokenizer.json` - The tokenizer

### Step 2: Generate Embeddings

```bash
airml embed \
  -m minilm/model.onnx \
  -t minilm/tokenizer.json \
  --text "Hello, world!"

# Output:
# {
#   "text": "Hello, world!",
#   "dimension": 384,
#   "embedding": [
#     0.123456, 0.234567, ...
#   ]
# }
```

### Step 3: Normalize for Similarity Search

```bash
# L2 normalized embeddings (recommended for cosine similarity)
airml embed \
  -m minilm/model.onnx \
  -t minilm/tokenizer.json \
  --text "Hello, world!" \
  --normalize
```

### Step 4: Different Output Formats

```bash
# JSON format (default)
airml embed -m model.onnx -t tokenizer.json --text "Hello" --output json

# Raw format (one number per line)
airml embed -m model.onnx -t tokenizer.json --text "Hello" --output raw > embedding.txt
```

## Tutorial 3: Using airML as a Library

### Step 1: Add Dependencies

```toml
# Cargo.toml
[dependencies]
airml-core = { path = "crates/airml-core" }
airml-preprocess = { path = "crates/airml-preprocess" }
airml-providers = { path = "crates/airml-providers", features = ["coreml"] }
```

### Step 2: Image Classification in Code

```rust
use airml_core::{InferenceEngine, SessionConfig};
use airml_preprocess::ImagePreprocessor;
use airml_providers::{auto_select_providers, CoreMLProvider};

fn main() -> anyhow::Result<()> {
    // Configure with CoreML
    let providers = vec![CoreMLProvider::default().into_dispatch()];
    let config = SessionConfig::new().with_providers(providers);

    // Load model
    let mut engine = InferenceEngine::from_file_with_config("resnet50.onnx", config)?;

    // Preprocess image
    let preprocessor = ImagePreprocessor::imagenet();
    let input = preprocessor.load_and_process("cat.jpg")?;

    // Run inference
    let outputs = engine.run(input.into_dyn())?;

    // Get predictions
    let logits = &outputs[0];
    let softmax = softmax(logits);
    let top_k = top_k_indices(&softmax, 5);

    for (idx, prob) in top_k {
        println!("{}: {:.2}%", idx, prob * 100.0);
    }

    Ok(())
}

fn softmax(logits: &ndarray::ArrayD<f32>) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

fn top_k_indices(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<_> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).collect()
}
```

### Step 3: Text Embeddings in Code

```rust
use airml_core::{InferenceEngine, SessionConfig};
use airml_preprocess::TextPreprocessor;
use airml_providers::auto_select_providers;

fn main() -> anyhow::Result<()> {
    // Load tokenizer
    let preprocessor = TextPreprocessor::from_file("tokenizer.json")?
        .with_max_length(128);

    // Load model
    let config = SessionConfig::new().with_providers(auto_select_providers());
    let mut engine = InferenceEngine::from_file_with_config("model.onnx", config)?;

    // Tokenize
    let tokenized = preprocessor.encode("Hello, world!")?;
    let (input_ids, attention_mask) = tokenized.to_array();

    // Run inference
    let outputs = engine.run_multiple(vec![
        input_ids.into_dyn().mapv(|x| x as f32),
        attention_mask.into_dyn().mapv(|x| x as f32),
    ])?;

    // Extract embedding (assuming [batch, hidden] output)
    let embedding: Vec<f32> = outputs[0].iter().copied().collect();

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

    println!("Embedding dimension: {}", normalized.len());
    println!("First 5 values: {:?}", &normalized[..5]);

    Ok(())
}
```

## Tutorial 4: Embedding Models in Binary

### Step 1: Setup

```rust
use airml_embed::EmbeddedModel;

// Embed model at compile time
static MODEL_BYTES: &[u8] = include_bytes!("../models/resnet50.onnx");
```

### Step 2: Use Embedded Model

```rust
fn main() -> anyhow::Result<()> {
    let model = EmbeddedModel::new(MODEL_BYTES);
    println!("Model size: {} bytes", model.size());

    let engine = model.into_engine()?;

    // Use engine as normal...
    Ok(())
}
```

### Step 3: With Custom Configuration

```rust
use airml_core::SessionConfig;
use airml_providers::CoreMLProvider;

fn main() -> anyhow::Result<()> {
    let config = SessionConfig::new()
        .with_providers(vec![CoreMLProvider::default().into_dispatch()])
        .with_intra_threads(4);

    let engine = EmbeddedModel::with_config(MODEL_BYTES, config)
        .into_engine()?;

    Ok(())
}
```

## Tutorial 5: Benchmarking and Optimization

### Step 1: Basic Benchmark

```bash
airml bench -m model.onnx -n 100 -w 10
```

Output:
```
Results:
------------------------------------------------------------
  Total iterations: 100
  Total time:       1234.567 ms

  Mean latency:     12.346 ms
  Median latency:   12.100 ms
  Min latency:      11.200 ms
  Max latency:      15.800 ms
  Std deviation:    0.890 ms

  Throughput:       81.00 inferences/sec

  P50:              12.100 ms
  P90:              13.200 ms
  P95:              14.100 ms
  P99:              15.500 ms
```

### Step 2: Compare Providers

```bash
# Create a comparison script
for provider in cpu coreml neural-engine; do
    echo "=== $provider ==="
    airml bench -m model.onnx -p $provider -n 100
done
```

### Step 3: Optimize Thread Count

```rust
// Test different thread configurations
let configs = vec![
    SessionConfig::new().with_intra_threads(1),
    SessionConfig::new().with_intra_threads(2),
    SessionConfig::new().with_intra_threads(4),
    SessionConfig::new().with_intra_threads(8),
];

for config in configs {
    // Benchmark each configuration
}
```

## Common Issues

### Model Not Found

```
Error: Model not found: /path/to/model.onnx
```

Solution: Check the file path and ensure the model exists.

### Unsupported Operator

```
Error: Failed to load model: Unsupported operator: CustomOp
```

Solution: The model uses operators not supported by ONNX Runtime. Try:
1. Re-export the model with `opset_version=17`
2. Use a different model architecture

### CoreML Not Available

```
Warning: CoreML not available, falling back to CPU
```

Solution:
1. Ensure you're on macOS
2. Build with `--features coreml`
3. Check `airml system` for available providers

### Out of Memory

```
Error: Failed to allocate memory
```

Solution:
1. Reduce batch size
2. Use a smaller model
3. Close other applications

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for internal details
- Check [API.md](API.md) for full API reference
- See examples in `/examples` directory
