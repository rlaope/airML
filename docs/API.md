# airML API Reference

Complete API documentation for airML crates.

## airml-core

### InferenceEngine

The main interface for loading and running ONNX models.

```rust
use airml_core::{InferenceEngine, SessionConfig};
```

#### Constructors

```rust
// Load from file with default config
let engine = InferenceEngine::from_file("model.onnx")?;

// Load from file with custom config
let config = SessionConfig::new().with_intra_threads(4);
let engine = InferenceEngine::from_file_with_config("model.onnx", config)?;

// Load from bytes (for embedded models)
let engine = InferenceEngine::from_bytes(model_bytes)?;

// Load from bytes with custom config
let engine = InferenceEngine::from_bytes_with_config(model_bytes, config)?;
```

#### Methods

```rust
// Run inference with single input
pub fn run(&mut self, input: ArrayD<f32>) -> Result<Vec<ArrayD<f32>>>

// Run inference with multiple inputs (matched by order)
pub fn run_multiple(&mut self, inputs: Vec<ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>>

// Run inference with named inputs
pub fn run_named(&mut self, inputs: Vec<(&str, ArrayD<f32>)>) -> Result<Vec<ArrayD<f32>>>

// Get model metadata
pub fn metadata(&self) -> &ModelMetadata

// Get input tensor info
pub fn inputs(&self) -> &[TensorInfo]

// Get output tensor info
pub fn outputs(&self) -> &[TensorInfo]
```

### SessionConfig

Configuration for ONNX Runtime sessions.

```rust
use airml_core::SessionConfig;
```

#### Builder Methods

```rust
let config = SessionConfig::new()
    .with_intra_threads(4)          // Threads within operators
    .with_inter_threads(2)          // Threads between operators
    .with_optimization_level(level) // Graph optimization
    .with_providers(providers);     // Execution providers
```

### ModelMetadata

Information about a loaded model.

```rust
pub struct ModelMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub version: Option<i64>,
    pub producer: Option<String>,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
}
```

### TensorInfo

Information about model inputs/outputs.

```rust
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<i64>,  // -1 for dynamic dimensions
    pub dtype: String,
}
```

### AirMLError

Error types for the core module.

```rust
pub enum AirMLError {
    ModelNotFound(String),
    ModelLoadError(String),
    InferenceError(String),
    PreprocessError(String),
    ConfigError(String),
    OrtError(String),
}
```

---

## airml-preprocess

### ImagePreprocessor

Image preprocessing for vision models.

```rust
use airml_preprocess::{ImagePreprocessor, ResizeMode};
```

#### Presets

```rust
// ImageNet preset (224x224, standard normalization)
let preprocessor = ImagePreprocessor::imagenet();

// CLIP preset (224x224, CLIP normalization)
let preprocessor = ImagePreprocessor::clip();

// YOLO preset (640x640, no normalization, letterbox)
let preprocessor = ImagePreprocessor::yolo(640);

// Custom preset
let preprocessor = ImagePreprocessor::custom(
    width,       // u32
    height,      // u32
    mean,        // [f32; 3]
    std,         // [f32; 3]
);
```

#### Methods

```rust
// Load image and preprocess
pub fn load_and_process<P: AsRef<Path>>(&self, path: P) -> Result<Array4<f32>>

// Preprocess already loaded image
pub fn process(&self, image: &DynamicImage) -> Result<Array4<f32>>
```

#### Fields

```rust
pub struct ImagePreprocessor {
    pub width: u32,
    pub height: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub resize_mode: ResizeMode,
}
```

### ResizeMode

How to resize images to target dimensions.

```rust
pub enum ResizeMode {
    Stretch,    // Stretch to fit (may distort)
    Crop,       // Center crop to fit
    Letterbox,  // Pad to fit (preserve aspect ratio)
}
```

### TextPreprocessor (NLP feature)

Text preprocessing with tokenization.

```rust
#[cfg(feature = "nlp")]
use airml_preprocess::{TextPreprocessor, TokenizedInput, TextPreprocessError};
```

#### Constructors

```rust
// Load from tokenizer.json file
let preprocessor = TextPreprocessor::from_file("tokenizer.json")?;

// Load from bytes
let preprocessor = TextPreprocessor::from_bytes(tokenizer_bytes)?;
```

#### Builder Methods

```rust
let preprocessor = TextPreprocessor::from_file("tokenizer.json")?
    .with_max_length(512)       // Maximum sequence length
    .with_padding(true)         // Pad to max_length
    .with_truncation(true);     // Truncate if too long
```

#### Methods

```rust
// Encode single text
pub fn encode(&self, text: &str) -> Result<TokenizedInput>

// Encode batch of texts
pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<TokenizedInput>>
```

### TokenizedInput

Result of tokenization.

```rust
pub struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
}

impl TokenizedInput {
    // Convert to ndarray for model input
    pub fn to_array(&self) -> (Array2<i64>, Array2<i64>)
}
```

### TextPreprocessError

Errors from text preprocessing.

```rust
pub enum TextPreprocessError {
    LoadError(String),
    EncodeError(String),
    TextTooLong { actual: usize, max: usize },
}
```

---

## airml-providers

### CpuProvider

CPU execution provider (always available).

```rust
use airml_providers::CpuProvider;

let provider = CpuProvider::default().into_dispatch();
```

### CoreMLProvider (coreml feature)

CoreML execution provider for macOS.

```rust
#[cfg(feature = "coreml")]
use airml_providers::{CoreMLProvider, ComputeUnits, CoreMLConfig};
```

#### Constructors

```rust
// Default (use all compute units)
let provider = CoreMLProvider::new();
let provider = CoreMLProvider::default();

// With custom config
let config = CoreMLConfig { ... };
let provider = CoreMLProvider::with_config(config);
```

#### Builder Methods

```rust
let provider = CoreMLProvider::default()
    .with_compute_units(ComputeUnits::CpuAndNeuralEngine)
    .with_subgraphs(true)           // Enable for control flow models
    .with_static_shapes(false)      // Require static input shapes
    .with_model_format(format)      // NeuralNetwork or MLProgram
    .with_cache_dir("/path/to/cache");
```

#### Convenience Methods

```rust
// Optimize for Neural Engine
let provider = CoreMLProvider::default().neural_engine_only();

// Use GPU only (no ANE)
let provider = CoreMLProvider::default().gpu_only();

// Use CPU only
let provider = CoreMLProvider::default().cpu_only();
```

#### Conversion

```rust
// Convert to ExecutionProviderDispatch for use with SessionConfig
let dispatch = provider.into_dispatch();
```

### ComputeUnits

Hardware targets for CoreML.

```rust
pub enum ComputeUnits {
    All,                  // CPU + GPU + Neural Engine
    CpuAndNeuralEngine,   // CPU + Neural Engine (no GPU)
    CpuAndGpu,            // CPU + GPU (no ANE)
    CpuOnly,              // CPU only
}
```

### CoreMLConfig

Full configuration for CoreML provider.

```rust
pub struct CoreMLConfig {
    pub compute_units: ComputeUnits,
    pub enable_subgraphs: bool,
    pub require_static_shapes: bool,
    pub model_format: Option<CoreMLModelFormat>,
    pub cache_dir: Option<String>,
}
```

### CoreMLModelFormat

Model format for CoreML.

```rust
pub enum CoreMLModelFormat {
    NeuralNetwork,  // Better compatibility with older macOS/iOS
    MLProgram,      // More operators, potentially better performance
}
```

### Utility Functions

```rust
// Check if running on Apple Silicon
pub fn is_apple_silicon() -> bool

// Auto-select best available providers
pub fn auto_select_providers() -> Vec<ExecutionProviderDispatch>

// List available provider names
pub fn available_providers() -> Vec<String>

// Get system information
pub fn system_info() -> SystemInfo
```

### SystemInfo

System capability information.

```rust
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub is_apple_silicon: bool,
    pub available_providers: Vec<String>,
}
```

---

## airml-embed

### EmbeddedModel

Wrapper for models embedded in binaries.

```rust
use airml_embed::EmbeddedModel;

// Embed at compile time
static MODEL: &[u8] = include_bytes!("model.onnx");
```

#### Constructors

```rust
// From bytes
let model = EmbeddedModel::new(MODEL);

// With custom config
let model = EmbeddedModel::with_config(MODEL, config);
```

#### Methods

```rust
// Get model bytes
pub fn bytes(&self) -> &[u8]

// Get model size
pub fn size(&self) -> usize

// Set configuration
pub fn config(self, config: SessionConfig) -> Self

// Convert to InferenceEngine
pub fn into_engine(self) -> Result<InferenceEngine>
```

### embed_model! Macro

Macro for embedding models.

```rust
use airml_embed::embed_model;

// Creates static EmbeddedModel
embed_model!(RESNET, "../models/resnet50.onnx");

fn main() {
    let engine = RESNET.clone().into_engine().unwrap();
}
```

---

## CLI Commands

### airml run

```
airml run [OPTIONS] --model <MODEL> --input <INPUT>

Options:
  -m, --model <MODEL>         ONNX model path
  -i, --input <INPUT>         Input image path
  -l, --labels <LABELS>       Labels file (one per line)
  -k, --top-k <K>             Top K predictions [default: 5]
  -p, --provider <PROVIDER>   Execution provider [default: auto]
      --preprocess <PRESET>   Preprocessing preset [default: imagenet]
      --raw                   Output raw tensors
  -v, --verbose               Verbose output
```

### airml info

```
airml info [OPTIONS] --model <MODEL>

Options:
  -m, --model <MODEL>   ONNX model path
  -v, --verbose         Detailed information
```

### airml bench

```
airml bench [OPTIONS] --model <MODEL>

Options:
  -m, --model <MODEL>        ONNX model path
  -n, --iterations <N>       Benchmark iterations [default: 100]
  -w, --warmup <N>           Warmup iterations [default: 10]
  -p, --provider <PROVIDER>  Execution provider [default: auto]
      --shape <SHAPE>        Input shape (e.g., "1,3,224,224")
```

### airml embed

```
airml embed [OPTIONS] --model <MODEL> --tokenizer <TOKENIZER> --text <TEXT>

Options:
  -m, --model <MODEL>          ONNX embedding model path
  -t, --tokenizer <TOKENIZER>  tokenizer.json path
      --text <TEXT>            Text to embed
      --max-length <N>         Max sequence length [default: 512]
  -p, --provider <PROVIDER>    Execution provider [default: auto]
      --output <FORMAT>        Output format (json, raw) [default: json]
      --normalize              L2 normalize embeddings
  -v, --verbose                Verbose output
```

### airml system

```
airml system

Displays:
  - Operating system
  - CPU architecture
  - Apple Silicon detection
  - Available execution providers
```
