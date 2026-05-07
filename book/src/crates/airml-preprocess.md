# airml-preprocess

Image and text preprocessing for ONNX models. Provides typed presets for the most common normalization pipelines so you don't have to hand-roll tensor transforms.

## Image preprocessing

```rust
use airml_preprocess::ImagePreprocessor;

// ImageNet preset: 224x224, RGB, standard normalization
let preprocessor = ImagePreprocessor::imagenet();

// CLIP preset: 224x224, CLIP normalization
let preprocessor = ImagePreprocessor::clip();

// YOLO preset: 640x640, letterbox, no normalization
let preprocessor = ImagePreprocessor::yolo(640);

// Custom
let preprocessor = ImagePreprocessor::custom(
    224, 224,
    [0.485, 0.456, 0.406],  // mean
    [0.229, 0.224, 0.225],  // std
);

// Load and preprocess
let tensor: Array4<f32> = preprocessor.load_and_process("image.jpg")?;

// Preprocess an already-loaded image
let img = image::open("image.jpg")?;
let tensor = preprocessor.process(&img)?;
```

### Resize modes

```rust
pub enum ResizeMode {
    Stretch,    // Distort to fit target size
    Crop,       // Center crop to fit
    Letterbox,  // Pad with black to preserve aspect ratio (YOLO default)
}
```

## Text preprocessing (nlp feature)

```rust
#[cfg(feature = "nlp")]
use airml_preprocess::{TextPreprocessor, TokenizedInput};

let preprocessor = TextPreprocessor::from_file("tokenizer.json")?
    .with_max_length(512)
    .with_padding(true)
    .with_truncation(true);

let encoded: TokenizedInput = preprocessor.encode("Hello, world.")?;
let (input_ids, attention_mask) = encoded.to_array();

// Batch
let batch = preprocessor.encode_batch(&["Hello", "World"])?;
```

## Preset normalization values

| Preset | Mean | Std |
|--------|------|-----|
| ImageNet | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |
| CLIP | [0.481, 0.457, 0.408] | [0.268, 0.261, 0.275] |
| YOLO | none | none |

## See also

- [airml-core](airml-core.md) — inference engine that consumes preprocessed tensors
- [`airml run --preprocess`](../cli/run.md) — CLI preprocessing flag
