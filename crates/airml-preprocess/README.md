# airml-preprocess

Image and text preprocessing utilities for airML ONNX inference pipelines.

`airml-preprocess` handles the data-preparation steps that sit between raw inputs and model
inference: resizing and normalizing images into `ndarray` tensors, and (with the `nlp` feature)
tokenizing text with HuggingFace-compatible tokenizers.

## Quickstart

### Image preprocessing

```rust
use airml_preprocess::{ImagePreprocessor, ResizeMode};

fn main() -> anyhow::Result<()> {
    let preprocessor = ImagePreprocessor::new(224, 224, ResizeMode::CenterCrop);

    // Load any image format supported by the `image` crate
    let img = image::open("cat.jpg")?;
    let tensor = preprocessor.process(&img)?;
    // tensor shape: [1, 3, 224, 224] — ready to pass to InferenceEngine::run()
    println!("tensor shape: {:?}", tensor.shape());
    Ok(())
}
```

### NLP tokenization (requires `nlp` feature)

```rust
#[cfg(feature = "nlp")]
{
    use airml_preprocess::TextPreprocessor;

    let preprocessor = TextPreprocessor::from_pretrained("tokenizer.json")?;
    let tokens = preprocessor.encode("Hello, airML!")?;
    println!("input_ids: {:?}", tokens.input_ids);
}
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `nlp` | no | Enables `TextPreprocessor` and `embed_texts` via the `tokenizers` crate |

## Links

- [Main project README](../../README.md)
- [airML on GitHub](https://github.com/airml/airml)
- [API docs on docs.rs](https://docs.rs/airml-preprocess)
