# airml-hub

Content-addressed model download and caching for airML (HuggingFace, registry, URL).

`airml-hub` provides zero-friction model acquisition. Point it at a HuggingFace repo, a
built-in registry alias, or a direct URL and it downloads the model once, verifies the
SHA-256 digest, and stores it in a content-addressed on-disk cache
(`~/Library/Caches/airml/models` on macOS, `~/.cache/airml/models` on Linux). Subsequent
calls return the cached path without any network traffic.

## Quickstart

```rust
use airml_hub::{Fetcher, ModelUri};

fn main() -> anyhow::Result<()> {
    let fetcher = Fetcher::new();

    // Registry alias
    let uri = ModelUri::parse("bge-small-en")?;
    let path = fetcher.resolve_to_path(&uri)?;
    println!("model cached at {}", path.display());

    // HuggingFace repo
    let hf_uri = ModelUri::parse("hf://Xenova/clip-vit-base-patch32")?;
    let path2 = fetcher.resolve_to_path(&hf_uri)?;
    println!("clip model at {}", path2.display());

    // Direct URL with sha256 verification
    let url_uri = ModelUri::parse(
        "https://example.com/model.onnx?sha256=abc123"
    )?;
    let path3 = fetcher.resolve_to_path(&url_uri)?;
    println!("custom model at {}", path3.display());

    Ok(())
}
```

Inspecting the cache:

```rust
use airml_hub::ModelCache;

let cache = ModelCache::default();
for stat in cache.list()? {
    println!("{} — {} bytes", stat.name, stat.size_bytes);
}
```

## Feature flags

This crate has no optional Cargo features.

## Links

- [Main project README](../../README.md)
- [airML on GitHub](https://github.com/airml/airml)
- [API docs on docs.rs](https://docs.rs/airml-hub)
