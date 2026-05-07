# airml-hub

Content-addressed model cache with HuggingFace and registry download support. Powers `airml pull` and provides a programmatic API for embedding model fetching in Rust applications.

## Usage

```rust
use airml_hub::{Hub, ModelId};

let hub = Hub::default();  // uses ~/.cache/airml

// Pull by registry id
let path = hub.pull("bge-small-en").await?;

// Pull from HuggingFace
let path = hub.pull("hf://Xenova/clip-vit-base-patch32").await?;

// Pull from URL
let path = hub.pull("https://example.com/model.onnx").await?;

// Check if cached
let cached = hub.is_cached("bge-small-en");
```

## Cache layout

Models are stored content-addressed by SHA256:

```
~/.cache/airml/
  bge-small-en.onnx       -> blobs/sha256-abc123...
  blobs/
    sha256-abc123...       (immutable blob)
```

Re-downloading the same content is a no-op. `--force` bypasses the cache check.

## Custom cache directory

```rust
let hub = Hub::with_cache_dir("/path/to/cache");
```

Or via environment variable:

```bash
export AIRML_CACHE_DIR=/path/to/cache
```

## Registry

The built-in registry (`src/registry.rs`) maps short ids to download URIs and expected SHA256 hashes. Community additions are welcome — see [Contributing](../contributing.md).

## See also

- [`airml pull`](../cli/pull.md) — CLI interface to hub
