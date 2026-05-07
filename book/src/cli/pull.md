# `airml pull`

Download and cache a model from the airML registry, a HuggingFace repository, or a direct URL. Models are stored content-addressed in `~/.cache/airml/` with SHA256 verification.

## Usage

```
airml pull <MODEL> [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Registry id (`bge-small-en`), `hf://owner/repo[/file]` URI, or direct HTTPS URL |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--list` | off | List all available registry models instead of pulling |
| `--cache-dir <PATH>` | `~/.cache/airml` | Override cache directory |
| `--force` | off | Re-download even if already cached |

## Examples

```bash
# Pull from registry by id
airml pull bge-small-en

# Pull from HuggingFace
airml pull hf://Xenova/clip-vit-base-patch32

# Pull a specific file from HuggingFace
airml pull hf://Xenova/whisper-tiny/onnx/encoder_model.onnx

# Pull from a direct URL
airml pull https://example.com/my-model.onnx

# List all registry models
airml pull --list
```

## Registry models

| ID | Source | Use case | Size |
|----|--------|----------|------|
| `bge-small-en` | BAAI | Text embedding | 133 MB |
| `all-minilm-l6-v2` | sentence-transformers | Text embedding | 90 MB |
| `clip-vit-b32` | Xenova/CLIP | Image+text | 605 MB |
| `mobilenetv3-small` | onnx/models | Image classification | 14 MB |
| `whisper-tiny-encoder` | Xenova/Whisper | Audio encoder | 80 MB |

Run `airml pull --list` for the current full registry.

## Cache layout

```
~/.cache/airml/
  bge-small-en.onnx          (symlink to content-addressed blob)
  blobs/
    sha256-abc123...          (actual file)
```

## See also

- [`airml run`](run.md) — run a cached model
- [airml-hub crate](../crates/airml-hub.md) — programmatic cache API
