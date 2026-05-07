# Quickstart

Get from zero to a working inference in under 60 seconds.

## Prerequisites

- macOS (Apple Silicon or Intel) or Linux (arm64 / x86_64)
- Rust 1.75+ (`rustup` recommended)

## Install

```bash
# From crates.io (once published)
cargo install airml

# Or directly from GitHub
cargo install --git https://github.com/rlaope/airML
```

## Install ONNX Runtime

airML needs the ONNX Runtime shared library. One command handles it:

```bash
airml install-runtime
```

This downloads the correct pre-built dylib for your platform and prints the path to set `ORT_DYLIB_PATH`. Add that export to your shell profile.

## Pull a model and run inference

```bash
# Pull a small text-embedding model (~133 MB)
airml pull bge-small-en

# Run inference
airml run -m bge-small-en --input "Hello, world."
```

## Verify the platform

```bash
airml system
```

This reports your OS, CPU architecture, Apple Silicon detection, and which execution providers (CPU, CoreML, Neural Engine) are available.

## Next steps

- [Installation](installation.md) — all install methods including pre-built binaries
- [Your first inference](first-inference.md) — image classification walkthrough
- [CLI Reference](../cli/overview.md) — all commands and flags
