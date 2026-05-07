# CLI Reference

airML is a single binary with subcommands for inference, model management, benchmarking, and serving.

## Global flags

These flags apply before any subcommand:

| Flag | Default | Description |
|------|---------|-------------|
| `--log-format <FORMAT>` | `text` | `text` (human-readable) or `json` (structured) |
| `--log-level <LEVEL>` | `info` | `trace`, `debug`, `info`, `warn`, or `error` |
| `--cache-dir <PATH>` | `~/.cache/airml` | Override the model cache directory |

`RUST_LOG` takes precedence over `--log-level`. Example:

```bash
RUST_LOG=airml_core=debug,airml_hub=info airml run -m bge-small-en --input "hi"
```

## Commands

| Command | Purpose |
|---------|---------|
| [`airml run`](run.md) | Run inference on an input |
| [`airml embed`](embed.md) | Generate text embeddings |
| [`airml info`](info.md) | Inspect model metadata |
| [`airml bench`](bench.md) | Measure inference latency |
| [`airml install-runtime`](install-runtime.md) | Auto-download ONNX Runtime dylib |
| [`airml pull`](pull.md) | Cache a model from registry, HuggingFace, or URL |
| [`airml generate`](generate.md) | LLM text generation (stub, v0.3) |
| [`airml serve`](serve.md) | OpenAI-compatible embeddings HTTP server |
| [`airml system`](system.md) | Print platform and provider info |

## Provider selection

All inference commands accept `--provider`:

| Value | Hardware |
|-------|----------|
| `auto` | Let `BackendOracle` choose (default) |
| `cpu` | CPU only |
| `coreml` | CoreML with all compute units |
| `neural-engine` | CoreML restricted to ANE + CPU |

On non-Apple platforms, `coreml` and `neural-engine` fall back to CPU.
