# `airml generate`

Autoregressive text generation from an ONNX language model with KV cache reuse. **This command is a stub in v0.2** — full implementation lands in v0.3.

## Status

| Version | Status |
|---------|--------|
| v0.2 | Stub — command exists, returns placeholder output |
| v0.3 | Real autoregressive generation, INT4/INT8 quantization passthrough |
| v0.3 target | ≥30 tok/s on M2 Pro for 1B INT4 model |

## Planned usage (v0.3)

```
airml generate --model <PATH> --prompt <TEXT> [OPTIONS]
```

## Planned options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <PATH>` | (required) | ONNX LLM path or registry id |
| `--prompt <TEXT>` | (required) | Input prompt |
| `--max-tokens <N>` | `256` | Maximum tokens to generate |
| `--temperature <F>` | `0.8` | Sampling temperature |
| `--top-p <F>` | `0.95` | Nucleus sampling threshold |
| `-p, --provider <P>` | `auto` | Execution provider (GPU recommended for LLMs) |

## Provider note

Language models with KV cache benefit from GPU rather than ANE — the auto-tuner sets `GPU only` for this model class. See [compute units guide](../apple-silicon/compute-units.md).

## Roadmap

See [ROADMAP.md v0.3](../roadmap.md) for the full milestone list including `IoBinding` integration and curated `llama-3.2-1b-int4` model.
