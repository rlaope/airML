# Deployment overview

airML ships as a single native binary with no runtime dependencies beyond the ONNX Runtime dylib. This makes it easy to deploy in containers, on bare metal, or as a serverless function.

## Deployment options

| Method | Best for | Guide |
|--------|---------|-------|
| Docker | Containerised workloads, Compose stacks | [Docker](docker.md) |
| systemd | Bare-metal Linux servers | [systemd](systemd.md) |
| AWS Lambda ARM | Serverless inference | [Lambda](lambda.md) |
| Homebrew | macOS developer machines | [Homebrew](homebrew.md) |

## Common requirements

All deployment methods require:

1. The `airml` binary
2. The ONNX Runtime shared library (`libonnxruntime.so` / `libonnxruntime.dylib`)
3. `ORT_DYLIB_PATH` environment variable pointing to the dylib
4. A model cache directory (default `~/.cache/airml`, override with `--cache-dir` or `AIRML_CACHE_DIR`)

## Production checklist

- Set `--log-format json` for structured logging
- Set `--auth-token` on `/v1/*` routes if exposing to the network
- Put a reverse proxy (nginx, Caddy) in front for TLS termination
- Mount the model cache as a persistent volume (Docker) or named directory (systemd)
- Scrape `/metrics` with Prometheus — see [Observability](../operations/observability.md)

## Full deployment guide

The canonical deployment documentation lives at [`docs/DEPLOYMENT.md`](https://github.com/rlaope/airML/blob/master/docs/DEPLOYMENT.md) in the repository.
