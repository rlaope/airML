# Docker

airML ships an official Docker image at `airml/airml`. Both `linux/amd64` and `linux/arm64` (Graviton/Ampere) are supported.

## One-off inference

```bash
docker run --rm \
  -v "$HOME/.cache/airml:/root/.cache/airml" \
  airml/airml:0.2 \
  run -m bge-small-en --input "Hello, world."
```

## HTTP inference server

```bash
docker run -d \
  --name airml \
  -p 8080:8080 \
  -v airml-cache:/root/.cache/airml \
  airml/airml:0.2 \
  serve --bind 0.0.0.0:8080
```

## Docker Compose (with nginx)

The `docker/docker-compose.yml` starts airML behind nginx on port 80:

```bash
docker compose -f docker/docker-compose.yml up -d

# Check logs
docker compose -f docker/docker-compose.yml logs -f airml
```

nginx proxies `http://localhost/` to `airml:8080`, passes the `Authorization` header through, and allows up to 10 MB request bodies.

## Building from source

```bash
# Current architecture
docker build -t airml/airml:local .

# Multi-arch (push to registry)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t airml/airml:0.2 \
  --push .
```

## Scratch image

A minimal scratch-based Dockerfile lives at `docker/scratch.Dockerfile`. It produces the smallest possible image by copying only the binary and dylib.

## Environment variables

| Variable | Description |
|----------|-------------|
| `ORT_DYLIB_PATH` | Path to `libonnxruntime.so` inside the container |
| `AIRML_CACHE_DIR` | Model cache directory (mount as a volume) |
| `RUST_LOG` | Log level/filter |

## See also

- [Deployment overview](overview.md)
- [Observability](../operations/observability.md)
