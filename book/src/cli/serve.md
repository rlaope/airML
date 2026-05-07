# `airml serve`

Start an OpenAI-compatible HTTP embeddings server. Any OpenAI client library works without modification. Requires the `nlp` feature: `cargo build --release --features nlp`.

## Usage

```
airml serve [OPTIONS]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--bind <HOST:PORT>` | `127.0.0.1:8080` | Bind address |
| `--default-model <ID>` | none | Model used when request omits the `model` field |
| `--auth-token <TOKEN>` | none | Require `Authorization: Bearer <TOKEN>` on `/v1/*` routes |
| `--max-request-bytes <N>` | `4194304` (4 MiB) | Request body size limit |
| `--cache-dir <PATH>` | `~/.cache/airml` | Override Hub cache directory |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/embeddings` | Generate embeddings (OpenAI-compatible) |
| `GET` | `/v1/models` | List registry models |
| `GET` | `/v1/embeddings/info?model=<id>` | Cache status for a model |
| `GET` | `/healthz` | Health check — always `{"status":"ok"}` |
| `GET` | `/metrics` | Prometheus metrics |

## Example

```bash
# Build with nlp feature
cargo build --release --features nlp

# Pull the model
airml pull bge-small-en

# Start the server
airml serve --bind 127.0.0.1:8080

# In another terminal
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-small-en","input":["Hello, world."]}' \
  | jq '.data[0].embedding[:5]'
```

## OpenAI client compatibility

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
response = client.embeddings.create(model="bge-small-en", input=["Hello"])
print(response.data[0].embedding[:5])
```

## Observability

Prometheus metrics are available at `/metrics`. See [Observability](../operations/observability.md) for the full metric list and a sample Grafana dashboard.

## Deployment

For production, run behind nginx or Caddy and use `--auth-token`. See:
- [Docker deployment](../deployment/docker.md)
- [systemd deployment](../deployment/systemd.md)
