# `airml embed`

Generate a text embedding vector from an ONNX encoder model. Requires the `nlp` feature at build time (`cargo build --features nlp`).

## Usage

```
airml embed --model <PATH> --tokenizer <PATH> --text <TEXT> [OPTIONS]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <PATH>` | (required) | ONNX embedding model path or registry id |
| `-t, --tokenizer <PATH>` | (required) | Path to `tokenizer.json` |
| `--text <TEXT>` | (required) | Text string to embed |
| `--max-length <N>` | `512` | Maximum token sequence length |
| `-p, --provider <P>` | `auto` | Execution provider |
| `--output <FORMAT>` | `json` | `json` or `raw` (space-separated floats) |
| `--normalize` | off | L2-normalize the output embedding |
| `--cache-dir <PATH>` | `~/.cache/airml` | Override cache directory |
| `-v, --verbose` | off | Show timing and provider details |

## Examples

```bash
# Embed a sentence and output JSON
airml embed -m bge-small-en -t tokenizer.json --text "Hello, world."

# Output raw floats for piping
airml embed -m bge-small-en -t tokenizer.json --text "Hello" --output raw

# L2-normalize for cosine similarity
airml embed -m bge-small-en -t tokenizer.json --text "cats" --normalize
```

## Output format

JSON output (default):

```json
{
  "embedding": [0.012, -0.034, 0.091, ...],
  "dimensions": 384,
  "model": "bge-small-en",
  "provider": "NeuralEngine",
  "latency_ms": 18.4
}
```

## Using the HTTP API instead

For batch embedding or production use, prefer [`airml serve`](serve.md) — it exposes an OpenAI-compatible `/v1/embeddings` endpoint.

## See also

- [`airml serve`](serve.md) — HTTP embeddings server
- [airml-preprocess](../crates/airml-preprocess.md) — tokenizer internals
