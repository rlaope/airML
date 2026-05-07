# Reddit – r/LocalLLaMA post

## Title

airML – Run BGE/MiniLM embeddings (and small LLMs via KV cache) on Apple Silicon, no Python

---

## Body

If you run local embeddings or small LLMs on an M-series Mac, airML removes
the Python layer entirely. No venv, no `pip install sentence-transformers`, no
waiting for torch to import.

---

### Cold start is the first thing you notice

`sentence-transformers` takes roughly 4 seconds just to import on a warm
machine — before you have done any inference. That is PyTorch loading,
tokenizer initialization, and the ONNX runtime (if you are using the
optimized path) all stacking up. airML's cold start is under 50 ms because
there is no interpreter, no runtime environment, and no dynamic library
discovery chain. It is a single native binary.

---

### Embedding benchmarks (M2 Pro, preliminary — validate before relying on these)

These are early numbers. The bench harness is in the repo under `bench/`.
If you run it on your chip please open a PR with the JSON result.

| Model | Provider | Mean latency | Note |
|---|---|---|---|
| bge-small-en (133 MB) | Neural Engine | ~12 ms | preliminary, M2 Pro |
| all-minilm-l6-v2 (90 MB) | Neural Engine | — | not yet measured |
| whisper-tiny-encoder (80 MB) | Neural Engine | — | encoder only, no decoder |

The `BackendOracle` in `airml-tune` picks ANE for static-shape text encoders
automatically. You do not set anything.

---

### One-command embeddings server

```bash
airml install-runtime
airml pull bge-small-en
airml serve --bind 127.0.0.1:8080
```

The server exposes `POST /v1/embeddings` in the OpenAI format. LangChain,
LlamaIndex, or any OpenAI client library connects without any code changes.
Point `OPENAI_BASE_URL` at localhost and go.

---

### LLM generation — early stage, honest about limits

`airml generate` exists but is a stub in v0.2. Real autoregressive generation
with KV cache lands in v0.3. The roadmap target is 30+ tok/s on a 1B INT4
model on M2 Pro. The groundwork (IoBinding in `airml-core`, KV cache module)
is in-progress.

If you need full LLM generation today, llama.cpp or mlx-lm are the right tools.
airML's v0.3 goal is specifically ONNX-exported models — models you already have
as ONNX files and want to run without Python.

---

### What airML does not do

- No CUDA / NVIDIA. For NVIDIA inference use llama.cpp with CUDA build or
  oobabooga.
- No training or fine-tuning.
- No model hub beyond ~20 curated entries. We verify each model's sha256 and
  test it against CoreML before adding it to the registry.
- No Python bindings. The point is to remove Python from the inference path.

---

### Install

```bash
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-aarch64.tar.gz | tar xz
sudo mv airml /usr/local/bin/
airml install-runtime
airml system   # shows chip, ORT version, available providers
```

Source: https://github.com/rlaope/airML

If you run the bench harness on M3 Pro or M3 Max I would really appreciate the
result JSON — right now I only have M2 Pro numbers.
