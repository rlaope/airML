# Hacker News – Show HN post

## Title

<!-- PREFERRED -->
Show HN: airML – Run any ONNX model on Apple Silicon as a single binary

<!-- Alternates (both under 80 chars) -->
<!-- Show HN: airML 0.2 – ONNX inference with auto-tuned CoreML compute units -->
<!-- Show HN: airML – A 2.4 MB Rust binary for ONNX inference, no Python -->

---

## Body

airML is a single native binary that runs ONNX models on Apple Silicon using
CoreML and the Neural Engine. It exists because every other path to ONNX
inference on macOS either requires Python or forces you to rewrite your model
in a new framework.

The usual install path for ONNX Runtime on Python is: create a venv, pip
install onnxruntime, figure out which CoreML EP flags to pass, discover that
the default execution provider is CPU, and spend an afternoon reading
OrtSessionOptions docs. Docker images for inference services routinely land
above 3 GB. candle is excellent but it requires you to re-implement your model
ops in Rust. ort (the Rust binding) is a library — it is not a deploy target
and it does not handle model fetching, caching, or backend selection for you.
None of these paths give you a binary you can scp to a machine and run.

What airML does:

- Single binary install. `curl ... | tar xz && sudo mv airml /usr/local/bin/`
  or `cargo install`. No Python, no pip, no venv.
- ONNX models load directly. Pull by registry ID, HuggingFace path, or raw URL.
- `BackendOracle` auto-tunes CoreML compute units per model family: Conv-heavy
  vision models go to ANE only; dynamic-shape text encoders get All compute
  units so CoreML can decide per shape; autoregressive LMs go to GPU because
  ANE struggles with that control flow.
- Content-addressed cache under `~/.cache/airml/`. Pull once, run everywhere.
- OpenAI-compatible HTTP API via `airml serve` — any OpenAI client library
  works without modification.

What airML does not do:

- Not a training framework. Use burn or candle.
- No CUDA / NVIDIA. If you need that, use candle.
- No Python bindings. That is the point.
- The curated registry stays small (target: ~20 models). We are not building
  Hugging Face Hub.

Quick demo:

```bash
# install binary + runtime
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-aarch64.tar.gz | tar xz
sudo mv airml /usr/local/bin/
airml install-runtime

# pull a model and run a benchmark
airml pull bge-small-en
airml bench -m bge-small-en -n 100
```

Feedback I am looking for: if a model fails to load, open an issue with the
model ID, your chip (M1/M2/M3 and Pro/Max/Ultra), and the ORT version
`airml system` prints. Mismatches in that combination are the most common
failure path right now.

Honest limit worth calling out: airML pins `ort = "=2.0.0-rc.11"` because
rc.12 introduced a VitisAI EP regression. We will migrate to the stable 2.0
release as soon as it lands. The rc is stable in practice for CoreML and CPU
execution providers on macOS — but you should know it is a release candidate.

Source: https://github.com/rlaope/airML
Docs: https://airml.github.io/airml
