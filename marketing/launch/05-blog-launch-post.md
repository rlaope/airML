# Blog – airml.dev/blog/launching-0.2

_Published: May 2026_

---

## Why we built airML

The honest answer is that I got tired of the install ceremony.

Every time I wanted to run an ONNX model on an M-series Mac in a production
context — a CLI tool, a server, a Lambda function — I ended up writing the same
scaffolding: create a venv, pin onnxruntime, figure out which CoreML execution
provider flags actually activate the Neural Engine, discover that the model
runs on CPU anyway, add logging, package into a Docker image that ends up
700 MB because Python.

The alternatives I looked at each solve part of the problem. candle is a full
Rust tensor library and it is excellent, but it requires you to re-implement
the model ops — you cannot hand it an existing ONNX file. The `ort` Rust crate
wraps ONNX Runtime cleanly, but it is a library, not a deploy target. tract
runs ONNX but has limited CoreML integration and no model fetch/cache layer.

The gap was: a binary that accepts any ONNX file, automatically picks the right
Apple Silicon execution path, caches models content-addressed, and exposes a
stable HTTP API. That is airML.

---

## What v0.2 ships

### 1. BackendOracle — automatic CoreML compute unit selection

The biggest piece of new infrastructure in v0.2 is `airml-tune` and its
`BackendOracle`. CoreML has four compute unit modes: `CPUOnly`,
`CPUAndNeuralEngine`, `CPUAndGPU`, and `All`. Picking the wrong one costs
anywhere from 20% to 10x in latency.

`BackendOracle::recommend_for_path` inspects the ONNX graph's op histogram and
maps it to a compute unit recommendation:

```
Conv-heavy (vision)          -> ANEOnly
Text encoder, static shapes  -> ANEOnly
Text encoder, dynamic shapes -> All (let CoreML decide per shape)
Autoregressive LM            -> CPUAndGPU (ANE struggles here)
Image+text dual encoder      -> All
```

You get this automatically with `--provider auto` (the default). Override with
`--provider neural-engine` or `--provider cpu` if you know better.

### 2. airml-hub — content-addressed model cache

`airml pull bge-small-en` downloads the model, verifies its sha256 against the
registry manifest, and stores it under `~/.cache/airml/` keyed by hash. Pull
once, run from any working directory. The registry currently covers five
models; `airml pull --list` shows what is available.

```bash
airml pull bge-small-en          # by registry ID
airml pull hf://BAAI/bge-small-en-v1.5/model.onnx   # by HF path
airml pull https://example.com/model.onnx            # by URL
```

### 3. airml install-runtime — one command ORT setup

ONNX Runtime's dylib is not bundled in the binary because of its size. v0.2
adds `airml install-runtime` which downloads the correct ORT release for your
platform and architecture, verifies the checksum, extracts it to
`~/.local/lib/airml/`, and sets `ORT_DYLIB_PATH` for the current shell. No
manual curl wrangling.

### 4. airml serve — OpenAI-compatible embeddings API

```bash
airml serve --bind 127.0.0.1:8080
```

Endpoints:

- `POST /v1/embeddings` — OpenAI-format embeddings (any OpenAI client works)
- `GET /v1/models` — list cached registry models
- `GET /healthz` — always `{"status":"ok"}`

Bearer token auth and CORS are both configurable via flags. The server is built
on Axum and uses Tokio's work-stealing runtime; it handles concurrent embedding
requests without spawning per-request ORT sessions.

---

## What is NOT in v0.2

**LLM generation.** `airml generate` exists in the CLI but is a stub. It will
print a warning and exit. Real autoregressive generation requires `IoBinding`
integration to eliminate per-call allocation and a KV cache management module.
Both land in v0.3.

**First benchmark numbers.** The bench harness (`crates/airml-bench/`) is
complete and the result schema is stable, but I do not have numbers from enough
chip variants to publish a comparison table. The ROADMAP.md v0.2 checklist
still has "First public benchmark numbers on M2/M3" unchecked. I would rather
ship no numbers than ship misleading ones.

**CUDA / NVIDIA.** This is a permanent anti-goal, not a deferral. Apple Silicon
inference is the entire scope. For NVIDIA use candle.

**Python bindings.** Also permanent. Removing Python from the inference path
is not a compromise position.

---

## Roadmap to 1.0

The full roadmap is in [ROADMAP.md](https://github.com/rlaope/airML/blob/master/ROADMAP.md).
Short version:

- **v0.3**: Real LLM generation — IoBinding, KV cache, `airml generate`,
  `llama-3.2-1b-int4` in registry. Target: 30+ tok/s on M2 Pro.
- **v0.4**: ONNX graph profiling — per-op histograms replace family heuristics
  in BackendOracle.
- **v0.5**: Linux ARM first-class, Docker scratch image, systemd template.
- **v0.6**: Public benchmark site at CI-published cadence.
- **v1.0**: Stable semver API, `ort` 2.0 stable (currently rc.11), fuzz
  coverage enforced in CI.

---

## How you can help

**Run the bench harness.** If you have an M3 Pro or M3 Max, the bench
directory has everything you need:

```bash
git clone https://github.com/rlaope/airML
cd airML
make -C bench bench MODEL=bge-small-en
```

PR the result JSON to `bench/results/`. That is the most useful thing right
now.

**File issues with failing models.** The issue template asks for model ID,
chip, and `airml system` output. That triple is enough to reproduce most
failures.

**Suggest curated registry models.** The bar is: ONNX export verified,
sha256-pinned, CoreML-compatible, tested on at least one M-series chip.
Nominations welcome in GitHub issues.

---

## Acknowledgments

`ort` by pykeio is the foundation — the Rust bindings to ONNX Runtime are
solid and the maintainers are responsive. candle by Hugging Face showed that
serious ML in Rust without Python is practical; airML is a different bet on the
same premise. Georgi Gerganov's work on llama.cpp established that small,
dependency-light binaries can be a serious production runtime — that aesthetic
is one we are explicitly trying to carry into the ONNX ecosystem.

The ONNX protobuf parser in `airml-tune` was written by reading the ONNX spec
directly. It is small, it is fuzzed, and it does exactly one thing.
