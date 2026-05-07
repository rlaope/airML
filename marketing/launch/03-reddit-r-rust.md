# Reddit – r/rust post

## Title

[Project] airML 0.2 – Rust ONNX inference runtime for Apple Silicon (auto-tunes CoreML compute units)

---

## Body

### What and why

airML is a CLI + library for running ONNX models on Apple Silicon without
Python, Docker, or pip. You install one binary, pull a model, and run
inference. That is the entire surface area.

The motivation: every existing path to ONNX inference on macOS involves either
a Python runtime or substantial manual wiring. `ort` (the Rust binding) is
excellent but it is a library — it does not pick CoreML compute units for you,
does not fetch models, and is not a deploy target. candle is excellent for
CUDA/NVIDIA and for cases where you want to re-implement the model ops
yourself. airML's bet is that most production Apple Silicon inference should be
"hand me an ONNX file and go."

Anti-goals, explicitly stated:

- No CUDA / NVIDIA support. Use candle.
- No training. Use burn.
- No Python bindings. That is the point.
- The model registry stays small (~20 curated entries with verified sha256s).
  We are not building a model hub.

---

### Architecture

The workspace has seven crates:

| Crate | Responsibility |
|---|---|
| `airml-core` | `InferenceEngine`, `SessionConfig`, session lifecycle |
| `airml-providers` | CoreML, CPU, and future EP wrappers |
| `airml-preprocess` | ImageNet normalization, NLP tokenizer glue (feature-gated) |
| `airml-embed` | `embed_model!` macro — bake an ONNX file into your binary at compile time |
| `airml-tune` | `BackendOracle` — dispatch table for CoreML compute unit selection |
| `airml-hub` | Content-addressed model cache, HuggingFace download, sha256 verification |
| `airml-bench` | Criterion-based benchmark harness, structured JSON result schema |

The CLI (`src/`) wires them together. Each crate is independently usable as a
library dependency.

---

### Tech I am proud of

**Hand-rolled ONNX protobuf parser.** `airml-tune` needs to inspect the op
histogram of an ONNX graph to decide which compute units to recommend. Rather
than pull in a full protobuf library for what amounts to reading a header, I
wrote a minimal parser that extracts the op type strings with zero non-workspace
dependencies. It gets fuzzed nightly via `cargo-fuzz` against the parser
surface.

**BackendOracle dispatch table.** The auto-tuner maps model family signals
(Conv depth, presence of dynamic shapes, sequence ops, KV cache patterns) to
CoreML compute unit enums. The dispatch is a plain match arm table — readable,
testable, no magic. Text encoders with static shapes go to `ANEOnly`;
autoregressive LMs go to `CPUAndGPU` because the ANE does not handle that
control flow well at the op level.

**IoBinding groundwork.** The v0.3 roadmap targets KV-cache LLM generation.
The foundation is `IoBinding` integration in `airml-core` to eliminate the
per-call allocation that makes naive autoregressive inference slow. The
`airml generate` command currently exists as a stub; the real implementation
lands in v0.3.

---

### What I would love help with

- **LLM ONNX exports.** Getting a clean ONNX export from llama.cpp or
  transformers for a 1B INT4 model that round-trips through CoreML without op
  substitutions is harder than it sounds. If you have done this, I want to
  talk.
- **Benchmark numbers on M3 Pro / M3 Max.** The existing result files are M2
  Pro. The `bench/` directory has a schema and a `make bench` target. If you
  run it and PR the JSON I will add it to the comparison table.
- **More curated registry models.** The bar is: ONNX export verified,
  sha256-pinned, CoreML-compatible, tested on at least one M-series chip.
  Nominations in issues welcome.

---

### Links

- Source: https://github.com/rlaope/airML
- Docs: https://airml.github.io/airml
- Benchmarks schema: `bench/results/schema.json`
- Contributing guide: `CONTRIBUTING.md`
