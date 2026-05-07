# Press release – airML 0.2

---

**FOR IMMEDIATE RELEASE**

**airML 0.2: Single-Binary ONNX Inference Runtime for Apple Silicon Ships with Automatic Neural Engine Dispatch**

*Open-source Rust project eliminates Python dependency from on-device ML inference on macOS*

---

**[Location] — May 2026** — airML, an open-source project, today released
version 0.2 of its ONNX inference runtime for Apple Silicon. The release
delivers a single native binary that runs any ONNX model on M-series Macs
using CoreML and the Apple Neural Engine, with no Python interpreter, no
Docker container, and no pip dependencies required.

The core addition in v0.2 is BackendOracle, an automatic dispatch system that
inspects an ONNX model's operator graph and selects the optimal CoreML compute
units without user configuration. Vision models with heavy convolution workloads
are routed to the Neural Engine; text encoders with dynamic input shapes receive
the All compute units mode so CoreML can adapt per inference call;
autoregressive language models are directed to GPU because the Neural Engine
does not handle that control flow efficiently at the operator level. The
dispatch decisions are encoded as a readable match table in the `airml-tune`
crate and are independently testable.

The release also includes `airml-hub`, a content-addressed model cache that
downloads models from the curated registry, HuggingFace, or any URL, verifies
sha256 checksums, and stores them under `~/.cache/airml/`. A companion command,
`airml install-runtime`, downloads and configures the correct ONNX Runtime
dylib for the user's platform in a single step, eliminating the manual library
setup that previously blocked new users.

For server-side deployments, `airml serve` starts an OpenAI-compatible
embeddings HTTP API on any bind address. Existing applications that use the
OpenAI client library for embeddings can point `OPENAI_BASE_URL` at a local
airML instance without code changes.

airML is explicit about what it does not do. It does not support NVIDIA GPU
inference — that use case is better served by candle. It does not include model
training or fine-tuning. It does not ship Python bindings. The model registry
is intentionally small, targeting approximately 20 curated entries with
verified checksums and confirmed CoreML compatibility.

The project currently pins ONNX Runtime at version 2.0.0-rc.11 while awaiting
the stable 2.0 release. The rc has been stable for CoreML and CPU execution
providers on macOS through the development period of v0.2.

airML 0.2 is available under the MIT license. The source is at
https://github.com/rlaope/airML. Documentation is at
https://airml.github.io/airml.

---

*airML is an independent open-source project. It is not affiliated with Apple,
Microsoft, or Hugging Face.*

*Media contact: file an issue at https://github.com/rlaope/airML/issues*
