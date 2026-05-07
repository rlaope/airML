# Glossary

**ANE** — Apple Neural Engine. The dedicated machine-learning accelerator built into Apple Silicon chips (M1 and later). Provides high throughput for matrix operations at very low power. airML targets the ANE via CoreML's `CpuAndNeuralEngine` or `All` compute units.

**BackendOracle** — The decision engine in `airml-tune` that analyzes an ONNX model's op-histogram and recommends the optimal CoreML `ComputeUnits` setting. Replaces manual provider selection for most models.

**BoundSession** — An ONNX Runtime session with pre-allocated I/O buffers via `IoBinding`. Eliminates per-call tensor allocation. Planned for `airml-core` in v0.3.

**ComputeUnits** — The CoreML enum that controls which hardware executes a model: `All` (CPU + GPU + ANE), `CpuAndNeuralEngine`, `CpuAndGpu`, or `CpuOnly`. Choosing the wrong value can cost 2–5× latency.

**EmbeddedModel** — An ONNX model baked into a Rust binary via `include_bytes!`. Provided by the `airml-embed` crate. Makes deployment a single-file copy with no external model assets.

**Fetcher** — The download subsystem inside `airml-hub`. Resolves model URIs (registry id, `hf://` URI, or direct URL) to a local content-addressed path, verifying SHA256 on arrival.

**IoBinding** — An ONNX Runtime API for pinning input and output tensors to specific memory locations before running a session. Eliminates per-inference allocation overhead. Important for high-throughput serving and for LLM KV-cache reuse.

**ModelFamily** — The classification `BackendOracle` assigns to a model based on its op-histogram: vision (Conv-heavy), text-encoder-static, text-encoder-dynamic, dual-encoder, or language-model. Each family maps to a recommended `ComputeUnits`.

**OpHistogram** — A frequency map of ONNX operator types in a model graph (e.g., `{Conv: 53, Relu: 52, Gemm: 1}`). Computed by `airml-tune`'s hand-rolled protobuf parser. The histogram is the primary signal for `BackendOracle` classification.

**ORT** — ONNX Runtime. The inference engine from Microsoft that airML wraps. airML uses the Rust bindings crate `ort`. The current pinned version is `2.0.0-rc.11`.

**Registry** — The curated set of ~20 models built into `airml-hub`. Each entry maps a short id (e.g., `bge-small-en`) to a download URI and expected SHA256. Designed to stay small and well-tested rather than grow into a general model hub.
