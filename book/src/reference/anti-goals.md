# Anti-goals

These are things airML explicitly does not do. Understanding what we won't build is as important as knowing what we will.

- **We don't compete with `candle` on CUDA.** If you need NVIDIA GPU inference, use `candle` or `ort` directly. airML's value is Apple Silicon and portable ONNX, not CUDA.

- **We don't train models.** Use `burn`, `candle`, or Python frameworks for training. airML is inference-only.

- **We don't ship a Python binding.** The whole point of airML is eliminating the Python runtime. A Python binding would contradict the project's core premise.

- **We don't host an iOS/Android SDK.** airML targets server and desktop CLI deployments. Mobile is out of scope.

- **We don't expand the registry beyond ~20 curated models.** Quality over quantity. Each registry model is tested, SHA256-pinned, and known to work with the auto-tuner. We are not a model hub.

- **We don't support Cloudflare Workers.** Workers run in V8 isolates without POSIX filesystem access. airML is a native binary that links against `libonnxruntime.so` — incompatible with that environment. See [Roadmap](../roadmap.md) for v0.6 tracking.

- **We don't support Windows.** macOS arm64, macOS x86_64, Linux arm64, and Linux x86_64 are the supported platforms. Windows support is not planned.

These constraints exist to keep the codebase small, the binary single, and the project focused. If you need any of the above, there are better tools for those jobs.
