# Changelog

All notable changes to airML are documented here.

This project adheres to [Semantic Versioning](https://semver.org/) from v1.0 onwards.

## [Unreleased]

## [0.2.0] - TBD

### Added
- `airml install-runtime` — auto-downloads ONNX Runtime dylib
- `airml pull <model>` — fetches models from registry, HF, or URL with sha256 verification
- `airml generate` — token-by-token LLM inference with KV cache reuse
- `airml serve` — OpenAI-compatible HTTP embeddings API
- `airml-tune` crate — `BackendOracle` automatically picks CoreML compute units
- `airml-hub` crate — content-addressed model cache
- `airml-bench` crate — criterion-based benchmark harness
- ONNX graph parser (hand-rolled protobuf, zero dependencies)
- 4 runnable examples under `examples/`
- Docker, systemd, and Homebrew packaging templates
- Nightly fuzz target on the ONNX parser

### Changed
- `--provider auto` now uses `BackendOracle::recommend_for_path` instead of always picking CoreML-All
- Synthetic input in `airml bench` replaced with pseudo-Gaussian (CLT-based)
- Workspace dependencies pinned to `ort = "=2.0.0-rc.11"` (rc.12 has a VitisAI EP regression)

### Fixed
- `airml-embed::embed_model!` macro now uses `LazyLock` for non-const construction
- ImageNet preprocessing channel order corrected to RGB

## [0.1.0]

Initial release.
