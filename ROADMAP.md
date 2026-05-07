# airML Roadmap

> Versions, not weeks. We ship when each milestone is done.

## v0.2 -- Foundations of the moat (current)

- [x] `airml-bench` -- criterion-based benchmark crate
- [x] `airml-tune` -- `BackendOracle` for automatic backend selection
- [x] `airml-hub` -- content-addressed model cache, HF download
- [x] `airml install-runtime` -- one-command ORT setup
- [x] `airml pull <model>` -- cache models by ID, URI, or URL
- [x] `examples/` -- 3 runnable examples
- [x] `--provider auto` rewired to `BackendOracle`
- [x] Realistic input distribution in `bench` (no more `(i*0.001) % 1.0`)
- [ ] First public benchmark numbers on M2/M3
- [ ] 20+ unit tests across all crates

## v0.3 -- LLM generation

- [ ] `IoBinding` integration in `airml-core` to eliminate per-call allocation
- [ ] KV-cache management module
- [ ] `airml generate` -- real autoregressive text generation
- [ ] INT4/INT8 quantization passthrough
- [ ] First curated LLM in registry: `llama-3.2-1b-int4`
- [ ] Target: >=30 tok/s on M2 Pro for 1B INT4

## v0.4 -- Real graph profiling

- [ ] ONNX graph parser inside `airml-tune`
- [ ] Op-histogram-based dispatch (not just family heuristics)
- [ ] Per-op latency profiling cache (informs future `recommend()` calls)
- [ ] `airml profile <model>` -- generate a per-op timing report

## v0.5 -- Deployment surface

- [ ] `airml serve` -- HTTP daemon with OpenAI-compatible embeddings API
- [ ] Linux ARM (Graviton/Ampere) first-class support
- [ ] Docker scratch image template
- [ ] systemd unit template

## v0.6 -- Public benchmark site

- [ ] `benchmarks.airml.dev` (CI-published static site)
- [ ] Comparison vs `ort` direct, `candle`, `tract`, `python+onnxruntime`, `mlx`
- [ ] "Where do you want to ship from?" cold-start vs P50 scatter plot
- [ ] Native Lambda runtime wrapper (`crates/airml-lambda`) — drop the AWS Lambda Adapter dependency, implement the Lambda Runtime API poll loop directly in Rust using the `lambda_runtime` crate

## v1.0 -- Stability

- [ ] Migrate to `ort` 2.0 stable
- [ ] Semver-stable public API (`airml-core`, `airml-embed`)
- [ ] `cargo public-api` enforcement in CI (gating; informational added in v0.2)
- [ ] Fuzz tests on ONNX parser surface (nightly fuzzing added in v0.2)

### Done in v0.2 (back-ported)

- [x] `cargo public-api` check in CI (informational only; gating deferred to v1.0)
- [x] Fuzz tests on ONNX parser surface (nightly job via `cargo-fuzz`; see `fuzz/`)

## Anti-goals (won't do)

See [README.md#anti-goals](README.md#anti-goals).
