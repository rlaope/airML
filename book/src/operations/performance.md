# Performance

## Benchmark methodology

Reproducible benchmarks live under `crates/airml-bench/`. Run them with:

```bash
cargo bench -p airml-bench
```

Numbers below come from `cargo bench -p airml-bench` on M2 Pro / macOS 14. Inputs are pseudo-Gaussian synthetic tensors (CLT-based) to avoid artificially cacheable patterns.

## Latency table

| Provider | Model | P50 | P95 |
|----------|-------|-----|-----|
| CPU | ResNet50 | — | — |
| CoreML (All) | ResNet50 | — | — |
| Neural Engine | ResNet50 | — | — |

> Help wanted: run `cargo bench -p airml-bench` on your hardware and PR results to `bench/results/`.

## Cold start comparison

| Metric | airML | Python (PyTorch) |
|--------|-------|-----------------|
| Binary size | ~50 MB | ~2 GB |
| Cold start | 0.01–0.05s | 2–5s |
| Memory usage | ~100 MB | ~500 MB+ |

## Provider selection impact

Choosing the wrong CoreML compute units can cost 2-5x latency. The auto-tuner eliminates this guesswork — see [Auto-tuner overview](../apple-silicon/auto-tuner.md).

## Submitting benchmark results

```bash
cargo bench -p airml-bench 2>&1 \
  | tee bench/results/$(uname -m)-$(sw_vers -productVersion 2>/dev/null || uname -r).txt
git add bench/results/
git commit -m "bench: add results from <your hardware>"
```

Open a PR. All hardware contributions are welcome.

## See also

- [airml-bench crate](../crates/airml-bench.md)
- [`airml bench` CLI](../cli/bench.md)
