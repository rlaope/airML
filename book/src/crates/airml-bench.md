# airml-bench

Criterion-based benchmark harness for measuring airML inference performance in CI. Complements the CLI `airml bench` command with reproducible, statistically sound benchmarks that track regressions across commits.

## Running benchmarks

```bash
cargo bench -p airml-bench
```

Results are written to `target/criterion/`. Criterion generates HTML reports at `target/criterion/report/index.html`.

## Contributing benchmark results

Run on your hardware and PR the results to `bench/results/`:

```bash
cargo bench -p airml-bench 2>&1 | tee bench/results/$(uname -m)-$(date +%Y%m%d).txt
```

See `bench/Makefile` for the full benchmark matrix and `bench/run-all.sh` for the automation script.

## Input data

The benchmark harness generates pseudo-Gaussian synthetic inputs using a CLT-based approximation. This avoids deterministic all-zeros inputs that cache artificially well and produce misleadingly low latency numbers.

## Performance numbers

Reproducible numbers from M2 Pro / macOS 14:

| Provider | Model | Latency | Throughput |
|----------|-------|---------|-----------|
| CPU | ResNet50 | — | — |
| CoreML (All) | ResNet50 | — | — |
| Neural Engine | ResNet50 | — | — |

> Help wanted: run `cargo bench -p airml-bench` on your hardware and PR results.

## See also

- [`airml bench` CLI](../cli/bench.md)
- [Performance](../operations/performance.md)
