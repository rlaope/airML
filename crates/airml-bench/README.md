# airml-bench

Benchmarking harness for airML inference. Provides realistic synthetic inputs,
cold-start measurement, percentile statistics, and Criterion integration.

## Running benchmarks

```sh
# CPU only (always available)
AIRML_BENCH_MODEL=models/mobilenetv3.onnx cargo bench -p airml-bench

# CoreML (macOS only, requires coreml feature)
AIRML_BENCH_MODEL=models/mobilenetv3.onnx cargo bench -p airml-bench \
    --features airml-providers/coreml
```

Criterion generates HTML reports automatically under `target/criterion/`.
Open `target/criterion/report/index.html` in a browser after the run.

## JSON export

Criterion stores raw measurement data as JSON at:

```
target/criterion/<benchmark-name>/new/estimates.json
target/criterion/<benchmark-name>/new/sample.json
```

## Programmatic use

```rust
use airml_bench::{BenchProvider, run_full_benchmark};
use std::path::Path;

let report = run_full_benchmark(
    Path::new("models/mobilenetv3.onnx"),
    BenchProvider::Auto,
    /*warmup=*/ 10,
    /*iterations=*/ 100,
)?;

println!("{}", report.to_markdown("mobilenetv3 / CPU"));
```

## Output format

`BenchReport::to_markdown` emits a single markdown table row suitable for
embedding into a larger benchmark table:

```
| label | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | stddev ms | min ms | max ms | throughput (inf/s) | cold_start ms | warm_start ms | model MB |
```
