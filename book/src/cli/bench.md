# `airml bench`

Measure inference latency for an ONNX model. Runs warmup iterations first, then reports P50/P95/P99 latency across N timed iterations.

## Usage

```
airml bench --model <PATH-OR-URI> [OPTIONS]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <PATH>` | (required) | ONNX model path or registry id |
| `-n, --iterations <N>` | `100` | Number of timed benchmark iterations |
| `-w, --warmup <N>` | `10` | Warmup iterations (not counted) |
| `-p, --provider <P>` | `auto` | Execution provider |
| `--shape <SHAPE>` | (from model) | Override input shape, e.g. `1,3,224,224` |

## Example

```bash
airml bench -m mobilenetv3-small -n 200 -w 20
```

```
Provider: NeuralEngine
Model:    mobilenetv3-small  [1, 3, 224, 224] -> [1, 1000]
Warmup:   20 iterations
Timed:    200 iterations

  Min:  10.1ms
  P50:  11.2ms
  P95:  13.8ms
  P99:  15.1ms
  Max:  18.4ms
```

## Compare providers

```bash
for p in cpu coreml neural-engine; do
  echo "=== $p ==="; airml bench -m mobilenetv3-small -p $p -n 100
done
```

## Input data

`airml bench` generates pseudo-Gaussian synthetic input tensors (CLT-based). This avoids deterministic patterns that might cache artificially well. Use your own data for production-representative benchmarks via [`airml-bench`](../crates/airml-bench.md) as a library.

## See also

- [airml-bench crate](../crates/airml-bench.md) — criterion-based benchmarks for CI
- [Performance](../operations/performance.md) — community benchmark results
