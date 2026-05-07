# airml-tune

The auto-tuner crate. Contains `BackendOracle`, which analyzes an ONNX model's graph structure and recommends the optimal CoreML compute units without requiring a test run.

## How it works

`BackendOracle::recommend_for_path` parses the ONNX protobuf graph, builds an op-histogram, and applies heuristics based on model family classification:

```rust
use airml_tune::BackendOracle;

let recommendation = BackendOracle::recommend_for_path("model.onnx")?;
println!("{:?}", recommendation.compute_units);
println!("{}", recommendation.reason);
```

## Heuristic table

| Model class | Auto pick | Reason |
|-------------|-----------|--------|
| Vision (Conv-heavy) | ANE only | ANE excels at convolution |
| Text encoder, static shapes | ANE only | Best ANE throughput |
| Text encoder, dynamic shapes | All compute units | Let CoreML decide per shape |
| Image+Text dual encoder | All compute units | Mixed workload |
| Language model (KV cache) | GPU only | ANE struggles with autoregressive control flow |

## ONNX graph parser

The graph parser (`src/graph_parser.rs`) is a hand-rolled protobuf reader with zero external dependencies. It is fuzz-tested nightly — see [Stability](../operations/stability.md).

## v0.4 improvements

v0.4 will replace family heuristics with actual per-op latency profiling. See the [roadmap](../roadmap.md).

## See also

- [Auto-tuner overview](../apple-silicon/auto-tuner.md)
- [Compute units guide](../apple-silicon/compute-units.md)
