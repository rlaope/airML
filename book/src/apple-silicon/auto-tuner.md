# Auto-tuner overview

`airml-tune` solves a real problem: CoreML has four compute unit settings, and picking the wrong one costs 2-5x latency. The auto-tuner parses your model's ONNX graph and recommends the right setting without requiring a test run.

## How it works

1. **Parse** — `BackendOracle::recommend_for_path` reads the ONNX protobuf and builds an op-histogram (a count of each operator type).
2. **Classify** — the histogram is matched against model family patterns (Conv-heavy = vision, MatMul-heavy with attention = text encoder, etc.).
3. **Recommend** — each family maps to a `ComputeUnits` setting based on empirical profiling.

```rust
use airml_tune::BackendOracle;

let rec = BackendOracle::recommend_for_path("model.onnx")?;
println!("Use: {:?}", rec.compute_units);
println!("Why: {}", rec.reason);
```

## Heuristic table

| Model class | Auto pick | Reason |
|-------------|-----------|--------|
| Vision (Conv-heavy) | `ANE only` | ANE excels at depthwise + pointwise conv |
| Text encoder, static shapes | `ANE only` | Highest ANE throughput for fixed-shape MatMuls |
| Text encoder, dynamic shapes | `All compute units` | Let CoreML reschedule per shape at runtime |
| Image+Text dual encoder (e.g. CLIP) | `All compute units` | Mixed workload; neither CPU nor ANE dominates |
| Language model (KV cache, autoregressive) | `GPU only` | ANE scheduling overhead hurts sequential decoding |

## Overriding the recommendation

Pass `--provider` to any inference command:

```bash
airml run -m clip-vit-b32 -i photo.jpg --provider neural-engine
```

Or in code:

```rust
use airml_providers::{CoreMLProvider, ComputeUnits};
let provider = CoreMLProvider::default()
    .with_compute_units(ComputeUnits::CpuAndNeuralEngine)
    .into_dispatch();
```

## v0.4 improvements

v0.4 will replace family heuristics with per-op latency profiling: the oracle will profile each op on real hardware once, cache the results, and use those numbers for future recommendations. See [Roadmap](../roadmap.md).

## See also

- [Compute units guide](compute-units.md)
- [CoreML model formats](model-formats.md)
- [airml-tune crate](../crates/airml-tune.md)
