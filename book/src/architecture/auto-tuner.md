# Auto-Tuner Internals

`airml-tune` contains a stateless `BackendOracle` that maps model characteristics to a `BackendRecommendation` without running any benchmarks.

```mermaid
flowchart TD
    A[ModelMetadata + ONNX graph bytes] --> B{Parse op histogram}
    B -->|parse ok| C[OpHistogram: counts per op_type]
    B -->|parse error / no file| D[Heuristic from input tensor names]
    C --> E{dominant_class thresholds}
    D --> F[infer_family from input names]
    F --> G[infer_op_class from ModelFamily]
    E -->|Conv > 30%| H[OpClass::ConvHeavy]
    E -->|MatMul/Gemm > 40%| I[OpClass::GemmHeavy]
    E -->|Attention > 20%| J[OpClass::AttentionHeavy]
    E -->|ControlFlow > 5%| K[OpClass::ControlFlow]
    E -->|fallback| L[OpClass::Mixed]
    G --> M[ModelProfile]
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M
    M --> N{BackendOracle::recommend}
    N -->|Vision + ConvHeavy| O[CoreMLAneOnly]
    N -->|TextEncoder + AttentionHeavy + static shapes| O
    N -->|TextEncoder + AttentionHeavy + dynamic shapes| P[CoreMLAll]
    N -->|ImageTextDual + Mixed| P
    N -->|LanguageModel + ControlFlow| Q[CoreMLGpuOnly]
    N -->|Unknown + dynamic| R[CpuOnlyWithReason]
    N -->|catch-all| P
```

## Two classification paths

**Metadata path** (`profile_from_metadata`): uses input tensor names and shapes only. Fast — no file I/O. Input names like `pixel_values`, `input_ids`, or `attention_mask` map directly to `ModelFamily` variants, which are then mapped to `OpClass`.

**Graph path** (`profile_from_path`): reads the raw ONNX protobuf and counts every `NodeProto.op_type` without pulling in `prost` or requiring `protoc`. The resulting `OpHistogram` applies fractional thresholds to determine `OpClass`. This path is more accurate for models with unusual input naming.

For best results, combine both: call `profile_from_metadata` to get family and dynamic-shape info, then patch `dominant_op_class` from the histogram if the model file is available.
