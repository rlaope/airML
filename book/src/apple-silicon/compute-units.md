# Compute units guide

CoreML exposes four compute unit settings that control which hardware runs your model. Choosing correctly can cut latency by 2-5x.

## The four settings

| Setting | Hardware used | Best for |
|---------|--------------|---------|
| `All` | CPU + GPU + Neural Engine | Mixed workloads; let CoreML decide |
| `CpuAndNeuralEngine` | CPU + ANE (no GPU) | Static-shape encoders; best ANE throughput |
| `CpuAndGpu` | CPU + GPU (no ANE) | Dynamic-shape models; GPU memory bandwidth |
| `CpuOnly` | CPU only | Debugging; compatibility fallback |

## Which setting to use

Use the auto-tuner (`--provider auto`) and let `BackendOracle` decide. If you need to override:

- **Vision models** (MobileNet, ResNet, EfficientNet): `neural-engine`
- **Text encoders, fixed input length** (BGE, MiniLM): `neural-engine`
- **Text encoders, variable input length**: `coreml` (All)
- **CLIP, dual encoders**: `coreml` (All)
- **Language models with KV cache**: `gpu_only` — ANE scheduling overhead hurts autoregressive decoding

## CoreML vs CPU fallback

CoreML compilation happens at first load and is cached in `~/Library/Caches/com.apple.coreml`. If compilation fails for a given setting, ORT falls back to CPU automatically. Set `RUST_LOG=debug` to see which provider ORT actually used:

```bash
RUST_LOG=airml_core=debug airml run -m mobilenetv3-small -i cat.jpg
```

## API

```rust
use airml_providers::{CoreMLProvider, ComputeUnits};

let provider = CoreMLProvider::default()
    .with_compute_units(ComputeUnits::CpuAndNeuralEngine)
    .into_dispatch();
```

## See also

- [Auto-tuner overview](auto-tuner.md)
- [CoreML model formats](model-formats.md)
- [airml-providers](../crates/airml-providers.md)
