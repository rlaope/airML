# airml-providers

Execution provider abstractions for ONNX Runtime. Wraps CoreML, CPU, and future GPU providers behind a uniform `into_dispatch()` interface consumed by `SessionConfig`.

## Key types

### `CpuProvider`

Always available, zero configuration:

```rust
use airml_providers::CpuProvider;
let provider = CpuProvider::default().into_dispatch();
```

### `CoreMLProvider` (coreml feature)

```rust
use airml_providers::{CoreMLProvider, ComputeUnits};

let provider = CoreMLProvider::default()
    .with_compute_units(ComputeUnits::CpuAndNeuralEngine)
    .with_subgraphs(true)
    .into_dispatch();
```

Convenience constructors:

```rust
CoreMLProvider::default().neural_engine_only()  // ANE + CPU
CoreMLProvider::default().gpu_only()            // GPU (no ANE)
CoreMLProvider::default().cpu_only()
```

### `ComputeUnits`

```rust
pub enum ComputeUnits {
    All,               // CPU + GPU + ANE
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}
```

### `CoreMLModelFormat`

```rust
pub enum CoreMLModelFormat {
    NeuralNetwork,  // Broader macOS/iOS compatibility
    MLProgram,      // More operators, potentially faster
}
```

## Utility functions

```rust
// Auto-select best providers for this platform
let providers = airml_providers::auto_select_providers();

// Check platform
let is_as = airml_providers::is_apple_silicon();
let info: SystemInfo = airml_providers::system_info();
```

## See also

- [Compute units guide](../apple-silicon/compute-units.md)
- [Auto-tuner overview](../apple-silicon/auto-tuner.md)
