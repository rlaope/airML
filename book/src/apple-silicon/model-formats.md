# CoreML model formats

When airML runs a model via CoreML, ORT converts the ONNX graph to a CoreML model internally. CoreML supports two model formats: NeuralNetwork and MLProgram.

## NeuralNetwork

The original CoreML format. Supported from macOS 10.13 / iOS 11.

- Broader operator coverage for older macOS/iOS targets
- More mature toolchain support
- Required for some older CoreML operator patterns

## MLProgram

Introduced with CoreML 5 (macOS 12 / iOS 15). The recommended format for new deployments.

- Supports more operators (including some Transformer attention patterns)
- Can achieve better performance on ANE for modern model architectures
- Required for some newer operator types that ORT generates

## Setting the format

airML defaults to letting ORT choose. To pin a format:

```rust
use airml_providers::{CoreMLProvider, CoreMLModelFormat};

let provider = CoreMLProvider::default()
    .with_model_format(CoreMLModelFormat::MLProgram)
    .into_dispatch();
```

## Compilation cache

CoreML compiles the model on first load and caches the `.mlmodelc` package at:

```
~/Library/Caches/com.apple.coreml/
```

If you change the model or the format setting, delete this cache to force recompilation:

```bash
rm -rf ~/Library/Caches/com.apple.coreml/
```

## Troubleshooting

If CoreML fails to compile your model:
1. Check `RUST_LOG=debug` output for the specific error
2. Try `MLProgram` if using `NeuralNetwork` (or vice versa)
3. Fall back to `--provider cpu` to confirm the model itself is valid
4. File an issue with the model provenance and the debug log

## See also

- [Compute units guide](compute-units.md)
- [Auto-tuner overview](auto-tuner.md)
