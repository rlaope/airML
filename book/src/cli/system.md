# `airml system`

Print platform capabilities: OS, CPU architecture, Apple Silicon detection, and available ONNX Runtime execution providers.

## Usage

```
airml system
```

No options or arguments.

## Example output

**Apple Silicon Mac:**

```
OS:               macOS 14.4
Arch:             aarch64
Apple Silicon:    true
ORT version:      1.23.1
ORT dylib:        /usr/local/lib/onnxruntime-osx-arm64-1.23.1/lib/libonnxruntime.dylib

Available providers:
  CPU             (always available)
  CoreML          (macOS 12+)
  NeuralEngine    (Apple Silicon, macOS 13+)
```

**Linux x86_64:**

```
OS:               Ubuntu 22.04
Arch:             x86_64
Apple Silicon:    false
ORT version:      1.23.1

Available providers:
  CPU             (always available)
```

## Use this to diagnose setup issues

If CoreML or NeuralEngine do not appear:

1. Confirm `ORT_DYLIB_PATH` is set: `echo $ORT_DYLIB_PATH`
2. Confirm the dylib exists: `ls $ORT_DYLIB_PATH`
3. Re-run `airml install-runtime` to reinstall

## See also

- [`airml install-runtime`](install-runtime.md) — install ONNX Runtime
- [Installation](../getting-started/installation.md)
