# `airml info`

Inspect an ONNX model: inputs, outputs, tensor shapes, data types, and producer metadata — without running inference.

## Usage

```
airml info --model <PATH-OR-URI> [OPTIONS]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <PATH>` | (required) | ONNX model path or registry id |
| `-v, --verbose` | off | Show all metadata fields including custom properties |

## Example

```bash
airml info -m mobilenetv3-small
```

```
Model: mobilenetv3-small
Producer: pytorch  version: 2.1.0
IR version: 8

Inputs:
  input   shape=[1, 3, 224, 224]  dtype=float32

Outputs:
  output  shape=[1, 1000]         dtype=float32

File size: 14.2 MB
```

With `--verbose`:

```bash
airml info -m mobilenetv3-small --verbose
```

```
... (above) ...

Custom metadata:
  task: image_classification
  dataset: imagenet
  accuracy_top1: 0.756
```

## Use before `airml run`

Use `airml info` to verify input shapes and choose the right `--preprocess` preset before running inference. If the input shape is `[1, 3, 224, 224]`, `imagenet` or `clip` preprocessing is appropriate.

## See also

- [`airml run`](run.md) — run inference
- [`airml bench`](bench.md) — benchmark the model
