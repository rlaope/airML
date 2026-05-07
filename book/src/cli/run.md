# `airml run`

Run inference on a single input file. The default pipeline is image classification: preprocess the image, run the ONNX model, apply softmax, and print the top-K labels.

## Usage

```
airml run --model <PATH-OR-URI> --input <FILE> [OPTIONS]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <URI>` | (required) | ONNX model path, registry id, or `hf://owner/repo[/file]` URI |
| `-i, --input <FILE>` | (required) | Input image file |
| `-l, --labels <FILE>` | none | Optional labels file (one label per line) |
| `-k, --top-k <N>` | `5` | Number of top predictions to display |
| `-p, --provider <P>` | `auto` | `cpu`, `coreml`, `neural-engine`, or `auto` |
| `--preprocess <PRESET>` | `imagenet` | `imagenet`, `clip`, `yolo`, or `none` |
| `--raw` | off | Print raw tensor values instead of softmax classification |
| `-v, --verbose` | off | Show timing and provider details |

## Examples

```bash
# Classify an image with auto-selected provider
airml run -m mobilenetv3-small -i cat.jpg

# Force Neural Engine only
airml run -m mobilenetv3-small -i cat.jpg --provider neural-engine

# Use a HuggingFace model with CLIP preprocessing
airml run -m hf://Xenova/clip-vit-base-patch32 -i photo.jpg --preprocess clip

# Show top 10 predictions with raw scores
airml run -m mobilenetv3-small -i cat.jpg --top-k 10 --raw

# Use a local ONNX file with a labels file
airml run -m ./models/resnet50.onnx -i dog.jpg -l imagenet_labels.txt
```

## Preprocessing presets

| Preset | Size | Mean | Std | Notes |
|--------|------|------|-----|-------|
| `imagenet` | 224×224 | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | Standard ImageNet normalization |
| `clip` | 224×224 | [0.481, 0.457, 0.408] | [0.268, 0.261, 0.275] | OpenAI CLIP normalization |
| `yolo` | 640×640 | none | none | Letterbox, no normalization |
| `none` | (unchanged) | none | none | Pass image bytes as-is |

## See also

- [`airml info`](info.md) — inspect model metadata before running
- [`airml bench`](bench.md) — measure inference latency
- [Auto-tuner overview](../apple-silicon/auto-tuner.md) — how `--provider auto` works
