# Your first inference

This walkthrough classifies an image using MobileNetV3-Small — a 14 MB model that runs on the Neural Engine.

## 1. Pull the model

```bash
airml pull mobilenetv3-small
```

Output:

```
Pulling mobilenetv3-small from registry...
  SHA256 verified
  Cached to ~/.cache/airml/mobilenetv3-small.onnx
```

## 2. Download a test image

```bash
curl -o cat.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg
```

## 3. Run inference

```bash
airml run -m mobilenetv3-small -i cat.jpg --top-k 3
```

Expected output:

```
Top 3 predictions:
  1. tabby cat           0.612
  2. tiger cat           0.218
  3. Egyptian cat        0.091
Inference time: 12ms  Provider: NeuralEngine
```

## 4. Inspect the model

```bash
airml info -m mobilenetv3-small
```

```
Model: mobilenetv3-small
Inputs:
  input  [1, 3, 224, 224]  float32
Outputs:
  output [1, 1000]         float32
Producer: pytorch
```

## 5. Benchmark latency

```bash
airml bench -m mobilenetv3-small -n 200 -w 20
```

```
Provider: NeuralEngine
Iterations: 200  Warmup: 20
  P50:  11.2ms
  P95:  13.8ms
  P99:  15.1ms
```

## Using a HuggingFace model

airML can pull directly from HuggingFace with the `hf://` URI:

```bash
airml run -m hf://Xenova/clip-vit-base-patch32 -i cat.jpg --preprocess clip
```

## Next steps

- [airml run reference](../cli/run.md) — all flags
- [Auto-tuner overview](../apple-silicon/auto-tuner.md) — how provider selection works
- [Library API](../crates/airml-core.md) — embed airML in your Rust app
