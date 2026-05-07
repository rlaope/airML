# candle baseline — v0.2 stub

> **Status**: ONNX import in candle is in beta as of early 2025.
> A working candle baseline is not included for v0.2 of airML benchmarks.
> This document explains the conversion path for when candle ONNX support matures.

## Why candle is listed

candle is Hugging Face's pure-Rust ML framework. It excels at:

- Training and fine-tuning small models in Rust
- CUDA acceleration via NVIDIA GPUs
- Pure-Rust deployment without ONNX Runtime

airML targets Apple Silicon / CoreML; candle targets CUDA/Metal.
The comparison is still useful for researchers choosing a Rust ML stack.

## How to convert an ONNX model to candle's safetensors format

candle does not natively run `.onnx` files at inference time — it loads
weights in the `safetensors` format and requires you to re-implement the
model architecture in Rust.

### Step 1 — Export weights from ONNX to safetensors

```bash
pip install safetensors onnx numpy

python3 - <<'EOF'
import onnx
import numpy as np
from safetensors.numpy import save_file

model = onnx.load("model.onnx")
tensors = {}
for init in model.graph.initializer:
    arr = np.array(onnx.numpy_helper.to_array(init))
    tensors[init.name] = arr
save_file(tensors, "model_weights.safetensors")
print(f"Saved {len(tensors)} tensors.")
EOF
```

### Step 2 — Implement the model in candle

Refer to the candle examples repository for reference implementations:

- MobileNet: <https://github.com/huggingface/candle/tree/main/candle-examples/examples/mobilenet>
- BERT/BGE: <https://github.com/huggingface/candle/tree/main/candle-examples/examples/bert>

### Step 3 — Load weights and benchmark

```rust
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

let device = Device::new_metal(0)?;  // Apple Silicon GPU via Metal
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&["model_weights.safetensors"], candle_core::DType::F32, &device)?
};
// ... construct your model, run inference loop ...
```

### Step 4 — Emit a benchmark JSON report

When running, print a JSON line matching `bench/results/schema.json`:

```json
{
  "schema_version": "1",
  "runtime": "candle",
  "runtime_version": "0.8.0",
  "label": "M2-Pro/Metal",
  ...
}
```

## When will a working baseline be added?

Track <https://github.com/huggingface/candle/issues/> for ONNX inference
progress. Once `candle-onnx` supports the full ONNX opset used by BGE-Small
and MobileNetV3, a `candle_baseline/` Cargo project will be added here.

## References

- candle docs: <https://huggingface.github.io/candle/>
- candle-onnx crate: <https://crates.io/crates/candle-onnx>
- safetensors format: <https://huggingface.co/docs/safetensors>
