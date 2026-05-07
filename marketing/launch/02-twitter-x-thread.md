# Twitter / X Thread

7 tweets. Character counts are exact (counted including spaces and punctuation).
Each tweet text is in a code block followed by the count.

---

**Tweet 1 — Hook**

```
We just shipped airML: a Rust ONNX runtime for Apple Silicon. The release
binary is 2.4 MB. No Python. No Docker. No pip install.

github.com/rlaope/airML

[GIF: airml pull bge-small-en -> airml bench showing 12 ms mean latency]
```
<!-- char count: 196 (without the GIF line, which is a media attachment) -->

---

**Tweet 2 — Why**

```
The Python ONNX path on macOS: create venv, pip install onnxruntime, spend
an hour on CoreML EP flags, end up running on CPU anyway.

Docker inference images are routinely 3 GB. Cold start is 4-5 seconds just
to import torch.

There is a better path.
```
<!-- char count: 272 -->

---

**Tweet 3 — What**

```
airML is a single binary.

Install:
  curl -L .../airml-macos-aarch64.tar.gz | tar xz
  sudo mv airml /usr/local/bin/
  airml install-runtime

Pull a model:
  airml pull bge-small-en

Run:
  airml bench -m bge-small-en -n 100
```
<!-- char count: 219 -->

---

**Tweet 4 — Auto-tuner**

```
BackendOracle picks CoreML compute units per model:

Conv-heavy vision    -> ANE only
Text encoder static  -> ANE only
Text encoder dynamic -> All units
LM w/ KV cache       -> GPU only

Override: --provider {cpu,coreml,neural-engine,auto}
```
<!-- char count: 239 -->

---

**Tweet 5 — HTTP daemon**

```
airml serve starts an OpenAI-compatible embeddings API:

  airml serve --bind 127.0.0.1:8080

  curl -X POST localhost:8080/v1/embeddings \
    -d '{"model":"bge-small-en","input":["hello"]}'

Any OpenAI client library works without modification.
```
<!-- char count: 246 -->

---

**Tweet 6 — Anti-goals**

```
What airML explicitly does not do:

- No CUDA / NVIDIA GPU (use candle for that)
- No training (use burn)
- No Python bindings — that is the point
- No model registry beyond ~20 curated entries
- No iOS / Android SDK

Apple Silicon inference, nothing else.
```
<!-- char count: 253 -->

---

**Tweet 7 — Call to action**

```
If a model fails to load, open an issue with:
- the model ID
- your chip (M1/M2/M3, Pro/Max/Ultra)
- output of `airml system`

That triple is enough to reproduce most failures.

Full docs: https://airml.github.io/airml
Source: https://github.com/rlaope/airML
```
<!-- char count: 264 -->
