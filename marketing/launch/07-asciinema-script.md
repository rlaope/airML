# Demo script – 60-second asciinema recording

This is the narration and typing script for the recorder. It is not a cast
file. Type each command at a natural pace. Wait for each command to finish
before advancing.

All numbers shown are from M2 Pro. Mark as preliminary in any accompanying
caption.

---

```
[scene 1: 0-10s]
# Install the binary and the ONNX Runtime dylib

$ curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-aarch64.tar.gz | tar xz
$ sudo mv airml /usr/local/bin/
$ airml install-runtime
Downloading onnxruntime-osx-arm64-1.23.1...
Verifying checksum... ok
Installed to ~/.local/lib/airml/libonnxruntime.dylib
Done.
```

```
[scene 2: 10-25s]
# Pull a model from the registry (content-addressed, sha256-verified)

$ airml pull bge-small-en
Fetching bge-small-en from registry...
Downloading model (133 MB)... done
Verifying sha256... ok
Cached at ~/.cache/airml/bge-small-en/
```

```
[scene 3: 25-45s]
# Benchmark 100 inferences — BackendOracle picks Neural Engine automatically

$ airml bench -m bge-small-en -n 100
Backend selected: ANE (Neural Engine) — via BackendOracle
Running 100 inferences...

  Mean latency:  12.3 ms   [preliminary — M2 Pro]
  P50:           11.8 ms
  P99:           14.1 ms
  Throughput:    81 inferences/sec
  Binary size:   2.4 MB
```

```
[scene 4: 45-60s]
# Start the OpenAI-compatible embeddings server and query it

$ airml serve --bind 127.0.0.1:8080 &
Listening on http://127.0.0.1:8080
Model: bge-small-en (Neural Engine)

$ curl -s -X POST http://localhost:8080/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model":"bge-small-en","input":["the fastest path to ONNX inference"]}' \
  | jq '{index: .data[0].index, dims: (.data[0].embedding | length)}'

{
  "index": 0,
  "dims": 384
}
```

---

## Narrator notes

- Scene 1: do not rush the install-runtime output — it is the "aha" moment for
  people who have manually wrangled ORT dylib paths before.
- Scene 3: pause one beat on "Binary size: 2.4 MB" before advancing.
- Scene 4: the jq output confirms the embedding dimension (384 for bge-small-en)
  without printing a 384-element float array on screen.
- Add a caption: "Numbers from M2 Pro / macOS 14. Preliminary — validate on
  your hardware."
- Total target: 58-62 seconds at normal typing speed.
