# Installation

## macOS (Apple Silicon) — recommended

```bash
# 1. Download the pre-built binary
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-aarch64.tar.gz | tar xz
sudo mv airml /usr/local/bin/

# 2. Install ONNX Runtime (preferred)
airml install-runtime

# Or install manually:
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-osx-arm64-1.23.1.tgz \
  | tar xz -C /usr/local/lib
export ORT_DYLIB_PATH=/usr/local/lib/onnxruntime-osx-arm64-1.23.1/lib/libonnxruntime.dylib
```

## macOS (Intel)

```bash
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-x86_64.tar.gz | tar xz
sudo mv airml /usr/local/bin/

curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-osx-x86_64-1.23.1.tgz \
  | tar xz -C /usr/local/lib
export ORT_DYLIB_PATH=/usr/local/lib/onnxruntime-osx-x86_64-1.23.1/lib/libonnxruntime.dylib
```

## Linux (x86_64)

```bash
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-linux-x86_64.tar.gz | tar xz
sudo mv airml /usr/local/bin/

curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz \
  | tar xz -C /usr/local/lib
export ORT_DYLIB_PATH=/usr/local/lib/onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so
```

## From source

```bash
git clone https://github.com/rlaope/airML.git
cd airML
cargo build --release --features coreml,nlp
```

The compiled binary will be at `target/release/airml`.

## Homebrew (coming soon)

```bash
brew tap airml/airml https://github.com/airml/homebrew-airml
brew install airml
airml install-runtime   # ORT is not bundled in the bottle
```

See [Homebrew deployment](../deployment/homebrew.md) for details on the tap status.

## Verify installation

```bash
airml system
```

Expected output on Apple Silicon:

```
OS: macOS 14.x
Arch: aarch64
Apple Silicon: true
Providers: CPU, CoreML, NeuralEngine
```
