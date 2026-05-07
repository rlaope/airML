# Homebrew

Homebrew installation is the recommended path for macOS developer machines.

## Status

The official tap is not yet published. Once available at `airml/homebrew-airml`:

```bash
brew tap airml/airml https://github.com/airml/homebrew-airml
brew install airml
```

ONNX Runtime is not bundled in the Homebrew bottle. Run after install:

```bash
airml install-runtime
```

## Tap formula

The formula lives at `Formula/airml.rb` in the repository. SHA256 values are automatically updated by `.github/workflows/release.yml` on every tagged release — maintainers do not need to update them manually.

## Until the tap is published

Use the pre-built binary:

```bash
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-macos-aarch64.tar.gz | tar xz
sudo mv airml /usr/local/bin/
airml install-runtime
```

Or install from source:

```bash
cargo install --git https://github.com/rlaope/airML
```

## See also

- [Installation](../getting-started/installation.md)
- [Deployment overview](overview.md)
