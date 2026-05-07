# `airml install-runtime`

Auto-download the correct ONNX Runtime shared library for your platform and print the path to set as `ORT_DYLIB_PATH`.

## Usage

```
airml install-runtime [OPTIONS]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--version <VER>` | latest supported | ORT version to install |
| `--dir <PATH>` | `/usr/local/lib` | Installation directory |
| `--dry-run` | off | Print what would be downloaded without downloading |

## Example

```bash
airml install-runtime
```

```
Detecting platform: macOS aarch64
Downloading onnxruntime-osx-arm64-1.23.1.tgz ...
  SHA256 verified
  Installed to /usr/local/lib/onnxruntime-osx-arm64-1.23.1/lib/libonnxruntime.dylib

Add to your shell profile:
  export ORT_DYLIB_PATH=/usr/local/lib/onnxruntime-osx-arm64-1.23.1/lib/libonnxruntime.dylib
```

## Platform matrix

| Platform | Downloaded file |
|----------|----------------|
| macOS arm64 | `onnxruntime-osx-arm64-{ver}.tgz` |
| macOS x86_64 | `onnxruntime-osx-x86_64-{ver}.tgz` |
| Linux x86_64 | `onnxruntime-linux-x64-{ver}.tgz` |
| Linux arm64 | `onnxruntime-linux-aarch64-{ver}.tgz` |

## Making it permanent

Add the printed `export` line to `~/.zshrc`, `~/.bashrc`, or equivalent. Alternatively, set it system-wide via `/etc/environment` on Linux.

## See also

- [Installation](../getting-started/installation.md) — full install instructions
- [systemd deployment](../deployment/systemd.md) — setting ORT_DYLIB_PATH as a service environment variable
