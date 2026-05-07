# airML

> The fastest way to run any ONNX model on Apple Silicon as a single binary.
> No Python, no Docker, no `pip install`.

airML packages ONNX Runtime and a curated set of models into a single native binary. You get sub-50ms cold starts, automatic Apple Neural Engine dispatch, and zero runtime dependencies. Install once, ship anywhere.

[![CI](https://github.com/airml/airml/actions/workflows/ci.yml/badge.svg)](https://github.com/airml/airml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![macOS arm64](https://img.shields.io/badge/platform-macOS%20arm64-lightgrey.svg)]()
[![Linux arm64 / x86_64](https://img.shields.io/badge/platform-Linux%20arm64%20%2F%20x86__64-lightgrey.svg)]()

## Why airML

| You want to...                                | airML                   | candle | ort       | tract |
|-----------------------------------------------|:-----------------------:|:------:|:---------:|:-----:|
| Ship a 50MB binary that runs ONNX             | yes                     | no     | no        | yes   |
| Use Apple Neural Engine without writing CoreML| yes                     | no     | manual    | no    |
| Auto-pick the best compute units per model    | yes (`airml-tune`)      | no     | no        | no    |
| Skip Python entirely                          | yes                     | yes    | yes (Rust)| yes   |
| Train models                                  | no                      | yes    | no        | no    |
| GPU on NVIDIA                                 | use `candle`            | yes    | yes       | no    |

## What's in this book

- **Getting Started** — install, run your first inference in 60 seconds.
- **CLI Reference** — every command, every flag.
- **Library API** — embed airML in your own Rust application.
- **Apple Silicon** — auto-tuner internals, compute unit selection.
- **Deployment** — Docker, systemd, Lambda, Homebrew.
- **Operations** — logging, metrics, stability guarantees.
- **Contributing** — roadmap, release process, changelog.
