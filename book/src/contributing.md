# How to contribute

We welcome:

- New benchmark results from your hardware (`crates/airml-bench/`)
- Curated model registry additions (`crates/airml-hub/src/registry.rs`)
- Bug reports with reproducible test cases
- Documentation improvements

Please check [ROADMAP.md](roadmap.md) to see if your idea is in scope. Anti-goals are listed in [reference/anti-goals.md](reference/anti-goals.md).

## Local development

```bash
cargo build
cargo test
cargo clippy --all-targets
```

For the full feature set:

```bash
cargo build --features coreml,nlp
cargo test --workspace --features coreml,nlp
cargo clippy --workspace --all-targets --features coreml,nlp -- -D warnings
```

## Coding style

- Follow `cargo fmt` defaults.
- All public items documented with `///`.
- No `unwrap()` in production paths — use `Result` and `?`.
- Errors via `thiserror` for crates, `anyhow` for binaries and examples.

## Fuzzing

Fuzz targets live in `fuzz/`. See `fuzz/README.md` for instructions on running locally with `cargo-fuzz`. The nightly CI job (`.github/workflows/fuzz.yml`) runs `graph_parser` for 20 minutes against `airml_tune::histogram_from_bytes`.

```bash
# Requires nightly Rust
cargo +nightly fuzz run graph_parser
```

## Submitting

Open a PR against `master`. CI runs build + clippy on macOS arm64, macOS x86_64, Linux arm64, and Linux x86_64.

## See also

- [Roadmap](roadmap.md)
- [Releasing](releasing.md)
