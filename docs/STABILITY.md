# Stability Policy

airML follows semver from v1.0 onwards. Until then:
- v0.x.y: minor versions (0.2 → 0.3) may include public API breakage. Patch versions (0.2.0 → 0.2.1) will not.
- Public APIs are tracked via `cargo public-api`; see `.github/workflows/public-api.yml`.
- Internal modules under `crates/*/src/` are not public unless re-exported in `lib.rs`.

## Hardened parsers

The ONNX graph parser at `crates/airml-tune/src/graph_parser.rs` is fuzz-tested nightly.
See `fuzz/README.md` for local fuzzing instructions.

## Reporting issues

File a GitHub issue with reproduction steps. Crash bugs in parsers are treated as
security issues — please email the maintainers privately first.
