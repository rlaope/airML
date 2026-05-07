# Stability

## Versioning policy

airML follows semver from v1.0 onwards. Until then:

- **Minor versions** (0.2 → 0.3) may include public API breakage.
- **Patch versions** (0.2.0 → 0.2.1) will not include breaking changes.

Public APIs are tracked via `cargo public-api`. The CI job `.github/workflows/public-api.yml` runs on every PR and reports any surface changes. The check is informational until v1.0, at which point it becomes a hard gate.

Internal modules under `crates/*/src/` are not public unless re-exported in `lib.rs`.

## Hardened parsers

The ONNX graph parser at `crates/airml-tune/src/graph_parser.rs` is fuzz-tested nightly via `cargo-fuzz`. The fuzz target `graph_parser` runs against `airml_tune::histogram_from_bytes` for 20 minutes per night.

Run locally:

```bash
# Requires nightly Rust
cargo +nightly fuzz run graph_parser
```

See `fuzz/README.md` for full instructions.

## Reporting issues

File a GitHub issue with reproduction steps. **Crash bugs in the ONNX parser are treated as security issues** — please email the maintainers privately before filing a public issue.

## Stable crates (post v1.0)

| Crate | Semver stable |
|-------|--------------|
| `airml-core` | v1.0+ |
| `airml-embed` | v1.0+ |
| `airml-providers` | v1.0+ |
| `airml-preprocess` | v1.0+ |
| `airml-tune` | v1.0+ (internal APIs excluded) |
| `airml-hub` | v1.0+ |
| `airml-bench` | v1.0+ |

## See also

- [Observability](observability.md)
- [Roadmap](../roadmap.md) — v1.0 milestone details
