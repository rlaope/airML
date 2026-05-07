# Security Policy

## Supported versions

The latest minor release is supported. Previous minor releases receive security fixes for 6 months.

## Reporting a vulnerability

Please **do not file a public GitHub issue** for security vulnerabilities.

Email the maintainers privately at security@airml.dev (or open a private GitHub Security Advisory if email is unavailable).

Include:
- Affected airML version (`airml --version`)
- Reproduction steps
- Impact assessment

We will acknowledge within 48 hours and aim to release a fix within 14 days for high-severity issues.

## Hardened parsers

The ONNX graph parser at `crates/airml-tune/src/graph_parser.rs` is fuzz-tested nightly via `.github/workflows/fuzz.yml`. Crashes found by fuzzing are treated as security issues.

## Out-of-scope

- Issues in upstream `ort`, `tokenizers`, or `ndarray` — please report to those projects directly.
- Issues that require attacker-controlled access to `~/.airml/` or `ORT_DYLIB_PATH`.
