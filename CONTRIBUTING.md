# Contributing to airML

We welcome:
- New benchmark results from your hardware (`crates/airml-bench/`)
- Curated model registry additions (`crates/airml-hub/src/registry.rs`)
- Bug reports with reproducible test cases
- Documentation improvements

Please check ROADMAP.md to see if your idea is in scope. Anti-goals are listed in README.md.

## Local development

```bash
cargo build
cargo test
cargo clippy --all-targets
```

## Coding style

- Follow `cargo fmt` defaults.
- All public items documented with `///`.
- No `unwrap()` in production paths -- use `Result` and `?`.
- Errors via `thiserror` for crates, `anyhow` for binaries / examples.

## Fuzzing

Fuzz targets live in `fuzz/`. See [`fuzz/README.md`](fuzz/README.md) for instructions on running them locally with `cargo-fuzz`. The nightly CI job (`.github/workflows/fuzz.yml`) runs `graph_parser` for 20 minutes against `airml_tune::histogram_from_bytes`.

## Submitting

Open a PR against `master`. CI runs build + clippy on macOS arm64, macOS x86_64, Linux arm64, Linux x86_64.

## Commit messages

Commits should follow [Conventional Commits](https://www.conventionalcommits.org/) so they are
picked up correctly by `git-cliff` when generating `CHANGELOG.md` (see `cliff.toml`):

| Prefix | Use for |
|--------|---------|
| `feat:` | new user-visible feature |
| `fix:` | bug fix |
| `perf:` | performance improvement |
| `refactor:` | internal restructuring, no behavior change |
| `docs:` | documentation only |
| `test:` | test additions or changes |
| `ci:` | CI configuration |
| `chore:` | maintenance (deps bumps, toolchain updates) |

Commits that do not match these prefixes are filtered out of the changelog automatically.
