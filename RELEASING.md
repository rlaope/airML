# Releasing airML

## Pre-release checklist

- [ ] `cargo test --workspace --features=coreml,nlp` passes
- [ ] `cargo clippy --workspace --all-targets --features=coreml,nlp -- -D warnings` passes
- [ ] `cargo public-api diff` reviewed for breaking changes
- [ ] `CHANGELOG.md` updated
- [ ] Version bumped in root + all member `Cargo.toml`s
- [ ] `bench/results/` updated with current numbers (optional)

## Release order

Member crates must be published first since they depend on each other:

1. `cargo publish -p airml-core`
2. `cargo publish -p airml-providers`
3. `cargo publish -p airml-preprocess`
4. `cargo publish -p airml-embed`
5. `cargo publish -p airml-tune` (depends on airml-core)
6. `cargo publish -p airml-hub` (independent)
7. `cargo publish -p airml-bench` (depends on airml-core, airml-providers)
8. `cargo publish` (root binary)

## Post-release

- Tag `git tag v{version} && git push --tags` — triggers `release.yml`.
- Update Homebrew formula SHAs in `Formula/airml.rb`.

## Automation

### CHANGELOG generation

`CHANGELOG.md` is regenerated automatically on every push to `master` by
`.github/workflows/changelog.yml` using `git-cliff` per `cliff.toml`.

To regenerate locally:

    cargo install git-cliff
    git cliff --output CHANGELOG.md

### Conventional commits

Commits should follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add airml serve subcommand`
- `fix: prevent panic in graph_parser on truncated input`
- `perf(core): pre-allocate KV cache buffer`
- `refactor(hub): extract URL parser to module`
- `docs(deploy): add Lambda ARM guide`
- `test(tune): cover dispatch heuristic edge cases`
- `ci: add coverage workflow`
- `chore(deps): bump ort to 2.0.0-rc.12`

Commits not matching this pattern are filtered out of the changelog by `cliff.toml`.

### Cross-platform builds

The current path uses the hand-written `release.yml` workflow. An alternative
using `cargo-dist` is available via the template `dist-workspace.toml`.
