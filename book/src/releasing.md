# Releasing

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

```bash
git tag v{version}
git push --tags
```

Pushing the tag triggers `.github/workflows/release.yml`, which:
- Builds release binaries for all platforms
- Creates a GitHub release with the tarballs
- Updates SHA256 values in `Formula/airml.rb`

## See also

- [Changelog](changelog.md)
- [Contributing](contributing.md)
