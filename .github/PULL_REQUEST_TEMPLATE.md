## Summary

<!-- One paragraph: what does this PR change and why? -->

## Type of change

<!-- Check one or more, prefix the PR title accordingly: -->

- [ ] `feat:` — new feature
- [ ] `fix:` — bug fix
- [ ] `perf:` — performance improvement
- [ ] `refactor:` — internal restructuring (no behavior change)
- [ ] `docs:` — documentation only
- [ ] `test:` — test additions/changes
- [ ] `ci:` — CI configuration

## Checklist

- [ ] `cargo test --workspace --features=coreml,nlp` passes locally
- [ ] `cargo clippy --workspace --all-targets --features=coreml,nlp -- -D warnings` passes
- [ ] Public API changes are intentional and noted in this PR description
- [ ] If the PR changes user-facing CLI behavior, the relevant `book/src/cli/*.md` page is updated
- [ ] If the PR adds dependencies, the new license is MIT/Apache/BSD compatible

## Breaking changes

<!-- If this is a breaking change for the public API, describe migration path -->

## Related issues

<!-- "Closes #123" or "Refs #456" -->
