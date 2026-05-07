# Launch checklist

## 1 week before

- [ ] All v0.2 milestones in ROADMAP.md checked off
- [ ] CI green on master for 7 consecutive days
- [ ] Bench numbers from at least 2 different M-series chips
- [ ] All curated registry models verified (sha256 confirmed)
- [ ] Homebrew tap created and tested (`brew install airml/tap/airml`)
- [ ] Domain airml.dev pointing to docs site

## 1 day before

- [ ] Tag v0.2.0 (triggers release.yml)
- [ ] Verify GitHub release artifacts download cleanly on all 4 targets
      (macos-aarch64, macos-x86_64, linux-x86_64, linux-aarch64)
- [ ] Update Homebrew formula SHAs from release artifacts
- [ ] Schedule HN post for Tuesday 8 AM PT
- [ ] Pre-warm community: post in r/rust on Monday

## Launch day

- [ ] HN post (Tuesday 8 AM PT)
- [ ] Twitter thread within 1 hour of HN post hitting top 30
- [ ] Reddit r/rust + r/LocalLLaMA at noon PT
- [ ] Blog post live with launch date
- [ ] Monitor issues / Discord / mentions for 6 hours

## Day +1

- [ ] Respond to all HN comments
- [ ] Triage issues
- [ ] Schedule any urgent fixes for v0.2.1

## Week +1

- [ ] Retrospective: what worked, what did not
- [ ] Update ROADMAP based on feedback
- [ ] Plan v0.3 sprint
