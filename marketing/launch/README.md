# marketing/launch/

Launch content for airML 0.2. All drafts are in plain markdown.
Numbers marked "preliminary" must be validated on target hardware before posting.

## Files

| File | Platform | Status |
|---|---|---|
| [01-hacker-news.md](01-hacker-news.md) | Hacker News (Show HN) | draft |
| [02-twitter-x-thread.md](02-twitter-x-thread.md) | Twitter / X (7-tweet thread) | draft |
| [03-reddit-r-rust.md](03-reddit-r-rust.md) | Reddit r/rust | draft |
| [04-reddit-r-localllama.md](04-reddit-r-localllama.md) | Reddit r/LocalLLaMA | draft |
| [05-blog-launch-post.md](05-blog-launch-post.md) | airml.dev/blog (~800 words) | draft |
| [06-press-release-style.md](06-press-release-style.md) | Press / tech journalists | draft |
| [07-asciinema-script.md](07-asciinema-script.md) | 60-second demo recording script | draft |
| [checklist.md](checklist.md) | Pre-launch operations checklist | — |

## Voice rules (enforced across all files)

- Direct, technical, no jargon-bingo.
- One concrete number wherever possible.
- No emojis.
- No "revolutionary", "game-changing", "disruptive", "best-in-class", "synergy",
  "robust".
- Honest about anti-goals.
- All performance numbers sourced from M2 Pro unless stated otherwise.
- Numbers not yet measured are marked "preliminary — validate before posting."
- Canonical URL: https://github.com/rlaope/airML (no airml.dev links until
  domain is confirmed live).

## Before posting

1. Confirm bench numbers on at least 2 M-series chips.
2. Replace all "preliminary" annotations with measured values or remove the
   claims entirely.
3. Verify the GitHub release artifacts exist at the URLs referenced in the
   install commands.
4. Check the HN title is 80 characters or fewer.
5. Re-count each tweet — 280-character limit is strict.
6. See [checklist.md](checklist.md) for full pre-launch gate.
