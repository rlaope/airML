#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v asciinema >/dev/null; then
  echo "Install asciinema first: brew install asciinema"
  exit 1
fi

asciinema rec airml-quickstart.cast \
  --title "airML quickstart" \
  --idle-time-limit 2 \
  --command "bash -c 'set -euo pipefail; \
    echo airml install-runtime; sleep 1; \
    airml install-runtime; \
    echo airml pull bge-small-en; sleep 1; \
    airml pull bge-small-en; \
    echo airml bench -m bge-small-en -n 100; sleep 1; \
    airml bench -m bge-small-en -n 100; \
    echo done.'"
