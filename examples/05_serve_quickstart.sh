#!/usr/bin/env bash
# 05_serve_quickstart.sh — start airml serve and send an embeddings request
set -euo pipefail

# 1. Ensure ONNX Runtime is installed
airml install-runtime

# 2. Pull the embedding model (cached after first run)
airml pull bge-small-en

# 3. Start the HTTP daemon in the background
airml serve --bind 127.0.0.1:8080 &
SERVE_PID=$!

# 4. Give the server a moment to bind
sleep 2

# 5. Send an embeddings request and print the first 5 values
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-small-en","input":["Hello, world."]}' \
  | jq '.data[0].embedding[:5]'

# 6. Check available models
curl -s http://127.0.0.1:8080/v1/models | jq '.data[].id'

# 7. Health check
curl -s http://127.0.0.1:8080/healthz

# 8. Shut down the server
kill "$SERVE_PID"
