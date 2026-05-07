#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT_DIR="${OUT_DIR:-deploy/lambda/dist}"
mkdir -p "$OUT_DIR"

echo "Building airML bootstrap binary for Lambda ARM..."
docker buildx build \
  --platform linux/arm64 \
  -f deploy/lambda/Dockerfile.builder \
  --output type=local,dest="$OUT_DIR" \
  .

# Package the function zip
cd "$OUT_DIR"
chmod +x bootstrap
zip -j airml-function.zip bootstrap
echo "Created $OUT_DIR/airml-function.zip ($(du -h airml-function.zip | cut -f1))"
