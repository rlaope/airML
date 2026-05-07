#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ORT_VERSION="${ORT_VERSION:-1.20.0}"
OUT_DIR="${OUT_DIR:-dist}"
mkdir -p "$OUT_DIR"

echo "Downloading ONNX Runtime $ORT_VERSION for Linux ARM..."
curl -L -o /tmp/ort.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"

# Lambda layer structure: /opt/lib/ on the runtime, so layer zip uses 'lib/' prefix.
WORK_DIR="$(mktemp -d)"
trap "rm -rf '$WORK_DIR'" EXIT
mkdir -p "$WORK_DIR/lib"

tar -xzf /tmp/ort.tgz -C "$WORK_DIR" --strip-components=1
cp "$WORK_DIR/lib/libonnxruntime.so.${ORT_VERSION}" "$WORK_DIR/lib/libonnxruntime.so"

(cd "$WORK_DIR" && zip -r "$OLDPWD/$OUT_DIR/airml-ort-layer-${ORT_VERSION}.zip" lib/)

echo "Created $OUT_DIR/airml-ort-layer-${ORT_VERSION}.zip"
echo ""
echo "To publish as a Lambda layer:"
echo "  aws lambda publish-layer-version \\"
echo "    --layer-name airml-ort \\"
echo "    --description 'ONNX Runtime ${ORT_VERSION} for Linux ARM' \\"
echo "    --zip-file fileb://$OUT_DIR/airml-ort-layer-${ORT_VERSION}.zip \\"
echo "    --compatible-runtimes provided.al2023 \\"
echo "    --compatible-architectures arm64 \\"
echo "    --license-info MIT"
