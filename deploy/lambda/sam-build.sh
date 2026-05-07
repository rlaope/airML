#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Build everything
./build-ort-layer.sh
./build-bootstrap.sh

# Deploy
sam deploy \
  --template-file template.yaml \
  --stack-name airml-embeddings \
  --capabilities CAPABILITY_IAM \
  --region "${AWS_REGION:-us-east-1}" \
  --resolve-s3 \
  --parameter-overrides "DefaultModel=${AIRML_DEFAULT_MODEL:-bge-small-en} AuthToken=${AIRML_AUTH_TOKEN:-}"
