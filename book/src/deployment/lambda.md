# AWS Lambda ARM

airML runs on Lambda ARM64 (`arm64` architecture, mapped to `aarch64-unknown-linux-gnu`). The key constraint is that Lambda's filesystem is read-only except for `/tmp`.

## Setup

1. Use the `airml-linux-aarch64.tar.gz` release tarball.
2. Bundle `libonnxruntime.so` in a Lambda layer or copy it to `/tmp` at cold start.
3. Set `ORT_DYLIB_PATH` to point into `/tmp` where you copied the dylib.
4. Set `AIRML_CACHE_DIR=/tmp/airml` so the model cache lands in `/tmp`.

## Example bootstrap script

```bash
#!/usr/bin/env bash
# bootstrap (Lambda custom runtime)
set -euo pipefail

# Copy ORT from Lambda layer on first invocation
if [[ ! -f /tmp/libonnxruntime.so ]]; then
  cp /opt/lib/libonnxruntime.so /tmp/libonnxruntime.so
fi

export ORT_DYLIB_PATH=/tmp/libonnxruntime.so
export AIRML_CACHE_DIR=/tmp/airml

while true; do
  EVENT=$(curl -sf \
    "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next")
  REQUEST_ID=$(curl -sI \
    "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next" \
    | grep -i lambda-runtime-aws-request-id \
    | awk '{print $2}' | tr -d '\r')

  RESULT=$(airml embed \
    --cache-dir /tmp/airml \
    --input "$(echo "${EVENT}" | jq -r '.text')")

  curl -sf -X POST \
    "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/${REQUEST_ID}/response" \
    -d "${RESULT}"
done
```

## Constraints

| Constraint | Value | Notes |
|-----------|-------|-------|
| `/tmp` size | 512 MB – 10 GB | Configure in function settings |
| Min memory | 512 MB | Recommended for model load |
| Cold start | 200–800ms | At 1 GB memory, ORT + model load |
| Architecture | `arm64` | Use `aarch64` release tarball |

## Inference command

```bash
airml run -m bge-small-en \
  --cache-dir /tmp/airml \
  --input "Hello from Lambda"
```

## Cloudflare Workers

**Not currently supported.** Workers run in V8 isolates without POSIX filesystem access. See [Roadmap](../roadmap.md) for v0.6 status.

## See also

- [Deployment overview](overview.md)
- [Docker](docker.md) — for ECS / Fargate on ARM
