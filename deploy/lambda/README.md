# airML on AWS Lambda ARM (Graviton)

## Why Lambda ARM

AWS Graviton3 processors deliver approximately 1.5x better price-performance than
equivalent x86 Lambda configurations. airML is uniquely well-suited to this target:

- **Binary size: ~2.4 MB.** Cold-start package extraction is near-instant.
- **No JVM, no interpreter.** The Rust binary starts in microseconds; all cold-start
  time is spent in ONNX Runtime model loading, not language runtime init.
- **Sub-100 ms cold starts** are realistic for small embedding models (bge-small-en)
  at 1 GB Lambda memory on Graviton3.
- **Cost model:** Lambda ARM charges ~20% less per GB-second than x86. For an
  embeddings workload doing 10 M requests/month at 512 MB / 200 ms average duration,
  ARM saves roughly $8–12/month vs x86 — and that is before the performance uplift.

---

## How it works

```
┌─────────────────────────────────────────────────────┐
│  Lambda invocation                                   │
│                                                      │
│  /opt/bootstrap  (AWS Lambda Adapter extension)      │
│       │  translates Lambda runtime API ↔ HTTP        │
│       ▼                                              │
│  airml serve --bind 127.0.0.1:8080                   │
│       │  OpenAI-compatible /v1/embeddings             │
│       ▼                                              │
│  ONNX Runtime  (/opt/lib/libonnxruntime.so)          │
│       │  loaded from the ORT Lambda layer            │
│       ▼                                              │
│  Model weights  (/tmp/airml/ — downloaded on first   │
│                  invocation, cached for warm calls)  │
└─────────────────────────────────────────────────────┘
```

### Components

| Component | Source | Lambda path |
|---|---|---|
| `bootstrap` binary | Built by `Dockerfile.builder` | Function zip root |
| `libonnxruntime.so` | ORT release tarball | `airml-ort` layer → `/opt/lib/` |
| AWS Lambda Adapter | Public AWS layer | Extension → `/opt/bootstrap` (sidecar) |
| Model weights | Downloaded at runtime | `/tmp/airml/` (ephemeral) |

### Why the AWS Lambda Adapter?

The current `airml` binary is a CLI / HTTP server, not a native Lambda runtime
(it does not poll the Lambda Runtime API). The
[AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)
is a sidecar extension that:

1. Polls `http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next`
2. Translates the API Gateway / HTTP API event into a plain HTTP request
3. Forwards it to `airml serve` on `127.0.0.1:8080`
4. Posts the response back to the Lambda Runtime API

This lets the existing `airml serve` work on Lambda with **zero Rust changes**.

A native `lambda_runtime`-based wrapper (`crates/airml-lambda`) is planned for
v0.6 and will eliminate this dependency.

---

## Cost model (embeddings workload example)

Assumptions:
- Model: `bge-small-en` (33 M params, fast)
- Average duration: 150 ms/request (warm), 400 ms (cold)
- Memory: 1024 MB
- Region: us-east-1, ARM64 pricing

| Volume | Requests/month | Compute cost | Cold-start fraction |
|---|---|---|---|
| Small | 1 M | ~$0.75 | 5% |
| Medium | 10 M | ~$7.50 | 1% |
| Large | 100 M | ~$75 | 0.1% |

Pricing uses $0.0000000133 per ms per GB (ARM Lambda, 2025). Free tier covers
the first 400,000 GB-seconds/month.

For sustained high-throughput workloads (>100 req/s continuous), a container on
ECS Fargate ARM or a Graviton EC2 instance will be cheaper. Lambda shines for
spiky, event-driven, or low-baseline workloads.

---

## Step-by-step: build, package, deploy

### Prerequisites

- Docker with `docker buildx` and QEMU (for cross-compilation on x86 hosts)
- AWS CLI v2, configured with credentials
- AWS SAM CLI (`brew install aws-sam-cli` or pip)
- An S3 bucket for SAM deployment artifacts (SAM can create one with `--resolve-s3`)

### 1. Enable Docker buildx for ARM cross-compilation

```bash
docker buildx create --use --name cross
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

### 2. Build the Lambda bootstrap binary

```bash
bash deploy/lambda/build-bootstrap.sh
# Output: deploy/lambda/dist/airml-function.zip
```

### 3. Build the ONNX Runtime Lambda layer

```bash
bash deploy/lambda/build-ort-layer.sh
# Output: deploy/lambda/dist/airml-ort-layer-1.20.0.zip
```

### 4. Publish the ORT layer to your AWS account

```bash
aws lambda publish-layer-version \
  --layer-name airml-ort \
  --description "ONNX Runtime 1.20.0 for Linux ARM" \
  --zip-file fileb://deploy/lambda/dist/airml-ort-layer-1.20.0.zip \
  --compatible-runtimes provided.al2023 \
  --compatible-architectures arm64
```

Note the `LayerVersionArn` from the output and update `template.yaml` if you
need to pin to a specific version number (the template uses `:1` by default).

### 5. Deploy with SAM

```bash
bash deploy/lambda/sam-build.sh
# Or manually:
sam deploy \
  --template-file deploy/lambda/template.yaml \
  --stack-name airml-embeddings \
  --capabilities CAPABILITY_IAM \
  --region us-east-1 \
  --resolve-s3 \
  --parameter-overrides "DefaultModel=bge-small-en AuthToken=mysecrettoken"
```

### 6. Test the endpoint

```bash
API_URL=$(aws cloudformation describe-stacks \
  --stack-name airml-embeddings \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
  --output text)

curl -s "$API_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mysecrettoken" \
  -d '{"model":"bge-small-en","input":"Hello from Lambda"}' | jq .
```

---

## Limitations

| Constraint | Detail |
|---|---|
| No CoreML | CoreML is macOS-only; ORT uses the CPU execution provider on Lambda |
| No GPU | Lambda does not offer GPU instances; use SageMaker for GPU inference |
| `/tmp` size | Default 512 MB; configurable up to 10 GB via `EphemeralStorage.Size` |
| Cold start | Model download on first invocation adds 2–10 s (network-dependent); use provisioned concurrency to eliminate |
| Concurrency | Each Lambda instance handles one request at a time; scale via concurrency, not threads |
| ORT dylib | Must be supplied via a Lambda layer; the function zip alone does not include it |
| ARM only | These scripts target `arm64` (Graviton). For x86 Lambda, change `--platform` and ORT tarball URL |

### Reducing cold starts

1. **Provisioned concurrency** — keeps instances warm, eliminates cold starts entirely.
2. **Pre-warm the model cache** — include a lightweight `init` call in your Lambda
   wrapper that loads the model during the init phase (before the handler loop).
3. **Smaller model** — `bge-small-en` (33 M params) loads ~3x faster than
   `bge-large-en` (335 M params).
4. **More memory** — Lambda allocates CPU proportionally to memory. 1 GB loads
   models roughly 2x faster than 512 MB.

---

## See also

- `deploy/lambda/template.yaml` — AWS SAM deployment template
- `deploy/lambda/Dockerfile.builder` — ARM cross-compilation builder
- `deploy/lambda/lambda-runtime-shim.md` — v0.6 native runtime plan
- [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)
- [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
