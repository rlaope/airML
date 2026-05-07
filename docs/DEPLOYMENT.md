# airML Deployment Guide

This guide covers production deployment options for airML v0.5+.

---

## Run with Docker

Pull the official image and run a one-off inference command:

```bash
docker run --rm \
  -v "$HOME/.cache/airml:/root/.cache/airml" \
  airml/airml:0.2 \
  run -m bge-small-en --input "Hello, world."
```

Start the HTTP inference server on port 8080:

```bash
docker run -d \
  --name airml \
  -p 8080:8080 \
  -v airml-cache:/root/.cache/airml \
  airml/airml:0.2 \
  serve --bind 0.0.0.0:8080
```

Build from source for the current host architecture:

```bash
docker build -t airml/airml:local .
```

For multi-arch images (amd64 + arm64):

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t airml/airml:0.2 \
  --push .
```

---

## Run with docker-compose

The `docker/docker-compose.yml` file starts airml behind nginx on port 80 with a
persistent model cache volume.

```bash
# From the project root:
docker compose -f docker/docker-compose.yml up -d
```

nginx proxies `http://localhost/` to `airml:8080`, passes the `Authorization`
header through, and allows up to 10 MB request bodies (sufficient for large
embedding batches).

To check logs:

```bash
docker compose -f docker/docker-compose.yml logs -f airml
```

---

## Run as a systemd service

This is the recommended path for bare-metal Linux servers (Debian/Ubuntu/RHEL).

### 1. Install prerequisites

```bash
# Download the airml binary
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-linux-x86_64.tar.gz \
  | tar xz
sudo install -m 755 airml /usr/local/bin/airml

# Install ONNX Runtime
sudo airml install-runtime
# Or manually:
# curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-x64-1.21.1.tgz \
#   | sudo tar xz -C /usr/local/lib --strip-components=2 --wildcards '*/lib/libonnxruntime.so*'
```

### 2. Run the installer

```bash
sudo bash deploy/install-systemd.sh
```

This creates the `airml` system user, installs `deploy/airml.service` into
`/etc/systemd/system/`, and starts the service.

### 3. Verify

```bash
systemctl status airml
curl http://127.0.0.1:8080/health
```

The service binds to `127.0.0.1:8080` by default. Put nginx or Caddy in front
to expose it externally — see `docker/nginx.conf` for a reference config.

### Uninstall

```bash
sudo systemctl stop airml
sudo systemctl disable airml
sudo rm /etc/systemd/system/airml.service
sudo systemctl daemon-reload
sudo userdel airml
```

---

## Install via Homebrew

> Once the official tap is published at `airml/homebrew-airml`, install with:

```bash
brew tap airml/airml https://github.com/airml/homebrew-airml
brew install airml
```

ONNX Runtime is not bundled in the Homebrew bottle. Run after install:

```bash
airml install-runtime
```

The tap formula (`Formula/airml.rb`) is automatically updated with real SHA256
values by `.github/workflows/release.yml` on every tagged release.

---

## Run on AWS Lambda ARM

For full deployment instructions, see
[`deploy/lambda/README.md`](../deploy/lambda/README.md).

### Quick summary

airML's ~2.4 MB binary and fast Rust startup make it an excellent fit for
Lambda ARM (Graviton). The deployment uses:

- `provided.al2023` custom runtime with a `bootstrap` binary
- An ORT Lambda layer (`/opt/lib/libonnxruntime.so`) built by `deploy/lambda/build-ort-layer.sh`
- The [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)
  to bridge the Lambda Runtime API to `airml serve` (no Rust changes required)
- AWS SAM for packaging and deployment (`deploy/lambda/template.yaml`)

### Key constraints

- Set `ORT_DYLIB_PATH=/opt/lib/libonnxruntime.so` (provided by the ORT layer).
- Lambda ARM64 maps to `aarch64-unknown-linux-gnu` — use `--platform linux/arm64`
  in Docker buildx.
- Model weights are downloaded to `/tmp/airml/` at first invocation; provision
  at least 2 GB `EphemeralStorage` for larger models.
- Provision at least 1024 MB memory for reliable model loading.
- Cold start (model download + ORT init) is typically 300–800 ms on Graviton3
  at 1 GB memory for `bge-small-en`.
- No CoreML on Linux; ORT uses the CPU execution provider.

### Deploy in four commands

```bash
bash deploy/lambda/build-ort-layer.sh
bash deploy/lambda/build-bootstrap.sh
aws lambda publish-layer-version --layer-name airml-ort \
  --zip-file fileb://deploy/lambda/dist/airml-ort-layer-1.20.0.zip \
  --compatible-runtimes provided.al2023 --compatible-architectures arm64
sam deploy --template-file deploy/lambda/template.yaml \
  --stack-name airml-embeddings --capabilities CAPABILITY_IAM --resolve-s3
```

See [`deploy/lambda/README.md`](../deploy/lambda/README.md) for the full
step-by-step guide, cost model, and architecture diagram.

---

## Run on Cloudflare Workers

**Not currently supported.**

Cloudflare Workers execute JavaScript/TypeScript (or compiled WASM) in V8
isolates. airML is a native binary that links against `libonnxruntime.so` — a
shared library that requires a POSIX OS, which Workers do not provide.

ONNX Runtime does publish an experimental WASM backend (`ort-wasm`), but it
requires browser-style imports and is not compatible with the airML CLI or
library surface as of v0.5.

This is tracked as a v0.6 milestone item. See
[ROADMAP.md](../ROADMAP.md#v06----public-benchmark-site) for status. Until
then, use [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/)
for managed inference at the edge, or deploy airML on a Cloudflare-connected
VPS and proxy through a Worker.
