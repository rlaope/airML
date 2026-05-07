# Lambda Runtime Shim — Architecture Note

## The problem

The `airml` binary is a CLI and HTTP server. It does not implement the
[Lambda Runtime API](https://docs.aws.amazon.com/lambda/latest/dg/runtimes-api.html),
which requires a process to:

1. Poll `GET http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next`
2. Process the event payload
3. `POST` the response to `.../runtime/invocation/{requestId}/response`
4. Loop back to step 1

Without this loop, Lambda considers the function broken and kills it after
the `Timeout` seconds.

## v0.5 supported path: AWS Lambda Web Adapter

The recommended approach for v0.5 is the
[AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)
— a public Lambda extension layer maintained by AWS Labs.

**How it works:**

```
Lambda Runtime API
       │
       ▼
AWS Lambda Adapter (sidecar, /opt/bootstrap via AWS_LAMBDA_EXEC_WRAPPER)
       │  translates invocation events ↔ HTTP/1.1
       ▼
airml serve --bind 127.0.0.1:8080
       │  standard OpenAI-compatible HTTP server
       ▼
ONNX Runtime
```

The adapter layer ARN for ARM64 (us-east-1, as of 2025):

```
arn:aws:lambda:us-east-1:753240598075:layer:LambdaAdapterLayerArm64:24
```

This ARN is region-specific. For other regions, see the
[adapter release page](https://github.com/awslabs/aws-lambda-web-adapter/releases).

### Required environment variables

```yaml
AWS_LAMBDA_EXEC_WRAPPER: /opt/bootstrap   # tells Lambda to run adapter before bootstrap
PORT: "8080"                              # adapter forwards to this port
```

These are already set in `template.yaml`.

### How `airml serve` becomes the handler

When `AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap` is set, Lambda executes:

```
/opt/bootstrap /var/task/bootstrap
```

The adapter (`/opt/bootstrap`) starts first, launches `airml serve` (the
function's `bootstrap`) as a subprocess on port 8080, waits for it to be
ready (health check), then begins the Lambda Runtime API polling loop —
forwarding each event as an HTTP POST to `http://127.0.0.1:8080/v1/embeddings`.

### Limitations of this approach

- Adds ~5 MB to the cold-start footprint (the adapter layer).
- The adapter adds a small per-request overhead (~1 ms) for HTTP translation.
- API Gateway request/response size limits still apply (6 MB sync).
- The adapter is a third-party binary in the critical path; audit its releases.

## v0.6 plan: native `lambda_runtime` wrapper

For v0.6, a thin Rust crate (`crates/airml-lambda`) will wrap `airml-core`
directly with the [`lambda_runtime`](https://crates.io/crates/lambda_runtime) crate:

```rust
// crates/airml-lambda/src/main.rs (sketch)
use lambda_runtime::{service_fn, Error, LambdaEvent};
use airml_core::embed;

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}

async fn handler(event: LambdaEvent<EmbedRequest>) -> Result<EmbedResponse, Error> {
    let embeddings = embed(&event.payload.input, &event.payload.model).await?;
    Ok(EmbedResponse { embeddings })
}
```

Benefits over the adapter approach:

- Eliminates the AWS Lambda Adapter layer dependency.
- Zero HTTP overhead — the event payload is deserialized directly.
- Full control over initialization: model loading can happen in `main()` before
  the handler loop, so it runs during the Lambda init phase (not billed as
  invocation time for provisioned concurrency).
- Smaller binary (no HTTP server stack needed).

**Status:** tracked in `ROADMAP.md` v0.6.

## Choosing between the two

| | v0.5 (Adapter) | v0.6 (native) |
|---|---|---|
| Ready now | Yes | No |
| Requires code changes | No | Yes (`crates/airml-lambda`) |
| Extra layer dependency | Yes (AWS Labs) | No |
| Per-request overhead | ~1 ms | ~0 ms |
| Init-phase model load | Via workaround | Native |
| Recommended for | Immediate deployments | Production long-term |
