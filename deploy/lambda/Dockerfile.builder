FROM --platform=linux/arm64 rust:1.82-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev ca-certificates curl tar gzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# Lambda function uses 'provided.al2023' which provides glibc 2.34+
# So we can use a standard glibc build, no musl needed.
RUN cargo build --release --features=nlp --bin airml

# Strip binary to reduce cold-start package size
RUN strip /build/target/release/airml

# Lambda bootstrap is a renamed binary
RUN cp /build/target/release/airml /build/bootstrap

# Output stage just for extraction
FROM scratch AS output
COPY --from=builder /build/bootstrap /bootstrap
