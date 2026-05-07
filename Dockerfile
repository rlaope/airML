# ─── Stage 1: builder ────────────────────────────────────────────────────────
# Builds the airml binary with the nlp feature set (coreml is macOS-only).
# ORT is downloaded here so the dylib can be copied into the runtime image.
#
# For cross-arch builds use:
#   docker buildx build --platform linux/amd64,linux/arm64 -t airml/airml:0.2 .
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Download ONNX Runtime shared library from Microsoft's official releases.
# Version must match the `ort` crate version pinned in Cargo.toml (2.0.0-rc.11 → ORT 1.21.x).
# We use ORT 1.21.1 which is the last stable release compatible with ort 2.0.0-rc.11.
ARG TARGETARCH
RUN set -eu; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-aarch64-1.21.1.tgz"; \
    else \
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-x64-1.21.1.tgz"; \
    fi; \
    curl -fsSL "${ORT_URL}" | tar xz -C /tmp && \
    find /tmp -name "libonnxruntime.so*" -exec cp {} /usr/local/lib/libonnxruntime.so \;

# Copy the full source tree and build.
COPY . .

RUN cargo build --release --features=nlp

# Strip the binary to reduce image size.
RUN strip target/release/airml

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
# Uses distroless/cc which ships glibc + libgcc — the minimum needed by ORT.
# The final image contains only the binary and the ORT dylib; no shell, no
# package manager, no build tools.
FROM gcr.io/distroless/cc-debian12

COPY --from=builder /build/target/release/airml /usr/local/bin/airml
COPY --from=builder /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

ENTRYPOINT ["airml"]
