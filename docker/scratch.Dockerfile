# ─── scratch.Dockerfile ──────────────────────────────────────────────────────
#
# WARNING: scratch images are NOT supported for airml because ONNX Runtime
# requires a libc (glibc or musl) at runtime. Even a fully statically-linked
# airml binary will dlopen() libonnxruntime.so, which itself pulls in glibc
# symbols. There is no production-ready fully-static ORT build from Microsoft.
#
# This file exists as a TEMPLATE for users who want to:
#   1. Build ORT from source with static linking enabled (unsupported / advanced).
#   2. Use a custom musl-based ORT build (community efforts only).
#
# Until a static ORT is available, use the distroless/cc image instead:
#   docker build -f Dockerfile -t airml/airml:0.2 .
#
# For the smallest supported production image size, distroless/cc-debian12
# is already ~20 MB compressed — nearly as small as scratch + libc.
# ─────────────────────────────────────────────────────────────────────────────

# TODO(users): Replace this base with a musl toolchain image if you have a
# static ORT build. e.g. `clux/muslrust` or `ghcr.io/cross-rs/aarch64-unknown-linux-musl`.
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    musl-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# TODO(users): Place your statically-linked libonnxruntime.a here.
# Microsoft does not ship one. You must build ORT from source with:
#   cmake -Donnxruntime_BUILD_SHARED_LIB=OFF ...
# and then set ORT_LIB_LOCATION to point to the output directory.
#
# COPY your-static-ort-build/libonnxruntime.a /usr/local/lib/
# ENV ORT_LIB_LOCATION=/usr/local/lib
# ENV ORT_STATIC=1

COPY . .

# TODO(users): Add --target x86_64-unknown-linux-musl (or aarch64) once you
# have a static ORT. The nlp feature requires tokenizers which links against
# libc; musl support may need additional patching.
RUN cargo build --release --features=nlp
RUN strip target/release/airml

# ─── Runtime stage ───────────────────────────────────────────────────────────
# TODO(users): Replace with `scratch` once you have confirmed your ORT build
# is truly static and all dylib dependencies are resolved.
# Test with: ldd target/release/airml  (should print "not a dynamic executable")
FROM scratch

# TODO(users): If ORT is truly static, no dylib copy is needed.
# If it is NOT static, you must use distroless/cc or debian:bookworm-slim.
COPY --from=builder /build/target/release/airml /airml

# TODO(users): Set this only if you have a static ORT with embedded path.
# ENV ORT_DYLIB_PATH=...

ENTRYPOINT ["/airml"]
