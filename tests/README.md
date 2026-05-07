# Integration tests

These tests live at the project root (not inside individual crates) because
they exercise the full binary surface: real model loading, real ORT calls,
and the airml-hub cache.

## Run with ORT installed

    airml install-runtime
    cargo test --test integration_e2e --test integration_hub --test integration_tune

## CI gating

`integration_e2e` is gated on `ORT_DYLIB_PATH`. It auto-skips when ORT isn't
available. The hub and tune tests do NOT require ORT and always run.

## Adding a new e2e test

1. Add a new function in `tests/integration_e2e.rs`.
2. Call `skip_if_no_ort("test_name")` at the top.
3. Use `models/synthetic-identity.onnx` as the standard fixture.
