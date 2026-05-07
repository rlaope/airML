# airML Cross-Runtime Benchmark Baselines

This directory contains minimal standalone programs for running the same
model through four different inference runtimes so results can be compared
on an equal footing.

Each baseline reads `MODEL_PATH` from the environment, runs warmup passes,
runs timed passes, and prints **one JSON object** to stdout matching the
schema at `bench/results/schema.json`.

## Runtimes

| Directory | Runtime | Notes |
|---|---|---|
| `ort_baseline/` | raw `ort` (ONNX Runtime Rust) | Optional CoreML via `--features coreml` |
| `tract_baseline/` | `tract-onnx` | CPU-only, no hardware accelerators |
| `python_baseline/` | Python `onnxruntime` | Requires Python 3.10+ and `onnxruntime` |
| `candle_baseline/` | `candle` | Stub — see README; ONNX support in beta |

## Running each baseline

### Prerequisites

```bash
# Pull a model first (from the airML root):
airml install-runtime
airml pull mobilenetv3-small   # saves to ~/.airml/models/
export MODEL_PATH="$HOME/.airml/models/mobilenetv3-small.onnx"
```

### 1. airML (the one being validated)

```bash
# From the repo root:
cargo bench -p airml-bench -- --bench inference
# Or use the CLI:
AIRML_BENCH_LABEL="M2-Pro/CoreML-ANE" airml bench "$MODEL_PATH" --provider auto
```

JSON output is written to `bench/results/airml__<model>__<label>.json`
when the `AIRML_BENCH_OUTPUT_DIR` env var points to `bench/results/`.

### 2. raw ORT (`ort_baseline/`)

```bash
cd bench/runtimes/ort_baseline

# CPU:
MODEL_PATH="$MODEL_PATH" cargo run --release -- \
    --label "M2-Pro/CPU" --iterations 100 --warmup 10

# With CoreML (Apple Silicon):
MODEL_PATH="$MODEL_PATH" cargo run --release --features coreml -- \
    --label "M2-Pro/CoreML" --iterations 100 --warmup 10
```

Pipe to a file:

```bash
MODEL_PATH="$MODEL_PATH" cargo run --release --features coreml -- \
    --label "M2-Pro/CoreML" \
    > ../../results/ort__mobilenetv3-small__M2-Pro-CoreML.json
```

### 3. tract (`tract_baseline/`)

```bash
cd bench/runtimes/tract_baseline

MODEL_PATH="$MODEL_PATH" cargo run --release -- \
    --label "M2-Pro/CPU" --iterations 100 --warmup 10 \
    > ../../results/tract__mobilenetv3-small__M2-Pro-CPU.json
```

tract runs on CPU only — no CoreML / ANE support.

### 4. Python onnxruntime (`python_baseline/`)

```bash
pip install onnxruntime numpy   # one-time

python3 bench/runtimes/python_baseline/bench.py \
    --model-path "$MODEL_PATH" \
    --label "M2-Pro/CPU" \
    --iterations 100 \
    --warmup 10 \
    > bench/results/python__mobilenetv3-small__M2-Pro-CPU.json

# With CoreML:
python3 bench/runtimes/python_baseline/bench.py \
    --model-path "$MODEL_PATH" \
    --label "M2-Pro/CoreML" \
    --provider CoreMLExecutionProvider \
    > bench/results/python__mobilenetv3-small__M2-Pro-CoreML.json
```

### 5. candle

See `candle_baseline/README.md`. A working baseline is not provided for v0.2
because candle's ONNX support is still in beta.

## Adding a new runtime

1. Create `bench/runtimes/<name>_baseline/` with its own `[workspace]` in `Cargo.toml`
   (or a standalone Python script).
2. Print one JSON object to stdout matching `bench/results/schema.json`.
   Set `"runtime"` to a new enum value and open a PR to add it to the schema.
3. Save the output to `bench/results/<runtime>__<model>__<label>.json`.
4. Re-run `python3 bench/site/build.py` to regenerate the comparison site.

## JSON schema

All output files must validate against `bench/results/schema.json` (Draft-07).
Required top-level keys: `schema_version`, `runtime`, `runtime_version`,
`label`, `model`, `model_size_mb`, `host`, `metrics`.

The `metrics` object must contain:
`p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `mean_ms`, `stddev_ms`,
`min_ms`, `max_ms`, `throughput_inf_per_sec`, `cold_start_ms`, `warm_start_ms`.

## Important: these projects are NOT workspace members

Each `ort_baseline/` and `tract_baseline/` Cargo project declares
`[workspace]` in its own `Cargo.toml` so that `cargo check` from the airML
root workspace does not pull in their heavy dependencies. Run `cargo check`
from inside each subdirectory when working on them.
