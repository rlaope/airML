# airML Benchmark Suite

Cross-runtime ONNX inference benchmarks.
The "shock graph": **cold-start vs P50 latency** across airml, ort, tract, and Python.

## Live site

- Production: <https://benchmarks.airml.dev> *(placeholder — update after first deploy)*
- GitHub Pages: <https://rlaope.github.io/airML/bench/>

## Quick start

```bash
# From the repo root — pull a model first
airml install-runtime
airml pull mobilenetv3-small

# Run everything and open the site
bash bench/run-all.sh
open bench/site/dist/index.html
```

Or via make:

```bash
cd bench
make bench          # build + run + site
make site           # regenerate site only (from existing results)
make help           # list all targets
```

## Directory layout

```
bench/
  results/          JSON benchmark reports (one file per runtime × model × label)
    schema.json     JSON Schema Draft-07 for all report files
    example_*.json  Example data (synthetic, labeled — replace with real runs)
  runtimes/
    ort_baseline/   raw ort Rust binary
    tract_baseline/ tract-onnx Rust binary
    python_baseline/bench.py  Python onnxruntime script
    candle_baseline/README.md candle conversion notes (no binary yet)
    README.md       How to run each baseline
  site/
    build.py        Static site generator (pure Python stdlib)
    style.css       Monospace dark/light theme
    dist/           Generated HTML (gitignored; created by build.py)
  Makefile          make bench / make site / make clean
  run-all.sh        Same pipeline as Makefile, bash-only
  README.md         This file
```

## JSON report schema

Every file in `bench/results/*.json` must validate against `bench/results/schema.json` (Draft-07).

Key fields:

| Field | Type | Description |
|---|---|---|
| `schema_version` | `"1"` | Always `"1"` for this generation |
| `runtime` | string | `airml` \| `ort` \| `tract` \| `python` \| `candle` |
| `runtime_version` | string | e.g. `"0.2.0"` |
| `label` | string | Human label, e.g. `"M2-Pro/CoreML-ANE"` |
| `model` | string | Model ID, e.g. `"mobilenetv3-small"` |
| `model_size_mb` | number | ONNX file size in MB |
| `host.os` | string | `"macos"` \| `"linux"` \| `"windows"` |
| `host.arch` | string | `"aarch64"` \| `"x86_64"` |
| `host.cpu_brand` | string | e.g. `"Apple M2 Pro"` |
| `host.ram_gb` | integer | Total system RAM in GB |
| `metrics.p50_ms` | number | 50th-percentile latency (ms) |
| `metrics.p90_ms` | number | 90th-percentile latency (ms) |
| `metrics.p95_ms` | number | 95th-percentile latency (ms) |
| `metrics.p99_ms` | number | 99th-percentile latency (ms) |
| `metrics.mean_ms` | number | Mean latency (ms) |
| `metrics.stddev_ms` | number | Std deviation (ms) |
| `metrics.min_ms` | number | Minimum latency (ms) |
| `metrics.max_ms` | number | Maximum latency (ms) |
| `metrics.throughput_inf_per_sec` | number | Inferences per second |
| `metrics.cold_start_ms` | number | First inference after load (ms) |
| `metrics.warm_start_ms` | number | Average of warmup runs (ms) |

Validate a file:

```bash
python3 -c "
import json, sys
schema = json.load(open('bench/results/schema.json'))
data   = json.load(open(sys.argv[1]))
print('top-level keys:', list(data.keys()))
print('metrics keys:', list(data['metrics'].keys()))
" bench/results/example_airml__mobilenetv3-small__M2-Pro-CoreML-ANE.json
```

## What the site shows

- **Scatter plot** (cold_start_ms on X, p50_ms on Y): points labeled by runtime.
  Lower-left is best. airML should dominate the lower-left corner (fast cold start, low P50).

- **Table** grouped by model, sorted airml | ort | tract | python | candle.
  Shows all eight latency percentiles, throughput, model size, and runtime version.

## Adding a new runtime

1. Create `bench/runtimes/<name>_baseline/` with `[workspace]` in `Cargo.toml`
   (keep it outside the main workspace to avoid bloating `cargo check`).
2. Print one JSON object to stdout matching `bench/results/schema.json`.
3. Save to `bench/results/<runtime>__<model>__<label>.json`.
4. Add the runtime name to `RUNTIME_ORDER` in `bench/site/build.py`.
5. Run `python3 bench/site/build.py` and open `bench/site/dist/index.html`.

## CI

`.github/workflows/bench.yml` runs on push to `master` when `bench/**` changes,
and on `workflow_dispatch`. It uses a macOS-14 (Apple Silicon M1) runner,
installs ONNX Runtime via `airml install-runtime`, pulls `mobilenetv3-small`,
runs all four runtimes, and deploys the generated site to `gh-pages`.

Keep CI under 10 minutes by using `mobilenetv3-small` (14 MB) and `--iterations 50`.
