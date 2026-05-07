#!/usr/bin/env bash
# airML cross-runtime benchmark runner
#
# Prerequisites (run once from the repo root):
#   airml install-runtime
#   airml pull mobilenetv3-small
#   airml pull bge-small-en
#
# Usage:
#   bash bench/run-all.sh
#   MODELS="mobilenetv3-small" bash bench/run-all.sh
#   ITERATIONS=50 WARMUP=5 bash bench/run-all.sh

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
RUNTIMES_DIR="${SCRIPT_DIR}/runtimes"
SITE_DIR="${SCRIPT_DIR}/site"

MODELS="${MODELS:-mobilenetv3-small bge-small-en}"
MODEL_DIR="${MODEL_DIR:-${HOME}/.airml/models}"
LABEL="${AIRML_BENCH_LABEL:-$(uname -m)/auto}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"

# ── Helpers ────────────────────────────────────────────────────────────────────

log()  { echo "==> $*"; }
info() { echo "    $*"; }
warn() { echo "[warn] $*" >&2; }
skip() { echo "[skip] $*" >&2; }

model_path() {
  echo "${MODEL_DIR}/$1.onnx"
}

safe_run() {
  # Run a command; on failure log a warning but don't abort the whole script.
  "$@" || warn "command failed (continuing): $*"
}

# ── Build runtimes ─────────────────────────────────────────────────────────────

log "Building ORT baseline..."
(cd "${RUNTIMES_DIR}/ort_baseline" && cargo build --release)

log "Building tract baseline..."
(cd "${RUNTIMES_DIR}/tract_baseline" && cargo build --release)

log "Compiling airml-bench..."
if cargo build --release --features=coreml -p airml-bench 2>/dev/null; then
  AIRML_FEATURES="coreml"
else
  cargo build --release -p airml-bench
  AIRML_FEATURES=""
fi

mkdir -p "${RESULTS_DIR}"

# ── Run benchmarks ─────────────────────────────────────────────────────────────

for MODEL_NAME in ${MODELS}; do
  MP="$(model_path "${MODEL_NAME}")"

  if [[ ! -f "${MP}" ]]; then
    skip "model not found: ${MP} — run 'airml pull ${MODEL_NAME}' first"
    continue
  fi

  log "Benchmarking model: ${MODEL_NAME}"
  LABEL_SAFE="${LABEL//\//-}"

  # airml
  info "airml..."
  AIRML_BENCH_LABEL="${LABEL}" \
  AIRML_BENCH_OUTPUT_DIR="${RESULTS_DIR}" \
  safe_run "${REPO_ROOT}/target/release/airml" bench "${MP}" \
    --iterations "${ITERATIONS}" --warmup "${WARMUP}" --provider auto

  # ORT
  info "ORT (CoreML if available)..."
  safe_run env MODEL_PATH="${MP}" \
    "${RUNTIMES_DIR}/ort_baseline/target/release/airml-bench-ort" \
      --label "${LABEL}" --iterations "${ITERATIONS}" --warmup "${WARMUP}" \
    > "${RESULTS_DIR}/ort__${MODEL_NAME}__${LABEL_SAFE}.json"

  # tract
  info "tract (CPU)..."
  safe_run env MODEL_PATH="${MP}" \
    "${RUNTIMES_DIR}/tract_baseline/target/release/airml-bench-tract" \
      --label "${LABEL}" --iterations "${ITERATIONS}" --warmup "${WARMUP}" \
    > "${RESULTS_DIR}/tract__${MODEL_NAME}__${LABEL_SAFE}.json"

  # Python
  info "Python onnxruntime..."
  safe_run python3 "${RUNTIMES_DIR}/python_baseline/bench.py" \
    --model-path "${MP}" \
    --label "${LABEL}" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    > "${RESULTS_DIR}/python__${MODEL_NAME}__${LABEL_SAFE}.json"

  info "Done: ${MODEL_NAME}"
done

# ── Generate site ──────────────────────────────────────────────────────────────

log "Generating site..."
python3 "${SITE_DIR}/build.py" \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${SITE_DIR}/dist"

log "All done. Open ${SITE_DIR}/dist/index.html to view results."
