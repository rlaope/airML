# airML cross-runtime benchmark suite
#
# Prerequisites (run once):
#   airml install-runtime
#   airml pull bge-small-en
#   airml pull mobilenetv3-small
#
# Usage:
#   make bench          # build all runtimes + run all models + generate site
#   make build-runtimes # compile ORT and tract baselines only
#   make run-benchmarks # run all runtimes against all models
#   make site           # regenerate the HTML site from existing results
#   make clean          # remove built binaries (keep results)

SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := bench

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS ?= mobilenetv3-small bge-small-en
MODEL_DIR ?= $(HOME)/.airml/models
RESULTS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))results
SITE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))site
RUNTIMES_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))runtimes
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..

AIRML_BENCH_LABEL ?= $(shell uname -m)/auto
WARMUP ?= 10
ITERATIONS ?= 100

.PHONY: bench build-runtimes run-benchmarks site clean help

## bench: full pipeline — build, run, generate site
bench: build-runtimes run-benchmarks site

## build-runtimes: compile ORT and tract baseline binaries
build-runtimes:
	@echo "==> Building ORT baseline..."
	cd "$(RUNTIMES_DIR)/ort_baseline" && \
	  cargo build --release
	@echo "==> Building tract baseline..."
	cd "$(RUNTIMES_DIR)/tract_baseline" && \
	  cargo build --release
	@echo "==> Compiling airml-bench..."
	cd "$(REPO_ROOT)" && \
	  cargo build --release --features=coreml -p airml-bench 2>/dev/null || \
	  cargo build --release -p airml-bench
	@echo "==> All runtimes built."

## run-benchmarks: run all runtimes for all models
run-benchmarks: $(foreach model,$(MODELS),run-model-$(model))

## run-model-<name>: run all runtimes for a single model
$(foreach model,$(MODELS),run-model-$(model)): run-model-%:
	$(eval MODEL_PATH := $(MODEL_DIR)/$*.onnx)
	@if [ ! -f "$(MODEL_PATH)" ]; then \
	  echo "[skip] model not found: $(MODEL_PATH) — run 'airml pull $*' first"; \
	  exit 0; \
	fi
	@echo "==> Benchmarking model: $*"

	@echo "  -> airml..."
	AIRML_BENCH_LABEL="$(AIRML_BENCH_LABEL)" \
	AIRML_BENCH_OUTPUT_DIR="$(RESULTS_DIR)" \
	  "$(REPO_ROOT)/target/release/airml" bench "$(MODEL_PATH)" \
	    --iterations "$(ITERATIONS)" --warmup "$(WARMUP)" \
	    --provider auto 2>/dev/null || true

	@echo "  -> ORT (CoreML)..."
	MODEL_PATH="$(MODEL_PATH)" \
	  "$(RUNTIMES_DIR)/ort_baseline/target/release/airml-bench-ort" \
	    --label "$(AIRML_BENCH_LABEL)" \
	    --iterations "$(ITERATIONS)" --warmup "$(WARMUP)" \
	  > "$(RESULTS_DIR)/ort__$*__$(subst /,-,$(AIRML_BENCH_LABEL)).json" || true

	@echo "  -> tract (CPU)..."
	MODEL_PATH="$(MODEL_PATH)" \
	  "$(RUNTIMES_DIR)/tract_baseline/target/release/airml-bench-tract" \
	    --label "$(AIRML_BENCH_LABEL)" \
	    --iterations "$(ITERATIONS)" --warmup "$(WARMUP)" \
	  > "$(RESULTS_DIR)/tract__$*__$(subst /,-,$(AIRML_BENCH_LABEL)).json" || true

	@echo "  -> Python onnxruntime..."
	python3 "$(RUNTIMES_DIR)/python_baseline/bench.py" \
	    --model-path "$(MODEL_PATH)" \
	    --label "$(AIRML_BENCH_LABEL)" \
	    --iterations "$(ITERATIONS)" --warmup "$(WARMUP)" \
	  > "$(RESULTS_DIR)/python__$*__$(subst /,-,$(AIRML_BENCH_LABEL)).json" 2>/dev/null || true

	@echo "  Done: $*"

## site: regenerate bench/site/dist/index.html from bench/results/
site:
	@echo "==> Generating site..."
	python3 "$(SITE_DIR)/build.py" \
	  --results-dir "$(RESULTS_DIR)" \
	  --output-dir "$(SITE_DIR)/dist"
	@echo "==> Site at: $(SITE_DIR)/dist/index.html"

## clean: remove compiled runtime binaries
clean:
	cd "$(RUNTIMES_DIR)/ort_baseline"   && cargo clean || true
	cd "$(RUNTIMES_DIR)/tract_baseline" && cargo clean || true

## help: print this help
help:
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/^## //' | column -t -s ':'
