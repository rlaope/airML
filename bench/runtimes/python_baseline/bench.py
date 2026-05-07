#!/usr/bin/env python3
"""airML Python/onnxruntime baseline benchmark.

Reads a model via onnxruntime (NOT optimum, NOT transformers), runs warmup
passes, then timed passes, and prints a single JSON line to stdout matching
the airML benchmark schema v1 (see bench/results/schema.json).

Requirements: onnxruntime>=1.16, numpy (stdlib otherwise).
Python 3.10+ compatible.

Usage:
    python3 bench.py --model-path model.onnx [--iterations 100] [--warmup 10] [--label "CPU"]
    python3 bench.py --model-path model.onnx --provider CUDAExecutionProvider
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="airML onnxruntime baseline benchmark — prints JSON to stdout."
    )
    p.add_argument("--model-path", required=True, help="Path to the ONNX model file.")
    p.add_argument("--iterations", type=int, default=100, help="Number of timed iterations (default: 100).")
    p.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10).")
    p.add_argument("--label", default="CPU", help="Human-readable label for this run (default: CPU).")
    p.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ORT execution provider (default: CPUExecutionProvider).",
    )
    return p.parse_args()


# ── Host info ─────────────────────────────────────────────────────────────────

def cpu_brand() -> str:
    """Best-effort CPU brand string — no third-party deps."""
    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            brand = result.stdout.strip()
            if brand:
                return brand
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def ram_gb() -> int:
    """Total RAM in GB — no third-party deps."""
    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            )
            return int(result.stdout.strip()) // (1024 ** 3)
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // (1024 * 1024)
        except Exception:
            pass
    return 0


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def make_input(session: ort.InferenceSession) -> dict:
    """Generate synthetic float32 inputs matching the model's input spec."""
    feeds = {}
    for inp in session.get_inputs():
        raw_shape = inp.shape
        shape = [
            d if isinstance(d, int) and d > 0
            else (1 if i == 0 else (3 if (i == 1 and len(raw_shape) == 4) else (224 if len(raw_shape) == 4 else 128)))
            for i, d in enumerate(raw_shape)
        ]
        feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feeds


def percentile(sorted_arr: list, p: float) -> float:
    idx = round((p / 100.0) * (len(sorted_arr) - 1))
    return sorted_arr[min(idx, len(sorted_arr) - 1)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    model_path = args.model_path
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Build session
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 1

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=[args.provider, "CPUExecutionProvider"],
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    feeds = make_input(session)
    output_names = [o.name for o in session.get_outputs()]

    # Cold start
    cold_start_t0 = time.perf_counter_ns()
    session.run(output_names, feeds)
    cold_start_ms = (time.perf_counter_ns() - cold_start_t0) / 1_000_000

    # Warmup
    warm_times_ms = []
    for _ in range(args.warmup):
        t0 = time.perf_counter_ns()
        session.run(output_names, feeds)
        warm_times_ms.append((time.perf_counter_ns() - t0) / 1_000_000)
    warm_start_ms = (sum(warm_times_ms) / len(warm_times_ms)) if warm_times_ms else cold_start_ms

    # Timed iterations
    times_ms = []
    for _ in range(args.iterations):
        t0 = time.perf_counter_ns()
        session.run(output_names, feeds)
        times_ms.append((time.perf_counter_ns() - t0) / 1_000_000)

    sorted_times = sorted(times_ms)
    mean_ms = sum(times_ms) / len(times_ms)
    variance = sum((x - mean_ms) ** 2 for x in times_ms) / len(times_ms)
    stddev_ms = variance ** 0.5
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    report = {
        "schema_version": "1",
        "runtime": "python",
        "runtime_version": ort.__version__,
        "label": args.label,
        "model": model_name,
        "model_size_mb": round(model_size_mb, 3),
        "host": {
            "os": platform.system().lower(),
            "arch": platform.machine().lower(),
            "cpu_brand": cpu_brand(),
            "ram_gb": ram_gb(),
        },
        "metrics": {
            "p50_ms": percentile(sorted_times, 50),
            "p90_ms": percentile(sorted_times, 90),
            "p95_ms": percentile(sorted_times, 95),
            "p99_ms": percentile(sorted_times, 99),
            "mean_ms": mean_ms,
            "stddev_ms": stddev_ms,
            "min_ms": sorted_times[0],
            "max_ms": sorted_times[-1],
            "throughput_inf_per_sec": throughput,
            "cold_start_ms": cold_start_ms,
            "warm_start_ms": warm_start_ms,
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
