#!/usr/bin/env python3
"""airML benchmark static-site generator.

Reads all JSON files from ``bench/results/`` (relative to this script's
parent directory), generates an HTML comparison page with:

  - A table grouped by model with one row per (runtime, label) pair.
  - An inline SVG scatter plot: cold_start_ms on X, p50_ms on Y.
  - Runtimes sorted: airml | ort | tract | python | candle | others.

Output: ``bench/site/dist/index.html``

No external dependencies — pure Python 3.10+ stdlib only.
No npm, no node, no third-party JS.

Schema expected (version "1"):
  {
    "schema_version": "1",
    "runtime": "airml",          # one of: airml, ort, tract, python, candle
    "runtime_version": "0.2.0",
    "label": "M2-Pro/CoreML-ANE",
    "model": "bge-small-en",
    "model_size_mb": 133.0,
    "host": {"os": ..., "arch": ..., "cpu_brand": ..., "ram_gb": ...},
    "metrics": {
      "p50_ms": 12.3, "p90_ms": 18.1, "p95_ms": 20.0, "p99_ms": 30.0,
      "mean_ms": 13.5, "stddev_ms": 2.1, "min_ms": 10.0, "max_ms": 35.0,
      "throughput_inf_per_sec": 74.0,
      "cold_start_ms": 250.0,
      "warm_start_ms": 14.0
    }
  }

Adding a new runtime:
  1. Drop a JSON file matching the schema into bench/results/.
  2. Add the runtime name to RUNTIME_ORDER below if you want a fixed sort position.
  3. Re-run this script: python3 bench/site/build.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

RUNTIME_ORDER = ["airml", "ort", "tract", "python", "candle"]

# Colours must stay in sync with style.css CSS variables.
RUNTIME_COLOURS = {
    "airml":  "#2563eb",
    "ort":    "#16a34a",
    "tract":  "#d97706",
    "python": "#9333ea",
    "candle": "#dc2626",
}
DEFAULT_COLOUR = "#888888"

SITE_TITLE = "airML benchmarks"
SITE_SUBTITLE = "Where do you want to ship from?"

LIVE_URL = "https://benchmarks.airml.dev"
GHPAGES_URL = "https://rlaope.github.io/airML/bench/"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all *.json files from results_dir. Skip schema.json and .gitkeep."""
    records = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "schema.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] skipping {p.name}: {exc}", file=sys.stderr)
            continue
        if data.get("schema_version") != "1":
            print(f"[warn] skipping {p.name}: unsupported schema_version", file=sys.stderr)
            continue
        records.append(data)
    return records


def runtime_sort_key(runtime: str) -> int:
    try:
        return RUNTIME_ORDER.index(runtime)
    except ValueError:
        return len(RUNTIME_ORDER)


def sort_records(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda r: (r.get("model", ""), runtime_sort_key(r.get("runtime", "")), r.get("label", "")),
    )


def fmt(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )

# ── SVG scatter plot ──────────────────────────────────────────────────────────

def build_scatter_svg(records: list[dict]) -> str:
    """Return an inline SVG scatter plot (cold_start_ms vs p50_ms)."""
    if not records:
        return '<p class="no-data">No data yet — run benchmarks and add JSON files to bench/results/.</p>'

    # Collect points
    points = []
    for r in records:
        m = r.get("metrics", {})
        cold = m.get("cold_start_ms")
        p50  = m.get("p50_ms")
        if cold is None or p50 is None:
            continue
        points.append({
            "x": float(cold),
            "y": float(p50),
            "runtime": r.get("runtime", "unknown"),
            "label": r.get("label", ""),
            "model": r.get("model", ""),
        })

    if not points:
        return '<p class="no-data">No metrics with cold_start_ms + p50_ms found.</p>'

    # SVG canvas
    W, H = 860, 480
    PAD_L, PAD_R, PAD_T, PAD_B = 72, 40, 30, 60

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    x_min, x_max = 0.0, max(xs) * 1.15 or 1.0
    y_min, y_max = 0.0, max(ys) * 1.15 or 1.0

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = PAD_L + (x - x_min) / (x_max - x_min) * (W - PAD_L - PAD_R)
        py = H - PAD_B - (y - y_min) / (y_max - y_min) * (H - PAD_T - PAD_B)
        return round(px, 1), round(py, 1)

    def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
        if hi <= lo:
            return [lo]
        step = (hi - lo) / max(n - 1, 1)
        magnitude = 10 ** math.floor(math.log10(step)) if step > 0 else 1
        nice_step = round(step / magnitude) * magnitude or magnitude
        ticks = []
        v = 0.0
        while v <= hi * 1.01:
            if v >= lo - 1e-9:
                ticks.append(v)
            v = round(v + nice_step, 10)
        return ticks[:n + 2]

    x_ticks = nice_ticks(x_min, x_max)
    y_ticks = nice_ticks(y_min, y_max)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'role="img" aria-label="Scatter plot: cold-start vs P50 latency">',
        f'<title>Cold-start vs P50 latency comparison</title>',
        # Background
        f'<rect width="{W}" height="{H}" fill="var(--bg,#fff)"/>',
        # Grid lines
        '<g stroke="var(--border,#ddd)" stroke-width="1" opacity="0.5">',
    ]

    for tv in y_ticks:
        _, py = to_px(x_min, tv)
        lines.append(f'<line x1="{PAD_L}" y1="{py}" x2="{W - PAD_R}" y2="{py}"/>')
    for tv in x_ticks:
        px, _ = to_px(tv, y_min)
        lines.append(f'<line x1="{px}" y1="{PAD_T}" x2="{px}" y2="{H - PAD_B}"/>')
    lines.append("</g>")

    # Axis labels (ticks)
    lines.append('<g font-family="monospace" font-size="11" fill="var(--fg-muted,#666)">')
    for tv in x_ticks:
        px, _ = to_px(tv, y_min)
        lines.append(f'<text x="{px}" y="{H - PAD_B + 16}" text-anchor="middle">{fmt(tv, 1)}</text>')
    for tv in y_ticks:
        _, py = to_px(x_min, tv)
        lines.append(f'<text x="{PAD_L - 6}" y="{py + 4}" text-anchor="end">{fmt(tv, 1)}</text>')
    lines.append("</g>")

    # Axis title labels
    cx = PAD_L + (W - PAD_L - PAD_R) / 2
    lines.append(
        f'<text x="{cx}" y="{H - 8}" text-anchor="middle" '
        f'font-family="monospace" font-size="12" fill="var(--fg-muted,#666)">'
        f'Cold-start (ms)</text>'
    )
    cy = PAD_T + (H - PAD_T - PAD_B) / 2
    lines.append(
        f'<text transform="rotate(-90,14,{cy})" x="14" y="{cy}" '
        f'text-anchor="middle" font-family="monospace" font-size="12" fill="var(--fg-muted,#666)">'
        f'P50 latency (ms)</text>'
    )

    # Data points
    for pt in points:
        px, py = to_px(pt["x"], pt["y"])
        colour = RUNTIME_COLOURS.get(pt["runtime"], DEFAULT_COLOUR)
        tip = f"{pt['runtime']} / {pt['label']} ({pt['model']}): cold={fmt(pt['x'])}ms p50={fmt(pt['y'])}ms"
        lines.append(
            f'<circle cx="{px}" cy="{py}" r="7" fill="{colour}" opacity="0.85">'
            f'<title>{html_escape(tip)}</title></circle>'
        )
        # Runtime label beside dot
        label_text = pt["runtime"]
        lines.append(
            f'<text x="{px + 10}" y="{py + 4}" '
            f'font-family="monospace" font-size="10" fill="{colour}">'
            f'{html_escape(label_text)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


# ── Table ─────────────────────────────────────────────────────────────────────

def build_table_html(records: list[dict]) -> str:
    if not records:
        return '<p class="no-data">No benchmark results found in bench/results/.</p>'

    cols = [
        ("Runtime",    "runtime"),
        ("Label",      "label"),
        ("Cold ms",    "cold_start_ms"),
        ("P50 ms",     "p50_ms"),
        ("P90 ms",     "p90_ms"),
        ("P99 ms",     "p99_ms"),
        ("Mean ms",    "mean_ms"),
        ("Stddev ms",  "stddev_ms"),
        ("Min ms",     "min_ms"),
        ("Max ms",     "max_ms"),
        ("inf/s",      "throughput_inf_per_sec"),
        ("Size MB",    "model_size_mb"),
        ("Version",    "runtime_version"),
    ]

    header_cells = "".join(f"<th>{c[0]}</th>" for c in cols)
    html = [
        '<div class="table-wrapper">',
        "<table>",
        f"<thead><tr>{header_cells}</tr></thead>",
        "<tbody>",
    ]

    current_model = None
    for r in records:
        model = r.get("model", "unknown")
        if model != current_model:
            current_model = model
            html.append(
                f'<tr class="model-group-header">'
                f'<td colspan="{len(cols)}">{html_escape(model)}</td></tr>'
            )

        runtime = r.get("runtime", "unknown")
        m = r.get("metrics", {})

        def cell(key: str) -> str:
            if key == "runtime":
                badge_cls = html_escape(runtime)
                return f'<td><span class="runtime-badge {badge_cls}">{html_escape(runtime)}</span></td>'
            if key == "label":
                return f'<td>{html_escape(r.get("label", "—"))}</td>'
            if key in ("model_size_mb", "runtime_version"):
                val = r.get(key)
            else:
                val = m.get(key)
            if val is None:
                return "<td>—</td>"
            if key == "runtime_version":
                return f"<td>{html_escape(str(val))}</td>"
            return f"<td>{fmt(val)}</td>"

        cells = "".join(cell(c[1]) for c in cols)
        html.append(f"<tr>{cells}</tr>")

    html += ["</tbody>", "</table>", "</div>"]
    return "\n".join(html)


# ── Legend ────────────────────────────────────────────────────────────────────

def build_legend_html(records: list[dict]) -> str:
    seen = sorted(
        {r.get("runtime", "unknown") for r in records},
        key=runtime_sort_key,
    )
    items = []
    for rt in seen:
        colour = RUNTIME_COLOURS.get(rt, DEFAULT_COLOUR)
        items.append(
            f'<span class="legend-item">'
            f'<span class="legend-dot {html_escape(rt)}" style="background:{colour}"></span>'
            f'{html_escape(rt)}</span>'
        )
    return f'<div class="legend">{"".join(items)}</div>'


# ── Full HTML page ────────────────────────────────────────────────────────────

def build_html(records: list[dict], generated_at: str) -> str:
    sorted_records = sort_records(records)
    scatter_svg = build_scatter_svg(sorted_records)
    table_html  = build_table_html(sorted_records)
    legend_html = build_legend_html(sorted_records)
    n_results   = len(sorted_records)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_escape(SITE_TITLE)}</title>
<link rel="stylesheet" href="../style.css">
<meta name="description" content="Cross-runtime ONNX inference benchmarks: airml vs ort vs tract vs Python">
</head>
<body>
<div class="container">

<header>
  <h1><span class="logo">airML</span> benchmarks</h1>
  <p class="subtitle">{html_escape(SITE_SUBTITLE)}</p>
  <p class="meta">
    {n_results} result(s) &mdash; generated {html_escape(generated_at)} &mdash;
    <a href="{html_escape(LIVE_URL)}" rel="noopener">benchmarks.airml.dev</a> &bull;
    <a href="https://github.com/rlaope/airML" rel="noopener">source</a>
  </p>
</header>

<section>
  <h2>Cold-start vs P50 latency (lower is better on both axes)</h2>
  <div class="scatter-wrapper">
    {scatter_svg}
  </div>
  {legend_html}
</section>

<section>
  <h2>All results</h2>
  {table_html}
</section>

<footer>
  <p>
    Benchmarks run on real Apple Silicon hardware (M2 Pro / macOS 14).
    All runtimes tested with the same ONNX model file and synthetic inputs.
    Example data is labeled with a <code>_note</code> field in the JSON.
    &mdash; <a href="https://github.com/rlaope/airML/tree/master/bench" rel="noopener">How to reproduce</a>
  </p>
</footer>

</div>
</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing JSON result files (default: bench/results/ relative to this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write index.html into (default: bench/site/dist/).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_dir = args.results_dir or (script_dir.parent / "results")
    output_dir  = args.output_dir  or (script_dir / "dist")

    if not results_dir.exists():
        print(f"[error] results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(results_dir)
    if not records:
        print(f"[warn] no valid JSON results found in {results_dir}", file=sys.stderr)

    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(records, generated_at)

    out_path = output_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out_path} ({len(html):,} bytes, {len(records)} results)")


if __name__ == "__main__":
    main()
