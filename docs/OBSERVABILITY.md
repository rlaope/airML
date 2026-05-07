# Observability

airML emits structured logs via `tracing` and Prometheus metrics for all inference calls.

## Logs

By default, logs print to stderr in human-readable text format. For production:

    airml --log-format=json --log-level=info serve

Override the level via the `RUST_LOG` env var (takes precedence over `--log-level`):

    RUST_LOG=airml_core=debug,airml_hub=info airml run -m bge-small-en

### Available flags

| Flag | Default | Description |
|---|---|---|
| `--log-format` | `text` | `text` (human-readable) or `json` (structured) |
| `--log-level` | `info` | `trace`, `debug`, `info`, `warn`, or `error` |

Both flags are global and can be placed before any subcommand.

## Metrics

When `airml serve` is running, metrics are exposed at:

    GET http://localhost:8080/metrics

Format is Prometheus text (version 0.0.4). Available metrics:

| Metric | Type | Description |
|---|---|---|
| `airml_inference_total` | counter | Total inference calls |
| `airml_inference_latency_seconds` | histogram | Per-call inference latency |
| `airml_model_load_seconds` | histogram | Model load time |

### Sample Prometheus scrape config

```yaml
scrape_configs:
  - job_name: airml
    static_configs:
      - targets: ['localhost:8080']
```

## Grafana dashboard

A starter dashboard JSON ships at `deploy/grafana-dashboard.json`.

Import it via **Dashboards → Import → Upload JSON file** in Grafana 11+.
It covers all three metrics with stat panels (totals) and time series (latency histograms).
