# PMX Prometheus + Grafana Observability

## Scope

This rollout adds a read-only observability sidecar for PMX on the same Windows host:

- dedicated Prometheus exporter at `127.0.0.1:9765`
- Alertmanager bridge at `127.0.0.1:9766`
- Prometheus at `127.0.0.1:9090`
- Alertmanager at `127.0.0.1:9093`
- Grafana at `127.0.0.1:3000`

This observability lane does **not** replace:

- `visualizations/live_dashboard.html`
- `visualizations/dashboard_data.json`
- business/trading dashboards
- existing trade/order execution logic

It complements operator monitoring only.

## Contract

- Metric prefix: `pmx_`
- Allowed labels in v1 only:
  - `severity`
  - `component`
  - `channel`
  - `job`
- Disallowed high-cardinality labels:
  - `run_id`
  - `message_id`
  - raw target ids
  - full ticker lists
  - per-session ids

Status code conventions:

- generic status metrics: `0=PASS/OK`, `1=WARN`, `2=FAIL`, `3=MISSING/ERROR`
- production gate status metrics: `0=READY/PASS`, `1=WARN/INCONCLUSIVE`, `2=FAIL/RED`, `3=MISSING/ERROR`
- model improvement status metrics: `0=PASS`, `1=WARN`, `2=FAIL`, `3=SKIP`, `4=ERROR`

## Data Sources

The exporter is read-only and uses cached collectors:

- every `15s`
  - `visualizations/dashboard_data.json`
  - `visualizations/performance/metrics_summary.json`
  - `logs/audit_gate/production_gate_latest.json`
  - `logs/automation/openclaw_maintenance_latest.json`
- every `30s`
  - `python scripts/openclaw_remote_workflow.py health --json`
  - `%USERPROFILE%\.openclaw\cron\jobs.json`
  - SQLite read check on `data/portfolio_maximizer.db`
- every `300s`
  - `python scripts/check_model_improvement.py --json`

Missing or corrupt inputs produce warnings and stale/unknown metrics. They must not make the exporter return HTTP 500 for normal scrape traffic.

## Initial Metrics

- `pmx_openclaw_gateway_up`
- `pmx_openclaw_primary_channel_up`
- `pmx_openclaw_channels_status_latency_ms`
- `pmx_openclaw_recovery_events_total`
- `pmx_cron_job_last_success_unixtime{job=...}`
- `pmx_cron_job_consecutive_errors{job=...}`
- `pmx_dashboard_snapshot_age_seconds`
- `pmx_production_gate_pass`
- `pmx_production_gate_status_code`
- `pmx_proof_runway_closed_trades`
- `pmx_proof_runway_remaining_days`
- `pmx_sqlite_health_ok`

Supporting metrics are also emitted for timestamps, cron thresholds, and model-improvement state to support alerting and Grafana panels without scraping live scripts directly.

## Alert Ownership

Default mode is **shadow mode**.

- Alertmanager sends webhooks to `scripts/pmx_alertmanager_bridge.py`
- Bridge default: `PMX_OBSERVABILITY_ALERT_SHADOW_MODE=1`
- In shadow mode:
  - if `PMX_OBSERVABILITY_OPENCLAW_SHADOW_TARGETS` is set, OpenClaw shadow notifications are sent there
  - if `PMX_EMAIL_TEST_TO` is set, email shadow notifications are sent there
  - otherwise, alerts are logged only

This prevents duplicate/noisy production paging during rollout.

## Windows Startup

Repo-owned startup/provisioning scripts:

- `scripts/start_pmx_observability_exporter.ps1`
- `scripts/start_pmx_alertmanager_bridge.ps1`
- `scripts/start_prometheus.ps1`
- `scripts/start_alertmanager.ps1`
- `scripts/start_grafana.ps1`
- `scripts/start_observability_stack.ps1`
- `scripts/stop_observability_stack.ps1`
- `scripts/status_observability_stack.ps1`
- `scripts/install_observability_stack.ps1`

Installer behavior:

- creates required repo-local data/log directories
- downloads repo-pinned official Windows zip archives when `-DownloadOfficialBinaries` is supplied without placeholders
- still allows explicit URL overrides for pinned/manual installs
- installs a per-user Startup shortcut:
  - `PMX-Observability-Stack.cmd`

Recommended install from PowerShell:

```powershell
Set-Location C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45
& .\scripts\install_observability_stack.ps1 -DownloadOfficialBinaries
```

Operator controls:

```powershell
& .\scripts\start_observability_stack.ps1
& .\scripts\stop_observability_stack.ps1
& .\scripts\status_observability_stack.ps1
```

`start_observability_stack.ps1` is idempotent: healthy services are detected and skipped instead of starting duplicate processes.
`stop_observability_stack.ps1` first requests localhost `/shutdown` for the Python sidecars, then falls back to process termination for any service still holding a port.
If startup reports `already_healthy_legacy` for a Python sidecar, the service is healthy but was started from an older build that does not expose graceful `/shutdown` yet; the next clean restart will move it onto the current contract.
`status_observability_stack.ps1 -Json` is the canonical way to see whether each service is `healthy`, whether a sidecar is `current` vs `legacy`, and which listener PIDs currently own the stack ports.

## Grafana Role

Grafana is ops/SRE-only in this rollout.

Provisioned dashboards:

- `PMX OpenClaw Health`
- `PMX Scheduler Health`
- `PMX Artifact Freshness`
- `PMX Gate State`

Grafana alerting is disabled in `observability/grafana/grafana.ini`. Alertmanager is the only new alert evaluator in v1.

## Success Metrics

Use these metrics to prove the observability rollout is helping:

- mean time to detect gateway/channel degradation `< 60s`
- mean time to recover after startup/wake improves by `>= 50%`
- cron freshness lag stays `< 5m` for required jobs
- dashboard snapshot age stays within `2x` expected refresh cadence
- `channels.status` p95 latency trends are visible and actionable
- unknown-cause operator incidents decrease by `>= 50%`

## Out of Scope

- attaching `/metrics` to `scripts/pmx_interactions_api.py`
- replacing `live_dashboard.html`
- Kafka integration
- Airflow integration
- adding Prometheus/Grafana logic to the trade path
