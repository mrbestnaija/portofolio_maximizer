# PMX Prometheus + Grafana + Loki Observability

## Scope

This rollout adds a read-only observability sidecar for PMX on the same Windows host:

- dedicated Prometheus exporter at `127.0.0.1:9765`
- Alertmanager bridge at `127.0.0.1:9766`
- Loki at `127.0.0.1:3100`
- Alloy log shipper at `127.0.0.1:12345`
- Prometheus at `127.0.0.1:9090`
- Alertmanager at `127.0.0.1:9093`
- Grafana at `127.0.0.1:3000`

This observability lane does **not** replace:

- `visualizations/live_dashboard.html`
- `visualizations/dashboard_data.json`
- business/trading dashboards
- existing trade/order execution logic

It complements operator monitoring and now mirrors the most important read-only PMX operating metrics into Grafana for faster operator decisions.

Logs are carried by Loki with Alloy as the repo-default shipper. This keeps log collection off the PMX app path and avoids adding Python logging/collector dependencies to the runtime environment. Promtail is intentionally not added to this repo-owned stack.

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

Log collection is also localhost-only:

- Alloy tails selected PMX/OpenClaw log files
- Loki stores those logs locally under `data/loki`
- Grafana reads logs through the provisioned Loki datasource

The shipped log set is intentionally narrow to reduce secret and noise risk:

- `logs/observability/*.log`
- `logs/run_audit/*.log`
- `logs/run_audit/*.txt`
- `logs/llm_activity/*.jsonl`
- `%USERPROFILE%\.openclaw\logs\*.jsonl`
- `%USERPROFILE%\.openclaw\logs\*.log`

Alloy keeps the Loki label set low-cardinality on purpose:

- static labels stay within `job` and `component`
- `channel` is extracted only from `logs/llm_activity/*.jsonl`
- `severity` is extracted only when a log line already exposes a stable level
- transient path labels such as `filename` are dropped before shipping to Loki
- oversized JSONL lines are truncated before write so Loki does not reject whole OpenClaw/LLM batches on Windows

Alloy boot behavior is intentionally split by feed:

- decision-critical feeds backfill from the start on first boot:
  - `logs/run_audit/*`
  - `logs/llm_activity/*`
  - `%USERPROFILE%\.openclaw\logs/*.jsonl`
- noisy infrastructure logs tail from the end:
  - `logs/observability/*`
  - `%USERPROFILE%\.openclaw\logs/*.log`

That keeps Grafana log boards useful immediately after install without flooding Loki with every historical infrastructure line on each restart.

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

Decision-grade PMX read-only metrics are also exported from `visualizations/dashboard_data.json`, including:

- current PnL and PnL percent
- trade count and win rate
- open-position counts and inferred exposure
- TS/LLM latency
- routing split and fallback usage
- forecaster RMSE/profit-factor/win-rate health and thresholds
- quant-validation fail fractions and thresholds
- operator-console issue, reconnect, short-circuit, and tool-call signals

Canonical dashboard contract:

- `visualizations/dashboard_data.json` is bridge-owned only
- `scripts/dashboard_db_bridge.py` is the sole canonical writer
- `scripts/run_auto_trader.py` now emits a producer artifact at `logs/automation/run_auto_trader_latest.json`
- the bridge reads that producer artifact for runtime-only supplements while keeping the canonical dashboard schema complete

Repo-owned required scheduler inventory:

- `config/observability_required_jobs.yml`
- each job declares `required_for_green`, `severity`, and `expected_cadence_seconds`
- the Prometheus exporter intersects runtime cron inventory with this file so only required jobs participate in green-bar freshness

## Alert Ownership

Default mode is **shadow mode**.

- Alertmanager sends webhooks to `scripts/pmx_alertmanager_bridge.py`
- Bridge default: `PMX_OBSERVABILITY_ALERT_SHADOW_MODE=1`
- In shadow mode:
  - if `PMX_OBSERVABILITY_OPENCLAW_SHADOW_TARGETS` is set, OpenClaw shadow notifications are sent there
  - if `PMX_EMAIL_TEST_TO` is set, email shadow notifications are sent there
  - otherwise, alerts are logged only

This prevents duplicate/noisy production paging during rollout.

Promotion sequence for strict green:

- Phase 1: shadow mode only
- Phase 2: 72-hour clean shadow soak with no required alerts firing continuously
- Phase 3: live delivery for P0 alerts only
- Phase 4: optional live P1 delivery after a second clean soak

## Windows Startup

Repo-owned startup/provisioning scripts:

- `scripts/start_pmx_observability_exporter.ps1`
- `scripts/start_pmx_alertmanager_bridge.ps1`
- `scripts/start_loki.ps1`
- `scripts/start_alloy.ps1`
- `scripts/start_prometheus.ps1`
- `scripts/start_alertmanager.ps1`
- `scripts/start_grafana.ps1`
- `scripts/start_observability_stack.ps1`
- `scripts/stop_observability_stack.ps1`
- `scripts/status_observability_stack.ps1`
- `scripts/install_observability_stack.ps1`

Installer behavior:

- creates required repo-local data/log directories
- downloads repo-pinned official Windows zip archives for Prometheus, Alertmanager, Grafana, Loki, and Alloy when `-DownloadOfficialBinaries` is supplied without placeholders
- still allows explicit URL overrides for pinned/manual installs
- installs a per-user Startup shortcut:
  - `PMX-Observability-Stack.cmd`

Recommended install from PowerShell:

```powershell
Set-Location <PROJECT_ROOT>
& .\scripts\install_observability_stack.ps1 -DownloadOfficialBinaries
```

Manual Loki ZIP import if you already downloaded the archive in a browser:

```powershell
Set-Location <PROJECT_ROOT>
& .\scripts\install_observability_stack.ps1 -LokiZipPath "$env:USERPROFILE\Downloads\loki-windows-amd64.exe.zip"
& .\scripts\start_loki.ps1
```

That imports the local ZIP into `tools\observability\loki` without redownloading the rest of the stack, and avoids the broken ad-hoc `C:\loki` workflow.

No-admin portable Alloy install if the Windows installer is blocked or times out:

```powershell
Set-Location <PROJECT_ROOT>
New-Item -ItemType Directory -Force -Path .\tools\observability\alloy | Out-Null
$alloyZip = Join-Path $env:TEMP "alloy-windows-amd64.exe.zip"
curl.exe -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C - "https://github.com/grafana/alloy/releases/download/v1.14.0/alloy-windows-amd64.exe.zip" -o $alloyZip
Expand-Archive -LiteralPath $alloyZip -DestinationPath .\tools\observability\alloy -Force
& .\scripts\start_alloy.ps1
```

`start_alloy.ps1` supports both install styles:

- portable repo-local binary under `tools\observability\alloy\alloy*.exe`
- Windows installer path under `%ProgramFiles%\GrafanaLabs\Alloy\alloy.exe`

`install_observability_stack.ps1` also accepts repo-local ZIP import flags:

- `-LokiZipPath "C:\path\to\loki-windows-amd64.exe.zip"`
- `-AlloyZipPath "C:\path\to\alloy-windows-amd64.exe.zip"`

Operator controls:

```powershell
& .\scripts\start_observability_stack.ps1
& .\scripts\stop_observability_stack.ps1
& .\scripts\status_observability_stack.ps1
& .\scripts\status_observability_stack.ps1 -RequireCurrent
```

`start_observability_stack.ps1` is idempotent: healthy services are detected and skipped instead of starting duplicate processes.
`stop_observability_stack.ps1` first requests localhost `/shutdown` for the Python sidecars, then falls back to process termination for any service still holding a port.
If startup reports `already_healthy_legacy` for a Python sidecar, the service is healthy but was started from an older build that does not expose graceful `/shutdown` yet; the next clean restart will move it onto the current contract.
`status_observability_stack.ps1 -Json` is the canonical way to see whether each service is `healthy`, whether a sidecar is `current` vs `legacy`, and which listener PIDs currently own the stack ports.
`status_observability_stack.ps1 -RequireCurrent` returns exit code `2` when the stack is healthy but still running legacy sidecars, and exit code `1` when any service is degraded.
If Loki/Alloy are not installed yet, the stack reports `status="partial"` with `optional_missing_count > 0` instead of failing the Prometheus/Alertmanager/Grafana core path.

Strict green entrypoint:

```powershell
python scripts/project_runtime_status.py --strict --pretty
```

Strict mode fails when:

- `production_gate` only passes through `INCONCLUSIVE_ALLOWED`
- `dashboard_data.json` is stale or missing required sections
- `logs/persistence_manager_status.json` is stale or not reconciled
- PMX runtime status and `status_observability_stack.ps1 -RequireCurrent` do not agree on green

## Grafana Role

Grafana is ops/SRE-only for alert ownership in this rollout. It remains read-only and out of the trade path, while also adding a unified PMX operator view built from the exporter cache.

Provisioned dashboards:

- `PMX OpenClaw Health`
- `PMX Scheduler Health`
- `PMX Artifact Freshness`
- `PMX Gate State`
- `PMX Unified Ops`
- `PMX Logs`

`PMX Unified Ops` is the decision-grade board for day-to-day PMX operations. It surfaces:

- PnL, win rate, trade count, open positions, and exposure
- latest signal confidence, expected return, and trade slippage
- signal counts, trade-event counts, and position-side mix
- proof runway and production-gate posture
- forecaster threshold health and RMSE ratio vs max
- quant-validation fail fractions vs max thresholds
- routing split, latency, and equity snapshots
- operator friction signals from the dashboard `operator_console` payload
- Loki-backed critical ops logs and recent PMX activity logs in the same dashboard
- Loki log-volume trends by job and interactive-channel activity over time
- Loki run-audit anomaly panels for gate, integrity, contamination, and orphaned-position triage

`PMX Logs` is the Grafana entry point for log triage. It shows:

- log volume by job and warning/error volume by job
- interactive channel activity counts from shipped `llm_activity` logs
- recent PMX logs across repo-owned jobs
- observability warnings/errors
- OpenClaw log flow in the same Grafana surface as the metrics dashboards

The intended operator workflow is now:

- use `PMX Unified Ops` for primary decision-making and incident triage
- drop into `PMX Logs` only when you want a wider log-only view

If `status_observability_stack.ps1 -Json` reports legacy Python sidecars, newly added exporter metrics may not appear in Grafana until those sidecars are restarted onto the current code. The dashboard JSON and Loki datasource will still be provisioned immediately, but the newest PMX metric panels depend on the refreshed exporter process.

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
