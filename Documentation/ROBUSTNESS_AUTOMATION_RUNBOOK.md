# Robustness Automation Runbook

**Status**: Active  
**Mode**: Read-only, advisory only  
**Runtime**: `simpleTrader_env\Scripts\python.exe` (Python 3.12 on this host)

This runbook covers the quality/visualization automation layer that was added on
top of existing gates. It does not change trading thresholds, routing config, or
capital-readiness criteria.

## Guardrails

- All automation scripts in this layer are read-only.
- Helper modules are library-style only, not CLIs:
  - `scripts/robustness_thresholds.py`
  - `scripts/quality_pipeline_common.py`
- Thresholds are sourced from canonical gate/config files and must not drift.
- `scripts/robustness_thresholds.py` may depend on gate/config sources, but the
  gate sources must not import back into the automation helpers.
- Trade readers use production-trade scope only:
  - `production_closed_trades` when present
  - otherwise the equivalent closed/non-diagnostic/non-synthetic filter

## Runtime Expectations

- `scripts/run_quality_pipeline.py` should normally complete in under 5 minutes.
- `scripts/build_training_dataset.py` should normally complete in under 5 minutes.
- If either job consistently exceeds that, split or optimize the workload rather
  than weakening thresholds.
- Cron should preserve stdout/stderr (or let OpenClaw capture them) so WARN/ERROR
  diagnostics remain debuggable.

## Canonical Commands

Daily quality pipeline:

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\run_quality_pipeline.py --json
```

Nightly training curation:

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\build_training_dataset.py --json
```

Dashboard refresh:

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\dashboard_db_bridge.py --once
.\simpleTrader_env\Scripts\python.exe scripts\check_dashboard_health.py
```

## Status Semantics

`run_quality_pipeline.py`:

- `PASS`: all steps succeeded without warnings
- `WARN`: partial data, insufficient data, zero healthy tickers, or threshold
  provenance changed
- `ERROR`: a step failed to produce its core output

Strict fail-closed notes:

- Missing required chart artifacts are integrity failures in strict mode:
  - `scripts/generate_performance_charts.py` emits `chart_missing:<name>`
  - chart stage `status=ERROR`
  - `scripts/run_quality_pipeline.py` escalates pipeline to `ERROR`
- Dashboard robustness cannot remain `OK` when chart links are stale/missing:
  - `scripts/dashboard_db_bridge.py` downgrades to `WARN`
- `scripts/data_sufficiency_monitor.py` CLI contract is fixed:
  - `0=SUFFICIENT`, `1=INSUFFICIENT`, `2=DATA_ERROR`
  - non-finite metrics are `DATA_ERROR`

`build_training_dataset.py`:

- `PASS`: outputs produced or dry-run completed without fail-closed behavior
- `WARN`: fail-closed by design (for example zero `HEALTHY` tickers)
- `ERROR`: dependency or write failure

Fail-closed rule:

- If a valid eligibility artifact exists and contains zero `HEALTHY` tickers,
  curation must:
  - write a summary JSON sidecar
  - report `fail_closed=true`
  - return a non-zero exit code

This prevents silent zero-row “success”.

## Artifact Map

| Producer | Artifact | Path | Consumers |
|---|---|---|---|
| `compute_ticker_eligibility.py` | Eligibility JSON | `logs/ticker_eligibility.json` | `run_quality_pipeline.py`, dashboard bridge, training curation |
| `compute_context_quality.py` | Context quality JSON | `logs/context_quality_latest.json` | `run_quality_pipeline.py`, dashboard bridge, chart generation |
| `generate_performance_charts.py` | Metrics summary JSON | `visualizations/performance/metrics_summary.json` | dashboard bridge |
| `generate_performance_charts.py` | Chart PNGs | `visualizations/performance/*.png` | live dashboard links |
| `build_training_dataset.py` | Training summary JSON | `logs/training_dataset_latest.json` | OpenClaw cron / operators |
| `build_training_dataset.py` | Curated trades parquet | `data/training/trades_filtered.parquet` | training workflows |
| `build_training_dataset.py` | Curated audits parquet | `data/training/audits_filtered.parquet` | training workflows |
| `dashboard_db_bridge.py` | Dashboard payload JSON | `visualizations/dashboard_data.json` | `live_dashboard.html` |

## Threshold Provenance

Every produced analytics artifact should carry threshold provenance:

- `thresholds.source_paths`
- `thresholds.source_hashes`

If the previous artifact exists and the source hashes change, emit a warning:

- `threshold_source_hash_changed`

This is a visibility safeguard, not an automatic threshold change.

## Known Live-State Limitation

Current production data can still yield:

- `run_quality_pipeline.py -> WARN`
- `build_training_dataset.py -> fail_closed`

Typical causes:

- zero `HEALTHY` tickers in current eligibility evidence
- insufficient trade/readiness data
- partial context joins when `time_series_forecasts.ts_signal_id` is missing

These are data conditions, not fail-open behavior.

## OpenClaw Host Wiring Consistency

- `scripts/run_openclaw_maintenance.ps1` now enforces OpenClaw exec environment
  before maintenance in both Windows and WSL execution paths.
- Enforcement command:

```powershell
python scripts/enforce_openclaw_exec_environment.py
```

- Runtime health visibility:

```powershell
python scripts/project_runtime_status.py --pretty
```

`openclaw_exec_env` health signals:
- `invalid_exec_host`
- `invalid_sandbox_mode`
- `missing_acp_default_agent`
- `exec_env_valid`
