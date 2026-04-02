# Institutional Workflow Runbook (ETL -> Forecasting -> Signals -> Execution -> Reporting)

> Runtime baseline: WSL + `simpleTrader_env` only.  
> Reference: `Documentation/RUNTIME_GUARDRAILS.md`.

This runbook is the canonical operating path for institutional-grade unattended workflows.

Related navigator:
- [DOCS_INDEX.md](DOCS_INDEX.md)

## 1) Preflight

1. Enter runtime:

```bash
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/portfolio_maximizer_v45
source simpleTrader_env/bin/activate
```

2. Record runtime fingerprint:

```bash
which python
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

3. Record repo state:

```bash
git status --porcelain
```

4. Freeze config scope for the run:
- `config/pipeline_config.yml`
- `config/forecasting_config.yml`
- `config/signal_routing_config.yml`
- `config/forecaster_monitoring.yml`
- `config/quant_success_config.yml`

## 2) Blocking Gate Baseline (Must Pass For Unattended Readiness)

Run:

```bash
python scripts/institutional_unattended_gate.py --json
python scripts/run_all_gates.py --json
```

Interpretation:
- If both pass: unattended-run baseline is green.
- If `institutional_unattended_gate` fails: fix hardening contracts first.
- If `run_all_gates` fails only on `production_audit_gate`: hardening is intact, but unattended readiness is blocked by lift/profitability policy.

Do not use `run_all_gates` skip flags for final readiness evidence.

## 3) Standard Execution Profiles

### A) End-to-end orchestrated run (recommended)

```bash
bash bash/run_end_to_end.sh
```

### B) Pipeline-only run

```bash
bash bash/run_pipeline.sh
```

### C) Auto-trader paper run

```bash
bash bash/run_auto_trader.sh
```

Notes:
- Auto-trader resumes persisted positions by default.
- To force fresh session: `NO_RESUME=1 bash bash/run_auto_trader.sh`.
- To reset persisted state: `bash/reset_portfolio.sh`.
- One-time migration for older DBs: `python scripts/migrate_add_portfolio_state.py`.

## 4) Post-Run Institutional Checks

```bash
python scripts/check_quant_validation_health.py
python scripts/check_forecast_audits.py --config-path config/forecaster_monitoring.yml
python scripts/production_audit_gate.py
```

Required artifacts to archive:
- `logs/audit_gate/production_gate_latest.json`
- `visualizations/dashboard_data.json`
- `visualizations/dashboard_snapshot.png` (if generated)
- Run logs under `logs/automation/` and/or `logs/pipeline_runs/`
- Forecast audits under `logs/forecast_audits/`

## 5) Lift-Gate Failure Triage

When `production_audit_gate` fails with lift failure:

1. Confirm effective audit count and violation profile:

```bash
python scripts/check_forecast_audits.py --config-path config/forecaster_monitoring.yml
```

2. Refresh and inspect gate decision context:

```bash
python scripts/production_audit_gate.py --allow-inconclusive-lift
```

3. Treat result as policy block, not hardening regression, unless institutional gate also fails.

## 6) Contract Test Set (Before Commit/Promotion)

```bash
python -m pytest tests/scripts/test_institutional_unattended_contract.py tests/scripts/test_institutional_unattended_gate.py tests/scripts/test_llm_runtime_install_policy.py tests/scripts/test_platt_calibration_contract.py tests/scripts/test_run_all_gates.py -q
python -m pytest -m "not gpu and not slow" --tb=short -q
```

## 7) Dashboard Serving

Windows manual launch:

```powershell
python scripts/windows_dashboard_manager.py launch
```

or:

```powershell
.\launch_live_dashboard.bat
```

This path refreshes `visualizations/dashboard_data.json` first, forces a fresh `production_gate_latest.json` once, then ensures the bridge, localhost HTTP server, and Prometheus exporter are running. The bridge continues to refresh the gate artifact when it becomes stale. The live watcher is available when explicitly requested with `--ensure-live-watcher`.

Serve repository root over HTTP:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/portfolio_maximizer_v45
```

Open:
- `http://127.0.0.1:8000/visualizations/live_dashboard.html`

Expected payload source:
- `visualizations/dashboard_data.json`

If dashboard shows missing prices/trades, verify producer-side emission from latest run before UI debugging.
