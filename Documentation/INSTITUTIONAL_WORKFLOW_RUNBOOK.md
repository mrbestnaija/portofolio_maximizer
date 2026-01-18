# Institutional Workflow Runbook (ETL → Forecasting → Signals → Execution → Reporting)

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).
> Do not use Windows interpreters/venvs — results are invalid. See `Documentation/RUNTIME_GUARDRAILS.md`.

This runbook defines the **audit-grade**, end-to-end operating procedure for Portfolio Maximizer runs, from ingestion and preprocessing through forecasting, signal generation, execution simulation, and reporting artifacts.

## 0) Preconditions (Institutional Hygiene)

1) **Enter the correct runtime**

```bash
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45
source simpleTrader_env/bin/activate
```

2) **Record the required fingerprint (paste into logs/issues/PRs)**

```bash
which python
python -V
python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})"
```

3) **Freeze config scope for the run**

- Record which configs are in effect (at minimum):
  - `config/pipeline_config.yml`
  - `config/forecasting_config.yml`
  - `config/signal_routing_config.yml`
  - `config/forecaster_monitoring.yml`
  - `config/quant_success_config.yml`

4) **Decide execution mode**

- **Research/CI smoke**: synthetic allowed.
- **Evaluation**: live providers only; do not enable synthetic flags.

## 1) Primary “Orchestrated” Workflows (Recommended)

### A. ETL + forecasting pipeline (batch)

Use when you want data ingestion → validation → preprocessing → storage → forecasting artifacts.

```bash
bash bash/run_pipeline.sh
```

Expected artifacts (examples):
- Logs: `logs/pipeline_runs/*`
- Data outputs: `data/processed/*` (pipeline-dependent)
- Forecast audits (if enabled): `logs/forecast_audits/*`

### B. Auto-trader (paper) run + dashboard emission (audit surface)

Use when you want forecasting → signals → paper execution → dashboard outputs.

```bash
bash bash/run_auto_trader.sh
```

Expected artifacts:
- Execution logs: `logs/automation/*`
- Dashboard JSON/PNG: `visualizations/dashboard_data.json`, `visualizations/dashboard_snapshot.png`
- Forecast audits (if enabled): `logs/forecast_audits/*`

### C. End-to-end run (ETL + auto-trader)

Use when you want “single command” audit trails across ingestion and execution.

```bash
bash bash/run_end_to_end.sh
```

## 2) Live Dashboard (Audit UI)

The dashboard renders **only run artifacts** (no demo data) from:
- `visualizations/dashboard_data.json` (emitted by `scripts/run_auto_trader.py`)

Serve the repo over HTTP:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45
```

Open:
- `http://127.0.0.1:8000/visualizations/live_dashboard.html`

**Payload expectations (for full trade/price/PnL panels):**
- `meta.tickers`, `signals[]` (ticker, action, confidence, expected_return, source, quality)
- `price_series` (per-ticker close series)
- `trade_events` (BUY/SELL events + realized PnL fields)
- `positions` (open positions, if any)

If the dashboard shows `Price series: 0` or `Trades: 0`, the run did not emit those fields (or no trades were executed).

## 3) Secondary “Direct Script” Workflow (Fine-Grained)

Use when debugging or generating artifacts independently.

### A. ETL only

```bash
python3 scripts/run_etl_pipeline.py --help
```

### B. Auto-trader only

```bash
python3 scripts/run_auto_trader.py --help
```

### C. Monitoring / gates

- Forecast regression audits:
  - `python3 scripts/check_forecast_audits.py --help`
- Dashboard health:
  - `python3 scripts/check_dashboard_health.py --help`
- Quant validation health (global):
  - `python3 scripts/check_quant_validation_health.py --help`

### D. Reporting (LLM performance)

```bash
python3 scripts/generate_llm_report.py --period monthly --format json
```

## 4) Post-Run Review Checklist (Institutional)

1) Confirm the run produced the canonical audit artifacts:
- `visualizations/dashboard_data.json`
- `logs/…` (run log(s) for the orchestrator)
- `logs/forecast_audits/*` (when enabled)

2) Run health checks:
- `python3 scripts/check_dashboard_health.py`
- `python3 scripts/check_quant_validation_health.py`
- `python3 scripts/check_forecast_audits.py`

3) Archive evidence (recommended):
- Keep the run log, dashboard JSON/PNG, and any audit summaries with a timestamped folder under `reports/` or `logs/`.

## 5) Common Failure Modes (Fast Triage)

- **Dashboard loads but shows “dashboard_data.json not found”**:
  - You haven’t produced `visualizations/dashboard_data.json` yet (run `bash/run_auto_trader.sh`), or you’re serving the wrong directory.
- **Dashboard shows tickers/signals but `Price series: 0`**:
  - `price_series` is missing from `visualizations/dashboard_data.json`; this indicates a producer-side emission gap.
- **Dashboard shows `Trades: 0`**:
  - No trades executed (often due to thresholds/quant gates), or producer did not emit `trade_events`.
