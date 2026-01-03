# Production Cron Automation - Portfolio Maximizer v45

**Last updated**: 2026-01-03  
**Scope**: Linux/Unix cron wiring for production‑style automation, aligned with:
- `Documentation/arch_tree.md`
- `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`
- `Documentation/implementation_checkpoint.md`

This document describes how to schedule the core ETL/trading/monitoring
tasks using a single entrypoint script: `bash/production_cron.sh`.

---

## 0. Windows / WSL Equivalent (Evidence Freshness)

For Windows hosts, the recommended operational equivalent to cron is:

- Windows Task Scheduler calls `schedule_backfill.bat` (repo root).
- `schedule_backfill.bat` defaults to `auto_trader_core` and invokes WSL:
  - `bash/production_cron.sh auto_trader_core`

This keeps time-series evidence building (core tickers + gating) reproducible on Windows while still using the Linux-style cron multiplexer as the single entry point.

## 1. Design Principles

- **Single entrypoint**: All cron jobs call `bash/production_cron.sh` with a task name.
- **Virtualenv‑first**: Uses the authorised `simpleTrader_env` interpreter only.
- **Config‑driven**: Respects existing YAML configs (pipeline, data sources, validation).
- **Safe defaults**: Read‑only monitoring and paper trading by default; no live execution
  unless explicitly configured in `config/` and environment variables.
- **Phase alignment**:
  - Phase 4.x: ETL, CV, checkpointing, logging.
  - Phase 5.2–5.5: LLM integration, monitoring, backfill.
  - Phase 5.7–5.9: Time series signal generation + autonomous profit engine.
  - Future Phase 5.x: Ticker discovery and optimizer cron stubs.

---

## 2. Cron Multiplexer Script

The cron wiring is centralised in:

- `bash/production_cron.sh`

Key tasks (first positional argument):

- `daily_etl` – Full ETL pipeline (default: live mode).
- `auto_trader` – Autonomous paper‑trading loop (`scripts/run_auto_trader.py`).
- `nightly_backfill` – Signal validation backfill (stub until modernised).
- `monitoring` – LLM/pipeline health + latency monitoring.
- `env_sanity` – Environment validation before trading hours.
- `ts_threshold_sweep` – TS threshold sweep over realised trades; writes JSON to `logs/automation/`.
- `transaction_costs` – Transaction cost estimation grouped by ticker/asset class to `logs/automation/`.
- `auto_trader_core` – Core tickers with trade-count gate (defaults: AAPL,MSFT,GC=F,COOP; stops once ≥30 total and ≥10 per-ticker closed trades).
- `ticker_discovery_stub` – Placeholder for future Phase 5.2 ticker discovery.
- `optimizer_stub` – Placeholder for future Phase 5.3 optimizer pipeline.
- `weekly_sleeve_maintenance` – Sleeve summary + promotion/demotion plan writer (see `bash/weekly_sleeve_maintenance.sh`).
- `synthetic_refresh` – Generate a synthetic dataset (config-driven) for offline regression/smoke testing; respects `CRON_SYNTHETIC_*` env overrides.

The script automatically:

- Resolves `PROJECT_ROOT` from its own location.
- Locates `simpleTrader_env/bin/python` (or `Scripts/python.exe` on Windows‑style layouts).
- Falls back to `python3`/`python` only if the venv binary is not available.
- Writes per‑task log files under `logs/cron/`, one file per invocation.

---

## 3. Example Crontab Entries

Assuming the repo lives at `/opt/portfolio_maximizer_v45`:

```cron
# ┌─ minute (0 - 59)
# │ ┌─ hour (0 - 23)
# │ │ ┌─ day of month (1 - 31)
# │ │ │ ┌─ month (1 - 12)
# │ │ │ │ ┌─ day of week (0 - 6) (Sunday to Saturday)
# │ │ │ │ │
# │ │ │ │ │

# 1. Daily ETL refresh (pre‑market)
15 5 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh daily_etl >> logs/cron/daily_etl.out 2>&1

# 2. Autonomous paper trader loop (every 30 minutes during market hours)
*/30 7-20 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh auto_trader --config config/pipeline_config.yml \
  >> logs/cron/auto_trader.out 2>&1

# 2b. Core ticker accumulation loop (halts once targets hit: ≥30 total, ≥10 per core ticker)
*/60 7-20 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  CRON_CORE_TICKERS="AAPL,MSFT,GC=F,COOP" \
  bash/bash/production_cron.sh auto_trader_core >> logs/cron/auto_trader_core.out 2>&1

# 3. Nightly signal validation backfill (stub, see Section 5)
5 2 * * * cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh nightly_backfill >> logs/cron/nightly_backfill.out 2>&1

# 4. Hourly monitoring & latency checks
5 * * * * cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh monitoring >> logs/cron/monitoring.out 2>&1

# 5. Pre‑open environment sanity check
0 5 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh env_sanity >> logs/cron/env_sanity.out 2>&1

# 6. Weekly ticker discovery stub (Phase 5.2+)
0 3 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh ticker_discovery_stub >> logs/cron/ticker_discovery.out 2>&1

# 7. Weekly optimizer stub (Phase 5.3+)
30 3 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh optimizer_stub >> logs/cron/optimizer_stub.out 2>&1

# 8. Weekly TS threshold sweep (uses realised trades in trade_executions)
0 4 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh ts_threshold_sweep >> logs/cron/ts_threshold_sweep.out 2>&1

# 9. Monthly transaction cost estimation (per asset class)
15 4 1 * * cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh transaction_costs >> logs/cron/transaction_costs.out 2>&1

# 10. Weekly sleeve summary + promotion/demotion recommendations
0 5 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/weekly_sleeve_maintenance.sh >> logs/cron/sleeve_maintenance.out 2>&1

# 11. Synthetic dataset refresh (offline regression)
0 1 * * 1 cd /opt/portfolio_maximizer_v45 && \
  CRON_SYNTHETIC_CONFIG="config/synthetic_data_config.yml" \
  CRON_SYNTHETIC_TICKERS="AAPL,MSFT" \
  bash/bash/production_cron.sh synthetic_refresh >> logs/cron/synthetic_refresh.out 2>&1
```

You can edit the schedule and paths to match your deployment environment.

---

## 4. Task Details

### 4.1 `daily_etl`

Maps to the core ETL stages documented in:

- `Documentation/arch_tree.md` – Phase 1–4.8 (ETL, validation, preprocessing, storage, CV).
- `scripts/run_etl_pipeline.py` – configurably runs:
  `data_extraction → data_validation → data_preprocessing → data_storage → time_series_forecasting → time_series_signal_generation → signal_router`.

Defaults (overridable via environment):

- `CRON_TICKERS` (default: `AAPL,MSFT,GOOGL`)
- `CRON_START_DATE` (default: `2020-01-01`)
- `CRON_END_DATE` (default: today)
- `CRON_EXEC_MODE` (default: `live`, may also be `synthetic` for offline regression)

### 4.2 `auto_trader`

Drives the “Autonomous Profit Engine” loop described in:

- `Documentation/arch_tree.md` (Week 5.9: Autonomous Profit Engine roll‑out).
- `scripts/run_auto_trader.py` – orchestrates:
  extraction → validation → forecasting → TS signal generation → routing → paper trading.

Typical cron usage is every N minutes during market hours, as a
stateless trigger into the auto‑trader, which internally respects
the configured risk and validation gates.

### 4.2b `auto_trader_core`

Same as `auto_trader` but with defaults targeted at the core tickers (`AAPL,MSFT,GC=F,COOP`) and a built‑in trade-count gate:

- Skips execution once **both** conditions are met in `trade_executions`:
  - Total closed trades ≥ `CRON_CORE_TOTAL_TARGET` (default: 30)
  - Per-core-ticker closed trades ≥ `CRON_CORE_PER_TICKER_TARGET` (default: 10)
- Env overrides:
  - `CRON_CORE_TICKERS` (comma list)
  - `CRON_CORE_DB_PATH` (default: `data/portfolio_maximizer.db`)
  - `CRON_CORE_TOTAL_TARGET`, `CRON_CORE_PER_TICKER_TARGET`

### 4.3 `nightly_backfill`

- `Documentation/implementation_checkpoint.md` – nightly signal validation backfill.
- `scripts/backfill_signal_validation.py` – executed via Task Scheduler (`PortfolioMaximizer_BackfillSignals`) and available for manual runs.
- Manual trigger: `bash/run_backfill.sh` (uses `simpleTrader_env/bin/python3` when present, logs to `logs/automation/backfill_<timestamp>.log`).

Current behaviour:

- If `scripts/backfill_signal_validation.py` exists, it is called with any extra args.
- If not present or not yet modernised, the job logs a stub message and exits cleanly.

### 4.4 `monitoring`

Aligns with:

- `Documentation/arch_tree.md` – Week 5.5 & 5.9 (Error Monitoring & Performance Optimization, Monitoring + Nightly Backfill Instrumentation).
- `scripts/monitor_llm_system.py` – logs LLM latency benchmarks and backtest summaries.

Run this hourly (or more frequently) to maintain a live view of:

- LLM latency and failover behaviour.
- Signal backtest statistics (if configured).

### 4.5 `env_sanity`

Lightweight guard rail before trading hours:

- Calls `scripts/validate_environment.py` when present.
- Intended checks: Python version, venv presence, configuration files,
  basic DB connectivity, and secret loading (see SECURITY* docs).

### 4.6 Future Stubs

Per `Documentation/arch_tree.md`:

- `ticker_discovery_stub`
  - Target: Phase 5.2 Ticker Discovery integration (`etl/ticker_discovery.*`).
  - Future behaviour: periodically refresh the ticker universe from Alpha Vantage,
    apply validation, and persist to a universe table for the optimizer.

- `optimizer_stub`
  - Target: Phase 5.3 Optimizer Pipeline (`run_optimizer_pipeline`).
  - Future behaviour: periodically run portfolio optimization across the
    validated universe using the enhanced `etl/portfolio_math.py` and
    a forthcoming `etl/portfolio_selection.py`.

Both stubs currently just write a one‑line log entry so you can safely
keep their cron entries in place while implementation progresses.

### 4.7 `synthetic_refresh`

- Generates a synthetic dataset via `scripts/generate_synthetic_dataset.py` (respects `CRON_SYNTHETIC_CONFIG`, `CRON_SYNTHETIC_TICKERS`, `CRON_SYNTHETIC_OUTPUT_ROOT`) and immediately validates it with `scripts/validate_synthetic_dataset.py`.
- Produces `data/synthetic/<dataset_id>/<ticker>.parquet` + `manifest.json` and a validation report under `logs/automation/`.
- Enablement is synthetic‑first only; keep live trading disabled. Promotion of synthetic outputs to live cron tasks requires GREEN/acceptable YELLOW quant health per `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` and the sequencing rules in `Documentation/NEXT_TO_DO_SEQUENCED.md`.

### 4.8 `sanitize_caches`

- Prunes cached data/log artifacts older than `CRON_SANITIZE_RETENTION` (default: 14 days).
- Invokes `scripts/sanitize_cache_and_logs.py`, targeting common data/log paths while skipping DVC stores by default.
- Override targets/patterns with:
  - `CRON_SANITIZE_DATA_DIRS` (comma-separated)
  - `CRON_SANITIZE_LOG_DIRS` (comma-separated)
  - `CRON_SANITIZE_PATTERNS` (comma-separated glob patterns)

---

## 5. Safety & Operational Notes

- **Database path**: Production tasks default to `data/portfolio_maximizer.db`.
  For synthetic/brutal runs or staging, override via `PORTFOLIO_DB_PATH`
  when invoking the pipeline (as already done in the brutal test harness).
- **LLM gating**: Time Series is the canonical stack (see
  `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`). LLM‑heavy
  jobs (e.g., `monitor_llm_system.py` with full LLM usage, or
  `run_auto_trader.py` with LLM enabled) should only be enabled after:
  - Profit‑critical tests are green (`tests/integration/test_profit_critical_functions.py`).
  - LLM performance tests are passing and latency within budget.
- **Brutal test suite**: For periodic deep validation, use:
  - `bash/bash/comprehensive_brutal_test.sh` (manual or via a separate cron).
  - This is intentionally not wired into the default production cron
    because it can run for hours and is best treated as a maintenance job.

---

## 6. Quant Threshold Sweeps & Cost Estimation

Two read-only helpers are now exposed through the cron multiplexer to keep TS thresholds and friction assumptions grounded in recent data:

- `scripts/sweep_ts_thresholds.py` – summarises realised performance over a grid of `(confidence_threshold, min_expected_return)` values per ticker (task: `ts_threshold_sweep`).
- `scripts/estimate_transaction_costs.py` – estimates commission / transaction costs by ticker or simple asset class buckets (task: `transaction_costs`).

Env overrides for these tasks:

```bash
# TS sweep overrides (defaults: 365-day lookback, 0.50/0.55/0.60 confidence grid, 0.001/0.002/0.003 min_return grid)
CRON_TS_SWEEP_TICKERS="AAPL,MSFT,GC=F,COOP" \
CRON_TS_SWEEP_CONFIDENCE="0.50,0.55,0.60,0.65" \
CRON_TS_SWEEP_MIN_RETURN="0.001,0.002,0.003,0.004" \
CRON_TS_SWEEP_MIN_TRADES=10 \
CRON_TS_SWEEP_OUTPUT="logs/automation/ts_threshold_sweep.json" \
  bash/bash/production_cron.sh ts_threshold_sweep

# Transaction cost overrides (defaults: 365-day lookback, asset_class grouping, min_trades=5)
CRON_COST_AS_OF="2025-12-05" \
CRON_COST_GROUPING="asset_class" \
CRON_COST_MIN_TRADES=5 \
CRON_COST_OUTPUT="logs/automation/transaction_costs.json" \
  bash/bash/production_cron.sh transaction_costs
```

Both scripts:
- Only read from the existing SQLite database; they do not modify configs.
- Emit machine-readable JSON under `logs/automation/` for use by higher-level tooling (e.g., a proposal → config diff helper or notebooks).

Helper to run the full chain (costs + TS sweep + config proposals) manually:

```bash
bash/bash/run_ts_sweep_and_proposals.sh

# Optional overrides
SWEEP_TICKERS="AAPL,MSFT,GC=F,COOP" \
SWEEP_SEL_MIN_PF=1.2 \
SWEEP_SEL_MIN_WR=0.55 \
COST_GROUPING="asset_class" \
PROPOSALS_OUTPUT="logs/automation/config_proposals.json" \
  bash/bash/run_ts_sweep_and_proposals.sh
```

### 6.1 Automation Dashboard Glue

To surface a unified “what should we change next?” view for humans and agents:

- `scripts/build_automation_dashboard.py`
  - Reads (when present):
    - `logs/automation/ts_threshold_sweep.json`
    - `logs/automation/transaction_costs.json`
    - `logs/automation/sleeve_summary.json`
    - `logs/automation/sleeve_promotion_plan.json`
    - `logs/automation/config_proposals.json`
    - best cached strategy config from `strategy_configs` (via `DatabaseManager`)
  - Writes:
    - `visualizations/dashboard_automation.json`
  - This file is read-only with respect to configs; it consolidates evidence for:
    - TS threshold tuning,
    - friction assumptions,
    - sleeve promotion/demotion,
    - higher-order strategy optimization output.

Example cron wiring (run after the other automation tasks have completed):

```bash
0 6 * * 1-5 cd /path/to/portfolio_maximizer_v45 && \
  "${PYTHON_BIN:-python}" scripts/build_automation_dashboard.py \
    --db-path "data/portfolio_maximizer.db" \
    --output "visualizations/dashboard_automation.json"
```

**Diagnostics caution**: `scripts/liquidate_open_trades.py`/`bash/force_close_and_sweep.sh` are evidence-gathering utilities that force-close trades with synthetic mark-to-market PnL. They are NOT suitable for real PnL reporting; use only to unblock sweeps in diagnostic workflows.

---

## 7. TS Model Search & Config Proposals

Time-series model search and configuration proposals are intentionally **decoupled** from the main cron loop. They are heavier, research-grade jobs that should normally be run:

- On demand (manual CLI) when you are actively evaluating TS models, or
- As low-frequency cron jobs (e.g., weekly) during research cycles.

### 7.1 Manual Invocations

Run a compact TS model search for a handful of tickers, persisting candidates into `ts_model_candidates`:

```bash
cd /opt/portfolio_maximizer_v45

# Example: rolling CV over a small SARIMAX/SAMOSSA grid for core tickers
simpleTrader_env/bin/python scripts/run_ts_model_search.py \
  --tickers "AAPL,MSFT,GC=F,COOP" \
  --lookback-days 730 \
  --horizon 5 \
  --step-size 20 \
  --max-folds 8 \
  --use-profiles \
  --db-path "data/portfolio_maximizer.db"

# Summarise best candidates per (ticker, regime)
simpleTrader_env/bin/python scripts/summarize_ts_candidates.py \
  --db-path "data/portfolio_maximizer.db" \
  --output "logs/automation/ts_model_candidates_summary.json"

# Generate advisory config proposals driven by stability + DM p-values
simpleTrader_env/bin/python scripts/generate_ts_model_config_proposals.py \
  --db-path "data/portfolio_maximizer.db" \
  --min-stability 0.4 \
  --max-dm-pvalue 0.10 \
  --output "logs/automation/ts_model_config_proposals.json"
```

Semantics:

- All three scripts are **read-only** with respect to configs; they write only to:
  - `ts_model_candidates` (SQLite table),
  - `logs/automation/ts_model_candidates_summary.json`,
  - `logs/automation/ts_model_config_proposals.json`.
- Any changes to `config/model_profiles.yml` or future TS override configs remain **human-reviewed**; proposals are advisory only.

### 7.2 Example Weekly Cron Wiring (Research Environment)

For a research/staging environment (not production), you may add a low-frequency cron block such as:

```cron
# Weekly TS model search + proposals (research-only)
0 2 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/bash/production_cron.sh ts_model_search >> logs/cron/ts_model_search.out 2>&1
```

Where `ts_model_search` is a small wrapper task in `bash/production_cron.sh` that:

1. Calls `scripts/run_ts_model_search.py` with your preferred ticker list and CV settings.
2. Calls `scripts/summarize_ts_candidates.py` to refresh `ts_model_candidates_summary.json`.
3. Calls `scripts/generate_ts_model_config_proposals.py` to refresh `ts_model_config_proposals.json`.

This keeps TS model search aligned with the same cron/automation patterns as threshold sweeps and cost estimation, while preserving a clear separation between **evidence generation** and **config changes**.

---

## 8. Quick Start Checklist

1. Ensure `simpleTrader_env` exists and is up to date:
   - `python3 -m venv simpleTrader_env`
   - `simpleTrader_env/bin/pip install -r requirements.txt`
2. Verify core scripts:
   - `scripts/run_etl_pipeline.py`
   - `scripts/run_auto_trader.py`
   - `scripts/monitor_llm_system.py` (optional but recommended)
3. Test manually:
   - `bash/bash/production_cron.sh daily_etl`
   - `bash/bash/production_cron.sh auto_trader --dry-run` (if supported)
4. Install crontab entries from Section 3, adjusting paths and schedules.
5. Monitor:
   - `logs/cron/*.log` for cron‑level output.
   - `logs/pipeline_run.log`, `logs/events/events.log`,
     and monitoring logs for deeper diagnostics.
