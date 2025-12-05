# Production Cron Automation – Portfolio Maximizer v45

**Last updated**: 2025-11-19  
**Scope**: Linux/Unix cron wiring for production‑style automation, aligned with:
- `Documentation/arch_tree.md`
- `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`
- `Documentation/implementation_checkpoint.md`

This document describes how to schedule the core ETL/trading/monitoring
tasks using a single entrypoint script: `bash/production_cron.sh`.

---

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
- `ticker_discovery_stub` – Placeholder for future Phase 5.2 ticker discovery.
- `optimizer_stub` – Placeholder for future Phase 5.3 optimizer pipeline.

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

### 4.3 `nightly_backfill` (stub)

Intended to support:

- `Documentation/implementation_checkpoint.md` – nightly signal validation backfill.
- `scripts/backfill_signal_validation.py` – currently under modernization to use
  timezone‑aware timestamps and updated SQLite adapters.

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

Two read-only helpers can be wired into cron to keep TS thresholds and friction assumptions grounded in recent data:

- `scripts/sweep_ts_thresholds.py` – summarises realised performance over a grid of `(confidence_threshold, min_expected_return)` values per ticker.
- `scripts/estimate_transaction_costs.py` – estimates commission / transaction costs by ticker or simple asset class buckets.

Example entries (adjust paths/schedules as needed):

```cron
# 8. Weekly TS threshold sweep (uses realised trades in trade_executions)
0 4 * * 1 cd /opt/portfolio_maximizer_v45 && \
  simpleTrader_env/bin/python scripts/sweep_ts_thresholds.py \
    --lookback-days 365 \
    --grid-confidence "0.50,0.55,0.60" \
    --grid-min-return "0.001,0.002,0.003" \
    --min-trades 10 \
    --output logs/automation/ts_threshold_sweep.json \
    >> logs/cron/ts_threshold_sweep.out 2>&1

# 9. Monthly transaction cost estimation (per asset class)
15 4 1 * * cd /opt/portfolio_maximizer_v45 && \
  simpleTrader_env/bin/python scripts/estimate_transaction_costs.py \
    --lookback-days 365 \
    --grouping asset_class \
    --min-trades 5 \
    --output logs/automation/transaction_costs.json \
    >> logs/cron/transaction_costs.out 2>&1
```

Both scripts:
- Only read from the existing SQLite database; they do not modify configs.
- Emit machine-readable JSON under `logs/automation/` for use by higher-level tooling (e.g., a proposal → config diff helper or notebooks).

---

## 7. Quick Start Checklist

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
