# Production Cron Automation - Portfolio Maximizer v45

**Last updated**: 2026-01-03  
**Scope**: Linux/Unix cron wiring for productionâ€‘style automation, aligned with:
- `Documentation/arch_tree.md`
- `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`
- `Documentation/implementation_checkpoint.md`

This document describes how to schedule the core ETL/trading/monitoring
tasks using a single entrypoint script: `bash/production_cron.sh`.

---

## 0. Windows / WSL Equivalent (Evidence Freshness)

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**  
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).  
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) â€” results are invalid.  
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

For Windows hosts, the recommended operational equivalent to cron is:

- Windows Task Scheduler calls `schedule_backfill.bat` (repo root).
- `schedule_backfill.bat` defaults to `auto_trader_core` and invokes WSL:
  - `bash/production_cron.sh auto_trader_core`

Optional daily automation:
- `bash/run_daily_trader.sh` (WSL) or `run_daily_trader.bat` (Windows Task Scheduler) runs a **daily + intraday** pass with `--resume` to keep positions across sessions.
- OpenClaw maintenance wrapper for Windows hosts without WSL distro:
  - `powershell -ExecutionPolicy Bypass -File scripts/run_openclaw_maintenance.ps1`

This keeps time-series evidence building (core tickers + gating) reproducible on Windows while still using the Linux-style cron multiplexer as the single entry point.

## 1. Design Principles

- **Single entrypoint**: All cron jobs call `bash/production_cron.sh` with a task name.
- **Virtualenvâ€‘first**: Uses the authorised `simpleTrader_env` interpreter only.
- **Configâ€‘driven**: Respects existing YAML configs (pipeline, data sources, validation).
- **Safe defaults**: Readâ€‘only monitoring and paper trading by default; no live execution
  unless explicitly configured in `config/` and environment variables.
- **Phase alignment**:
  - Phase 4.x: ETL, CV, checkpointing, logging.
  - Phase 5.2â€“5.5: LLM integration, monitoring, backfill.
  - Phase 5.7â€“5.9: Time series signal generation + autonomous profit engine.
  - Future Phase 5.x: Ticker discovery and optimizer cron stubs.

---

## 2. Cron Multiplexer Script

The cron wiring is centralised in:

- `bash/production_cron.sh`

Key tasks (first positional argument):

- `daily_etl` â€“ Full ETL pipeline (default: live mode).
- `auto_trader` â€“ Autonomous paperâ€‘trading loop (`scripts/run_auto_trader.py`).
- `nightly_backfill` â€“ Signal validation backfill (stub until modernised).
- `monitoring` â€“ LLM/pipeline health + latency monitoring.
- `env_sanity` â€“ Environment validation before trading hours.
- `ts_threshold_sweep` â€“ TS threshold sweep over realised trades; writes JSON to `logs/automation/`.
- `transaction_costs` â€“ Transaction cost estimation grouped by ticker/asset class to `logs/automation/`.
- `auto_trader_core` â€“ Core tickers with trade-count gate (defaults: AAPL,MSFT,GC=F,COOP; stops once â‰¥30 total and â‰¥10 per-ticker closed trades).
- `ticker_discovery_stub` â€“ Placeholder for future Phase 5.2 ticker discovery.
- `optimizer_stub` â€“ Placeholder for future Phase 5.3 optimizer pipeline.
- `weekly_sleeve_maintenance` â€“ Sleeve summary + promotion/demotion plan writer (see `bash/weekly_sleeve_maintenance.sh`).
- `synthetic_refresh` â€“ Generate a synthetic dataset (config-driven) for offline regression/smoke testing; respects `CRON_SYNTHETIC_*` env overrides.
- `training_priority_cycle` – Prioritized forecaster/LLM training-finetune chain driven by `config/training_priority.yml`.
- `self_improvement_review_forward` – Forward pending self-improvement proposals to human reviewers via OpenClaw targets (WhatsApp/Discord/Telegram).
- `openclaw_maintenance` – OpenClaw stale-session lock cleanup + gateway/channel self-heal guard.

### Provenance + dashboards

- ETL and auto_trader runs now emit `logs/automation/db_provenance_<run>.json` (run_id, dataset_id, origin, generator_version) for auditability and dashboard badges. Dashboards with `data_origin=synthetic` or `mixed` are explicitly **not profitability proof artifacts**; use the badge to gate promotions.

The script automatically:

- Resolves `PROJECT_ROOT` from its own location.
- Locates `simpleTrader_env/bin/python` (WSL only; Windows layouts are unsupported).
- Aborts if the WSL virtualenv is missing (no fallbacks; see `Documentation/RUNTIME_GUARDRAILS.md`).
- Writes perâ€‘task log files under `logs/cron/`, one file per invocation.

---

## 3. Example Crontab Entries

Assuming the repo lives at `/opt/portfolio_maximizer_v45`:

```cron
# â”Œâ”€ minute (0 - 59)
# â”‚ â”Œâ”€ hour (0 - 23)
# â”‚ â”‚ â”Œâ”€ day of month (1 - 31)
# â”‚ â”‚ â”‚ â”Œâ”€ month (1 - 12)
# â”‚ â”‚ â”‚ â”‚ â”Œâ”€ day of week (0 - 6) (Sunday to Saturday)
# â”‚ â”‚ â”‚ â”‚ â”‚
# â”‚ â”‚ â”‚ â”‚ â”‚

# 1. Daily ETL refresh (preâ€‘market)
15 5 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh daily_etl >> logs/cron/daily_etl.out 2>&1

# 2. Autonomous paper trader loop (every 30 minutes during market hours)
*/30 7-20 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh auto_trader --config config/pipeline_config.yml \
  >> logs/cron/auto_trader.out 2>&1

# 2b. Core ticker accumulation loop (halts once targets hit: â‰¥30 total, â‰¥10 per core ticker)
*/60 7-20 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  CRON_CORE_TICKERS="AAPL,MSFT,GC=F,COOP" \
  bash/production_cron.sh auto_trader_core >> logs/cron/auto_trader_core.out 2>&1

# 3. Nightly signal validation backfill (stub, see Section 5)
5 2 * * * cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh nightly_backfill >> logs/cron/nightly_backfill.out 2>&1

# 4. Hourly monitoring & latency checks
5 * * * * cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh monitoring >> logs/cron/monitoring.out 2>&1

# 5. Preâ€‘open environment sanity check
0 5 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh env_sanity >> logs/cron/env_sanity.out 2>&1

# 6. Weekly ticker discovery stub (Phase 5.2+)
0 3 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh ticker_discovery_stub >> logs/cron/ticker_discovery.out 2>&1

# 7. Weekly optimizer stub (Phase 5.3+)
30 3 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh optimizer_stub >> logs/cron/optimizer_stub.out 2>&1

# 8. Weekly TS threshold sweep (uses realised trades in trade_executions)
0 4 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh ts_threshold_sweep >> logs/cron/ts_threshold_sweep.out 2>&1

# 9. Monthly transaction cost estimation (per asset class)
15 4 1 * * cd /opt/portfolio_maximizer_v45 && \
  bash/production_cron.sh transaction_costs >> logs/cron/transaction_costs.out 2>&1

# 10. Weekly sleeve summary + promotion/demotion recommendations
0 5 * * 1 cd /opt/portfolio_maximizer_v45 && \
  bash/weekly_sleeve_maintenance.sh >> logs/cron/sleeve_maintenance.out 2>&1

# 11. Synthetic dataset refresh (offline regression)
0 1 * * 1 cd /opt/portfolio_maximizer_v45 && \
  CRON_SYNTHETIC_CONFIG="config/synthetic_data_config.yml" \
  CRON_SYNTHETIC_TICKERS="AAPL,MSFT" \
  bash/production_cron.sh synthetic_refresh >> logs/cron/synthetic_refresh.out 2>&1

# 12. Weekly prioritized forecaster retraining + proposal refresh
30 1 * * 6 cd /opt/portfolio_maximizer_v45 && \
  CRON_TRAINING_PROFILE="forecasters" \
  bash/production_cron.sh training_priority_cycle >> logs/cron/training_forecasters.out 2>&1

# 13. Weekly LLM finetune dataset refresh (+ optional trainer hook)
0 2 * * 6 cd /opt/portfolio_maximizer_v45 && \
  CRON_TRAINING_PROFILE="llm" \
  bash/production_cron.sh training_priority_cycle >> logs/cron/training_llm.out 2>&1

# 14. Forward pending self-improvement proposals to human reviewers
15 2 * * 1-5 cd /opt/portfolio_maximizer_v45 && \
  CRON_REVIEW_TARGETS="whatsapp:+15551234567,telegram:@mychat,discord:channel:123456789012345678" \
  bash/production_cron.sh self_improvement_review_forward >> logs/cron/self_improvement_review_forward.out 2>&1

# 15. OpenClaw maintenance guard (stale locks + gateway/channel guard)
0 3 * * 0 cd /opt/portfolio_maximizer_v45 && \
  CRON_OPENCLAW_MAINTENANCE_APPLY=1 \
  CRON_OPENCLAW_DISABLE_BROKEN_CHANNELS=1 \
  CRON_OPENCLAW_RESTART_GATEWAY_ON_FAILURE=1 \
  bash/production_cron.sh openclaw_maintenance >> logs/cron/openclaw_maintenance.out 2>&1
```

You can edit the schedule and paths to match your deployment environment.

---

## 4. Task Details

### 4.1 `daily_etl`

Maps to the core ETL stages documented in:

- `Documentation/arch_tree.md` â€“ Phase 1â€“4.8 (ETL, validation, preprocessing, storage, CV).
- `scripts/run_etl_pipeline.py` â€“ configurably runs:
  `data_extraction â†’ data_validation â†’ data_preprocessing â†’ data_storage â†’ time_series_forecasting â†’ time_series_signal_generation â†’ signal_router`.

Defaults (overridable via environment):

- `CRON_TICKERS` (default: `AAPL,MSFT,GOOGL`)
- `CRON_START_DATE` (default: `2020-01-01`)
- `CRON_END_DATE` (default: today)
- `CRON_EXEC_MODE` (default: `live`, may also be `synthetic` for offline regression)

### 4.2 `auto_trader`

Drives the â€œAutonomous Profit Engineâ€ loop described in:

- `Documentation/arch_tree.md` (Week 5.9: Autonomous Profit Engine rollâ€‘out).
- `scripts/run_auto_trader.py` â€“ orchestrates:
  extraction â†’ validation â†’ forecasting â†’ TS signal generation â†’ routing â†’ paper trading.

Typical cron usage is every N minutes during market hours, as a
stateless trigger into the autoâ€‘trader, which internally respects
the configured risk and validation gates.

### 4.2b `auto_trader_core`

Same as `auto_trader` but with defaults targeted at the core tickers (`AAPL,MSFT,GC=F,COOP`) and a builtâ€‘in trade-count gate:

- Skips execution once **both** conditions are met in `trade_executions`:
  - Total closed trades â‰¥ `CRON_CORE_TOTAL_TARGET` (default: 30)
  - Per-core-ticker closed trades â‰¥ `CRON_CORE_PER_TICKER_TARGET` (default: 10)
- Env overrides:
  - `CRON_CORE_TICKERS` (comma list)
  - `CRON_CORE_DB_PATH` (default: `data/portfolio_maximizer.db`)
  - `CRON_CORE_TOTAL_TARGET`, `CRON_CORE_PER_TICKER_TARGET`

### 4.3 `nightly_backfill`

- `Documentation/implementation_checkpoint.md` â€“ nightly signal validation backfill.
- `scripts/backfill_signal_validation.py` â€“ executed via Task Scheduler (`PortfolioMaximizer_BackfillSignals`) and available for manual runs.
- Manual trigger: `bash/run_backfill.sh` (uses `simpleTrader_env/bin/python3` when present, logs to `logs/automation/backfill_<timestamp>.log`).

Current behaviour:

- If `scripts/backfill_signal_validation.py` exists, it is called with any extra args.
- If not present or not yet modernised, the job logs a stub message and exits cleanly.

### 4.4 `monitoring`

Aligns with:

- `Documentation/arch_tree.md` â€“ Week 5.5 & 5.9 (Error Monitoring & Performance Optimization, Monitoring + Nightly Backfill Instrumentation).
- `scripts/monitor_llm_system.py` â€“ logs LLM latency benchmarks and backtest summaries.

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

Both stubs currently just write a oneâ€‘line log entry so you can safely
keep their cron entries in place while implementation progresses.

### 4.7 `synthetic_refresh`

- Generates a synthetic dataset via `scripts/generate_synthetic_dataset.py` (respects `CRON_SYNTHETIC_CONFIG`, `CRON_SYNTHETIC_TICKERS`, `CRON_SYNTHETIC_OUTPUT_ROOT`) and immediately validates it with `scripts/validate_synthetic_dataset.py`.
- Produces `data/synthetic/<dataset_id>/<ticker>.parquet` + `manifest.json` and a validation report under `logs/automation/`.
- Enablement is syntheticâ€‘first only; keep live trading disabled. Promotion of synthetic outputs to live cron tasks requires GREEN/acceptable YELLOW quant health per `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` and the sequencing rules in `Documentation/NEXT_TO_DO_SEQUENCED.md`.

### 4.8 `sanitize_caches`

- Prunes cached data/log artifacts older than `CRON_SANITIZE_RETENTION` (default: 14 days).
- Invokes `scripts/sanitize_cache_and_logs.py`, targeting common data/log paths while skipping DVC stores by default.
- Override targets/patterns with:
  - `CRON_SANITIZE_DATA_DIRS` (comma-separated)
  - `CRON_SANITIZE_LOG_DIRS` (comma-separated)
  - `CRON_SANITIZE_PATTERNS` (comma-separated glob patterns)

---

### 4.9 `training_priority_cycle`

- Runs `scripts/run_training_priority_cycle.py` with profile/target filtering.
- Source of truth is `config/training_priority.yml` (priority ordering, commands, prereqs, criticality).
- Supports both forecaster retraining flows and LLM fine-tune dataset/trainer hooks.
- Key env overrides:
  - `CRON_TRAINING_PROFILE` = `forecasters|llm|all` (default `forecasters`)
  - `CRON_TRAINING_TARGET` (default `local_cron`)
  - `CRON_TRAINING_MAX_PRIORITY` (optional cutoff)
  - `CRON_TRAINING_CONTINUE_ON_ERROR=1` (optional)
  - `CRON_TRAINING_DRY_RUN=1` (optional planning mode)

### 4.10 `self_improvement_review_forward`

- Runs `scripts/forward_self_improvement_reviews.py` to notify reviewers about pending proposals under `logs/llm_activity/proposals`.
- Sends sanitized summaries only (target file + compact description); no diff payloads or secret values are forwarded.
- Uses OpenClaw targets from `OPENCLAW_TARGETS` / `OPENCLAW_TO` by default, so one run can fan out to WhatsApp, Discord, and Telegram.
- Deduplicates by proposal id using persistent state in `logs/automation/self_improve_review_forward_state.json`.
- Key env overrides:
  - `CRON_REVIEW_TARGETS`, `CRON_REVIEW_TO`, `CRON_REVIEW_CHANNEL`
  - `CRON_REVIEW_MAX_ITEMS`, `CRON_REVIEW_MAX_AGE_DAYS`, `CRON_REVIEW_MIN_INTERVAL_MINUTES`
  - `CRON_REVIEW_RESEND_PENDING=1`, `CRON_REVIEW_DRY_RUN=1`, `CRON_REVIEW_FORCE=1`

### 4.11 `openclaw_maintenance`

- Runs `scripts/openclaw_maintenance.py` through the cron multiplexer.
- Intended to prevent long-lived OpenClaw instability from stale session locks and unhealthy gateway runtime state.
- Can optionally disable broken non-primary channels (Telegram/Discord) when they repeatedly fail auth/config checks.
- Writes report JSON to `logs/automation/openclaw_maintenance_latest.json` by default.
- Key env overrides:
  - `CRON_OPENCLAW_MAINTENANCE_APPLY=1` to apply changes (default in multiplexer).
  - `CRON_OPENCLAW_PRIMARY_CHANNEL` (default `whatsapp`).
  - `CRON_OPENCLAW_SESSION_STALE_SECONDS` (default `7200`).
  - `CRON_OPENCLAW_DISABLE_BROKEN_CHANNELS=1`.
  - `CRON_OPENCLAW_RESTART_GATEWAY_ON_FAILURE=1`.
  - `CRON_OPENCLAW_REPORT_FILE` for alternate report path.
  - `CRON_OPENCLAW_STRICT=1` to fail the task on unresolved errors.

## 5. Safety & Operational Notes

- **Database path**: Production tasks default to `data/portfolio_maximizer.db`.
  For synthetic/brutal runs or staging, override via `PORTFOLIO_DB_PATH`
  when invoking the pipeline (as already done in the brutal test harness).
- **LLM gating**: Time Series is the canonical stack (see
  `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`). LLMâ€‘heavy
  jobs (e.g., `monitor_llm_system.py` with full LLM usage, or
  `run_auto_trader.py` with LLM enabled) should only be enabled after:
  - Profitâ€‘critical tests are green (`tests/integration/test_profit_critical_functions.py`).
  - LLM performance tests are passing and latency within budget.
- **Brutal test suite**: For periodic deep validation, use:
  - `bash/comprehensive_brutal_test.sh` (manual or via a separate cron).
  - This is intentionally not wired into the default production cron
    because it can run for hours and is best treated as a maintenance job.

---

## 6. Quant Threshold Sweeps & Cost Estimation

Two read-only helpers are now exposed through the cron multiplexer to keep TS thresholds and friction assumptions grounded in recent data:

- `scripts/sweep_ts_thresholds.py` â€“ summarises realised performance over a grid of `(confidence_threshold, min_expected_return)` values per ticker (task: `ts_threshold_sweep`).
- `scripts/estimate_transaction_costs.py` â€“ estimates commission / transaction costs by ticker or simple asset class buckets (task: `transaction_costs`).

Env overrides for these tasks:

```bash
# TS sweep overrides (defaults: 365-day lookback, 0.50/0.55/0.60 confidence grid, 0.001/0.002/0.003 min_return grid)
CRON_TS_SWEEP_TICKERS="AAPL,MSFT,GC=F,COOP" \
CRON_TS_SWEEP_CONFIDENCE="0.50,0.55,0.60,0.65" \
CRON_TS_SWEEP_MIN_RETURN="0.001,0.002,0.003,0.004" \
CRON_TS_SWEEP_MIN_TRADES=10 \
CRON_TS_SWEEP_OUTPUT="logs/automation/ts_threshold_sweep.json" \
  bash/production_cron.sh ts_threshold_sweep

# Transaction cost overrides (defaults: 365-day lookback, asset_class grouping, min_trades=5)
CRON_COST_AS_OF="2025-12-05" \
CRON_COST_GROUPING="asset_class" \
CRON_COST_MIN_TRADES=5 \
CRON_COST_OUTPUT="logs/automation/transaction_costs.json" \
  bash/production_cron.sh transaction_costs
```

Both scripts:
- Only read from the existing SQLite database; they do not modify configs.
- Emit machine-readable JSON under `logs/automation/` for use by higher-level tooling (e.g., a proposal â†’ config diff helper or notebooks).

Helper to run the full chain (costs + TS sweep + config proposals) manually:

```bash
bash/run_ts_sweep_and_proposals.sh

# Optional overrides
SWEEP_TICKERS="AAPL,MSFT,GC=F,COOP" \
SWEEP_SEL_MIN_PF=1.2 \
SWEEP_SEL_MIN_WR=0.55 \
COST_GROUPING="asset_class" \
PROPOSALS_OUTPUT="logs/automation/config_proposals.json" \
  bash/run_ts_sweep_and_proposals.sh
```

### 6.1 Automation Dashboard Glue

To surface a unified â€œwhat should we change next?â€ view for humans and agents:

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
  bash/production_cron.sh ts_model_search >> logs/cron/ts_model_search.out 2>&1
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
   - `bash/production_cron.sh daily_etl`
   - `bash/production_cron.sh auto_trader --dry-run` (if supported)
4. Install crontab entries from Section 3, adjusting paths and schedules.
5. Monitor:
   - `logs/cron/*.log` for cronâ€‘level output.
   - `logs/pipeline_run.log`, `logs/events/events.log`,
     and monitoring logs for deeper diagnostics.
