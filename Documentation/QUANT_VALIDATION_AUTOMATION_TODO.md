# Quant Validation Automation & Threshold Optimization TODO

**Last updated**: 2026-01-18
**Scope**: Automating quant validation sample growth, TS threshold sweeps, cost calibration, and sleeve‑level promotion/demotion via cron and CLI helpers.
**Current status (2026-01-07)**: Phases 1–5 are implemented; Phase 6 (dashboard glue + standardized outputs) remains pending (see `Documentation/PROJECT_STATUS.md`).

## Delta (2026-01-18)

- Phase 6 “dashboard glue” is partially complete: `visualizations/live_dashboard.html` is non-fictitious (empty states until data exists) and polls `visualizations/dashboard_data.json` every 5s. Canonical producer is `scripts/dashboard_db_bridge.py` (DB→JSON) started by bash orchestrators; snapshots persist to `data/dashboard_audit.db` by default (`--persist-snapshot`).

This document turns the current quant‑validation findings into a concrete, automation‑friendly TODO list. All items are config‑ or script‑level only (no core architecture rewrites) and should be backed by stats/backtests before being made permanent.

**Project-wide sequencing**: See `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` for the higher-level, step-by-step plan that ties TS horizon alignment, execution cost modeling, and run-local reporting into the quant-validation workflow.

---

## Phase 1 – Grow Quant‑Validation Sample Size

- [x] Align quant lookbacks
  - Set `quant_validation.lookback_days = 365` in `config/quant_success_config.yml` and `config/quant_success_config.hyperopt.yml`.
  - Ensure `scripts/run_auto_trader.py` is always invoked with `--lookback-days >= quant_validation.lookback_days` (default is already `365`).

- [x] Maintain robust per‑ticker series
  - Raised `MIN_SERIES_POINTS` in `scripts/run_auto_trader.py` to expand padding and stabilise sparse histories.

- [x] Schedule regular TS cycles for core names
  - Use `bash/production_cron.sh` (see `Documentation/CRON_AUTOMATION.md`) to run `auto_trader` daily for a curated list of **core tickers** (e.g., `AAPL,MSFT,GC=F,COOP` plus a few backtest‑strong names that already clear PF/WR/n criteria).
  - Target: accumulate ≥30 closed trades overall and ≥10 closed trades in each of the core names before tightening thresholds.

---

## Phase 2 – Threshold Sweeper for Time‑Series Parameters

- [x] Add `scripts/sweep_ts_thresholds.py`
  - CLI accepts:
    - `--tickers`: comma‑separated list of tickers (default: realised‑trade universe).
    - `--grid-confidence`: JSON or simple list of `confidence_threshold` values.
    - `--grid-min-return`: list of `min_expected_return` values.
    - Optional: `--grid-max-risk`, `--min-trades`, `--output`.
  - For each `(ticker, θ)` combination:
    - Reconstruct realised or backtest PnL using existing DB history (`etl/database_manager.py`) and/or backtest helpers.
    - Compute annualised PnL, profit factor (PF), win rate (WR), and trade count `n`.

- [x] Selection rule per ticker
  - Choose θ that maximises annualised PnL subject to:
    - `PF ≥ 1.1`,
    - `WR ≥ 0.5`,
    - `n ≥ min_trades` (e.g., 30 overall, 10 per core name).
  - Write chosen thresholds to a machine‑readable artifact (JSON) and, in a later phase, map them into config (e.g., per‑ticker block in `config/signal_routing_config.yml` or a dedicated thresholds table).

- [x] Automation
  - Cron wrapper added via `bash/production_cron.sh ts_threshold_sweep` (defaults: 365‑day lookback, grid confidence 0.50/0.55/0.60, grid min_return 0.001/0.002/0.003, min_trades=10).
- [x] Selection artifact
  - `scripts/sweep_ts_thresholds.py` now emits a per‑ticker `selection` block chosen by annualized PnL with PF/WR/min_trades gates. Use `bash/run_ts_sweep_and_proposals.sh` to run sweep + proposals in one step.
- [x] Diagnostic liquidation caution
  - `scripts/liquidate_open_trades.py` (and `bash/force_close_and_sweep.sh`) are for evidence collection only; they force-close trades with synthetic MTM PnL and must not be used for real PnL reporting.

---

## Phase 3 – Transaction Cost Calibration

- [x] Add `scripts/estimate_transaction_costs.py`
  - CLI accepts:
    - `--as-of`: optional date; default = “now”.
    - `--lookback-days`: window for trades to analyse.
    - `--grouping`: how to bucket instruments (e.g., `asset_class`, `ticker`).
    - `--output`: JSON path for proposed cost parameters.
  - Use `etl/database_manager` trade tables to compute median and percentile round‑trip costs (fees + slippage) per group.

- [x] Feed costs back into configs
  - `scripts/generate_signal_routing_overrides.py` writes `logs/automation/signal_routing_overrides.yml` with friction buffers and min_expected_return per asset class for review.

- [x] Schedule
  - Cron wrapper added via `bash/production_cron.sh transaction_costs` (defaults: 365‑day lookback, asset_class grouping, min_trades=5; outputs under `logs/automation/transaction_costs*.json`).
- [x] Apply to configs
  - Per‑ticker overrides and asset‑class friction buffers added to `config/signal_routing_config.yml` for review (CL=F, MSFT, MTN; US_EQUITY, FX).

---

## Phase 4 – Risk Buckets & Sleeve‑Level Performance

- [x] Formalise risk buckets
  - `risk/barbell_policy.BarbellConfig` / `config/barbell.yml` tag safe/core/spec sleeves with per‑bucket caps; optional gating is wired into `run_auto_trader.py`.

- [x] Per‑sleeve metrics
  - `scripts/summarize_sleeves.py` emits PF, WR, annualised PnL, and trade count per ticker with bucket tags for dashboards/cron.

- [x] Promotion / demotion rules
  - `scripts/evaluate_sleeve_promotions.py` consumes the JSON emitted by `scripts/summarize_sleeves.py` (sleeve-labelled metrics, consistent with `config/barbell.yml`) and proposes promotions/demotions once PF/WR/n thresholds are met (see `logs/automation/sleeve_promotion_plan.json`).

- [x] Automation
  - `bash/weekly_sleeve_maintenance.sh` schedules sleeve summaries + promotion/demotion review and writes artifacts to `logs/automation/sleeve_*.json`.

---

## Phase 5 – Execution Log & Time‑of‑Day Filters

- [x] Enhance execution logging
  - `run_auto_trader.py` logs execution/skip events (ticker, side, entry, mid‑price, timestamp, source) to `logs/automation/execution_log.jsonl`.

- [x] Slippage analysis
  - `scripts/analyze_slippage_windows.py` consumes the execution log to compute mid‑price slippage distributions by asset class and hour (commission proxy retained).

- [x] No‑trade windows
  - `execution.no_trade_windows` lives in `config/signal_routing_config.yml` and is enforced before execution; skipped events are recorded.

---

## Phase 6 – Glue, Guardrails, and Reporting

- [ ] Make all new automation scripts config‑driven and read‑only with respect to source code.
- [ ] Standardise outputs to `logs/automation/*.json` and `visualizations/dashboard_automation.json` so dashboards and agents can consume the latest recommendations (sleeve + execution artifacts now live under `logs/automation/`; dashboard glue remains).
- [ ] Keep changes consistent with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md` and `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`: no permanent threshold changes without quantified justification (PF/WR/n, cost‑aware).

---

## Automation Helpers Overview

To keep this roadmap actionable, the following helpers are already implemented:

- `scripts/sweep_ts_thresholds.py`
  - Sweeps `(confidence_threshold, min_expected_return)` over a grid and writes realised performance summaries to `logs/automation/ts_threshold_sweep.json`.

- `scripts/estimate_transaction_costs.py`
  - Aggregates `commission` and `realized_pnl` from `trade_executions` and writes per‑group cost stats to `logs/automation/transaction_costs.json`.

- `scripts/generate_config_proposals.py`
  - Ingests the two JSON artifacts above and produces `logs/automation/config_proposals.json` with **proposed** changes for:
    - Per‑ticker TS thresholds (candidates for `config/signal_routing_config.yml`), and
    - Cost‑aware `min_expected_return` suggestions (candidates for `config/quant_success_config.yml` and related configs).
  - This script never edits configs directly; it is the bridge between cron‑driven evidence and human‑reviewed config diffs.

For cron wiring examples of the sweep and cost scripts, see `Documentation/CRON_AUTOMATION.md`.
