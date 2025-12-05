# Quant Validation Automation & Threshold Optimization TODO

**Last updated**: 2025-11-20  
**Scope**: Automating quant validation sample growth, TS threshold sweeps, cost calibration, and sleeve‑level promotion/demotion via cron and CLI helpers.

This document turns the current quant‑validation findings into a concrete, automation‑friendly TODO list. All items are config‑ or script‑level only (no core architecture rewrites) and should be backed by stats/backtests before being made permanent.

---

## Phase 1 – Grow Quant‑Validation Sample Size

- [ ] Align quant lookbacks  
  - Set `quant_validation.lookback_days = 365` in `config/quant_success_config.yml` and `config/quant_success_config.hyperopt.yml`.  
  - Ensure `scripts/run_auto_trader.py` is always invoked with `--lookback-days >= quant_validation.lookback_days` (default is already `365`).

- [ ] Maintain robust per‑ticker series  
  - If sparse histories cause instability, consider raising `MIN_SERIES_POINTS` in `scripts/run_auto_trader.py` so the `_ensure_min_length` padding targets a slightly larger window.

- [ ] Schedule regular TS cycles for core names  
  - Use `bash/production_cron.sh` (see `Documentation/CRON_AUTOMATION.md`) to run `auto_trader` daily for a curated list of **core tickers** (e.g., `AAPL,MSFT,GC=F,COOP` plus a few backtest‑strong names that already clear PF/WR/n criteria).  
  - Target: accumulate ≥30 closed trades overall and ≥10 closed trades in each of the core names before tightening thresholds.

---

## Phase 2 – Threshold Sweeper for Time‑Series Parameters

- [ ] Add `scripts/sweep_ts_thresholds.py`  
  - CLI should accept:  
    - `--tickers`: comma‑separated list of tickers (default: core universe).  
    - `--grid-confidence`: JSON or simple list of `confidence_threshold` values.  
    - `--grid-min-return`: list of `min_expected_return` values.  
    - Optional: `--grid-max-risk`, `--min-trades`, `--output`.  
  - For each `(ticker, θ)` combination:  
    - Reconstruct realised or backtest PnL using existing DB history (`etl/database_manager.py`) and/or backtest helpers.  
    - Compute annualised PnL, profit factor (PF), win rate (WR), and trade count `n`.

- [ ] Selection rule per ticker  
  - Choose θ that maximises annualised PnL subject to:  
    - `PF ≥ 1.1`,  
    - `WR ≥ 0.5`,  
    - `n ≥ min_trades` (e.g., 30 overall, 10 per core name).  
  - Write chosen thresholds to a machine‑readable artifact (JSON) and, in a later phase, map them into config (e.g., per‑ticker block in `config/signal_routing_config.yml` or a dedicated thresholds table).

- [ ] Automation  
  - Wire the sweeper into a weekly/monthly cron task after ETL + auto‑trader runs or after `bash/run_post_eval.sh` (hyperopt) so it always works from fresh data.

---

## Phase 3 – Transaction Cost Calibration

- [ ] Add `scripts/estimate_transaction_costs.py`  
  - CLI should accept:  
    - `--as-of`: optional date; default = “now”.  
    - `--lookback-days`: window for trades to analyse.  
    - `--grouping`: how to bucket instruments (e.g., `asset_class`, `ticker`).  
    - `--output`: JSON path for proposed cost parameters.  
  - Use `etl/database_manager` trade tables to compute median and percentile round‑trip costs (fees + slippage) per group.

- [ ] Feed costs back into configs  
  - Derive recommended `friction_buffer` and `min_expected_return` per asset class such that both sit a few bps above observed median round‑trip cost.  
  - Optionally adjust `quant_validation.success_criteria.min_expected_profit` in `config/quant_success_config.yml` to remain consistent with observed costs and capital base.

- [ ] Schedule  
  - Run cost estimation monthly/quarterly under cron and log proposals to `logs/automation/transaction_costs_*.json` for manual review before changing configs.

---

## Phase 4 – Risk Buckets & Sleeve‑Level Performance

- [ ] Formalise risk buckets  
  - Extend `risk/barbell_policy.BarbellConfig` / `config/barbell.yml` to tag tickers as `core` vs `speculative` with per‑bucket capital and per‑trade risk caps.

- [ ] Per‑sleeve metrics  
  - Extend `PaperTradingEngine.get_performance_metrics()` or add `scripts/summarize_sleeves.py` to emit PF, WR, annualised PnL, and trade count per `(ticker, route_type, bucket)` tuple.

- [ ] Promotion / demotion rules  
  - Promote “good sleeves”: if PF/WR and `n` exceed thresholds, allow a small size increase or slightly lower `min_expected_profit`.  
  - Demote “weak sleeves”: raise `min_expected_profit` or disable the sleeve (especially for speculative bucket names) until metrics improve.

- [ ] Automation  
  - Schedule sleeve summary + promotion/demotion review weekly; write proposed config diffs to `logs/automation/sleeve_updates_*.json`.

---

## Phase 5 – Execution Log & Time‑of‑Day Filters

- [ ] Enhance execution logging  
  - Ensure each execution record contains: `ticker`, `side`, `shares`, `fill_price`, approximated `mid_price` at fill, timestamp, and `route_type` (TS vs LLM).

- [ ] Slippage analysis  
  - Implement `scripts/analyze_slippage_windows.py` to compute slippage distributions by hour (or 30‑minute bins) per asset class and flag “bad” time windows.

- [ ] No‑trade windows  
  - Add an `execution.no_trade_windows` block (per asset class) to the relevant config file and enforce it in the execution engine so trades are skipped when slippage is historically poor.

---

## Phase 6 – Glue, Guardrails, and Reporting

- [ ] Make all new automation scripts config‑driven and read‑only with respect to source code.  
- [ ] Standardise outputs to `logs/automation/*.json` and `visualizations/dashboard_automation.json` so dashboards and agents can consume the latest recommendations.  
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

