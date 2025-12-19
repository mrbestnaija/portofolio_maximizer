# Monetization Readiness & Guardrails

> Source of truth for monetization gating, synthetic-first constraints, and usage tracking.  
> Keep aligned with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`, `Documentation/SYNTHETIC_DATA_GENERATOR_TODO.md`, and `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`.

## 1) Purpose
- Prevent alert/licensing rollout until the strategy meets profitability and risk standards.
- Ensure synthetic-first operation while live free-tier data is too thin (commodities/illiquid classes).
- Track usage so low-value features can be paused before they consume maintenance time.

## 2) Monetization Gate (hard stop)
- Status enums: `BLOCKED | EXPERIMENTAL | READY_FOR_PUBLIC`.
- Thresholds (365d window): `annual_return >= 0.10`, `sharpe >= 1.0`, `max_drawdown <= 0.25`, `trade_count >= 30`.
- Entry points:
  - CLI: `python -m tools.check_monetization_gate --window 365` (or `./simpleTrader_env/bin/python -m tools.check_monetization_gate --window 365` with `PYTHONPATH=.` set in shell) – exits non-zero on BLOCKED.
  - Modules should call the gate before sending alerts/reports (see `src/core/monetization_gate.py`).
- Line-budget guardrail: total monetization code ≤700 LOC; gate ≤250 LOC (per REWARD plan).

## 3) Synthetic-First Policy
- Pre-prod and brutal runs must use persisted synthetic datasets until quant health is GREEN/YELLOW on live data.
- Required flags: `ENABLE_SYNTHETIC_PROVIDER=1`, `SYNTHETIC_ONLY=1`, `--execution-mode synthetic --data-source synthetic`, or `SYNTHETIC_DATASET_ID`/`PATH` pointing to `data/synthetic/latest.json`.
- Live trading/alerts stay disabled until:
  - Quant validation clears RED (`scripts/check_quant_validation_health.py`).
  - Gate status is `READY_FOR_PUBLIC`.
  - Synthetic/living data segregation confirmed (no synthetic rows in live dashboards/training).

## 4) Usage Tracking & Reality Checks
- `src/monitoring/monetization_usage_tracker.py` logs counts for alerts/reports/exporters (JSON/SQLite).
- CLI: `python -m tools.monetization_reality_check` prints 30d usage + keep/refactor/delete recommendations.
- Alert pipeline (`tools.push_daily_signals`) must:
  - Invoke monetization gate first.
  - Record per-channel sends to usage tracker.
  - Honor channel feature flags in `config/monetization.yml`.

## 5) Operational Checklist (per run or cron)
- Run `python -m tools.check_monetization_gate --window 365`; abort on BLOCKED.
- Ensure synthetic flags set for pre-prod; clear them before any live/digest intended for external users.
- For alerts: `python -m tools.push_daily_signals --config config/monetization.yml --date YYYY-MM-DD` (fails closed when gate BLOCKED).
- For reports/factsheets: only generate if gate is READY or explicit `--allow-experimental`.
- Keep secrets in Docker secrets/env (see `SECURITY_IMPLEMENTATION_SUMMARY.md`); never bake keys into configs.
- If you change DBs (e.g., `data/test_database.db` vs `data/portfolio_maximizer.db`), pass `--db-path ...` to the gate and repopulate performance_metrics in that DB before rerunning.
- The current gate was unblocked by inserting synthetic trades/metrics; replace these with real pipeline/auto-trader runs as soon as available.

## 6) Evidence & Logging
- Gate decisions: write to `logs/automation/monetization_gate.log`.
- Usage tracker: `logs/automation/monetization_usage.jsonl` (rotated by cron if configured).
- Synthetic runs: `logs/automation/synthetic_runs.log` with auto-prune (14 days) per synthetic roadmap.
- Quant health: `logs/signals/quant_validation.jsonl` + `scripts/check_quant_validation_health.py` (global) and `scripts/summarize_quant_validation.py` (per-ticker).

## 7) Promotion Criteria (from REWARD plan)
- Require ≥90 days of audited performance before turning on direct monetization.
- Indirect path first: foreign broker demo trades, newsletter/blog, Telegram/email alerts with disclaimers.
- Options/bridge modules must remain behind feature flags until gate is READY.

## 8) References
- `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`
- `Documentation/SYNTHETIC_DATA_GENERATOR_TODO.md`
- `Documentation/UNIFIED_ROADMAP.md`
- `Documentation/SECURITY_IMPLEMENTATION_SUMMARY.md`
- `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`
