# Monetization Gate (Local-Only Snapshot)

- **States**: `READY`, `EXPERIMENTAL`, `BLOCKED`. Default remains `BLOCKED` until quant health is GREEN on live/paper.
- **Inputs**: quant health (`scripts/check_quant_validation_health.py`), usage tracking, profitability metrics from DB (paper/live), synthetic regressions.
- **Synthetic-first**: While feeds are shallow, pre-prod stays synthetic (`ENABLE_SYNTHETIC_PROVIDER=1`, `--execution-mode synthetic`). Monetization reports must never include synthetic-only PnL.
- **Controls**:
  - BLOCKED: no alerts/exports; LLM-heavy jobs disabled; only synthetic/brutal allowed.
  - EXPERIMENTAL: limited alerts, guarded dashboards; LLM capped; live extraction allowed only if quant health >= YELLOW.
  - READY: full alerts/exports permitted; synthetic refresh remains as regression guardrail.
- **Ops**:
  - Run gate: `python -m tools.check_monetization_gate --window 365 [--db-path ...]`.
  - Push daily signals only if gate READY/EXPERIMENTAL: `python -m tools.push_daily_signals --config config/monetization.yml`.
  - Reality check: `python -m tools.monetization_reality_check`.
- **Data hygiene**: no secrets in configs; separate synthetic vs live DBs; avoid mixing synthetic rows in dashboards/training once gate transitions to READY.
