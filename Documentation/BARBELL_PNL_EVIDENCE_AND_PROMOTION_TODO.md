# Barbell PnL Evidence & Promotion TODO (Gate Before Production)

## Current evidence snapshot (local DB @ `data/portfolio_maximizer.db`)

### A) Walk-forward (TS-only signal regeneration)
- Result: **inconclusive** with current defaults; the walk-forward harness produces **0 executed trades** under realistic thresholds and engine guardrails.
- Root causes observed in runs:
  - `TimeSeriesSignalGenerator` mostly outputs `HOLD` due to low confidence/edge.
  - When thresholds are loosened, `PaperTradingEngine` still rejects many signals as **edge/cost too low** and/or **position size too small**.

### B) Trade-history counterfactual (scale realized PnL by bucket)
- Window: `2025-12-19` → `2026-01-13`
- Trades with realized PnL: **34**
- Observation: realized PnL in the DB is **100% positive** (no losses) and is recorded on a **single date** (`2025-12-19`), producing `profit_factor = inf` and `max_drawdown = 0`.
- Barbell counterfactual using current `config/barbell.yml` multipliers:
  - TS_ONLY `total_profit`: **2935**
  - BARBELL_SIZED `total_profit`: **2427**
  - Δ `total_profit`: **-508** (barbell reduces profits when there are no losses to reduce)
- Conclusion: **not convincing / not promotable** on current data; we need a loss-containing sample and/or real signal payloads.

## Promotion criteria (must be met)

- **Trade count**: `total_trades >= 30` per window, and `>= 3` independent windows (or runs) with non-overlapping dates.
- **Loss presence**: at least `>= 5` losing trades per window (otherwise drawdown/PF conclusions are not meaningful).
- **Primary** (paired comparison TS_ONLY vs BARBELL_SIZED on same signal stream):
  - `Δtotal_return_pct > 0` OR `Δprofit_factor > 0` with `Δmax_drawdown <= 0`
  - AND `Δtotal_trades` not catastrophically negative (e.g., `>-30%` unless explicitly intended)
- **Secondary**:
  - Exposure by bucket respects `safe_min/safe_max` and `risk_max` constraints (when allocator is enabled).
  - No regression in runtime, guardrail rejects, or audit completeness.

## TODO A — Make the evidence step statistically valid

- [ ] Persist **actionable** non-`HOLD` signals into `trading_signals` (and/or `llm_signals`) for every run.
- [ ] Ensure `trade_executions.signal_id` is populated (or add a durable join key like `(run_id, ticker, signal_timestamp)`).
- [ ] Fix/verify `trade_executions.realized_pnl`:
  - [ ] Confirm negative PnL trades are recorded when they occur.
  - [ ] Add invariants: realized PnL distribution must include losses in realistic regimes; raise an audit alert if not.
- [ ] Add a “**signal replay**” evidence mode:
  - Input: `trading_signals` + `ohlcv_data`
  - Replay with identical ordering/timestamps and compute TS_ONLY vs BARBELL_SIZED by scaling **confidence → sizing** only.
  - Output: JSON report + CSV summary in `reports/` with metrics + bucket exposure timeline.
- [ ] Add a “**multi-window runner**”:
  - Iterate windows: `30d`, `90d`, `180d`, `365d` (or rolling monthly)
  - Emit `reports/barbell_pnl_matrix_<ts>.json` with per-window outcomes.
- [ ] Add statistical robustness:
  - [ ] Bootstrap CIs for `Δtotal_return_pct`, `Δmax_drawdown`.
  - [ ] Paired tests (block bootstrap by week/month) to avoid i.i.d. assumptions.
  - [ ] Report **effect size** + CI, not just point estimates.

## TODO B — If evidence becomes convincing: production implementation plan

### B1) Feature-flagged sizing overlay (first production step)
- [ ] Add `ENABLE_BARBELL_SIZING` flag (env + config) independent of `enable_barbell_allocation`.
- [ ] Centralize sizing modifier:
  - Input: `(ticker, base_confidence, barbell_cfg)`
  - Output: `effective_confidence` (clipped `[0,1]`) + `bucket` tag for audit.
- [ ] Wire into execution path (single choke point):
  - `models/signal_router` or `execution/paper_trading_engine.execute_signal` (choose one; avoid duplicating logic).
- [ ] Audit logging:
  - Store `(bucket, base_confidence, effective_confidence, multipliers)` in `trade_executions.provenance` or a new table.
- [ ] Dashboard:
  - Show per-trade bucket + confidence before/after and realized PnL by bucket.

### B2) Portfolio/NAV allocator integration (second step)
- [ ] Enable `risk.barbell_policy.BarbellConstraint.project_to_feasible()` behind `enable_barbell_allocation`.
- [ ] Integrate into NAV allocator (`risk/nav_allocator.py`) so portfolio weights respect:
  - Safe bucket `safe_min/safe_max`
  - Risk bucket maxes and per-position caps
- [ ] Add reconciliation:
  - Pre/post projection weights, and residual tracking (`L1` drift between desired vs feasible).

### B3) Guardrails + monitoring (must ship with prod wiring)
- [ ] Add runtime assertions:
  - Bucket weights sum sanity, caps enforced, no NaNs.
  - If `enable_barbell_allocation=false`, ensure no-op invariants hold.
- [ ] Monitoring metrics (per run + rolling):
  - `bucket_exposure_{safe,core,spec}`, `trade_pnl_by_bucket`, `reject_rate_by_bucket`
  - `Δreturn_pct`, `Δmax_drawdown`, `profit_factor`, `turnover`, `cost_bps`
- [ ] Alert thresholds:
  - `safe_weight < safe_min` or `risk_weight > risk_max` for sustained periods.
  - Sudden jump in reject rates after enabling sizing overlay.

### B4) Testing & verification
- [ ] Unit tests:
  - Confidence scaling by bucket (already exists) + guardrail clipping.
  - Allocator projection invariants: sums preserved, caps respected.
- [ ] Integration tests:
  - End-to-end run writes (signals → trades → dashboard payload) and includes barbell audit fields.
- [ ] Backtest regression suite:
  - Snapshot known periods and ensure deterministic outputs with fixed seeds/config.

### B5) Rollout protocol
- [ ] Shadow mode (compute effective confidence, **do not act**): log only.
- [ ] Canary mode (small capital slice or small universe): compare TS_ONLY vs BARBELL_SIZED daily.
- [ ] Promotion:
  - Requires Promotion criteria above + signoff.
- [ ] Kill switch:
  - Immediate disable via env/config; preserve audit trail.
