# NAV-Centric Barbell Integration – TODO Checklist

## 0. Scope & Principles

**Current status (2026-01-07)**: Barbell policy exists (`config/barbell.yml`, `risk/barbell_policy.py`), but NAV tracking/allocator + risk buckets remain unimplemented (see `Documentation/PROJECT_STATUS.md`).

- [ ] Treat **NAV as the central risk budget** for all sleeves (safe, TS core, ML, LLM, tail hedges, NGX/frontier).
- [ ] Keep **Time Series (TS) forecasts as primary signal source**, with **LLM strictly as fallback/overlay**.
- [ ] Preserve existing **Taleb barbell shell** (`config/barbell.yml`, `risk/barbell_policy.py`) and add NAV/risk-budget logic only as an *outer* layer.
- [ ] Avoid Medallion-style HFT; target **daily / multi-day horizons** with NAV-based scaling and modest gross leverage.

**Project-wide sequencing**: See `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` for the prerequisite TS horizon + execution realism + reporting fixes that should be completed before promoting NAV allocator work to production.

---

## Delta (2026-01-18)

- Dashboard reporting now supports trade-level visualization (`trade_events`, `price_series`, `positions`) in `visualizations/dashboard_data.json`, and `visualizations/live_dashboard.html` polls it every 5 seconds. Canonical producer is `scripts/dashboard_db_bridge.py` (DB→JSON) started by bash orchestrators; audit snapshots persist to `data/dashboard_audit.db` by default (`--persist-snapshot`).

## 1. NAV Tracking & Exposure Surface

- [ ] Create `portfolio/nav_tracker.py` (or equivalent helper):
  - [ ] Define a lightweight `PortfolioSnapshot` (cash, positions, optional FX).
  - [ ] Implement `compute_nav(snapshot, prices) -> float` as the single source of truth.
- [ ] Expose NAV to:
  - [ ] `scripts/run_auto_trader.py` (auto-trader loop).
  - [ ] Backtest runners in `backtesting/` (so historical simulations match live sizing logic).
- [ ] Document NAV conventions:
  - [ ] Base currency and FX handling (if applicable).
  - [ ] How derivative/option exposures will be valued once `options_trading.enabled=true` (reference `config/options_config.yml` but keep inert for now).

---

## 2. Risk Buckets & NAV Budgets

- [ ] Add `config/risk_buckets.yml` to describe **NAV budgets per bucket**:
  - [ ] `safe` (cash/T-bills/short-duration ETFs).
  - [ ] `ts_core` (primary TS forecaster sleeve).
  - [ ] `ml_secondary` (non-TS ML models).
  - [ ] `llm_fallback` (LLM overlay; tiny NAV fraction).
  - [ ] `tail_hedge` (far OTM options / synthetic convexity; future).
  - [ ] `ngx_risk` / `em_risk` (frontier and EM sleeves).
- [ ] For each bucket, define:
  - [ ] `base_nav_frac`, `min_nav_frac`, `max_nav_frac` (fractions of total NAV).
  - [ ] Notes on interaction with `config/barbell.yml` (safe vs risk buckets).
- [ ] Write a small loader (e.g. `risk/risk_buckets_config.py`) to provide a typed view of these budgets to the allocator.

---

## 3. NAV Allocator & Barbell Shell

- [ ] Implement a **NAV allocator** module (e.g. `risk/nav_allocator.py`) that:
  - [ ] Accepts per-bucket *relative* weights (e.g. from TS core / ML / LLM engines).
  - [ ] Reads NAV and `risk_buckets.yml` budgets.
  - [ ] Produces **pre-barbell absolute target weights**:
    - `w_symbol = rel_weight_in_bucket × bucket_nav_frac × bucket_risk_scale`.
- [ ] Wire NAV allocator into:
  - [ ] `scripts/run_auto_trader.py` before orders are computed.
  - [ ] Backtesting engine(s) before portfolio transitions are simulated.
- [ ] Keep `risk/barbell_policy.BarbellConstraint` as the outer Taleb layer:
  - [ ] Call `project_to_feasible(weights)` *after* NAV allocation to enforce:
    - Safe vs risk bucket caps from `config/barbell.yml`.
    - Per-market caps (NGX/crypto/EM) once implemented.

---

## 4. Dynamic Risk Scaling (Medallion-Style Overlay)

- [ ] Add `risk/risk_scaler.py` to compute **per-bucket risk scaling factors**:
  - [ ] Define `BucketRiskState` (realized vol, current drawdown, max_drawdown, target_vol).
  - [ ] Define `RiskScalerConfig` per bucket (vol sensitivity, DD sensitivity, min/max scale).
  - [ ] Implement `compute_scaling_factor(bucket_name, state) -> float`.
- [ ] Track simple per-bucket PnL time series:
  - [ ] Backtests: derive from simulated equity curves.
  - [ ] Live/paper: derive from `etl.database_manager.get_performance_summary()` plus bucket attribution.
- [ ] Integrate scaling into NAV allocator:
  - [ ] `effective_bucket_nav_frac = base_nav_frac × risk_scaler_factor`.
  - [ ] Apply *before* barbell projection, with clamps to `[min_nav_frac, max_nav_frac]`.

---

## 5. Signal Priority: TS Core First, LLM as Fallback

- [ ] Create a small signal envelope in `signals/types.py`:
  - [ ] `Signal(symbol, direction, strength, edge_score, confidence, source)`.
- [ ] Implement **TS core signal engine** (if not already explicit):
  - [ ] Wrap existing `TimeSeriesForecaster` logic to output `Signal(source="time_series_core")`.
  - [ ] Define an `edge_score` (e.g. expected_return / predicted_vol) and `confidence` per symbol.
- [ ] Implement **LLM fallback engine** as a separate bucket:
  - [ ] Only generate LLM signals when:
    - TS and ML signals are missing or below their configured thresholds.
  - [ ] Tag all such signals as `source="llm_fallback"`.
- [ ] Implement a **signal router** helper that:
  - [ ] Aggregates signals from all sources.
  - [ ] Assigns each signal to a bucket (`ts_core`, `ml_secondary`, `llm_fallback`, etc.).
  - [ ] Produces per-bucket *relative* weights for the NAV allocator.

---

## 6. NAV-Centric Barbell Execution Path

- [ ] In `scripts/run_auto_trader.py`, refactor the main cycle to follow:
  1. **Snapshot**: read positions, prices, compute `NAV` (NAVTracker).
  2. **Signal generation**:
     - TS core (primary).
     - ML secondary (optional).
     - LLM fallback (only where higher-priority sources are weak/silent).
  3. **Bucket aggregation**: map signals → bucket-relative weights.
  4. **NAV allocation**: apply `risk_buckets.yml` budgets and per-bucket risk scaling.
  5. **Barbell enforcement**: call `BarbellConstraint.project_to_feasible`.
  6. **Order sizing**: convert final target weights to orders and pass to `PaperTradingEngine`.
- [ ] Ensure new logic is **fully bypassed** (no behaviour change) when:
  - [ ] `barbell.enable_barbell_allocation == false` and NAV allocator feature flag is off.

---

## 7. Monitoring, CI & Quant Health Hooks

- [ ] Ensure `scripts/check_quant_validation_health.py` is part of CI and brutal runs:
  - [ ] Fail when `FAIL` fraction or negative `expected_profit` fraction exceeds `forecaster_monitoring.quant_validation` ceilings.
- [ ] Expose NAV + bucket allocations in dashboards:
  - [ ] Extend `visualizations/dashboard_data.json` payload with:
    - Current NAV.
    - Per-bucket NAV fractions (safe, ts_core, ml_secondary, llm_fallback, etc.).
    - Per-bucket risk scaling factors.
  - [ ] Show when **barbell quant gate** is actively disabling risk-bucket tickers.
- [ ] Add a simple regression test:
  - [ ] Backtest run that verifies safe NAV never drops below configured floor under the NAV allocator + barbell shell.

---

## 8. Documentation & Complexity Control

- [ ] Keep all **NAV/barbell/Medallion-style details** centralised in this file and a single architecture overview (e.g. `Documentation/NAV_RISK_BUDGET_ARCH.md`).
- [ ] In existing docs:
  - [ ] `Documentation/BARBELL_OPTIONS_MIGRATION.md` – reference this TODO for NAV-centric budgeting; avoid duplicating formulas and defer MTM specifics to `Documentation/MTM_AND_LIQUIDATION_IMPLEMENTATION_PLAN.md`.
  - [ ] `Documentation/BARBELL_INTEGRATION_TODO.md` – link to this file for NAV/bucket work; keep that file focused on high-level phases (A/B, antifragility, NGX).
- [ ] When implementing, prefer:
  - [ ] Small, composable modules (`NAVTracker`, `NAVAllocator`, `RiskScaler`).
  - [ ] Clear feature flags so the legacy spot-only path remains easy to reason about.

---

## 9. Barbell PnL Evaluation Protocol (Do Not Promote Without Evidence)

**Goal**: Quantify whether barbell constraints improve realized PnL outcomes versus TS-only sizing, **before** enabling any barbell/NAV allocator logic in production-like runs.

### 9.1 What this protocol does (and does not do)

- **Does**: compare two backtests on the same OHLCV window using the same forecasting/signal logic:
  - `TS_ONLY`: baseline walk-forward backtest.
  - `BARBELL_SIZED`: same backtest, but applies barbell bucket sizing multipliers (safe/core/spec) to the signal confidence before execution sizing (feature-flagged in the harness; does not change production config files).
- **Does not**: implement a full NAV allocator, bucket budgets, or options legs. This is a *first-order sizing audit* to decide if barbell sizing is promising.

### 9.2 Preconditions

- Run under WSL `simpleTrader_env` and record the runtime fingerprint (see `Documentation/RUNTIME_GUARDRAILS.md`).
- Ensure the DB contains OHLCV for the evaluation tickers and window (or run ETL first).

### 9.3 Evaluation universe (must include safe + risk)

Include at least:
- Safe bucket tickers from `config/barbell.yml` (e.g., `SHY,BIL,IEF`)
- Core/speculative tickers from `config/barbell.yml` (e.g., `MSFT,CL=F,MTN,AAPL,BTC-USD`)

### 9.4 Run command (feature-flagged harness)

```bash
./simpleTrader_env/bin/python scripts/run_barbell_pnl_evaluation.py \
  --tickers "SHY,BIL,IEF,MSFT,CL=F,MTN,AAPL,BTC-USD" \
  --lookback-days 180 \
  --forecast-horizon 14 \
  --report-path reports/barbell_pnl_eval.json \
  --write-promotion-evidence reports/barbell_pnl_evidence_latest.json
```

For promotion gating + evidence readiness tasks, track progress in:
- `Documentation/BARBELL_PNL_EVIDENCE_AND_PROMOTION_TODO.md`

If (and only if) `reports/barbell_pnl_evidence_latest.json` shows `passed=true`, you may enable the feature-flagged sizing overlay in `scripts/run_auto_trader.py`:

```bash
export ENABLE_BARBELL_SIZING=1
export BARBELL_PROMOTION_EVIDENCE_PATH=reports/barbell_pnl_evidence_latest.json
```

### 9.5 Acceptance criteria (minimum bar to claim “improves PnL”)

Barbell is only considered “helpful” when all are true:
- `BARBELL_SIZED.total_return > TS_ONLY.total_return` (PnL up in USD)
- `BARBELL_SIZED.profit_factor >= TS_ONLY.profit_factor` (no PF regression)
- `BARBELL_SIZED.max_drawdown <= TS_ONLY.max_drawdown` (risk not worse)
- Trade count remains sufficient for inference (`total_trades >= 30`) or the result is marked “inconclusive”.

### 9.6 Outputs (audit artifacts)

The harness writes a JSON report with both metric bundles and deltas:
- `reports/barbell_pnl_eval.json`

This report should be attached whenever proposing:
- enabling `barbell.enable_barbell_allocation`, or
- adding NAV allocator integration.
