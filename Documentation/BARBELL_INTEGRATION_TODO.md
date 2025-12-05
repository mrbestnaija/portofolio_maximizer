# TODO – Taleb Barbell Strategy Integration

> Scope: implement a **Taleb-style barbell framework** across allocation, LLM Phase A, ML Phase B, antifragility tests, and NGX/frontier exposure – without violating guardrails/phase gating in `UNIFIED_ROADMAP.md`.

---

## 0. Pre-Flight Checks (Meta)

- [ ] Re-read:
  - [ ] `UNIFIED_ROADMAP.md` (current v45 status + Phase A/B plan)
  - [ ] `AGENT_INSTRUCTION.md` / `AGENT_DEV_CHECKLIST.md` (guardrails, phase gates)
- [ ] Confirm:
  - [ ] LLM pipeline (`ai_llm/`) is in place and tests pass
  - [ ] `portfolio_math.py` public API is stable
  - [ ] Paper trading engine spec (`execution/paper_trading_engine.py`) matches roadmap assumptions

---

## 1. Config & Domain Model – Barbell Concepts

### 1.1 Global Barbell Config

- [x] Create `config/barbell.yml`:
  - [x] `safe_bucket.symbols` – list of tickers treated as “safe”
  - [x] `risk_bucket.symbols` – list of tickers/venues treated as “risk”
  - [x] `safe_bucket.min_weight` / `safe_bucket.max_weight`
  - [x] `risk_bucket.max_weight` (implicitly `1 - safe_weight`)
  - [x] Flags:
    - [x] `enable_barbell_allocation: false` (feature-flag, disabled by default)
    - [x] `enable_barbell_validation: false`
    - [x] `enable_antifragility_tests: false` (toggle per run)

### 1.2 Instrument & Strategy Metadata

- [ ] Extend instrument metadata (wherever universe is defined) with:
  - [ ] `risk_class: "safe" | "risk" | "neutral"` (default = `neutral`)
  - [ ] Optional: `convexity_role: "tail_hedge" | "beta"` for later use
- [ ] Add optional strategy-level tags (Phase B):
  - [ ] `strategy_type: "safe" | "risk"` in strategy config
- [ ] Tests:
  - [ ] Fail if an instrument is tagged both safe and risk
  - [ ] Fail if `safe_bucket.min_weight > safe_bucket.max_weight`
  - [ ] Warn if no instruments fall into a bucket

---

## 2. Portfolio Math & Allocation Layer

### 2.1 Barbell Constraint Logic

- [x] Implement `BarbellConstraint` (see `risk/barbell_policy.py`):
  - [x] `bucket_weights(weights, instrument_metadata) -> (w_safe, w_risk, w_other)` via symbol lists.
  - [x] Constraint: `w_safe ∈ [min_safe, max_safe]`, `w_risk ≤ max_risk`.
  - [x] `project_to_barbell_feasible_region(weights)`:
    - [x] Minimal adjustment to satisfy barbell bounds.
    - [x] Preserve non-negativity and leverage constraints; no behaviour change when barbell is disabled.
- [ ] Integrate into optimizer and NAV allocator:
  - [ ] If optimizer supports constraints: add barbell explicitly.
  - [ ] Else: post-process via `project_to_barbell_feasible_region`.
  - [ ] Coordinate with NAV-centric risk budgeting work in `Documentation/NAV_BAR_BELL_TODO.md` so safe/risk bucket weights and NAV fractions stay consistent.
  - [x] Config toggle: `enable_barbell_allocation=false` leaves behavior unchanged (current default).

### 2.2 Reporting & Analytics

- [ ] Log `{safe_weight, risk_weight, other_weight}` per portfolio/backtest
- [ ] Persist time-series of bucket weights
- [ ] Diagnostics: count barbell violations; distribution of `safe_weight`
- [ ] Tests: unit for constraint math; integration on toy universe (2 safe, 2 risk)

---

## 3. Phase A – LLM Barbell Integration

### 3.1 Signal Validator Hooks (`ai_llm/signal_validator.py`)

- [ ] Extend signal schema to carry `ticker` → `risk_class`; optional `risk_bucket_override`
- [ ] `_validate_barbell_constraints(signal, current_portfolio)`:
  - [ ] Compute current `{w_safe, w_risk}`
  - [ ] Simulate proposed trade; reject/downsize if `w_safe` < min or `w_risk` > max
  - [ ] Return warnings in `ValidationResult`
- [ ] Auto-classify LLM signals:
  - [ ] If `ticker ∈ risk_bucket.symbols` → risk trade
  - [ ] If `ticker ∈ safe_bucket.symbols` → safe trade
  - [ ] Else neutral (config: disallow or count toward risk)

### 3.2 Paper Trading Engine (`execution/paper_trading_engine.py`)

- [ ] In `execute_signal`, run barbell validation; reject/scale as needed
- [ ] Track after each trade:
  - [ ] `safe_weight`, `risk_weight`
  - [ ] Barbell-based rejections/scalings
- [ ] Summary stats: avg weights, worst attempted violations

### 3.3 Tests – Phase A

- [ ] Unit: `_validate_barbell_constraints` on synthetic portfolios
- [ ] Integration: risk-heavy signal stream never breaches `safe_min`; caps enforced

---

## 4. Phase B – ML & Strategy Barbell Integration

### 4.1 Strategy Tagging

- [ ] Add `strategy_type: "safe" | "risk"` per strategy; document rationale
- [ ] Loader enforces presence; warns on mismatched symbols vs type

### 4.2 Meta Allocation

- [ ] Meta allocator combines strategies respecting:
  - [ ] Strategy-level barbell (e.g., 70–90% safe, 10–30% risk)
  - [ ] Asset-level barbell (Section 2)
- [ ] Config: `barbell.strategy.safe_min_weight`, `barbell.strategy.risk_max_weight`

### 4.3 Backtest Evaluation

- [ ] Report safe vs risk strategy contributions
- [ ] Track bleed in calm periods; payoff in volatile/crisis periods
- [ ] Tests: toy safe + risk strategies honor caps

---

## 5. Antifragility Test Suite

### 5.1 Scenario Generators

- [ ] `vol_scaled_scenarios(returns, scales=[1.25, 1.5, 2.0])`
- [ ] `jump_injection_scenarios(returns, jump_prob, jump_size_dist)`
- [ ] `param_perturbation_scenarios(model_params, epsilons)`

### 5.2 Antifragility Metrics

- [ ] `delta_pnl_vs_vol_scale(portfolio, scenarios)`
- [ ] `tail_ratio(portfolio_returns)`
- [ ] `probability_of_ruin(portfolio_returns, threshold)`
- [ ] Integrate into backtest reports: vol amplification, jump behavior, ruin probability

### 5.3 Gating Rules

- [ ] Safe bucket: max drawdown threshold under all scenarios
- [ ] Risk bucket: no dramatic ruin-probability increase; improved/neutral under higher vol
- [ ] CI: antifragility tests are non-optional gate for “risk” strategy promotion

---

## 6. Emerging Market / NGX Integration & Barbell

- [ ] Tag NGX equities as `risk` by default; NGX bills/sovereigns as `safe` or `local_safe` mapped to safe bucket via config
- [ ] Extend config with per-market caps (e.g., `max_weight_ngx`, `max_weight_em`)
- [ ] Update `BarbellConstraint` for nested constraints: global `{w_safe, w_risk}` and per-market caps
- [ ] Backtest NGX strategies: ensure confinement to risk bucket; measure crisis-period contribution and antifragility impact

---

## 7. Monitoring, Metrics & Ops

- [ ] Runtime metrics: `safe_weight_current`, `risk_weight_current`, barbell violations, barbell-scaled trades, bucket PnL
- [ ] Dashboard/report: time series of safe vs risk weights, cumulative PnL by bucket, distribution of barbell rejections

---

## 8. Documentation

- [ ] Create `docs/BARBELL_DESIGN.md` (concept, config, before/after examples)
- [ ] Update `UNIFIED_ROADMAP.md` with Phase A/B barbell milestones
- [ ] Cross-link NGX/emerging market docs to barbell design (NGX equities live in risk bucket)

---

## 9. Final Validation Before Any Live Capital

- [ ] Run full historical backtests with barbell enabled across DM + EM/NGX proxies
- [ ] Compare baseline vs barbell-constrained portfolios
- [ ] Confirm: no increase in ruin probability; tail behavior improves/does not worsen; guardrails/phase gating respected

---

## 10. Data Providers & NAV Feed for NAV-Centric Barbell (High ROI, Low Intrusion)

> Goal: make the barbell NAV-aware using a small number of high-leverage provider upgrades, without breaking existing spot-only behaviour.

### 10.1 Provider Metadata & Routing

- [ ] Extend `config/data_sources_config.yml` with minimal provider metadata (no behaviour change when unset):
  - [ ] `asset_classes: ["equity", "etf", "fx", "options", ...]`
  - [ ] `supports.spot_ohlcv`, `supports.options_chains`, `supports.options_history`, `supports.greeks`, `supports.rates`, `supports.nav_feed`
- [ ] Update `etl/data_source_manager.DataSourceManager` to *optionally* route by `(asset_class, usage)` when metadata is present, falling back to existing priority/fallback logic otherwise:
  - [ ] Daily spot (equity/index): yfinance → finnhub → alpha_vantage (current behaviour)
  - [ ] Intraday spot: finnhub → yfinance (when configured)
  - [ ] Options chains/history (Phase O1/O2): prefer dedicated options-capable provider when configured (e.g. finnhub/polygon), else fall back to yfinance’s current-chain-only API with reduced scope
  - [ ] Rates: alpha_vantage (and future FRED) for risk-free curves

### 10.2 Options Data Layer (Schema + Manager)

- [ ] Design `options_contracts` and `options_quotes` tables (Phase O1 in `BARBELL_OPTIONS_MIGRATION.md`):
  - [ ] Per-contract reference data: `underlying_symbol`, `contract_id`, `option_type`, `strike_price`, `expiry_date`, `exercise_style`, `multiplier`
  - [ ] Per-contract market data per bar: `bid`, `ask`, `last`, `volume`, `open_interest`, `implied_volatility`, optional Greeks
- [ ] Implement a thin `OptionsDataManager` that:
  - [ ] Reads from existing providers (yfinance/Finnhub/next options vendor) using their native APIs
  - [ ] Normalises into the shared schema and persists via `etl/database_manager.DatabaseManager`
  - [ ] Exposes NAV-aware queries for barbell evaluators (premium at risk per underlying, IV term structure snapshots)
- [ ] Keep Phase O1/O2 limited to a **small, liquid universe** (e.g. SPY/QQQ/large tech) to maximise reward-to-effort.

### 10.3 NAV Feed & Broker Integration (Production-Only)

- [ ] Reuse existing broker work (`execution/ctrader_client.py`, future massive.com/polygon.io/Tradier clients) to surface:
  - [ ] Account NAV / equity by currency
  - [ ] Margin used / available
  - [ ] Realised PnL and commission/fee history
- [ ] Plumb a **read-only NAV endpoint** into the barbell/NAV allocator so:
  - [ ] Backtests continue to use simulated NAV from historical prices (no change)
  - [ ] Live/paper runs can optionally override simulated NAV with broker NAV when the broker client is configured (feature-flagged, disabled by default)
- [ ] Add a minimal dashboard tile / log summary for NAV vs barbell buckets so operators can see safe/risk weights as fractions of *actual* NAV when broker APIs are enabled.
