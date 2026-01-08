# NAV-Centric Barbell Architecture (TS-First, LLM Fallback)

> **Purpose**: One-page view of how Time Series (TS) core signals, NAV-centric risk budgeting, the Taleb barbell shell, and LLM fallback fit together in `portfolio_maximizer` v45.
> **Current status (2026-01-07)**: Barbell policy is implemented; NAV tracker/allocator + risk bucket budgets are still planned (see `Documentation/PROJECT_STATUS.md`).

**Project-wide sequencing**: See `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` for the step-by-step plan to make TS signals bar-aware, horizon-consistent, and execution-cost-aligned before scaling NAV/bucket automation.

---

## 1. High-Level Flow

Textual data/control flow:

1. **Data & Forecasts**
   - ETL + caching → TS forecaster (`forcester_ts/*`, `etl/time_series_forecaster.py`).
   - TS ensemble outputs per-ticker forecast bundles + diagnostics.
2. **Signal Generation (TS-First)**
   - `models/time_series_signal_generator.TimeSeriesSignalGenerator`:
     - Converts TS forecasts → structured TS signals (action, expected_return, confidence, risk).
     - Enforces quant-success criteria from `config/quant_success_config.yml` and logs audits to `logs/signals/quant_validation.jsonl`.
   - `models.signal_router.SignalRouter`:
     - Treats TS as **primary**, LLM as **fallback/redundancy** only when enabled.
3. **Signal Buckets**
   - TS core signals and any secondary ML/LLM signals are wrapped into a unified `Signal` envelope and assigned to buckets:
     - `safe` (cash/T-bills/short-duration ETFs; optional safe strategies).
     - `ts_core` (primary TS ensemble-based sleeve).
     - `ml_secondary` (non-TS ML sleeves, optional).
     - `llm_fallback` (tiny NAV slice; only when TS/ML are weak or silent).
     - Future: `tail_hedge`, `ngx_risk`, `em_risk` for options/frontier exposure.
4. **NAV Allocator**
   - A NAV tracker computes total NAV from positions + prices.
   - Bucket-relative weights (per sleeve) are scaled by:
     - Per-bucket NAV budgets from `config/risk_buckets.yml`.
     - Per-bucket risk scaling factors (based on realized vol/drawdown), as described in `Documentation/NAV_BAR_BELL_TODO.md`.
   - Output: **pre-barbell absolute target weights per symbol**, expressed as fractions of NAV.
5. **Barbell Shell (Taleb)**
   - `config/barbell.yml` + `risk/barbell_policy.BarbellConstraint`:
     - Enforce global safe vs risk bucket caps (e.g. 70–95% safe, ≤25% risk).
     - Enforce per-market caps (NGX, crypto, EM) once configured.
   - A barbell-aware quant gate (when `enable_barbell_validation=true`) can temporarily disable risk-bucket tickers if `logs/signals/quant_validation.jsonl` indicates unhealthy TS signal behaviour (wired through `scripts/check_quant_validation_health.py` and `scripts/run_auto_trader.py`).
6. **Orders & Execution**
   - Final target weights (post-NAV + barbell) are translated into orders.
   - `execution.paper_trading_engine.PaperTradingEngine` executes trades, logs fills and realized PnL, and updates the portfolio.
   - Dashboards (`visualizations/dashboard_data.json`) include:
     - PnL, win_rate, routing stats.
     - `forecaster_health` (forecaster + TS metrics).
     - `quant_validation_health` (global PASS/FAIL/expected_profit metrics from `quant_validation.jsonl`).
   - Diagnostic liquidation (`scripts/liquidate_open_trades.py`) is reserved for research workflows (e.g. quant threshold sweeps); it now honours `asset_class` / `instrument_type` hints and, when present, decomposes `instrument_type='synthetic'` trades via the `synthetic_legs` table and `etl/synthetic_pricer.py`. See `Documentation/MTM_AND_LIQUIDATION_IMPLEMENTATION_PLAN.md` and `Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md` for the current mark-to-market policies used in experiments.

---

## 2. Responsibilities by Layer

- **TS Core (Primary Signals)**
  - Modules: `forcester_ts/*`, `etl/time_series_forecaster.py`, `models/time_series_signal_generator.py`.
  - Role: Generate the main profit-critical signals (BUY/SELL/HOLD + metrics) from TS forecasts.
  - Guardrails: Tier-1 stack (`Documentation/QUANT_TIME_SERIES_STACK.md`), quant-success thresholds (`config/quant_success_config.yml`), TS-first routing contract (`config/signal_routing_config.yml`, `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`).

- **LLM Fallback (Overlay Only)**
  - Modules: `ai_llm/*`, `models.signal_router.SignalRouter`, `scripts/run_auto_trader.py`.
  - Role: Provide optional overlay/fallback signals where TS is weak or unavailable.
  - Guardrails: Feature flags, small NAV budget (via `config/risk_buckets.yml`), and barbell/risk buckets treat LLM as part of the **risk sleeve**, never the safe core.

- **NAV & Buckets**
  - Modules/Configs: `portfolio/nav_tracker.py` (planned), `config/risk_buckets.yml`, `Documentation/NAV_BAR_BELL_TODO.md`.
  - Role: Convert bucket-relative target weights into portfolio-level weights using NAV, per-bucket NAV budgets, and dynamic risk scaling, before any barbell projection.

- **Barbell Shell**
  - Modules/Configs: `config/barbell.yml`, `risk/barbell_policy.py`, `Documentation/BARBELL_OPTIONS_MIGRATION.md`, `Documentation/BARBELL_INTEGRATION_TODO.md`.
  - Role: Enforce Taleb-style safe vs risk allocation, per-market caps, and optional barbell-aware gating around TS/LLM risk sleeves.

- **Monitoring & CI Hooks**
  - Modules/Docs: `scripts/check_quant_validation_health.py`, `scripts/summarize_quant_validation.py`, `Documentation/SYSTEM_ERROR_MONITORING_GUIDE.md`, `Documentation/CHECKPOINTING_AND_LOGGING.md`.
  - Role: Fail CI/brutal runs when quant validation degrades, surface TS/LLM health and barbell status in logs/dashboards.

---

## 3. Where to Read More

- NAV + bucket design and TODOs: `Documentation/NAV_BAR_BELL_TODO.md`.
- Taleb barbell & options migration: `Documentation/BARBELL_OPTIONS_MIGRATION.md`.
- Barbell integration roadmap (LLM Phase A, ML Phase B, antifragility, NGX): `Documentation/BARBELL_INTEGRATION_TODO.md`.
- TS-first implementation details: `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`.
- Unified roadmap and project status: `Documentation/UNIFIED_ROADMAP.md`, `Documentation/implementation_checkpoint.md`, `Documentation/arch_tree.md`.
