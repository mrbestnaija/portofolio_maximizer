# Documentation Update Summary

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**  
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).  
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.  
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

---

## 2025-11-27 – TS Baseline, Ensemble Governance, NAV Shell

- **SAMOSSA as TS baseline**
  - SAMOSSA is now treated as the **primary Time Series baseline** for regression metrics and ensemble comparisons; SARIMAX is retained as a secondary candidate/fallback when SAMOSSA metrics are missing.
  - Forecast evaluation uses SAMOSSA-centric metrics in `forcester_ts/ensemble.derive_model_confidence`, so confidence scores are anchored to SAMOSSA performance instead of SARIMAX.

- **Directional-accuracy aware metrics**
  - `forcester_ts/metrics.compute_regression_metrics` now records `directional_accuracy` alongside RMSE, sMAPE, tracking error, and `n_observations`.
  - Ensemble confidence and TS health checks use directional accuracy as a trading-aligned signal (hit-rate on the sign of forecast vs realised return), not just RMSE/sMAPE.

- **Ensemble vs SAMOSSA brutal gate**
  - New CLI: `scripts/compare_forecast_models.py` aggregates `time_series_forecasts.regression_metrics` per ticker for `COMBINED` (ensemble) vs a baseline (`SAMOSSA` by default, `SARIMAX` optional).
  - `bash/comprehensive_brutal_test.sh` gained an “Ensemble vs SAMOSSA Regression Check” substage that:
    - Prints `Ens_RMSE`, `Base_RMSE`, `RMSE_Ratio`, `Ens_DA`, `Base_DA`, `DA_Delta`, and a flag per ticker.
    - Fails the TS stage when the fraction of underperforming tickers (RMSE ratio above threshold and/or DA below allowed delta) exceeds `--max-underperform-fraction`.
  - This keeps the TS ensemble **subordinate to the SAMOSSA baseline** when it does not add clear statistical and trading value.

- **NAV-centric barbell wiring (TS-first)**
  - `Documentation/NAV_RISK_BUDGET_ARCH.md` and `Documentation/NAV_BAR_BELL_TODO.md` now describe the TS-first, NAV-centric barbell shell:
    - TS core signals (Time Series forecaster + signal generator) feed **risk buckets** (`ts_core`, `ml_secondary`, `llm_fallback`, etc.).
    - A NAV allocator scales bucket-relative weights by per-bucket NAV budgets and risk scalers before the Taleb-style barbell shell (`config/barbell.yml`, `risk/barbell_policy.py`) enforces safe vs risk caps and per-market limits.
    - LLM remains a **capped fallback/overlay** in the risk sleeve; safe instruments and TS core retain priority over experimental ML/LLM components.

- **Documentation touched**
  - `Documentation/CRITICAL_REVIEW.md` – added “2025-11-26 TS Regression Governance Update” describing SAMOSSA baseline, directional accuracy, and the ensemble-vs-baseline brutal gate.
  - `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md` – extended “Regression Metrics & Backtesting” to include directional accuracy and referenced `scripts/compare_forecast_models.py`; cross-linked to NAV/barbell docs.
  - `Documentation/NAV_RISK_BUDGET_ARCH.md` – one-page architecture diagram for TS core → buckets → NAV allocator → barbell shell → orders, with LLM as capped fallback.
- Agent-facing guides (`AGENT_INSTRUCTION.md`, `AGENT_DEV_CHECKLIST.md`, roadmap/checkpoint docs) now point to the NAV/barbell docs as the **single source of truth** for TS-first risk wiring.

## 2025-11-30 – Sentiment overlay (profit-gated) scaffolding

- Added `Documentation/SENTIMENT_SIGNAL_INTEGRATION_PLAN.md` describing profit-gated sentiment ingestion/feature fusion, phased rollout (offline → shadow → limited impact → opt-in), and safeguards (burst/disagreement clamps, missing-data passthrough).
- Created `config/sentiment.yml` (disabled by default) and `tests/sentiment/test_sentiment_config_scaffold.py` to enforce strict gating (Sharpe ≥ 1.1, drawdown ≤ 0.22, PnL > 0 for 90/180d) until profitability clears the benchmark.
- Cross-referenced in `Documentation/arch_tree.md`, `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`, and `Documentation/implementation_checkpoint.md` for tracking without enabling runtime hooks yet.

## 2025-12-19 – Monetization guardrails + synthetic-first policy

- New `Documentation/MONITIZATION.md` centralizes the monetization gate (READY/EXPERIMENTAL/BLOCKED thresholds), synthetic-first pre-prod requirement, and usage-tracking expectations from `REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.
- CLIs noted: `python -m tools.check_monetization_gate --window 365`, `python -m tools.push_daily_signals --config config/monetization.yml`, `python -m tools.monetization_reality_check`.
- Reinforces that alerts/reports remain blocked when gate says BLOCKED and that pre-prod stays synthetic until quant health turns GREEN/YELLOW on live data.
- Synthetic generator Phase 2/3 hooks landed: `config/synthetic_data_config.yml` gains microstructure depth/order-imbalance knobs and liquidity shock events; `etl/synthetic_extractor.py` now emits Depth/OrderImbalance channels to better emulate illiquid markets without live feeds.
- Monetization gate currently unblocked with synthetic trades/metrics; swap in real pipeline/auto-trader runs ASAP. If switching DBs (e.g., `data/test_database.db`), call the gate with `--db-path ...` and ensure performance_metrics are populated in that store first.

## 2025-12-25 – Verification Snapshot + Status Reconcile

- Added `Documentation/PROJECT_STATUS.md` as the canonical “what is verified right now” snapshot (compile + focused pytest run).
- Updated integration/testing docs to remove stale “BLOCKED (2025-11-15)” framing now that the structural fixes are in place; remaining gating is profitability/quant-health evidence on live/paper data.

## 2025-12-26 – MVS Paper Window PASS + Exit/DB Hardening

- Added `scripts/run_mvs_paper_window.py` (deterministic replay runner) to quickly generate ≥30 **realised** trades from stored OHLCV for MVS validation.
- Verified **MVS PASS** on the 365‑day replay window; report artifact: `reports/mvs_paper_window_20251226_183023.md`.
- Exit safety: `ai_llm/signal_validator.SignalValidator` now treats exposure‑reducing exits as always executable (won’t block liquidation/time exits with trend/regime guardrails).
- WSL SQLite reliability: `etl/database_manager.DatabaseManager` now cleans stale `*-wal`/`*-shm` artifacts on `/mnt/*` paths and refreshes mirrors defensively to avoid stale/unsynced state.

---

## 2025-11-24 – Frontier Universe & Data-Source-Aware Tickers

- Introduced `Documentation/DATA_SOURCE_AWARE_TICKER_UNIVERSE.md` and `etl/data_universe.py` for a data-source-aware, config-driven ticker discovery flow.
- Wired `scripts/run_auto_trader.py` and multi-ticker scripts through `DataUniverse` so frontier tickers (Nigeria → Bulgaria atlas) are consistently included when `--include-frontier-tickers` is enabled.
- Added `tests/etl/test_data_universe.py` for offline/no-network coverage.

---

## 2025-11-18 – SQLite Self-Heal & Forecast Instrumentation

- `etl/database_manager.py` now:
  - Detects `"database disk image is malformed"` / `"database is locked"` errors,
  - Backs up the corrupted store,
  - Rebuilds a clean SQLite file,
  - Resets the connection and retries the write.
- TS instrumentation (`forcester_ts/instrumentation.py`) logs dataset diagnostics and regression metrics (RMSE, sMAPE, tracking error) to `logs/forecast_audits/*.json` so dashboards and brutal logs share a common evidence base.

---

## 2025-11-06 – TS Signal Generation Refactor (Snapshot)

- Time Series ensemble promoted to **DEFAULT signal generator**, with LLM as fallback/redundancy:
  - `models/time_series_signal_generator.py` converts TS forecasts to trading signals.
  - `models/signal_router.py` routes TS-first, LLM-fallback signals.
  - `models/signal_adapter.py` provides a unified signal envelope.
- Test coverage increased from ~196 to **246 tests** (38 new unit + 12 new integration tests).
- All major roadmap, checkpoint, and to-do docs were synchronized to say:
  - “Time Series ensemble is DEFAULT signal generator.”
  - “LLM signals serve as fallback/redundancy.”

For detailed historical context on the refactor, see the earlier snapshot docs:
- `REFACTORING_IMPLEMENTATION_COMPLETE.md`
- `REFACTORING_STATUS.md`
- `TESTING_IMPLEMENTATION_SUMMARY.md`
- `INTEGRATION_TESTING_COMPLETE.md`
- `TIME_SERIES_FORECASTING_IMPLEMENTATION.md`
