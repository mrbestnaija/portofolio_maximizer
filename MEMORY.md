# Phase 7.16 Baseline

## Status

Phase 7.16 auto-learning components exist in the codebase and were hardened to avoid
fail-open behavior, stale snapshot short-circuiting, and order-cache threshold dodges.

As of 2026-02-28, the repo also includes:

- An additive Monte Carlo forecast layer (`forcester_ts/monte_carlo_simulator.py`) that
  consumes existing forecast outputs without changing training, order selection, or
  default-model routing.
- Weather-context hydration through the real signal path (`utils/weather_context.py`,
  `ai_llm/signal_validator.py`, `models/time_series_signal_generator.py`,
  `models/signal_router.py`, `execution/paper_trading_engine.py`).
- A corrected emerging-market claim audit: broker-side stock enablement in
  `config/xtb_config.yml` does not by itself mean live XTB equity execution is implemented.
  Current truthful repo status is `partial` until a concrete XTB runtime adapter exists.
- Repo-level documentation has been synchronized so `README.md` and the XTB
  documentation set no longer overstate runtime broker readiness.

## Implemented Components

- `forcester_ts/order_learner.py`
  - AIC-ranked SQLite cache for learned model orders.
  - Regime-aware lookup with legacy `UNKNOWN` fallback compatibility.
  - Skip-grid now requires both qualified fit count and finite AIC evidence.

- `forcester_ts/model_snapshot_store.py`
  - `joblib` snapshot persistence with `manifest.json`.
  - Path traversal guard on snapshot load.

- `forcester_ts/var_backtest.py`
  - VaR computation, Kupiec POF, Christoffersen conditional coverage, pinball loss.

- `forcester_ts/shapley_attribution.py`
  - Exact power-set Shapley attribution for ensemble error decomposition.

- `forcester_ts/walk_forward_learner.py`
  - Rolling/expanding fold harness.
  - `order_used` now reflects live `OrderLearner` suggestions when present.

- `forcester_ts/monte_carlo_simulator.py`
  - Opt-in Monte Carlo price-path summaries for forecast uncertainty.
  - Explicit `SKIP` reasons when base forecast or dispersion inputs are missing.
  - Bounded path count (`MIN_PATHS=250`, `MAX_PATHS=10000`) to prevent low-quality
    simulations and runaway runtime.

## Hardening Applied

- `OrderLearner`
  - Non-finite AIC fits are ignored instead of being stored as zero-AIC winners.
  - Skip-grid cannot activate before the minimum qualification threshold.
  - Qualified coverage counts only rows with finite AIC evidence.

- `TimeSeriesForecaster`
  - Snapshot restore/save is now wired into `SARIMAX`, `GARCH`, and `SAMOSSA`.
  - Snapshot restore is exact-match only:
    - `strict_hash=True`
    - `max_obs_delta=0`
  - This preserves performance standards by preventing stale-model reuse on changed data.
  - `forecast(mc_enabled=True, ...)` now adds a Monte Carlo summary under
    `results["monte_carlo"]` without changing `mean_forecast`, ensemble logic, or
    order-learning thresholds.
  - `_regime_result` is initialized in the constructor so `forecast()` no longer
    short-circuits in restore/stub flows that do not call `fit()` first.

- `scripts/verify_emerging_market_claims.py`
  - No longer marks `emerging_market_equity_execution` as implemented from config alone.
  - Requires a concrete XTB execution adapter before the claim can graduate from
    `partial` to `implemented`.
  - Weather-risk auditing now verifies the full downstream wiring path, not just a
    single validator symbol.

- `scripts/check_order_learner_health.py`
  - Fails closed by default.
  - Missing DB, snapshot corruption/missing files, and smoke-test failures are errors.
  - `--allow-warn-pass` is the explicit opt-out for legacy fail-open behavior.

## Verification Evidence

- `python scripts/check_order_learner_health.py`
  - Result: `WARN`
  - Reason: existing backfilled `model_order_stats` rows are present, but none currently
    carry finite `best_aic` evidence, so qualified coverage is intentionally `0`.

- `python -m pytest tests/forcester_ts/test_order_learner.py tests/scripts/test_check_order_learner_health.py tests/forcester_ts/test_walk_forward_learner.py -q`
  - Result: `48 passed`

- `python -m pytest tests/forcester_ts/test_forecaster_snapshot_integration.py tests/forcester_ts/test_model_snapshot_store.py -q`
  - Result: `17 passed`

- `python -m pytest tests/forcester_ts/test_forecaster_rolling_cv_vs_random_walk.py tests/forcester_ts/test_forecaster_vs_random_walk_baseline.py -q`
  - Result: `4 passed`

- `python -m pytest tests/forcester_ts/test_forecaster_monte_carlo.py tests/forcester_ts/test_forecaster_vs_random_walk_baseline.py tests/forcester_ts/test_forecaster_rolling_cv_vs_random_walk.py -q`
  - Result: `8 passed`

- `python -m pytest tests/utils/test_weather_context.py tests/ai_llm/test_signal_validator.py tests/models/test_time_series_signal_generator.py tests/models/test_signal_router.py tests/execution/test_paper_trading_engine.py tests/scripts/test_verify_emerging_market_claims.py -q`
  - Result: `80 passed`

- `python scripts/verify_emerging_market_claims.py --json`
  - Result:
    - `emerging_market_equity_execution.status = "partial"`
    - `weather_risk_overlay.status = "implemented"`
    - summary = `implemented=5, partial=1, dormant=1, unsupported=3`

- `python -m pytest -m "not gpu and not slow" --tb=short -q`
  - Result: `1200 passed, 3 skipped, 28 deselected, 7 xfailed`

## Outstanding Note

This baseline hardens the auto-learning path and fixes claim-to-implementation gaps in
the order cache, snapshot reuse, walk-forward metadata, and health gating. It does not
yet solve the separate SARIMAX convergence warning volume in `logs/warnings/warning_events.log`.

## Documentation Sync

- `README.md` now reflects the Phase 7.16 baseline, additive Monte Carlo
  forecasting, and the current fast regression lane result.
- `Documentation/XTB_MIGRATION_SUMMARY.md` and
  `Documentation/XTB_INTEGRATION_GUIDE.md` are explicit that XTB work in this
  repo is currently configuration/planning only until a runtime adapter is added.
