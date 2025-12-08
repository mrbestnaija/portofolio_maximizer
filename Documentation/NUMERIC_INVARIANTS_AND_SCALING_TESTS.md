## Numeric invariants and scaling checks

This document summarizes the key numeric invariants around normalization, scaling,
renormalization, and floating-point precision in the forecasting and dashboard
stack, and how they are enforced by tests.

### 1. Ensemble weights and convexity

Components:
- `forcester_ts.ensemble.EnsembleCoordinator._normalize`
- `forcester_ts.forecaster.TimeSeriesForecaster._enforce_convexity`

Invariants:
- Negative and zero weights are discarded before normalization.
- Remaining weights are strictly non-negative and form a convex combination
  (sum to 1.0 within a small floating-point tolerance).
- Repeated renormalization of existing weights does not drift away from a
  proper simplex (no accumulation of numeric error).

Coverage:
- `tests/forcester_ts/test_ensemble_and_scaling_invariants.py::test_ensemble_normalize_filters_non_positive_and_sums_to_one`
- `tests/forcester_ts/test_ensemble_and_scaling_invariants.py::test_ensemble_weights_stable_under_re_normalization`
- `tests/forcester_ts/test_ensemble_and_scaling_invariants.py::test_enforce_convexity_handles_degenerate_and_negative_weights`

### 2. SAMOSSA normalization and scaling

Component:
- `forcester_ts.samossa.SAMOSSAForecaster`

Invariants:
- When `normalize=True`:
  - Input series are standardized to approximately zero-mean and unit-variance
    before decomposition.
  - Reported `normalized_mean` and `normalized_std` in `get_model_summary()`
    match this standardization within a small tolerance.
- When `normalize=False`:
  - `scale_mean` is 0.0 and `scale_std` is 1.0 (identity scaling).
  - `normalized_mean` and `normalized_std` reflect the raw cleaned series
    statistics (so downstream code can still reason about the data scale).

Coverage:
- `tests/forcester_ts/test_ensemble_and_scaling_invariants.py::test_samossa_normalization_produces_zero_mean_unit_std`
- `tests/forcester_ts/test_ensemble_and_scaling_invariants.py::test_samossa_without_normalization_reports_raw_stats`

### 3. GARCH scaling / rescaling (renormalization of scale)

Component:
- `forcester_ts.garch.GARCHForecaster`

Invariants:
- Returns may be scaled by a factor (e.g. 100x) before fitting to satisfy the
  `arch` library’s recommended scale for stable convergence.
- Forecasts are rescaled back to the original return scale:
  - Variance scales as `scale_factor^2` and is divided by that factor.
  - Mean scales linearly and is divided by `scale_factor`.
  - Volatility is always `sqrt(variance_forecast)` after rescaling.
- When `_scale_factor == 1.0`, forecasts are left unchanged.

Coverage:
- `tests/forcester_ts/test_ensemble_and_scaling_invariants.py::test_garch_rescaling_inverse_of_input_scaling`
- `tests/forcester_ts/test_ensemble_and_scaling_invariants.py::test_garch_no_rescaling_when_scale_factor_one`

### 4. How to run these checks

From the project root:

```bash
pytest tests/forcester_ts/test_ensemble_and_scaling_invariants.py
```

This runs a focused suite that:
- Exercises normalization / renormalization of ensemble weights.
- Verifies SAMOSSA’s normalization and reporting of scale.
- Validates GARCH scaling and rescaling logic without requiring a live `arch`
  installation or actual model fitting.

The tests intentionally use tight `pytest.approx` tolerances to catch subtle
floating-point and scaling regressions in future edits.

### 5. Time Series signal gating and diagnostics

Recent changes (2025-12-04) added downstream **quant-success gating** and diagnostic toggles:

- `models.time_series_signal_generator.TimeSeriesSignalGenerator` now attaches a quant-validation profile (driven by `config/quant_success_config.yml`) and demotes BUY/SELL to HOLD when `status == "FAIL"` outside diagnostic modes. This logic sits on top of the numeric invariants described above.
- Diagnostic runs (`DIAGNOSTIC_MODE`/`TS_DIAGNOSTIC_MODE`) still relax thresholds and disable quant validation so scaling invariants can be exercised without profit-factor/win-rate gates interfering. Production tests should run with these toggles off to observe the full effect of quant gating on realised PnL.

For how these gates are monitored and tuned over time, see:

- `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` for GREEN/YELLOW/RED tiers and CI/brutal behaviour.
- `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md` for automation around TS threshold sweeps, transaction cost calibration, and config proposals.
- `Documentation/OPTIMIZATION_IMPLEMENTATION_PLAN.md` (Phase 4) for institutional-grade TS hyper-parameter search using:
  - `scripts/run_ts_model_search.py` (rolling-window CV over small SARIMAX/SAMOSSA grids, writing to `ts_model_candidates`), and
  - `etl/statistical_tests.py` (Diebold–Mariano-style tests and rank stability helpers) to ensure candidate selection is statistically robust, not just numerically stable.
