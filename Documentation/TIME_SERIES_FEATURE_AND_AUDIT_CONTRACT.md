# Time-Series Feature And Audit Contract

This contract defines the minimum provenance and policy fields that every
production-facing time-series forecast artifact must carry.

> Canonical objective policy: `Documentation/REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md`
> **Barbell asymmetry is the primary economic objective. The system optimizes for asymmetric upside with bounded downside, not for symmetric textbook efficiency metrics.**

## Required repo-wide policy fields

- `objective_mode`: Must default to `domain_utility`.
- `forecast_horizon_bars`: Canonical horizon count in bars.
- `forecast_horizon_units`: Must be `"bars"` when the horizon is emitted explicitly.
- `expected_close_source`: How the expected close timestamp was resolved.
- `posture`: `GENUINE_PASS`, `WARMUP_COVERED_PASS`, or `FAIL`.
- `matrix_health`: Structural diagnostics for the matrix/design path.
- `utility_breakdown`: Primary-objective breakdown for the emitted signal or audit record.

## Required audit fields

- `dataset_hash`: Stable hash for the dataset snapshot used by the report or run.
- `db_max_ohlcv_date`: Maximum OHLCV date visible to the run at execution time.
- `config_hash`: Stable hash over the config files that shaped the run.
- `git_commit`: Source commit for the code that produced the artifact.
- `config_paths`: Absolute config paths included in `config_hash`.

## Required forecast policy fields

- `effective_default_model`: The model actually allowed to drive the live default path.
- `ensemble_index_mismatch`: `true` when the ensemble recovered via index intersection.
- `exog_policy`: SARIMAX fit-time exogenous feature alignment policy.
- `forecast_exog_policy`: SARIMAX forecast-time exogenous policy.
- `residual_diagnostics`: Per-model normalized residual diagnostics.
- `active_rank`: MSSA-RL rank selected by the bounded policy for the emitted forecast.
- `q_state`: MSSA-RL policy state used for the emitted forecast.

## Objective field vocabulary

### Primary objective fields

- `expected_profit`
- `omega_ratio`
- `profit_factor`
- `terminal_directional_accuracy`
- `max_drawdown`
- `expected_shortfall`
- `utility_breakdown`

### Diagnostic-only fields by default

- `win_rate`
- `sharpe_ratio`
- `sortino_ratio`
- `brier_score`
- one-step directional metrics unless a local contract promotes them explicitly

## Required behaviors

- Ensemble index mismatch must be fail-soft at forecast time and fail-closed in audit/CI gates.
- Missing residual diagnostics for a model-backed default path must be treated as a governance failure after warmup.
- Backtests and baseline snapshots must persist provenance fields alongside metrics so comparisons are reproducible.
- Config changes that alter exogenous alignment, differencing, residual gates, GARCH fallback behavior, or MSSA-RL action/rank policy must update the relevant config files and preserve `config_hash` traceability.
