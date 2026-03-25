# Time-Series Feature And Audit Contract

This contract defines the minimum provenance and policy fields that every
production-facing time-series forecast artifact must carry.

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

## Required behaviors

- Ensemble index mismatch must be fail-soft at forecast time and fail-closed in audit/CI gates.
- Missing residual diagnostics for a model-backed default path must be treated as a governance failure after warmup.
- Backtests and baseline snapshots must persist provenance fields alongside metrics so comparisons are reproducible.
- Config changes that alter exogenous alignment, differencing, residual gates, GARCH fallback behavior, or MSSA-RL action/rank policy must update the relevant config files and preserve `config_hash` traceability.
