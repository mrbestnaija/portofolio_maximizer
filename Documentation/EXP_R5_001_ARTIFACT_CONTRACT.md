# EXP-R5-001 Artifact Contract (Frozen)

Contract version: `exp-r5-001.v1`  
Owner: Agent A (producer), Agent B (consumer/reporting)

## Purpose

Freeze the residual experiment payload contract before parser wiring to prevent adapter drift.

## Canonical producer path

`artifacts.residual_experiment`

## Required keys (nullable allowed)

- `y_hat_anchor` (array[float] or null)
- `y_hat_residual_ensemble` (array[float] or null)
- `rmse_anchor` (float or null)
- `rmse_residual_ensemble` (float or null)
- `rmse_ratio` (float or null)
- `da_anchor` (float or null)
- `da_residual_ensemble` (float or null)
- `corr_anchor_residual` (float or null)

## Compatibility fallbacks

- `artifacts.evaluation_metrics.mssa_rl.{rmse,directional_accuracy|da}`
- `artifacts.evaluation_metrics.residual_ensemble.{rmse,directional_accuracy|da}`

Fallbacks are used only when the experiment signal is present.

## Status mapping

- `PASS`: residual experiment metrics available.
- `SKIP` + `RESIDUAL_EXPERIMENT_NOT_FITTED`: artifact emitted but `residual_status="inactive"` and `residual_active=false` (Phase 1 / model not fitted yet).
- `SKIP` + `RESIDUAL_EXPERIMENT_NOT_AVAILABLE`: experiment not emitted yet.
- `ERROR` + `RESIDUAL_EXPERIMENT_PAYLOAD_MALFORMED`: payload shape/type invalid.

## Compact parse diagnostics

Audit parse errors are reported as aggregate counters and filename samples, not one log line per file.

## Example payload (captured from Agent A artifact shape)

See fixture:

- [forecast_audit_agent_a_residual_real.json](c:/Users/Bestman/personal_projects/portfolio_maximizer_v45/portfolio_maximizer_v45/tests/scripts/fixtures/forecast_audit_agent_a_residual_real.json)
