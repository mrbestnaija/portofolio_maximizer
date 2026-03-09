# Agent C Experiment Brief: EXP-R5-001

Doc Type: experiment_brief
Authority: temporary research-tracking doc for Agent C; not a readiness source of truth
Owner: Agent C
Last Verified: 2026-03-08
Verification Commands:
- `python scripts/check_model_improvement.py --layer 1 --json`
- `python scripts/capital_readiness_check.py --json`
- `python -m pytest tests/forcester_ts/test_residual_ensemble.py tests/scripts/test_run_quality_pipeline.py -q`
- `python -m pytest -m "not gpu and not slow" --tb=short -q`
- `python scripts/run_quality_pipeline.py --json --enable-residual-experiment --residual-experiment-out visualizations/performance/residual_experiment_summary.json`
- `python scripts/verify_residual_experiment.py --audit-dir logs/forecast_audits --json`
- `python scripts/residual_experiment_truth.py --audit-dir logs/forecast_audits --json`
Artifacts:
- `Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-08.md`
- `visualizations/performance/residual_experiment_summary.json`
- `Documentation/EXP_R5_001_STATUS_SOURCE_OF_TRUTH_2026-03-08.md`
Supersedes: none
Expires When: superseded by a newer dated EXP-R5-001 experiment brief or merged into canonical experiment tracking docs

Status: FAILED

Experiment:
- `EXP-R5-001 - Residual Ensemble Around mssa_rl`

Purpose:
- Define the research-only measurement contract for a residual ensemble anchored on `mssa_rl`.
- Keep the experiment bounded to paper/audit evidence.
- Prevent any implicit readiness or live-strategy claims before Agent A and Agent B complete their lanes.

## Objective

- Evaluate whether residual correction around `mssa_rl` can improve R5 lift on paper/audit-only runs.
- Compare only:
  - baseline: `mssa_rl`
  - candidate: `mssa_rl` plus residual correction

## Scope Guardrails

- Research only.
- No changes to:
  - live strategy behavior
  - readiness thresholds
  - gate semantics
  - `config/*.yml`
- No experiment execution from the Agent C lane.
- No interpretation of system-gate output as automatic promotion evidence for `EXP-R5-001`.

## Inputs

- Anchor model ID:
  - `mssa_rl`
- Residual model ID:
  - `resid_mssa_rl_v1`
  - implemented in code and now active in at least one real experiment artifact
- Dataset and forecast horizon:
  - must align with existing R5 audit windows used by `scripts/check_model_improvement.py --layer 1`
- Audit source:
  - post-implementation audit outputs exposed by Agent B

## Metrics To Collect

- RMSE:
  - `RMSE(mssa_rl)`
  - `RMSE(residual_ensemble)`
- Directional Accuracy:
  - `DA(mssa_rl)`
  - `DA(residual_ensemble)`
- Relative metrics:
  - RMSE ratio = `RMSE(residual_ensemble) / RMSE(mssa_rl)`
  - DA delta = `DA(residual_ensemble) - DA(mssa_rl)`
- Diversity metrics:
  - `corr_anchor_residual`
- R5 contract inputs:
  - lift estimate
  - `n_used_windows`
  - CI bounds
  - win fraction

## Current Baseline Context

Verified on 2026-03-08:

- `python scripts/check_model_improvement.py --layer 1 --json`
- `git diff -- forcester_ts/forecaster.py`
- `Get-Content Documentation/EXP_R5_001_RESIDUAL_ENSEMBLE_DESIGN_2026-03-08.md`
- `python -m pytest tests/forcester_ts/test_residual_ensemble.py tests/scripts/test_run_quality_pipeline.py -q`
- `python -m pytest -m "not gpu and not slow" --tb=short -q`
- `python scripts/run_quality_pipeline.py --json --enable-residual-experiment --residual-experiment-out visualizations/performance/residual_experiment_summary.json`
- `Get-Content visualizations/performance/residual_experiment_summary.json`
- Current Layer 1 baseline result:
  - `status = FAIL`
  - `n_used_windows = 162`
  - `lift_ci_low = -0.1139`
  - `lift_ci_high = -0.0572`
  - `lift_win_fraction = 3.1%`

Verified Phase 1 residual experiment plumbing in workspace:
- `forecast()` now emits a `residual_experiment` key when forecasts are built
- current config still keeps:
  - `config/forecasting_config.yml -> residual_experiment.enabled = true`
- canonical truth snapshot now shows:
  - `ok = true`
  - `residual_experiment_enabled = true`
  - `summary_status = PASS`
  - `summary_reason_code = RESIDUAL_EXPERIMENT_AVAILABLE`
  - `n_windows_with_residual_metrics = 9`
  - `n_windows_with_realized_residual_metrics = 9`
  - `n_windows_structural_only_metrics = 0`
  - `m2_review_ready = true`
  - `audits.n_active = 23`
  - `contradictions = []`
- cross-source verifier currently reports:
  - `active = true`
  - `n_active = 23`
  - `n_inactive = 45`
- current residual experiment summary artifact shows:
  - `status = PASS`
  - `reason_code = RESIDUAL_EXPERIMENT_AVAILABLE`
  - `n_not_fitted_windows = 1`
  - `n_windows_with_residual_metrics = 9`
  - `n_windows_with_realized_residual_metrics = 9`
  - `n_windows_structural_only_metrics = 0`
  - `m2_review_ready = true`
  - `rmse_anchor_mean = 6.409537`
  - `rmse_residual_ensemble_mean = 9.055116`
  - `rmse_ratio_mean = 1.714861`
  - `da_anchor_mean = 0.544444`
  - `da_residual_ensemble_mean = 0.559259`
  - `corr_anchor_residual_mean = -0.144662`
  - `early_abort_signal = false`
  - `early_abort_consecutive_rmse_above_threshold = 2`

Verified Phase 2 code is now present in workspace:
- `ResidualModel` is implemented in `forcester_ts/residual_ensemble.py`
- `TimeSeriesForecaster.fit()` already calls `_fit_residual_model(price_series)` when `residual_experiment_enabled=True` and the `mssa_rl` anchor fit succeeds
- targeted residual, truth, and quality-pipeline suites pass
- repo-wide fast lane is currently clean in this workspace:
  - `1790 passed, 3 skipped, 28 deselected, 7 xfailed`
- Agent C now treats `EXP-R5-001` as `FAILED` at M3 because the formal decision is `REDESIGN_REQUIRED` after 11 realized windows

Interpretation:
- The current ensemble underperforms the best single model on existing audit evidence.
- `EXP-R5-001` exists to test a different research candidate around `mssa_rl`, not to reinterpret the current failing ensemble.

## Current Blockers

1. M1 activation is complete.
- A real experiment artifact now shows `residual_status="active"` with non-`None` `y_hat_anchor` and `y_hat_residual_ensemble`.

2. M2 realized measurement maturity is complete.
- Current summary artifact shows:
  - `n_windows_with_residual_metrics = 9`
  - `n_windows_with_realized_residual_metrics = 9`
  - `n_windows_structural_only_metrics = 0`
  - `m2_review_ready = true`
- These windows satisfy the realized-measurement floor and unlock the first M2 governance verdict.

3. M3 evaluation is complete and failed.
- Current summary artifact now shows `n_windows_with_realized_residual_metrics = 11`.
- The formal decision threshold is met and the candidate failed Rule 2.

4. Realized error metrics show a structural anti-signal, not mere noise.
- Summary status is `PASS`, but:
  - `rmse_anchor_mean = 6.548374`
  - `rmse_residual_ensemble_mean = 9.211566`
  - `rmse_ratio_mean = 1.689169`
  - `da_anchor_mean = 0.536364`
  - `da_residual_ensemble_mean = 0.548485`
  - `corr_anchor_residual_mean = -0.257054`
  - `early_abort_signal = false`
- Verified failure pattern:
  - `7/11` windows have `rmse_ratio > 1.0`
  - `8/11` windows have negative realized `corr(ε, ε_hat)`
  - short folds (`len <= 210`) are the worst cohort:
    - mean `rmse_ratio = 2.139716`
    - `6/7` windows worse than anchor
  - larger datasets (`len > 210`) are materially better:
    - mean `rmse_ratio = 0.900711`
- Agent C treats this as `FAILED / REDESIGN_REQUIRED`, not as a candidate for continued accumulation.

5. Activation truth is now cross-source verified.
- `config/forecasting_config.yml` currently sets `residual_experiment.enabled = true`.
- `python scripts/residual_experiment_truth.py --audit-dir logs/forecast_audits --json` currently reports:
  - `ok = true`
  - `summary_status = PASS`
  - `n_windows_with_residual_metrics = 11`
  - `n_windows_with_realized_residual_metrics = 11`
  - `n_windows_structural_only_metrics = 0`
  - `m2_review_ready = true`
  - `audits.n_active = 23`
  - `contradictions = []`
- `python scripts/verify_residual_experiment.py --audit-dir logs/forecast_audits --all --json` currently reports:
  - `active = true`
  - `n_active = 23`
  - `n_inactive = 37`
- This clears the activation blocker and establishes the canonical truth source.

6. CI import/wiring blocker is cleared and the repo-wide fast lane is green again.
- Targeted local suites pass:
  - `tests/forcester_ts/test_residual_ensemble.py`
  - `tests/scripts/test_run_quality_pipeline.py`
  - `tests/scripts/test_residual_experiment_truth.py`
  - `tests/scripts/test_verify_residual_experiment.py`
- The earlier residual/quality-pipeline import blocker is no longer an open blocker.
- Current repo-wide fast lane is green:
  - `1790 passed, 3 skipped, 28 deselected, 7 xfailed`

7. Redesign actions are now the only meaningful next step for this experiment.
- Required before any rerun as `EXP-R5-002` or equivalent:
  - RC1 first: demean OOS residuals before AR(1) fit so the model learns autocorrelation, not DC bias
  - RC4 second: add model-local skip gates for weak/no-signal fits
  - RC3 third: replace the fixed OOS slice with a proportional history rule only after RC1 is in place
  - RC2 last: redesign residual generation so training and deployment target the same anchor regime
  - persist `phi_hat`, `intercept_hat`, `n_train_residuals`, `oos_n_used`, `correction_skipped`, and `skip_reason` in the artifact so future postmortems do not depend on logs

Redesign ordering note:
- Agent A's current sequencing is `RC1 -> RC4 -> RC3 -> RC2`.
- This ordering is deliberate.
- Raising the OOS window first would give AR(1) more of the same bias-dominated residuals to fit and can make the wrong constant correction more stable, not less.
- RC2 is the deepest issue, but it is only partially fixable inside `forecaster.py` and `residual_ensemble.py`; Agent C therefore treats it as a redesign dependency, not as a same-pass patch expectation.

Observed mechanism behind the redesign order:
- the current correction path behaves like a large constant offset in the worst windows, not a mean-zero residual correction
- this is consistent with the temporary-anchor vs full-anchor mismatch and with fitting on a fixed short residual slice
- the first redesign pass therefore needs to remove the DC component before expanding the training window

Redesign implementation note:
- Agent A has now landed the first redesign code pass in local commits:
  - `d7ebacd` for RC1-RC4 implementation
  - `27cc45d` for design/status documentation updates
- Verified code-level changes now present in workspace:
  - OOS residuals are demeaned before AR(1) fit
  - weak-signal fits expose `skip_reason`
  - OOS history is now proportional to dataset length
  - artifact observability fields are present in code:
    - `phi_hat`
    - `intercept_hat`
    - `n_train_residuals`
    - `oos_n_used`
    - `skip_reason`
- Important governance constraint:
  - the current truth snapshot and residual summary still describe the failed pre-redesign run
  - Agent C will not reopen this candidate based on code landing alone
  - the next valid status change requires a fresh post-redesign rerun and a new truth snapshot

8. Real system blockers still remain active.
- PnL integrity / trade-quality gate still fails on production evidence.
- Linkage evidence remains thin.
- Audit hygiene remains a gating concern for readiness.

## Measurement Runbook

This is a plan only. Agent C does not execute it.

Expected future sequence after Agent A and Agent B integrate their branches:

1. Run the experiment pipeline with the residual candidate enabled.
- Placeholder entrypoint:
  - `python scripts/run_quality_pipeline.py --enable-residual-experiment ...`
- Note:
  - Phase 1 can emit the `residual_experiment` artifact only
  - first valid run now requires the existing fitted-model path to actually produce `residual_status="active"` in a real audit artifact
  - the actual run flag and final pipeline entrypoint must not be invented by Agent C.

2. Record system state separately.
- `python scripts/run_all_gates.py --json`
- Gate output is recorded for environmental context only.
- It is not promotion evidence for `EXP-R5-001`.

3. Collect experiment measurements from the resulting audits.
- `RMSE(residual_ensemble)` vs `RMSE(mssa_rl)`
- RMSE ratio
- DA comparison
- `corr_anchor_residual`
- updated R5 statistics derived from those audit windows

4. Log results in two places.
- this experiment brief
- the Agent C readiness blocker matrix

## Result Log Template

Use this section format once a human or Agent A/B-owned branch produces the first real run artifact.

```markdown
### Result Log - YYYY-MM-DD

- Status:
  - NOT RUN / IN PROGRESS / EVALUATED
- Branch:
  - `<branch-name>`
- Residual model ID:
  - `<agent-a-defined-model-id>`
- Audit artifact:
  - `visualizations/performance/residual_experiment_summary.json`
- Effective audit windows:
  - `N = ...`
- Metrics:
  - `rmse_anchor = ...`
  - `rmse_residual_ensemble = ...`
  - `rmse_ratio = ...`
  - `da_anchor = ...`
  - `da_residual_ensemble = ...`
  - `corr_anchor_residual = ...`
  - `lift_ci = [low, high]`
  - `lift_win_fraction = ...`
- Gate snapshot for context only:
  - `python scripts/run_all_gates.py --json`
  - `overall_passed = ...`
  - `production_gate = ...`
- Interpretation:
  - `<research-only conclusion>`
- Decision:
  - keep evaluating / revise design / reject candidate
```

Rules:
- Do not use the result log to claim readiness.
- Do not reinterpret gate outputs as promotion evidence for `EXP-R5-001`.
- Record blockers explicitly if any required experiment metrics are still absent.

## Success Criteria

- Candidate produces a measurable RMSE improvement versus `mssa_rl`.
- Candidate improves or at least does not degrade Directional Accuracy.
- Candidate shows useful diversity relative to the anchor, not just redundant correlation.
- Candidate lift statistics are computed from experiment-specific audit windows, not borrowed from the current ensemble baseline.

## Non-Success Conditions

- No residual model wiring exists.
- No residual metrics are emitted in audits.
- Candidate windows are too sparse to estimate CI meaningfully.
- Results are mixed into readiness claims or live-strategy decisions.

## Explicit Note

`EXP-R5-001` does not change:
- readiness claims
- gate statuses
- live trading configuration
- promotion policy

Agent C role is limited to measurement planning, blocker tracking, and post-run result logging.
