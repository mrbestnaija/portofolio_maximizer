# Agent C Experiment Brief: EXP-R5-001

Doc Type: experiment_brief
Authority: temporary research-tracking doc for Agent C; not a readiness source of truth
Owner: Agent C
Last Verified: 2026-03-08
Verification Commands:
- `python scripts/check_model_improvement.py --layer 1 --json`
- `python scripts/capital_readiness_check.py --json`
Artifacts:
- `Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-08.md`
- `logs/performance/residual_experiment_summary.json`
Supersedes: none
Expires When: superseded by a newer dated EXP-R5-001 experiment brief or merged into canonical experiment tracking docs

Status: NOT RUN

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
  - pending Agent A implementation and naming
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
- Current Layer 1 baseline result:
  - `status = FAIL`
  - `n_used_windows = 162`
  - `lift_ci_low = -0.1139`
  - `lift_ci_high = -0.0572`
  - `lift_win_fraction = 3.1%`

Verified Phase 1 residual experiment plumbing in workspace:
- `forecast()` now emits a `residual_experiment` key when forecasts are built
- when `residual_experiment_enabled=True` and no residual model is fitted yet:
  - `residual_status = "inactive"`
  - reason = `residual_model_not_fitted (Phase 2 pending)`
- first valid experiment run therefore still requires Phase 2

Interpretation:
- The current ensemble underperforms the best single model on existing audit evidence.
- `EXP-R5-001` exists to test a different research candidate around `mssa_rl`, not to reinterpret the current failing ensemble.

## Current Blockers

1. Residual model not wired.
- Phase 1 artifact plumbing is present, but the fitted residual model is still absent.
- Agent A Phase 2 work remains incomplete.

2. Residual metrics not exposed in audits.
- Agent B work incomplete.

3. No experiment-specific effective audit count exists yet.
- Current baseline has 162 Layer 1 windows, but `EXP-R5-001` has produced zero valid candidate windows so far.
- Current status remains `NOT RUN`, because `residual_status="inactive"` is not a fitted experiment run.

4. Real system blockers still remain active.
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
  - first valid run requires Phase 2 fitted-model wiring from Agent A
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
  - `logs/performance/residual_experiment_summary.json`
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
