# Agent B Handoff - EXP-R5-001 Truth/Contradiction Guard

Date: 2026-03-08

## Scope delivered

1. Added single-source status script:
   - `scripts/residual_experiment_truth.py`
2. Added CI anti-contradiction tests:
   - `tests/scripts/test_residual_experiment_truth.py`
3. Updated status script to surface canonical summary sidecar:
   - `scripts/verify_residual_experiment.py`
   - `tests/scripts/test_verify_residual_experiment.py`
4. Added canonical-path status reference doc:
   - `Documentation/EXP_R5_001_STATUS_SOURCE_OF_TRUTH_2026-03-08.md`
5. Added residual measurement-maturity fields (additive, backward-compatible):
   - `n_windows_with_realized_residual_metrics`
   - `n_windows_structural_only_metrics`
   - `n_active_windows_missing_realized_metrics`
   - `m2_review_ready`
   - warning: `residual_experiment_realized_metrics_unavailable`
   - warning: `residual_experiment_missing_realized_metrics_windows:<N>`

## Canonical summary path (enforced in status tooling)

`visualizations/performance/residual_experiment_summary.json`

## Contradiction codes

`scripts/residual_experiment_truth.py` fails (`exit 1`) on:

1. `ACTIVE_AUDITS_BUT_SUMMARY_SKIP`
2. `ACTIVE_AUDITS_BUT_ZERO_MEASURED_WINDOWS`

## Validation evidence

### Targeted tests

Command:

`python -m pytest tests/scripts/test_run_quality_pipeline.py tests/scripts/test_residual_experiment_truth.py tests/scripts/test_verify_residual_experiment.py -q`

Result:

- `23 passed`

### Fast lane

Command:

`python -m pytest -m "not gpu and not slow" --tb=short -q`

Result:

- `1788 passed, 3 skipped, 28 deselected, 7 xfailed`

### Runtime snapshots

Truth snapshot command:

`python scripts/residual_experiment_truth.py --audit-dir logs/forecast_audits --json`

Observed key fields:

- `ok = true`
- `summary_status = PASS`
- `n_windows_with_residual_metrics = 9`
- `n_windows_with_realized_residual_metrics = 0`
- `n_windows_structural_only_metrics = 9`
- `m2_review_ready = false`
- `audits.n_active = 23`
- `audits.n_inactive = 35`
- `contradictions = []`

Verifier command:

`python scripts/verify_residual_experiment.py --audit-dir logs/forecast_audits --json`

Observed nuance:

- Default behavior scans latest audit only; latest was inactive (`n_active=0`, `n_inactive=1`) while summary was PASS.
- For full audit scan in this script, use `--all`.

## Agent A integration notes

1. Use `residual_experiment_truth.py` as final status decision source for EXP-R5-001 activation consistency.
2. Keep `verify_residual_experiment.py` for latest-window quick checks; use `--all` for full scan.
3. Do not use non-canonical summary paths in status decisions.
4. M2 review floor should use `n_windows_with_realized_residual_metrics` (not structural-only count).

## Agent C governance notes

1. Promotion out of `NOT RUN` should require truth snapshot `ok=true` and no contradiction codes.
2. When reporting measured progress, reference:
   - `summary_status`
   - `n_windows_with_residual_metrics`
   - `n_windows_with_realized_residual_metrics`
   - `audits.n_active`
3. Treat contradiction code presence as blocker regardless of summary status string.
