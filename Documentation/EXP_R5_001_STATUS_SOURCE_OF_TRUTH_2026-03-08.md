# EXP-R5-001 Status Source of Truth

## Current experiment status (2026-03-09)

**REDESIGN COMPLETE** — RC1–RC4 applied (commit d7ebacd).  Experiment is in re-accumulation.
- Audits before 2026-03-09 reflect the OLD AR(1) (no demeaning, no phi gate) — treat as historical.
- Audits from 2026-03-09 onwards reflect the corrected AR(1).
- Do NOT use pre-redesign audits to evaluate M2/M3 thresholds for the new cycle.
- `phi_hat` and `skip_reason` fields are now present in all new audit artifacts.
  - `residual_status="active"` AND `skip_reason=null` AND `phi_hat >= 0.15` → correction applied.
  - `residual_status="inactive"` with `skip_reason` set → gate fired (expected for weak autocorrelation).

## Canonical summary sidecar path

Use only:

`visualizations/performance/residual_experiment_summary.json`

Do not use alternate locations (for example `logs/performance/...`) for EXP-R5-001 status decisions.

## Canonical status commands

1. Audit-level activation check:

`python scripts/verify_residual_experiment.py --audit-dir logs/forecast_audits --json`

2. Cross-source contradiction check (single source for Agent A/C handoff):

`python scripts/residual_experiment_truth.py --audit-dir logs/forecast_audits --json`

## Measurement maturity fields (from summary sidecar)

Use these for M2 readiness instead of structural-only counts:

- `n_windows_with_realized_residual_metrics`
- `n_windows_structural_only_metrics`
- `n_active_windows_missing_realized_metrics`
- `m2_review_ready`

## Contradiction rules (must fail CI/ops checks)

1. `ACTIVE_AUDITS_BUT_SUMMARY_SKIP`
2. `ACTIVE_AUDITS_BUT_ZERO_MEASURED_WINDOWS`

If either appears, treat EXP-R5-001 status as inconsistent and block status promotion until fixed.
