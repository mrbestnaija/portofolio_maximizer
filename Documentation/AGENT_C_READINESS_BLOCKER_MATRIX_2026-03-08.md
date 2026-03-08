# Agent C Readiness Blocker Matrix (2026-03-08)

Doc Type: blocker_matrix
Authority: temporary Agent C blocker tracker; not a gate-semantics source of truth
Owner: Agent C
Last Verified: 2026-03-08
Verification Commands:
- `python scripts/project_runtime_status.py --pretty`
- `python scripts/capital_readiness_check.py --json`
- `python scripts/check_model_improvement.py --layer 1 --json`
- `python -m scripts.dashboard_db_bridge --once --db-path data\\portfolio_maximizer.db --output logs\\dashboard_data_review_tmp.json`
Artifacts:
- `visualizations/dashboard_data.json`
- `logs/overnight_denominator/live_denominator_latest.json`
- `logs/audit_gate/production_gate_latest.json`
Supersedes: `Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-07.md`
Expires When: superseded by a newer dated blocker matrix or merged into canonical runtime status docs

Purpose: keep Agent C on measurement, sequencing, and evidence only.

This document does not authorize strategy changes, experiment execution, or
readiness claims. It supersedes both older Agent C blocker matrices.

## Verified Inputs

Commands run on 2026-03-08 in the current session:

- `python scripts/project_runtime_status.py --pretty`
- `python scripts/capital_readiness_check.py --json`
- `python scripts/check_model_improvement.py --layer 1 --json`
- `python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json`
- `python -m pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_live_dashboard_wiring.py tests/scripts/test_check_model_improvement.py -q`

## Current State

### 1. Runtime Health

Source: `python scripts/project_runtime_status.py --pretty`

- overall status: `degraded`
- failed check: `production_gate`
- production gate still reports:
  - `GATES_FAIL`
  - `THIN_LINKAGE`
  - `EVIDENCE_HYGIENE_FAIL`
  - `matched=0/1`

Interpretation:
- Phase 3 remains blocked.
- This is still evidence-bound, not a wiring-only gate artifact issue.

### 2. Capital Readiness

Source: `python scripts/capital_readiness_check.py --json`

```json
{
  "ready": false,
  "verdict": "FAIL",
  "reasons": [
    "R3: win_rate=40.0% < 45%, profit_factor=0.80 < 1.30",
    "R5: ensemble lift is definitively negative CI=[-0.1139, -0.0572] across 162 windows (win_fraction=3.1%)"
  ]
}
```

| Gate | Status | Root Cause |
|---|---|---|
| R1 adversarial | PASS | 0 confirmed CRITICAL/HIGH findings |
| R2 gate artifact | PASS | current gate artifact is fresh and readable |
| R3 trade quality | FAIL | WR=40% < 45%, PF=0.80 < 1.30 |
| R4 calibration | PASS | Brier below threshold |
| R5 lift CI | FAIL | CI definitively negative across 162 windows |
| R6 lifecycle | PASS | no high lifecycle violations |

Interpretation:
- The old R5 threshold-dodge is closed.
- Current capital-readiness failure is real, not a wiring placeholder.

### 3. Layer 1 Lift Semantics

Source: `python scripts/check_model_improvement.py --layer 1 --json`

- `status = FAIL`
- `n_used_windows = 162`
- `lift_ci_low = -0.1139`
- `lift_ci_high = -0.0572`
- `lift_win_fraction = 3.1%`
- summary correctly says:
  - `lift CI [-0.1139, -0.0572] definitively negative`

Open issue:
- Layer 1 still emits deprecated naive UTC timestamps via `datetime.utcnow()`.
- This is not currently causing a gate dodge, but it is time-semantics debt.

### 4. Dashboard Truth

Sources:
- `python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json`
- live `visualizations/dashboard_data.json`

Live served payload now includes:
- `payload_schema_version = 2`
- `payload_digest`
- `performance_unknown = false`
- `positions_stale = true`
- `positions_source = trade_executions_fallback_stale`
- `data_origin = mixed`

Interpretation:
- the running bridge now matches the current schema and provenance semantics
- remaining dashboard concern is not runtime drift; it is the meaning of stale fallback-derived position/performance fields

### 5. Eligibility Gate Semantics

Source:
- `python scripts/apply_ticker_eligibility_gates.py --eligibility logs\definitely_missing_eligibility.json --output logs\ticker_eligibility_gates_tmp.json --json`

Current behavior on missing input:
- `status = WARN`
- `gate_written = true`
- all ticker lists empty
- process exits success

Interpretation:
- this is still fail-open on missing evidence
- ownership/policy decision remains open

### 6. Watcher Lane

Source: `logs/overnight_denominator/live_denominator_latest.json`

- `status = WAITING`
- `fresh_trade_rows = 1`
- `fresh_linkage_included = 1`
- `fresh_production_valid_matched = 0`

Interpretation:
- denominator recovery exists but is still thin
- no readiness claim is justified
- next live evidence window is the next trading day, Monday, **2026-03-09**

## Blocker Matrix

| Surface | Current result | Blocking owner | Unblock condition |
|---|---|---|---|
| Runtime health | `degraded` | Trading cycles + gate owners | `production_gate` passes |
| Production audit gate | FAIL | Trading cycles | clear `GATES_FAIL`, `THIN_LINKAGE`, `EVIDENCE_HYGIENE_FAIL` |
| Capital readiness | FAIL | Trading cycles + ensemble architecture | R3 clears and R5 no longer definitively negative |
| Dashboard truth (live runtime) | PASS | Agent B | keep served payload aligned with bridge schema and truth fields |
| Provenance classification | PASS | Agent B | mixed trade/live sources now produce `data_origin = mixed` |
| Eligibility gate semantics | OPEN | Agent A | explicit policy: WARN-only or fail-closed |
| Fresh TRADE denominator | `linkage_included=1`, `matched=0` | Trading cycles | `fresh_linkage_included > 1` across multiple cycles and `fresh_production_valid_matched >= 1` |
| Experiment: EXP-R5-001 | NOT RUN | Agent A + Agent B + Agent C | Phase 1 artifact is emitted but `residual_status=inactive`; Phase 2 fitted residual model, residual metrics in audits, and experiment-specific effective audits are still required |
| Experiment execution | blocked | Agent C protocol | all preconditions above satisfied |

## Agent C Operating Rules

- Do not start experiments.
- Do not recommend strategy changes.
- Do not interpret `WAITING` as progress.
- Do not interpret corrected reporting semantics as readiness progress.
- Report only verified outputs from commands and artifacts.

## Promotion Rule For Agent C

Agent C may move from blocker tracking to experiment-ready planning only when all
of the following are true:

1. fresh `TRADE` exclusions stay near zero across multiple cycles
2. `fresh_linkage_included > 1`
3. at least one fresh production-valid matched row appears
4. `production_audit_gate` passes
5. `capital_readiness_check.py` clears R3
6. live dashboard/runtime truth blockers are explicitly resolved

None of items 1-6 are currently satisfied end-to-end.
