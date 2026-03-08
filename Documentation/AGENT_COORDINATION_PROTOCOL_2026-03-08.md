# Multi-Agent Coordination Protocol (2026-03-08)

This document is the single source of truth for how Agents A, B, and C collaborate.
It supersedes any prior informal coordination notes.

## Integration Gate Rule (Non-Negotiable)

**No agent-produced code may enter master without explicit approval and integration
by the human + Claude Code pair.**

The sequence is:

```
Agent proposes change (code, config, doc)
        ↓
Agent commits to a personal branch or patch file
        ↓
Human reviews the diff via Claude Code
        ↓
Claude Code integrates what passes review; rejects or defers the rest
        ↓
Merged to master only after targeted tests pass
```

Agents do NOT self-merge. Agents do NOT commit to master directly.
If a commit appears on master from an agent without the review step, it will be reverted.

**Evidence from this session**: Two agent commits (Agent B `106b7aa` and Agent C `7416e17`)
were reverted in full because they arrived on master without the review gate.
Safe subsets were subsequently re-integrated one concern at a time.

---

## Task Delegation Principle

**Agent A compute is a constrained resource. It must be reserved for work with direct
impact on model quality, signal accuracy, and PnL. Lower-priority tasks that require
less precision in implementation must be routed to Agent B or Agent C.**

### Agent A Priority Ladder

Agent A works tasks strictly in this order. If a task does not appear at tier P0 or P1,
Agent A should flag it for delegation before starting.

| Tier | Category | Examples |
|---|---|---|
| **P0 — Reserve for A only** | Forecasting model correctness | GARCH/SAMOSSA convergence, ensemble weight logic, CI computation, Hurst/regime numerical correctness |
| **P0 — Reserve for A only** | Gate wiring integrity | Key name bugs in `capital_readiness_check.py`, wrong FAIL/WARN classification, CI span semantics |
| **P0 — Reserve for A only** | Signal accuracy | Anything that changes what signal fires, when, or with what confidence |
| **P1 — A preferred, may delegate** | Regression test fixes for P0 code | Tests covering model output correctness; A writes these but may delegate fixture/helper scaffolding to B |
| **P2 — Delegate to B** | Audit script compatibility | pandas/numpy API warnings, dtype coercions, FutureWarnings in `exit_quality_audit.py`, `ensemble_health_audit.py` |
| **P2 — Delegate to B** | Reporting/logging additions | New fields in dashboard payload, adding `exit_reason` to feeds, stale detection flags |
| **P2 — Delegate to B** | Pipeline orchestration changes | New steps in `run_quality_pipeline.py`, eligibility gate wiring, sidecar scripts |
| **P2 — Delegate to B** | Test fixture maintenance | Shared DB schemas in test helpers, import scaffolding for new scripts |
| **P3 — Delegate to C** | Documentation updates | Readiness blocker matrix, experiment briefs, acceptance test stubs (NOT-YET-RUN) |
| **P3 — Delegate to C** | Observation and reporting | Running gate commands and recording verified outputs, updating status tables |

### Delegation Rule

When Agent A receives a task request, it must classify it before acting:

1. If P0/P1 → proceed.
2. If P2 → propose the task to Agent B with a one-sentence spec; do not implement.
3. If P3 → propose the task to Agent C; do not implement.
4. If unclear → flag to Claude Code / human for routing decision.

Agent A must never use P0 compute to perform P2/P3 work even if it is faster for A to
do it. The constraint is not capability — it is allocation.

---

## Agent Roles and Domain Boundaries

### Agent A — Model, Signal, and PnL Integrity

**Domain** (P0 — Agent A exclusively):
- Forecasting layer accuracy: `forcester_ts/`, `models/time_series_signal_generator.py`
- Gate wiring correctness: `scripts/capital_readiness_check.py`, `scripts/check_model_improvement.py`, `scripts/run_all_gates.py`
- Numerical stability: `forcester_ts/regime_detector.py`, GARCH, SAMOSSA
- Ensemble selection logic, CI computation, signal routing correctness

**In scope**:
- Fix wrong key names in gate logic (e.g., `n_used` vs `n_used_windows`)
- Fix mislabeled CI intervals (`spans zero` vs `definitively negative`)
- Fix NaN/inf in regime detection
- Ensemble convergence, weight redistribution, DA-cap logic
- Lazy-import / import-path isolation **only** when it blocks a P0 gate fix

**Out of scope — delegate to B**:
- Dashboard payload shape (`scripts/dashboard_db_bridge.py`)
- pandas/numpy compatibility fixes in audit scripts
- Pipeline orchestration changes (`run_quality_pipeline.py`, eligibility sidecar)
- Reporting field additions (exit_reason, stale flags, unknown metrics display)

**Out of scope — delegate to C**:
- Documentation updates (readiness matrix, experiment briefs)
- Acceptance test stubs

**Must not touch without explicit sign-off**:
- Trade execution / PnL integrity (`execution/`, `integrity/`)
- Experiment configuration (`config/quant_success_config.yml`, routing config)
- Anything that would change what signals fire or what trades execute

**Current outstanding A-work (as of 2026-03-08)**: None.
- `hurst=0.5` constant-series semantic contract: DONE — `tests/forcester_ts/test_regime_detector_stability.py:82` asserts `_calculate_hurst_exponent([1.0]*5) == 0.5` (Phase 7.40, committed b925045).
- `etl/regime_detector.py` degenerate t-test finite-clamp: DONE — `confidence` and `transition_probability` no longer NaN on flat inputs; `tests/etl/test_regime_detector_stability.py` covers this path.
- `run_all_gates.py` pre_institutional artifact: DONE — gate status written before institutional P4 and fails closed on stale/missing prior-gate evidence.
- All Phase 7.40 fixes integrated and verified.

---

### Agent B — Infrastructure, Reporting, and Compatibility

**Domain** (receives P2 delegations from A):
- Dashboard bridge: `scripts/dashboard_db_bridge.py`
- Pipeline orchestration: `scripts/run_quality_pipeline.py`
- Eligibility gate sidecar: `scripts/apply_ticker_eligibility_gates.py`
- Exit quality audit: `scripts/exit_quality_audit.py`
- Audit script compatibility (pandas/numpy API, dtype warnings, FutureWarnings)
- Reporting field additions and logging improvements

**In scope**:
- Make dashboard metrics truthful (None vs 0.0 for unknown, stale detection)
- Wire eligibility gate output into pipeline steps
- Fix pandas/numpy compatibility issues in audit scripts
- Add reporting fields to trade event feeds (exit_reason, stale flags, etc.)
- Test fixture scaffolding for shared DB schemas used by multiple test files
- Pipeline step additions that do not change gate pass/fail logic

**Out of scope** (must not touch without explicit sign-off):
- Gate pass/fail semantics (`run_all_gates.py`, `capital_readiness_check.py`)
- Forecasting model code (`forcester_ts/`, `models/`)
- Experiment entry criteria (Agent C domain)
- Live HTML dashboard (`visualizations/live_dashboard.html`) — requires separate
  review since it ships to users; diff must be shown before integration

**Current outstanding B-work (as of 2026-03-08)**:
- `visualizations/live_dashboard.html` from reverted commit `106b7aa` — not yet reviewed.
  Dashboard code changes must be shown as a standalone diff before integration.
  The 3 new JS/HTML behaviors (stale flag, unknown metrics display, exit_reason column)
  need to match the backend payload fields now in place.

---

### Agent C — Experiment Planning and Measurement Reporting

**Domain** (receives P3 delegations from A; measurement-only):
- Experiment design documents: `Documentation/AGENT_C_EXPERIMENT_BRIEFS_*.md`
- Readiness tracking: `Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_*.md`
- Acceptance test templates for C1, C2, C4 (not implementation)
- Recording and cross-referencing verified command outputs against entry criteria

**In scope**:
- Report verified outputs from commands and artifacts
- Maintain experiment briefs with current global block status
- Prepare acceptance test structure (code stubs in `tests/`, clearly marked as NOT-YET-RUN)
- Update readiness blocker matrix when gate state changes
- Translate gate outputs into plain-language readiness summaries for human review

**Out of scope (hard prohibited)**:
- No strategy changes
- No experiment execution
- No readiness claims while global entry criteria are unmet
- No interpretation of watcher WAITING status as progress
- No code changes to production files (all code proposals route through B or A)

**Agent C is currently blocked**. See `AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-08.md`.

**Protocol note (2026-03-08)**: Agent C created `scripts/windows_persistence_manager.py` and
`scripts/run_persistence_manager.bat`. These are production scripts — per the delegation table,
infrastructure orchestration is a P2 task belonging to Agent B. The implementation was
accepted because it was safe (calls only existing entry points, no gate/signal code touched,
7 tests pass, fast lane clean). In future: Agent C must raise infrastructure scripts as a
one-sentence spec to Agent B and not implement them directly.

---

## Shared File Ownership Rules

| File | Owner | Others must |
|---|---|---|
| `scripts/capital_readiness_check.py` | A | Ask A before editing |
| `scripts/check_model_improvement.py` | A | Ask A before editing |
| `forcester_ts/regime_detector.py` | A | Ask A before editing |
| `scripts/dashboard_db_bridge.py` | B | Diff reviewed before merge |
| `scripts/run_quality_pipeline.py` | B | Diff reviewed before merge |
| `scripts/apply_ticker_eligibility_gates.py` | B | Diff reviewed before merge |
| `integrity/pnl_integrity_enforcer.py` | Claude Code / human | No agent self-edits |
| `scripts/run_all_gates.py` | Claude Code / human | No agent self-edits |
| `config/*.yml` | Claude Code / human | No agent self-edits |
| `AGENTS.md` | Claude Code / human | No agent self-edits |
| `Documentation/AGENT_C_*` | C (propose), Claude Code (approve) | |

---

## Integration Checklist (Required Before Any Merge)

1. [ ] Run `python scripts/run_all_gates.py --json` — record `overall_passed`
2. [ ] Run `python -m integrity.pnl_integrity_enforcer` — confirm `ALL PASSED`
3. [ ] Run targeted tests for every file touched — record pass count
4. [ ] Run `python -m pytest tests/ -q --tb=short -x` (or fast lane) — no new failures
5. [ ] Verify gate state did not regress vs previous run
6. [ ] Document what was integrated, what was deferred, and why

Partial integration (integrating the safe subset and deferring the risky subset) is
explicitly allowed and preferred over all-or-nothing decisions.

---

## What "Approved and Integrated" Means for Each Agent

An agent's work is approved when:
1. The diff has been read in full by Claude Code
2. Every non-trivial change has a targeted test
3. All touched tests pass
4. The blocker matrix and/or experiment briefs are updated to reflect the new state
5. The change is committed to master by Claude Code (not by the agent directly)

Work sitting in a branch or patch file is **not integrated** even if correct.

---

## Current Verified System State (2026-03-08)

All facts below were verified by running commands in this session.

### Gate Status

| Gate | Result | Fixable by code? |
|---|---|---|
| `ci_integrity_gate` | PASS | — |
| `check_quant_validation_health` | PASS | — |
| `production_audit_gate` | FAIL | No — data-driven |
| `institutional_unattended_gate` | FAIL | No — cascades from above |

`production_audit_gate` fails because:
- Total PnL is negative (`profitable=False`) — real trade outcomes, not wiring
- `matched=0/1` — insufficient fresh linked audit windows
- `EVIDENCE_HYGIENE_FAIL` — non-trade context contamination in audit population

**These require live trading cycles on Monday, 2026-03-10. No code can unblock them.**

### Capital Readiness

| Gate | Status | Root Cause |
|---|---|---|
| R1 adversarial | PASS | 0/21 confirmed adversarial findings |
| R2 gate artifact | FAIL | cascades from production_audit_gate |
| R3 trade quality | FAIL | WR=40%<45%, PF=0.80<1.30 — real metrics |
| R4 calibration | PASS | Brier below threshold |
| R5 lift CI | FAIL | CI=[-0.1139,-0.0572] definitively negative, 162 windows |
| R6 lifecycle | PASS | cleared |

R5 is now correctly a hard FAIL (Phase 7.40 fixed the `n_used` → `n_used_windows`
key read bug that was silently keeping the hard-fail branch as dead code).

### Resolved This Session

- `ci_integrity_gate` ORPHANED_POSITION: ids 249,250,251,253 whitelisted
- Dashboard truth: stale positions → fallback, `None` vs 0.0 unknown metrics, `exit_reason` in trade feed
- Layer 1 CI: definitively-negative CI (`ci_high<0`, n>=20) now hard FAIL
- `apply_ticker_eligibility_gates.py` + pipeline step 1b: integrated
- `exit_quality_audit.py` np.where fix: integrated (no more pandas 2.x ChainedAssignmentError)
- Agent C docs (blocker matrix, experiment briefs): restored and updated

### Deferred (Needs Separate Review)

- `visualizations/live_dashboard.html` from Agent B — HTML/JS diff not yet reviewed
