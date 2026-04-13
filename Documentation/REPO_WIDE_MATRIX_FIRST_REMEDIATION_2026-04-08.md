# Repo-Wide Matrix-First Remediation Policy - 2026-04-08

**Status**: Implemented documentation policy; code/config rollout in progress
**Scope**: Forecasting stack, signal scoring, audit contracts, gate posture, monitoring language
**Canonical role**: Repo-wide source of truth for objective semantics and matrix-first remediation wording

## Canonical Economic Objective

**Barbell asymmetry is the primary economic objective. The system optimizes for asymmetric upside with bounded downside, not for symmetric textbook efficiency metrics.**

This policy is binding across canonical docs, configs, audit contracts, and monitoring language.
If any older document implies that Sharpe ratio, win rate, balanced accuracy, or generic forecast
neatness is the primary objective, that statement is superseded by this document.

## Current Status Labels

- `Implemented`: already present in code/config/docs and should be treated as current policy.
- `Accepted Plan`: approved target state that canonical docs may reference without claiming code completion.
- `Historical Context`: evidence or prior reasoning retained for traceability, not for current defaults.

## Implemented Policy Defaults

- `objective_mode: domain_utility` is the repo-wide default name for the primary economic objective.
- Horizon semantics are bars, not calendar days.
  Required terms: `forecast_horizon_bars`, `forecast_horizon_units="bars"`, `expected_close_source`.
- Readiness posture must distinguish:
  - `GENUINE_PASS`
  - `WARMUP_COVERED_PASS`
  - `FAIL`
- Warmup-covered states are compatibility evidence only; they do not count as unattended-run readiness.

## Documentation Hygiene Posture

- `Documentation/DOCUMENTATION_INDEX.md` is the canonical navigator for repository documentation.
- `Documentation/history/` is the canonical home for archived status snapshots, session notes, and run logs.
- Repo-wide docs should link directly to archived `history/` files instead of keeping duplicate mirrors or long-lived relocation stubs.
- Canonical top-level entry points are:
  - `README.md` for repo overview
  - `CLAUDE.md` for agent guidance
  - `SECURITY.md` for responsible disclosure
  - `Documentation/HEARTBEAT.md` for the live status snapshot
- Duplicate mirrors of canonical docs should be removed once inbound references are updated in the same hygiene pass.

## Primary vs Diagnostic Fields

### Primary objective fields

- `expected_profit`
- `omega_ratio`
- `payoff_asymmetry`
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
- one-step directional metrics unless a local contract explicitly promotes them

Diagnostic metrics remain important, but they protect or explain the primary objective; they do not
replace it.

## Matrix-First Contract Language

- Matrix diagnostics must be explicit and audit-visible through `matrix_health`.
- `matrix_health` should carry shape, rank, singular-value summaries, condition number, NaN/Inf counts,
  EVR where applicable, and model-specific stability indicators.
- `utility_breakdown` should explain how the emitted signal or audit artifact aligns with the primary
  asymmetric objective.
- `expected_close_source` must state how the close timestamp was resolved so causality is inspectable.

## Conflict Resolution

Use this order when documents disagree about objective semantics:

1. Code and tests
2. This document
3. `Documentation/CORE_PROJECT_DOCUMENTATION.md`
4. `Documentation/PROJECT_STATUS.md`
5. Historical or supporting notes

## Historical Evidence Links

These remain valuable evidence records, but their objective framing is subordinate to this policy:

- `Documentation/DOMAIN_CALIBRATION_REMEDIATION_2026-04-05.md`
- `Documentation/REPO_WIDE_GATE_LIFT_REMEDIATION_2026-03-29.md`
- `Documentation/MSSA_RL_OFFLINE_REMEDIATION_20260404.md`
