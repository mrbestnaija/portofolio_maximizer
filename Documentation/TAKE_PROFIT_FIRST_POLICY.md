# TAKE_PROFIT-First Policy

## Purpose

This repository optimizes for `exit_reason == TAKE_PROFIT`, not for binary directional accuracy.
Directional accuracy remains useful as a diagnostic and research signal, but it is not the
primary production objective under this policy.

## Metric Hierarchy

When multiple quality measures disagree, use the following order:

1. TAKE_PROFIT capture rate
2. Payoff asymmetry
3. Omega ratio versus the NGN hurdle
4. Win rate
5. Directional accuracy

## Domain Constants

The code implementation lives in `etl/domain_objective.py` and defines the canonical
constants for this policy:

- `SYSTEM_OBJECTIVE = "TAKE_PROFIT_CAPTURE"`
- `MIN_OMEGA_VS_HURDLE = 1.0`
- `MIN_TAKE_PROFIT_FREQUENCY = 0.095`
- `TARGET_AMPLITUDE_MULTIPLIER = 2.0`
- `TAKE_PROFIT_FILTER_THRESHOLD_FALLBACK = 0.15`

These constants are the repo-wide source of truth. Selection scripts, eligibility gates,
forecast-audit checks, and downstream reporting must import them rather than duplicating
their own shadow copies.

## Operational Notes

- TAKE_PROFIT-first analysis uses outcome-linked trade history and keeps label timing
  (`holding_period_at_exit`) visible so fast target hits can be separated from slow
  terminal exits.
- Directional classifiers are research-only until a TAKE_PROFIT edge is demonstrated on
  outcome-linked data.
- If a document or config file still implies directional accuracy is the main objective,
  this policy supersedes it.
