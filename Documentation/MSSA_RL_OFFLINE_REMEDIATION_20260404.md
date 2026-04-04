## MSSA-RL Offline Remediation 2026-04-04

### Purpose

This note records the remediation contract for the MSSA-RL offline-policy redesign after the
second-sweep audit of `Documentation/GATE_LIFT_FIRST_PRINCIPLES_AUDIT_20260329.md`.

The goal is to make the forecasting stack more robust without weakening any gate thresholds,
lift thresholds, or ensemble confidence caps.

### Audit-Confirmed Problems

1. MSSA-RL runtime wiring was split across two incompatible contracts.
   - `forcester_ts/mssa_rl.py` had already moved to an offline artifact model.
   - Runtime config and tests still assumed an online Q-table learner.

2. MSSA-RL config construction was not backward-safe.
   - `TimeSeriesForecaster` instantiated `MSSARLConfig(**mssa_rl_kwargs)` directly.
   - Legacy YAML keys such as `reward_mode` could crash config-driven runs.

3. Fail-closed behavior was incomplete.
   - Missing or invalid artifacts correctly blocked forecast generation.
   - Residual-diagnostics degradation could still pass because readiness only checked key
     presence, not `white_noise == True`.

4. Ensemble containment was incomplete.
   - MSSA-RL could still receive confidence and candidate weight even when not fit for
     live participation.

5. The new trainer workflow was not fully standalone.
   - `scripts/train_mssa_rl_policy.py` depended on import context instead of working
     directly from repo root.

6. MSSA-RL still failed the random-walk non-collapse contract even when supplied with a
   valid offline artifact.
   - The robust response is containment and fail-closed behavior until the policy earns
     promotion with fresh evidence.

### Remediation Contract

The codebase must follow these rules after this remediation:

1. Production MSSA-RL is an offline-policy model, not an online Q-learning model.
   - `fit()` computes reconstructions, state features, and diagnostics.
   - `forecast()` may only run when a valid offline artifact is loaded and the policy is ready.

2. MSSA-RL is fail-closed by default.
   - Missing artifact, invalid artifact, stale artifact, unsupported state, insufficient support,
     or degraded residual diagnostics must prevent live forecast output.

3. Residual-diagnostics readiness is strict.
   - A policy is not `ready` unless residual diagnostics exist and `white_noise` is explicitly true.

4. MSSA-RL stays containment-only in ensemble selection.
   - If policy readiness is not proven, MSSA-RL must not contribute confidence or receive
     meaningful candidate weight in the ensemble selector.

5. Runtime configuration must reflect the offline-policy contract.
   - Runtime config keys:
     - `policy_artifact_path`
     - `policy_fail_closed`
     - `min_policy_state_support`
     - `reward_horizon`
   - Deprecated online-learning knobs may remain in code only for backward compatibility,
     but they must not drive live behavior.

6. Tests must verify the offline contract directly.
   - Replace Q-table mutation tests with artifact/fail-closed tests.
   - Integration tests must either provide a valid artifact fixture or assert the expected
     fail-closed behavior.

### Non-Negotiables

- Do not relax `min_lift_rmse_ratio`.
- Do not relax `strict_preselection_max_rmse_ratio`.
- Do not relax `max_violation_rate`.
- Do not promote MSSA-RL by increasing its ensemble weight while it still fails the
  random-walk non-collapse benchmark.

### Implementation Order

1. Document the contract and remediation scope.
2. Make MSSA-RL runtime construction backward-safe.
3. Align YAML/runtime config with the offline-policy model.
4. Harden policy readiness and fail-closed behavior.
5. Exclude non-ready MSSA-RL from ensemble confidence and candidate scoring.
6. Replace stale Q-learning tests with offline-policy contract tests.
7. Re-run targeted MSSA/GARCH/SAMOSSA/monitoring lanes and the fast regression lane.

