# Agent C Readiness Blocker Matrix (2026-03-07)

Purpose: keep Agent C on measurement, sequencing, and evidence only.

This document does not authorize any strategy change, experiment execution, or readiness claim.

## Verified Inputs

Commands run on 2026-03-07:

- `python scripts/capital_readiness_check.py --json`
- `python scripts/run_all_gates.py --json`
- `python scripts/check_forecast_audits.py --audit-dir logs/forecast_audits --db data/portfolio_maximizer.db --max-files 500`
- `python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db`
- `python scripts/check_model_improvement.py --layer 1 --json`
- `python -m pytest tests/scripts/test_capital_readiness_check.py tests/scripts/test_run_quality_pipeline.py tests/scripts/test_validate_profitability_proof_hardening.py tests/forcester_ts/test_regime_detector_stability.py -q`
- `python -m pytest tests/scripts/test_apply_ticker_eligibility_gates.py -q`
- direct SQLite checks on `portfolio_positions` and `performance_metrics`
- `logs/overnight_denominator/live_denominator_latest.json`

## Current State

### 1. Fresh denominator lane

Source: `logs/forecast_audits_cache/latest_summary.json`

- `n_outcome_windows_eligible = 1`
- `n_outcome_windows_matched = 0`
- `n_outcome_windows_missing = 1`
- `n_linkage_denominator_included = 1`
- `n_outcome_windows_non_trade_context = 48`
- `n_outcome_windows_invalid_context = 5`
- `n_outcome_windows_missing_execution_metadata = 3`

Interpretation:

- Denominator recovery exists, but it is still too thin.
- Fresh production-valid matching has not started.
- Non-trade contamination remains a real hygiene blocker in the broader audit population.

### 2. Watcher lane

Source: `logs/overnight_denominator/live_denominator_latest.json`

- `status = WAITING`
- `cycles_completed = 0`
- `run_id = 20260307_070513`
- `sleep_seconds = 86400`
- `weekdays_only = true`

Interpretation:

- No new denominator evidence is expected until the next trading day.
- Weekend idle time is not evidence progress or regression.

### 3. Production gate

Source: `python scripts/run_all_gates.py --json`

- `overall_passed = false`
- `phase3_ready = false`
- `phase3_reason = GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL`
- `production_audit_gate` reports `matched=0/1`

Interpretation:

- Phase 3 remains blocked.
- Agent C must not discuss readiness or experiment execution while this remains true.

### 4. Capital readiness

Source: `python scripts/capital_readiness_check.py --json`

- `ready = false`
- `verdict = FAIL`
- `R2` fails because `run_all_gates.py` currently reports `overall_passed = false`
- `R3` now fails on real trade metrics:
  - `win_rate = 40.0% < 45%`
  - `profit_factor = 0.80 < 1.30`
- `R5` now hard-fails with:
  - `ensemble lift is definitively negative CI=[-0.1139, -0.0572] across 162 windows`

Interpretation:

- Capital readiness is now failing for real gate reasons, not the earlier R3 wiring crash.
- The R5 threshold-dodge in `capital_readiness_check.py` appears closed.
- The remaining semantic issue is upstream: Layer 1 still describes the same negative CI as `spans zero`.

## Critical Complement To Parallel Work

### 1. R5 hard-fail wiring is now live, but Layer 1 wording is still wrong

Evidence:

- targeted tests in `tests/scripts/test_capital_readiness_check.py` now pass
- live runtime now emits:
  - `R5: ensemble lift is definitively negative CI=[-0.1139, -0.0572] across 162 windows ...`
- `python scripts/check_model_improvement.py --layer 1 --json` still emits a summary that says:
  - `lift CI [-0.1139, -0.0572] spans zero`

Interpretation:

- The end-to-end R5 hard-fail contract in `capital_readiness_check.py` now appears to be wired correctly.
- The remaining issue is reporting inconsistency between Layer 1 and capital readiness.
- Agent A should treat the Layer 1 wording bug as a merge blocker because it still downplays definitive negative lift in shared reporting.

### 2. Layer 1 summary text is still semantically wrong

Evidence:

- `python scripts/check_model_improvement.py --layer 1 --json` reports:
  - `lift_ci_low = -0.1139`
  - `lift_ci_high = -0.0572`
  - summary still says `spans zero`
- `scripts/check_model_improvement.py` currently labels any `ci_low <= 0` as `spans zero`

Interpretation:

- Even after the capital-readiness fix lands, Layer 1 will still mislabel a fully negative interval unless this branch is tightened to require `ci_low <= 0 <= ci_high`.
- Agent C should not treat current lift wording as trustworthy.

### 3. Quality-pipeline eligibility wiring is promising but not merge-complete

Evidence:

- targeted tests now pass for:
  - `tests/scripts/test_run_quality_pipeline.py`
  - `tests/scripts/test_apply_ticker_eligibility_gates.py`

Interpretation:

- The integration direction looks correct.
- The remaining architectural question is failure policy:
  - missing eligibility input still degrades to `WARN + empty ticker sets`
- Agent B should decide whether that is intentional or whether downstream automation needs fail-closed semantics.

### 4. Regime-detector stability improved, but C2 is still semantically blocked

Evidence:

- targeted finite-output tests pass in `tests/forcester_ts/test_regime_detector_stability.py`
- current implementation clamps Hurst tau to epsilon and can return a near-zero flat-series Hurst instead of a neutral `0.5`

Interpretation:

- The NaN/inf problem is improved.
- The semantic contract for flat-series Hurst is still unresolved.
- Experiment `C2` stays blocked until Agent A chooses and tests the intended flat-series meaning explicitly.

### 5. Profitability-proof import hardening looks correct but is secondary

Evidence:

- targeted tests in `tests/scripts/test_validate_profitability_proof_hardening.py` pass
- `scripts/validate_profitability_proof.py` now resolves the repo root before import and falls back cleanly when the guarded connect import is missing

Interpretation:

- This is a useful wiring fix.
- It is not the current critical-path blocker for Agent C.
- The higher-priority risk remains thin denominator evidence, Layer 1 lift mislabeling, and dashboard truth semantics.

### 5. Dashboard truth path

Source: `visualizations/dashboard_data.json` plus direct DB checks

- `checks = ["performance_metrics missing; run performance aggregation."]`
- `robustness.status = STALE`
- `positions_count = 2`
- `portfolio_positions_latest_date = 2026-02-19`
- `performance_metrics_count = 0`
- payload still shows:
  - `pnl_pct = 0.0`
  - `sharpe = 0.0`
  - `sortino = 0.0`
  - `max_drawdown = 0.0`

Interpretation:

- The dashboard is connected, but not fully authoritative.
- Unknown performance fields are still visually indistinguishable from real zero values.
- Stale `portfolio_positions` data is still present in the DB-backed view.

## Blocker Matrix

| Surface | Current result | Blocking owner | Unblock condition |
|---|---|---|---|
| Fresh TRADE denominator | `linkage_included = 1`, `matched = 0` | A/B (producer + reporting truth) | `fresh_linkage_included > 1` across multiple cycles and `fresh_production_valid_matched >= 1` |
| Phase 3 readiness | `false` | A | `GATES_FAIL`, `THIN_LINKAGE`, and `EVIDENCE_HYGIENE_FAIL` all cleared |
| Capital readiness | `FAIL` | A | `R2`, `R3`, and `R5` all need to clear on real evidence, not wiring placeholders |
| Lift semantics | capital-readiness now hard-fails negative lift, but Layer 1 still mislabels the same interval as `spans zero` | A | Layer 1 wording no longer calls a fully negative CI `spans zero` |
| Dashboard truth | stale positions + zero-filled unknown performance | B | stale/unknown semantics rendered honestly; `exit_reason` present in trade feed |
| Experiment execution | blocked | Agent C protocol | do not start until denominator and reporting preconditions are satisfied |

## Agent C Operating Rules

- Do not start experiments.
- Do not recommend strategy changes.
- Do not interpret waiting as progress.
- Report only verified outputs from commands and artifacts.

## Promotion Rule For Agent C

Agent C may move from "blocked experiment planning" to "experiment candidate ready for implementation review" only when all of the following are true:

1. fresh `TRADE` exclusions stay near zero across multiple cycles
2. `fresh_linkage_included > 1`
3. at least one fresh production-valid matched row appears
4. dashboard/readiness reporting truth blockers are explicitly resolved or waived by Agent A
