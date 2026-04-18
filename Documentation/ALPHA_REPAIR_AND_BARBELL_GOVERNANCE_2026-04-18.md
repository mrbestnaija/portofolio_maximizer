# Alpha Repair and Barbell Governance Review

Date: 2026-04-18

## What Was Hardened

- Candidate evaluation is now benchmark-relative and causal.
- Live candidate sizing knobs now only reduce additive exposure, not exits, so alpha repair does not distort the close-side lifecycle.
- Benchmark-relative NGN metrics now expose `alpha`, `information_ratio`, `beta`, and `tracking_error` for downstream scoring and audit.
- Barbell acceptance no longer depends on a single attractive `omega_ratio` or a high `payoff_asymmetry`.
- The simulator now publishes an explicit `anti_barbell_evidence` bundle from executed trades:
  - `omega_monotonicity_ok` and `omega_cliff_ok` for threshold sensitivity
  - `omega_right_tail_ok` for right-tail confidence
  - `es_to_edge_bounded` for left-tail containment
  - `barbell_path_risk_ok` for liquidity/path risk
  - `regime_realism_ok` for regime realism
  - `anti_barbell_ok` as the aggregate fail-closed verdict
- The optimization layer now treats those booleans as hard constraints, so the optimizer cannot promote a candidate on upside shape alone.

## Config Drift Fixed

- `config/pipeline_config.yml` now matches `config/forecasting_config.yml` on the ensemble controls that affect live behavior:
  - `confidence_scaling`
  - `track_directional_accuracy`
  - `prefer_diversified_candidate`
  - `diversity_tolerance`
  - `da_floor`
  - `da_weight_cap`
- `scripts/validate_forecasting_configs.py` now rejects drift in those fields instead of silently accepting mismatched ensemble semantics.

## Why This Matters

This closes the main anti-barbell failure modes:

- bad threshold selection
- right-tail overestimation
- unbounded left tail
- path/liquidity risk blind spots
- regime realism bypass

Raw upside metrics remain visible for audit, but they are no longer sufficient for acceptance.

## Verification

Focused tests:

- `python -m pytest tests/execution/test_paper_trading_engine.py tests/etl/test_portfolio_math_ngn.py -q`
- `python -m pytest tests/backtesting/test_candidate_simulator.py tests/scripts/test_run_backtest_for_candidate.py tests/scripts/test_run_strategy_optimization.py tests/etl/test_strategy_optimizer.py tests/scripts/test_validate_forecasting_configs.py -q`

Fast regression lane:

- `python -m pytest -m "not gpu and not slow" --tb=short -q`

Observed results:

- live sizing + NGN alpha tests: `69 passed`
- focused alpha/config tests: `27 passed`
- fast regression lane: `2515 passed, 6 skipped, 45 deselected, 11 xfailed`

## Review Note

This patch is ready for Claude review and integration, but it is intentionally not merged here. The workspace still contains unrelated pre-existing OpenClaw edits that should be reviewed separately.
