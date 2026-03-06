# Phase 7.35 Outcome Linkage Hardening (Pre-Phase 3)

Date: 2026-03-06

## Scope

- Fix outcome denominator corruption from cross-ticker dedupe suppression.
- Fix causal eligibility anchoring (`entry_ts + forecast_horizon`).
- Add fail-closed invalid-context guards.
- Improve denominator truth metrics (`ts_*` coverage vs total closed trades).
- Add adversarial anti-regression checks for linkage semantics.

## Implemented Changes

1. `scripts/check_forecast_audits.py`
- Added separate dedupe maps:
  - RMSE: `(start, end, length, forecast_horizon)`
  - Outcome linkage: `(ticker, start, end, length, forecast_horizon)`
- Added `compute_expected_close(signal_context, dataset)`:
  - primary: `entry_ts + signal_context.forecast_horizon`
  - fallback: dataset end+horizon only when signal context is absent.
- Added explicit outcome taxonomy states:
  - `MATCHED`, `OUTCOME_MISSING`, `NOT_DUE`, `INVALID_CONTEXT`, `NON_TRADE_CONTEXT`
- Added integrity reasons:
  - `CAUSALITY_VIOLATION`, `HORIZON_MISMATCH`, `MISSING_SIGNAL_ID`,
    `EXPECTED_CLOSE_UNAVAILABLE`, `AMBIGUOUS_MATCH`, `OUTCOME_WINDOW_OPEN`
- Added telemetry counters:
  - `n_outcome_deduped_windows`
  - `n_outcome_windows_not_due`
  - `n_outcome_windows_invalid_context`
  - `n_outcome_windows_outcomes_not_loaded`

2. `scripts/outcome_linkage_attribution_report.py`
- Added denominator split metrics:
  - `total_ts_trades`
  - `linked_ts_trades`
  - `linked_ts_trade_ratio`
  - `ts_trade_coverage`

3. `scripts/adversarial_diagnostic_runner.py`
- Added checks:
  - `TCON-06` ticker-aware outcome dedupe enforcement
  - `TCON-07` signal-anchored expected-close enforcement
  - `TCON-08` `NOT_DUE` classification enforcement

## Test Coverage Added

- `tests/scripts/test_check_forecast_audits.py`
  - cross-ticker dedupe suppression regression
  - causality violation classification
  - horizon mismatch classification
  - open-window `NOT_DUE` classification
- `tests/scripts/test_outcome_linkage_attribution_report.py`
  - TS-vs-legacy denominator metrics
- `tests/scripts/test_adversarial_diagnostic_runner.py`
  - TCON-06/TCON-07/TCON-08 anti-regression logic

## Verification Commands

```powershell
python -m pytest tests/scripts/test_check_forecast_audits.py tests/scripts/test_outcome_linkage_attribution_report.py tests/scripts/test_adversarial_diagnostic_runner.py -q
python scripts/check_forecast_audits.py --audit-dir logs/forecast_audits --db data/portfolio_maximizer.db --max-files 500
python scripts/outcome_linkage_attribution_report.py --json
python scripts/adversarial_diagnostic_runner.py --json --severity LOW
python -m pytest -m "not gpu and not slow" --tb=short -q
```

## Phase 3 Gate Note

Phase 3 remains blocked until refreshed telemetry confirms:

- `outcome_matched >= 10`
- `matched / eligible >= 0.8`
- `linked_ts_trades / total_ts_trades >= 0.8`
