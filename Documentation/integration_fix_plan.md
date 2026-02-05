# Integration Fix Plan

## Status (2025-12-25)
Structural fixes from the 2025-11-15 brutal run are now implemented and re-verified. Remaining gating is profitability/quant-health (not engineering breakage). See `Documentation/PROJECT_STATUS.md` for the latest verified snapshot.

## Integration Remediation Tasks

1. **Repair Database Corruption**
   - Recover or rebuild `data/portfolio_maximizer.db`, then extend `DatabaseManager._connect` so `"database disk image is malformed"` follows the existing disk-I/O reset/mirror branch.
   - Re-run `PRAGMA integrity_check;` to confirm the store is healthy before triggering any integration flows.
   - _2025-11-19 update_: `DatabaseManager._connect` now routes `"database disk image is malformed"` through the disk I/O fallback before any rebuild, and the current `data/portfolio_maximizer.db` passes `PRAGMA integrity_check`.

2. **Fix MSSA Change Point Handling**
   - In `scripts/run_etl_pipeline.py:1755-1764`, convert `change_points` from the MSSA bundle into an explicit list instead of relying on boolean coercion of a `DatetimeIndex`.
   - Re-run Stage 7 + Stage 8 to ensure `TimeSeriesSignalGenerator` receives a populated forecast bundle on every ticker.
   - _2025-11-19 update_: Added `_normalize_change_points` to `scripts/run_etl_pipeline.py` so DatetimeIndex/Series/ndarray payloads produce ISO strings without triggering the ambiguous truth-value exception.

3. **Repair Visualization Hook**
   - Remove the unsupported `axis=` argument from the Matplotlib `FigureBase.autofmt_xdate()` call so Stage 7/8 dashboards render and can be asserted again.
   - _2025-11-19 update_: `etl/visualizer.py` now patches `Figure.autofmt_xdate` via `_monkey_patch_autofmt_axis_kwarg` to drop the `axis` kwarg globally before dashboards render, preventing the Matplotlib crash reported in brutal logs (lines 2626+).

4. **Silence SARIMAX Warning Flood**
   - Update `forcester_ts/forecaster.py` and `forcester_ts/sarimax.py` to drop the legacy Period coercion and tighten the SARIMAX order search grid so pandas/statsmodels stop emitting warning spam during brutal runs.
   - _2025-11-19 update_: `forcester_ts/sarimax.py` now treats frequency as a stored hint only (no forced `asfreq`/`PeriodIndex` coercion), while the existing `_should_skip_order` budget and capped grid search continue to bound unconverged SARIMAX candidates; regression tests in `tests/etl/test_time_series_forecaster.py::TestSARIMAX` all pass.

5. **Modernize Signal Backfill Timestamps**
   - Refactor `scripts/backfill_signal_validation.py:281-292` to use timezone-aware timestamps plus sqlite adapters, eliminating the Python 3.12 `datetime.utcnow()` deprecation warnings.
   - _2025-12-25 update_: `scripts/backfill_signal_validation.py` now normalizes timestamps with timezone-aware UTC datetimes and registers sqlite adapters/converters.

6. **Re-run Integration Suite**
   - After each fix, execute `pytest tests/integration/test_time_series_signal_integration.py -v --tb=short` until all integration tests complete end-to-end, then update `Documentation/INTEGRATION_TESTING_COMPLETE.md` with the new status.
   - _2025-12-25 update_: `tests/integration/test_time_series_signal_integration.py` now completes quickly and passes end-to-end (10 tests) after reusing a single forecast bundle fixture to keep runtime bounded.

## Guardrail Checklist (From `AGENT_INSTRUCTION.md`)

- Confirm the prior phase is working, demand profitability evidence (>10% annualized backtest), and keep designs configuration-driven and backward compatible before making changes.
- Stay inside the Tier-1 Python/statsmodels/SAMOSSA stack defined in `Documentation/QUANT_TIME_SERIES_STACK.md`, using only free data sources.
- For every remediation above, document how it drives >1% return improvement, justify the added complexity, validate upstream dependencies, and define a rollback plan before deployment.
