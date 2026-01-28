# Comprehensive Brutal Testing Suite

## Overview

The `comprehensive_brutal_test.sh` script provides exhaustive, multi-hour testing of the entire Portfolio Maximizer project. It tests all stages, components, data sources, and configurations with multiple iterations to ensure robust operation.

### Synthetic input for brutal/offline runs
- Generate reproducible synthetic datasets with `scripts/generate_synthetic_dataset.py` (manifest + dataset_id under `data/synthetic/<dataset_id>/`) and validate them via `scripts/validate_synthetic_dataset.py`.
- Set `ENABLE_SYNTHETIC_PROVIDER=1` and either `SYNTHETIC_DATASET_PATH` or `SYNTHETIC_DATASET_ID` before invoking `bash/comprehensive_brutal_test.sh` to force the adapter to read the persisted dataset instead of regenerating in-process. Keep live trading disabled; promotion to live data requires GREEN/acceptable YELLOW quant health per `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`.

### Robust test runbook (TS ensemble gating)
- Required env: `TS_FORECAST_AUDIT_DIR=logs/forecast_audits` (or custom path), `TS_FORECAST_MONITOR_CONFIG=config/forecaster_monitoring_ci.yml` for CI/brutal (fast gating) or `config/forecaster_monitoring.yml` for research dashboards, and synthetic date spans long enough to yield ≥ `regression_metrics.min_effective_audits` holdout windows (CI defaults: 3/5; research: 10/20).
- Sequence for CI/brutal: (1) run synthetic pipeline `scripts/run_etl_pipeline.py --execution-mode synthetic --prefer-gpu` (or `bash/comprehensive_brutal_test.sh`), (2) call `simpleTrader_env/bin/python scripts/check_forecast_audits.py --config-path $TS_FORECAST_MONITOR_CONFIG` to enforce RMSE gate + holding-period rules, (3) run `simpleTrader_env/bin/python scripts/check_quant_validation_health.py` for broader quant gates.
- Expectations: during holding period the gate is inconclusive unless `fail_on_violation_during_holding_period` trips; after holding period, ensemble must beat `baseline_model=BEST_SINGLE` within `max_rmse_ratio_vs_baseline` or be demoted to research-only. Promotion requires meeting `promotion_margin` (CI: 0%, research: 2%).

### 🚨 2025-11-15 Brutal Run Findings (blocking)
- `logs/pipeline_run.log:16932-17729` plus `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` prove the SQLite store is corrupted (“rowid … out of order”, “row … missing from index”), so every writer in `etl/database_manager.py:689`/`:1213` now raises `database disk image is malformed`. The brutal suite currently exercises a broken database.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` show the time-series stage repeatedly failing with `ValueError: The truth value of a DatetimeIndex is ambiguous` after the MSSA serialization block in `scripts/run_etl_pipeline.py:1755-1764` evaluates `change_points or []`. Because the exception fires after the “Saved forecast …” messages, the suite still logs IDs 871‑960 and then reports “Generated forecasts for 0 ticker(s)”.
- The visualization hook crashes immediately afterwards (`FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` at log lines 2626, 2981, …), so the dashboard export artefacts listed later in this README are currently not produced.
- Hardening notes in `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md` claim the DatetimeIndex ambiguity was fixed, yet pandas/statsmodels continue to emit `PeriodDtype[B]` FutureWarnings and `ValueWarning`/`ConvergenceWarning` spam because `forcester_ts/forecaster.py:128-136` still does a deprecated `PeriodIndex` round-trip and `_select_best_order` in `forcester_ts/sarimax.py:136-183` keeps unstable combinations in the grid.
- `scripts/backfill_signal_validation.py:281-292` still uses `datetime.utcnow()` and sqlite’s default converters, which triggers the Python 3.12 deprecation warnings documented in `logs/backfill_signal_validation.log:15-22`. The brutal job will continue to warn until the script is modernised.

**Immediate actions before rerunning the brutal suite**
1. Recover or recreate `data/portfolio_maximizer.db`, then update `DatabaseManager._connect` so `"database disk image is malformed"` follows the same reset/mirror path we already use for `"disk i/o error"`.
2. Patch `scripts/run_etl_pipeline.py` to copy MSSA `change_points` to a list (without boolean coercion), re-run `python scripts/run_etl_pipeline.py --stage time_series_forecasting`, and verify Stage 8 receives usable forecasts.
3. Drop the unsupported `axis=` argument when calling `FigureBase.autofmt_xdate()` in the dashboard loader, generate a PNG, and attach it to the brutal output folder.
4. Replace the Period coercion inside `forcester_ts/forecaster.py`, tighten the SARIMAX search space (`forcester_ts/sarimax.py:136-183`), and extend regression tests so pandas/statsmodels warnings stop polluting the brutal logs.
5. Update `scripts/backfill_signal_validation.py` to use timezone-aware timestamps (`datetime.now(timezone.utc)`) and explicit sqlite adapters before the nightly run that the brutal harness triggers.

> **2025-11-19 remediation note**  
> Actions 1-4 are now implemented in code (`etl/database_manager.py`, `scripts/run_etl_pipeline.py`, `etl/visualizer.py`, `forcester_ts/forecaster.py`, `forcester_ts/sarimax.py`) and tracked in `Documentation/integration_fix_plan.md`. Synthetic/brutal runs have also been redirected to an isolated SQLite file (`data/test_database.db`) via the `PORTFOLIO_DB_PATH` override in `bash/comprehensive_brutal_test.sh`, so the primary `data/portfolio_maximizer.db` is no longer stressed by these tests. Subsequent work has modernised `scripts/backfill_signal_validation.py` and validated a fresh brutal run; the remaining gating factor for declaring the brutal gate fully GREEN is global quant validation health (currently RED per `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` and `scripts/check_quant_validation_health.py`).

### ✅ 2025-11-16 Regression Fixes (validated via `logs/pipeline_run.log:22237-22986`)
- `etl/database_manager.py` now normalises SQLite paths, backs up corrupt stores, and automatically mirrors `/mnt/*` deployments into `/tmp` without throwing warnings—all recent runs show clean INFO logs while writes succeed.
- `scripts/run_etl_pipeline.py` serialises MSSA change points explicitly, so Stage 7/8 complete and IDs 361‑393 persist without DatetimeIndex crashes.
- `etl/visualizer.py` monkey patches `Figure.autofmt_xdate` to ignore unsupported `axis` kwargs, restoring dashboard PNG exports for brutal artefacts.
- `forcester_ts/forecaster.py` suppresses KPSS `InterpolationWarning`s and `forcester_ts/sarimax.py` demotes the fallback-order notice to INFO, preventing the warning storm highlighted above.
- Finnhub availability is now treated as optional; when `FINNHUB_API_KEY` is absent the manager logs a single INFO line instead of spamming warnings.
- Outstanding item (historical): `scripts/backfill_signal_validation.py` still needs timezone-aware adapters before nightly brutal tasks can be marked green. This has since been addressed; the script now uses timezone-aware timestamps and explicit sqlite adapters, and its datetime converter is covered by `tests/scripts/test_backfill_signal_validation.py`.
- **DB isolation for tests (Nov 19)**: `bash/comprehensive_brutal_test.sh` exports `PORTFOLIO_DB_PATH=data/test_database.db` for synthetic runs so brutal/test pipelines write into a dedicated SQLite file instead of the production `data/portfolio_maximizer.db`.
- **Transparency upgrade (Nov 16)**: `forcester_ts/instrumentation.py` captures dataset shape/frequency/statistics per stage and `TimeSeriesVisualizer` renders those summaries directly on dashboards (see `logs/forecast_audits/*.json`). Every brutal report now references concrete data dimensions rather than hand-wavy descriptions.

### 2025-12-04 Update (TS/LLM guardrails + MVS visibility)
- Time Series signals now respect a **quant-success hard gate**: `models/time_series_signal_generator.TimeSeriesSignalGenerator` attaches a `quant_profile` from `config/quant_success_config.yml` and demotes BUY/SELL to HOLD when `status == "FAIL"` outside diagnostic modes. Brutal TS/forecaster stages will therefore see HOLD-only regimes unless the quant-success thresholds are met.
- Automated trading uses an **LLM readiness gate**: `scripts/run_auto_trader.py` only allows LLM fallback when `data/llm_signal_tracking.json` reports at least one validated signal; brutal/diagnostic runs must explicitly enable LLM behaviour and should not treat it as production-ready until this gate passes.
- End-to-end and live orchestrators (`bash/run_end_to_end.sh`, `bash/run_pipeline_live.sh`) now emit **MVS-style summaries** after runs (total trades, total profit, win rate, profit factor, MVS PASS/FAIL) using `DatabaseManager.get_performance_summary()` over either full history or an operator-specified window (`MVS_START_DATE` / `MVS_WINDOW_DAYS`). Brutal reports should cross-reference these metrics when judging readiness for live capital.

## Features

### ✅ Comprehensive Stage Testing
- **Data Extraction**: Tests all data sources (yfinance, Alpha Vantage, Finnhub)
- **Data Validation**: Validates data quality and integrity
- **Data Preprocessing**: Tests data cleaning and normalization
- **Data Storage**: Verifies database persistence
- **Time Series Forecasting**: Tests SARIMAX, GARCH, SAMOSSA, MSSA-RL models
- **Signal Generation**: Tests Time Series signal generation
- **Signal Routing**: Tests signal routing with Time Series primary, LLM fallback
- **LLM Integration**: Legacy Ollama-backed checks (disabled by default; only runs when explicitly enabled)

### ✅ Multi-Source Testing
- Tests each data source individually
- Validates data extraction from multiple providers
- Verifies data consistency across sources

### ✅ Performance Benchmarking
- Runs 10 performance iterations
- Measures execution time, memory usage, CPU utilization
- Generates performance statistics and reports

### ✅ Error Handling Tests
- Invalid ticker handling
- Invalid date range handling
- Missing data source handling
- Graceful failure testing

### ✅ Stress Testing
- 20 consecutive pipeline runs
- Tests system stability under load
- Identifies memory leaks or resource issues

### ✅ Database Integrity Tests
- Table existence verification
- Data consistency checks
- Foreign key integrity validation
- Data count verification

## Usage

### Basic Usage

```bash
# Run with default settings (4 hours, 5 iterations per test)
./bash/comprehensive_brutal_test.sh
```

### Custom Configuration

Set environment variables before running:

```bash
# Set test duration (hours)
export TEST_DURATION_HOURS=6

# Set iterations per test
export ITERATIONS_PER_TEST=10

# Set tickers to test (frontier list appended automatically via --include-frontier-tickers)
export TICKERS_LIST="AAPL,MSFT,GOOGL,TSLA,AMZN,NVDA"

# Set date range
export START_DATE="2020-01-01"
export END_DATE="2024-01-01"

# Run the test
./bash/comprehensive_brutal_test.sh
```

### On Windows (WSL/Git Bash)

```bash
# Make executable (if needed)
chmod +x bash/comprehensive_brutal_test.sh

# Run
bash bash/comprehensive_brutal_test.sh
```

The brutal runner passes `--include-frontier-tickers`, so you do **not** need to manually append the Nigeria → Bulgaria symbols; `etl/frontier_markets.py` handles that automatically for every multi-ticker stage.

## Test Structure

### Test Execution Order

1. **Environment Setup**
   - Virtual environment activation
   - Package verification
   - Ollama availability check
   - Test database creation

2. **Stage-by-Stage Testing**
   - Each stage tested individually
   - Multiple iterations per stage
   - All data sources tested

3. **Multi-Source Testing**
   - Cross-source validation
   - Data consistency checks

4. **Performance Benchmarking**
   - 10 performance runs
   - Statistics generation

5. **Error Handling Tests**
   - Invalid input handling
   - Graceful failure testing

6. **Stress Testing**
   - 20 consecutive runs
   - System stability validation

7. **Database Integrity Tests**
   - Schema validation
   - Data integrity checks

8. **Report Generation**
   - Final report with all results
   - Performance statistics
   - Pass/fail summaries

## Output Structure

```
logs/brutal/results_YYYYMMDD_HHMMSS/
├── logs/                          # Individual test logs
│   ├── stage_data_extraction.log
│   ├── stage_data_validation.log
│   ├── stage_time_series_forecasting.log
│   └── ...
├── reports/                       # Generated reports
│   └── final_report.md
├── artifacts/                     # Test artifacts
│   └── test_database.db
├── performance/                   # Performance data
│   ├── performance_results.csv
│   └── time_*.txt
├── stage_summary.csv              # Stage pass/fail summary
└── test.log                       # Main test log
```

## Report Format

The final report includes:

- **Test Summary**: Overall pass/fail statistics
- **Stage Results**: Detailed results for each stage
- **Performance Statistics**: Average, min, max, std dev
- **Logs and Artifacts**: Paths to all generated files
- **Test Configuration**: All test parameters

## Expected Duration

- **Default**: ~4 hours
- **Extended**: 6+ hours (with more iterations)
- **Quick**: 2-3 hours (with fewer iterations)

Duration depends on:
- Number of iterations per test
- Number of tickers tested
- Date range size
- System performance
- Network speed (for data sources)

## Requirements

### System Requirements
- Linux/WSL/Git Bash (for bash script execution)
- Python 3.8+
- Virtual environment activated
- Sufficient disk space (several GB for logs and artifacts)
- Network connectivity (for data sources)

### Python Packages
- pandas
- numpy
- yfinance
- sqlalchemy
- All project dependencies

### Optional
- Ollama (deprecated; only needed if you explicitly re-enable legacy LLM tests)
- `bc` command (for calculations, usually pre-installed)
- `time` command (for performance measurements)

## Interpreting Results

### Pass/Fail Criteria

- **Stage Test**: Passes if pipeline completes without errors
- **Data Validation**: Passes if data is extracted and stored correctly
- **Database Test**: Passes if tables exist and contain data
- **Performance Test**: Always passes (measures performance, not correctness)

### Common Issues

1. **Data Source Failures**
   - Check API keys (Alpha Vantage, Finnhub)
   - Verify network connectivity
   - Check rate limits

2. **Ollama Disabled/Not Available**
   - Legacy LLM checks are skipped by default
   - To re-enable for experiments: set `PM_ENABLE_OLLAMA=1` and ensure `ollama serve` is running

3. **Database Errors**
   - Check disk space
   - Verify SQLite permissions
   - Check for database locks

4. **Performance Degradation**
   - Review performance results
   - Check system resources
   - Look for memory leaks

## Best Practices

1. **Run During Off-Hours**: Tests can take hours
2. **Monitor System Resources**: Ensure sufficient RAM/disk
3. **Check Logs Regularly**: Review logs for early issues
4. **Save Results**: Archive results for comparison
5. **Run Periodically**: Regular testing catches regressions

## Troubleshooting

### Script Won't Run

```bash
# Check if bash is available
which bash

# Check script permissions
ls -l bash/comprehensive_brutal_test.sh

# Make executable
chmod +x bash/comprehensive_brutal_test.sh
```

### Tests Failing

1. Check individual stage logs in `logs/` directory
2. Review main test log: `test.log`
3. Verify environment setup
4. Check database connectivity
5. Verify data source availability

### Performance Issues

1. Reduce `ITERATIONS_PER_TEST`
2. Test fewer tickers
3. Use shorter date ranges
4. Check system resources

## Example Output

```
=== Comprehensive Brutal Testing Suite ===
[INFO] Test Results Directory: logs/brutal/results_20251109_120000
[INFO] Expected Duration: 4 hours
[INFO] Iterations per Test: 5

=== Environment Setup ===
[PASS] Virtual environment activated
[PASS] Required packages available
[PASS] Ollama found
[PASS] Test database created

=== Stage 1: Data Extraction ===
[PASS] Iteration 1 with yfinance (45s)
[PASS] Iteration 2 with yfinance (43s)
...

=== Testing Complete ===
[INFO] Total Duration: 3h 45m
[INFO] Results: logs/brutal/results_20251109_120000
[INFO] Report: reports/final_report.md
```

## Notes

- Tests are designed to be **brutal** - they will find issues
- Some tests may take a long time (especially forecasting)
- Database files can grow large during testing
- Network-dependent tests may fail if APIs are unavailable
- LLM tests require Ollama to be running

## Support

For issues or questions:
1. Check individual stage logs
2. Review the final report
3. Check system resources
4. Verify environment setup

---

**Last Updated**: 2025-11-09  
**Version**: 1.0  
**Status**: Production Ready

