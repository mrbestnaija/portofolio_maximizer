# Comprehensive Brutal Testing Suite

## Overview

The `comprehensive_brutal_test.sh` script provides exhaustive, multi-hour testing of the entire Portfolio Maximizer project. It tests all stages, components, data sources, and configurations with multiple iterations to ensure robust operation.

## Features

### ✅ Comprehensive Stage Testing
- **Data Extraction**: Tests all data sources (yfinance, Alpha Vantage, Finnhub)
- **Data Validation**: Validates data quality and integrity
- **Data Preprocessing**: Tests data cleaning and normalization
- **Data Storage**: Verifies database persistence
- **Time Series Forecasting**: Tests SARIMAX, GARCH, SAMOSSA, MSSA-RL models
- **Signal Generation**: Tests Time Series signal generation
- **Signal Routing**: Tests signal routing with Time Series primary, LLM fallback
- **LLM Integration**: Tests LLM market analysis and signal generation (if Ollama available)

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

# Set tickers to test
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
- Ollama (for LLM integration tests)
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

2. **Ollama Not Available**
   - LLM tests will be skipped
   - Other tests continue normally

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

