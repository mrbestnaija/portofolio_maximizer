# Comprehensive Brutal Test Report
**Date**: Thu Nov 20 00:21:25 WAT 2025
**Duration**: 2h 14m (8063s)
**Test Root**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251119_220702

## Test Summary

**Total Tests**: 34
**Passed**: 32
**Failed**: 2
**Pass Rate**: 94.1%

## Stage Results

| Stage | Passed | Failed |
|-------|--------|--------|
| stage | passed | failed |
| profit_critical | 3 | 0 |
| etl_unit_tests | 5 | 1 |
| time_series_forecasting | 2 | 0 |
| signal_routing | 2 | 0 |
| integration_tests | 2 | 0 |
| llm_integration | 0 | 0 |
| security_tests | 1 | 0 |
| pipeline_execution | 4 | 0 |
| profit_validation | 0 | 1 |
| monitoring_suite | 1 | 1 |
| nightly_backfill | 1 | 0 |
| database_integrity | 4 | 0 |

## Test Artifacts

- **Logs**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251119_220702/logs
- **Reports**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251119_220702/reports
- **Artifacts**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251119_220702/artifacts
- **Performance**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251119_220702/performance

## Compliance

✅ **AGENT_INSTRUCTION.md**: Focused on profit-critical functions
✅ **AGENT_DEV_CHECKLIST.md**: Used existing test patterns
✅ **API_KEYS_SECURITY.md**: No API keys exposed in logs
✅ **CHECKPOINTING_AND_LOGGING.md**: Used existing logging systems
✅ **arch_tree.md**: Tested existing modules only

## Recommendations

- ⚠️  Review failed tests in /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251119_220702/logs
- ⚙️  LLM fallback suite intentionally skipped (BRUTAL_ENABLE_LLM=0). Time Series brutal checks are the approval gate per TIME_SERIES_FORECASTING_IMPLEMENTATION.md.
- ✅ All profit-critical tests should pass before production
