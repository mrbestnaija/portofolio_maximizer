# Comprehensive Brutal Test Report
**Date**: Sun Nov 16 07:05:20 WAT 2025
**Duration**: 6h 0m (21626s)
**Test Root**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251116_010454

## Test Summary

**Total Tests**: 36
**Passed**: 32
**Failed**: 4
**Pass Rate**: 88.9%

## Stage Results

| Stage | Passed | Failed |
|-------|--------|--------|
| stage | passed | failed |
| profit_critical | 3 | 0 |
| etl_unit_tests | 5 | 1 |
| time_series_forecasting | 2 | 0 |
| signal_routing | 2 | 0 |
| integration_tests | 2 | 0 |
| llm_integration | 2 | 2 |
| security_tests | 1 | 0 |
| pipeline_execution | 3 | 0 |
| profit_validation | 0 | 1 |
| monitoring_suite | 1 | 1 |
| nightly_backfill | 1 | 0 |
| database_integrity | 2 | 0 |

## Test Artifacts

- **Logs**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251116_010454/logs
- **Reports**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251116_010454/reports
- **Artifacts**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251116_010454/artifacts
- **Performance**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251116_010454/performance

## Compliance

✅ **AGENT_INSTRUCTION.md**: Focused on profit-critical functions
✅ **AGENT_DEV_CHECKLIST.md**: Used existing test patterns
✅ **API_KEYS_SECURITY.md**: No API keys exposed in logs
✅ **CHECKPOINTING_AND_LOGGING.md**: Used existing logging systems
✅ **arch_tree.md**: Tested existing modules only

## Recommendations

- ⚠️  Review failed tests in /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251116_010454/logs
- ✅ All profit-critical tests should pass before production
