# Comprehensive Brutal Test Report
**Date**: Wed Nov 12 05:41:07 WAT 2025
**Duration**: 0h 41m (2469s)
**Test Root**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251112_045958

## Test Summary

**Total Tests**: 38
**Passed**: 35
**Failed**: 3
**Pass Rate**: 92.1%

## Stage Results

| Stage | Passed | Failed |
|-------|--------|--------|
| stage | passed | failed |
| profit_critical | 3 | 0 |
| etl_unit_tests | 5 | 1 |
| time_series_forecasting | 2 | 0 |
| signal_routing | 2 | 0 |
| integration_tests | 2 | 0 |
| llm_integration | 3 | 1 |
| security_tests | 1 | 0 |
| pipeline_execution | 3 | 0 |
| profit_validation | 0 | 1 |
| monitoring_suite | 2 | 0 |
| nightly_backfill | 0 | 1 |
| database_integrity | 4 | 0 |

## Test Artifacts

- **Logs**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251112_045958/logs
- **Reports**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251112_045958/reports
- **Artifacts**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251112_045958/artifacts
- **Performance**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251112_045958/performance

## Compliance

✅ **AGENT_INSTRUCTION.md**: Focused on profit-critical functions
✅ **AGENT_DEV_CHECKLIST.md**: Used existing test patterns
✅ **API_KEYS_SECURITY.md**: No API keys exposed in logs
✅ **CHECKPOINTING_AND_LOGGING.md**: Used existing logging systems
✅ **arch_tree.md**: Tested existing modules only

## Recommendations

- ⚠️  Review failed tests in /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251112_045958/logs
- ✅ All profit-critical tests should pass before production
