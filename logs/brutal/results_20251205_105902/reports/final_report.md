# Comprehensive Brutal Test Report
**Date**: Fri Dec  5 13:09:08 WAT 2025
**Duration**: 2h 10m (7806s)
**Test Root**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251205_105902

## Test Summary

**Total Tests**: 42
**Passed**: 42
**Failed**: 0
**Pass Rate**: 100.0%
**Quant validation health (global)**: RED

## Stage Results

| Stage | Passed | Failed |
|-------|--------|--------|
| stage | passed | failed |
| profit_critical | 3 | 0 |
| etl_unit_tests | 6 | 0 |
| time_series_forecasting | 2 | 0 |
| signal_routing | 2 | 0 |
| integration_tests | 2 | 0 |
| llm_integration | 4 | 0 |
| security_tests | 1 | 0 |
| pipeline_execution | 4 | 0 |
| profit_validation | 1 | 0 |
| monitoring_suite | 2 | 0 |
| nightly_backfill | 1 | 0 |
| database_integrity | 4 | 0 |

## Test Artifacts

- **Logs**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251205_105902/logs
- **Reports**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251205_105902/reports
- **Artifacts**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251205_105902/artifacts
- **Performance**: /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45/logs/brutal/results_20251205_105902/performance

## Compliance

✅ **AGENT_INSTRUCTION.md**: Focused on profit-critical functions
✅ **AGENT_DEV_CHECKLIST.md**: Used existing test patterns
✅ **API_KEYS_SECURITY.md**: No API keys exposed in logs
✅ **CHECKPOINTING_AND_LOGGING.md**: Used existing logging systems
✅ **arch_tree.md**: Tested existing modules only

## Recommendations

- ✅ All profit-critical tests should pass before production
