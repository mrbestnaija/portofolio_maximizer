# System Error Monitoring Guide - 2025-10-22

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

## Overview

This guide provides comprehensive information about the enhanced error monitoring and management system implemented for the Portfolio Maximizer v45 project.

### Frontier Market Telemetry (2025-11-15)
- Monitoring dashboards must now track the Nigeria → Bulgaria frontier ticker atlas introduced via `etl/frontier_markets.py` and the `--include-frontier-tickers` flag. Synthetic runs remain the default, but alerts should note whether a failure originated while exercising frontier symbols (per `bash/test_real_time_pipeline.sh` Step 10 and the brutal frontier stage) so spread/liquidity anomalies are triaged separately from mega-cap incidents.

### 🚨 2025-11-15 Brutal Run Findings (monitoring priority)
- The newest brutal run logs (`logs/pipeline_run.log:16932-17729`) plus `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` show persistent `database disk image is malformed` errors (rowids out of order/missing from index). The monitoring stack must treat this as a P0 signal—`DatabaseManager._connect` needs the same reset/mirror handling for this message as it already has for `"disk i/o error"`.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` now emit repeated `ValueError: The truth value of a DatetimeIndex is ambiguous` because `scripts/run_etl_pipeline.py:1755-1764` evaluates `mssa_result.get('change_points') or []`. Alerting should be updated to flag this regression until the MSSA serialization code is patched.
- Dashboard generation crashes with `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (`logs/pipeline_run.log:2626, 2981, …`), so the monitoring guide must note that visualization evidence is currently unavailable.
- Pandas/statsmodels warning storms (PeriodDtype + convergence) have returned because `forcester_ts/forecaster.py:128-136` and `_select_best_order` in `forcester_ts/sarimax.py:136-183` remain unpatched despite previous hardening claims. *(Nov 18 update: the frequency coercion was removed and SARIMAX series are now auto-scaled with a capped grid; monitoring should still watch `logs/warnings/warning_events.log` for any new ConvergenceWarnings.)*
- `scripts/backfill_signal_validation.py:281-292` continues to use `datetime.utcnow()` with sqlite’s default converters, leading to the Python 3.12 deprecation warnings surfaced in `logs/backfill_signal_validation.log:15-22`. Monitoring should treat these warnings as actionable items rather than noise.

**Monitoring action items**
1. Add explicit alerts for “database disk image is malformed” so corruption is escalated immediately.
2. Track occurrences of the MSSA `DatetimeIndex` ambiguity (`scripts/run_etl_pipeline.py:1755-1764`) and block promotion until the change_points handling is fixed.
3. Capture and alert on the Matplotlib `autofmt_xdate(axis=…)` regression so visualization failures are not silent.
4. Expand warning classification to include pandas/statmodels FutureWarnings coming from `forcester_ts/forecaster.py`/`forcester_ts/sarimax.py`.
5. Flag deprecation warnings originating in `scripts/backfill_signal_validation.py` until the script uses timezone-aware timestamps and sqlite adapters.

### ✅ 2025-11-16 Monitoring Update
- Database corruption detection is now proactive: `etl/database_manager.py` backs up malformed files and falls onto a `/tmp` mirror without emitting warnings (see `logs/pipeline_run.log:22245-22250`).
- MSSA serialization no longer trips DatetimeIndex exceptions, so the monitoring stack can downgrade that alert to “resolved” while Stage 7+ metrics remain populated.
- The visualization hook was hardened (`etl/visualizer.py`), eliminating the `autofmt_xdate(axis=…)` alert and restoring dashboard evidence this guide references.
- KPSS and SARIMAX fallback chatter has been demoted to INFO by guards in `forcester_ts/forecaster.py` and `forcester_ts/sarimax.py`, reducing noise in `pipeline_events`.
- Outstanding alert: keep watching for the UTC deprecation warnings from `scripts/backfill_signal_validation.py` until its timezone overhaul lands.
- Data-focused instrumentation (`forcester_ts/instrumentation.py`) now records dataset shape/frequency/statistics every run, and the comprehensive dashboard prints the summary directly on the figure—monitoring teams can open `logs/forecast_audits/*.json` to inspect the exact data profile that triggered an alert.
- Nov 18 update: the same database manager now rebuilds a fresh SQLite store automatically when “database disk image is malformed” occurs mid-run, so alerting can focus on the timestamp of the self-heal rather than hundreds of repeated write failures.

---

## 🚨 **Error Monitoring System**

### **Components**

1. **Enhanced Error Monitor** (`scripts/error_monitor.py`)
   - Real-time error monitoring
   - Automated alerting system
   - Error categorization and analysis
   - Historical error reporting

2. **Cache Management System** (`scripts/cache_manager.py`)
   - Automated Python cache clearing
   - Cache health monitoring
   - Import validation
   - Performance optimization

3. **Method Signature Validation** (`tests/etl/test_method_signature_validation.py`)
   - Automated testing for method signature changes
   - Parameter validation
   - Backward compatibility testing
   - Performance testing

### **Error Categories**

- **TypeError**: Method signature mismatches, parameter type errors
- **ValueError**: Invalid parameter values, data validation errors
- **ConnectionError**: Network connectivity issues, API failures
- **Other**: Miscellaneous errors not in above categories

---

## 🔧 **Usage Instructions**

### **1. Error Monitoring**

#### **Basic Monitoring**
```bash
# Run error monitoring
python scripts/error_monitor.py

# Monitor with specific thresholds
python scripts/error_monitor.py --max-errors-per-hour 10
```

#### **Automated Monitoring**
```bash
# Add to crontab for hourly monitoring
0 * * * * cd /path/to/project && python scripts/error_monitor.py

# Add to systemd for continuous monitoring
systemctl enable portfolio-maximizer-error-monitor
```

### **2. Cache Management**

#### **Manual Cache Clearing**
```bash
# Clear all caches
python scripts/cache_manager.py

# Check cache health only
python scripts/cache_manager.py --check-only
```

#### **Automated Cache Management**
```bash
# Run daily cache cleanup
0 2 * * * cd /path/to/project && python scripts/cache_manager.py

# Use generated cleanup script
./scripts/cleanup_cache.sh
```

### **3. Method Signature Testing**

#### **Run Signature Tests**
```bash
# Run all signature validation tests
python -m pytest tests/etl/test_method_signature_validation.py -v

# Run specific test
python -m pytest tests/etl/test_method_signature_validation.py::TestMethodSignatureValidation::test_train_validation_test_split_signature -v
```

---

## 📊 **Monitoring Configuration**

### **Error Thresholds**

```yaml
# config/error_monitoring_config.yml
error_thresholds:
  max_errors_per_hour: 5
  max_errors_per_day: 20
  critical_error_types:
    - TypeError
    - ValueError
    - ConnectionError
  alert_cooldown_minutes: 30
```

### **Alert Settings**

```yaml
alerts:
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your-email@gmail.com"
    password: "your-password"
    recipients:
      - "admin@company.com"
  
  file:
    enabled: true
    path: "logs/alerts/"
    max_files: 100
```

---

## 🚨 **Alert System**

### **Alert Triggers**

1. **Error Threshold Exceeded**
   - More than 5 errors per hour
   - More than 20 errors per day
   - Any critical error type detected

2. **System Health Issues**
   - Cache corruption detected
   - Import failures
   - Performance degradation

### **Alert Types**

1. **File Alerts**
   - Saved to `logs/alerts/error_alert_YYYYMMDD_HHMMSS.txt`
   - Contains detailed error information
   - Includes system status and recommendations

2. **Log Alerts**
   - Written to system log
   - Includes error summary and action taken
   - Timestamped for tracking

### **Alert Content**

```
🚨 ERROR ALERT - Portfolio Maximizer v45
=====================================

Timestamp: 2025-10-22T20:45:00
Total Errors (24h): 8

Error Breakdown:
  - TypeError: 5
  - ValueError: 2
  - Other: 1

Recent Errors:
  - 2025-10-22T20:44:30: TypeError: DataStorage.train_validation_test_split()...
  - 2025-10-22T20:44:45: ValueError: Invalid parameter value...
  - 2025-10-22T20:45:00: ConnectionError: API timeout...

System Status: WARNING
```

---

## 📈 **Error Reporting**

### **Daily Error Report**

```bash
# Generate 7-day error report
python scripts/error_monitor.py --report-days 7

# Generate detailed report
python scripts/error_monitor.py --report-days 30 --detailed
```

### **Report Contents**

- **Error Statistics**: Total errors, error types, trends
- **Performance Metrics**: Error rates, resolution times
- **System Health**: Cache status, import validation
- **Recommendations**: Action items for improvement

### **Sample Report**

```json
{
  "period_days": 7,
  "total_errors": 15,
  "errors_by_day": {
    "2025-10-15": 3,
    "2025-10-16": 2,
    "2025-10-17": 5,
    "2025-10-18": 1,
    "2025-10-19": 2,
    "2025-10-20": 1,
    "2025-10-21": 1
  },
  "error_types": {
    "TypeError": 8,
    "ValueError": 4,
    "ConnectionError": 2,
    "Other": 1
  },
  "average_errors_per_day": 2.1,
  "most_common_error": "TypeError",
  "generated_at": "2025-10-22T20:45:00"
}
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. Method Signature Errors**
```
TypeError: DataStorage.train_validation_test_split() got an unexpected keyword argument 'test_size'
```

**Solution**:
1. Clear Python cache: `python scripts/cache_manager.py`
2. Restart Python process
3. Verify method signature: `python -c "import inspect; from etl.data_storage import DataStorage; print(inspect.signature(DataStorage.train_validation_test_split))"`

#### **2. Import Errors**
```
ModuleNotFoundError: No module named 'etl.data_storage'
```

**Solution**:
1. Check Python path: `python -c "import sys; print(sys.path)"`
2. Verify file exists: `ls -la etl/data_storage.py`
3. Clear cache and reimport: `python scripts/cache_manager.py`

#### **3. Cache Corruption**
```
ImportError: cannot import name 'DataStorage' from 'etl.data_storage'
```

**Solution**:
1. Clear all caches: `python scripts/cache_manager.py`
2. Verify file integrity: `python -m py_compile etl/data_storage.py`
3. Reinstall if necessary: `pip install -e .`

### **Debug Commands**

```bash
# Check cache health
python scripts/cache_manager.py --check-only

# Validate critical files
python scripts/cache_manager.py --validate-files

# Test method signatures
python -m pytest tests/etl/test_method_signature_validation.py -v

# Check error log
tail -f logs/errors/errors.log

# Monitor system in real-time
python scripts/error_monitor.py --watch
```

---

## 📋 **Maintenance Tasks**

### **Daily Tasks**
- [ ] Check error monitor alerts
- [ ] Review error log for new issues
- [ ] Verify system health status

### **Weekly Tasks**
- [ ] Generate error report
- [ ] Clear old cache files
- [ ] Review error trends and patterns
- [ ] Update error thresholds if needed

### **Monthly Tasks**
- [ ] Archive old error logs
- [ ] Review and update monitoring configuration
- [ ] Analyze error patterns for system improvements
- [ ] Update documentation

---

## 🎯 **Best Practices**

### **Error Prevention**
1. **Always test method signature changes** before deployment
2. **Clear Python cache** after code changes
3. **Use automated testing** for critical components
4. **Monitor error rates** continuously

### **Error Response**
1. **Acknowledge alerts** immediately
2. **Investigate root cause** thoroughly
3. **Apply fixes** systematically
4. **Document resolution** for future reference

### **System Maintenance**
1. **Regular cache cleanup** to prevent issues
2. **Monitor system performance** continuously
3. **Update monitoring thresholds** based on usage
4. **Keep documentation current**

---

## 📚 **Additional Resources**

### **Related Documentation**
- `Documentation/ERROR_FIXES_SUMMARY_2025-10-22.md`
- `Documentation/SYSTEM_STATUS_2025-10-22.md`
- `Documentation/LLM_ENHANCEMENTS_IMPLEMENTATION_SUMMARY_2025-10-22.md`

### **Configuration Files**
- `config/error_monitoring_config.yml`
- `config/pipeline_config.yml`
- `config/llm_config.yml`

### **Log Files**
- `logs/errors/errors.log` - Current error log
- `logs/alerts/` - Alert notifications
- `logs/archive/errors/` - Archived error logs
- `logs/signals/quant_validation.jsonl` - Quant validation audit log for trading/algo efficiency triage

---

**Last Updated**: October 22, 2025  
**Version**: 1.0  
**Status**: ✅ **ACTIVE**

