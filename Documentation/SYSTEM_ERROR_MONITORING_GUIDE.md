# System Error Monitoring Guide - 2025-10-22

## Overview

This guide provides comprehensive information about the enhanced error monitoring and management system implemented for the Portfolio Maximizer v45 project.

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
