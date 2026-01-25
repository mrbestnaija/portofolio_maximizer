# Log Organization Summary

**Date**: 2026-01-25
**Action**: Structured logs/ directory for better organization and maintainability

---

## Changes Made

### 1. Created Subdirectory Structure

```
logs/
├── phase7.5/          # Phase 7.5: Regime Detection Integration logs
├── phase7.6/          # Phase 7.6: Threshold Tuning logs
├── phase7.7/          # Phase 7.7: Per-Regime Weight Optimization logs
├── phase7.8/          # Phase 7.8: Future (all regimes optimization)
├── archive/           # Old/large logs (compressed)
├── automation/        # Autonomous trading loop logs
├── brutal/            # Comprehensive testing suite logs
├── cron/              # Scheduled job logs
├── errors/            # Error-specific logs
├── events/            # System event logs
├── forecast_audits/   # Forecast audit trail
├── forecast_audits_cache/  # Cached audit summaries
├── hyperopt/          # Hyperparameter optimization logs
├── live_runs/         # Live trading execution logs
├── performance/       # Performance profiling logs
├── signals/           # Signal generation logs
├── stages/            # Pipeline stage-specific logs
└── warnings/          # Warning-level logs
```

### 2. Moved Phase-Specific Logs

**Phase 7.5** (5 files, 792KB total):
- `phase7.5_aapl_success.log` (82K)
- `phase7.5_msft_validation.log` (87K)
- `phase7.5_nvda_validation.log` (87K)
- `phase7.5_multi_ticker_validation.log` (279K)
- `phase7.5_multi_ticker_final.log` (105K)

**Phase 7.6** (1 file, 87KB):
- `phase7.6_aapl_tuned_thresholds.log` (87K)

**Phase 7.7** (2 files, 35KB):
- `phase7.7_weight_optimization.log` (35K)
- `phase7.7_validation_aapl.log` (181B → rerunning with correct params)

### 3. Created Documentation

**New Files**:
1. [logs/README.md](../logs/README.md) - Comprehensive log structure documentation
   - Directory structure explanation
   - Phase-specific log summaries
   - Log retention policies
   - Search patterns and examples
   - Troubleshooting guide
   - Best practices

2. [bash/organize_logs.sh](../bash/organize_logs.sh) - Automated log organization script
   - Moves phase logs to subdirectories
   - Archives large logs (>50MB, >7 days old)
   - Compresses archived logs with gzip
   - Dry-run mode for testing
   - Summary statistics

---

## Usage

### Manual Organization

```bash
# Organize logs by phase
bash bash/organize_logs.sh

# Dry run (preview changes)
bash bash/organize_logs.sh --dry-run
```

### Automated Organization (Recommended)

Add to cron or Windows Task Scheduler:

```bash
# Daily at 2 AM
0 2 * * * cd /path/to/portfolio_maximizer_v45 && bash bash/organize_logs.sh
```

---

## Benefits

1. **Easier Navigation**: Phase-specific logs grouped together
2. **Better Searchability**: Know exactly where to find logs for specific phases
3. **Disk Space Management**: Automated archiving and compression of old logs
4. **Historical Analysis**: Permanent retention of phase logs for future reference
5. **Clean Root Directory**: Only core logs (pipeline.log, pipeline_run.log) at top level
6. **Scalability**: Easy to add new phase directories as project evolves

---

## Log Retention Policy

| Category | Retention | Compression |
|----------|-----------|-------------|
| Phase logs (7.5, 7.6, 7.7, etc.) | **Permanent** | No (kept for analysis) |
| pipeline.log | Last 5 rotations | No (managed by logger) |
| pipeline_run.log | Last 30 days | Archive monthly |
| automation/ | Last 30 days | Delete older |
| live_runs/ | Last 90 days | Archive quarterly |
| archive/ | Permanent | gzip -9 |

---

## Search Patterns

### Find All Errors in Phase 7.7

```bash
grep -r "ERROR" logs/phase7.7/
```

### Count Regime Detections by Type

```bash
grep "regime=" logs/phase7.5/*.log | cut -d'=' -f2 | cut -d',' -f1 | sort | uniq -c
```

**Output**:
```
      3 CRISIS
      2 MODERATE_MIXED
      5 MODERATE_TRENDING
      1 HIGH_VOL_TRENDING
```

### Find RMSE Improvements

```bash
grep "RMSE.*->" logs/phase7.7/phase7.7_weight_optimization.log
```

**Output**:
```
RMSE: 19.2599 -> 6.7395 (+65.01%)
```

### Find Optimization Results

```bash
grep -A 5 "## MODERATE_TRENDING" logs/phase7.7/phase7.7_weight_optimization.log
```

---

## Disk Space Monitoring

### Check Total Logs Size

```bash
du -sh logs/
```

**Current**: ~18MB

### Check Phase Sizes

```bash
du -sh logs/phase7.*
```

**Output**:
```
792K    logs/phase7.5
128K    logs/phase7.6
37K     logs/phase7.7
```

### Find Large Logs

```bash
find logs/ -size +10M -ls
```

**Output**:
```
16M     logs/pipeline_run.log
```

---

## Automation Scripts

### organize_logs.sh

**Purpose**: Organize logs by phase and archive old files

**Features**:
- Automatic phase detection and sorting
- Archive logs >50MB and >7 days old
- Compress archived logs with gzip
- Dry-run mode for testing

**Usage**:
```bash
# Preview changes
bash bash/organize_logs.sh --dry-run

# Apply changes
bash bash/organize_logs.sh
```

### Future Scripts (Recommended)

**archive_large_logs.sh** (Daily):
```bash
#!/usr/bin/env bash
# Archive pipeline_run.log if > 50MB
if [[ $(stat -c%s logs/pipeline_run.log) -gt 52428800 ]]; then
    mv logs/pipeline_run.log logs/archive/pipeline_run_$(date +%Y%m%d).log
    gzip logs/archive/pipeline_run_*.log
    touch logs/pipeline_run.log
fi
```

**compress_old_logs.sh** (Weekly):
```bash
#!/usr/bin/env bash
# Compress logs older than 7 days in archive/
find logs/archive -name "*.log" -mtime +7 -exec gzip -9 {} \;
```

**clean_old_archives.sh** (Monthly):
```bash
#!/usr/bin/env bash
# Delete compressed archives older than 365 days
find logs/archive -name "*.gz" -mtime +365 -delete
```

---

## Integration with Monitoring

### Grafana Loki

Parse logs for time-series metrics:

```yaml
# promtail-config.yml
scrape_configs:
  - job_name: portfolio_maximizer
    static_configs:
      - targets:
          - localhost
        labels:
          job: portfolio_maximizer
          __path__: /path/to/logs/**/*.log
    pipeline_stages:
      - regex:
          expression: '(?P<level>ERROR|WARN|INFO) - (?P<message>.*)'
      - labels:
          level:
```

### Prometheus Alerts

Alert on error rates:

```yaml
# alertmanager.yml
groups:
  - name: portfolio_maximizer
    rules:
      - alert: HighErrorRate
        expr: rate(log_errors_total[5m]) > 0.1
        annotations:
          summary: "High error rate detected in logs"
```

---

## Best Practices

1. ✅ **Always organize after major phase completion**
2. ✅ **Run organize_logs.sh weekly** (automated via cron)
3. ✅ **Keep phase logs permanently** (critical for analysis)
4. ✅ **Compress archives immediately** (save disk space)
5. ✅ **Monitor disk usage** (alert at 80% capacity)
6. ✅ **Document key findings** in phase-specific markdown files
7. ✅ **Use grep for quick searches** (faster than opening large files)
8. ✅ **Archive pipeline_run.log monthly** (can grow to 100MB+)

---

## Quick Reference

| Task | Command |
|------|---------|
| Organize logs | `bash bash/organize_logs.sh` |
| Dry run | `bash bash/organize_logs.sh --dry-run` |
| Check disk usage | `du -sh logs/` |
| Find large logs | `find logs/ -size +10M` |
| Compress log | `gzip logs/archive/old.log` |
| Search errors | `grep -r "ERROR" logs/phase7.7/` |
| Count regimes | `grep "regime=" logs/phase7.5/*.log \| cut -d'=' -f2 \| sort \| uniq -c` |

---

## Future Enhancements

1. **Automated log rotation** for pipeline_run.log (max 100MB per file)
2. **Log aggregation** to centralized monitoring (Loki, ELK stack)
3. **Real-time alerting** on ERROR patterns (PagerDuty, Slack)
4. **Log parsing dashboard** in Grafana (RMSE trends, regime distribution)
5. **Automated cleanup** of very old archives (>1 year)

---

## Summary

The logs/ directory is now well-organized with:
- ✅ Phase-specific subdirectories for easy navigation
- ✅ Comprehensive README with search patterns and examples
- ✅ Automated organization script (organize_logs.sh)
- ✅ Clear retention policies
- ✅ Integration-ready for monitoring tools

**Total logs size**: ~18MB (manageable)
**Phase logs organized**: 7.5, 7.6, 7.7
**Documentation created**: 2 files (README.md, LOG_ORGANIZATION_SUMMARY.md)
**Scripts created**: 1 (organize_logs.sh)

---

**Date**: 2026-01-25 17:15:00 UTC
**Status**: ✅ Complete
