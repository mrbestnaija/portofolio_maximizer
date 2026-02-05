# Quick Reference - Optimized Portfolio Maximizer

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

**Version**: 2.0
**Date**: 2025-10-14
**Primary Model**: **Qwen 14B Chat (q4_K_M)**
**Focus**: **Quantifiable Profit on Trade**

---

## 🚀 **Quick Start**

### **1. Run Optimized Pipeline**
```bash
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45
source simpleTrader_env/bin/activate

# Ensure Ollama is running
ollama serve

# Run pipeline with Qwen 14B model (automatic)
python scripts/run_etl_pipeline.py \
  --tickers AAPL,MSFT,GOOGL \
  --include-frontier-tickers \
  --enable-llm \
  --verbose

# All data automatically saved to database
```

`--include-frontier-tickers` automatically merges the Nigeria → Bulgaria ticker atlas from
`etl/frontier_markets.py`, ensuring every “optimized pipeline” invocation exercises the guided
frontier-market liquidity scenarios documented in `Documentation/arch_tree.md`.
SQLite self-heal: as of 2025‑11‑18 `etl/database_manager.py` automatically backs up and rebuilds the
database if “database disk image is malformed” appears, so brutal/test runs keep saving forecasts
without manual intervention.

### **2. Generate Profit Report**
```bash
# Text report
python scripts/generate_llm_report.py --format text

# JSON report
python scripts/generate_llm_report.py --format json --output metrics.json

# HTML dashboard
python scripts/generate_llm_report.py --format html --output dashboard.html
```

### **3. Export Forecast Audit (Interpretable AI)**
```bash
# Save instrumentation report for the latest Time Series run
python - <<'PY'
from forcester_ts.forecaster import TimeSeriesForecaster
from forcester_ts.forecaster import TimeSeriesForecasterConfig
from pathlib import Path

cfg = TimeSeriesForecasterConfig(ensemble_kwargs={"audit_log_dir": "logs/forecast_audits"})

> For canonical interpretation of ensemble status (per-forecast policy label vs aggregate audit gate) and current evidence, see `ENSEMBLE_MODEL_STATUS.md`.
forecaster = TimeSeriesForecaster(config=cfg)
# ... fit + forecast with your series, then:
forecaster.save_audit_report(Path("logs/forecast_audits/manual_audit.json"))
PY
```

### **4. Check Database**
```bash
# Open database
sqlite3 data/portfolio_maximizer.db

# View total profit
SELECT SUM(realized_pnl) as total_profit FROM trade_executions;

# View latest signals
SELECT * FROM llm_signals ORDER BY signal_date DESC LIMIT 10;

# Performance summary
.mode column
.headers on
SELECT * FROM performance_metrics ORDER BY metric_date DESC LIMIT 1;
```

---

## 📊 **Key Quantifiable Metrics**

### **Primary Success Criterion: PROFIT**
```sql
-- Total Profit (USD)
SELECT SUM(realized_pnl) FROM trade_executions;

-- Win Rate (%)
SELECT
    COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) * 100.0 / COUNT(*)
FROM trade_executions;

-- Profit Factor
SELECT
    SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl END) /
    ABS(SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl END))
FROM trade_executions;
```

### **Minimum Viable System (MVS) Criteria**
```python
# Quick check
from etl.database_manager import DatabaseManager
db = DatabaseManager()
perf = db.get_performance_summary()

mvs_passed = (
    perf['total_profit'] > 0 and          # ✓ Profitable
    perf['win_rate'] > 0.45 and            # ✓ Win rate > 45%
    perf['profit_factor'] > 1.0 and        # ✓ Profit factor > 1.0
    perf['total_trades'] >= 30             # ✓ Minimum 30 trades
)

print(f"MVS Status: {'✅ PASS' if mvs_passed else '❌ FAIL'}")
```

---

## 🔧 **Configuration Overview**

### **Active LLM Model**
- **Primary**: `qwen:14b-chat-q4_K_M` (9.4GB RAM)
- **Fallback 1**: `deepseek-coder:6.7b-instruct-q4_K_M` (4.1GB RAM)
- **Fallback 2**: `codellama:13b-instruct-q4_K_M` (7.9GB RAM)

### **Database Schema**
```
portfolio_maximizer.db
├── ohlcv_data          (Price/volume data)
├── llm_analyses        (Market analysis results)
├── llm_signals         (Trading signals)
├── llm_risks           (Risk assessments)
├── portfolio_positions (Current positions)
├── trade_executions    (P&L tracking)
└── performance_metrics (Quantitative metrics)
```

### **Key Files**
- **Pipeline**: `scripts/run_etl_pipeline.py`
- **Reporting**: `scripts/generate_llm_report.py`
- **Database**: `etl/database_manager.py`
- **LLM Config**: `config/llm_config.yml`
- **Success Criteria**: `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md`

---

## 📈 **Success Metrics Tiers**

### **Tier 1: Profitability** (Primary)
| Metric | Target | Formula |
|--------|--------|---------|
| Total Profit | > $0 | Σ(Sell - Buy - Fees) |
| Win Rate | > 50% | Wins / Total × 100% |
| Profit Factor | > 1.0 | Gross Profit / Gross Loss |

### **Tier 2: Risk-Adjusted**
| Metric | Target | Formula |
|--------|--------|---------|
| Sharpe Ratio | > 0.5 | (Return - RF) / Volatility |
| Max Drawdown | < 20% | max(Peak - Trough) |

### **Tier 3: Alpha Generation**
| Metric | Target | Formula |
|--------|--------|---------|
| Alpha | > 2% | Portfolio Return - Benchmark |
| Information Ratio | > 0.5 | Alpha / Tracking Error |

---

## 🎯 **Success Validation**

### **MVS (Minimum Viable System)**
```
Must pass ALL:
1. ✓ Total Profit > $0
2. ✓ Win Rate > 45%
3. ✓ Profit Factor > 1.0
4. ✓ Validation Period ≥ 30 days
```

### **PRS (Production Ready System)**
```
Must pass ALL:
1. ✓ Total Profit > $1,000 in 30 days
2. ✓ Win Rate > 50%
3. ✓ Profit Factor > 1.5
4. ✓ Sharpe Ratio > 0.5
5. ✓ Annual Return > 10%
6. ✓ Max Drawdown < 20%
7. ✓ Alpha > 2%
8. ✓ Validation ≥ 60 days
9. ✓ Capital ≥ $25,000
```

---

## 🗄️ **Database Queries**

### **Performance Summary**
```sql
-- Overall stats
SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(realized_pnl) as total_profit,
    AVG(realized_pnl) as avg_profit_per_trade,
    MAX(realized_pnl) as largest_win,
    MIN(realized_pnl) as largest_loss
FROM trade_executions;
```

### **LLM Performance**
```sql
-- Latency by task
SELECT
    'Analysis' as task,
    AVG(latency_seconds) as avg_latency,
    MAX(latency_seconds) as max_latency
FROM llm_analyses
UNION ALL
SELECT
    'Signals',
    AVG(latency_seconds),
    MAX(latency_seconds)
FROM llm_signals
UNION ALL
SELECT
    'Risk',
    AVG(latency_seconds),
    MAX(latency_seconds)
FROM llm_risks;
```

### **Signal Accuracy**
```sql
-- Validated signals
SELECT
    action,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    AVG(backtest_annual_return) as avg_backtest_return
FROM llm_signals
WHERE validation_status = 'validated'
GROUP BY action;
```

---

## 📊 **Reporting Commands**

### **Standard Reports**
```bash
# Profit/Loss summary
python scripts/generate_llm_report.py --format text | grep "Total Profit"

# Full detailed report
python scripts/generate_llm_report.py --format text --output report.txt

# JSON for automation
python scripts/generate_llm_report.py --format json > metrics.json

# Get win rate
cat metrics.json | jq '.reports.profit_loss.metrics.win_rate_pct'
```

### **Backtesting**
```bash
# Backtest signals
python scripts/backtest_llm_signals.py \
  --signals data/llm_signals.json \
  --data data/training/*.parquet \
  --capital 100000 \
  --verbose

# Track signal performance
python scripts/track_llm_signals.py --report
```

---

## 🔍 **Troubleshooting**

### **Issue: LLM Not Responding**
```bash
# Check Ollama
ollama list

# Restart Ollama
pkill ollama
ollama serve

# Test model
ollama run qwen:14b-chat-q4_K_M "Hello"
```

### **Issue: Database Locked**
```bash
# Close all connections
ps aux | grep portfolio_maximizer.db
# Kill processes if needed

# Verify database integrity
sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"
```

### **Issue: Low Profit**
```python
# Review signals
from etl.database_manager import DatabaseManager
db = DatabaseManager()
signals = db.get_latest_signals(limit=20)

for sig in signals:
    print(f"{sig['ticker']}: {sig['action']} (conf: {sig['confidence']:.0%})")
```

---

## 📁 **File Structure**

```
portfolio_maximizer_v45/
├── config/
│   ├── llm_config.yml              (Qwen 14B primary)
│   └── pipeline_config.yml         (Success criteria)
│
├── etl/
│   └── database_manager.py         (700+ lines, 7 tables)
│
├── scripts/
│   ├── run_etl_pipeline.py         (Main pipeline)
│   ├── generate_llm_report.py      (Profit reporting)
│   ├── backtest_llm_signals.py     (Validation)
│   └── track_llm_signals.py        (Performance tracking)
│
├── Documentation/
│   ├── QUANTIFIABLE_SUCCESS_CRITERIA.md  (6-tier metrics)
│   ├── OPTIMIZATION_SUMMARY.md           (This optimization)
│   └── QUICK_REFERENCE_OPTIMIZED_SYSTEM.md  (This file)
│
└── data/
    ├── portfolio_maximizer.db      (SQLite database)
    └── llm_signal_tracking.json    (Signal registry)
```

---

## ⚡ **Performance Expectations**

### **LLM Latency (Qwen 14B)**
- Market Analysis: 10-40s per ticker
- Signal Generation: 8-30s per ticker
- Risk Assessment: 10-35s per ticker
- **Total**: ~30-100s per ticker

### **Database Performance**
- Save OHLCV: ~0.01s per row
- Save LLM Analysis: ~0.001s
- Query Performance: ~0.1s for summaries
- **Storage**: ~1MB per 1000 rows

---

## 🎯 **Daily Workflow**

### **Morning**
```bash
# 1. Run pipeline
python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,GOOGL --include-frontier-tickers --enable-llm

# 2. Check profit
sqlite3 data/portfolio_maximizer.db \
  "SELECT SUM(realized_pnl) FROM trade_executions;"

# 3. View today's signals
sqlite3 data/portfolio_maximizer.db \
  "SELECT * FROM llm_signals WHERE signal_date = date('now');"
```

### **Weekly**
```bash
# 1. Generate comprehensive report
python scripts/generate_llm_report.py --format text > weekly_report.txt

# 2. Check success criteria
cat weekly_report.txt | grep -A10 "OVERALL SUCCESS CRITERIA"

# 3. Backup database
cp data/portfolio_maximizer.db backups/db_$(date +%Y%m%d).db
```

### **Monthly**
```bash
# 1. Performance review
python scripts/generate_llm_report.py --format html --output monthly_dashboard.html

# 2. Validate PRS criteria
# (Check if ready for live trading)

# 3. Export metrics for analysis
python scripts/generate_llm_report.py --format json > monthly_metrics.json
```

---

## 🔒 **Safety Features**

### **Built-in Protections**
- ✅ **Advisory Only**: LLM signals require validation
- ✅ **30-Day Minimum**: Must backtest before trading
- ✅ **Profit Validation**: Must show actual profit
- ✅ **Risk Limits**: Max 25% per position
- ✅ **Database Persistence**: All data saved

### **Production Checklist**
- [ ] MVS criteria passed (6/6)
- [ ] 30+ days backtesting
- [ ] Total profit > $1,000
- [ ] Win rate > 50%
- [ ] Database populated
- [ ] Reports generated
- [ ] Risk management plan

---

## 📚 **Documentation Links**

- **Quantifiable Criteria**: `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md`
- **Optimization Summary**: `Documentation/OPTIMIZATION_SUMMARY.md`
- **LLM Integration**: `Documentation/LLM_INTEGRATION.md`
- **Database Schema**: `etl/database_manager.py` (docstrings)
- **Success Formulas**: See QUANTIFIABLE_SUCCESS_CRITERIA.md

---

## 💡 **Pro Tips**

1. **Monitor Latency**: Qwen 14B is slower but more accurate
2. **Check Profit Daily**: Primary success metric
3. **Validate Signals**: Always backtest before trading
4. **Use Database**: All metrics in SQL for analysis
5. **Track Win Rate**: Key indicator of system health
6. **Review Risks**: Don't ignore risk assessments
7. **Backup Often**: Database contains all history

---

## 🎓 **Success Formula**

```
PROFIT = (Qwen 14B Reasoning) ×
         (Persistent Data) ×
         (Quantifiable Criteria) ×
         (Standardized Reporting) ×
         (Continuous Validation)

MEASURE EVERYTHING.
VALIDATE EVERYTHING.
PROFIT IS THE ONLY TRUE SUCCESS METRIC.
```

---

**STATUS**: ✅ **OPTIMIZED AND PRODUCTION READY**
**MODEL**: **Qwen 14B Chat (q4_K_M)**
**DATABASE**: **SQLite with 7 tables**
**FOCUS**: **QUANTIFIABLE PROFIT ON TRADE**
