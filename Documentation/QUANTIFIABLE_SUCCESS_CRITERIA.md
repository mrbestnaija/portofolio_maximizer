# Quantifiable Success Criteria for Portfolio Maximizer

**Version**: 1.0  
**Date**: 2025-10-14  
**Status**: Production Standard  
**Primary Goal**: **MEASURABLE PROFIT ON TRADE**

> **Delta (2026-04-08):** Canonical objective policy now lives in
> `Documentation/REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md`.
> **Barbell asymmetry is the primary economic objective. The system optimizes for asymmetric upside with bounded downside, not for symmetric textbook efficiency metrics.**
> For time-series signal evaluation, asymmetric payoff quality is primary; Sharpe, Sortino, and win rate are secondary diagnostics unless a local exception is documented explicitly.

> **Telemetry update (Nov 16, 2025):** Time-series forecaster instrumentation now emits per-model audit reports (dataset summary + RMSE/sMAPE/tracking error), so every KPI listed here can be traced back to concrete runs (`logs/forecast_audits/*.json` when enabled).

---

## 🎯 **PRIMARY SUCCESS CRITERION: PROFIT**

### **Definition**
Success is measured by **actual realized profit** from executed trades, not predictive metrics or theoretical returns.

For the time-series forecasting and signal stack, realized profit must be interpreted through the
barbell-asymmetry policy above. A lower hit rate can still be healthy when convex upside, payoff
asymmetry, and downside containment improve total economic utility.

### **Formula**
```
Total Profit = Σ(Sell Price - Buy Price - Commission) × Shares

Profit per Trade = Total Profit / Number of Trades

ROI = (Total Profit / Initial Capital) × 100%

Annual ROI = ((1 + ROI)^(365 / Days)) - 1) × 100%
```

---

## 📊 **TIER 1: PROFITABILITY METRICS** (Primary KPIs)

### **1.1 Total Profit (USD)**
- **Definition**: Net profit/loss from all executed trades
- **Target**: > $0 (profitable)
- **Minimum for Production**: > $1,000 in 30 days
- **Ideal**: > 10% of initial capital per quarter

**Measurement**:
```sql
SELECT SUM(realized_pnl) as total_profit 
FROM trade_executions 
WHERE realized_pnl IS NOT NULL
```

### **1.2 Profit per Trade (USD)**
- **Definition**: Average profit/loss per trade execution
- **Target**: > $0
- **Minimum**: > $50 per trade
- **Ideal**: > $200 per trade

**Calculation**:
```python
profit_per_trade = total_profit / num_trades
```

### **1.3 Win Rate (%)**
- **Definition**: Percentage of profitable trades
- **Target**: > 50%
- **Minimum**: > 45%
- **Ideal**: > 60%

**Calculation**:
```python
win_rate = (winning_trades / total_trades) × 100%
```

### **1.4 Profit Factor**
- **Definition**: Gross profit ÷ Gross loss
- **Target**: > 1.0
- **Minimum**: > 1.2
- **Ideal**: > 2.0

**Calculation**:
```python
profit_factor = sum(winning_trades) / abs(sum(losing_trades))
```

**Interpretation**:
- < 1.0: Losing system
- 1.0-1.5: Marginal
- 1.5-2.0: Good
- > 2.0: Excellent

> **Barbell / Risk-Bucket Note**  
> For Taleb-style barbell strategies, the **safe leg** must meet or exceed these PF/WR thresholds at the portfolio level.  
> The **risk leg** (e.g. long OTM options, synthetic convex exposures) is allowed to have lower win rate and more volatile trade-level PnL, but:
> - Its capital share is bounded by barbell guardrails (`config/barbell.yml`, `config/options_config.yml`), and  
> - It must improve or at least not degrade portfolio-level tail behaviour (drawdown, tail ratio, CVaR) once barbell constraints are applied.  
> In other words, risk-taking in the barbell risk bucket is intentional and acceptable **only** when sized according to the barbell allocation and evaluated on **total portfolio** PF/WR and tail metrics, not per-trade Sharpe alone.

### **1.5 Win/Loss Ratio**
- **Definition**: Average win ÷ Average loss
- **Target**: > 1.0
- **Minimum**: > 0.8 (with high win rate)
- **Ideal**: > 1.5

**Calculation**:
```python
win_loss_ratio = avg_winning_trade / abs(avg_losing_trade)
```

---

## 📈 **TIER 2: RISK-ADJUSTED METRICS**

### **2.1 Sharpe Ratio**
- **Definition**: (Return - Risk-free rate) / Volatility
- **Target**: > 0.0
- **Minimum**: > 0.5
- **Ideal**: > 1.0

**Formula**:
```
Sharpe = (R_p - R_f) / σ_p

Where:
R_p = Portfolio return
R_f = Risk-free rate (assume 0.04 = 4%)
σ_p = Portfolio volatility (std dev of returns)
```

**Interpretation**:
- < 0: Poor (losses or excessive risk)
- 0-1: Acceptable
- 1-2: Good
- > 2: Excellent

### **2.2 Sortino Ratio**
- **Definition**: Return / Downside deviation
- **Target**: > Sharpe ratio
- **Minimum**: > 0.7
- **Ideal**: > 1.5

**Formula**:
```
Sortino = (R_p - R_f) / σ_down

Where σ_down = std dev of negative returns only
```

### **2.3 Maximum Drawdown (%)**
- **Definition**: Largest peak-to-trough decline
- **Target**: < 20%
- **Maximum Acceptable**: < 30%
- **Ideal**: < 15%

**Formula**:
```
MDD = max(1 - P_t / max(P_{0:t})) × 100%

Where P_t = portfolio value at time t
```

### **2.4 Calmar Ratio**
- **Definition**: Annual return / Max drawdown
- **Target**: > 0.5
- **Minimum**: > 0.3
- **Ideal**: > 1.0

**Formula**:
```
Calmar = Annual Return / |Max Drawdown|
```

---

## 🎓 **TIER 3: ALPHA GENERATION METRICS**

### **3.1 Alpha (Excess Return)**
- **Definition**: Return vs benchmark (S&P 500)
- **Target**: > 0%
- **Minimum**: > 2% annually
- **Ideal**: > 5% annually

**Formula**:
```
Alpha = Portfolio Return - Benchmark Return
```

**Per AGENT_INSTRUCTION.md**:
- Must beat buy-and-hold baseline
- Required for production deployment

### **3.2 Beta (Market Correlation)**
- **Definition**: Correlation with market movements
- **Target**: 0.5 - 1.5 (some correlation, not excessive)
- **Ideal**: 0.7 - 1.0

**Formula**:
```
Beta = Cov(R_p, R_m) / Var(R_m)

Where:
R_p = Portfolio returns
R_m = Market returns (S&P 500)
```

### **3.3 Information Ratio**
- **Definition**: Alpha / Tracking error
- **Target**: > 0.5
- **Ideal**: > 1.0

**Formula**:
```
IR = Alpha / σ_alpha

Where σ_alpha = std dev of excess returns
```

---

## 🔍 **TIER 4: SIGNAL QUALITY METRICS**

### **4.1 Signal Accuracy (%)**
- **Definition**: % of signals with correct direction
- **Target**: > 50%
- **Minimum**: > 55%
- **Ideal**: > 65%

**Measurement**:
```python
# Compare predicted direction vs actual direction
correct_signals = sum(predicted_direction == actual_direction)
accuracy = correct_signals / total_signals
```

### **4.2 Signal Confidence Calibration**
- **Definition**: Correlation between confidence and accuracy
- **Target**: Positive correlation
- **Ideal**: r > 0.5

**Measurement**:
```python
correlation = np.corrcoef(confidence_scores, accuracy_scores)[0,1]
```

### **4.3 LLM Latency (Seconds)**
- **Definition**: Time to generate analysis/signal
- **Target**: < 60s per ticker
- **Maximum**: < 180s per ticker

**Measurement**:
```sql
SELECT AVG(latency_seconds) FROM llm_analyses
```

### **4.4 False Positive Rate**
- **Definition**: % of BUY signals that result in losses
- **Target**: < 50%
- **Maximum**: < 60%

---

## ⚙️ **TIER 5: OPERATIONAL METRICS**

### **5.1 Total Trades**
- **Definition**: Number of executed trades
- **Minimum for Validation**: 30 trades
- **Ideal for Statistics**: 100+ trades

### **5.2 Average Holding Period (Days)**
- **Definition**: Average days per position
- **Target**: Context-dependent
- **Day Trading**: < 1 day
- **Swing Trading**: 3-10 days
- **Position Trading**: 10-90 days

### **5.3 Commission Impact (%)**
- **Definition**: Total fees as % of total return
- **Target**: < 2% of returns
- **Maximum**: < 5% of returns

**Calculation**:
```python
commission_impact = total_commissions / abs(total_returns)
```

### **5.4 Capital Utilization (%)**
- **Definition**: % of capital actively invested
- **Target**: 70-90%
- **Warning**: < 50% or > 95%

---

## ✅ **COMPOSITE SUCCESS CRITERIA**

### **Minimum Viable System (MVS)**
Must pass ALL of these to deploy:

1. ✅ **Profitable**: Total profit > $0
2. ✅ **Win Rate**: > 45%
3. ✅ **Profit Factor**: > 1.0
4. ✅ **Validation Period**: 30+ days
5. ✅ **Minimum Trades**: 30+ executions
6. ✅ **Positive Alpha**: Beats buy-and-hold

**SQL Query for MVS Check**:
```sql
SELECT 
    CASE 
        WHEN total_profit > 0 
         AND win_rate > 0.45 
         AND profit_factor > 1.0 
         AND days_active >= 30 
         AND total_trades >= 30 
         AND alpha > 0 
    THEN 'PASS' 
    ELSE 'FAIL' 
    END as mvs_status
FROM performance_metrics
```

### **Production Ready System (PRS)**
Must pass ALL of these for live trading:

1. ✅ **Profitable**: Total profit > $1,000 in 30 days
2. ✅ **Win Rate**: > 50%
3. ✅ **Profit Factor**: > 1.5
4. ✅ **Sharpe Ratio**: > 0.5
5. ✅ **Annual Return**: > 10%
6. ✅ **Max Drawdown**: < 20%
7. ✅ **Alpha**: > 2% annually
8. ✅ **Validation**: 60+ days, 50+ trades
9. ✅ **Capital**: $25,000+ available

### **Elite Performance System (EPS)**
Aspirational targets:

1. 🌟 **Annual Return**: > 20%
2. 🌟 **Sharpe Ratio**: > 1.5
3. 🌟 **Win Rate**: > 60%
4. 🌟 **Profit Factor**: > 2.5
5. 🌟 **Alpha**: > 10% annually
6. 🌟 **Max Drawdown**: < 10%

---

## 📋 **Validation Checklist**

### **Before Paper Trading**
- [ ] MVS criteria passed
- [ ] 30+ days backtest
- [ ] 30+ trades executed
- [ ] All metrics calculated
- [ ] Database populated
- [ ] Reports generated

### **Before Live Trading**
- [ ] PRS criteria passed
- [ ] 60+ days paper trading
- [ ] $25,000+ capital
- [ ] Risk management plan
- [ ] Stop-loss protocols
- [ ] Position sizing rules

---

## 📊 **Reporting Requirements**

### **Daily Reports**
- Total P&L (running total)
- Open positions
- Today's trades
- Current drawdown

### **Weekly Reports**
- Weekly P&L
- Win rate trend
- Top performers
- Risk metrics

### **Monthly Reports**
- All Tier 1-5 metrics
- Comparative analysis vs S&P 500
- Signal accuracy breakdown
- Model performance review

---

## 🎯 **Success Formula Summary**

```
SUCCESS = (Profit > 0) 
        AND (Win Rate > 50%) 
        AND (Profit Factor > 1.5)
        AND (Sharpe > 0.5)
        AND (Annual Return > 10%)
        AND (Alpha > 0)
        AND (Max Drawdown < 20%)
        AND (Validation Days >= 30)
```

---

## 🔄 **Continuous Improvement Metrics**

Track these over time to measure improvement:

1. **Month-over-month profit growth**
2. **Sharpe ratio trend**
3. **Win rate stability**
4. **Drawdown frequency reduction**
5. **Signal accuracy improvement**

---

## 📖 **References**

- AGENT_INSTRUCTION.md: Validation requirements
- pipeline_config.yml: Validation thresholds
- database_manager.py: Metric calculations
- generate_llm_report.py: Reporting implementation

---

**Status**: ✅ **PRODUCTION STANDARD**  
**Primary KPI**: **TOTAL PROFIT (USD)**  
**Minimum for Success**: **MVS Criteria (6/6)**  
**Goal for Live Trading**: **PRS Criteria (9/9)**


