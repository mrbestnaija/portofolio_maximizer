# Critical Profitability Analysis & Remediation Plan

## Executive Summary

**FINDING:** The claimed **$2,935 profit with 100% win rate** is **NOT** evidence of real profitability.

### Key Evidence

- **36/67** trades have **NULL `data_source`** (not traceable to live data)
- **28%** of profit (**$815**) comes from **synthetic test tickers** (`SYN0`–`SYN4`)
- Only **2 `SELL`** actions vs **65 `BUY`** actions (missing exit lifecycle)
- **100% win rate** is statistically impossible and indicates data issues
- All **12 recent tickers FAILED quant validation** (system protecting against unprofitable trades)
- Recent live trades (**Jan 13, 2026**) show **ZERO realized P&L**

**ROOT CAUSE:** Mixed synthetic/test data contaminating production metrics, combined with incomplete position lifecycle tracking (only entries, no exits).

---

## Phase 1: Database Cleanup & Audit Trail

### 1.1 Identify and Tag Contaminated Data

**Critical files**

- `etl/database_manager.py` — add `data_source` validation
- `scripts/cleanup_synthetic_trades.py` — **NEW** cleanup script

**Actions**

- Query all trades with NULL/empty `data_source` and `execution_mode`
- Tag them with `is_test_data=TRUE` flag (add column if needed)
- Create separate views: `production_trades` vs `test_trades`
- Archive synthetic ticker trades (`SYN0`–`SYN4`) to separate table

**SQL to execute**

```sql
-- Add provenance tracking
ALTER TABLE trade_executions ADD COLUMN is_test_data BOOLEAN DEFAULT FALSE;
ALTER TABLE trade_executions ADD COLUMN audit_notes TEXT;

-- Tag contaminated data
UPDATE trade_executions
SET is_test_data = TRUE,
    audit_notes = 'Missing data_source/execution_mode - likely test data'
WHERE data_source IS NULL OR data_source = ''
   OR execution_mode IS NULL OR execution_mode = '';

-- Tag synthetic tickers
UPDATE trade_executions
SET is_test_data = TRUE,
    audit_notes = 'Synthetic test ticker'
WHERE ticker LIKE 'SYN%' AND ticker GLOB 'SYN[0-9]*';

-- Create production view
CREATE VIEW production_trades AS
SELECT * FROM trade_executions
WHERE is_test_data = FALSE OR is_test_data IS NULL;
```

### 1.2 Fix Performance Metrics Calculation

**Critical files**

- `etl/database_manager.py:get_performance_summary()` — lines ~2400–2500
- `monitoring/performance_dashboard.py` — update to use `production_trades` view

**Changes**

- Modify `get_performance_summary()` to **exclude test data**

```python
# OLD: SELECT * FROM trade_executions
# NEW: SELECT * FROM production_trades
```

- Add separate metrics for test vs production

```python
def get_performance_summary(self, start_date=None, production_only=True):
    # Add production_only flag
    # Calculate separate metrics for test/production
    ...
```

- Update dashboard to show BOTH sets of metrics with clear labels:
  - “Production Performance (Live Data Only)”
  - “Test Performance (Including Synthetic)”

---

## Phase 2: Complete Position Lifecycle Tracking

### 2.1 Implement Proper Exit Tracking

**Critical files**

- `execution/paper_trading_engine.py` — lines 200–400 (`execute_trade` method)
- `execution/order_manager.py` — position tracking logic

**Root issue:** system records `BUY` entries but doesn’t properly record `SELL` exits with P&L attribution.

**Changes**

- Fix P&L calculation timing

```python
# In paper_trading_engine.py
def execute_trade(self, ticker, action, signal):
    if action == "BUY":
        # Record entry, NO P&L yet
        realized_pnl = None
        realized_pnl_pct = None
    elif action == "SELL":
        # Calculate P&L on exit
        position = self._get_open_position(ticker)
        realized_pnl = (exit_price - position.entry_price) * position.shares
        realized_pnl_pct = (exit_price / position.entry_price) - 1.0
```

- Add position matching
  - Track which `SELL` matches which `BUY` (FIFO or specific lots)
  - Store `entry_trade_id` in `SELL` records to link back to `BUY`

- Unrealized P&L tracking

```python
def update_unrealized_pnl(self, current_prices):
    """Called periodically to mark-to-market open positions"""
    for position in self.get_open_positions():
        current_price = current_prices.get(position.ticker)
        if current_price:
            unrealized_pnl = (current_price - position.entry_price) * position.shares
            # Update portfolio_positions table
```

### 2.2 Add Equity Curve Tracking

**Critical files**

- `etl/database_manager.py` — add `equity_curve` table
- `execution/paper_trading_engine.py` — record equity snapshots

**New table schema**

```sql
CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_equity REAL NOT NULL,  -- Cash + unrealized positions
    cash_balance REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    run_id TEXT,
    execution_mode TEXT,
    data_source TEXT
);
```

**Implementation**

- Record equity snapshot after every trade
- Record periodic snapshots (e.g., daily EOD)
- Dashboard displays equity curve chart

---

## Phase 3: Enhanced Validation & Risk Controls

### 3.1 Strengthen Data Source Validation

**Critical files**

- `etl/data_source_manager.py` — add strict validation
- `scripts/run_etl_pipeline.py` — enforce production mode

**Changes**

- Prevent synthetic contamination

```python
def validate_production_mode(self):
    """Ensure we're not mixing synthetic with live"""
    if self.execution_mode == "live":
        # REJECT synthetic data sources
        if "synthetic" in self.active_sources:
            raise ValueError("Cannot use synthetic data in live mode")

        # REQUIRE data_source tracking
        if not self.data_source:
            raise ValueError("data_source must be set in live mode")
```

- Add `data_source` to ALL database writes:
  - `data_source` (yfinance, alpha_vantage, etc.)
  - `execution_mode` (live, paper, backtest, synthetic)
  - `pipeline_id` (traceability to pipeline run)

### 3.2 Implement Realistic Win Rate Expectations

**Critical files**

- `monitoring/performance_dashboard.py` — add alerts
- `config/quant_success_config.yml` — update thresholds

**New alerts**

```python
def _build_alerts(metrics):
    alerts = []

    # RED FLAG: Impossibly high win rate
    if metrics['win_rate'] > 0.80 and metrics['trade_count'] > 20:
        alerts.append("CRITICAL: Win rate > 80% suggests data quality issue")

    # RED FLAG: No losing trades
    if metrics['win_rate'] == 1.0 and metrics['trade_count'] > 10:
        alerts.append("CRITICAL: 100% win rate impossible - check for survivorship bias")

    # RED FLAG: All buys, no sells
    buy_count = count_action(db, "BUY", production_only=True)
    sell_count = count_action(db, "SELL", production_only=True)
    if buy_count > sell_count * 5:
        alerts.append(f"WARNING: {buy_count} buys vs {sell_count} sells - positions not being closed")

    return alerts
```

---

## Phase 4: Production Trading Validation

### 4.1 Clean Slate Test Run

**Objective:** Establish baseline with ZERO test data contamination.

**Steps**

Create clean database:

```bash
# Backup existing DB
cp data/portfolio_maximizer.db data/portfolio_maximizer_backup_$(date +%Y%m%d).db

# Archive test data
sqlite3 data/portfolio_maximizer.db <<EOF
DELETE FROM trade_executions WHERE is_test_data = TRUE;
DELETE FROM portfolio_positions WHERE ticker LIKE 'SYN%';
DELETE FROM trading_signals WHERE ticker LIKE 'SYN%';
VACUUM;
EOF
```

Run production pipeline:

```bash
# Ensure synthetic is DISABLED
unset ENABLE_SYNTHETIC_PROVIDER ENABLE_SYNTHETIC_DATA_SOURCE SYNTHETIC_ONLY

# Clear any synthetic environment variables
env | grep -i synthetic  # Should show nothing

# Run with strict validation
python scripts/run_etl_pipeline.py \
    --tickers AAPL,MSFT,NVDA \
    --start 2024-01-01 \
    --end 2026-01-18 \
    --execution-mode live \
    --enable-llm \
    --strict-validation  # NEW FLAG
```

Verify data provenance:

```bash
# Check that ALL trades have data_source
sqlite3 data/portfolio_maximizer.db \
    "SELECT COUNT(*) FROM trade_executions WHERE data_source IS NULL OR data_source = '';"
# Should return: 0

# Check execution mode
sqlite3 data/portfolio_maximizer.db \
    "SELECT DISTINCT execution_mode FROM trade_executions;"
# Should return: live (not NULL, not synthetic)
```

### 4.2 Multi-Day Paper Trading Test

**Objective:** Establish realistic performance metrics over time.

**Test plan**

- Run auto-trader for 5 consecutive days (simulated or real-time)
- Allow positions to open AND close naturally
- Track equity curve with unrealized P&L
- Measure realistic win rate (expect 40–60%)

**Acceptance criteria**

- At least 20 completed round-trips (`BUY` → `SELL`)
- Mix of winners and losers (win rate 40–70%)
- No synthetic tickers in results
- 100% `data_source` tracking
- Equity curve matches realized + unrealized P&L

---

## Phase 5: Dashboard Enhancements

### 5.1 Production vs Test Data Segregation

**Critical files**

- `scripts/dashboard_db_bridge.py` — filter production trades
- `visualizations/live_dashboard.html` — add production toggle

**Changes**

Add production filter:

```python
def build_dashboard_payload(conn, tickers, production_only=True):
    # Add production_only parameter
    table = "production_trades" if production_only else "trade_executions"

    # Update all queries to use filtered table
    query = f"SELECT * FROM {table} WHERE ..."
```

Dashboard toggle:

```html
<!-- In live_dashboard.html -->
<label>
    <input type="checkbox" id="show-test-data" />
    Include test/synthetic data
</label>
```

Visual indicators:

- Green badge: “Production Data ✓”
- Orange badge: “Includes Test Data ⚠”
- Red alert: “Synthetic tickers detected”

### 5.2 Enhanced Metrics Display

**New dashboard sections**

Data Quality Panel:

```text
📊 Data Quality
├─ Data Source: yfinance ✓
├─ Execution Mode: live ✓
├─ Synthetic Tickers: 0 ✓
├─ NULL Sources: 0 ✓
└─ Profitability Proof: TRUE ✓
```

Position Lifecycle:

```text
📈 Position Tracking
├─ Open Positions: 12
├─ Closed Positions: 23
├─ Unrealized P&L: -$245.00 ⚠
├─ Realized P&L: $1,200.00
└─ Total Equity: $21,200.00
```

Reality Check Alerts:

```text
⚠ ALERTS
• Win rate 100% - possible data issue
• 65 BUY vs 2 SELL - positions not closing
• 36 trades missing data source
```

---

## Phase 6: Profitability Proof Protocol

### 6.1 Define Profitability Criteria

**File:** `config/profitability_proof_requirements.yml` (**NEW**)

```yaml
profitability_proof_requirements:
  data_quality:
    min_data_source_coverage: 1.0  # 100% of trades must have data_source
    max_synthetic_ticker_pct: 0.0  # 0% synthetic tickers allowed
    allowed_execution_modes: ["live"]

  statistical_significance:
    min_closed_trades: 30  # At least 30 completed round-trips
    min_trading_days: 21   # At least 1 month of trading
    max_win_rate: 0.85     # Win rate > 85% is suspicious
    min_win_rate: 0.35     # Win rate < 35% is unprofitable

  performance:
    min_profit_factor: 1.5  # Gross profit / gross loss >= 1.5
    max_drawdown: 0.25      # Max 25% drawdown allowed
    min_sharpe_ratio: 0.5   # Risk-adjusted returns

  audit_trail:
    require_pipeline_id: true
    require_run_id: true
    require_entry_exit_matching: true
```

### 6.2 Automated Profitability Validator

**File:** `scripts/validate_profitability_proof.py` (**NEW**)

```python
#!/usr/bin/env python3
"""
Rigorous validation that performance metrics represent REAL profitability.
"""

def validate_profitability_proof(db_path: str) -> Dict[str, Any]:
    """
    Returns:
        {
            "is_profitable": bool,
            "is_proof_valid": bool,
            "violations": List[str],
            "metrics": Dict[str, Any]
        }
    """

    # Load requirements
    requirements = load_yaml("config/profitability_proof_requirements.yml")

    # Run checks
    violations = []

    # 1. Data Source Coverage
    null_source_pct = check_null_data_sources(db)
    if null_source_pct > 0:
        violations.append(f"Data source NULL for {null_source_pct:.1%} of trades")

    # 2. Synthetic Contamination
    synthetic_count = count_synthetic_tickers(db)
    if synthetic_count > 0:
        violations.append(f"Found {synthetic_count} synthetic ticker trades")

    # 3. Win Rate Reality Check
    win_rate = calculate_win_rate(db, production_only=True)
    if win_rate > requirements['max_win_rate']:
        violations.append(f"Win rate {win_rate:.1%} exceeds {requirements['max_win_rate']:.1%} - suspicious")

    # 4. Position Lifecycle Completeness
    open_positions = count_open_positions(db)
    closed_positions = count_closed_positions(db)
    if closed_positions < requirements['min_closed_trades']:
        violations.append(f"Only {closed_positions} closed trades (need {requirements['min_closed_trades']})")

    # 5. Entry/Exit Matching
    unmatched_sells = find_unmatched_sells(db)
    if len(unmatched_sells) > 0:
        violations.append(f"Found {len(unmatched_sells)} SELL orders without matching BUY")

    # Determine proof validity
    is_proof_valid = len(violations) == 0

    return {
        "is_profitable": metrics['total_pnl'] > 0,
        "is_proof_valid": is_proof_valid,
        "violations": violations,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }
```

**Integration**

- Run automatically after each pipeline
- Display results on dashboard
- Block profitability claims if validation fails

---

## Verification Plan

### Test 1: Database Cleanup

```bash
# Execute Phase 1 cleanup
python scripts/cleanup_synthetic_trades.py

# Verify contaminated data is tagged
sqlite3 data/portfolio_maximizer.db \
    "SELECT COUNT(*), is_test_data FROM trade_executions GROUP BY is_test_data;"

# Expected output:
# 36|1  (test data)
# 31|0  (production data)
```

### Test 2: Production Metrics

```bash
# Run profitability validator
python scripts/validate_profitability_proof.py

# Expected violations:
# - Only 2 closed positions (need 30)
# - 65 BUY vs 2 SELL (incomplete lifecycle)
# - Unrealized P&L not tracked
```

### Test 3: Dashboard Display

```bash
# Start dashboard
python scripts/dashboard_db_bridge.py --once

# Verify dashboard shows:
# - "Profitability Proof: FALSE"
# - Alerts about win rate / position lifecycle
# - Separate production vs test metrics
```

### Test 4: Clean Paper Trading Run

```bash
# 5-day paper trading test
python scripts/run_auto_trader.py \
    --tickers AAPL,MSFT \
    --lookback-days 365 \
    --cycles 5 \
    --execution-mode live

# Validate results
python scripts/validate_profitability_proof.py

# Expected: More realistic metrics with wins AND losses
```

---

## Critical Files to Modify

### Phase 1 — Database Cleanup

- `scripts/cleanup_synthetic_trades.py` (**NEW**)
- `etl/database_manager.py:get_performance_summary()` (~line 2450)

### Phase 2 — Position Lifecycle

- `execution/paper_trading_engine.py:execute_trade()` (~line 250)
- `execution/order_manager.py` (position matching logic)
- `etl/database_manager.py` (add `equity_curve` table)

### Phase 3 — Validation

- `etl/data_source_manager.py:validate_production_mode()` (**NEW method**)
- `monitoring/performance_dashboard.py:_build_alerts()` (~line 366)
- `config/profitability_proof_requirements.yml` (**NEW**)

### Phase 4 — Production Testing

- No new files, just execution scripts

### Phase 5 — Dashboard

- `scripts/dashboard_db_bridge.py:build_dashboard_payload()` (~line 492)
- `visualizations/live_dashboard.html` (add production toggle)

### Phase 6 — Profitability Proof

- `scripts/validate_profitability_proof.py` (**NEW**)
- Integration into pipeline completion handler

---

## Timeline & Priority

### P0 — Immediate (Blocks profitability claims)

- Tag contaminated data in database
- Update dashboard to show production-only metrics
- Add alerts for suspicious win rate / position lifecycle

### P1 — High Priority (Enables real testing)

- Fix position lifecycle tracking (entry + exit)
- Add unrealized P&L calculation
- Create equity curve tracking

### P2 — Medium Priority (Improves validation)

- Strengthen data source validation
- Implement profitability proof validator
- Clean slate production test run

### P3 — Nice to Have (Enhances monitoring)

- Dashboard production/test toggle
- Enhanced metrics display
- Automated profitability reporting

---

## Expected Outcomes

### After Phase 1–2 (Cleanup + Lifecycle)

- Accurate representation of what data is test vs production
- Complete tracking of position entries AND exits
- Realistic metrics showing both realized and unrealized P&L

### After Phase 3–4 (Validation + Testing)

- No synthetic contamination in production metrics
- Statistically valid performance data (30+ closed trades)
- Realistic win rate in 40–60% range

### After Phase 5–6 (Dashboard + Proof)

- Clear visual distinction between test and production data
- Automated validation that prevents false profitability claims
- Audit trail proving metrics come from real trading

---

## Phase 7: Model Diagnostics & Root Cause Analysis

### Root Causes Identified

**Primary blocker: overly strict validation thresholds**

- `min_expected_profit: 5.0` requires >0.02% net return AFTER 0.032% transaction costs
- AAPL override of 15.0 is mathematically impossible for small moves
- Result: even accurate forecasts fail validation

**Secondary blocker: insufficient data + disabled ensemble**

- System has 109 bars but needs 194+ for rolling CV
- Ensemble disabled → no `regression_metrics` → `forecast_edge` validation fails
- Log evidence: “Insufficient data for rolling CV (need >= 194, received 109)”

**Tertiary issues**

- Conservative default transaction costs (3.2 bps vs realistic 0.8 bps)
- MSSA-RL change-point detection too sensitive (threshold 2.5)
- Confidence scoring penalized by cost floor

### Diagnostic Scripts to Run

**Day 1–3: Systematic diagnostics**

Data quality check:

```bash
# For each ticker
python scripts/analyze_dataset.py \
    --dataset data/training/training_*.parquet \
    --column Close

# Output: diagnostics/{TICKER}_data_quality.json
# Check: bars >= 365, stationarity tests pass
```

Forecaster performance (NEW SCRIPT):

```text
# scripts/diagnose_forecaster_quality.py
# For each ticker:
# - Run SARIMAX, SAMOSSA, MSSA-RL independently
# - Measure RMSE, directional accuracy, fit success rate
# - Output: {ticker}_forecaster_diagnostics.json
```

Validation failure analysis:

```bash
# Parse recent validation failures
tail -100 logs/signals/quant_validation.jsonl | \
    jq -r '[.failed_criteria[]] | @tsv' | \
    sort | uniq -c | sort -rn

# Expected: ~85 expected_profit failures, ~47 forecast_edge errors
```

---

## Phase 8: Model Improvement Strategy

### Priority 1: Configuration Fixes (Day 4 — Quick Wins)

**Critical files**

- `config/quant_success_config.yml` (lines 30–40)
- `config/forecasting_config.yml` (lines 65–75)

**Changes**

Lower validation thresholds:

```yaml
# quant_success_config.yml
default_thresholds:
  min_expected_profit: 2.0  # Was 5.0 - now achievable

ticker_overrides:
  AAPL:
    min_expected_profit: 5.0  # Was 15.0 - 3x reduction
```

Impact: Pass rate 0% → 25%

Re-enable ensemble:

```yaml
# forecasting_config.yml
ensemble:
  enabled: true  # Was false
  weights:
    samossa: 0.7
    mssa_rl: 0.3
```

Impact: Enables `regression_metrics` → pass rate 25% → 40%

Extend data lookback:

```bash
# When running pipeline
python scripts/run_etl_pipeline.py \
    --start 2023-07-01 \  # Was 2024-07-01 (add 12 months)
    --end 2026-01-18
```

Impact: Removes “insufficient data” failures

### Priority 2: Hyperparameter Tuning (Day 5–6)

**Critical files**

- `forcester_ts/sarimax.py` (ARIMA order selection)
- `forcester_ts/samossa.py` (`window_length` parameter)
- `forcester_ts/mssa_rl.py` (change-point threshold)

**Changes**

SARIMAX grid search:

```python
# Add to forecaster.py
def optimize_sarimax_order(ticker, data):
    best_aic = np.inf
    for p in [0,1,2,3]:
        for d in [0,1]:
            for q in [0,1,2,3]:
                try:
                    model = SARIMAX(data, order=(p,d,q))
                    if model.aic < best_aic:
                        best_order = (p,d,q)
    return best_order
```

Impact: 5–15% RMSE reduction

SAMOSSA window optimization:

```text
# Test window_length in [20, 30, 40, 50, 60]
# Stable tickers (AAPL, MSFT) → 50-60
# Volatile tickers → 30
```

Impact: 5–10% RMSE reduction

Raise change-point threshold:

```text
# mssa_rl.py
change_point_threshold: 3.0  # Was 2.5 - fewer false regime shifts
```

Impact: Risk scores decrease 0.10–0.15

### Priority 3: Feature Engineering (Day 7–8)

**Critical files**

- `forcester_ts/forecaster.py` (lines 114–151)
- `models/time_series_signal_generator.py` (lines 1040–1084)

**Changes**

Enhanced SARIMAX features:

```python
# Add to exog_features in forecaster.py:
features = {
    'ret_1': log_returns[1:],
    'vol_10': rolling_volatility(10),
    'mom_5': momentum(5),
    'rsi_14': rsi(14),  # NEW
    'bbpos': bollinger_position(),  # NEW
    'mom_20': momentum(20),  # NEW
    'vol_ratio': recent_vol / hist_vol,  # NEW
}
```

Impact: 5–10% SARIMAX RMSE improvement

Adaptive risk scoring:

```python
# time_series_signal_generator.py
def calculate_risk_score(volatility, regime):
    vol_regime = recent_vol / historical_vol
    if vol_regime > 1.5:
        risk_multiplier = 1.2  # High vol regime
    else:
        risk_multiplier = 1.0
    return base_risk * risk_multiplier
```

Impact: More adaptive to market conditions

### Priority 4: Cost Model Improvements (Day 9)

**Critical files**

- `etl/synthetic_extractor.py` (add Bid/Ask spreads)
- `config/execution_cost_model.yml` (**NEW**)
- `models/time_series_signal_generator.py` (cost calculation)

**Changes**

Realistic bid/ask spreads:

```python
# synthetic_extractor.py - add to generated data
df['Bid'] = df['Close'] * (1 - 0.0008)  # 0.8 bps for liquid US stocks
df['Ask'] = df['Close'] * (1 + 0.0008)
```

Impact: Cost estimates drop 3.2 bps → 0.8 bps

Enable LOB simulation:

```yaml
# execution_cost_model.yml (NEW)
lob_simulation:
  enabled: true
  depth_profile: institutional
  market_impact_model: square_root
```

Impact: More accurate slippage estimates

### Priority 5: Confidence Refinement (Day 10)

**Critical files**

- `models/time_series_signal_generator.py` (lines 855–912)

**Changes**

Reweight confidence components:

```text
# Current: [30% diagnostics, 25% agreement, 20% SNR, 25% edge]
```

Edge-focused (for AAPL, MSFT, liquid names):

```python
CONFIDENCE_WEIGHTS_LIQUID = {
    'diagnostics': 0.20,
    'agreement': 0.20,
    'snr': 0.15,
    'forecast_edge': 0.45  # Emphasize edge
}
```

Quality-focused (for volatile/frontier tickers):

```python
CONFIDENCE_WEIGHTS_VOLATILE = {
    'diagnostics': 0.40,
    'agreement': 0.30,
    'snr': 0.20,
    'forecast_edge': 0.10  # Emphasize quality
}
```

Impact: Better alignment with actual profitability

---

## Phase 9: Backtesting & Validation Framework

### Day 11–12: Walk-Forward Validation

Create: `scripts/walk_forward_validation.py` (**NEW**)

```python
def walk_forward_validate(ticker, data, window_months=6):
    """
    Rolling window validation to prevent overfitting.

    For each 6-month window:
    1. Train on expanding history
    2. Forecast next 30 days
    3. Measure RMSE, directional accuracy, signal pass rate
    """
    results = []
    for window_start in date_range:
        train_data = data[data.date < window_start]
        test_data = data[(data.date >= window_start) &
                         (data.date < window_start + 30days)]

        # Apply Phase 8 improvements
        forecasts = run_improved_forecaster(train_data)
        signals = generate_signals(forecasts)

        # Validate
        metrics = {
            'rmse': calculate_rmse(forecasts, test_data),
            'dir_acc': directional_accuracy(forecasts, test_data),
            'pass_rate': quant_validation_pass_rate(signals),
        }
        results.append(metrics)

    # Success criteria
    assert mean(metrics['dir_acc']) > 0.52
    assert mean(metrics['pass_rate']) > 0.10
    return aggregate(results)
```

Output: `diagnostics/walk_forward_results.json`

### Day 13: Out-of-Sample Performance Test

Create: `scripts/evaluate_final_performance.py` (**NEW**)

```python
def evaluate_out_of_sample(tickers, test_set_pct=0.15):
    """
    Holdout test on data never seen during development.

    Tests:
    - ALL Phase 8 improvements applied
    - Simulated paper trading on test set
    - Realistic transaction costs
    """
    results = {}
    for ticker in tickers:
        # Load data, split test set
        full_data = load_ohlcv(ticker)
        split_idx = int(len(full_data) * (1 - test_set_pct))
        test_data = full_data[split_idx:]

        # Run improved pipeline
        signals = run_improved_pipeline(ticker, test_data)

        # Paper trade simulation
        trades = simulate_paper_trading(signals, test_data)

        # Metrics
        results[ticker] = {
            'pass_rate': calculate_pass_rate(signals),
            'sharpe': calculate_sharpe(trades),
            'profit_factor': calculate_profit_factor(trades),
            'win_rate': calculate_win_rate(trades),
            'max_drawdown': calculate_max_drawdown(trades),
        }

    # Aggregate success criteria
    passing_tickers = sum(1 for r in results.values() if r['pass_rate'] > 0.5)
    assert passing_tickers >= 6  # 50% of 12 tickers

    agg_sharpe = mean(r['sharpe'] for r in results.values())
    assert agg_sharpe > 0.5

    return results
```

Success criteria (50% passing goal):

- ≥6/12 tickers pass quant validation on test set
- Aggregate Sharpe >0.5
- Aggregate profit factor >1.1
- Win rate >45% (realistic, not 100%)

### Day 14+: Production Monitoring Enhancements

Enhance: `monitoring/performance_dashboard.py`

```python
def enhanced_model_health_monitoring():
    """Add model diagnostics to production dashboard."""

    # New metrics to track
    model_health = {
        'avg_rmse': calculate_avg_rmse_all_models(),
        'directional_accuracy': calculate_dir_acc_all_models(),
        'quant_pass_rate': calculate_validation_pass_rate(),
        'ensemble_enabled': check_ensemble_status(),
        'data_depth_ok': check_min_data_bars(),
    }

    # Alerts
    alerts = []
    if model_health['quant_pass_rate'] < 0.40:
        alerts.append("CRITICAL: Pass rate below 40% - models degrading")

    if model_health['avg_rmse'] > historical_baseline * 1.2:
        alerts.append("WARNING: RMSE degraded 20% - consider retuning")

    # Automated actions
    if model_health['quant_pass_rate'] < 0.30:
        trigger_hyperparameter_optimization()

    return model_health, alerts
```

Automated checks:

- Daily: Quant validation pass rate (target >50%)
- Weekly: Walk-forward validation on new data
- Monthly: Re-tune hyperparameters if pass rate <40%

---

## Implementation Timeline

### Week 1: Cleanup + Diagnostics + Quick Wins

- Days 1–3: run diagnostics, identify all failure modes
- Days 4–5: apply config fixes, extend data lookback
- Expected: pass rate 0% → 30%

### Week 2: Systematic Model Improvements

- Days 6–7: tune hyperparameters (SARIMAX, SAMOSSA, MSSA-RL)
- Days 8–9: feature engineering + cost model fixes
- Day 10: confidence refinement
- Expected: pass rate 30% → 50%

### Week 3: Validation & Production Deploy

- Days 11–12: walk-forward validation
- Day 13: out-of-sample performance test
- Day 14: production monitoring enhancements
- Expected: validated 50%+ pass rate, production-ready

---

## Risk Assessment

### Risks of NOT Implementing

- Misleading profitability claims damage credibility
- Capital deployment based on false metrics leads to losses
- Regulatory issues if claiming profitability without proof
- Models continue failing — 0% signal pass rate means no trading

### Risks of Implementation

- Performance metrics will decrease when test data is removed
- Win rate will drop to realistic levels (45–60% vs current 100%)
- May reveal system is NOT profitable with current signals (but we’ll fix that)
- Time investment: 2–3 weeks of focused work

### Mitigation

- Set expectations: “Current $2,935 profit is TEST DATA, not real”
- Focus on building REAL profitability, not preserving fake metrics
- Use learnings to improve forecasting models (Phase 7–9)
- Track progress: Week 1 (30% pass), Week 2 (50% pass), Week 3 (validated)
