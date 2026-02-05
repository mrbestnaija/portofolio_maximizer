# Exit Eligibility Audit & Proof Mode Specification

**Date**: 2026-01-29
**Phase**: 7.9+ (Cross-Session Persistence)
**Status**: Diagnostic -- 0 closed trades after 34 executions across 23 tickers

---

## Step 1: Current Portfolio State Diagnosis

### Exit Field Population

From the live `portfolio_state` table (1 open position as of 2026-01-29):

| Field | AAPL (short -2) | Coverage |
|-------|-----------------|----------|
| stop_loss | 262.14 | 100% |
| target_price | 257.92 | 100% |
| max_holding_days | 30 | 100% |

**Finding: Stops, targets, and max_holding are ALL populated.** The upstream
`TimeSeriesSignalGenerator._calculate_targets()` computes them for every BUY/SELL
signal. The fields are never null for active signals.

### Distance to Exit (AAPL short @ 258.12)

| Exit Type | Level | Distance | Reachable? |
|-----------|-------|----------|------------|
| Stop loss | 262.14 | +1.56% (price must rise) | Yes -- normal daily range |
| Target | 257.92 | -0.08% (price must drop) | Trivially -- AAPL moves more than this intraday |
| Time exit | 30 bars | Depends on interval | 30 daily bars = ~6 weeks; 30 hourly bars = ~4 trading days |

**Target distance is 0.08%.** This means the forecast predicted a near-zero move
downward. The target will be hit on virtually any down-tick, triggering TAKE_PROFIT.

### Why This Hasn't Triggered Yet

Exits are only evaluated when `execute_signal()` is called for a given ticker.
This happens once per ticker per cycle. With only 1-2 sessions run so far on a
single trading day (2026-01-28), the market data hasn't refreshed with new prices
that would trigger the exit. The next run with fresh market data will likely
trigger the TAKE_PROFIT on AAPL.

---

## Step 2: Root Cause -- Zero Closed Trades

### Trade Execution History (34 trades, 23 tickers, 1 day)

```
Ticker   Direction   Trades   Pattern
AAPL     SELL        3        Stacked shorts (no reversal)
ABBV     BUY         1        Single open
AMZN     SELL        1        Single open
BA       BUY         1        Single open
BIL      SELL        5        Stacked shorts (5 separate SELL trades)
COIN     BUY         1        Single open
...
MTN      BUY         5        Stacked longs (5 separate BUY trades)
MSFT     SELL        2        Stacked shorts
```

**Critical finding: No ticker has ever received BOTH a BUY and a SELL signal.**

Every ticker got one direction and repeated it. This is why `_store_trade_execution`
never calculates realized PnL -- the `prior_shares > 0 && action == SELL` (or the
short equivalent) condition is never met because:

1. Fresh session starts with 0 shares for each ticker
2. First signal opens a position (e.g., SELL opens short)
3. Subsequent signals for the same ticker repeat the same direction (another SELL)
4. The position grows but never closes

### Five Contributing Factors

| # | Factor | Evidence | Impact |
|---|--------|----------|--------|
| 1 | **No signal reversals** | 0 tickers with both BUY+SELL | No natural closes |
| 2 | **Only 1 trading day** | All trades dated 2026-01-28 | Insufficient history for regime change |
| 3 | **Exits not yet checked** | Target is 0.08% away but no new run yet | TAKE_PROFIT hasn't fired |
| 4 | **Position stacking** | MTN: 5 BUYs, BIL: 5 SELLs, AAPL: 3 SELLs | Adding to same direction instead of reversing |
| 5 | **max_holding_days=30** | Default from `forecast_bundle.get('horizon', 30)` | TIME_EXIT takes 30 bars (~6 weeks daily, ~4 days intraday) |

---

## Step 3: Exit Eligibility Log Line

Add a diagnostic log line at the top of `execute_signal()` for every open position,
emitted once per ticker per run cycle. This makes exit proximity visible in logs.

### Format

```
[EXIT_CHECK] AAPL: shares=-2, entry=258.12, last=257.50, stop=262.14(+1.56%),
target=257.92(-0.08%), max_hold=30, bars_held=3, exit_reason=TAKE_PROFIT
```

### Implementation Location

`execution/paper_trading_engine.py`, inside `execute_signal()`, immediately after
the `_evaluate_exit_reason()` call at line 286. Log for EVERY ticker that has an
open position, regardless of whether an exit triggers.

```python
# After line 286 in execute_signal():
if current_position != 0 and current_price:
    _stop = self.portfolio.stop_losses.get(ticker)
    _target = self.portfolio.target_prices.get(ticker)
    _max_hold = self.portfolio.max_holding_days.get(ticker)
    _bars = self.portfolio.holding_bars.get(ticker, 0)
    _stop_dist = f"{(_stop - current_price) / current_price * 100:+.2f}%" if _stop else "N/A"
    _tgt_dist = f"{(_target - current_price) / current_price * 100:+.2f}%" if _target else "N/A"
    logger.info(
        "[EXIT_CHECK] %s: shares=%d, entry=%.2f, last=%.2f, "
        "stop=%s(%s), target=%s(%s), max_hold=%s, bars_held=%s, exit=%s",
        ticker, current_position, self.portfolio.entry_prices.get(ticker, 0),
        current_price, _stop, _stop_dist, _target, _tgt_dist,
        _max_hold, _bars, forced_exit_reason or "NONE",
    )
```

---

## Step 4: Proof Mode -- Guaranteed Round Trips

### Purpose

Proof Mode creates round trips (open + close) reliably so that:
- `validate_profitability_proof.py` has closed trades to evaluate
- The execution core's P&L logic is verified end-to-end
- Cross-session persistence is proven (BUY session N, SELL session N+1)

This is NOT a trading strategy. It is a testing harness.

### Specification

#### 4a. Tighter max_holding_bars

For intraday (1h) positions, set `max_holding_bars = 6` (6 hours).
For daily (1d) positions, set `max_holding_bars = 5` (1 trading week).

This ensures TIME_EXIT fires within a single automation cycle, creating a
guaranteed close for every open position.

**Implementation**: Override `forecast_horizon` in `run_auto_trader.py` when
a `--proof-mode` flag is active:

```python
@click.option("--proof-mode", is_flag=True, default=False,
              help="Tighten exits to force round trips for profitability validation")

# In signal execution, before passing to engine:
if proof_mode:
    signal["forecast_horizon"] = 6 if intraday else 5
```

#### 4b. ATR/Volatility-Based Stops and Targets

Replace the current `target_price = forecast_price` (which can be 0.08% away)
with ATR-scaled levels:

| Parameter | Formula | Rationale |
|-----------|---------|-----------|
| Stop loss | `entry +/- 1.5 * ATR(14)` | 1.5x ATR gives room for noise |
| Target | `entry +/- 2.0 * ATR(14)` | 2:1 reward-to-risk ratio |

**Current behavior** (`_calculate_targets` at line 1198-1212):
- Target = forecast price (can be arbitrarily close to current price)
- Stop = current_price * (1 +/- vol*0.5), clamped to 1.5%-5%

**Proof mode override**: When `--proof-mode`, compute ATR from market_data and
set wider, symmetric stops/targets.

```python
if proof_mode and market_data is not None and len(market_data) >= 14:
    atr = (market_data['High'] - market_data['Low']).rolling(14).mean().iloc[-1]
    if action == 'BUY':
        signal["stop_loss"] = current_price - 1.5 * atr
        signal["target_price"] = current_price + 2.0 * atr
    else:
        signal["stop_loss"] = current_price + 1.5 * atr
        signal["target_price"] = current_price - 2.0 * atr
```

#### 4c. Flatten Before Reverse

Enforce that a position must pass through zero before switching sides.
Currently `_update_portfolio` allows flips (long -> short in one trade via
oversized SELL). In proof mode, cap trade size to close the existing position
first.

**Current behavior** (line 626, 634): Position sizing already caps at current
position size for closing trades. The issue is that the system opens NEW
positions on the same ticker before closing old ones (stacking).

**Proof mode rule**: If a signal's action conflicts with the existing position
(BUY when short, SELL when long), treat it as a close-only trade. Do not open
a new position in the opposite direction until the next cycle.

```python
# In execute_signal(), after position sizing:
if proof_mode and current_position != 0:
    if (action == "BUY" and current_position < 0) or \
       (action == "SELL" and current_position > 0):
        # Close only -- cap at existing position size
        position_size = min(position_size, abs(current_position))
```

---

## Step 5: Verification Checklist

### Pre-requisites

- [ ] Portfolio state has open positions with stops/targets/max_hold set
- [ ] At least 2 sessions have run with `--resume`
- [ ] Market data has refreshed (different trading day or intraday bar)

### Proof Mode Validation

Run with `--proof-mode` on 3-5 tickers over 2-3 sessions:

```bash
# Session 1: Open positions
RISK_MODE=research_production ./simpleTrader_env/Scripts/python.exe \
  scripts/run_auto_trader.py --tickers AAPL,MSFT,NVDA \
  --cycles 1 --resume --proof-mode

# Session 2: Positions should close via TIME_EXIT (max_hold=5/6)
RISK_MODE=research_production ./simpleTrader_env/Scripts/python.exe \
  scripts/run_auto_trader.py --tickers AAPL,MSFT,NVDA \
  --cycles 3 --resume --proof-mode

# Verify:
python -c "
import sqlite3
conn = sqlite3.connect('data/portfolio_maximizer.db')
closed = conn.execute('''
    SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NOT NULL
''').fetchone()[0]
print(f'Closed trades with realized PnL: {closed}')
conn.close()
"
```

### Success Criteria

| Metric | Threshold | Why |
|--------|-----------|-----|
| Closed trades | > 0 | At least some positions closed |
| realized_pnl populated | 100% of closing trades | P&L logic works |
| holding_period_days populated | 100% of closing trades | Holding period tracked |
| Cross-session close | >= 1 trade | BUY in session N, SELL in session N+1 |

### What to Check in Logs

```
[EXIT_CHECK] AAPL: shares=-2, entry=258.12, last=255.00, ...exit=TAKE_PROFIT
```

If `exit=NONE` for all tickers across multiple runs, investigate:
1. Are prices actually changing between runs? (Cached data?)
2. Is `_evaluate_exit_reason` being reached? (Add debug log)
3. Are forced exits being blocked by signal validation? (Check line 375)

---

## Signal Flow Summary

```
TimeSeriesSignalGenerator._calculate_targets()
    target_price = forecast_price        <-- can be very close to current (0.08%)
    stop_loss = price * (1 +/- vol*0.5)  <-- 1.5% to 5%
         |
    TimeSeriesSignal(forecast_horizon=30) <-- default 30 bars
         |
    SignalRouter._signal_to_dict()        <-- preserves all fields
         |
    PaperTradingEngine.execute_signal()
         |
         +-- _evaluate_exit_reason()      <-- checks stop/target/time
         |       stop_loss  -> STOP_LOSS
         |       target     -> TAKE_PROFIT
         |       bars_held  -> TIME_EXIT
         |
         +-- Trade(stop_loss, target_price, forecast_horizon)
         |
         +-- _update_portfolio()
         |       portfolio.stop_losses[ticker] = trade.stop_loss
         |       portfolio.target_prices[ticker] = trade.target_price
         |       portfolio.max_holding_days[ticker] = trade.forecast_horizon
         |
         +-- _store_trade_execution()     <-- realized P&L calculated here
                  SELL + prior_shares > 0  -> close long
                  BUY  + prior_shares < 0  -> cover short
```

---

## Conclusion

The execution core is sound. The zero-closed-trade situation stems from:

1. **Only 1 trading day of data** -- no opportunity for signal reversals
2. **No signal reversals** -- every ticker kept the same direction across all cycles
3. **Exits not yet triggered** -- targets are very close (0.08% for AAPL) but
   the system hasn't run again with fresh prices
4. **max_holding_days=30** -- TIME_EXIT needs 30 bars, which is 6 weeks of daily
   runs or ~4 days of hourly runs

**Immediate fix**: Run the daily trader script again. With fresh market data,
the AAPL TAKE_PROFIT (target 257.92, only 0.08% below entry) should trigger
on the next run if AAPL drops at all.

**Structural fix**: Proof Mode (`--proof-mode`) is now implemented with:
- `max_holding_bars = 5` (daily) / `6` (intraday) to force TIME_EXIT
- ATR-based stops/targets instead of forecast-price targets
- Flatten-before-reverse to prevent position stacking

This creates guaranteed round trips for profitability validation without
changing the production trading strategy.

---

## Implementation Status (2026-01-29)

All features implemented and verified:

| Feature | File | Status |
|---------|------|--------|
| `[EXIT_CHECK]` diagnostic log | `execution/paper_trading_engine.py` | Implemented |
| `--proof-mode` CLI flag | `scripts/run_auto_trader.py` | Implemented |
| ATR-based stops/targets | `scripts/run_auto_trader.py` (_execute_signal) | Implemented |
| Flatten-before-reverse | `execution/paper_trading_engine.py` (execute_signal) | Implemented |
| holding_bars persistence | `etl/database_manager.py` + migration | Implemented |
| entry_bar/last_bar timestamps | `etl/database_manager.py` + migration | Implemented |

### Verification Results

**First closed trade achieved:**
```
AAPL BUY 1@256.59 PnL=-0.2975 (-0.12%) held=0d
[EXIT_CHECK] AAPL: shares=-1, entry=256.29, last=256.44, stop=263.81(+2.87%),
  target=246.62(-3.83%), max_hold=5, bars_held=5, exit=TIME_EXIT
```

- TIME_EXIT fired correctly at `bars_held >= max_holding_days` (5 >= 5)
- Forced exit converted SELL signal to BUY (correct for short cover)
- Realized P&L calculated: `(entry - cover) * shares = (256.29 - 256.59) * 1 = -0.30`
- Portfolio state saved with 0 positions (fully flat)

### Bug Found and Fixed During Implementation

**sqlite3.Row `.get()` access**: The `load_portfolio_state` method used `.get()`
for new columns, but `sqlite3.Row` objects don't support `.get()` despite having
`keys()`. Fixed by using `try/except KeyError` bracket access.
