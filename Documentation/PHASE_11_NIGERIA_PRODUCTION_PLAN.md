# Phase 11 — Nigeria Production Path
## Status: PLANNING (implementation blocked until THIN_LINKAGE >= 10)
## Created: 2026-04-06
## Author: Adversarial review + critical corrections

---

## Hard Prerequisite — THIN_LINKAGE Gate

**The warmup exemption expires 2026-04-15 (9 days from creation of this document).**

Current: `matched = 1/309 (0.32%)`. Post-warmup requirement: `matched >= 10`.

Every session until April 15 is dedicated to evidence accumulation:

```bash
# Run after each session
python scripts/production_audit_gate.py
python scripts/outcome_linkage_attribution_report.py
```

**No Phase 11 code is wired until `matched >= 10`.**

---

## What Phase 11 Is

Nigeria is the deployment jurisdiction. Three structural facts make standard
quantitative finance assumptions invalid:

| Assumption | US/EU Standard | Nigeria Reality |
|---|---|---|
| Hurdle rate | ~5% USD risk-free | ~31% (28% CPI + 3% P2P friction) |
| Return distribution | Near-Gaussian | Fat-tailed (EM FX, crypto) |
| Objective function | Sharpe ratio | Omega ratio (distribution-free) |
| PnL denomination | USD | NGN via USDT P2P bridge |
| Trade timing | Any market hours | WAT-gated (London open overlap optimal) |
| Broker access | Interactive Brokers, Schwab | OANDA, IC Markets, Bybit (confirmed Nigeria-accessible) |

Phase 11 adds the infrastructure layer to address all six without breaking
the existing 2208-test research stack.

---

## Critical Review of Prior Proposal (2026-04-06)

A prior proposal was rejected. Key failures documented for future reference:

| Claim | Actual |
|---|---|
| "94.2% quant FAIL rate" | Fixed in Phase 7.10b to ~27.7% |
| "Transaction cost hardcoded at 0.8 bps" | `execution_cost_model.yml` already has per-asset LOB model |
| "portfolio_math.py at root" | Actual path: `etl/portfolio_math.py` |
| "Replace portfolio_math.py" | Would break `monitoring/performance_dashboard.py` and remove 6 existing functions |
| `from portfolio_math import fractional_kelly` | Wrong path; wrong function name |
| `np.sign(units)` in oanda_executor without numpy import | Bug |
| `hmmlearn` as main dependency | Not in requirements.txt; goes in requirements-ml.txt only |
| Bybit P2P `api2.bybit.com/fiat/otc/item/online` | Undocumented internal endpoint; no SLA |
| `pytz.timezone("Africa/Lagos")` | Conflicts with project convention; use `datetime.timezone(timedelta(hours=1))` |
| HMM covariance method (dead code) | Not wired to select_weights; no tests; adds heavy dep |

All corrections accepted. Phased integration approach adopted below.

---

## Phase A — Additive Math Extension (IMPLEMENTED 2026-04-06)

**Commit**: see git log  
**Test file**: `tests/etl/test_portfolio_math_ngn.py` (39 tests)  
**Zero breakage**: no existing function modified, no existing test affected

Functions added to `etl/portfolio_math.py`:

| Function | Purpose |
|---|---|
| `omega_ratio(returns, threshold)` | Omega vs NGN hurdle. Replaces Sharpe as barbell objective |
| `fractional_kelly_fat_tail(returns, rf, lambda)` | Quarter-Kelly with kurtosis dampener for EM/crypto |
| `effective_ngn_return(usd_return, spot_change, friction)` | USD -> NGN via USDT bridge |
| `portfolio_metrics_ngn(returns)` | Wrapper extending existing enhanced_metrics with NGN keys |

Constants added:
- `NGN_ANNUAL_INFLATION = 0.28` (env override: `NGN_ANNUAL_INFLATION`)
- `NGN_P2P_FRICTION = 0.03` (env override: `NGN_P2P_FRICTION`)
- `DAILY_NGN_THRESHOLD` — computed daily compound hurdle (~31% annual)

**The `beats_ngn_hurdle` flag is the canonical Phase 11 success criterion.**
A system generating `omega_ratio > 1.0` is outperforming the structural
devaluation rate — the minimum bar for Nigeria deployment.

---

## Phase B — New Modules, No Wiring (begins after THIN_LINKAGE >= 10)

Create as new standalone importable modules. No changes to existing files.

### `nigeria/fx_layer.py`
- Tracks NGN/USD conversion costs for USDT P2P withdrawals
- **Data source**: `CBN_OFFICIAL_RATE` env var (not Bybit internal API)
- Fallback constants: `cbn_official=1580.0`, `parallel_usdt=1640.0`, `spread_pct=0.038`
- If live rate is needed: use Bybit's official OAuth P2P API, not `api2.bybit.com`
- Key method: `effective_ngn_pnl(usd_pnl) -> dict` with full cost decomposition

### `nigeria/broker_cost_model.py`
- Empirical cost function: `c(instrument, broker, time) = base_cost * phi(t)`
- `phi(t)` = WAT time-of-day multiplier (London open overlap = 1.0x; overnight = 3.0x)
- **No pytz** — use `datetime.timezone(datetime.timedelta(hours=1))` for WAT
- Method `calibrate_from_mt5_logs(path)` calibrates phi(t) from actual MT5 history
- Requires 50+ MT5 trades before phi(t) is meaningful

### `utils/wat_execution_filter.py`
- Blocks execution outside high-liquidity windows from Lagos (UTC+1)
- **No pytz** — use stdlib `datetime.timezone(datetime.timedelta(hours=1))`
- FX windows: 09:00-12:00 WAT (London open), 14:00-17:00 WAT (NY pre-open)
- Crypto blackout: 01:00-05:00 WAT (lowest volume)
- Method: `is_executable(ts=None) -> bool`

### `execution/oanda_executor.py`
- OANDA v20 REST live executor (confirmed Nigeria-accessible)
- Environment variables required: `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_API_URL`
- Default `OANDA_API_URL` = practice endpoint until live validation complete
- Uses `fractional_kelly_fat_tail` from `etl.portfolio_math` (correct import path)
- Uses `WATExecutionFilter` to gate execution
- WAT filter block: logs seconds to next window, does not raise exception

### Tests for Phase B
Each new module requires a parallel `tests/` file before any wiring:
- `tests/nigeria/test_fx_layer.py`
- `tests/nigeria/test_broker_cost_model.py`
- `tests/utils/test_wat_execution_filter.py`
- `tests/execution/test_oanda_executor.py` (mock OANDA API)

---

## Phase C — Feature Flag in run_auto_trader.py (+1 week after Phase B)

Add `--executor {paper,oanda}` argument. Default remains `paper`. All existing
cron jobs, test suites, and overnight scripts are unaffected.

```python
# run_auto_trader.py addition (only)
parser.add_argument(
    "--executor",
    choices=["paper", "oanda"],
    default="paper",
    help="Execution backend. 'oanda' requires OANDA_API_KEY + OANDA_ACCOUNT_ID env vars.",
)
```

When `--executor oanda` is passed:
1. `WATExecutionFilter.is_executable()` is checked before signal routing
2. `OANDAExecutor.execute_signal()` replaces `PaperTradingEngine`
3. NGN PnL breakdown logged after each fill via `NGNFXLayer.effective_ngn_pnl()`

When `--executor paper` (default): zero change to existing behaviour.

---

## Phase D — HMM Regime-Conditional Covariance (+2 weeks after Phase C)

**Dependency**: add `hmmlearn>=0.3` to `requirements-ml.txt` only (not base `requirements.txt`).
Do not add to `requirements.txt` — it introduces sklearn version constraints that
may conflict with existing dependencies.

**Wiring**: add `EnsembleConfig.regime_covariance_method: {rolling, hmm}` config key.
Default `rolling` preserves backward compatibility. `hmm` activates the HMM path only
when explicitly configured.

**The HMM method must be wired into `select_weights` with test coverage** before
being merged. A standalone dead method must not be committed.

**Tests required before merge**:
- `test_hmm_covariance_returns_valid_matrix` (positive semi-definite, correct shape)
- `test_hmm_covariance_identifies_at_least_two_regimes` (on 2-year synthetic data)
- `test_rolling_covariance_still_default` (backward compatibility)

---

## Phase E — OANDA Candles as Data Source (+3 weeks after Phase D)

The system forecasts AAPL/MSFT/NVDA via yfinance. Before OANDA execution is useful,
the data source must serve instruments OANDA trades: EURUSD, XAUUSD, GBPUSD.

**Options**:
1. Wire OANDA `/v3/accounts/{id}/instruments/{instrument}/candles` as an extractor
   parallel to `etl/yfinance_extractor.py`
2. Use `etl/openbb_extractor.py` with OANDA as a provider (if OpenBB supports it)

This is the largest change in the Nigeria path. It requires:
- New `etl/oanda_extractor.py`
- Config integration in `config/pipeline_config.yml`
- Fallback chain: OANDA candles -> yfinance -> alpha_vantage (for FX)
- Tests: `tests/etl/test_oanda_extractor.py`

---

## Sequencing Summary

```
Phase A  [DONE 2026-04-06]  etl/portfolio_math.py additive extension + 39 tests
Phase B  [post gate clear]  nigeria/ + utils/ + execution/ new modules — no wiring
Phase C  [+1 week]          --executor oanda flag in run_auto_trader.py
Phase D  [+2 weeks]         HMM covariance in requirements-ml.txt + config flag
Phase E  [+3 weeks]         OANDA /candles as data source for FX instruments
```

---

## Implementation Rules (Non-Negotiable)

1. **No pytz** — use `datetime.timezone(datetime.timedelta(hours=1))` for WAT/UTC+1
2. **No Bybit internal API** — use `CBN_OFFICIAL_RATE` env var or official OAuth
3. **No replacement of existing files** — additive only until explicitly justified
4. **No hmmlearn in requirements.txt** — requirements-ml.txt only
5. **Dead code not accepted** — every method must be wired and tested before merge
6. **Import paths** — always `from etl.portfolio_math import ...` (never root-level)
7. **No new features while THIN_LINKAGE < 10** — evidence sprint is the only priority

---

## Success Criterion for Nigeria Deployment

```python
from etl.portfolio_math import portfolio_metrics_ngn
m = portfolio_metrics_ngn(live_returns)
assert m["beats_ngn_hurdle"] is True       # omega_ratio > 1.0
assert m["omega_ratio"] > 1.2              # 20% above hurdle (conservative buffer)
assert m["fractional_kelly_fat_tail"] > 0  # positive edge confirmed
```

A system meeting these criteria is demonstrably outperforming the structural
NGN devaluation rate and is ready for live Nigeria deployment.
