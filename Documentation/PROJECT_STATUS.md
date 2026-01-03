# Project Status - Portfolio Maximizer

**Last verified**: 2025-12-27  
**Dependency sanity check**: 2025-12-27  
**Scope**: Engineering/integration health + paper-window MVS validation (not live profitability)
**Document updated**: 2026-01-03  

**Metric definitions (canonical)**: `Documentation/METRICS_AND_EVALUATION.md` (implementations in `etl/database_manager.py`, `etl/portfolio_math.py`, `etl/statistical_tests.py`).

## Verified Now

- Code compiles cleanly (`python -m compileall` on core packages)
- Focused test run passes (signal validator + paper trading engine + DB schema + diagnostics): **31 tests**
- Time Series execution validation prefers TS provenance edge (`net_trade_return` / `roundtrip_cost_*`) over historical drift fallbacks
- Portfolio impact checks include concentration caps + optional correlation warnings (when correlations can be computed from stored OHLCV)
- Position lifecycle management supports stop/target/time exits (so HOLD signals can still close positions when risk controls trigger)
- Trade execution telemetry persists mid-price + mid-slippage (bps) in `trade_executions` for bps-accurate cost priors
- Dependency note: `arch==8.0.0` enables full GARCH; if missing, `forcester_ts.garch.GARCHForecaster` falls back to EWMA for test/dev continuity

### Verification Commands (Repro)

```bash
# From repo root
python -m compileall -q ai_llm analysis backtesting etl execution forcester_ts models monitoring recovery risk scripts tools

pytest -q \
  tests/test_diagnostic_tools.py \
  tests/ai_llm/test_signal_validator.py \
  tests/execution/test_paper_trading_engine.py \
  tests/execution/test_order_manager.py \
  tests/etl/test_database_manager_schema.py
```

### MVS Snapshot (Verified from DB)

Full-history (realised trades only):
- Total trades: 31
- Total profit: 15.18 USD
- Win rate: 51.6%
- Profit factor: 1.28
- Status: **PASS**

Recent 60-day window (realised trades only):
- Total trades: 6
- Total profit: -4.27 USD
- Win rate: 33.3%
- Profit factor: 0.66
- Status: **FAIL**

**Interpretation:** the system can clear the minimum bar on a replay window / accumulated history, but still needs enough *recent* trades and positive edge in actual paper/live windows.

### MVS Paper Window (Historical Verified Replay)

Command:

```bash
python scripts/run_mvs_paper_window.py \
  --tickers AAPL,MSFT,GOOGL \
  --window-days 365 \
  --max-holding-days 2 \
  --entry-momentum-threshold 0.003 \
  --reset-window-trades
```

Result (realised trades only):
- Total trades: 31
- Total profit: 15.18 USD
- Win rate: 51.6%
- Profit factor: 1.28
- Status: **PASS**

Report artifact: `reports/mvs_paper_window_20251226_183023.md`

## Current Status (Reality-Based)

- 🟢 **Engineering / Integration**: Unblocked (core pipeline pieces compile and tests above pass)
- 🟡 **Profitability / Quant Health**: Full-history MVS is PASS, but recent windows can still FAIL due to low trade count and weak edge; paper/live still needs sustained evidence (and quant-validation health GREEN/YELLOW)
- ⚪ **LLM (Ollama) Live Inference**: Optional; integration tests skip unless Ollama is running and `RUN_OLLAMA_TESTS=1`

## Pending Tasks (Highest Value Next)

1. Drive **recent-window MVS PASS** on actual paper/live runs (≥30 realised trades, positive PnL, WR/PF thresholds) using `bash/run_end_to_end.sh` / `scripts/run_auto_trader.py`.
2. Calibrate `signal_routing.time_series.cost_model.default_roundtrip_cost_bps` using the persisted mid-slippage telemetry (`scripts/estimate_transaction_costs.py` → `scripts/generate_config_proposals.py` → `scripts/generate_signal_routing_overrides.py`).
3. Use `backtesting/candidate_simulator.py` walk-forward harness to validate thresholds/cost-model choices without lookahead before promoting configs.
4. Archive fresh brutal/end-to-end artifacts under `logs/`/`reports/` and refresh quant-health classification based on the recent window (not full history).
