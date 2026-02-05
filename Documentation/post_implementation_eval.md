# Post-Implementation Testing & Evaluation Playbook

This playbook operationalizes the quantitative, visual evaluation of forecasts and signals. It aligns with: `CRITICAL_REVIEW.md`, `QUANTIFIABLE_SUCCESS_CRITERIA.md`, `OPTIMIZATION_IMPLEMENTATION_PLAN.md`, `TIME_SERIES_FORECASTING_IMPLEMENTATION.md`, `LLM_ENHANCEMENTS_IMPLEMENTATION_SUMMARY_2025-10-22.md`, `PROFIT_CALCULATION_FIX.md`, `QUANT_TIME_SERIES_STACK.md`, `UNIFIED_ROADMAP.md`, `REFACTORING_IMPLEMENTATION_COMPLETE.md`.

## Metrics (per run and trend)
- Hit rate, profit factor, CAGR, max drawdown, Sharpe/Sortino, average/median trade PnL.
- Calibration: expected_return vs realized_return buckets; residual mean/std and sMAPE.
- Signal mix: BUY/SELL/HOLD counts; zero-size trade rejects; quality gate pass rate.
- Drift: PSI + vol_psi per split; overlap warnings; coverage/missing/outliers quality.
- Latency: mean/p95 per stage (TS, LLM, routing); cache hit rate.

## Data & splits
- Use rolling/expanding CV (`--use-cv`) to log `split_drift_latest.json`; enforce PSI/vol_psi < 0.2 where possible.
- Ensure test isolation: persist split metadata + hash; watch overlap warnings.
- Expand coverage to uncorrelated markets (equities + commodities + crypto/FX) to dilute correlation and seek positive PnL: e.g., `AAPL,MSFT,CL=F,GC=F,BTC-USD,EURUSD=X`.

## Runs to execute
1) **ETL + CV drift check**
   ```
   python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,CL=F,GC=F,BTC-USD,EURUSD=X --use-cv --n-splits 5 --start 2020-01-01 --end 2024-01-01 --execution-mode auto
   ```
   - Inspect `logs/pipeline_run.log`, `visualizations/split_drift_latest.json`, DB tables `split_drift_metrics`, `data_quality_snapshots`.

2) **Backtest loop (profit + calibration)**
   ```
   python scripts/run_auto_trader.py --tickers AAPL,MSFT,CL=F,GC=F,BTC-USD,EURUSD=X --lookback-days 365 --forecast-horizon 30 --cycles 1 --sleep-seconds 0 --initial-capital 25000 --enable-llm
   ```
   - Review `visualizations/dashboard_data.json` for PnL, win_rate, equity (realized + MTM), latency, quality.

3) **Batch sensitivity (optional)**
   - Sweep `min_return`, `confidence_threshold`, `max_risk` and compare calibration + profit factor across runs; store JSON artifacts per run ID.

## Visuals to update/inspect
- Dashboard: equity (MTM + realized overlay), latency, quality distribution, routing mix, recent signals.
- Add-on plots (for notebooks or future dashboard):
  - Calibration curve: expected vs realized buckets.
  - Residual histogram + QQ.
  - Confusion-style table: action vs market move.

## Quality & drift gates
- Quality_score >= 0.5; coverage > 0.9; missing_pct < 2%; outlier_frac < 5%.
- Drift gate: PSI/vol_psi <= 0.2 (warn >0.2).
- Reject low-quality windows; flag zero-size trades and re-tune position sizing if frequent.

## Pass/Fail bar (quantifiable success)
- Win rate >= 0.55, profit factor > 1.3, max drawdown < 12%, Sharpe > 1.2 over the test window.
- Positive realized PnL across the mixed, uncorrelated basket.
- Calibration bias |expected - realized| mean < 2% and slope ~1.0 (no strong under/over-predict).

## To automate next
- Nightly cron/CI: ETL + backtest on mixed tickers; persist dashboard JSON/PNG + drift JSON; fail build on metric regression.
- Add tests: dashboard JSON schema validation; drift threshold assertion; calibration bucket sanity; zero-size trade rate < threshold.

