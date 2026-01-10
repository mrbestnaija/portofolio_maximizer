# Project Status - Portfolio Maximizer

**Last verified**: 2026-01-07  
**Dependency sanity check**: 2026-01-04  
**Scope**: Engineering/integration health + paper-window MVS validation (not live profitability)
**Document updated**: 2026-01-07  

**Metric definitions (canonical)**: `Documentation/METRICS_AND_EVALUATION.md` (implementations in `etl/database_manager.py`, `etl/portfolio_math.py`, `etl/statistical_tests.py`).

**Sequenced optimization roadmap (2026-01)**: `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` (bar-aware trading loop, horizon-consistent TS signals, execution cost alignment, run-local reporting).

## Verified Now

- Code compiles cleanly (`python -m compileall` on core packages)
- Full pytest suite passes: **529 tests** (new chunked OHLCV test added; last full-suite run still green)
- Brutal harness completes end-to-end with quant-validation health GREEN (see `logs/brutal/results_20260103_220403/reports/final_report.md`)
- LLM monitoring script no longer errors on missing `llm_db_manager` (see `scripts/monitor_llm_system.py`)
- Time Series execution validation prefers TS provenance edge (`net_trade_return` / `roundtrip_cost_*`) over historical drift fallbacks
- Auto-trader loop is bar-aware (`scripts/run_auto_trader.py` skips repeated cycles on the same bar; optional persisted bar-state)
- Auto-trader parallel pipeline defaults ON for candidate prep + forecasts with GPU-first when available (`ENABLE_GPU_PARALLEL=1` + CUDA/torch present), otherwise CPU threads (override via `ENABLE_PARALLEL_TICKER_PROCESSING=0` / `ENABLE_PARALLEL_FORECASTS=0`); stress evidence in `logs/automation/stress_parallel_20260107_202403/comparison.json`
- Dependency baseline now includes `torch==2.9.1` in `requirements.txt`; optional `requirements-ml.txt` (when present) retains CUDA extras (CuPy/NVIDIA libs) for full GPU stacks
- TS signals use the horizon-end forecast target for `expected_return`/`target_price` (`models/time_series_signal_generator.py`)
- TS confidence is edge/uncertainty-aware and emits diagnostics provenance; quant validation supports `validation_mode=forecast_edge` using rolling CV regression metrics (`models/time_series_signal_generator.py`, `config/quant_success_config.yml`)
- Forecaster health uses persisted horizon-end forecast snapshots + lagged regression backfill so `get_forecast_regression_summary` stays run-fresh (`scripts/run_auto_trader.py`, `etl/database_manager.py`)
- Lifecycle exits treat `forecast_horizon` as bar count (intraday-safe) (`execution/paper_trading_engine.py`)
- Run reporting uses run-local PF/WR scoped by `run_id` and preserves lifetime metrics separately (`etl/database_manager.py`, `scripts/run_auto_trader.py`)
- DataSourceManager supports chunked OHLCV extraction via `chunk_size` / `DATA_SOURCE_CHUNK_SIZE`, with batching tested in `tests/etl/test_data_source_manager_chunking.py`
- Portfolio impact checks include concentration caps + optional correlation warnings (when correlations can be computed from stored OHLCV)
- Position lifecycle management supports stop/target/time exits (so HOLD signals can still close positions when risk controls trigger)
- Trade execution telemetry persists mid-price + mid-slippage (bps) in `trade_executions` for bps-accurate cost priors
- Dependency note: `arch==8.0.0` enables full GARCH; if missing, `forcester_ts.garch.GARCHForecaster` falls back to EWMA for test/dev continuity
- CI notes: GitHub Actions runs `CI / test` on `ubuntu-latest` with Python 3.10 and executes `pip check` + `pytest -m "not gpu"`; project/issue automation workflows that require `PROJECTS_TOKEN` must be treated as non-blocking and skip when secrets are unavailable; Git workflow is remote-first (remote `master` is canonical) — see `Documentation/GIT_WORKFLOW.md`.

### Verification Commands (Repro)

```bash
# From repo root
./simpleTrader_env/bin/python -m compileall -q ai_llm analysis backtesting etl execution forcester_ts models monitoring recovery risk scripts tools

./simpleTrader_env/bin/python -m pytest -q \
  tests/test_diagnostic_tools.py \
  tests/ai_llm/test_signal_validator.py \
  tests/execution/test_paper_trading_engine.py \
  tests/execution/test_order_manager.py \
  tests/scripts/test_forecast_persistence.py \
  tests/etl/test_database_manager_schema.py \
  tests/etl/test_data_source_manager_chunking.py
```

### Brutal Run Snapshot (2026-01-03)

Artifact bundle:
- `logs/brutal/results_20260103_220403/reports/final_report.md`
- `logs/brutal/results_20260103_220403/test.log`
- `logs/brutal/results_20260103_220403/logs/pipeline_execution.log`

Headline outcomes:
- 42/42 stages PASSED (pass rate 100%)
- Quant validation health (global): GREEN
- Pipeline execution ran in SYNTHETIC mode (by design for brutal validation)

Notable warnings (do not fail the suite, but matter for production readiness):
- `logs/brutal/results_20260103_220403/logs/monitoring_run.log`: monitoring summary reported `overall_status=DEGRADED` (latency above 5s benchmark, and signal-backtest inputs missing)
- `logs/brutal/results_20260103_220403/logs/profitability_validation.log`: profitability status WARNING (expected-return below threshold; no realised PnL in the test DB)

### Operational Notes (2026-01-04)

- `bash/repo_cleanup.sh` requires LF line endings; `.gitattributes` now enforces `eol=lf` for `*.sh` to avoid `pipefail\\r` errors on Windows checkouts.
- WSL is the stable runtime for full test runs and monitoring in this environment; Windows `python` is not available via PATH on this machine.

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
