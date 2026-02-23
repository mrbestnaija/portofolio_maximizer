# Project Status - Portfolio Maximizer

**Last verified (focused)**: 2026-01-29
**Last full-suite**: 2026-01-07
**Dependency sanity check**: 2026-01-04
**Scope**: Engineering/integration health + paper-window MVS validation (not live profitability)
**Document updated**: 2026-01-29

**Metric definitions (canonical)**: `Documentation/METRICS_AND_EVALUATION.md` (implementations in `etl/database_manager.py`, `etl/portfolio_math.py`, `etl/statistical_tests.py`).

**Sequenced optimization roadmap (2026-01)**: `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` (bar-aware trading loop, horizon-consistent TS signals, execution cost alignment, run-local reporting).

## Phase 7.11: Directional Accuracy Enhancement Roadmap — PLANNED

**Status**: Plan written, not yet implemented.
**Plan doc**: `Documentation/QUANT_FAIL_RATE_RECOVERY_PLAN_2026-02-19.md` (Part C)
**Baseline**: 41-46% directional accuracy (post 7.10b). Target: ≥52% over 50+ live trades.

### Part C Roadmap Summary
| # | Change | Tier | Files | Expected DA lift |
|---|--------|------|-------|-----------------|
| C1 | Directional Consensus Gate (≥2/3 models agree) | 1 | `time_series_signal_generator.py` | +3-5 pp |
| C2 | Hurst Exponent Directional Policy (trend=follow, revert=fade) | 1 | `time_series_signal_generator.py` | +3-5 pp |
| C3 | Rolling IC Feature Culling (Spearman IC < 0.03 → drop) | 1 | `time_series_signal_generator.py` | +2-4 pp |
| C4 | EMA Momentum Pre-Filter (anti-whipsaw gate) | 1 | `time_series_signal_generator.py` | +3-5 pp |
| C5 | Direction Classifier (sklearn LR on lagged features) | 2 | new `models/direction_classifier.py` | +4-7 pp |
| C6 | Isotonic Regression Calibration (upgrade from Platt) | 2 | `time_series_signal_generator.py` | +2-3 pp |
| C7 | Volume Confirmation Gate (suppress low-volume signals) | 2 | `time_series_signal_generator.py` | +3-5 pp |
| C8 | Asymmetric Directional Loss (RMSE + λ·direction_error) | 2 | `samossa.py`, `mssa_rl.py` | +2-4 pp |
| C9 | Walk-Forward Hit Rate Tracking (auto-penalize lagging models) | 3 | `ensemble.py`, `database_manager.py` | +2-5 pp |
| C10 | Wavelet Denoising before SSA (PyWavelets db4) | 3 | `preprocessor.py` | +2-4 pp |
| C11 | Fourier Spectral Cycle Phase Detection | 3 | `samossa.py` | +1-3 pp |

**Implementation order**: C1 → C2 → C4 → C5 → C3 → C6 → C7 → C8 → C9 → C10 → C11

**All Part C config flags default to `enabled: false`** until validated on ≥50 live trades.

---

## Phase 7.10b: Quant FAIL Rate Recovery (2026-02-19) — COMPLETE

**Status**: Implemented and tested (802 tests passing, 0 failures).
**Plan doc**: `Documentation/QUANT_FAIL_RATE_RECOVERY_PLAN_2026-02-19.md`

### What changed
| Part | Change | Expected impact |
|------|--------|-----------------|
| A1-A2 | Weighted scoring (0.60 threshold) replaces ALL-MUST-PASS | FAIL rate 94.2% → ~60-70% |
| A3 | `--exclude-mode proof` strips artificial proof-mode entries from gate | FAIL rate calc more accurate |
| A4 | `min_expected_profit` $1.0→$5.0 abs + 0.2% relative (OR) | Consistent $50 floor |
| B1 | GARCH: skewt dist + AR(1) mean + ADF stationarity + GJR fallback | Directional signal; 28% IGARCH reduction |
| B2 | SAMoSSA: ARIMA residuals (was polyfit) + auto window (T//3) | Bug fix; proper residual model |
| B3 | MSSA-RL: Q-values wired to forecast direction + slope-capped trend forecast | Activates dead Q-learning code |
| B4 | Ensemble: auto_directional candidate from per-model CV hit rates | Data-driven weight selection |
| B5 | Confidence: Platt scaling (active only when PASS rate ≥ 15%) | Calibration when data is reliable |

### Quality audit findings + fixes applied
- `pipeline_config.yml` synced with new GARCH/SAMoSSA/MSSA-RL params (was out of sync)
- SAMoSSA window: hard 40-cap removed; uses full T//3 per paper recommendation
- MSSA-RL slope: 5% cumulative drift cap prevents divergent long-horizon forecasts
- Platt scaling: 15% minimum PASS rate guard prevents calibration on overwhelmingly-FAIL history
- Co-agent compatibility: all changes use optional params; existing callers unaffected

## Ensemble Status (Canonical)

For any external-facing statement about the **time-series ensemble** (SAMOSSA/MSSA-RL/GARCH/SARIMAX), use `ENSEMBLE_MODEL_STATUS.md` as the single source of truth.

This matters because the system exposes:
- A **per-forecast policy label** (`KEEP` / `RESEARCH_ONLY` / `DISABLE_DEFAULT`) recorded by the forecaster, and
- A separate **aggregate audit gate decision** produced by `scripts/check_forecast_audits.py`.

Do not conflate these.

## Verified Now

- Code compiles cleanly (`python -m compileall` on core packages)
- Full pytest suite passes: **529 tests** (last full-suite run 2026-01-07; still green)
- Brutal harness completes end-to-end with quant-validation health GREEN (see `logs/brutal/results_20260103_220403/reports/final_report.md`)
- LLM monitoring script no longer errors on missing `llm_db_manager` (see `scripts/monitor_llm_system.py`)
- Time Series execution validation prefers TS provenance edge (`net_trade_return` / `roundtrip_cost_*`) over historical drift fallbacks
- Auto-trader loop is bar-aware (`scripts/run_auto_trader.py` skips repeated cycles on the same bar; optional persisted bar-state)
- Auto-trader supports **cross-session portfolio persistence** (`--resume` default; `PaperTradingEngine` loads/saves DB state, reset via `bash/reset_portfolio.sh` or `scripts/migrate_add_portfolio_state.py`)
- Auto-trader parallel pipeline defaults ON for candidate prep + forecasts with GPU-first when available (`ENABLE_GPU_PARALLEL=1` + CUDA/torch present), otherwise CPU threads (override via `ENABLE_PARALLEL_TICKER_PROCESSING=0` / `ENABLE_PARALLEL_FORECASTS=0`); stress evidence in `logs/automation/stress_parallel_20260107_202403/comparison.json`
- Dependency baseline now includes `torch==2.9.1` in `requirements.txt`; optional `requirements-ml.txt` (when present) retains CUDA extras (CuPy/NVIDIA libs) for full GPU stacks
- TS signals use the horizon-end forecast target for `expected_return`/`target_price` (`models/time_series_signal_generator.py`)
- TS confidence is edge/uncertainty-aware and emits diagnostics provenance; quant validation supports `validation_mode=forecast_edge` using rolling CV regression metrics (`models/time_series_signal_generator.py`, `config/quant_success_config.yml`)
- Forecaster health uses persisted horizon-end forecast snapshots + lagged regression backfill so `get_forecast_regression_summary` stays run-fresh (`scripts/run_auto_trader.py`, `etl/database_manager.py`)
- Lifecycle exits treat `forecast_horizon` as bar count (intraday-safe) (`execution/paper_trading_engine.py`)
- Run reporting uses run-local PF/WR scoped by `run_id` and preserves lifetime metrics separately (`etl/database_manager.py`, `scripts/run_auto_trader.py`)
- DataSourceManager supports chunked OHLCV extraction via `chunk_size` / `DATA_SOURCE_CHUNK_SIZE`, with batching tested in `tests/etl/test_data_source_manager_chunking.py`
- Live dashboard is real-time and non-fictitious: `visualizations/live_dashboard.html` polls `visualizations/dashboard_data.json` every 5s and renders trade/price/PnL panels. Canonical producer is `scripts/dashboard_db_bridge.py` (DB→JSON) started by bash orchestrators; snapshots persist to `data/dashboard_audit.db` by default (`DASHBOARD_PERSIST=1`). Provenance checks are enforced via `scripts/audit_dashboard_payload_sources.py` (warns/fails when synthetic/demo contamination is detected or when trade source metadata is missing). Trade events default to the **latest run_id**, and positions fall back to `trade_executions` when `portfolio_positions` is empty.
- Portfolio impact checks include concentration caps + optional correlation warnings (when correlations can be computed from stored OHLCV)
- Position lifecycle management supports stop/target/time exits (so HOLD signals can still close positions when risk controls trigger)
- Trade execution telemetry persists mid-price + mid-slippage (bps) in `trade_executions` for bps-accurate cost priors
- Dependency note: `arch==8.0.0` enables full GARCH; if missing, `forcester_ts.garch.GARCHForecaster` falls back to EWMA for test/dev continuity
- CI notes: GitHub Actions runs `CI / test` on `ubuntu-latest` with Python 3.10 and executes `pip check` + `pytest -m "not gpu"`; project/issue automation workflows that require `PROJECTS_TOKEN` must be treated as non-blocking and skip when secrets are unavailable; Git workflow is remote-first (remote `master` is canonical) — see `Documentation/GIT_WORKFLOW.md`.

### Verification Commands (Repro)

```bash
# Focused validations (2026-01-29)
./simpleTrader_env/bin/python -m pytest -q \
  tests/ai_llm/test_signal_validator.py \
  tests/execution/test_paper_trading_engine.py

./simpleTrader_env/bin/python -m pytest -q \
  tests/forecasting/test_parameter_learning.py::TestParameterLearning::test_mssa_rl_learns_rank_from_variance \
  tests/forecasting/test_parameter_learning.py::TestParameterLearning::test_mssa_rl_change_points_data_dependent \
  tests/etl/test_time_series_forecaster.py::TestMSSARL

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
