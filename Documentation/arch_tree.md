# UPDATED TO-DO LIST: Portfolio Maximizer - Current Implementation Status

## CURRENT PROJECT STATUS: PARTIALLY BLOCKED – 2025-11-15 brutal run issues largely remediated; higher-order hyperopt + validator/backfill work still in progress (TS model candidates + institutional-grade search scaffolding now in place)
**All Core Phases Complete**: ETL + Analysis + Visualization + Caching + k-fold CV + Multi-Source + Config-Driven + Checkpointing & Logging + Error Monitoring + Performance Optimization + Remote Synchronization Enhancements (LLM now operates purely as fallback/redundancy per TIME_SERIES_FORECASTING_IMPLEMENTATION.md)
**Recent changes (2025-12-19)**: Synthetic stack upgraded with profiles + copula/tail shocks + richer features/calibration + txn-cost microstructure (TxnCostBps/ImpactBps) and persisted features/calibration artifacts; `SYNTHETIC_DATASET_ID=latest` now points to `syn_6c850a7d0b99` (manifest + features + calibration). Forecasting/ETL statistical hardening landed (SARIMAX shift-safe backtransform, ensemble variance screening + row-wise convex blending, MSSA-RL standardized CUSUM, synthetic/live isolation, leak-free post-split normalization). GPU preference wiring remains in place (`PIPELINE_DEVICE` auto-detects CUDA with CPU fallback). Added cache/log sanitizer (`scripts/sanitize_cache_and_logs.py`) and cron hook `sanitize_caches` to keep artifacts within 14-day retention by default.
**Recent Achievements**:
- 2025-12-19 Delta (forecasting + ETL hardening): SARIMAX log-shift inversion + Jarque–Bera compatibility, ensemble one-sided variance screening + minimum weight pruning + row-wise blending under partial forecasts, MSSA-RL standardized CUSUM mean-shift detection, and ETL isolation/slicing/leak-free scaling. Regression coverage: `tests/forcester_ts/test_ensemble_and_scaling_invariants.py`, `tests/etl/test_time_series_forecaster.py`, `tests/integration/test_time_series_signal_integration.py`, and core ETL/synthetic suites.
- 2025-11-30 Sentiment overlay plan captured (`Documentation/SENTIMENT_SIGNAL_INTEGRATION_PLAN.md`); `config/sentiment.yml` remains disabled with strict gating and `tests/sentiment/test_sentiment_config_scaffold.py` guarding activation until profitability beats the benchmark.
- 2025-12-04 Delta (TS/LLM guardrails + MVS reporting): TimeSeriesSignalGenerator now treats quant validation as a hard gate for TS trades (FAILED profiles demote BUY/SELL to HOLD outside diagnostic modes, using `config/quant_success_config.yml`), `scripts/run_auto_trader.py` only enables LLM fallback once `data/llm_signal_tracking.json` reports at least one validated signal (LLM remains research-only otherwise), and `bash/run_end_to_end.sh`/`bash/run_pipeline_live.sh` clear DIAGNOSTIC_*/LLM_FORCE_FALLBACK envs and print MVS-style profitability summaries via `DatabaseManager.get_performance_summary()` after each run.
- 2025-12-04 Delta (Quant monitoring + brutal integration): `scripts/check_quant_validation_health.py` now reads `config/forecaster_monitoring.yml` to classify global quant health as GREEN/YELLOW/RED (strict RED gate at `max_fail_fraction=0.90`, softer YELLOW warning band), `scripts/summarize_quant_validation.py` uses the same config for per-ticker GREEN/YELLOW/RED tiers, and `bash/comprehensive_brutal_test.sh` embeds the global classification in `final_report.md` as **Quant validation health (global)** so every brutal run is self-describing.
- 2025-12-03 Delta (diagnostic mode + invariants): DIAGNOSTIC_MODE/TS/EXECUTION toggles relax TS thresholds (confidence=0.10, min_return=0, max_risk=1.0, volatility filter off) and make PaperTradingEngine permissive (>=1 share) while bypassing LLM latency guards in diagnostics; volume_ma_ratio now guards zero/NaN volume. Numeric/scaling invariants and dashboard/quant health tests pass in `simpleTrader_env` (`tests/forcester_ts/test_ensemble_and_scaling_invariants.py`, `tests/forcester_ts/test_metrics_low_level.py`, dashboard payload + quant health scripts). Reduced-universe diagnostic run (MTN, SOL, GC=F, EURUSD=X; cycles=1; horizon=10; cap=$25k) executed 4 trades with PnL -0.06%, updated `visualizations/dashboard_data.json`; positions: long MTN 10, short SOL 569, short GC=F 1, short EURUSD=X 792; quant_validation fail_fraction 0.932 (<0.98) and negative_expected_profit_fraction 0.488 (<0.60).
- 2025-12-07: GPU-parallel, energy-aware runner checklist added (`Documentation/GPU_PARALLEL_RUNNER_CHECKLIST.md`) plus a shard-per-GPU orchestration stub (`bash/run_gpu_parallel.sh`) and a trade-count-aware rebuild helper (`bash/auto_rebuild_and_sweep.sh`) to rebuild evidence only when needed.
- 2025-12-07 Delta (DB recovery + power-aware rebuild): Restored the latest good eval snapshot to `data/portfolio_maximizer.db` after corruption (preserved corrupt copy), re-tightened per-ticker TS thresholds (CL=F 0.55/0.005, AAPL 0.65/0.010) with high-notional names removed from diagnostics, disabled LLM fallback/redundancy for faster sampling, and added `bash/auto_rebuild_and_sweep.sh` to rebuild trade history only when realised trades are below target, then refresh slippage and TS sweeps.
- Remote Sync (2025-11-06): Pipeline entry point refactoring, data persistence auditing, LLM graceful failure, comprehensive documentation updates ⭐ NEW
- Phase 4.6: Platform-agnostic architecture
- Phase 4.7: Configuration-driven CV
- Phase 4.8: Checkpointing and event logging with 7-day retention
- Phase 5.2: LLM Integration Complete (Ollama) ⭐ COMPLETE
- Phase 5.3: Profit Calculation Fix Applied (Oct 14, 2025) ⭐ CRITICAL
- Phase 5.4: Ollama Health Check Fixed (Oct 22, 2025) ⭐ COMPLETE
- Phase 5.5: Error Monitoring & Performance Optimization (Oct 22, 2025) ⭐ NEW
- Week 5.6: Statistical validation suite + paper trading integration (Nov 02, 2025) ⭐ NEW
- Week 5.6: Visual analytics dashboard with market/commodity context overlays (Nov 02, 2025) ⭐ NEW
- Week 5.6: Signal validator backtests publish statistical/bootstrapped metrics to monitoring (Nov 02, 2025) ⭐ NEW
- Week 5.6: LLM latency guard telemetry now visible in system monitor dashboards (Nov 02, 2025) ⭐ NEW
- Week 5.6: SQLite “disk I/O” auto-retry added for OHLCV ingestion (Nov 02, 2025) ⭐ NEW
- Week 5.6: `--config config.yml` alias resolves to `config/pipeline_config.yml` (Nov 02, 2025) ⭐ NEW
- Week 5.6: All pipeline/utility logs streamed to `logs/` directory (Nov 02, 2025) ⭐ NEW
- Week 5.7: Time-series models extracted into `forcester_ts/` (SARIMAX, GARCH, SAMOSSA, MSSA-RL) with shared orchestration (Nov 06, 2025) ⭐ NEW
- Week 5.7: Dashboard pipeline emits forecast/signal PNGs via `etl/dashboard_loader.py` + `TimeSeriesVisualizer.plot_forecast_dashboard` (Nov 06, 2025) ⭐ NEW
- Week 5.7: Token-throughput failover auto-selects faster Ollama models when tokens/sec degrade (`ai_llm/ollama_client.py`, Nov 12, 2025) ⭐ NEW
- Week 5.8: Time Series Signal Generation Refactoring IMPLEMENTED (Nov 06, 2025) ⭐ NEW - **ROBUST TESTING REQUIRED**
  - Time Series ensemble is DEFAULT signal generator (models/time_series_signal_generator.py) - **TESTING REQUIRED**
  - Signal Router routes TS primary + LLM fallback (models/signal_router.py) - **TESTING REQUIRED**
  - Unified signal interface for backward compatibility (models/signal_adapter.py) - **TESTING REQUIRED**
  - Unified trading_signals database table - **TESTING REQUIRED**
  - Regression metrics (RMSE / sMAPE / tracking error) persisted to SQLite feed the router + dashboards (forecester_ts/forecaster.py, DatabaseManager.save_forecast regression_metrics column) - **LIVE**
  - Complete pipeline integration with 50 tests written (38 unit + 12 integration) - **NEEDS EXECUTION & VALIDATION**
- Week 5.9: Monitoring + Nightly Backfill Instrumentation (Nov 09, 2025) ⭐ NEW
  - `scripts/monitor_llm_system.py` logs latency benchmarks (`logs/latency_benchmark.json`), emits `llm_signal_backtests` summaries, and saves JSON run reports for dashboards.
  - `schedule_backfill.bat` replays validator jobs nightly; register via Windows Task Scheduler (02:00 daily) to keep Time Series + LLM metrics fresh.
  - `models/time_series_signal_generator.py` hardened (volatility scalar conversion + HOLD provenance timestamps) and regression-tested via `pytest tests/models/test_time_series_signal_generator.py -q` plus the targeted integration smoke.
  - `simpleTrader_env/` (authorised virtual environment) is the sole supported interpreter across Windows/WSL; all other ad-hoc venvs were removed to keep configuration consistent.
- Week 5.9: Autonomous Profit Engine roll-out (Nov 12, 2025) ⭐ NEW
  - `scripts/run_auto_trader.py` chains extraction → validation → forecasting → Time Series signal generation → signal routing → execution (PaperTradingEngine) with optional LLM fallback, keeping cash/positions/trade history synchronized each cycle.
  - `README.md` + `Documentation/UNIFIED_ROADMAP.md` now present the platform as an **Autonomous Profit Engine**, highlight the hands-free loop in Key Features, and add a Quick Start recipe plus project-structure pointer so operators can launch the trader immediately.
  - See `Documentation/NAV_RISK_BUDGET_ARCH.md` and `Documentation/NAV_BAR_BELL_TODO.md` for the NAV-centric barbell wiring (TS-first, LLM capped fallback) that wraps this loop.
  - `scripts/run_etl_pipeline.py` stage planner updated: `data_storage` is part of the core stage list, Time Series forecasting/signal routing run before any LLM stage, and LLM work is appended only as fallback after the router.
  - `scripts/run_auto_trader.py` now adds the repo root via `site.addsitedir(...)` before importing project packages so the runtime works even without an editable install or manual PYTHONPATH adjustments.

### Synthetic event & microstructure additions (2025-12-19)
- `config/synthetic_data_config.yml` gains profile support (`config/synthetic_data_profiles.yml`), t-copula/tail-scale shocks, macro regime change events, intraday seasonality, size-aware slippage, and exec-cost proxies (TxnCostBps/ImpactBps) alongside existing regime/event/microstructure defaults.
- `scripts/generate_synthetic_dataset.py` now persists features (SMA/vol/RSI/MACD/Bollinger/zscores), calibration stats (optional real-data reference), manifests, and latest pointer; retention pruning enforces `keep_last`. `SYNTHETIC_DATASET_ID=latest` resolves to `syn_6c850a7d0b99`.
- `etl/synthetic_extractor.py` reads profile overrides (`SYNTHETIC_PROFILE`), supports t-copula shocks, emits txn-cost columns, and records events/regimes in attrs for downstream consumers.

### GPU defaults and PIPELINE_DEVICE (2025-12-08)
- `scripts/run_etl_pipeline.py`, backtest stubs (`scripts/backtest_llm_signals.py`, `scripts/run_backtest_for_candidate.py`), and GAN runner (`bash/run_gan_stub.sh` → `scripts/train_gan_stub.py`) auto-detect CUDA (torch/cupy) and set `PIPELINE_DEVICE=cuda` when available, falling back to CPU.
- New `--prefer-gpu/--no-prefer-gpu` flags thread through the runners; env `PIPELINE_DEVICE` is logged and respected across stages so TS/ETL/backtests stay aligned with operator intent.
- GAN stub checkpoints under `models/synthetic/gan_stub` consume the same synthetic parquet to keep GPU/CPU parity during quick experiments.

### Broader refresh (timelines, phase statuses, run IDs)
- Timeline: Phase 1/2 synthetic event/regime + market-hours/microstructure hooks are live; tail/corr profiles and feature/calibration persistence added; Phase 2 calibration sweeps + ML generator remain open; GPU preference wiring is landed across pipeline/backtests/GAN stub.
- Phase status: Synthetic stack is green with Spread/Slippage/TxnCost/Impact outputs; GPU auto-detect defaults to CUDA when present with silent CPU fallback; GAN stub remains optional/torch-gated.
- Run IDs: `data/synthetic/latest.json` → `syn_6c850a7d0b99` (profiles + copula + features/calibration); prior baselines `syn_682c415bb89c`, `syn_fc8a37128efc`, and `syn_714ae868f78b` remain for comparison. Latest smoke ran via `scripts/brutal_synthetic_smoke.py` (with PYTHONPATH set) / `scripts/run_etl_pipeline.py --execution-mode synthetic`.
- Week 5.10: Higher-Order Hyperopt & Regime-Aware Backtesting (Nov 24, 2025) ⭐ NEW
  - `bash/run_post_eval.sh` now acts as a higher-order hyper-parameter driver around ETL → auto-trader → strategy optimization, treating evaluation windows, `min_expected_profit`, and `time_series.min_expected_return` as tunable knobs.
  - Stochastic, non-convex search uses a bandit-style explore/exploit policy (30% explore / 70% exploit by default, dynamically adjusted per trial) and logs trials to `logs/hyperopt/hyperopt_<RUN_ID>.log`.
  - Hyperopt candidate ranges are tightened using historic quant-validation metrics for profitable tickers (e.g., AAPL, COOP, GC=F, EURUSD=X), and the best configuration per run is re-executed as `<RUN_ID>_best` with metrics surfaced in `visualizations/dashboard_data.json`.
  - `bash/run_end_to_end.sh` and `bash/run_auto_trader.sh` honour `HYPEROPT_ROUNDS>0` by delegating to `bash/run_post_eval.sh`, making higher-order hyperopt the default orchestration mode when enabled.
  - 2025-12-07 TS model search scaffold:
    - `ts_model_candidates` table in `etl/database_manager.py` stores per-(ticker, regime, candidate_name) CV metrics, stability, and scalar scores.
    - `scripts/run_ts_model_search.py` runs rolling-window CV for compact SARIMAX/SAMOSSA grids and records candidates into `ts_model_candidates`.
    - `etl/statistical_tests.py` provides Diebold–Mariano-style comparison and rank stability helpers so candidate selection is statistically grounded.
    - `scripts/build_automation_dashboard.py` consolidates TS sweeps, transaction costs, sleeve promotion plans, config proposals, and best strategy/TS model candidates into `visualizations/dashboard_automation.json` for institutional-grade review.
- `bash/comprehensive_brutal_test.sh` (Nov 12) run: profit-critical + ETL suites passed, but `tests/etl/test_data_validator.py` is missing and the Time Series block timed out with a `Broken pipe`, so TS/LLM regression coverage remains outstanding. *(Nov 16 update: the script now defaults to **Time Series-first** execution�LLM tests only run when `BRUTAL_ENABLE_LLM=1`, keeping the brutal gate aligned with `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`.)*

## Architecture Overview

```
                   v 
        +---------------------------------------------+
        |            Output Layer                     |
        +---------------------------------------------+
                |                    |
                v                    v
    +--------------------+  +--------------------+
    | JSON Reports       |  | PNG Visualizations |
    | - Analysis         |  | - 8 plots          |
    | - Metrics          |  | - 1.6 MB total     |
    +--------------------+  +--------------------+
```

### Data Flow

```
External Data Sources
    |
    +--> Yahoo Finance API --+
    |                        |
                             |
                             v
                    +-----------------+
                    |   Cache Check   |<--- 24h validity
                    +-----------------+
                             |
                    +--------+--------+
                    |                 |
                Hit v                 v Miss
            +-----------+      +-----------+
            |  Cache    |      |  Network  |
            |  (Fast)   |      |  (Fetch)  |
            +-----------+      +-----------+
                    |                 |
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    |   Raw Storage   |
                    |    (Parquet)    |
                    +-----------------+
                             |
                             v
                    +-----------------+
                    |   Validation    |
                    |    (Quality)    |
                    +-----------------+
                             |
                             v
                    +-----------------+
                    |  Preprocessing  |
                    |   (Transform)   |
                    +-----------------+
                             |
                             v
                    +-----------------+
                    | Train/Val/Test  |
                    |   Split 70/15/15|
                    +-----------------+
                             |
                 +-----------+-----------+
                 |           |           |
                 v           v           v
        +-----------+ +-----------+ +-----------+
        | Training  | | Validation| |  Testing  |
        |   (704)   | |   (151)   | |    (151)  |
        +-----------+ +-----------+ +-----------+
                             |
                 +-----------+-----------+
                 |           |           |
                 v           v           v
        +-----------+ +-----------+ +-----------+
        | Analysis  | | Portfolio | | Backtest  |
        |           | |    Opt    | |  (Future) |
        +-----------+ +-----------+ +-----------+
                             |
                             v
                    +-----------------+
                    | Visualizations  |
                    |     & Reports   |
                    +-----------------+
```

### Module Dependencies

```
scripts/run_etl_pipeline.py
    |
    +--> etl/yfinance_extractor.py
    |       +--> etl/data_storage.py (cache)
    |       +--> retry logic, rate limiting
    |
    +--> etl/data_validator.py
    |       +--> statistical validation
    |
    +--> etl/preprocessor.py
    |       +--> missing data handling
    |       +--> normalization
    |
    +--> etl/data_storage.py
            +--> train/val/test split
            +--> parquet I/O

scripts/analyze_dataset.py
    |
    +--> etl/time_series_analyzer.py
            +--> ADF test (statsmodels)
            +--> ACF/PACF computation
            +--> Statistical summary
            +--> JSON report generation
```

### Frontier Market Multi-Ticker Coverage (Nov 2025) ?? NEW
- `etl/frontier_markets.py` centralizes the Nigeria ? Bulgaria ticker sets and exposes `merge_frontier_tickers()` so every training/test flow can append the curated symbols without duplicating lists.
- `scripts/run_etl_pipeline.py` now ships a `--include-frontier-tickers` flag that appends the curated symbols whenever a run contains multiple tickers. The flag is wired through `bash/run_pipeline_live.sh`, `bash/run_pipeline_dry_run.sh`, `.bash/full_test_run.sh`, `bash/test_real_time_pipeline.sh` (Step 10 synthetic multi-run), and `bash/comprehensive_brutal_test.sh` (frontier training stage) to keep `.bash/` and `.script/` orchestration in sync.
- Frontier coverage list (also referenced in `Documentation/UNIFIED_ROADMAP.md`, `TO_DO_LIST_MACRO.mdc`, and the security docs to keep requirements synchronized):
  - **Nigeria (NGX)**: `MTNN`, `AIRTELAFRI`, `ZENITHBANK`, `GUARANTY`, `FBNH`
  - **Kenya (NSE)**: `EABL`, `KCB`, `SCANGROUP`, `COOP`
  - **South Africa (JSE)**: `NPN`, `BIL`, `SAB`, `SOL`, `MTN`
  - **Vietnam (HOSE)**: `VHM`, `GAS`, `BID`, `SSI`
  - **Bangladesh (DSE)**: `BRACBANK`, `LAFSURCEML`, `IFADAUTOS`, `RELIANCE`
  - **Sri Lanka (CSE)**: `COMBANK`, `HNB`, `SAMP`, `LOLC`
  - **Pakistan (PSX)**: `OGDC`, `MEBL`, `LUCK`, `UBL`
  - **Kuwait (KSE)**: `ZAIN`, `NBK`, `KFH`, `MAYADEEN`
  - **Qatar (QSE)**: `QNBK`, `DUQM`, `QISB`, `QAMC`
  - **Romania (BVB)**: `SIF1`, `TGN`, `BRD`, `TLV`
  - **Bulgaria (BSE)**: `5EN`, `BGO`, `AIG`, `SYN`
- All multi-ticker documentation snippets now highlight `--include-frontier-tickers` so readers do not accidentally omit the frontier set when recreating tests.

- Week 5.9: Quant success governance + routing configs (Nov 13, 2025) ?-? NEW
  - `config/quant_success_config.yml` defines the Sharpe/Sortino/VaR thresholds the Time Series signal generator must satisfy before a signal is eligible for routing/execution.
  - `config/signal_routing_config.yml` stores the TS-first, LLM-fallback feature flags consumed by both `scripts/run_auto_trader.py` and `scripts/run_etl_pipeline.py`.
  - `logs/signals/quant_validation.jsonl` logs every scored signal (ticker, metrics, pass/fail) so brutal/dry-run invocations can surface quantitative guardrail breaches immediately after stage timings.

- Week 5.10: Demo-first broker frosting (Nov 12, 2025) ⭐ NEW
  - `execution/ctrader_client.py` and `execution/order_manager.py` replace the massive.com/polygon.io stub with a demo-ready cTrader Open API client that handles OAuth tokens, order placement, and lifecycle persistence while the order manager enforces the 2% per signal risk cap, daily trade limit, and risk-manager circuit breakers before submitting trades.
  - `config/ctrader_config.yml` documents the demo/live endpoints, risk thresholds, and gating rules.
  - New unit tests (`tests/execution/test_ctrader_client.py`, `tests/execution/test_order_manager.py`) cover configuration loading, order placement, and lifecycle gating, keeping the new broker stack regression-tested.
- Portfolio mathematics engine upgraded to institutional-grade metrics and optimisation (`etl/portfolio_math.py`)
- Signal validator aligned with 5-layer quantitative guardrails (statistical significance, Kelly sizing)
- Comprehensive error monitoring system with automated alerting ⭐ NEW
- Advanced LLM performance optimization and signal quality validation ⭐ NEW
- 200+ tests (100% passing) + enhanced risk/optimisation coverage + LLM integration tests + error monitoring tests


### ?? 2025-11-15 Brutal Run Regression (blocking)
- `logs/pipeline_run.log:16932-17729` and a direct `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` call reported `database disk image is malformed`, �rowid � out of order,� and �row � missing from index� for dozens of pages. Every OHLCV/forecast write in `etl/database_manager.py:689` and `:1213` is now rejected, so the project is running against a corrupted primary datastore.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, �` capture the same `ValueError: The truth value of a DatetimeIndex is ambiguous` after inserting ~90 SARIMAX/SAMOSSA/MSSA rows per ticker. The culprit is the `change_points = mssa_result.get('change_points') or []` branch inside `scripts/run_etl_pipeline.py:1755-1764`. Because the exception fires after the DB writes, the code logs �Saved forecast �� and then overwrites the ticker entry with an `error`, so downstream stages see �No valid forecast available.�
- Immediately after the failed stage, the visualization hook raises `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (`logs/pipeline_run.log:2626, 2981, �`), so the dashboard export promised in this document is also broken.
- The earlier hardening notes in `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md` (lines 9-24) claimed the DatetimeIndex ambiguity was resolved, yet the brutal log proves the regression remains. We also continue to emit pandas/statsmodels warnings because `forcester_ts/forecaster.py:128-136` still does a deprecated Period round-trip and `_select_best_order` in `forcester_ts/sarimax.py:136-183` keeps unconverged combinations in the search space. *(2025-11-18 update: the forecaster now stores frequency hints instead of forcing a `PeriodIndex`, rescales series before fitting, and limits the SARIMAX grid once repeated non-convergences appear; any residual warnings still land in `logs/warnings/warning_events.log` for review.)*
- `scripts/backfill_signal_validation.py:281-292` still calls `datetime.utcnow()` and relies on sqlite3�s default converters, producing the deprecation warnings documented in `logs/backfill_signal_validation.log:15-22`.
- **2025-11-19 remediation note**: Database corruption, MSSA change-point ambiguity, the Matplotlib `axis=` crash, and SARIMAX warning storms now have concrete fixes in `etl/database_manager.py`, `scripts/run_etl_pipeline.py`, `etl/visualizer.py`, and `forcester_ts/forecaster.py`/`forcester_ts/sarimax.py`. Synthetic/brutal pipelines write into `data/test_database.db` via `PORTFOLIO_DB_PATH`, keeping `data/portfolio_maximizer.db` reserved for production. The remaining gating item for lifting the global BLOCKED flag is modernising `scripts/backfill_signal_validation.py` per `Documentation/integration_fix_plan.md` and validating a fresh brutal run.
- `forcester_ts/instrumentation.py` (Nov 16) records fit/forecast telemetry, dataset snapshots (shape, missingness, frequency), ensemble weights, and benchmarking metrics (RMSE / sMAPE / tracking error). Configure `ensemble_kwargs.audit_log_dir` or set `TS_FORECAST_AUDIT_DIR` to emit JSON audits under `logs/forecast_audits/`, satisfying the interpretable-AI requirement from `AGENT_DEV_CHECKLIST.md`.

**Required actions before claiming production readiness**
1. Back up `data/portfolio_maximizer.db`, run `sqlite3 � ".recover"` (or start from a clean file), and teach `DatabaseManager._connect` to handle `"database disk image is malformed"` the same way we already handle `"disk i/o error"` (reset connection or operate off a POSIX mirror) so corruption is detected immediately.
2. Patch `scripts/run_etl_pipeline.py:1755-1764` to pull `change_points` once, detect `None`, and convert concrete iterables to `list` before serialising so pandas never gets coerced to `bool`. Re-run `python scripts/run_etl_pipeline.py --stage time_series_forecasting` to confirm AAPL/MSFT retain usable forecasts.
3. Remove the unsupported `axis=` argument when calling `FigureBase.autofmt_xdate()` (dashboard loader) and capture a fresh PNG to prove visualization works again.
4. Replace the PeriodDtype round-trip with `Series.asfreq()` (or a resample) inside `forcester_ts/forecaster.py`, tighten the SARIMAX order grid via config, and add regression tests so the FutureWarning/ConvergenceWarning spam ceases.
5. Update `scripts/backfill_signal_validation.py` to use timezone-aware timestamps (`datetime.now(datetime.UTC)`) and register sqlite adapters before the scheduled nightly job runs again.

### Config Inventory (Nov 2025; updated Dec 2025)
- `config/pipeline_config.yml` defines the default stage planner ordering that both the ETL pipeline and autonomous trader invoke.
- `config/forecasting_config.yml` tunes the SARIMAX/GARCH/SAMOSSA ensemble parameters consumed by `forcester_ts/forecaster.py`.
- `config/llm_config.yml` lists the Ollama models, latency guardrails, and token-throughput failover policies enforced by `ai_llm/ollama_client.py`.
- `config/signal_routing_config.yml` centralizes the TS-first, LLM-fallback feature flags shared by `scripts/run_etl_pipeline.py` and `scripts/run_auto_trader.py`.
- `config/quant_success_config.yml` encodes the per-signal thresholds (Sharpe/Sortino/drawdown, min_expected_profit) that gate Time Series signals before routing/execution and feed quant validation logs (`logs/signals/quant_validation.jsonl`).
- `config/forecaster_monitoring.yml` plus `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` define the GREEN/YELLOW/RED quant validation thresholds used by `scripts/summarize_quant_validation.py` (per-ticker) and `scripts/check_quant_validation_health.py` (global), including the hard RED CI gate at `max_fail_fraction=0.90`. Higher-order automation around these thresholds is tracked in `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md`.
- `config/ai_companion.yml` lists the Tier-1 knowledge base + dependency guardrails that automation launchers consume before invoking SARIMAX/SAMOSSA workloads.
- `config/ctrader_config.yml` captures demo/live endpoints plus per-signal risk caps for the cTrader order manager.

### Artifact & Log Inventory (Nov 2025; updated Dec 2025)
- `bash/comprehensive_brutal_test.sh` orchestrates the profit-critical, ETL, TS/LLM, and execution suites; the most recent run (2025-11-12) persisted its reports under `logs/brutal/` alongside the console summary. *(Set `BRUTAL_ENABLE_LLM=1` if you need the legacy LLM fallback stage�otherwise the script only exercises the forecaster stack.)*
- Recent brutal runs (for example `logs/brutal/results_20251204_190220/`) use TS-first defaults (`BRUTAL_ENABLE_LLM=0`, frontier stage without `--enable-llm`) and, when quant logs are present, call `scripts/check_quant_validation_health.py` to write the global GREEN/YELLOW/RED quant health classification into `final_report.md`.
- `logs/brutal/results_*/artifacts/portfolio_maximizer.db.bak`, `.bak-shm`, and `test_database.db` capture SQLite snapshots from brutal runs so validation teams can diff executions before enabling brokers.
- `logs/signals/quant_validation.jsonl` retains the quant-success audit rows that the brutal and dry-run scripts surface immediately after stage timings, keeping the TS-first + LLM-fallback contract observable.

### ⚠️ Validation Status (Nov 12, 2025)
- `bash/comprehensive_brutal_test.sh` execution summary:
  - Profit-critical functions, profit-factor, and LLM profit-report suites: ? PASS.
  - ETL suites (`test_data_storage`, `test_preprocessor`, `test_time_series_cv`, `test_data_source_manager`, `test_checkpoint_manager`): ? PASS (92 tests) but `tests/etl/test_data_validator.py` not found (needs restoration).
  - Time Series forecasting / router suites: ? NOT RUN � script timed out with `Broken pipe` output before those stages, so no TS-first regression coverage exists yet.
- `logs/errors/errors.log` (Nov 02–07) still reports blocking runtime issues: `DataStorage.train_validation_test_split()` TypeError (unexpected `test_size`), zero-fold CV `ZeroDivisionError`, SQLite `disk I/O error` during OHLCV persistence/migrations, and missing `pyarrow`/`fastparquet` to serialize checkpoints. These must be cleared before rerunning ETL or the autonomous loop on live data.

---

## IMMEDIATE PRIORITIES (WEEK 1-2)

### PHASE 5.1: COMPLETE MULTI-SOURCE DATA EXTRACTION
**Status**: ✅ FOUNDATION COMPLETE - Phase 4.6 implemented platform-agnostic architecture

#### **TASK 5.1.1: Implement Alpha Vantage Extractor**
```python
# etl/alpha_vantage_extractor.py - STUB READY FOR IMPLEMENTATION
# Current: ✅ 140-line stub (BaseExtractor pattern established)
# Priority: MEDIUM - Foundation ready, API implementation needed

class AlphaVantageExtractor(BaseExtractor):
    def extract_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Implement actual Alpha Vantage API integration"""
        # Use config from alpha_vantage_config.yml
        # API: 5 calls/min free, 75 calls/min premium
        # Transform to match existing OHLCV schema
        # Integrate with existing cache system
```

#### **TASK 5.1.2: Implement Finnhub Extractor**
```python
# etl/finnhub_extractor.py - STUB READY FOR IMPLEMENTATION
# Current: ✅ 145-line stub (BaseExtractor pattern established)
# Priority: MEDIUM - Foundation ready, API implementation needed

class FinnhubExtractor(BaseExtractor):
    def extract_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Implement actual Finnhub API integration"""
        # Use config from finnhub_config.yml
        # API: 60 calls/min free, 300 calls/min premium
        # Transform to match existing OHLCV schema
        # Integrate with existing cache system
```

#### **TASK 5.1.3: DataSourceManager - Production Ready**
```python
# etl/data_source_manager.py - ✅ COMPLETE (340 lines)
# Status: Strategy + Factory + Chain of Responsibility patterns implemented
# Phase 4.6: Multi-source orchestration with failover (18 tests passing)

def extract_ohlcv(self, tickers: List[str], start_date: str, end_date: str,
                 prefer_source: Optional[str] = None) -> pd.DataFrame:
    """Production-ready multi-source extraction with failover"""
    # ✅ Current: Dynamic extractor selection (yfinance active)
    # ✅ Failover: P(success) = 1 - ∏(1 - p_i) = 99.99% (3 sources)
    # Ready: Add Alpha Vantage/Finnhub when implementations complete
```

### PHASE 5.2: TICKER DISCOVERY SYSTEM INTEGRATION
**Status**: NEW - Leverage existing multi-source architecture

#### **TASK 5.2.1: Create Ticker Discovery Module**
```python
# NEW: etl/ticker_discovery/__init__.py
# Integrate with existing config architecture

- `etl/ticker_discovery/` -> NEW MODULE (Future Phase 5)
  - `etl/ticker_discovery/__init__.py`
  - `etl/ticker_discovery/base_ticker_loader.py` -> abstract class aligning with existing extractor interfaces
  - `etl/ticker_discovery/alpha_vantage_loader.py` -> bulk ticker downloads (Alpha Vantage CSV + cache)
  - `etl/ticker_discovery/ticker_validator.py` -> yfinance-powered validation service
  - `etl/ticker_discovery/ticker_universe.py` -> master list orchestration/persistence

#### **TASK 5.2.2: Alpha Vantage Bulk Ticker Loader**
```python
# NEW: etl/ticker_discovery/alpha_vantage_loader.py
# Use existing alpha_vantage_config.yml settings

class AlphaVantageTickerLoader(BaseTickerLoader):
    def download_listings(self) -> pd.DataFrame:
        """Download daily CSV of all US listings"""
        # URL: https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=YOUR_API_KEY
        # Returns: symbol, name, exchange, assetType, status
        # Integrate with existing cache (24h validity)
    
    def get_active_equities(self) -> List[str]:
        """Filter for active stocks/ETFs using existing validation patterns"""
```

#### **TASK 5.2.3: Ticker Validation Service**
```python
# NEW: etl/ticker_discovery/ticker_validator.py
# Leverage existing yfinance_extractor.py validation logic

class TickerValidator:
    def validate_ticker(self, symbol: str) -> bool:
        """Use existing yfinance infrastructure to validate tickers"""
        # Reuse cache-first strategy from yfinance_extractor
        # Use existing data validation patterns
        # Return True if ticker exists and has valid data
```

### PHASE 5.3: PORTFOLIO OPTIMIZER PIPELINE ENHANCEMENT
**Status**: ENHANCE EXISTING - Build on current ETL foundation

#### **TASK 5.3.1: Create Optimizer-Ready Pipeline**
```python
# ✅ ENHANCED: scripts/run_etl_pipeline.py (modular CVSettings + LLMComponents orchestrator)
# Add ticker discovery integration to existing pipeline

def run_optimizer_pipeline(ticker_source="manual", portfolio_size=50):
    """Enhanced pipeline with ticker discovery options"""
    # Option 1: Manual ticker list (existing behavior)
    # Option 2: Discover from Alpha Vantage universe
    # Option 3: Pre-validated portfolio candidates
    # Reuse existing: extraction → validation → preprocessing → storage
```

#### **TASK 5.3.2: Portfolio Selection Module**
```python
# NEW: etl/portfolio_selection.py
# Simple ranking for portfolio optimization foundation

class BasicPortfolioSelector:
    def select_from_universe(self, universe: List[str], count: int) -> List[str]:
        """Simple selection using existing data patterns"""
        # Use existing portfolio_math.py calculations
        # Rank by liquidity (volume * price)
        # Filter by data quality (existing validation)
```

## UPDATED DIRECTORY STRUCTURE

```
portfolio_maximizer_v45/
|-- config/                          # ? EXISTING - COMPLETE
|   |-- pipeline_config.yml          # ? 6.5 KB - Production ready
|   |-- data_sources_config.yml      # ? Multi-source configured
|   |-- yfinance_config.yml          # ? 2.6 KB - Production ready
|   |-- alpha_vantage_config.yml     # ? Configured - needs API keys
|   |-- finnhub_config.yml           # ? Configured - needs API keys
|   |-- preprocessing_config.yml     # ? 4.8 KB - Production ready
|   |-- validation_config.yml        # ? 7.7 KB - Production ready
|   |-- storage_config.yml           # ? 5.9 KB - Production ready
|   |-- analysis_config.yml          # ? MIT standards
|   `-- ai_companion.yml             # ? Tier-1 AI companion guardrails (knowledge base + stack metadata)
|-- etl/                             # ? PHASE 4.8 COMPLETE - 3,986 lines ? UPDATED
|   |-- base_extractor.py            # ? 280 lines - Abstract Factory (Phase 4.6)
|   |-- data_source_manager.py       # ? 340 lines - Multi-source orchestration (Phase 4.6)
|   |-- yfinance_extractor.py        # ? 498 lines - BaseExtractor impl (Phase 4.6)
|   |-- alpha_vantage_extractor.py   # ? 140-line stub - Ready for API impl
|   |-- finnhub_extractor.py         # ? 145-line stub - Ready for API impl
|   |-- data_validator.py            # ? 117 lines - Production ready
|   |-- preprocessor.py              # ? 101 lines - Production ready
|   |-- data_storage.py              # ? 210+ lines - Production ready (+CV/run metadata, Remote Sync 2025-11-06) ? UPDATED
|   |-- time_series_cv.py            # ? 336 lines - Production ready (5.5x coverage)
|   |-- checkpoint_manager.py        # ? 362 lines - State persistence (Phase 4.8) ? NEW
|   |-- pipeline_logger.py           # ? 415 lines - Event logging (Phase 4.8) ? NEW
|   |-- portfolio_math.py            # ? 45 lines - Production ready
|   |-- statistical_tests.py         # ? Statistical validation suite (Phase 5.6) ? NEW
|   |-- time_series_analyzer.py      # ? 500+ lines - Production ready
|   |-- visualizer.py                # ? 600+ lines - Production ready
|   `-- ticker_discovery/            # ? NEW MODULE (Future Phase 5)
|       |-- base_ticker_loader.py    # ? Create abstract class
|       |-- alpha_vantage_loader.py  # ? Bulk ticker downloads
|       |-- ticker_validator.py      # ? Validation service
|       `-- ticker_universe.py       # ? Master list management
|-- models/                          # ?? TIME SERIES SIGNAL GENERATION (Nov 6, 2025) - 800+ lines ? NEW - TESTING REQUIRED
|   |-- __init__.py                  # ?? Package exports - TESTING REQUIRED
|   |-- time_series_signal_generator.py # ?? 350 lines - Converts TS forecasts to trading signals (DEFAULT) - TESTING REQUIRED
|   |-- signal_router.py             # ?? 250 lines - Routes TS primary + LLM fallback - TESTING REQUIRED
|   `-- signal_adapter.py            # ?? 200 lines - Unified signal interface for backward compatibility - TESTING REQUIRED
|-- ai_llm/                          # ? PHASE 5.2-5.5 COMPLETE - 1,500+ lines ? UPDATED
|   |-- ollama_client.py             # ? 440+ lines - Local LLM integration (Phase 5.5) + fast-mode latency tuning (Phase 5.6)
|   |-- market_analyzer.py           # ? 180 lines - Market analysis (Phase 5.2)
|   |-- signal_generator.py          # ? 198 lines - Signal generation (Phase 5.2) + timestamp/backtest metadata (Phase 5.6) - NOW FALLBACK
|   |-- signal_validator.py          # ? 150 lines - Signal validation (Phase 5.2) + SSA diagnostics (Phase 5.6)
|   |-- risk_assessor.py             # ? 120 lines - Risk assessment (Phase 5.2)
|   |-- performance_monitor.py       # ? 208 lines - LLM performance monitoring (Phase 5.5) ? NEW
|   |-- signal_quality_validator.py  # ? 378 lines - 5-layer signal validation (Phase 5.5) ? NEW
|   |-- llm_database_integration.py  # ? 421 lines - LLM data persistence (Phase 5.5) ? NEW
|   `-- performance_optimizer.py     # ? 359 lines - Model selection optimization (Phase 5.5) ? NEW
|-- execution/                       # ? PHASE 5.6 - Paper trading + broker stack ? UPDATED
|   |-- __init__.py                  # ? Module marker + cTrader exports
|   |-- paper_trading_engine.py      # ? Realistic simulation & persistence (Phase 5.6)
|   |-- ctrader_client.py            # ? Demo-first cTrader Open API client (Phase 5.10)
|   `-- order_manager.py             # ? Lifecycle manager enforcing risk gates + persistence (Phase 5.10)
|-- .local_automation/
|   |-- developer_notes.md           # Automation playbook
|   `-- settings.local.json          # Tooling configuration
|-- scripts/                         # ? PHASE 4.7-5.5 COMPLETE - 1,200+ lines ? UPDATED
|   |-- run_etl_pipeline.py          # ? 1,900+ lines - Modular orchestrator (Remote Sync + TS Refactoring 2025-11-06) ? UPDATED
|   |-- backfill_signal_validation.py# ? Backfills pending signals & recomputes accuracy (Phase 5.6) ? NEW
|   |-- analyze_dataset.py           # ? 270+ lines - Production ready
|   |-- visualize_dataset.py         # ? 200+ lines - Production ready
|   |-- validate_environment.py      # ? Environment checks
|   |-- error_monitor.py             # ? 286 lines - Error monitoring system (Phase 5.5) ? NEW
|   |-- cache_manager.py             # ? 359 lines - Cache management system (Phase 5.5) ? NEW
|   |-- monitor_llm_system.py        # ? 418 lines - LLM system monitoring + latency/backtest reporting (Phase 5.6) ? NEW
|   |-- test_llm_implementations.py  # ? 150 lines - LLM implementation testing (Phase 5.5) ? NEW
|   |-- deploy_monitoring.sh         # ? 213 lines - Monitoring deployment script (Phase 5.5) ? NEW
|   `-- refresh_ticker_universe.py   # ? NEW - Weekly ticker updates
|-- schedule_backfill.bat            # ? Task Scheduler wrapper for nightly signal backfills (Phase 5.6)
|-- visualizations/                  # ? Context-rich dashboards (Phase 5.6) ? UPDATED
|   |-- Close_dashboard.png          # ? Legacy price dashboard
|   |-- Volume_dashboard.png         # ? Market conditions + commodities overlays (Phase 5.6)
|   `-- training/                    # ? Sample training-set plots
|-- bash/                            # ? PHASE 4.7 COMPLETE - Validation scripts ? UPDATED
|   |-- run_cv_validation.sh         # ? CV validation suite (5 tests + 88 unit tests)
|   |-- test_config_driven_cv.sh     # ? Config-driven demonstration
|   |-- run_pipeline_dry_run.sh      # ? Synthetic/no-network pipeline exerciser
|   `-- run_pipeline_live.sh         # ? Live/auto pipeline runner with stage summaries
|-- logs/                            # ? PHASE 4.8 - Event & activity logging (7-day retention) ? NEW
|   |-- pipeline.log                 # ? Main pipeline log (10MB rotation)
|   |-- events/events.log            # ? Structured JSON events (daily rotation)
|   |-- errors/errors.log            # ? Error log with stack traces
|   `-- stages/                      # Reserved for stage-specific logs
|-- data/
|   |-- checkpoints/                 # ? PHASE 4.8 - Pipeline checkpoints (7-day retention) ? NEW
|   |-- raw/                         # Raw extracted data + cache
|   |-- processed/                   # Cleaned and transformed data
|   |-- training/                    # Training set
|   |-- validation/                  # Validation set
|   `-- testing/                     # Test set
|-- tests/                           # ? PHASE 5.2-5.5 COMPLETE - 200+ tests ? UPDATED
|   |-- etl/                         # ? 121 tests - 100% passing
|   |   |-- test_checkpoint_manager.py
|   |   |-- test_data_source_manager.py
|   |   |-- test_time_series_cv.py
|   |   |-- test_method_signature_validation.py
|   |   |-- test_statistical_tests.py
|   |   `-- test_visualizer_dashboard.py
|   |-- ai_llm/                      # ? 50+ tests - 100% passing (Phase 5.2-5.5) ? UPDATED
|   |   |-- test_ollama_client.py
|   |   |-- test_market_analyzer.py
|   |   |-- test_signal_generator.py
|   |   |-- test_signal_validator.py
|   |   `-- test_llm_enhancements.py
|   `-- execution/                   # ? 4 tests - Paper trading + broker regression (Phase 5.6-5.10) ? NEW
|       |-- test_paper_trading_engine.py
|       |-- test_ctrader_client.py
|       `-- test_order_manager.py
`-- Documentation/                   # ? PHASE 4.8-5.6 - 25+ files ? UPDATED
    |-- implementation_checkpoint.md                # High-level status checkpoint
    |-- TIME_SERIES_FORECASTING_IMPLEMENTATION.md   # TS-first architecture details
    |-- QUANT_VALIDATION_MONITORING_POLICY.md       # GREEN/YELLOW/RED quant gates
    |-- QUANT_VALIDATION_AUTOMATION_TODO.md         # Quant automation & TS threshold sweeps
    |-- MTM_AND_LIQUIDATION_IMPLEMENTATION_PLAN.md  # MTM / liquidation roadmap (diagnostic)
    |-- NAV_RISK_BUDGET_ARCH.md                     # NAV-centric barbell architecture
    |-- NAV_BAR_BELL_TODO.md                        # Barbell/NAV integration TODOs
    |-- BARBELL_OPTIONS_MIGRATION.md                # Options/derivatives migration plan
    |-- RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md   # Research log & publication outline
    |-- REFACTORING_IMPLEMENTATION_COMPLETE.md
    |-- REFACTORING_STATUS.md
    |-- TESTING_IMPLEMENTATION_SUMMARY.md
    |-- INTEGRATION_TESTING_COMPLETE.md
    |-- CHECKPOINTING_AND_LOGGING.md
    |-- IMPLEMENTATION_SUMMARY_CHECKPOINTING.md
    |-- CV_CONFIGURATION_GUIDE.md
    |-- IMPLEMENTATION_SUMMARY.md
    |-- SYSTEM_ERROR_MONITORING_GUIDE.md
    |-- ERROR_FIXES_SUMMARY_2025-10-22.md
    |-- LLM_ENHANCEMENTS_IMPLEMENTATION_SUMMARY_2025-10-22.md
    |-- RECOMMENDED_ACTIONS_IMPLEMENTATION_SUMMARY_2025-10-22.md
    `-- [other docs...]
```


## INTEGRATION WITH EXISTING ARCHITECTURE

### Leverage Current Strengths (Phase 4.6 + 4.7):
- ✅ **Cache System**: 100% hit rate, 20x speedup - Reuse for ticker data
- ✅ **Validation**: Existing data_validator.py - Extend for ticker validation
- ✅ **Configuration**: 8 YAML files - Add ticker discovery settings
- ✅ **Cross-Validation**: k-fold CV (5.5x coverage) - Use for portfolio backtesting
- ✅ **Multi-source**: DataSourceManager (Phase 4.6) - Extend for ticker discovery
- ✅ **Platform-Agnostic**: BaseExtractor pattern - Consistent interface across sources
- ✅ **Config-Driven**: Zero hard-coded defaults - Full YAML + CLI control

### Heuristics, Full Models, and ML Calibration (Conceptual Stack)

- **Heuristics (fast, transparent)**:
  - Quant validation tiers (GREEN/YELLOW/RED) from `config/forecaster_monitoring.yml` + `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` + `scripts/check_quant_validation_health.py` / `scripts/summarize_quant_validation.py`.
  - MVS-style summaries from `DatabaseManager.get_performance_summary()` surfaced by `bash/run_pipeline_live.sh` / `bash/run_end_to_end.sh`.
  - Barbell quant gates and simple NAV guards in `risk/barbell_policy.py`, `Documentation/NAV_RISK_BUDGET_ARCH.md`, `Documentation/NAV_BAR_BELL_TODO.md`.
- **Full models (official NAV/risk)**:
  - TS ensemble (`forcester_ts/*`, `etl/time_series_forecaster.py`) + portfolio math (`etl/portfolio_math.py`) + auto trader / backtesting pipelines (`scripts/run_auto_trader.py`, `scripts/run_etl_pipeline.py`, `scripts/run_strategy_optimization.py`).
  - These are the source of truth for official NAV calculations, drawdown metrics, and PnL-based research conclusions.
- **ML calibrators (future, regime-aware)**:
  - Higher-order hyper-parameter and bandit-style search around thresholds (see `Documentation/STOCHASTIC_PNL_OPTIMIZATION.md`, `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md`).
  - Future ML components should learn **when to tighten/loosen heuristics** (e.g., regime-aware `min_expected_return` bands) based on full-model outcomes, not replace the TS ensemble or portfolio math directly.

### Build on Production Foundation:
```
Phase 4.6: Multi-Source Architecture (COMPLETE ✅)
├── BaseExtractor (280 lines) - Abstract Factory pattern
├── DataSourceManager (340 lines) - Multi-source orchestration
├── YFinanceExtractor (498 lines) - BaseExtractor implementation
├── Alpha Vantage stub (140 lines) - Ready for API
└── Finnhub stub (145 lines) - Ready for API

Phase 4.7: Configuration-Driven CV (COMPLETE ✅)
├── Pipeline config enhanced - Zero hard-coded defaults
├── CLI override system - 3-tier priority (CLI > Config > Defaults)
├── Bash validation scripts - 5 pipeline tests + 88 unit tests
└── Documentation - CV_CONFIGURATION_GUIDE.md (3.3 KB)

Phase 5 (NEXT): Complete Multi-Source + Ticker Discovery
    ↓
Implement Alpha Vantage/Finnhub API integration
    ↓
Integrate Ticker Discovery (NEW MODULE)
    ↓
Enhanced Portfolio Pipeline (OPTIMIZER READY)
```

## ACTION PLAN (2 WEEKS)

### WEEK 1: Complete Multi-Source Implementation
1. **Implement** `AlphaVantageExtractor` (use existing config)
2. **Implement** `FinnhubExtractor` (use existing config) 
3. **Test** multi-source fallback with `DataSourceManager`
4. **Update** tests for new extractors (leverage existing test patterns)

### WEEK 2: Ticker Discovery Integration
1. **Create** `etl/ticker_discovery/` module structure
2. **Implement** `AlphaVantageTickerLoader` for bulk downloads
3. **Build** `TickerValidator` using existing yfinance infrastructure
4. **Enhance** `run_etl_pipeline.py` with ticker discovery options
5. **Create** `refresh_ticker_universe.py` maintenance script

## SUCCESS CRITERIA

### Critical Requirements:
- [x] **ZERO** breaking changes to existing portfolio optimization ✅ Phase 4.6/4.7
- [x] **100%** cache performance maintained (20x speedup) ✅ Phase 4.6
- [x] **All 88 tests** continue passing (100% coverage) ✅ Phase 4.6/4.7
- [x] **Existing pipelines** unaffected (backward compatibility) ✅ Phase 4.7

### New Capabilities:
- [ ] Alpha Vantage data extraction operational
- [ ] Finnhub data extraction operational  
- [ ] Multi-source fallback working (3 sources)
- [ ] Ticker discovery from Alpha Vantage bulk data
- [ ] Automatic ticker validation with yfinance
- [ ] Portfolio-ready ticker universe management

### Quality Assurance:
- [ ] New tests for ticker discovery (85%+ coverage)
- [ ] Performance within 10% of baseline
- [ ] API rate limit compliance
- [ ] Error handling and graceful degradation

## CONFIGURATION READINESS

### Existing Config Files (READY):
- `data_sources_config.yml` - Already configured for multi-source
- `alpha_vantage_config.yml` - Structured, needs API key
- `finnhub_config.yml` - Structured, needs API key  
- `pipeline_config.yml` - Ready for ticker discovery integration

### API Key Integration:
```python
# .env - ADD NEW KEYS (maintain existing structure) # remove secret credential to dot environment 
ALPHA_VANTAGE_API_KEY='UFJ93EBWE29IE2RR'
FINNHUB_API_KEY='d3f4cb1r01qh40fgqdjgd3f4cb1r01qh40fgqdk0'
# Existing YFINANCE_API_KEY (if any) remains
```

## RISK MITIGATION

### Low Risk Implementation (Phase 4.6 Complete):
- ✅ **Stubs Exist**: alpha_vantage_extractor.py and finnhub_extractor.py (Phase 4.6)
- ✅ **Config Ready**: YAML files pre-configured for new sources (Phase 4.6)
- ✅ **Patterns Established**: BaseExtractor, DataSourceManager operational (Phase 4.6)
- ✅ **Tests Comprehensive**: 100+ tests provide safety net (Phase 4.6/4.7)
- ✅ **Architecture Complete**: Abstract Factory + Strategy patterns implemented (Phase 4.6)

### Rollback Safety (Production Safeguards):
- ✅ Existing `yfinance_extractor.py` remains primary source (tested, working)
- ✅ New sources are fallback only (failover pattern implemented)
- ✅ All changes in separate, optional modules (no breaking changes)
- ✅ Can disable multi-source and revert to yfinance-only easily (config toggle)
- ✅ Configuration-driven (zero code changes needed for source selection)

**STATUS**: ✅ PHASES 4.6 & 4.7 COMPLETE
- **Multi-source architecture**: Platform-agnostic foundation ready (Phase 4.6)
- **Configuration-driven CV**: Zero hard-coded defaults (Phase 4.7)
- **Test coverage**: 100+ tests, 100% passing (Phase 4.6/4.7)
- **Next phase**: API implementation for Alpha Vantage/Finnhub + ticker discovery




## Recent Additions (2025-11-22)
- Data quality gating + snapshots (data_quality_snapshots), quality-aware routing, and dashboard quality display.
- Latency telemetry per ticker (latency_metrics) from TS/LLM routing; averages surfaced on dashboards.
- Dashboard JSON + PNG emission from run_auto_trader.py with routing, equity, win-rate, quality, latency.
- New orchestration helpers: bash/run_auto_trader.sh, bash/run_end_to_end.sh, git_sync.sh for safe rebase/push.

- Split diagnostics: rolling-window CV default (k=5); split metadata logging with drift checks (PSI/mean/std); test isolation enforced and warned when overlap detected.

- Hotfix 2025-11-23: ETL data_storage UnboundLocalError resolved (removed nested pandas imports); ASCII-only logging enforced in DataSourceManager to prevent cp1252 errors; CV drift metrics persisted and dashboard JSON emission extended.
- Strategy optimization layer: etl/strategy_optimizer.py and scripts/run_strategy_optimization.py (config/strategy_optimization_config.yml) implement stochastic, regime-aware PnL tuning without hardcoded strategies; see Documentation/STOCHASTIC_PNL_OPTIMIZATION.md for design.
