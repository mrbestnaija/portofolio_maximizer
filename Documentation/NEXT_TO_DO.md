# UPDATED TO-DO LIST: Portfolio Maximizer v45 - ML Integration & Optimization

> **Current verified snapshot (2025-12-26)**: `Documentation/PROJECT_STATUS.md` (engineering unblocked; **paper-window MVS now PASS**, live/paper still gated).

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

## CURRENT PROJECT STATUS: 🟡 GATED (live/paper profitability evidence) – engineering unblocked
**Infrastructure in place**: ETL + Analysis + Visualization + Caching + k-fold CV + Multi-Source + Config-Driven + Checkpointing + LLM Integration  
**Recent Achievements**:

### ✅ 2025-12-26 Delta (MVS PASS – paper window replay)
- Added `scripts/run_mvs_paper_window.py` to replay stored OHLCV through `PaperTradingEngine` and rapidly generate ≥30 **realised** trades for MVS evaluation.
- Verified **MVS PASS** on a 365‑day paper window (AAPL/MSFT/GOOGL): 31 realised trades, +$15.18 total profit, 51.6% win rate, 1.28 profit factor. Report: `reports/mvs_paper_window_20251226_183023.md`.
- Exit safety: `SignalValidator` no longer blocks exposure-reducing exits (SELL closing a long / BUY covering a short), so liquidation/time exits are executable under guardrails.

### 🔄 2025-12-03 Delta (diagnostic mode + invariants)
- DIAGNOSTIC_MODE/TS/EXECUTION relax TS thresholds (confidence=0.10, min_return=0, max_risk=1.0, volatility filter off), disable quant validation, and allow PaperTradingEngine to size at least 1 share; LLM latency guard bypassed in diagnostics; `volume_ma_ratio` now guards zero/NaN volume.
- Numeric/scaling invariants and dashboard/quant health tests pass in `simpleTrader_env` (`tests/forcester_ts/test_ensemble_and_scaling_invariants.py`, `tests/forcester_ts/test_metrics_low_level.py`, dashboard payload + quant health scripts).
- Diagnostic reduced-universe run (MTN, SOL, GC=F, EURUSD=X; cycles=1; horizon=10; cap=$25k) executed 4 trades with PnL -0.06%, updated `visualizations/dashboard_data.json`; positions: long MTN 10, short SOL 569, short GC=F 1, short EURUSD=X 792; quant_validation fail_fraction 0.932 (<0.98) and negative_expected_profit_fraction 0.488 (<0.60).

### 🔄 2025-11-24 Delta (currency update)
- Data-source-aware ticker resolution added via `etl/data_universe.py` (wired into `scripts/run_auto_trader.py`); keeps explicit + frontier as default and allows provider discovery when empty inputs.
- LLM fallback now enabled by default on the auto-trader flag to improve routing redundancy without changing thresholds.
- Dashboard JSON emitter hardened (datetime → isoformat) to avoid serialization warnings during live runs.
- Barbell migration groundwork documented in `BARBELL_INTEGRATION_TODO.md`; future optimization tasks must treat long-vol/tail-hedge legs as convexity purchases and evaluate them with Sortino/Omega/CVaR + scenario analysis rather than Sharpe alone.
- Forecaster monitoring config (`config/forecaster_monitoring.yml`) added to centralise TS health thresholds (profit_factor, win_rate, RMSE) reused by brutal CLIs, hyperopt, and dashboards.
- `bash/run_post_eval.sh` and `scripts/run_strategy_optimization.py` now respect these thresholds so hyperopt only treats regimes as profitable when the TS ensemble passes PF/WR/RMSE checks.
 - Global barbell shell and policy scaffold added:
   - `config/barbell.yml` defines safe/risk buckets and feature flags (`enable_barbell_allocation`, `enable_barbell_validation`, `enable_antifragility_tests`).
   - `risk/barbell_policy.py` exposes `BarbellConstraint` for safe vs risk bucket calculations and optional projection into the barbell-feasible region (no-op while barbell allocation is disabled).
 - Options/barbell migration documented and partially wired:
   - `config/options_config.yml` remains the single source of truth for options feature flags and risk caps (`max_options_weight`, `max_premium_pct_nav`, allowed underlyings).
   - `Documentation/BARBELL_OPTIONS_MIGRATION.md` and `Documentation/BARBELL_INTEGRATION_TODO.md` updated to reflect the initial config/policy implementation and the feature-flag contract for future options/derivatives work.

### 🚨 2025-11-15 Brutal Run Findings (blocking)
- `logs/pipeline_run.log:16932-17729` and `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` confirmed the SQLite store is corrupted (`database disk image is malformed`, “rowid … out of order/missing from index”), so every persistence-dependent task in this list is blocked until the DB is rebuilt and `DatabaseManager._connect` handles this error like the existing disk-I/O branch.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` demonstrate Stage 7 failing on every ticker with `ValueError: The truth value of a DatetimeIndex is ambiguous` because `scripts/run_etl_pipeline.py:1755-1764` evaluates `mssa_result.get('change_points') or []`. As soon as ~90 forecast rows are inserted the stage logs “Generated forecasts for 0 ticker(s)”, so all downstream Time Series/LLM tasks stall.
- The visualization hook subsequently throws `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (lines 2626, 2981, …), meaning the dashboards mentioned in this file cannot be produced right now.
- Pandas/statsmodels warning spam remains (`forcester_ts/forecaster.py:128-136` Period round-trip; `_select_best_order` in `forcester_ts/sarimax.py:136-183` keeps unconverged grids) even though the Nov‑09 hardening note claimed otherwise.
- `scripts/backfill_signal_validation.py:281-292` still calls `datetime.utcnow()` with sqlite’s default converters, triggering the Python 3.12 deprecation warnings seen in `logs/backfill_signal_validation.log:15-22`.

**Blocking tasks that must precede every other TODO**
1. Recover/recreate `data/portfolio_maximizer.db` and update `DatabaseManager._connect` so `"database disk image is malformed"` leverages the reset/mirror logic.
2. Fix the MSSA `change_points` path (copy the `DatetimeIndex` to a list) and rerun `python scripts/run_etl_pipeline.py --stage time_series_forecasting` to rehydrate Stage 8.
3. Remove the unsupported `axis=` argument before invoking `FigureBase.autofmt_xdate()` so visualization checkpoints generate artefacts again.
4. Replace the deprecated Period coercion and tighten the SARIMAX grid to stop warning spam from drowning out actionable errors. *(Completed Nov 16; any residual warnings are now routed to `logs/warnings/warning_events.log` for auditing via `etl/warning_recorder.py`.)*
5. Update `scripts/backfill_signal_validation.py` to use timezone-aware timestamps and sqlite adapters prior to the nightly job described later in this document.
- ✅ **2025-11-16 status**: Items 1–4 shipped in `etl/database_manager.py`, `scripts/run_etl_pipeline.py`, `etl/visualizer.py`, `forcester_ts/forecaster.py`, and `forcester_ts/sarimax.py`. See `logs/pipeline_run.log:22237-22986` for the clean ETL execution. Item 5 (validator modernization) remains outstanding.
- ✅ **2025-11-16 telemetry**: `forcester_ts/instrumentation.py` now records dataset shape/statistics snapshots and per-model benchmarking metrics (RMSE, sMAPE, tracking error). Forecast runs automatically emit JSON audits under `logs/forecast_audits/` and the comprehensive dashboard prints the dataset summary inline.
- Phase 4.8: Checkpointing & Event Logging (Oct 7, 2025)
- Phase 5.1: Alpha Vantage & Finnhub APIs Complete (Oct 7, 2025)
- Phase 5.2: LLM Integration Complete (Ollama) (Oct 8, 2025)
- Phase 5.3: Profit Calculation Fix Applied (Oct 14, 2025) ⚠️ CRITICAL
- Monitoring deployment script validated end-to-end (Oct 23, 2025)
- Autonomous trading entry point (`scripts/run_auto_trader.py`) now chains extraction → validation → forecasting → signal routing → execution with optional LLM fallback, keeping cash/positions/trade history synchronized each cycle.
- README + roadmap reposition the platform as an **Autonomous Profit Engine** and document the new loop in Key Features, Quick Start, and infrastructure bullet lists.

### 🔄 2025-11-24 Options & Barbell Delta
- `config/options_config.yml` added as the **single source of truth** for options/derivative feature flags and barbell guardrails:
  - `options_trading.enabled`: master toggle (default: `false`).
  - `barbell.max_options_weight` and `barbell.max_premium_pct_nav`: caps for options exposure inside the risk sleeve.
  - `selection.moneyness` and `selection.expiry_days`: default seeds for OTM selection bands that higher-order hyperopt may refine.
- `Documentation/BARBELL_OPTIONS_MIGRATION.md` documents the migration plan from spot-only portfolios to options/derivatives under a barbell allocation, focusing on:
  - Clear definitions of OTM options and their role as convex risk-leg instruments.
  - Data model changes (options_quotes, underlyings, strikes, expiries) that must be implemented without altering existing ETL/auto-trader behaviour when options are disabled.
  - Phased rollout (O1–O4) with brutal-suite gates for each phase.
- Pipeline stage planner now forces Time Series forecasting/signal routing before any LLM work; LLM stages run strictly as fallback after the router.

**⚠️ STATUS ALERT**: Documentation is ahead of execution in some areas, but paper-trading evidence now exists (MVS PASS via paper-window replay). The remaining gate is sustained **live/paper** evidence (≥30 realised trades in real time) plus acceptable quant-validation health.

### 2025-11-12 Update
- ✅ `forcester_ts/` is the canonical forecasting stack; `TimeSeriesSignalGenerator` now handles GARCH volatility series safely and stamps HOLD provenance timestamps. Tests: `pytest tests/models/test_time_series_signal_generator.py -q` + targeted integration smoke.
- ✅ `scripts/monitor_llm_system.py` emits latency benchmarks + `llm_signal_backtests` summaries; JSON reports live under `logs/`.
- ✅ `schedule_backfill.bat` ships for nightly validator replays (Task Scheduler registration still required).
- ⚠️ deepseek-coder:6.7b latency remains 15–38 s (goal <5 s); explore prompt slimming, alternate models, or streaming fallback before enabling paper trading.
- ⚠️ SQLite occasionally returns `disk I/O error` when adding new `llm_signals` columns—rerun migration after closing DB handles; track in `SYSTEM_STATUS_2025-10-22.md`.
- 🚧 Paper trading & broker wiring deferred until latency + nightly jobs stay green for 48 h.
- ⚠️ ETL blockers found in `logs/errors/errors.log` (Nov 2–7):  
  - `DataStorage.train_validation_test_split()` raises `TypeError` when CV passes `test_size`.
  - `ZeroDivisionError` when CV returns zero folds (insufficient history).
  - `sqlite3.OperationalError: disk I/O error` during OHLCV persistence—database file or disk needs attention.
  - Missing parquet engine (`pyarrow`/`fastparquet`) prevents checkpoint serialization on Windows, halting extraction.
  - LLM migrations skipped because of the same disk issue; latency monitor still reports 15–39 s inference + sub-5 tokens/sec throughput.
  Immediate remediation is required before rerunning ETL or the autonomous loop against live data.
- 🧪 `bash/comprehensive_brutal_test.sh` (Nov 12) highlights current test status:
  - Profit-critical functions + profit-factor + report generation: ✅ PASS.
  - ETL suites (`test_data_storage`, `test_preprocessor`, `test_time_series_cv`, `test_data_source_manager`, `test_checkpoint_manager`): ✅ PASS (92 tests) but `tests/etl/test_data_validator.py` missing.
  - Time Series forecasting/LLM suites: ❌ NOT EXECUTED — script timed out with `Broken pipe` after ETL phase. Need to fix missing tests + timeout before promoting TS-first architecture.
- ✅ Live TS signals verified via `logs/ts_signal_demo.json`; the generator now normalises pandas payloads, captures decision context, and surfaces BUY/SELL output outside the test harness.
- ✅ `etl/checkpoint_manager.py` replaces metadata atomically (Path.replace) so Windows runs no longer fail with `[WinError 183]` when saving repeated checkpoints.
- ⚠️ `scripts/backfill_signal_validation.py` now bootstraps `sys.path` for CLI invocations; production still needs a scheduled task pointing at `schedule_backfill.bat`.

### Implementation Gaps (Nov 6, 2025 snapshot)
- ✅ LLM signal persistence now records `signal_type`, timestamps, and backtest metrics (`llm_signal_backtests` feeds dashboards).
- ✅ 5-layer validator + statistical diagnostics execute inside the pipeline with regression coverage.
- ✅ SAMOSSA + SARIMAX hybrid forecasts persist explained-variance diagnostics; RL/CUSUM promotion remains gated on profitability milestones.
- ✅ `TimeSeriesForecaster.evaluate()` now computes RMSE / sMAPE / tracking-error for every model + ensemble; metrics are written to SQLite so dashboards and the ensemble grid-search can rely on realised performance instead of static heuristics.
- ✅ Ensemble confidence scoring blends those metrics with AIC/EVR and variance-ratio tests, and MSSA-RL gains a CuPy-accelerated path (optional) plus change-point weighting so regime breaks are handled automatically.
- ✅ Time Series ensemble is the **default signal generator** (per `Documentation/REFACTORING_IMPLEMENTATION_COMPLETE.md` / `Documentation/REFACTORING_STATUS.md`): `models/time_series_signal_generator.py` + `signal_router.py` route TS output first, with LLM retained solely for fallback/redundancy.
- ⚠️ Paper trading engine, broker integration, stress testing, and regime detection remain outstanding.
- 🟡 Nightly validation wrapper `schedule_backfill.bat` is ready—needs Task Scheduler registration in production environments.

### Next Steps — Critical Gap Closure
1. ✅ Confirmed `etl/portfolio_math.py` (enhanced engine) is the promoted dependency. Verification logged via automated tests and `scripts/run_etl_pipeline.py` portfolio metrics output.
2. ✅ Repair the signal and risk persistence contract (Nov 5, 2025): `ai_llm/llm_database_integration.py` migrates `llm_risk_assessments` to accept `'extreme'` and new tests validate the schema (`python -m pytest tests/ai_llm/test_llm_enhancements.py::TestLLMDatabaseIntegration::test_risk_assessment_extreme_persisted`).
3. ✅ Persist validator telemetry via LLMSignalTracker (Nov 5, 2025): `scripts/run_etl_pipeline.py` now registers each signal/decision with `LLMSignalTracker`, and `python -m pytest tests/scripts/test_track_llm_signals.py` locks the wiring in CI.
4. ✅ Statistical scoring restored: `SignalValidator.backtest_signal_quality` now feeds bootstrap/Ljung–Box results into `llm_signal_backtests` and updates per-signal metrics.
5. ✅ Latency guard upgraded: `ai_llm/ollama_client.py` now auto-switches models when token throughput drops below the configured `token_rate_failover_threshold`; keep logging sub-5 s benchmarks via `logs/latency_benchmark.json` to confirm the guard holds.
6. ✅ Time-series stage now performs a rolling hold-out per ticker, calls `forecaster.evaluate(...)`, and stores RMSE/sMAPE/tracking-error in `time_series_forecasts.regression_metrics`; keep piping those fields into dashboards and reports.
7. ✅ Enforce the “Time Series first, LLM fallback” routing path (`config/signal_routing_config.yml`, `models/signal_router.py`) across every environment so the refactored architecture stays consistent with documentation.
8. 🟡 Register `schedule_backfill.bat` with Windows Task Scheduler (e.g. `schtasks /Create /TN PortfolioMaximizer_BackfillSignals /TR "\"C:\path\to\schedule_backfill.bat\"" /SC DAILY /ST 02:00 /F`) so nightly validation/backtests stay current.
9. 🚧 Promote paper trading engine + broker integration once monitoring and nightly jobs are confirmed stable (Phase A next major workstream).
10. ⚠️ Remediate ETL blockers logged Nov 2–7 (`DataStorage.train_validation_test_split()` TypeError, zero-fold CV `ZeroDivisionError`, SQLite `disk I/O error`, missing `pyarrow`/`fastparquet`) before attempting further live yfinance/Alpha Vantage/Finnhub runs.
11. 🆕 Validate `scripts/run_auto_trader.py` and hook it into the paper-vs-live adapters (XTB/cTrader) once profit KPIs clear the guardrails; execute the pending Time Series + signal-router tests (and re-run `bash/comprehensive_brutal_test.sh`) to certify the autonomous workflow without timeouts.

**Execution Guardrail**  
- Treat this as an autonomous blocker run using `Documentation/AGENT_DEV_CHECKLIST.md`, `Documentation/AGENT_INSTRUCTION.md`, and `Documentation/arch_tree.md` for navigation.  
- **Do not** proceed to downstream implementation (paper trading, broker integration, dashboards, Phase B ML) until the remaining yellow items are marked “Verified Complete” inside `Documentation/implementation_checkpoint.md`.

**📚 NEW**: 
- A comprehensive sequenced implementation plan has been created. See **`Documentation/SEQUENCED_IMPLEMENTATION_PLAN.md`** and **`Documentation/NEXT_TO_DO_SEQUENCED.md`** for the complete 12-week implementation plan with critical fixes prioritized first, then LLM operationalization (Phase A) and ML enhancement (Phase B).
- **Stub Implementation Review**: Complete review of all missing/incomplete implementations documented in **`Documentation/STUB_IMPLEMENTATION_PLAN.md`**. The cTrader client + order manager items are now delivered; remaining blockers include the performance dashboard, disaster recovery, production deployment scripts, and demo-to-live validation.
- **🟡 Time Series Signal Generation Refactoring IMPLEMENTED** (Nov 6, 2025) - **ROBUST TESTING REQUIRED**: See **`Documentation/REFACTORING_IMPLEMENTATION_COMPLETE.md`** for details. Time Series ensemble is now the DEFAULT signal generator with LLM as fallback. Includes 50 tests written (38 unit + 12 integration) - **NEEDS EXECUTION & VALIDATION**, unified database schema - **TESTING REQUIRED**, and complete pipeline integration - **TESTING REQUIRED**.

---

## 🚨 CRITICAL ARCHITECTURE UPDATE: ML-FIRST QUANTITATIVE APPROACH

### *Fundamental Correction Required*
The previous "ML optional" designation represents a **fundamental architectural flaw** that contradicts quantitative trading principles. ML must be the **core engine**, not decoration.

**CORRECTED DATA FLOW:**
```
DATA LAYER → ETL → FEATURE ENGINEERING → ML FORECASTING → QUANTITATIVE SIGNALS → PORTFOLIO OPTIMIZATION
                        ↑                                      ↓
                  Feature Importance                   Probabilistic Position Sizing
                        ↓                                      ↓
                  Model Interpretation                 Risk-Adjusted Allocation
```

---

## IMMEDIATE PRIORITIES (WEEK 1-2)

### PHASE 5.1: COMPLETE MULTI-SOURCE DATA EXTRACTION
**Status**: ✅ COMPLETE - All extractors implemented with production-grade features (2025-10-07)

#### **TASK 5.1.1: Implement Alpha Vantage Extractor** ✅ COMPLETE
```python
# etl/alpha_vantage_extractor.py - ✅ PRODUCTION READY (518 lines)
# Features: Full API integration, rate limiting, cache strategy
```

#### **TASK 5.1.2: Implement Finnhub Extractor** ✅ COMPLETE
```python
# etl/finnhub_extractor.py - ✅ PRODUCTION READY (532 lines)
# Features: Full API integration, Unix timestamp handling, production error handling
```

---

## 🎯 NEW: PHASE 6 - QUANTITATIVE ML INTEGRATION (CORE ENGINE)

### PHASE 6.1: ML FORECASTING PIPELINE
**Status**: NEW - Core quantitative prediction engine

#### **TASK 6.1.1: Create Quantitative Forecasting Pipeline**
```python
# NEW: ml/forecasting/quantitative_forecaster.py
# CORE ENGINE: ML-driven price prediction

class QuantitativeForecastingPipeline:
    """
    Production ML pipeline for quantitative price prediction
    Multi-horizon, multi-model ensemble approach
    """
    
    def create_forecasting_targets(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create multi-horizon, risk-adjusted targets for ML training"""
        horizons = [1, 5, 21, 63]  # 1day, 1week, 1month, 1quarter
        
        targets = {}
        for horizon in horizons:
            # Forward returns (primary target)
            targets[f'return_{horizon}d'] = prices.pct_change(horizon).shift(-horizon)
            
            # Risk-adjusted targets
            targets[f'sharpe_{horizon}d'] = (
                targets[f'return_{horizon}d'] / prices.rolling(horizon).std()
            )
            
            # Binary classification: significant moves
            targets[f'signal_{horizon}d'] = (
                targets[f'return_{horizon}d'].abs() > prices.rolling(63).std()
            ).astype(int)
            
        return pd.DataFrame(targets)
```

#### **TASK 6.1.2: Feature Engineering for Quantitative Prediction**
```python
# NEW: ml/features/quantitative_feature_engine.py
# Technical, statistical, and regime features for ML

class QuantitativeFeatureEngine:
    """
    Features specifically designed for price forecasting
    Builds on existing ETL foundation
    """
    
    def create_forecasting_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Technical, statistical, and regime features for ML"""
        
        # Price-based features
        features = {}
        
        # Momentum and trend (for direction prediction)
        features['momentum_1_20'] = ohlcv['close'] / ohlcv['close'].shift(20) - 1
        features['trend_strength'] = self.calculate_adx(ohlcv, period=14)
        
        # Mean reversion signals
        features['bollinger_position'] = (
            (ohlcv['close'] - ohlcv['close'].rolling(20).mean()) / 
            (2 * ohlcv['close'].rolling(20).std())
        )
        
        # Volatility regime features
        features['volatility_ratio'] = (
            ohlcv['close'].rolling(10).std() / 
            ohlcv['close'].rolling(63).std()
        )
        
        # Statistical features
        features['hurst_exponent'] = self.rolling_hurst(ohlcv['close'], window=100)
        features['variance_ratio'] = self.variance_ratio_test(ohlcv['close'], periods=[2, 5, 10])
        
        return pd.DataFrame(features).dropna()
```

#### **TASK 6.1.3: Multi-Model Ensemble Training**
```python
# NEW: ml/models/ensemble_trainer.py
# Robust ensemble for quantitative forecasting

class QuantitativeEnsembleTrainer:
    """
    Multi-model ensemble for robust forecasting
    Walk-forward validation for time series
    """
    
    def train_quantitative_ensemble(self, features: pd.DataFrame, targets: pd.DataFrame):
        """Multi-model ensemble with performance-based weighting"""
        models = {
            'lstm_temporal': TemporalConvNet(lookback=60, features=features.shape[1]),
            'xgboost_features': XGBRegressor(n_estimators=1000, max_depth=8),
            'linear_robust': BayesianRidge(),  # For baseline and uncertainty
            'regime_adaptive': RegimeAdaptiveModel(regime_model=GaussianHMM(n_components=4))
        }
        
        # Walk-forward validation for time series
        cv_scores = self.timeseries_cross_validate(models, features, targets)
        
        return EnsembleModel(models, weighting='performance_based')
```

### PHASE 6.2: ML-DRIVEN STRATEGY ENGINE
**Status**: NEW - Core signal generation replacing rule-based approach

#### **TASK 6.2.1: ML Strategy Engine**
```python
# NEW: trading/ml_strategy_engine.py
# CORE: ML predictions drive ALL trading decisions

class MLDrivenStrategyEngine:
    """
    Quantitative strategy engine with ML as core signal generator
    Replaces rule-based approach with data-driven forecasting
    """
    
    def __init__(self):
        self.forecast_models = {
            'short_term': LSTMForecaster(lookback=20, horizon=5),
            'medium_term': XGBoostForecaster(features=50, horizon=21),
            'regime_detection': HMMRegimeClassifier(states=4)
        }
        
    def generate_quantitative_signals(self, features: pd.DataFrame) -> Dict:
        """ML predictions drive ALL trading decisions"""
        # Ensemble forecasts with uncertainty
        returns_forecast = self.ensemble_forecast(features)
        regime_probabilities = self.detect_market_regime(features)
        confidence_intervals = self.calculate_prediction_intervals(features)
        
        return {
            'expected_returns': returns_forecast,
            'regime_probabilities': regime_probabilities,
            'forecast_confidence': confidence_intervals,
            'position_sizes': self.kelly_position_sizing(returns_forecast, confidence_intervals)
        }
```

#### **TASK 6.2.2: ML-Optimized Barbell Strategy**
```python
# NEW: strategies/ml_barbell_optimizer.py
# Quantitative Barbell optimization using ML forecasts

class MLBarbellOptimizer:
    """
    ML-optimized Barbell strategy with dynamic allocation
    Safe sleeve: ML-driven bond duration timing
    Risky sleeve: ML-driven leverage and selection
    """
    
    def optimize_barbell_allocation(self, ml_signals: Dict, current_portfolio: Portfolio) -> Allocation:
        # Safe sleeve: ML-driven bond duration timing
        safe_allocation = self.optimize_safe_sleeve(
            ml_signals['rate_forecasts'], 
            ml_signals['inflation_expectations']
        )
        
        # Risky sleeve: ML-driven leverage and selection
        risky_allocation = self.optimize_risky_sleeve(
            ml_signals['expected_returns'],
            ml_signals['regime_probabilities'],
            ml_signals['covariance_forecast']
        )
        
        # Dynamic allocation based on regime confidence
        barbell_ratio = self.calculate_optimal_barbell_ratio(
            ml_signals['regime_confidence'],
            ml_signals['market_volatility']
        )
        
        return Allocation(safe_allocation, risky_allocation, barbell_ratio)
```

### PHASE 6.3: QUANTITATIVE RISK MANAGEMENT
**Status**: NEW - ML-aware risk management

#### **TASK 6.3.1: Model Risk Management**
```python
# NEW: risk/model_risk_manager.py
# Monitor and manage ML model risks in production

class ModelRiskManager:
    """
    Monitor and manage ML model risks in production
    Ensures quantitative strategy robustness
    """
    
    def monitor_forecast_decay(self, predictions: pd.DataFrame, actuals: pd.Series):
        """Detect when models stop working - critical for production"""
        forecast_errors = np.abs(predictions - actuals)
        rolling_accuracy = 1 - forecast_errors.rolling(63).mean()
        
        # Alert if accuracy drops below threshold
        if rolling_accuracy.iloc[-1] < 0.55:  # 55% accuracy threshold
            self.trigger_model_retraining()
    
    def validate_signal_persistence(self, signals: pd.DataFrame) -> bool:
        """Ensure signals have reasonable persistence - prevent over-trading"""
        signal_changes = signals.diff().abs().sum()
        if signal_changes > len(signals) * 0.8:  # Too many changes
            return False
        return True
```

#### **TASK 6.3.2: Quantitative Backtesting**
```python
# NEW: backtesting/quantitative_backtester.py
# ML-aware backtesting with proper strategy evaluation

class QuantitativeBacktester:
    """
    ML-aware backtesting with proper strategy evaluation
    Uses ML predictions for position sizing and validation
    """
    
    def backtest_ml_strategy(self, ml_predictions: pd.DataFrame, 
                           prices: pd.DataFrame, 
                           transaction_costs: float = 0.001):
        """Proper backtesting for quantitative strategies"""
        
        # Use ML predictions for position sizing
        positions = self.ml_to_positions(ml_predictions)
        
        # Calculate returns with costs
        strategy_returns = positions.shift(1) * prices.pct_change() - (
            positions.diff().abs() * transaction_costs
        )
        
        # Risk-adjusted performance metrics
        performance = {
            'sharpe_ratio': self.calculate_sharpe(strategy_returns),
            'max_drawdown': self.calculate_max_drawdown(strategy_returns),
            'information_ratio': self.calculate_information_ratio(strategy_returns, prices),
            'hit_rate': self.calculate_hit_rate(ml_predictions, prices),
            'profit_factor': self.calculate_profit_factor(strategy_returns)
        }
        
        return performance
```

---

## UPDATED DIRECTORY STRUCTURE WITH ML INTEGRATION

```
portfolio_maximizer_v45/
├── config/                          # ✅ EXISTING - COMPLETE
│   ├── pipeline_config.yml          # ✅ 6.5 KB - Production ready
│   ├── ml_pipeline_config.yml       # ⬜ NEW - ML pipeline configuration
│   └── [other config files...]      # ✅ Existing configs
│
├── etl/                             # ✅ PHASE 5.1 COMPLETE - 4,259 lines
│   ├── [existing ETL modules...]    # ✅ All production ready
│   └── advanced_analysis/           # ⭐ ENHANCED for ML features
│       ├── feature_engineer.py      # ⬜ Enhanced for ML features
│       └── [other analysis modules...]
│
├── ml/                              # ⭐ NEW ML MODULE (CORE ENGINE)
│   ├── __init__.py
│   ├── forecasting/                 # Quantitative prediction
│   │   ├── quantitative_forecaster.py     # ⬜ Core forecasting pipeline
│   │   ├── multi_horizon_predictor.py     # ⬜ Multi-timeframe predictions
│   │   └── ensemble_model.py              # ⬜ Model combination
│   │
│   ├── features/                    # Feature engineering for ML
│   │   ├── quantitative_feature_engine.py # ⬜ ML-specific features
│   │   ├── technical_feature_generator.py # ⬜ Technical indicators
│   │   └── regime_feature_detector.py     # ⬜ Market regime features
│   │
│   ├── models/                      # ML model implementations
│   │   ├── ensemble_trainer.py            # ⬜ Multi-model training
│   │   ├── temporal_conv_net.py           # ⬜ LSTM/TCN for time series
│   │   ├── xgboost_forecaster.py          # ⬜ Tree-based models
│   │   └── regime_adaptive_model.py       # ⬜ Market regime adaptation
│   │
│   └── validation/                  # ML model validation
│       ├── walk_forward_validator.py      # ⬜ Time series CV
│       ├── model_performance_tracker.py   # ⬜ Production monitoring
│       └── feature_importance_analyzer.py # ⬜ Model interpretation
│
├── trading/                         # ⭐ ENHANCED TRADING MODULE
│   ├── ml_strategy_engine.py        # ⬜ NEW - ML-driven strategy engine
│   ├── quantitative_signal_generator.py   # ⬜ ML signal generation
│   └── [existing trading modules...]      # ✅ Maintain existing
│
├── strategies/                      # ⭐ ENHANCED STRATEGIES
│   ├── ml_barbell_optimizer.py      # ⬜ NEW - ML-optimized Barbell
│   ├── regime_aware_allocator.py    # ⬜ NEW - Dynamic allocation
│   └── [existing strategies...]            # ✅ Maintain existing
│
├── risk/                            # ⭐ ENHANCED RISK MANAGEMENT
│   ├── model_risk_manager.py        # ⬜ NEW - ML model risk
│   ├── probabilistic_position_sizer.py    # ⬜ Kelly-based sizing
│   └── [existing risk modules...]          # ✅ Maintain existing
│
├── backtesting/                     # ⭐ ENHANCED BACKTESTING
│   ├── quantitative_backtester.py   # ⬜ NEW - ML-aware backtesting
│   ├── strategy_evaluator.py        # ⬜ NEW - Performance attribution
│   └── [existing backtesting...]           # ✅ Maintain existing
│
└── scripts/                         # ⭐ ENHANCED SCRIPTS
    ├── run_ml_pipeline.py           # ⬜ NEW - ML training pipeline
    ├── generate_ml_signals.py       # ⬜ NEW - Daily signal generation
    ├── monitor_model_performance.py # ⬜ NEW - Model health monitoring
    └── [existing scripts...]               # ✅ Maintain existing
```

---

## QUANTITATIVE SUCCESS CRITERIA

### *ML-First Performance Targets*
```python
QUANTITATIVE_SUCCESS_CRITERIA = {
    'forecast_accuracy': '> 55% directional accuracy across horizons',
    'risk_adjusted_returns': 'Sharpe ratio > 1.2 in backtesting',
    'strategy_capacity': '> $10M without significant decay',
    'model_stability': '< 5% performance variance across market regimes',
    'feature_importance': 'Economically interpretable feature weights',
    'max_drawdown': '< 15% in stress periods',
    'hit_rate': '> 52% for binary classification signals'
}
```

### *Continuous Improvement Cycle*
```
Model Prediction → Strategy Execution → Performance Analysis → Feature Refinement → Model Retraining
        ↑                                                                               ↓
   Real-time Signals                                                          Walk-Forward Validation
```

---

## IMPLEMENTATION ROADMAP (12 WEEKS)

### *Phase 1: Core ML Foundation (Weeks 1-4)*
1. **Quantitative Forecasting Pipeline** 
   - Feature engineering for price prediction
   - Multi-horizon target creation
   - Ensemble model development

2. **ML Infrastructure**
   - Walk-forward validation framework
   - Model performance tracking
   - Feature importance analysis

### *Phase 2: ML-Driven Strategy (Weeks 5-8)*
3. **Quantitative Signal Generation**
   - Probabilistic position sizing (Kelly criterion)
   - Regime-aware signal adjustment
   - Forecast combination methods

4. **ML-Optimized Barbell**
   - Dynamic safe sleeve optimization
   - Regime-based risky sleeve leverage
   - Risk-parity position sizing

### *Phase 3: Production Integration (Weeks 9-12)*
5. **Risk Management & Monitoring**
   - Model performance tracking
   - Strategy capacity analysis
   - Production deployment with fail-safes

6. **Performance Optimization**
   - Latency optimization for real-time signals
   - Model compression for production
   - Automated retraining pipelines

---

## RISK MITIGATION & BACKWARD COMPATIBILITY

### *Critical Safeguards:*
- ✅ **Existing ETL pipeline remains unchanged** - ML is additive
- ✅ **Rule-based strategies remain operational** - Fallback option
- ✅ Focused verification suite passes (124 tests; see `Documentation/PROJECT_STATUS.md`)
- ✅ **Configuration-driven ML deployment** - Can disable via config
- ✅ **Gradual rollout capability** - Start with paper trading

### *Model Risk Controls:*
```python
# ml/validation/model_risk_controls.py
MODEL_RISK_CONTROLS = {
    'max_position_size': 0.1,  # 10% per position
    'minimum_forecast_confidence': 0.55,
    'maximum_drawdown_trigger': 0.15,
    'model_retraining_frequency': 'weekly',
    'emergency_stop_accuracy': 0.45  # Stop if accuracy drops below 45%
}
```

## ðŸš€ PHASE 7: TIME-SERIES MODEL UPGRADE ROADMAP

### **Time-Series Model Upgrade Overview**
- Run SAMOSSA, SARIMAX, and GARCH forecasts in parallel with existing LLM signals.
- Maintain backward compatibility by routing through `signal_router.py` with feature flags and unchanged downstream interfaces.
- Evaluate models with rolling cross-validation, walk-forward tests, and loss-focused metrics (profit factor, drawdown).
- Promote the most consistent performer to default only after statistically significant, loss-reducing outperformance; keep LLM as fallback.
- Expand monitoring so regressions trigger automatic reversion to the current LLM-only configuration.

### **Week 7: SAMOSSA + SARIMAX Foundations**
```python
# TASK 7.1: Time-Series Feature Engineering Upgrade (Days 43-45)  # ⏳ Pending
# File: etl/time_series_feature_builder.py (NEW - 250 lines)
class TimeSeriesFeatureBuilder:
    def build_features(self, price_history: pd.DataFrame) -> pd.DataFrame:
        """Create lag, seasonal, and volatility features for SAMOSSA/SARIMAX"""
        # Seasonal decomposition and holiday effects
        # Rolling statistics and differencing
        # Persist outputs via database_manager.py feature store helpers

# TASK 7.2: SAMOSSA Forecaster (Days 46-47)  # ✅ Delivered 2025-11-05 (etl/time_series_forecaster.py::SAMOSSAForecaster)
# Notes: SSA Page matrix, ≥90% energy capture, residual ARIMA fallback, hooks for CUSUM per SAMOSSA_algorithm_description.md
class SAMOSSAForecaster:
    def fit(self, features: pd.DataFrame) -> None:
        """Train Seasonal Adaptive Multi-Order Smoothing model"""
        # Extend model registry for checkpoint storage
        # Configurable seasonality plus adaptive smoothing parameters

    def forecast(self, horizon: int) -> ForecastResult:
        """Return price forecasts with confidence intervals"""
        # Output matches existing signal schema for parity with LLM signals

# TASK 7.3: SARIMAX Pipeline Refresh (Days 48-49)  # ✅ Delivered (see etl/time_series_forecaster.py::SARIMAXForecaster)
class SARIMAXForecaster:
    def fit_and_forecast(self, market_data: pd.DataFrame) -> ForecastResult:
        """Production-ready SARIMAX with automated tuning"""
        # Use pmdarima auto_arima with guardrails
        # Consume cached exogenous features from ETL pipeline
        # Persist diagnostics for regime-aware switching
```

### **Week 8: GARCH + Parallel Inference Integration**
```python
# TASK 7.4: GARCH Volatility Engine (Days 50-52)  # ✅ Delivered (etl/time_series_forecaster.py::GARCHForecaster)
# Notes: Supports GARCH/EGARCH/GJR-GARCH, surfaces AIC/BIC + volatility horizons
class GARCHVolatilityEngine:
    def fit(self, returns: pd.Series) -> None:
        """Estimate volatility clusters via GARCH(p, q)"""
        # Use arch package for variance forecasts
        # Emit risk-adjusted metrics for signal fusion

    def forecast_volatility(self, steps: int = 1) -> VolatilityResult:
        """Expose volatility forecast to risk sizing logic"""
        # Integrate with portfolio_math_enhanced.py metrics

# TASK 7.5: Parallel Model Runner (Days 53-54)  # ⏳ Pending (extend to async orchestration + provenance logging)
# File: models/time_series_runner.py (NEW - 260 lines)
class TimeSeriesRunner:
    def run_all(self, context: MarketContext) -> List[Signal]:
        """Execute SAMOSSA, SARIMAX, GARCH, and LLM pipelines in parallel"""
        # Async execution via existing task orchestrator
        # Normalize outputs into unified signal schema
        # Attach provenance metadata for performance dashboards

# TASK 7.6: Backward-Compatible Signal Routing (Days 55-56)  # ⏳ Pending
# File: signal_router.py (UPDATE - 180 lines)
class SignalRouter:
    def route(self, signals: List[Signal]) -> SignalBundle:
        """Merge legacy LLM signals with new time-series models"""
        # Feature flag toggles for gradual rollout
        # Priority ordering based on confidence and risk score
        # Downstream consumers see unchanged interface
```

> **SAMOSSA Implementation Status (Nov 05, 2025)**  
> • SSA decomposition + residual ARIMA forecasting is live inside `etl/time_series_forecaster.py`, emitting explained-variance diagnostics and confidence bands.  
> • CUSUM change-point scoring, Q-learning intervention policies, and GPU/CuPy optimisation remain on the backlog until paper-trading metrics satisfy `QUANTIFIABLE_SUCCESS_CRITERIA.md` gates.  
> • Monitoring tasks must ingest SAMOSSA diagnostics before toggling feature flags for downstream consumers.

### **Week 9: Cross-Validation + Evaluation Framework**
```python
# TASK 7.7: Rolling Cross-Validation Harness (Days 57-59)
# File: analysis/time_series_validation.py (NEW - 300 lines)
class TimeSeriesValidation:
    def evaluate(self, models: List[BaseModel]) -> ValidationReport:
        """Blocked CV and walk-forward evaluation across horizons"""
        # Profit factor, max drawdown, hit rate, volatility-adjusted return
        # Statistical tests (Diebold-Mariano, paired t-tests)
        # Persist results for dashboards and CI

# TASK 7.8: Performance Dashboard Extension (Days 60-61)
# File: monitoring/performance_dashboard.py (UPDATE - 200 lines)
class PerformanceDashboard:
    def render_time_series_tab(self, reports: List[ValidationReport]) -> Dashboard:
        """Compare LLM vs SAMOSSA/SARIMAX/GARCH performance"""
        # Rolling metrics, drawdown curves, loss distribution
        # Highlight leading model per asset and regime

# TASK 7.9: Risk & Compliance Review (Days 62-63)
# File: risk/model_governance.py (NEW - 150 lines)
class ModelGovernance:
    def certify(self, report: ValidationReport) -> GovernanceDecision:
        """Ensure new models satisfy risk limits before promotion"""
        # Enforce drawdown, VaR/ES thresholds, and audit logs
        # Document fallback procedures and approvals
```

### **Week 10: Promotion, Fallback, and Automation**
```python
# TASK 7.10: Dynamic Model Selection Logic (Days 64-66)
# File: models/model_selector.py (NEW - 220 lines)
class ModelSelector:
    def choose_primary_model(self, reports: List[ValidationReport]) -> ModelDecision:
        """Promote the most consistent, low-loss model"""
        # Weighted scoring (profit factor, drawdown, Sharpe, stability)
        # Minimum uplift thresholds relative to LLM baseline
        # Automatic reversion when performance degrades

# TASK 7.11: Integration Tests and Regression Suite (Days 67-68)
# File: tests/test_time_series_models.py (NEW - 280 lines)
class TestTimeSeriesModels(unittest.TestCase):
    def test_parallel_pipeline_integrity(self) -> None:
        """Validate routing, fallback, and metrics instrumentation"""
        # Simulate feature-flag toggles and failure scenarios

# TASK 7.12: Deployment Playbook Update (Days 69-70)
# File: docs/deployment_playbook.mdc (UPDATE - 120 lines)
def document_time_series_rollout():
    """Update runbooks for rolling out the new default model"""
    # Include rollback checklist, monitoring KPIs, approval steps
```

### **Phase 7 Success Criteria**
- [ ] SAMOSSA, SARIMAX, and GARCH pipelines deliver signals alongside LLM output
- [ ] Feature flags enable immediate fallback to the current LLM-only routing
- [ ] Rolling CV shows >= 3% profit-factor uplift with <= 50% drawdown increase versus baseline
- [ ] Governance sign-off documented with a tested fallback plan
- [ ] Dynamic selector promotes the top model automatically and regression suite passes

## QUANTITATIVE ML INTEGRATION BENEFITS

### *Enhanced Capabilities:*
1. **Predictive Power**: ML forecasts vs. lagging indicators
2. **Regime Adaptation**: Dynamic strategy adjustment to market conditions
3. **Risk Management**: Probabilistic position sizing with uncertainty
4. **Feature Discovery**: ML identifies non-obvious predictive patterns
5. **Continuous Improvement**: Automated model retraining and refinement

### *Performance Expectations:*
- **55%+** directional forecast accuracy (vs. 50% random)
- **1.2+** Sharpe ratio in backtesting
- **<15%** maximum drawdown in stress periods
- **Adaptive** to different market regimes (bull/bear/sideways)

**STATUS**: ✅ PHASES 4.6 & 4.7 COMPLETE | 🎯 PHASE 6 READY FOR IMPLEMENTATION
- **Production Foundation**: ETL + Multi-source + Configuration-driven
- **ML Integration**: Core quantitative engine replacing optional approach
- **Risk Management**: Comprehensive model risk controls
- **Backward Compatibility**: All existing functionality preserved
```

## Key ML Integration Improvements:

### 🚨 **Critical Architecture Correction**
- ML moved from "optional" to **core engine** 
- Quantitative forecasting drives ALL trading decisions
- Rule-based approaches become fallback only

### 🎯 **Quantitative ML Pipeline**
1. **Multi-horizon forecasting** (1d, 1w, 1m, 1q)
2. **Ensemble models** (LSTM, XGBoost, Bayesian, Regime-adaptive)
3. **Walk-forward validation** for time series
4. **Feature importance** for model interpretation

### 🔧 **Production-Ready ML Infrastructure**
- Model risk management with performance monitoring
- Automated retraining pipelines
- Probabilistic position sizing (Kelly criterion)
- Comprehensive backtesting with transaction costs

### ⚡ **Backward Compatibility**
- Existing ETL pipeline remains **unchanged**
- Rule-based strategies remain **operational** 
- Focused verification suite passes (124 tests; see `Documentation/PROJECT_STATUS.md`)
- Configuration-driven deployment

The updated architecture makes ML the **central quantitative engine** while maintaining all existing production capabilities and adding sophisticated forecasting, risk management, and adaptive strategy optimization.

