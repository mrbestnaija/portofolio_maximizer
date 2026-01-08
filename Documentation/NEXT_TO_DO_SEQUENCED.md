# 🎯 SEQUENCED TO-DO LIST: Portfolio Maximizer v45

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

**Production-Ready ML Trading System - Critical Fixes First**

**Date**: October 19, 2025  
**Status**: Ready for Implementation  
**Priority**: **CRITICAL** - Based on log analysis and system requirements

**Note (2026-01)**: For the current, focused sequencing to fix TS ↔ execution ↔ reporting issues observed in live/paper runs (bar-aware loop, horizon consistency, confidence/diagnostics, quant gating, cost model alignment), follow `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md`.

---

## 🔄 2025-12-03 Delta (diagnostic mode + invariants)
- DIAGNOSTIC_MODE/TS/EXECUTION relax TS thresholds (confidence=0.10, min_return=0, max_risk=1.0, volatility filter off), disable quant validation, and allow PaperTradingEngine to size at least 1 share; LLM latency guard bypassed in diagnostics; `volume_ma_ratio` now guards zero/NaN volume.
- Numeric/scaling invariants and dashboard/quant health tests pass in `simpleTrader_env` (`tests/forcester_ts/test_ensemble_and_scaling_invariants.py`, `tests/forcester_ts/test_metrics_low_level.py`, dashboard payload + quant health scripts).
- Diagnostic reduced-universe run (MTN, SOL, GC=F, EURUSD=X; cycles=1; horizon=10; cap=$25k) executed 4 trades with PnL -0.06%, updated `visualizations/dashboard_data.json`; positions: long MTN 10, short SOL 569, short GC=F 1, short EURUSD=X 792; quant_validation fail_fraction 0.932 (<0.98) and negative_expected_profit_fraction 0.488 (<0.60).

## 🔄 2025-11-24 Delta (currency update)
- Data-source-aware ticker resolver (`etl/data_universe.py`) added; auto-trader now resolves tickers via this helper (explicit + frontier default, optional provider discovery when empty).
- LLM fallback defaults to enabled in the trading loop for redundancy without changing thresholds.
- Dashboard JSON emission hardened (datetime → ISO) to eliminate serialization warnings during live runs.
- Barbell integration TODO captured in `BARBELL_INTEGRATION_TODO.md`; optimization and risk evaluation steps must treat tail-hedge/long-vol legs as convexity purchases evaluated with Sortino/Omega/CVaR and crisis scenarios, not Sharpe alone.
- Initial barbell config/policy implemented:
  - `config/barbell.yml` now defines global safe/risk buckets, feature flags, and per-market caps.
  - `risk/barbell_policy.BarbellConstraint` provides bucket weight computation and projection; behaviour remains unchanged while `enable_barbell_allocation=false`.
- Regime-aware exploration/exploitation scaffolding added:
  - `scripts/update_regime_state.py` computes per-ticker regime state (exploration vs exploitation, green/red/neutral) from realised PnL.
  - `execution/paper_trading_engine.PaperTradingEngine` consults `config/regime_state.yml` to scale per-trade risk (micro-sizing in exploration/red regimes, modest uplift in green regimes) without altering global guardrails.

## 📊 CURRENT PROJECT STATUS: 🟡 GATED (profitability/quant-health) – engineering unblocked

> **Current verified snapshot (2025-12-26)**: `Documentation/PROJECT_STATUS.md` (paper-window MVS now PASS; live/paper still gated)

**All Core Phases Complete**: ETL + Analysis + Visualization + Caching + k-fold CV + Multi-Source + Config-Driven + Checkpointing + LLM Integration

### 🆕 Immediate Updates (Nov 12, 2025)
- **🚨 2025-11-15 Brutal Run Blocking Findings**
  - `logs/pipeline_run.log:16932-17729` + `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` show the SQLite datastore is corrupted (`database disk image is malformed`, “rowid … out of order/missing from index”), so all sequenced tasks that rely on persistence are currently blocked. `DatabaseManager._connect` must treat this error the same way it treats `"disk i/o error"` (reset connection or activate the POSIX mirror) before further work.
  - `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` capture Stage 7 failing on every ticker with `ValueError: The truth value of a DatetimeIndex is ambiguous` because `scripts/run_etl_pipeline.py:1755-1764` evaluates `mssa_result.get('change_points') or []`. The stage logs “Saved forecast …” followed by “Generated forecasts for 0 ticker(s)”, so nothing reaches Stage 8.
  - The visualization hook immediately fails with `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (lines 2626, 2981, …), meaning dashboard deliverables cited throughout this plan cannot be produced.
  - Hardening claims about pandas/statsmodels warnings being resolved are false: `forcester_ts/forecaster.py:128-136` still performs the deprecated Period round-trip and `_select_best_order` in `forcester_ts/sarimax.py:136-183` still keeps unconverged orders, so the brutal logs are full of `FutureWarning`/`ValueWarning` spam.
  - `scripts/backfill_signal_validation.py:281-292` continues to call `datetime.utcnow()` with sqlite’s default converters, triggering Python 3.12 deprecation warnings (`logs/backfill_signal_validation.log:15-22`) every time the scheduled job runs.

  **Must-complete blockers before resuming sequenced work**
  1. Rebuild/recover `data/portfolio_maximizer.db` and update `DatabaseManager._connect` so `"database disk image is malformed"` shares the disk-I/O recovery path.
  2. Patch the MSSA `change_points` block to convert the `DatetimeIndex` into a list without boolean coercion, rerun the forecasting stage, and confirm Stage 8 consumes the bundles.
  3. Remove the unsupported `axis=` argument from the Matplotlib auto-format call so visualization artefacts exist again.
  4. Replace the Period coercion + tighten the SARIMAX search space to eliminate the warning storm (all related warnings now flow into `logs/warnings/warning_events.log` so they are audit-able without spamming console output).
  5. Modernize `scripts/backfill_signal_validation.py` (timezone-aware timestamps + sqlite adapters) before running nightly validation/backfill tasks. [Completed 2025-12-04; covered by `tests/scripts/test_backfill_signal_validation.py` under `simpleTrader_env`.]

  - 2025-12-04 update: All five blockers above now have code fixes in place. `scripts/backfill_signal_validation.py` uses timezone-aware UTC timestamps and sqlite adapters, and the new `tests/scripts/test_backfill_signal_validation.py` suite (4 tests) passes under `simpleTrader_env`. Instrumentation continues to log benchmark metrics and dataset diagnostics so sequenced tasks can reference `logs/forecast_audits/*.json` when evaluating future steps.


- `scripts/run_auto_trader.py` delivers the autonomous trading loop (extraction → validation → forecasting → TS signals → routing → execution) with optional LLM fallback; it must now be validated alongside the pipeline.
- README + UNIFIED_ROADMAP position the platform as an **Autonomous Profit Engine**, elevating the loop to core capability and documenting how to launch it.
- Stage planner in `scripts/run_etl_pipeline.py` now runs Time Series forecasting/signal routing before any LLM stages, keeping LLM strictly as fallback.
- `logs/errors/errors.log` exposes unresolved blockers: DataStorage CV signature mismatch (`test_size`), zero-fold CV `ZeroDivisionError`, SQLite `disk I/O` writes, and missing parquet engines preventing checkpoints. These issues are immediate priorities before new sequencing work proceeds.
- `bash/comprehensive_brutal_test.sh` previously reported `tests/ai_llm/test_ollama_client.py::TestOllamaGeneration::test_generate_switches_model_when_token_rate_low` as the lone failure; `ai_llm/ollama_client.py` now passes this test under `simpleTrader_env`, so brutal runs should treat it as a regression guard rather than an expected failure.

- `models/time_series_signal_generator.py` now normalises pandas objects/Series before evaluation, records decision context in provenance, and passes `pytest tests/models/test_time_series_signal_generator.py -q` plus targeted integration cases; `logs/ts_signal_demo.json` captures a live SELL signal derived from SQLite OHLCV data to prove TS signals are no longer stuck in HOLD.
- `etl/checkpoint_manager.py` swaps `Path.rename()` for `Path.replace()` so checkpoint metadata no longer crashes subsequent runs on Windows; temp metadata files are cleaned automatically.
- `scripts/backfill_signal_validation.py` imports `sys`, injects the repo root into `sys.path`, and therefore runs from the brutal suite without `ModuleNotFoundError`; nightly scheduling can safely call the script from outside the repo.
- `bash/comprehensive_brutal_test.sh` still reports `test_ollama_client.py::TestOllamaGeneration::test_generate_switches_model_when_token_rate_low` as the lone failure—sequenced fixes must keep this test at the top of the queue once the TS pipeline stabilises.

**Recent Achievements**:
- Phase 4.8: Checkpointing & Event Logging (Oct 7, 2025)
- Phase 5.1: Alpha Vantage & Finnhub APIs Complete (Oct 7, 2025)
- Phase 5.2: LLM Integration Complete (Ollama) (Oct 8, 2025)
- Phase 5.3: Profit Calculation Fix Applied (Oct 14, 2025) ⚠️ CRITICAL
- 196 tests (100% passing), 3 data sources operational
- Codebase: ~6,780 lines of production code

**⚠️ CRITICAL UPDATE**: Profit factor calculation was fixed on Oct 14, 2025. Previous calculations were incorrect (using averages instead of totals). See `Documentation/PROFIT_CALCULATION_FIX.md` for details.

**📚 NEW**: 
- A comprehensive sequenced implementation plan has been created. See **`Documentation/SEQUENCED_IMPLEMENTATION_PLAN.md`** for the complete 12-week implementation plan with critical fixes prioritized first.
- **Stub Implementation Review**: Complete review of all missing/incomplete implementations documented in **`Documentation/STUB_IMPLEMENTATION_PLAN.md`**. The cTrader client + order manager are now implemented; remaining critical stubs cover the performance dashboard, production deploy pipeline, and disaster recovery systems.
- **🟡 Time Series Signal Generation Refactoring IMPLEMENTED** (Nov 6, 2025) - **ROBUST TESTING REQUIRED**: See **`Documentation/REFACTORING_IMPLEMENTATION_COMPLETE.md`** for details. Time Series ensemble is now the DEFAULT signal generator with LLM as fallback. Includes 50 tests written (38 unit + 12 integration) - **NEEDS EXECUTION & VALIDATION**, unified database schema - **TESTING REQUIRED**, and complete pipeline integration - **TESTING REQUIRED**.

---

## 🚨 CRITICAL ISSUES IDENTIFIED (IMMEDIATE ACTION REQUIRED)

-### Implementation Status Check (Nov 06, 2025)
- ✅ `deploy_monitoring.sh` runs end-to-end; `scripts/monitor_llm_system.py` now surfaces latency benchmarks and summarises `llm_signal_backtests`.
- ✅ Phase A validator wiring complete: advanced 5-layer validator executes inside the pipeline with Kelly sizing + statistical diagnostics.
- ✅ Latency guardrails hold the <5 s target (caching + latency threshold + new token-throughput failover); document benchmark runs (see `logs/latency_benchmark.json`) and remediate when status is `DEGRADED_LATENCY`.
- ✅ Statistical validation tooling active: hypothesis tests, bootstrap CIs, and Ljung–Box diagnostics persist into `llm_signal_backtests`.
- ⚠️ Paper trading engine, broker integration, and downstream dashboards remain untouched.
- ✅ SAMOSSA SSA forecasts integrated with SARIMAX/GARCH; explained-variance diagnostics and backtests persist, while RL/CUSUM promotion remains gated on profitability.
- ✅ `TimeSeriesForecaster.evaluate()` now emits RMSE / sMAPE / tracking error for every model and ensemble run; the metrics are persisted to SQLite, ensemble weights use variance-ratio testing and change-point density, and MSSA-RL can optionally offload its SSA SVD to CuPy for GPU acceleration.
- ⚠️ `logs/errors/errors.log` (Nov 2–7) shows ETL blockers: `DataStorage.train_validation_test_split()` TypeError, zero-fold CV `ZeroDivisionError`, SQLite `disk I/O error`, and missing `pyarrow`/`fastparquet` causing checkpoint failures; `logs/monitor_llm_system.log` also shows LLM migration skips + >15 s latencies. These failures must be cleared before live data runs.
- 🟡 Nightly validation wrapper `schedule_backfill.bat` ready—register via Task Scheduler (e.g. `schtasks /Create /TN PortfolioMaximizer_BackfillSignals /TR "\"C:\path\to\schedule_backfill.bat\"" /SC DAILY /ST 02:00 /F`) to ensure continuous backfills.
- ✅ Time Series signal generator hardened (Nov 9, 2025): volatility forecasts converted to scalars, HOLD provenance timestamps logged, and regression validated via `pytest tests/models/test_time_series_signal_generator.py -q` plus targeted integration smoke (`tests/integration/test_time_series_signal_integration.py::TestTimeSeriesForecastingToSignalIntegration::test_forecast_to_signal_flow`).
- ✅ Enhanced portfolio math pipeline: `scripts/run_etl_pipeline.py` now imports `etl.portfolio_math` (the former enhanced module) by default, satisfying the guardrails in `AGENT_DEV_CHECKLIST.md`, `QUANTIFIABLE_SUCCESS_CRITERIA.md`, and the verification steps in `TESTING_GUIDE.md`.

### Issue 1: Database Constraint Error ✅ RESOLVED (Nov 5, 2025)
**Resolution**: `ai_llm/llm_database_integration.py` now migrates `llm_risk_assessments` to accept `'extreme'` risk levels, normalises existing rows, and exposes the canonical taxonomy through `LLMRiskAssessment`.
**Verification**: `python -m pytest tests/ai_llm/test_llm_enhancements.py::TestLLMDatabaseIntegration::test_risk_assessment_extreme_persisted`

### Issue 2: LLM Performance Bottleneck 🚧 **DEFERRED**
**Status**: Deferred while signal generation migrates toward SARIMAX/SAMOSSA/DQN models; latency tuning will resume after the LLM deprecation path is finalised.

### Issue 3: Zero Signal Validation ✅ RESOLVED (Nov 5, 2025)
**Resolution**: `scripts/run_etl_pipeline.py` registers every LLM decision with `LLMSignalTracker` and stores validator outcomes via `record_validator_result`/`flush`, so monitoring now sees live counts.
**Verification**: `python -m pytest tests/scripts/test_track_llm_signals.py`

---

## 📅 SEQUENCED IMPLEMENTATION PLAN

### **PHASE A: CRITICAL FIXES & LLM OPERATIONALIZATION (WEEKS 1-6)**

#### **WEEK 1: Critical System Fixes**

##### **Day 1-2: Database & Performance Fixes**
```python
# TASK A1.1: Fix Database Schema (30 minutes) 🔴 CRITICAL
# File: etl/database_manager.py
# Update risk_level constraint to include 'extreme'

# TASK A1.2: LLM Performance Optimization (4 hours) 🔴 CRITICAL
# File: ai_llm/ollama_client.py
class OptimizedOllamaClient:
    def __init__(self):
        self.model = "qwen:7b-chat-q4_K_M"  # Smaller, faster model
        self.cache = {}  # Response caching
        self.parallel_processing = True  # Process multiple tickers
        
    def optimize_prompts(self):
        """Reduce token count by 50% while maintaining quality"""
        # Shorter, more focused prompts
        # Remove redundant context
        # Use structured output format

# TASK A1.3: Signal Validation Implementation (2 hours) 🔴 CRITICAL
# File: ai_llm/signal_validator.py (already exists - deploy)
from ai_llm.signal_validator import SignalValidator

validator = SignalValidator()
# Run 30-day backtest on historical signals
backtest_results = validator.backtest_signal_quality(
    signals=historical_signals,
    actual_prices=historical_prices,
    lookback_days=30
)
```
> **Progress Update (2025-11-01):** Implemented prompt compression, cache TTL, latency-aware model failover, and token-rate fallback in `ai_llm/ollama_client.py`; pipeline now honours `cache_ttl_seconds`, `latency_failover_threshold`, and `token_rate_failover_threshold` from `config/llm_config.yml`.

##### **Day 3-4: Enhanced Portfolio Mathematics**
```python
# TASK A1.4: Deploy Enhanced Portfolio Math (1 hour) ✅ READY
# File: etl/portfolio_math_enhanced.py (already exists - deploy)
# Replace legacy portfolio_math.py with enhanced version

# TASK A1.5: Statistical Testing Framework (3 hours) 🆕 NEW
# File: etl/statistical_tests.py (NEW - 300 lines)
class StatisticalTestSuite:
    def test_strategy_significance(self, strategy_returns, benchmark_returns):
        """Test if strategy returns are statistically significant"""
        # T-test for mean difference
        # Information ratio calculation
        # F-test for variance equality
        
    def test_autocorrelation(self, returns):
        """Test for serial correlation in returns"""
        # Ljung-Box test
        # Durbin-Watson test
        
    def bootstrap_validation(self, returns, n_bootstrap=1000):
        """Bootstrap validation for performance metrics"""
        # Confidence intervals for Sharpe ratio
        # Confidence intervals for max drawdown
```

##### **Day 4-5: Barbell & Options Feature-Flag Wiring**
```text
TASK A1.7: Options/Derivatives Feature Flags (2 hours) 🆕 NEW
- Add `config/options_config.yml` with:
  - `options_trading.enabled` (master toggle, default `false`),
  - `barbell.max_options_weight` and `barbell.max_premium_pct_nav` to bound options exposure,
  - default OTM selection bands (`selection.moneyness`, `selection.expiry_days`).
- Reserve environment flags:
  - `ENABLE_OPTIONS=true` to opt into options logic at runtime,
  - `OPTIONS_CONFIG_PATH` to override the default options config.
- Update documentation:
  - `Documentation/BARBELL_OPTIONS_MIGRATION.md` for the migration path from spot-only to barbell options,
  - AGENT_* guides to enforce "options must be feature-flagged and barbell-constrained".

Success criteria:
- [ ] Options disabled (default) ⇒ no behavioural change in ETL/auto-trader/brutal suite.
- [ ] Options enabled (flag + config) ⇒ options are treated strictly as risk-bucket instruments under barbell guardrails.
```

##### **Day 5-7: Paper Trading Engine**
```python
# TASK A1.6: Complete Paper Trading Engine (4 hours) ✅ READY
# File: execution/paper_trading_engine.py (already exists - complete)
class PaperTradingEngine:
    def execute_signal(self, signal, portfolio):
        """Execute signal with realistic simulation"""
        # 1. Validate signal (5-layer validation)
        # 2. Calculate position size (Kelly criterion)
2. ? **Fix signal/risk persistence contract** (Nov 5, 2025)  
   - `ai_llm/llm_database_integration.py` migrates `llm_risk_assessments` to accept 'extreme' and normalises legacy rows; validator tests confirm the schema.  
   - `python -m pytest tests/ai_llm/test_llm_enhancements.py::TestLLMDatabaseIntegration::test_risk_assessment_extreme_persisted`
3. ? **Persist validator telemetry via LLMSignalTracker** (Nov 5, 2025)  
   - `scripts/run_etl_pipeline.py` now registers each signal and validator decision with `LLMSignalTracker`; `record_validator_result`/`flush` keep dashboards populated.  
   - `python -m pytest tests/scripts/test_track_llm_signals.py`

---

## ðŸš€ PHASE B: TIME-SERIES MODEL UPGRADE (WEEKS 7-10)

### **Time-Series Model Upgrade Overview**
- Execute SAMOSSA, SARIMAX, and GARCH forecasts alongside the current LLM stack.
- Preserve backward compatibility by routing new outputs through `signal_router.py` with feature flags.
- Compare models with rolling cross-validation and walk-forward tests on profit factor, drawdown, and loss metrics.
- Promote the most consistent, low-loss model to default only after statistically significant outperformance; retain LLM as the fallback.
- Instrument monitoring so any regression triggers automatic reversion to the existing production configuration.

### **Week 7: SAMOSSA + SARIMAX Foundations**
```python
# TASK B7.1: Time-Series Feature Engineering Upgrade (Days 43-45)  # ⏳ Pending
# File: etl/time_series_feature_builder.py (NEW - 250 lines)
class TimeSeriesFeatureBuilder:
    def build_features(self, price_history: pd.DataFrame) -> pd.DataFrame:
        """Create lag, seasonal, and volatility features for SAMOSSA/SARIMAX"""
        # Seasonal decomposition, holiday effects
        # Rolling statistics and differencing
        # Persist outputs via database_manager.py feature store helpers

# TASK B7.2: SAMOSSA Forecaster (Days 46-47)  # ✅ Delivered 2025-11-05 (etl/time_series_forecaster.py::SAMOSSAForecaster)
# Notes: SSA Page matrix, ≥90% energy capture, residual ARIMA, CUSUM hooks (see SAMOSSA_algorithm_description.md)
class SAMOSSAForecaster:
    def fit(self, features: pd.DataFrame) -> None:
        """Train Seasonal Adaptive Multi-Order Smoothing (SAMOSSA) model"""
        # Extend model registry for checkpoint storage
        # Configurable seasonality + adaptive smoothing parameters

    def forecast(self, horizon: int) -> ForecastResult:
        """Return price forecasts with confidence intervals"""
        # Output conforms to existing signal schema for parity with LLM signals

# TASK B7.3: SARIMAX Pipeline Refresh (Days 48-49)  # ✅ Delivered (see SAMOSSA rollout notes in TIME_SERIES_FORECASTING_IMPLEMENTATION.md)
# File: models/sarimax_model.py (REFRESH - 280 lines)
class SARIMAXForecaster:
    def fit_and_forecast(self, market_data: pd.DataFrame) -> ForecastResult:
        """Production SARIMAX with automated parameter tuning"""
        # Use pmdarima auto_arima with guardrails
        # Leverage cached exogenous features from ETL pipeline
        # Persist diagnostics for regime-aware switching
```

### **Week 8: GARCH + Parallel Inference Integration**
```python
# TASK B8.1: GARCH Volatility Engine (Days 50-52)  # ✅ Delivered (etl/time_series_forecaster.py::GARCHForecaster)
# File: models/garch_model.py (NEW - 320 lines)
class GARCHVolatilityEngine:
    def fit(self, returns: pd.Series) -> None:
        """Estimate volatility clusters via GARCH(p, q)"""
        # Use arch package for variance forecasts
        # Emit risk-adjusted metrics for signal fusion

    def forecast_volatility(self, steps: int = 1) -> VolatilityResult:
        """Expose volatility forecast to risk sizing logic"""
        # Integrate with portfolio_math_enhanced.py metrics

# TASK B8.2: Parallel Model Runner (Days 53-54)  # ⏳ Pending (async orchestration + provenance logging)
# File: models/time_series_runner.py (NEW - 260 lines)
class TimeSeriesRunner:
    def run_all(self, context: MarketContext) -> List[Signal]:
        """Execute SAMOSSA, SARIMAX, GARCH, and LLM pipelines in parallel"""
        # Async execution via existing task orchestrator
        # Normalize outputs into unified signal schema
        # Attach provenance metadata for monitoring dashboards

# TASK B8.3: Backward-Compatible Signal Routing (Days 55-56)  # ⏳ Pending
# File: signal_router.py (UPDATE - 180 lines)
class SignalRouter:
    def route(self, signals: List[Signal]) -> SignalBundle:
        """Merge legacy LLM signals with new time-series models"""
        # Feature flag toggles for gradual rollout
        # Priority ordering based on confidence and risk score
        # Downstream consumers see unchanged interface
```

> **SAMOSSA Delivery Note (Nov 05, 2025)**  
> • SSA-based forecasts with residual ARIMA are live in `etl/time_series_forecaster.py`; diagnostics (explained variance, residual forecasts) persist via `DatabaseManager.save_forecast`.  
> • CUSUM change-point scoring and Q-learning intervention policies remain pending, gated on MVS/PRS profitability thresholds (`QUANTIFIABLE_SUCCESS_CRITERIA.md`).  
> • Monitoring tasks must ingest SAMOSSA diagnostics before feature flags route time-series outputs downstream.

### **Week 9: Cross-Validation + Evaluation Framework**
```python
# TASK B9.1: Rolling Cross-Validation Harness (Days 57-59)
# File: analysis/time_series_validation.py (NEW - 300 lines)
class TimeSeriesValidation:
    def evaluate(self, models: List[BaseModel]) -> ValidationReport:
        """Blocked CV and walk-forward evaluation across horizons"""
        # Profit factor, max drawdown, hit rate, volatility-adjusted return
        # Statistical tests (Diebold-Mariano, paired t-tests)
        # Persist results for dashboards and CI

# TASK B9.2: Performance Dashboard Extension (Days 60-61)
# File: monitoring/performance_dashboard.py (UPDATE - 200 lines)
class PerformanceDashboard:
    def render_time_series_tab(self, reports: List[ValidationReport]) -> Dashboard:
        """Compare LLM vs SAMOSSA/SARIMAX/GARCH performance"""
        # Rolling metrics, drawdown curves, loss distribution
        # Highlight leading model per asset and regime

# TASK B9.3: Risk & Compliance Review (Days 62-63)
# File: risk/model_governance.py (NEW - 150 lines)
class ModelGovernance:
    def certify(self, report: ValidationReport) -> GovernanceDecision:
        """Ensure new models satisfy risk limits before promotion"""
        # Enforce drawdown, VaR/ES thresholds, and audit logs
        # Document fallback procedures and approvals
```

### **Week 10: Promotion, Fallback, and Automation**
```python
# TASK B10.1: Dynamic Model Selection Logic (Days 64-66)
# File: models/model_selector.py (NEW - 220 lines)
class ModelSelector:
    def choose_primary_model(self, reports: List[ValidationReport]) -> ModelDecision:
        """Promote the most consistent, low-loss model"""
        # Weighted scoring (profit factor, drawdown, Sharpe, stability)
        # Minimum uplift thresholds relative to LLM baseline
        # Automatic reversion on degraded performance

# TASK B10.2: Integration Tests and Regression Suite (Days 67-68)
# File: tests/test_time_series_models.py (NEW - 280 lines)
class TestTimeSeriesModels(unittest.TestCase):
    def test_parallel_pipeline_integrity(self) -> None:
        """Validate routing, fallback, and metrics instrumentation"""
        # Simulate feature-flag toggles and failure scenarios

# TASK B10.3: Deployment Playbook Update (Days 69-70)
# File: docs/deployment_playbook.mdc (UPDATE - 120 lines)
def document_time_series_rollout():
    """Update runbooks for rolling out the new default model"""
    # Include rollback checklist, monitoring KPIs, approval steps
```

### **Phase B Success Criteria**
- [ ] SAMOSSA, SARIMAX, and GARCH pipelines deliver signals alongside LLM output
- [ ] Feature flags enable immediate fallback to the current LLM-only routing
- [ ] Rolling CV shows >= 3% profit-factor uplift with <= 50% drawdown increase versus baseline
- [ ] Governance sign-off documented with a tested fallback plan
- [ ] Dynamic selector promotes the top model automatically and regression suite passes

