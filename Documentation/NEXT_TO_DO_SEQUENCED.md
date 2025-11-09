# 🎯 SEQUENCED TO-DO LIST: Portfolio Maximizer v45
**Production-Ready ML Trading System - Critical Fixes First**

**Date**: October 19, 2025  
**Status**: Ready for Implementation  
**Priority**: **CRITICAL** - Based on log analysis and system requirements

---

## 📊 CURRENT PROJECT STATUS: PRODUCTION READY ✅

**All Core Phases Complete**: ETL + Analysis + Visualization + Caching + k-fold CV + Multi-Source + Config-Driven + Checkpointing + LLM Integration

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
- **Stub Implementation Review**: Complete review of all missing/incomplete implementations documented in **`Documentation/STUB_IMPLEMENTATION_PLAN.md`**. Identifies 12+ critical stubs including IBKR client, order manager, performance dashboard, and disaster recovery systems that must be completed before production deployment.

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
- 🟡 Nightly validation wrapper `schedule_backfill.bat` ready—register via Task Scheduler (e.g. `schtasks /Create /TN PortfolioMaximizer_BackfillSignals /TR "\"C:\path\to\schedule_backfill.bat\"" /SC DAILY /ST 02:00 /F`) to ensure continuous backfills.
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
