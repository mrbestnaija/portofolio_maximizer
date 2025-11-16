# Stub & Incomplete Implementation Plan
**Comprehensive Review and Replacement Strategy**

**Date**: 2025-11-12  
**Status**: 🔴 BLOCKED – 2025-11-15 brutal run regressions  
**Priority**: HIGH - Complete all stubs before production deployment

### Nov 12, 2025 Update
- Time Series signal generator refactor is exercised via logs/ts_signal_demo.json; stubs for router/broker now depend on these BUY/SELL payloads rather than HOLD placeholders.
- Checkpoint/metadata utilities are stable on Windows thanks to Path.replace, unblocking future stub work that writes checkpoints repeatedly.
- Validator/backfill stubs can assume scripts/backfill_signal_validation.py is callable from scheduled jobs; only the scheduler glue remains.

### 🚨 2025-11-15 Brutal Run Regression
- `logs/pipeline_run.log:16932-17729` and `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` confirmed the database that these stubs rely on is corrupted (`database disk image is malformed`, “rowid … out of order/missing from index”). `DatabaseManager._connect` must treat this error like the existing disk-I/O path before any stub replacement can be validated.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` show Stage 7 still failing with `ValueError: The truth value of a DatetimeIndex is ambiguous` because `scripts/run_etl_pipeline.py:1755-1764` evaluates `mssa_result.get('change_points') or []`. The refactors cited above therefore leave Stage 8 without data.
- The visualization hook crashes with `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (lines 2626, 2981, …), so dashboard/pipeline stubs that depend on PNG artefacts cannot be closed yet.
- Pandas/statsmodels warning spam persists because `forcester_ts/forecaster.py:128-136` forces a deprecated `PeriodIndex` round-trip and `_select_best_order` in `forcester_ts/sarimax.py:136-183` keeps unconverged orders, contradicting the “forecaster stubs resolved” statement. *(Resolved with warning capture on Nov 16 — review `logs/warnings/warning_events.log` for the preserved warning stream.)*
- `scripts/backfill_signal_validation.py:281-292` still uses `datetime.utcnow()` and sqlite’s default converters, generating Python 3.12 deprecation warnings (`logs/backfill_signal_validation.log:15-22`) whenever validator stubs run.

**Blocking fixes**
1. Rebuild/recover `data/portfolio_maximizer.db` and update `DatabaseManager._connect` so `"database disk image is malformed"` triggers the reset/mirror flow.
2. Patch the MSSA `change_points` branch (cast to list) and rerun the forecasting stage to feed Stage 8.
3. Remove the unsupported `axis=` argument from the Matplotlib auto-format call so visualization-dependent stubs can be validated.
4. Replace the deprecated Period coercion + tighten SARIMAX order search to silence warnings and stabilise the forecasting core.
5. Modernize `scripts/backfill_signal_validation.py` with timezone-aware timestamps + sqlite adapters before scheduling nightly backfills that exercise these stubs.
- ✅ `2025-11-16` note: Fixes 1–4 are now complete (see `logs/pipeline_run.log:22237-22986` for the clean pipeline run). Remaining blocker for this plan is the validator modernization in item 5.

---

## 📋 Executive Summary

This document identifies all stub implementations, incomplete code, and placeholders that need to be completed or replaced with production-ready implementations. The review covers the entire codebase with focus on:

- Stub classes and functions
- Incomplete implementations (pass statements, NotImplementedError)
- Mock/fake implementations that need real logic
- TODO/FIXME comments indicating incomplete work
- Missing critical functionality
- ✅ UPDATE (Nov 8, 2025): The forecasting stubs around regression metrics, ensemble heuristics, and MSSA-RL GPU acceleration have been fully implemented (`forcester_ts/metrics.py`, `forcester_ts/ensemble.py`, `forcester_ts/mssa_rl.py`). No further action is required for those items; they now serve as references for future Phase B enhancements.
- ✅ UPDATE (Nov 9, 2025): `models/time_series_signal_generator.py` is no longer a stub—volatility handling, provenance metadata, and regression coverage have been completed and verified via `pytest tests/models/test_time_series_signal_generator.py -q` plus the targeted integration smoke test. Remove it from the open-stub list; remaining signal-related stubs now focus on broker/paper-trading glue.
- ✅ UPDATE (Nov 12, 2025): `models/signal_router.py` and the TS-first pipeline reordering are production code (see `Documentation/UNIFIED_ROADMAP.md`). Remove Signal Router from the open-stub list.
- ⚠️ UPDATE (Nov 12, 2025): `bash/comprehensive_brutal_test.sh` highlighted new gaps—`tests/etl/test_data_validator.py` is missing, and the Time Series forecasting tests timed out with a `Broken pipe`. Section 13 documents these testing placeholders.

---

## 🔍 Codebase Analysis Results

### Files Reviewed: 100+ Python files
### Stubs Identified: 15+ incomplete implementations
### Priority: CRITICAL - Blocking production deployment

---

## 🚨 CRITICAL STUBS (Must Complete Before Production)

### 1. **Broker Integration - cTrader Client** ✅ IMPLEMENTED
**Location**: `execution/ctrader_client.py`  
**Status**: Replaces the missing IBKR stub with a demo-first cTrader Open API
client that reads credentials from `.env`, handles OAuth token refresh, and
exposes order/account helpers for the order manager.  
**Priority**: CRITICAL - Required for live trading (demo first per roadmap)

**Key Features**:
- Demo + live endpoints configurable via environment variables (`USERNAME_CTRADER`,
  `PASSWORD_CTRADER`, `APPLICATION_NAME_CTRADER`, optional
  `CTRADER_ACCOUNT_ID`/`CTRADER_APPLICATION_SECRET`).
- OAuth password + refresh-token workflows with automatic retries and
  configurable timeouts.
- Dataclass-wrapped orders, account snapshots, and placement responses so
  downstream code remains testable.
- Credential loader accepts both `KEY=value` and `KEY:'value'` syntaxes to match
  the regenerated `.env` file noted in `RECOVER_ENV_FILE.md`.

**Success Criteria**:
- [x] Demo authentication flow coded (`execution/ctrader_client.py`).
- [x] Order placement + cancellation helpers exposed for `OrderManager`.
- [x] Error handling/logging + token refresh implemented.
- [ ] 50+ successful demo trades recorded (requires manual run against broker).
- [ ] Live credentials smoke-tested once demo KPIs clear guardrails.
- [x] Unit tests cover configuration + payload generation (added in this PR).

---

### 2. **Order Management System** ✅ IMPLEMENTED
**Location**: `execution/order_manager.py`  
**Status**: Ships the full lifecycle manager described in the roadmap. Ties the
new cTrader client to `RealTimeRiskManager` and `DatabaseManager`, enforces the
2% position cap, and persists executions for dashboards.  
**Priority**: CRITICAL - Required for live trading

**Highlights**:
- Demo-first (default `mode="demo"`) with injectable cTrader client for tests
  and future live runs.
- Pre-trade checks cover confidence thresholds, free margin, 2% position caps,
  daily trade limits, and circuit-breaker status from `RealTimeRiskManager`.
- Confidence-weighted sizing + database persistence via
  `DatabaseManager.save_trade_execution` for executed fills.
- Structured `LifecycleResult` dataclass informs automation + monitoring.

**Success Criteria**:
- [x] Order lifecycle management implemented with dependency injection.
- [x] Pre-trade validation + risk gating enforced automatically.
- [x] Database writes triggered for executed trades.
- [ ] Fill monitoring/post-trade reconciliation with live executions (pending
  real broker runs).
- [x] Integration-ready API used by forthcoming automation scripts.

---

### 3. **Production Performance Dashboard** ❌ NOT IMPLEMENTED
**Location**: `monitoring/performance_dashboard.py` (MISSING)  
**Status**: Referenced in documentation but file does not exist  
**Priority**: HIGH - Required for monitoring

**Required Implementation**:
```python
# NEW: monitoring/performance_dashboard.py (300 lines)
class PerformanceDashboard:
    """Real-time performance monitoring"""
    
    def generate_live_metrics(self) -> Dashboard:
        """Generate real-time dashboard"""
        
        db = DatabaseManager()
        pm = PortfolioMath()
        
        # Get recent performance
        trades = db.get_recent_trades(days=30)
        portfolio = db.get_current_portfolio()
        llm_signals = db.get_llm_signals(days=30)
        
        # Calculate metrics
        metrics = {
            # Trading performance
            'total_trades': len(trades),
            'win_rate': pm.calculate_win_rate(trades),
            'profit_factor': pm.calculate_profit_factor(trades),
            'sharpe_ratio': pm.calculate_sharpe_ratio(portfolio),
            
            # LLM signal quality
            'signal_accuracy': self._calculate_signal_accuracy(llm_signals, trades),
            'avg_confidence': np.mean([s.confidence for s in llm_signals]),
            
            # Risk metrics
            'current_drawdown': pm.calculate_current_drawdown(portfolio),
            'portfolio_volatility': pm.calculate_volatility(portfolio),
            'var_95': pm.calculate_var(portfolio, 0.95),
            
            # System health
            'uptime': self._calculate_uptime(),
            'data_quality': self._calculate_data_quality(),
            'latency_ms': self._calculate_avg_latency()
        }
        
        return Dashboard(
            metrics=metrics,
            charts=self._generate_charts(trades, portfolio),
            alerts=self._get_active_alerts(),
            timestamp=datetime.now()
        )
```

**Dependencies**:
- `etl/database_manager.py`
- `etl/portfolio_math.py`
- Web framework (Flask/FastAPI) for serving dashboard
- Charting library (Plotly/Bokeh)

**Success Criteria**:
- [ ] Live dashboard operational
- [ ] Metrics updating every 1 minute
- [ ] Historical charts (30-day, 90-day, 1-year)
- [ ] Alert visualization
- [ ] Export to CSV/JSON
- [ ] Web interface accessible

---

### 4. **Production Deployment Pipeline** ❌ NOT IMPLEMENTED
**Location**: `deployment/production_deploy.py` (MISSING)  
**Status**: Referenced in documentation but file does not exist  
**Priority**: HIGH - Required for production deployment

**Required Implementation**:
```python
# NEW: deployment/production_deploy.py (400 lines)
class ProductionDeployer:
    """Production deployment with health checks"""
    
    def deploy_trading_system(self) -> DeploymentResult:
        """Deploy with comprehensive validation"""
        
        # Environment validation
        env_validator = EnvironmentValidator()
        if not env_validator.validate():
            return DeploymentResult(status='FAILED', 
                                   reason='Environment validation failed')
        
        # API key validation
        api_validator = APIKeyValidator()
        if not api_validator.validate_all_keys():
            return DeploymentResult(status='FAILED',
                                   reason='API keys invalid')
        
        # System health check
        health = self._system_health_check()
        if health.status != 'HEALTHY':
            return DeploymentResult(status='FAILED',
                                   reason=f'Health check failed: {health.issues}')
        
        # Deploy components
        self._deploy_signal_validator()
        self._deploy_paper_trading_engine()
        self._deploy_risk_manager()
        self._deploy_performance_dashboard()
        
        # Start monitoring
        self._start_monitoring_services()
        
        return DeploymentResult(status='SUCCESS',
                               components_deployed=4,
                               health_status=health)
```

**Dependencies**:
- All core components
- Environment configuration
- Health check utilities

**Success Criteria**:
- [ ] Production deployment pipeline working
- [ ] All components deployed successfully
- [ ] System health monitoring active
- [ ] Automated rollback on failure
- [ ] Uptime >99.9% for 2 weeks

---

### 5. **Disaster Recovery System** ❌ NOT IMPLEMENTED
**Location**: `recovery/disaster_recovery.py` (MISSING)  
**Status**: Referenced in documentation but file does not exist  
**Priority**: HIGH - Required for production reliability

**Required Implementation**:
```python
# NEW: recovery/disaster_recovery.py (350 lines)
class DisasterRecovery:
    """Automated disaster recovery"""
    
    def handle_system_failure(self, failure: SystemFailure) -> RecoveryResult:
        """Automatic recovery procedures"""
        
        if failure.type == 'MODEL_FAILURE':
            # Fallback to simpler models
            return self._fallback_to_simple_model()
            
        elif failure.type == 'DATA_FAILURE':
            # Switch to backup data source
            return self._failover_data_source()
            
        elif failure.type == 'BROKER_FAILURE':
            # Close positions if can't connect
            return self._emergency_position_closure()
            
        elif failure.type == 'RISK_BREACH':
            # Automatic position reduction
            return self._automatic_risk_reduction()
    
    def _failover_data_source(self) -> RecoveryResult:
        """Fail over to backup data source"""
        # Use existing DataSourceManager
        dsm = DataSourceManager()
        
        # Try data sources in priority order
        for source in ['yfinance', 'alpha_vantage', 'finnhub']:
            if dsm.test_data_source(source):
                dsm.set_primary_source(source)
                return RecoveryResult(status='RECOVERED',
                                     action=f'Switched to {source}')
        
        return RecoveryResult(status='FAILED',
                             action='All data sources unavailable')
```

**Dependencies**:
- `etl/data_source_manager.py`
- `etl/checkpoint_manager.py`
- `execution/order_manager.py`

**Success Criteria**:
- [ ] Disaster recovery system operational
- [ ] Data source failover tested
- [ ] Model fallback tested
- [ ] Emergency position closure tested
- [ ] Recovery from checkpoints verified

---

## ⚠️ INCOMPLETE IMPLEMENTATIONS (Need Completion)

### 6. **Real-Time Risk Manager - Automatic Actions** 🟡 PARTIAL
**Location**: `risk/real_time_risk_manager.py` (Lines 318-357)  
**Status**: Logging only - needs actual execution integration  
**Priority**: MEDIUM - Currently logs actions but doesn't execute

**Current State**:
```python
def _execute_automatic_action(self, action: str, positions: Dict[str, int]):
    """Execute automatic risk mitigation action"""
    logger.info(f"Executing automatic risk action: {action}")
    
    if action == 'CLOSE_ALL_POSITIONS':
        logger.critical("EMERGENCY: Closing all positions due to risk breach")
        # In production, would call execution engine to close positions
        # For now, just log the action
        for ticker in positions.keys():
            logger.critical(f"  → CLOSE position in {ticker}")
```

**Required Completion**:
```python
def _execute_automatic_action(self, action: str, positions: Dict[str, int]):
    """Execute automatic risk mitigation action"""
    logger.info(f"Executing automatic risk action: {action}")
    
    if action == 'CLOSE_ALL_POSITIONS':
        logger.critical("EMERGENCY: Closing all positions due to risk breach")
        # Integrate with order manager
        from execution.order_manager import OrderManager
        order_manager = OrderManager()
        
        for ticker, shares in positions.items():
            close_order = Order(
                ticker=ticker,
                action='SELL',
                quantity=shares,
                order_type='MARKET',
                reason='RISK_BREACH'
            )
            order_manager.manage_order_lifecycle(close_order)
    
    elif action == 'REDUCE_POSITIONS':
        logger.warning("RISK MITIGATION: Reducing positions by 50%")
        from execution.order_manager import OrderManager
        order_manager = OrderManager()
        
        for ticker, shares in positions.items():
            reduced_shares = shares // 2
            reduce_order = Order(
                ticker=ticker,
                action='SELL',
                quantity=reduced_shares,
                order_type='MARKET',
                reason='RISK_MITIGATION'
            )
            order_manager.manage_order_lifecycle(reduce_order)
```

**Dependencies**:
- `execution/order_manager.py` (must be implemented first)

**Success Criteria**:
- [ ] Automatic position closure working
- [ ] Position reduction working
- [ ] Stop loss tightening working
- [ ] Integration tests passing

---

### 7. **Time-Series Feature Builder** ❌ NOT IMPLEMENTED
**Location**: `etl/time_series_feature_builder.py` (MISSING)  
**Status**: Referenced in Phase B documentation but not implemented  
**Priority**: MEDIUM - Required for advanced ML models

**Required Implementation**:
```python
# NEW: etl/time_series_feature_builder.py (250 lines)
class TimeSeriesFeatureBuilder:
    """Create lag, seasonal, and volatility features for SAMOSSA/SARIMAX"""
    
    def build_features(self, price_history: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-series features"""
        
        features = {}
        
        # Lag features
        for lag in [1, 5, 10, 20]:
            features[f'price_lag_{lag}'] = price_history['Close'].shift(lag)
            features[f'return_lag_{lag}'] = price_history['Close'].pct_change(lag)
        
        # Seasonal decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(
            price_history['Close'],
            model='additive',
            period=252  # Annual seasonality
        )
        features['trend'] = decomposition.trend
        features['seasonal'] = decomposition.seasonal
        features['residual'] = decomposition.resid
        
        # Holiday effects (simplified)
        features['is_month_end'] = price_history.index.is_month_end.astype(int)
        features['is_quarter_end'] = price_history.index.is_quarter_end.astype(int)
        
        # Rolling statistics
        for window in [5, 10, 20, 60]:
            features[f'rolling_mean_{window}'] = price_history['Close'].rolling(window).mean()
            features[f'rolling_std_{window}'] = price_history['Close'].rolling(window).std()
            features[f'rolling_skew_{window}'] = price_history['Close'].rolling(window).skew()
        
        # Differencing
        features['diff_1'] = price_history['Close'].diff(1)
        features['diff_5'] = price_history['Close'].diff(5)
        
        # Persist via database_manager
        from etl.database_manager import DatabaseManager
        db = DatabaseManager()
        db.save_feature_store(features, 'time_series_features')
        
        return pd.DataFrame(features).dropna()
```

**Dependencies**:
- `statsmodels` for seasonal decomposition
- `etl/database_manager.py` for feature store

**Success Criteria**:
- [ ] Lag features created
- [ ] Seasonal decomposition working
- [ ] Holiday effects included
- [ ] Rolling statistics calculated
- [ ] Features persisted to database
- [ ] Integration with SAMOSSA/SARIMAX

---

### 8. **Parallel Model Runner** ❌ NOT IMPLEMENTED
**Location**: `models/time_series_runner.py` (MISSING)  
**Status**: Referenced in Phase B documentation but not implemented  
**Priority**: MEDIUM - Required for parallel model execution

**Required Implementation**:
```python
# NEW: models/time_series_runner.py (260 lines)
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor

class TimeSeriesRunner:
    """Execute SAMOSSA, SARIMAX, GARCH, and LLM pipelines in parallel"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def run_all(self, context: MarketContext) -> List[Signal]:
        """Execute all models in parallel"""
        
        # Create tasks for each model
        tasks = [
            self._run_samossa(context),
            self._run_sarimax(context),
            self._run_garch(context),
            self._run_llm(context)
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Normalize outputs into unified signal schema
        signals = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Model execution failed: {result}")
                continue
            
            # Convert to unified signal format
            signal = self._normalize_signal(result)
            signal.provenance = self._get_provenance(result)
            signals.append(signal)
        
        return signals
    
    async def _run_samossa(self, context: MarketContext) -> ForecastResult:
        """Run SAMOSSA forecaster"""
        from etl.time_series_forecaster import SAMOSSAForecaster
        forecaster = SAMOSSAForecaster()
        forecaster.fit(context.market_data)
        return forecaster.forecast(context.horizon)
    
    async def _run_sarimax(self, context: MarketContext) -> ForecastResult:
        """Run SARIMAX forecaster"""
        from etl.time_series_forecaster import SARIMAXForecaster
        forecaster = SARIMAXForecaster()
        return forecaster.fit_and_forecast(context.market_data)
    
    async def _run_garch(self, context: MarketContext) -> VolatilityResult:
        """Run GARCH volatility engine"""
        from etl.time_series_forecaster import GARCHForecaster
        forecaster = GARCHForecaster()
        forecaster.fit(context.returns)
        return forecaster.forecast_volatility(context.horizon)
    
    async def _run_llm(self, context: MarketContext) -> Signal:
        """Run LLM signal generator"""
        from ai_llm.signal_generator import LLMSignalGenerator
        from ai_llm.ollama_client import OllamaClient
        
        client = OllamaClient()
        generator = LLMSignalGenerator(client)
        return generator.generate_signal(context.market_data)
    
    def _normalize_signal(self, result: Any) -> Signal:
        """Normalize model output to unified signal schema"""
        # Implementation depends on result type
        pass
    
    def _get_provenance(self, result: Any) -> Dict:
        """Extract provenance metadata"""
        return {
            'model_type': type(result).__name__,
            'timestamp': datetime.now(),
            'confidence': getattr(result, 'confidence', 0.5)
        }
```

**Dependencies**:
- `etl/time_series_forecaster.py` (SAMOSSA, SARIMAX, GARCH)
- `ai_llm/signal_generator.py`
- `asyncio` for parallel execution

**Success Criteria**:
- [ ] Parallel execution working
- [ ] All models execute concurrently
- [ ] Unified signal schema output
- [ ] Provenance metadata attached
- [ ] Error handling for failed models
- [ ] Performance benchmarks <5s total

---

### 10. **Time-Series Validation Framework** ❌ NOT IMPLEMENTED
**Location**: `analysis/time_series_validation.py` (MISSING)  
**Status**: Referenced in Phase B documentation but not implemented  
**Priority**: MEDIUM - Required for model evaluation

**Required Implementation**:
```python
# NEW: analysis/time_series_validation.py (300 lines)
class TimeSeriesValidation:
    """Blocked CV and walk-forward evaluation across horizons"""
    
    def evaluate(self, models: List[BaseModel]) -> ValidationReport:
        """Comprehensive model evaluation"""
        
        results = []
        
        for model in models:
            # Walk-forward validation
            cv_results = self._walk_forward_validation(model)
            
            # Calculate metrics
            metrics = {
                'profit_factor': self._calculate_profit_factor(cv_results),
                'max_drawdown': self._calculate_max_drawdown(cv_results),
                'hit_rate': self._calculate_hit_rate(cv_results),
                'volatility_adjusted_return': self._calculate_var_adjusted_return(cv_results),
                'sharpe_ratio': self._calculate_sharpe(cv_results)
            }
            
            # Statistical tests
            statistical_tests = {
                'diebold_mariano': self._diebold_mariano_test(cv_results),
                'paired_t_test': self._paired_t_test(cv_results)
            }
            
            results.append(ValidationReport(
                model_name=model.__class__.__name__,
                metrics=metrics,
                statistical_tests=statistical_tests,
                cv_results=cv_results
            ))
        
        # Persist results for dashboards and CI
        self._persist_results(results)
        
        return results
    
    def _walk_forward_validation(self, model: BaseModel) -> List[CVResult]:
        """Walk-forward cross-validation"""
        # Implementation with proper time-series CV
        pass
    
    def _calculate_profit_factor(self, results: List[CVResult]) -> float:
        """Calculate profit factor"""
        total_profit = sum(r.profit for r in results if r.profit > 0)
        total_loss = abs(sum(r.profit for r in results if r.profit < 0))
        return total_profit / total_loss if total_loss > 0 else 0.0
    
    def _diebold_mariano_test(self, results: List[CVResult]) -> Dict:
        """Diebold-Mariano test for forecast accuracy"""
        # Implementation using statsmodels
        pass
```

**Dependencies**:
- `statsmodels` for statistical tests
- `etl/time_series_cv.py` for walk-forward CV
- Database for result persistence

**Success Criteria**:
- [ ] Walk-forward validation working
- [ ] Profit factor calculated
- [ ] Statistical tests implemented
- [ ] Results persisted to database
- [ ] Integration with dashboards

---

## 🔧 MINOR INCOMPLETE ITEMS (Low Priority)

### 11. **Exception Handlers with Pass** 🟡 MINOR
**Location**: `forcester_ts/sarimax.py` (Lines 227, 234)  
**Status**: Exception handlers silently pass - should log  
**Priority**: LOW - Should log exceptions

**Current State**:
```python
try:
    lb_df = acorr_ljungbox(...)
    diagnostics["ljung_box_pvalue"] = float(lb_df["lb_pvalue"].iloc[-1])
except Exception:  # pragma: no cover
    pass

try:
    _, jb_pvalue = jarque_bera(residuals)
    diagnostics["jarque_bera_pvalue"] = float(jb_pvalue)
except Exception:  # pragma: no cover
    pass
```

**Required Fix**:
```python
try:
    lb_df = acorr_ljungbox(...)
    diagnostics["ljung_box_pvalue"] = float(lb_df["lb_pvalue"].iloc[-1])
except Exception as e:  # pragma: no cover
    logger.warning(f"Ljung-Box test failed: {e}")
    diagnostics["ljung_box_pvalue"] = None

try:
    _, jb_pvalue = jarque_bera(residuals)
    diagnostics["jarque_bera_pvalue"] = float(jb_pvalue)
except Exception as e:  # pragma: no cover
    logger.warning(f"Jarque-Bera test failed: {e}")
    diagnostics["jarque_bera_pvalue"] = None
```

---

### 12. **Cache Manager - Exception Handler** 🟡 MINOR
**Location**: `scripts/cache_manager.py` (Line 103)  
**Status**: Exception handler silently passes  
**Priority**: LOW - Should log exceptions

**Current State**:
```python
def _get_dir_size(self, path: Path) -> int:
    """Get total size of directory"""
    total_size = 0
    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except:
        pass
    return total_size
```

**Required Fix**:
```python
def _get_dir_size(self, path: Path) -> int:
    """Get total size of directory"""
    total_size = 0
    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception as e:
        logger.warning(f"Error calculating directory size for {path}: {e}")
    return total_size
```

---

### 13. **ETL Data Validator Test Suite Missing** ❌ NOT IMPLEMENTED
**Location**: `tests/etl/test_data_validator.py` (MISSING)  
**Status**: `bash/comprehensive_brutal_test.sh` (Nov 12, 2025) warns that this test file is absent, so the ETL suite skips all validator coverage.  
**Priority**: MEDIUM - Required to keep QA parity with documented validation steps.

**Required Implementation**:
```python
# NEW: tests/etl/test_data_validator.py (~20 tests)
import pandas as pd
import pytest

from etl.data_validator import DataValidator

class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator()

    def test_positive_prices_required(self):
        df = pd.DataFrame({'Open': [1, -2], 'High': [2, 3], 'Low': [0.5, 0.2], 'Close': [1.5, -1], 'Volume': [100, 200]})
        report = self.validator.validate_ohlcv(df)
        assert not report['passed']
        assert any('negative/zero prices' in err for err in report['errors'])

    # ... additional tests for volume non-negativity, missing data thresholds,
    # outlier detection, custom price column validation, etc.
```

**Success Criteria**:
- [ ] File restored under `tests/etl/`
- [ ] Price positivity, volume non-negativity, and missing-data checks covered
- [ ] Pytest suite (`bash/comprehensive_brutal_test.sh`) no longer emits WARN for missing validator tests
- [ ] Added to CI to guard against future regressions

**Test Impact**: Until this file is restored, the brutal suite halts before the Time Series forecasting block (Broken pipe timeout on Nov 12, 2025). Reinstating this test suite is the gating item before re-running the comprehensive tests and capturing TS/LLM regression evidence.

---

## 📊 Implementation Priority Matrix

| Component | Priority | Effort | Dependencies | Status |
|-----------|----------|--------|--------------|--------|
| cTrader Client | CRITICAL | High | None | ✅ Completed |
| Order Manager | CRITICAL | High | cTrader Client | ✅ Completed |
| Performance Dashboard | HIGH | Medium | Database | ❌ Not Started |
| Production Deployer | HIGH | Medium | All Components | ❌ Not Started |
| Disaster Recovery | HIGH | Medium | Order Manager | ❌ Not Started |
| Risk Manager Actions | MEDIUM | Low | Order Manager | 🟡 Partial |
| Time-Series Features | MEDIUM | Medium | Database | ❌ Not Started |
| Parallel Model Runner | MEDIUM | Medium | All Models | ❌ Not Started |
| Data Validator Test Suite | MEDIUM | Low | DataValidator | ❌ Not Started |
| Time-Series Validation | MEDIUM | High | Models | ❌ Not Started |
| Exception Logging | LOW | Low | None | 🟡 Partial |

---

## 🎯 Implementation Roadmap

### Phase 1: Critical Trading Infrastructure (Weeks 1-2)
1. ~~**IBKR Client**~~ ➜ **cTrader Client** (Week 1, Days 1-3) ✅ Completed
   - Demo + live Open API endpoints with OAuth token refresh
   - Dataclass order payloads + placement responses
   - Environment-driven credential loader (`execution/ctrader_client.py`)
   - Error handling, retries, and session lifecycle management

2. **Order Manager** (Week 1, Days 4-5) ✅ Completed
   - Pre-trade checks (confidence, free margin, daily limits, circuit breakers)
   - Order lifecycle orchestration via `CTraderClient`
   - Fill persistence through `DatabaseManager.save_trade_execution`
   - Risk manager integration for automatic gating

3. **Risk Manager Integration** (Week 2, Days 1-2)
   - Connect automatic actions to Order Manager
   - Test position closure
   - Test position reduction

### Phase 2: Monitoring & Deployment (Weeks 3-4)
4. **Performance Dashboard** (Week 3, Days 1-3)
   - Web interface
   - Real-time metrics
   - Historical charts
   - Alert visualization

5. **Production Deployer** (Week 3, Days 4-5)
   - Environment validation
   - Health checks
   - Component deployment
   - Rollback mechanism

6. **Disaster Recovery** (Week 4, Days 1-3)
   - Failure detection
   - Automatic failover
   - Recovery procedures
   - Testing

### Phase 3: Advanced Features (Weeks 5-6)
7. **Time-Series Features** (Week 5, Days 1-2)
   - Feature engineering
   - Seasonal decomposition
   - Database persistence

8. **Parallel Model Runner** (Week 5, Days 3-5)
   - Async execution
   - Signal normalization
   - Provenance tracking

9. ~~**Signal Router** (Week 6, Days 1-2)~~ ✅ Completed Nov 2025
    - Feature flags (TS-primary, LLM fallback) implemented in `models/signal_router.py`
    - Stage planner reordered in `scripts/run_etl_pipeline.py`

10. **Time-Series Validation** (Week 6, Days 3-5)
    - Walk-forward CV
    - Statistical tests
    - Result persistence

### Phase 4: Polish & Testing (Week 7)
11. **Exception Logging** (Week 7, Day 1)
    - Fix silent exception handlers
    - Add proper logging

12. **Integration Testing** (Week 7, Days 2-5)
    - End-to-end tests
    - Performance benchmarks
    - Documentation updates

---

## ✅ Success Criteria

### Critical Components
- [ ] cTrader demo account connected (runtime validation + 50 sample trades)
- [ ] Orders can be placed and monitored
- [ ] Risk manager can automatically close positions
- [ ] Performance dashboard shows live metrics
- [ ] Production deployment pipeline working
- [ ] Disaster recovery tested and operational

### Advanced Features
- [ ] Time-series features generated and stored
- [ ] Models execute in parallel
- [ ] Signal router maintains backward compatibility
- [ ] Validation framework produces reliable metrics

### Code Quality
- [ ] All exception handlers log properly
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Performance benchmarks met

---

## 📝 Notes

1. **Dependencies**: Many components depend on others. Follow the dependency order in the roadmap.

2. **Testing**: Each component should have unit tests before integration.

3. **Documentation**: Update relevant documentation files as components are completed.

4. **Backward Compatibility**: Maintain backward compatibility, especially for signal routing.

5. **Feature Flags**: Use feature flags for gradual rollout of new features.

---

**Last Updated**: 2025-11-12  
**Status**: ACTIVE  
**Next Review**: After brutal test suite completes without warnings/timeouts
