# Stub & Incomplete Implementation Plan

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

**Comprehensive Review and Replacement Strategy**

**Date**: 2025-11-12  
**Status**: 🟢 UNBLOCKED – 2026-01-07 (see `Documentation/PROJECT_STATUS.md`)  
**Priority**: HIGH - Complete all stubs before production deployment
**Current status (2026-01-07)**: Core trading, TS wiring, and quant monitoring are implemented; remaining partial items are dashboard web UI/scheduling, MTM options pricing (Phase 3), NAV allocator/risk buckets, and demo-to-live promotion evidence.

### 🔄 2025-11-24 Delta (currency update)
- New data-source-aware ticker resolver (`etl/data_universe.py`) wired into `scripts/run_auto_trader.py`; explicit + frontier remains default, provider discovery optional when no inputs.
- LLM fallback defaults to on in the trading loop; guardrails unchanged.
- Dashboard JSON emission hardened (ISO timestamps) to stop serialization warnings during live runs.
- Barbell integration tracked via `BARBELL_INTEGRATION_TODO.md`; future stub replacements around portfolio math, paper trading, and NGX/frontier exposure must respect safe/risk buckets and tail-aware evaluation (Sortino/Omega/CVaR + scenarios).

### 2025-12-17 Delta (synthetic-first pre-production)
- Free-tier liquidity gaps (commodities/illiquid classes) keep pre-production in **synthetic-first** mode. Use `config/data_sources_config.yml` provider `synthetic` + `etl/synthetic_extractor.py`, `scripts/generate_synthetic_dataset.py`, `scripts/validate_synthetic_dataset.py`, and `bash/run_gpu_parallel.sh MODE=synthetic` to generate/validate persisted datasets for brutal/regression runs.
- Isolation rules: `ENABLE_SYNTHETIC_PROVIDER=1`/`SYNTHETIC_ONLY=1` for tests; remove synthetic flags and point `PORTFOLIO_DB_PATH` back to production before live. Synthetic outputs must not enter live trading, dashboards, or training once validation ends (`AGENT_INSTRUCTION.md`).
- Logging/retention: synthetic generation/validation append to `logs/automation/synthetic_runs.log` with auto-prune at 14 days via `scripts/prune_synthetic_logs.py`, aligning with `CHECKPOINTING_AND_LOGGING.md`, `QUANT_VALIDATION_MONITORING_POLICY.md`, and `TIME_SERIES_FORECASTING_IMPLEMENTATION.md`.
- Pipeline readiness: `scripts/run_etl_pipeline.py --execution-mode synthetic --data-source synthetic` now loads persisted datasets via `SYNTHETIC_DATASET_ID`/`SYNTHETIC_DATASET_PATH`, logs `dataset_id`/`generator_version` in `PipelineLogger`, and falls back to deterministic in-process generation if none is provided.
- Smoke verification 2025-12-17: `pipeline_20251217_220920` ran end-to-end on `syn_1dcce391f1ea` using the synthetic provider after refreshing `numpy`/`scipy`/`pyarrow`; only warnings were CV fold overlap/drift and missing viz deps (`kiwisolver`), no stage failures.

### 2025-12-18 Delta (dashboard snapshot)
- Implemented a performance dashboard snapshot generator in `monitoring/performance_dashboard.py`. It computes trade + risk metrics from `DatabaseManager`, emits JSON/CSV artifacts, and surfaces basic alerts for data quality, drawdown, and latency. Web UI + scheduled refresh remain pending.
- Stub wiring additions:
  - Added deployment helper `deployment/production_deploy.py` (env/API validation, health check, dashboard snapshot hook).
  - Added disaster recovery helper `recovery/disaster_recovery.py` (data-source failover, broker/risk mitigations, model fallback).
  - Added TS utilities: `etl/time_series_feature_builder.py`, `models/time_series_runner.py`, and `analysis/time_series_validation.py`.
  - Minor hardening: SARIMAX diagnostics now log instead of pass; cache manager warns on dir size errors.

### Nov 12, 2025 Update
- Time Series signal generator refactor is exercised via logs/ts_signal_demo.json; stubs for router/broker now depend on these BUY/SELL payloads rather than HOLD placeholders.
- Checkpoint/metadata utilities are stable on Windows thanks to Path.replace, unblocking future stub work that writes checkpoints repeatedly.
- Validator/backfill stubs can assume scripts/backfill_signal_validation.py is callable from scheduled jobs; only the scheduler glue remains.

### Nov 15, 2025 Frontier Coverage Update
- Added `etl/frontier_markets.py` + `--include-frontier-tickers` plumbing so every stub exercise in `.bash/`/`.script/` land includes the Nigeria → Bulgaria ticker atlas provided in the liquidity guide. Synthetic runs remain default until live ticker suffix mapping (e.g., NGX `.LG`) is finalized; document whether each stub validation used synthetic or live data.
- Update stub replacement PRDs to reference `Documentation/arch_tree.md` and `README.md` for the canonical frontier list—future placeholder removals must preserve this coverage requirement.

### Nov 18, 2025 SQLite Recovery
- `etl/database_manager.py` now auto-backs up corrupted SQLite files and rebuilds them when `database disk image is malformed` trips the brutal suite. Any stub work touching persistence (checkpointing, validation backfills, trade logs) should rely on this code path instead of crafting ad-hoc repair logic.

### Nov 20, 2025 Quant Validation Automation Update
- Quant validation automation helpers are now scaffolded as standalone, read-only CLIs:
  - `scripts/sweep_ts_thresholds.py` sweeps `(confidence_threshold, min_expected_return)` and writes realised performance summaries to `logs/automation/ts_threshold_sweep.json`.
  - `scripts/estimate_transaction_costs.py` estimates commission / transaction cost statistics by ticker or asset class and writes `logs/automation/transaction_costs.json`.
  - `scripts/generate_config_proposals.py` ingests both artifacts and produces `logs/automation/config_proposals.json` with **proposed** TS threshold and cost-aware `min_expected_return` adjustments for human review.
- These helpers are wired into the roadmap via `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md` and cron examples in `Documentation/CRON_AUTOMATION.md`; they provide the evidence layer for tightening `time_series.min_expected_return` and `quant_validation.success_criteria.min_expected_profit` without hardcoding new thresholds in code.

### Heuristics, Full Models, and Future ML Calibrators
- Stub replacements must respect the three-layer stack described in `Documentation/AGENT_INSTRUCTION.md` and `Documentation/arch_tree.md`:
  - **Heuristics** (fast, transparent): quant validation GREEN/YELLOW/RED tiers, MVS summaries, barbell gates and simple NAV guards – used for real-time monitoring and quick routing/guardrail decisions.
  - **Full models** (authoritative): TS ensemble + portfolio math, auto trader, backtesting and strategy optimisation – the single source of truth for official NAV, drawdown, and PnL-based research.
  - **ML calibrators** (future): regime-aware learning around heuristics (e.g., thresholds, bands), driven by full-model outcomes and documented in `Documentation/STOCHASTIC_PNL_OPTIMIZATION.md` and `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md`; never silently replace TS/portfolio engines.
- Any new stubs added to this file should clearly indicate which layer they belong to and avoid introducing undocumented ML behaviour before spot-only TS profitability and quant health (GREEN/YELLOW) are demonstrated in brutal reports.

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
- ✅ `2025-12-28` note: Brutal suite is green; this regression block is historical only (see `Documentation/PROJECT_STATUS.md`).

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
- ✅ UPDATE (Dec 28, 2025): Missing validator test restored and brutal suite no longer fails with `Broken pipe` (see `Documentation/PROJECT_STATUS.md` for the latest brutal run evidence).
- ✅ UPDATE (Nov 16, 2025): `forcester_ts/instrumentation.py` now emits dataset snapshots plus RMSE/sMAPE/tracking-error benchmarks for every forecast run, and `TimeSeriesVisualizer` renders those summaries on dashboards. Future stub replacements must consume `logs/forecast_audits/*.json` instead of inventing ad-hoc diagnostics.

---

## 🔍 Codebase Analysis Results

### Files Reviewed: 100+ Python files
### Stubs Identified: 15+ incomplete implementations
### Priority: CRITICAL - Blocking production deployment

---

## 🚨 CRITICAL STUBS (Must Complete Before Production)

### 1. **Broker Integration - cTrader Client** ✅ IMPLEMENTED
**Location**: `execution/ctrader_client.py`  
**Status**: Replaces the missing massive.com/polygon.io stub with a demo-first cTrader Open API
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

### 3. **Production Performance Dashboard** 🟡 PARTIAL
**Location**: `monitoring/performance_dashboard.py`  
**Status**: Snapshot generator implemented (JSON/CSV export + alerts); web UI and scheduler pending  
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
 - Existing heuristic outputs (MVS summaries, quant health, barbell gates) as documented in `Documentation/arch_tree.md`, `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`, and `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` – the dashboard should surface these **heuristics alongside full-model metrics**, not invent new, untracked rules.

**Success Criteria**:
- [ ] Live dashboard operational
- [ ] Metrics updating every 1 minute
- [ ] Historical charts (30-day, 90-day, 1-year)
- [ ] Alert visualization
- [x] Export to CSV/JSON
- [ ] Web interface accessible

---

### 4. **Production Deployment Pipeline** 🟡 PARTIAL
**Location**: `deployment/production_deploy.py`  
**Status**: Deploy helper added (env/API key validation, health check, dashboard emit hooks); monitoring/rollback remain  
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

### 5. **Disaster Recovery System** 🟡 PARTIAL
**Location**: `recovery/disaster_recovery.py`  
**Status**: Implemented failover hooks (data source switch, model fallback, broker/risk mitigation); needs deeper integration tests  
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

### 7. **Time-Series Feature Builder** 🟡 PARTIAL
**Location**: `etl/time_series_feature_builder.py`  
**Status**: Implemented lags/returns/rolling stats/seasonal decomposition; persistence to parquet; DB wiring optional  
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
- [x] Lag features created
- [x] Seasonal decomposition working
- [x] Holiday effects included
- [x] Rolling statistics calculated
- [ ] Features persisted to database
- [ ] Integration with SAMOSSA/SARIMAX

---

### 8. **Parallel Model Runner** 🟡 PARTIAL
**Location**: `models/time_series_runner.py`  
**Status**: Implemented async runner with SARIMAX/GARCH/LLM hooks; normalization + provenance light; full error handling still needed  
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
- [x] Parallel execution working
- [ ] All models execute concurrently
- [ ] Unified signal schema output
- [ ] Provenance metadata attached
- [ ] Error handling for failed models
- [ ] Performance benchmarks <5s total

---

### 10. **Time-Series Validation Framework** 🟡 PARTIAL
**Location**: `analysis/time_series_validation.py`  
**Status**: Walk-forward CV scaffold added with MAE/RMSE/profit factor metrics; persistence to logs/automation/time_series_validation.json  
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
- Existing CV/quant infrastructure:
  - Rolling/blocked CV patterns from `TIME_SERIES_CV.md`.
  - Numeric/scaling invariants from `Documentation/NUMERIC_INVARIANTS_AND_SCALING_TESTS.md`.
  - Quant monitoring & automation from `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` and `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md`.
  - The framework should orchestrate and report on these **full-model evaluations** (TS ensemble + portfolio metrics), not replace the underlying forecaster or quant success helper.

**Success Criteria**:
- [x] Walk-forward validation working
- [x] Profit factor calculated
- [ ] Statistical tests implemented
- [ ] Results persisted to database
- [ ] Integration with dashboards

---

## 🔧 MINOR INCOMPLETE ITEMS (Low Priority)

### 11. **Exception Handlers with Pass** 🟢 DONE
**Location**: `forcester_ts/sarimax.py` (Lines 227, 234)  
**Status**: Exception handlers now log warnings and preserve diagnostics  
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

### 12. **Cache Manager - Exception Handler** 🟢 DONE
**Location**: `scripts/cache_manager.py` (Line 103)  
**Status**: Exception handler now logs warnings on directory size errors  
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

### 13. **ETL Data Validator Test Suite** ✅ IMPLEMENTED
**Location**: `tests/etl/test_data_validator.py`  
**Status**: Restored and exercised; brutal suite no longer warns about missing validator tests (see `Documentation/PROJECT_STATUS.md`).  
**Priority**: MEDIUM - Keep QA parity with documented validation steps.

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
- [x] File restored under `tests/etl/`
- [x] Price positivity, volume non-negativity, and missing-data checks covered
- [x] Brutal suite no longer emits WARN for missing validator tests
- [x] Included in CI/brutal coverage

---

### 14. **Higher-Order Hyper-Parameter Orchestration & Regime-Aware Backtesting** 🟡 PARTIAL
**Location**: `bash/run_post_eval.sh`, `scripts/run_strategy_optimization.py`, `backtesting/candidate_backtester.py`, `scripts/run_backtest_for_candidate.py`  
**Status**: Driver and plumbing in place; candidate-level evaluation and regime-aware Q-learning still stubbed  
**Priority**: HIGH – Controls how the system searches for profitable configurations

**Current State**:
- `bash/run_post_eval.sh` implements a higher-order hyperopt driver that:
  - Wraps ETL → auto-trader → strategy optimization in a stochastic loop.
  - Treats evaluation windows, `min_expected_profit`, and `time_series.min_expected_return`
    as tunable higher-order hyper-parameters.
  - Uses a bandit-like explore/exploit policy (30% explore / 70% exploit by default,
    adjusted after each trial) and logs trials to `logs/hyperopt/hyperopt_<RUN_ID>.log`.
  - Dynamically tightens candidate ranges based on historic profitability (AAPL, COOP, GC=F,
    EURUSD=X) using realistic expected-profit and expected-return bands.
- `etl/strategy_optimizer.py` and `scripts/run_strategy_optimization.py` support stochastic
  candidate sampling and scoring, but still rely on aggregate DB performance summaries per
  regime instead of full candidate-specific backtests.
- `backtesting/candidate_backtester.py` provides a simple rule-based simulator, and
  `scripts/run_backtest_for_candidate.py` exposes only a summary view of performance; neither
  is yet wired into the higher-order hyperopt loop as the canonical candidate evaluator.

**Required Completion**:
- Replace aggregate `DatabaseManager.get_performance_summary` usage in the optimizer’s
  `evaluation_fn` with calls into `backtesting/candidate_backtester.backtest_candidate` so
  each candidate is scored on its own simulated PnL (total_profit, profit_factor, win_rate,
  max_drawdown, total_trades).
- Extend `bash/run_post_eval.sh` to:
  - Select different hyper-parameter candidate grids per regime (e.g., default vs high-vol),
  - Support per-asset-class bands for `min_expected_profit` and `min_expected_return`
    (e.g., FX vs frontier equities) driven from config rather than hardcoded arrays.
- Implement a lightweight Q-table/bandit state (SQLite/JSON) that records
  `(regime, hyper-parameter combo) -> reward` and is reused across runs to bias sampling
  toward historically profitable regions instead of starting from a blank slate each time.
- Enhance `scripts/run_backtest_for_candidate.py` so it:
  - Accepts structured candidate JSON,
  - Invokes `backtesting/candidate_backtester.backtest_candidate` under the appropriate regime,
  - Returns a metrics dict compatible with `StrategyOptimizer` and the hyperopt driver.
- Add tests:
  - Unit tests for the higher-order driver (small synthetic DB) to confirm that best_score
    improves over a few rounds and that candidate ranges are respected,
  - Integration tests tying `bash/run_post_eval.sh` + `run_strategy_optimization.py` together
    under a fixed synthetic regime.

**Dependencies**:
- `etl/database_manager.py` (performance summary and equity curves)
- `backtesting/candidate_backtester.py`
- `Documentation/STOCHASTIC_PNL_OPTIMIZATION.md` (design contract for non-convex search)

**Success Criteria**:
- [ ] StrategyOptimizer’s evaluation function uses candidate-level backtests, not only aggregate DB metrics.
- [ ] Higher-order hyperopt respects regime- and asset-class-specific guardrails from config.
- [ ] Q-table/bandit state persists across runs and systematically biases sampling toward profitable regions.
- [ ] End-to-end hyperopt tests demonstrate improved realized PnL under stable regimes.

---

### 15. **Options/Derivatives Integration (Barbell-Constrained)** 🟡 IN PROGRESS
**Location**: `config/options_config.yml`, `config/barbell.yml`, `risk/barbell_policy.py`, `Documentation/BARBELL_OPTIONS_MIGRATION.md`, `Documentation/BARBELL_INTEGRATION_TODO.md`  
**Status**: Core config + barbell policy scaffold in place; options ETL/execution still feature-flagged and inert  
**Priority**: MEDIUM – Future extension once spot-only barbell is profitable

**Current State**:
- `config/options_config.yml` defines:
  - `options_trading.enabled` master toggle (default `false`),
  - barbell guardrails (`max_options_weight`, `max_premium_pct_nav`),
  - default selection bands for OTM options (moneyness, expiry),
  - per-asset-class limits (equity index, single-name equity, FX, commodity, frontier equity),
  - a first set of allowed underlyings for barbell-style options (AAPL, MSFT, SPY, QQQ).
- `config/barbell.yml` now provides the **global barbell shell**:
  - Safe and risk buckets (symbol lists) with `min_weight` / `max_weight` bounds,
  - Feature flags (`enable_barbell_allocation`, `enable_barbell_validation`, `enable_antifragility_tests`),
  - Per-market caps for EM/NGX/crypto inside the risk sleeve.
- `risk/barbell_policy.py` implements the initial `BarbellConfig` + `BarbellConstraint` helper:
  - `bucket_weights(weights) -> (w_safe, w_risk, w_other)` for instrumentation and analytics,
  - `project_to_feasible(weights)` that enforces barbell bounds when `enable_barbell_allocation: true`,
  - No behaviour change when barbell allocation is disabled (no-op projection).
- `Documentation/BARBELL_OPTIONS_MIGRATION.md` describes:
  - What OTM options are and why they are used as convex risk-leg instruments,
  - Data model extensions (options_quotes, underlyings, strikes, expiries, IV/Greeks),
  - Phased migration (O1–O4) from spot-only to options/derivatives under a barbell allocation.
- `Documentation/BARBELL_INTEGRATION_TODO.md` tracks the remaining barbell tasks (LLM/ML integration, antifragility suite) and is now partially satisfied at the config/policy layer.

**Required Completion**:
- Implement options ETL and storage alongside existing OHLCV tables (without changing spot behaviour when `options_trading.enabled` is false).
- Add feature-flagged options/synthetic convex strategies to the risk sleeve only, honouring barbell caps and quant-success guardrails.
- Extend brutal/backtest harness to include tail/antifragility tests for options strategies (premium-at-risk aware).
- Keep options codepaths entirely inert unless `ENABLE_OPTIONS=true` and `options_trading.enabled: true` are both set.

**Dependencies**:
- `etl/data_source_manager.py` and new options extractor(s)
- `risk/barbell_policy.py` (or equivalent barbell enforcement)
- `Documentation/BARBELL_OPTIONS_MIGRATION.md` for design reference

**Success Criteria**:
- [ ] Spot-only system remains unchanged when options are disabled.
- [ ] Options risk is always bounded by `max_options_weight` and `max_premium_pct_nav`.
- [ ] Options strategies pass brutal + antifragility tests before being considered for live/paper promotion.

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
| Higher-Order Hyperopt & Regime Backtesting | HIGH | High | Optimizer, Backtester | 🟡 Partial |
| Data Validator Test Suite | MEDIUM | Low | DataValidator | ❌ Not Started |
| Time-Series Validation | MEDIUM | High | Models | ❌ Not Started |
| Exception Logging | LOW | Low | None | 🟡 Partial |

---

## 🎯 Implementation Roadmap

### Phase 1: Critical Trading Infrastructure (Weeks 1-2)
1. ~~**massive.com/polygon.io Client**~~ ➜ **cTrader Client** (Week 1, Days 1-3) ✅ Completed
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

11. **Higher-Order Hyperopt & Regime Backtesting** (Week 6, Days 4-5)
    - Wire candidate-level backtests into `StrategyOptimizer` evaluation_fn
    - Enable per-regime/per-asset hyper-parameter grids from config
    - Persist Q-table/bandit state to bias future sampling

### Phase 4: Polish & Testing (Week 7)
12. **Exception Logging** (Week 7, Day 1)
    - Fix silent exception handlers
    - Add proper logging

13. **Integration Testing** (Week 7, Days 2-5)
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

