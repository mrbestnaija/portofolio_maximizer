# Performance & Scalability Optimization TODO

**Date**: 2026-01-07  
**Status**: 🟡 READY FOR IMPLEMENTATION  
**Priority**: HIGH - Performance optimization aligned with project guidelines  
**Project Phase**: Post-Core Implementation Enhancement  

---

## Implementation Guidelines Compliance

This optimization plan follows the established project patterns documented in:
- `Documentation/PROJECT_STATUS.md` - Current baseline verification
- `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` - Sequenced implementation approach
- `Documentation/AGENT_INSTRUCTION.md` - Approved time-series stack and vectorization requirements
- `Documentation/implementation_checkpoint.md` - Evidence-driven development standards

**Key Constraints**:
- Must maintain current test coverage (141+ tests passing)
- Follow incremental, reversible changes per PROJECT_WIDE_OPTIMIZATION_ROADMAP
- Use approved Tier-1 stack: NumPy/pandas vectorization, statsmodels, arch
- Evidence-based validation after each phase
- Preserve existing brutal test suite compatibility

---

## Phase 1: Data Processing Performance Optimization (Week 1)

### 1.1 Vectorization Audit & Enhancement ⚡
**Priority**: CRITICAL - Core performance foundation

**Current State Analysis**:
- ✅ `etl/yfinance_extractor.py` uses vectorized pandas operations
- ⚠️ `scripts/run_auto_trader.py` has sequential ticker processing (lines 1107-1233)
- ⚠️ Quality metrics computation uses row-by-row operations

**Implementation Tasks**:
```python
# Target: _split_ticker_frame vectorization
- [ ] Replace sequential ticker frame splitting with vectorized operations
- [ ] Optimize quality metrics calculation using pandas.agg()
- [ ] Batch forecast generation across tickers where possible
- [ ] Add performance benchmarking to brutal test suite

# Evidence Requirements:
- [ ] Benchmark before/after on 10+ tickers (current vs optimized)
- [ ] Document speedup metrics in logs/performance/optimization_*.json
- [ ] Verify no regression in forecast accuracy or signal quality
```

**Test Integration**:
```bash
# Verification commands per PROJECT_WIDE_OPTIMIZATION_ROADMAP
./simpleTrader_env/bin/python -m py_compile scripts/run_auto_trader.py
./simpleTrader_env/bin/python -m pytest -q tests/etl/test_yfinance_extractor.py
CYCLES=1 SLEEP_SECONDS=0 ENABLE_LLM=0 bash bash/run_auto_trader.sh
```

### 1.2 Memory Optimization 🧠
**Priority**: HIGH - Scalability foundation

**Current Issues**:
- Large OHLCV windows loaded entirely into memory; chunking support landed (2026-01-07) via `DataSourceManager.extract_ohlcv(..., chunk_size=...)` or `DATA_SOURCE_CHUNK_SIZE`, but defaults still pull full ticker lists unless configured per run.
- Database connections not pooled

**Implementation Tasks**:
```python
# Target: Streaming data processing
- [x] Implement chunked OHLCV processing in DataSourceManager (chunk_size param + DATA_SOURCE_CHUNK_SIZE; covered by tests/etl/test_data_source_manager_chunking.py)
- [ ] Add memory usage monitoring to performance dashboard
- [ ] Optimize DataFrame memory footprint using categorical dtypes
- [ ] Implement connection pooling in DatabaseManager

# Memory Targets:
- [ ] <500MB memory usage for 50 tickers, 2 years daily data
- [ ] <2GB for full frontier market universe processing
- [ ] Document memory benchmarks in brutal test artifacts
```

### 1.3 Caching Enhancement 🚀
**Priority**: MEDIUM - Building on existing 20x speedup

**Current State**: 24h validity, Parquet format, 100% hit rate after first run

**Enhancement Tasks**:
```yaml
# Target: Smart cache invalidation
- [x] Implement market-hours-aware cache invalidation (default-on)
- [x] Add incremental update capability (delta vs full refresh) via `ENABLE_CACHE_DELTAS=1`
- [x] Cache warming for frequently accessed tickers (`ENABLE_CACHE_WARMING=1`, optional list via `CACHE_WARM_TICKERS`)
- [ ] Compressed cache storage optimization

# Cache Performance Targets:
- [ ] <50ms cache hit latency for any ticker
- [ ] <5s cache miss with incremental update
- [ ] Document cache analytics in monitoring dashboard
```

---

## Phase 2: Parallel Processing Architecture (Week 2)

### 2.1 Ticker-Level Parallelization ⚙️
**Priority**: HIGH - Core scalability improvement

**Implementation Strategy**:
```python
# Target: ThreadPoolExecutor for I/O bound operations
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelTradingOrchestrator:
    def __init__(self, max_workers=4):
        self.max_workers = min(max_workers, len(os.sched_getaffinity(0)))
    
    def process_tickers_parallel(self, ticker_list):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_ticker, ticker): ticker 
                for ticker in ticker_list
            }
            return self._collect_results(futures)

# Implementation Tasks:
- [x] Refactor run_auto_trader.py ticker loop for parallel execution (candidate prep + forecast)
- [ ] Add parallel data extraction in DataSourceManager  
- [x] Implement thread-safe logging and metrics collection (DB writes/execution remain sequential)
- [x] Add parallelization config flags for testing/debugging
```

**Safety Requirements**:
```python
# Thread Safety Checklist:
- [ ] DatabaseManager connection per thread or pooling
- [x] Thread-safe signal router state management (router usage remains sequential)
- [x] Isolated forecast model instances per worker
- [x] Atomic metrics aggregation (per-cycle ordering preserved)

**Parallel Defaults & Evidence (2026-01-07)**:
- `ENABLE_PARALLEL_TICKER_PROCESSING` + `ENABLE_PARALLEL_FORECASTS` now default to on (override via env).
- Stress evidence: `logs/automation/stress_parallel_20260107_202403/comparison.json` shows matching outputs and faster parallel elapsed time.
```

### 2.2 Model Inference Optimization 🤖
**Priority**: MEDIUM - Computational efficiency

**Current State**: Sequential SARIMAX/GARCH/SAMoSSA execution per ticker

**Implementation Tasks**:
```python
# Target: Batch model inference
- [ ] Implement batched forecast generation where models support it
- [ ] Optimize SARIMAX parameter estimation with warm starts
- [ ] Add model result caching for repeated parameter sets
- [ ] Profile bottlenecks in forcester_ts ensemble execution

# Performance Targets:
- [ ] 50% reduction in total forecasting time for 10+ ticker universe
- [ ] <30s end-to-end execution for full brutal test suite
- [ ] Document model performance in logs/forecast_audits/
```

---

## Phase 3: Architecture Improvements (Week 3)

### 3.1 Event-Driven Architecture 📡
**Priority**: MEDIUM - Foundation for real-time processing

**Current State**: Polling-based data updates in trading loop

**Implementation Design**:
```python
# Target: Event-driven market data processing
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class MarketDataEvent:
    ticker: str
    timestamp: datetime
    ohlcv_data: pd.DataFrame
    source: str

class EventHandler(ABC):
    @abstractmethod
    async def handle_event(self, event: MarketDataEvent) -> None:
        pass

class TradingEventProcessor:
    def __init__(self):
        self.handlers = []
        self.event_queue = asyncio.Queue()
    
    async def process_market_data(self, event: MarketDataEvent):
        # Trigger forecasting, signal generation, execution
        await self.event_queue.put(event)

# Implementation Tasks:
- [ ] Design event schema for market data updates
- [ ] Implement async event processing pipeline
- [ ] Add event-based signal triggering
- [ ] Maintain backward compatibility with polling mode
```

### 3.2 Configuration Consolidation 📋
**Priority**: MEDIUM - Maintainability improvement

**Current Issues**: 30+ YAML files create complexity

**Consolidation Strategy**:
```yaml
# Target: Unified configuration with includes
portfolio_maximizer:
  # Core settings with environment overrides
  data_sources: !include data_sources.yml
  forecasting: !include forecasting.yml
  execution: !include execution.yml
  monitoring: !include monitoring.yml
  
  # Environment-specific overlays
  environments:
    development: !include config/dev_overrides.yml
    production: !include config/prod_overrides.yml
    testing: !include config/test_overrides.yml

# Implementation Tasks:
- [ ] Create unified config schema with validation
- [ ] Implement hierarchical config loading with overrides
- [ ] Add config validation with clear error messages
- [ ] Maintain backward compatibility during transition
```

---

## Phase 4: Monitoring & Observability Enhancement (Week 4)

### 4.1 Real-Time Performance Monitoring 📊
**Priority**: HIGH - Production readiness requirement

**Current State**: File-based dashboard updates, basic logging

**Enhancement Tasks**:
```python
# Target: Structured monitoring with metrics
import structlog
from prometheus_client import Counter, Histogram, Gauge

class PerformanceMonitor:
    def __init__(self):
        self.trade_counter = Counter('trades_total', 'Total trades executed', ['ticker', 'action'])
        self.latency_histogram = Histogram('operation_duration_seconds', 
                                         'Operation latency', ['operation'])
        self.pnl_gauge = Gauge('portfolio_pnl_dollars', 'Current PnL in USD')
    
    def record_trade(self, ticker: str, action: str, latency: float):
        self.trade_counter.labels(ticker=ticker, action=action).inc()
        self.latency_histogram.labels(operation='trade_execution').observe(latency)

# Implementation Tasks:
- [ ] Add structured logging with correlation IDs
- [ ] Implement real-time metrics collection
- [ ] Create monitoring dashboard with live updates
- [ ] Add alerting for performance degradation
```

### 4.2 Business Metrics Tracking 💰
**Priority**: HIGH - Trading performance visibility

**Implementation Tasks**:
```python
# Target: Granular PnL tracking
- [ ] Per-strategy PnL attribution
- [ ] Per-timeframe performance metrics
- [ ] Slippage and execution cost tracking
- [ ] Real-time drawdown monitoring

# Metrics Dashboard Requirements:
- [ ] Live equity curve updates
- [ ] Rolling Sharpe/Sortino ratios
- [ ] Win rate and profit factor trends
- [ ] Model performance attribution
```

---

## Phase 5: Resource Optimization (Week 5)

### 5.1 CPU Optimization 🔧
**Priority**: MEDIUM - Computational efficiency

**Optimization Targets**:
```python
# Target: CPU-intensive operations
- [ ] Profile forecast model computation bottlenecks
- [ ] Optimize statistical test calculations in validation
- [ ] Implement efficient rolling window operations
- [ ] Add CPU affinity configuration for production

# Performance Benchmarks:
- [ ] <10% CPU usage during normal operation
- [ ] <50% CPU during forecast generation
- [ ] Efficient utilization across available cores
```

### 5.2 Memory Management 💾
**Priority**: MEDIUM - System stability

**Implementation Tasks**:
```python
# Target: Memory leak prevention and optimization
- [ ] Add memory profiling to brutal test suite
- [ ] Implement garbage collection tuning
- [ ] Optimize large DataFrame operations with chunking
- [ ] Add memory leak detection in continuous runs

# Memory Targets:
- [ ] Stable memory usage over extended runs (>24h)
- [ ] <100MB growth per hour during continuous operation
- [ ] Graceful handling of memory pressure
```

---

## Testing & Validation Requirements

### Performance Regression Tests 🧪
Following PROJECT_STATUS.md verification patterns:

```bash
# Pre-implementation baseline capture
./simpleTrader_env/bin/python -c "
import time
start = time.time()
exec(open('scripts/run_etl_pipeline.py').read())
print(f'Baseline ETL time: {time.time() - start:.2f}s')
"

# Performance benchmark suite
./simpleTrader_env/bin/python -m pytest tests/performance/ -v --benchmark-only

# Memory usage validation
./simpleTrader_env/bin/python scripts/run_auto_trader.py --cycles 1 --memory-profile

# Brutal test with performance monitoring
PERFORMANCE_MONITORING=1 bash/comprehensive_brutal_test.sh
```

### Evidence Collection Standards 📋
Per implementation_checkpoint.md patterns:

```bash
# Performance artifacts to generate:
- logs/performance/optimization_baseline_YYYYMMDD.json
- logs/performance/optimization_results_YYYYMMDD.json  
- logs/performance/memory_profile_YYYYMMDD.json
- logs/performance/latency_benchmarks_YYYYMMDD.json

# Comparison requirements:
- Before/after execution time for key operations
- Memory usage patterns over time
- Accuracy preservation verification
- Throughput improvements quantified
```

---

## Success Criteria & Metrics

### Phase 1 Success Metrics:
- [ ] **50%+ reduction** in ticker processing time for 10+ ticker universe
- [ ] **<500MB memory usage** for standard brutal test run
- [ ] **Zero regression** in forecast accuracy or signal quality
- [ ] **All existing tests pass** with no behavioral changes

### Phase 2 Success Metrics:
- [ ] **2-4x throughput improvement** with parallel processing
- [ ] **Linear scalability** up to available CPU cores
- [ ] **Thread safety verified** through stress testing
- [ ] **Robust error handling** in concurrent scenarios

### Phase 3 Success Metrics:
- [ ] **Event processing latency <100ms** for market data updates
- [ ] **90% reduction** in configuration complexity/duplication
- [ ] **Backward compatibility maintained** for existing workflows
- [ ] **Clear migration path** documented

### Phase 4 Success Metrics:
- [ ] **Real-time monitoring** with <5s update intervals
- [ ] **Comprehensive alerting** on performance degradation
- [ ] **Rich business metrics** for trading performance analysis
- [ ] **Production-ready observability** stack

### Phase 5 Success Metrics:
- [ ] **<10% baseline CPU usage** during normal operation
- [ ] **Stable memory profile** over extended runs
- [ ] **Efficient resource utilization** across hardware profiles
- [ ] **Graceful degradation** under resource pressure

---

## Risk Mitigation & Rollback Strategy

### Implementation Safety:
- **Feature flags**: All optimizations behind configurable flags
- **Incremental rollout**: One phase at a time with validation
- **Performance monitoring**: Continuous measurement during implementation
- **Automated testing**: Extended brutal test coverage for regressions

### Rollback Procedures:
```bash
# Immediate rollback capability
git checkout optimization_baseline
./simpleTrader_env/bin/python -m pytest -q tests/
CYCLES=1 bash bash/run_auto_trader.sh  # Verify baseline functionality

# Selective feature rollback via config
export ENABLE_PARALLEL_PROCESSING=false
export ENABLE_EVENT_DRIVEN=false
export ENABLE_ADVANCED_CACHING=false
```

---

## Implementation Timeline

**Week 1**: Data Processing & Vectorization Optimization  
**Week 2**: Parallel Processing Architecture  
**Week 3**: Event-Driven Design & Configuration Consolidation  
**Week 4**: Monitoring & Observability Enhancement  
**Week 5**: Resource Optimization & Final Integration  

**Verification After Each Week**:
- Run complete test suite
- Execute 1-cycle auto-trader validation
- Review performance metrics vs baseline
- Update documentation with evidence

This optimization plan maintains alignment with the project's established patterns while delivering significant performance improvements through incremental, testable changes.
