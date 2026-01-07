# Optimization & Hardening Backlog

Use this as a menu of optimizations; only pull items when profitability or stability impact is proven and in alignment with `Documentation/AGENT_INSTRUCTION.md` phase gates/Core Directive. Favor configuration-driven, incremental changes validated by backtests and targeted benchmarks.

## Performance & Scalability
### Data Processing
- Vectorize loops in `_split_ticker_frame` and quality metrics; eliminate per-row iteration in hot paths.
- Chunk large OHLCV windows to control memory when handling many tickers (already available via `DataSourceManager.extract_ohlcv(..., chunk_size=...)` or `DATA_SOURCE_CHUNK_SIZE`).
- Parallelize ticker processing across tickers (thread/process pools) with deterministic ordering and safe resource limits.
- Add database connection pooling to reuse connections across calls.

### Caching Enhancements
- Smart invalidation that respects market hours and weekends (avoid rigid 24h expiry).
- Incremental/delta updates for recent data instead of full refetches.
- Cache warming for frequently accessed tickers/forecasts.

## Code Quality & Maintainability
### Architecture
- Consolidate the 30+ YAML configs with layered includes/overrides; reduce configuration explosion.
- Decompose the ~700-line `main()` into focused orchestrator components (e.g., `DataManager`, `SignalRouter`, `ExecutionEngine`) with single responsibilities.
- Normalize exception handling patterns across modules.
- Add type hints in critical paths to improve safety and readability.

### Code Structure Example
```python
class TradingOrchestrator:
    def __init__(self, config):
        self.data_manager = DataManager(config)
        self.signal_router = SignalRouter(config)
        self.execution_engine = ExecutionEngine(config)

    def run_trading_cycle(self):
        # Focused single responsibility; avoid monoliths
        ...
```

## Reliability & Robustness
- Circuit breakers around external APIs to prevent cascading failures.
- Graceful degradation with fallbacks when components fail; more robust bar-state persistence.
- Testing gaps: expand end-to-end workflow coverage, add performance regression tests, and introduce chaos/failure-injection scenarios.

## Monitoring & Observability
### Metrics & Alerting
- Replace file-based dashboards with real-time monitoring.
- Track latency/throughput and granular PnL (per-strategy, per-timeframe).

### Logging Improvements
```python
logger.info(
    "trade.executed",
    extra={
        "ticker": ticker,
        "action": action,
        "price": price,
        "pnl": realized_pnl,
        "strategy": strategy_name,
    },
)
```

## Security & Compliance
- Move API key management to secrets tooling with rotation; avoid long-lived env vars.
- Add data anonymization for sensitive trading data and an audit trail for compliance.

## Resource Optimization
### Memory Management
- Support streaming/chunked extraction instead of loading full history into memory.
```python
for chunk in extractor.extract_data_chunked(ticker, start_date, end_date):
    process_chunk(chunk)
```

### CPU/GPU Optimization
- Batch model inference across tickers when possible.
- Consider GPU acceleration for heavy forecasting workloads when justified by benchmarks and budget.

## Specific Implementation Sketches
1. **Parallel ticker processing**
```python
from concurrent.futures import ThreadPoolExecutor

def process_ticker_batch(ticker_list):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_ticker, ticker) for ticker in ticker_list]
        return [f.result() for f in futures]
```

2. **Configuration consolidation**
```yaml
portfolio_maximizer:
  data_sources: !include data_sources.yml
  forecasting: !include forecasting.yml
  execution: !include execution.yml
  monitoring: !include monitoring.yml
```

3. **Event-driven architecture**
```python
class MarketDataEvent:
    def __init__(self, ticker, ohlcv_data):
        self.ticker = ticker
        self.data = ohlcv_data

class TradingEventHandler:
    def on_market_data(self, event: MarketDataEvent):
        # Trigger forecasting and signal generation
        ...
```

These options target performance, maintainability, scalability, and robustness while preserving the quantitative guardrails and cost constraints defined in the main agent instructions.
