# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Portfolio Maximizer is an autonomous quantitative trading system that extracts financial data, forecasts market regimes, routes trading signals, and executes trades automatically. It's a production-ready Python system with institutional-grade ETL pipelines, LLM integration, and comprehensive testing.

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment (required for all operations)
source simpleTrader_env/bin/activate  # Linux/Mac
simpleTrader_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Build and Test Commands
```bash
# Run full test suite
pytest tests/

# Run tests with coverage
pytest tests/ --cov=etl --cov-report=html

# Run specific test categories
pytest tests/ -m "not slow"           # Skip slow tests
pytest tests/ -m integration          # Integration tests only
pytest tests/ -m security             # Security tests only

# Run tests for specific modules
pytest tests/etl/test_yfinance_cache.py -v
pytest tests/execution/test_order_manager.py -v
```

### Core Pipeline Operations
```bash
# Run ETL pipeline (main data processing)
python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --start 2020-01-01 --end 2024-01-01 --execution-mode auto --enable-llm

# Run autonomous trading loop
python scripts/run_auto_trader.py --tickers AAPL,MSFT,NVDA --lookback-days 365 --cycles 5

# Run pipeline with bash wrapper (recommended)
bash/run_pipeline.sh --mode live --tickers AAPL --enable-llm
bash/run_pipeline_live.sh  # Shortcut for live mode
bash/run_auto_trader.sh    # Autonomous trading with defaults
```

### Data Analysis and Validation
```bash
# Analyze time series data
python scripts/analyze_dataset.py --dataset data/training/training_*.parquet --column Close

# Generate visualizations
python scripts/visualize_dataset.py --dataset data/training/training_*.parquet --output-dir visualizations/

# Validate environment and dependencies
python scripts/validate_environment.py
```

### Synthetic Data and Testing
```bash
# Generate synthetic data for testing
python scripts/generate_synthetic_dataset.py

# Run brutal testing suite
bash/comprehensive_brutal_test.sh

# Quick smoke test
bash/run_synthetic_smoke.sh
```

## High-Level Architecture

### Core System Layers
The system follows a 7-layer architecture:

1. **Extraction Layer** (`etl/`): Multi-source data extraction with intelligent caching
   - `yfinance_extractor.py`: Yahoo Finance with 20x speedup caching
   - `data_source_manager.py`: Multi-source coordination with failover
   - `synthetic_extractor.py`: Synthetic data generation for testing

2. **Storage Layer** (`etl/data_storage.py`): Parquet-based storage with train/val/test splits

3. **Validation Layer** (`etl/data_validator.py`): Statistical validation and outlier detection

4. **Preprocessing Layer** (`etl/preprocessor.py`): Missing data handling and normalization

5. **Analysis Layer** (`etl/time_series_analyzer.py`): MIT-standard time series analysis

6. **Forecasting Layer** (`forcester_ts/`, `models/`): Multiple forecasting models
   - SARIMAX, GARCH, SAMoSSA (Singular Spectrum Analysis + RL)
   - Ensemble routing through `SignalRouter`

7. **Execution Layer** (`execution/`): Order management and paper trading
   - `paper_trading_engine.py`: Risk-managed position sizing
   - `order_manager.py`: Order lifecycle management

### Key Components

**Data Pipeline Orchestration:**
- `scripts/run_etl_pipeline.py`: Main orchestrator with CV, LLM integration
- Configuration-driven via YAML files in `config/`
- Checkpointing and resumption via `etl/checkpoint_manager.py`

**Autonomous Trading:**
- `scripts/run_auto_trader.py`: Continuous trading loop
- Real-time signal generation and execution
- Risk management through `risk/barbell_policy.py`

**LLM Integration (`ai_llm/`):**
- `ollama_client.py`: Local LLM server integration
- `signal_generator.py`: LLM-powered signal generation
- `market_analyzer.py`: Fundamental analysis via LLM

**Testing Infrastructure:**
- 141+ tests across ETL, LLM, integration, and security modules
- Property-based testing for financial calculations
- Security validation for credential handling

## Configuration Management

The system uses modular YAML configuration files in `config/`:
- `pipeline_config.yml`: Main orchestration settings
- `yfinance_config.yml`: Data extraction parameters
- `llm_config.yml`: LLM integration settings
- `quant_success_config.yml`: Trading success criteria
- `signal_routing_config.yml`: Signal routing logic

Configuration supports:
- Environment variable overrides
- Hyperparameter optimization (`.hyperopt.yml` files)
- Per-environment settings

## Data Flow Architecture

```
Data Sources → Extraction → Validation → Preprocessing → Forecasting → Signal Router → Execution
     ↓              ↓            ↓             ↓            ↓            ↓           ↓
   Cache       Checkpoint   Quality      Feature      Model       Signal      Paper
   Layer        Manager      Checks      Builder     Ensemble     Router     Trading
```

**Key Data Paths:**
- Raw data: `data/raw/` (cached extracts)
- Processed: `data/training/`, `data/validation/`, `data/testing/`
- Checkpoints: `data/checkpoints/` (pipeline state)
- Visualizations: `visualizations/` (analysis plots)

## Development Patterns

### Error Handling
- Graceful degradation: synthetic fallback when live data fails
- Comprehensive logging with structured JSON for monitoring
- Circuit breaker patterns for external API calls

### Performance Optimization
- Intelligent caching with 24h validity (20x speedup)
- Vectorized operations throughout (NumPy/Pandas)
- Parquet format for 10x faster I/O vs CSV
- Connection pooling for LLM and data sources

### Testing Strategy
- Unit tests for core calculations and data processing
- Integration tests for pipeline workflows
- Security tests for credential handling
- Performance benchmarks for critical paths

### Code Organization
- Clear separation between extraction, processing, and execution
- Configuration-driven behavior to avoid hardcoded parameters
- Consistent logging and error handling patterns
- Type hints and comprehensive docstrings

## Troubleshooting

**Common Issues:**
- Virtual environment not activated → All operations require `source simpleTrader_env/bin/activate`
- Cache corruption → Clear with `rm data/raw/*.parquet`
- Test failures → Check Python version (3.10+ required)
- LLM integration issues → Verify Ollama server: `curl http://localhost:11434/api/tags`

**Environment Validation:**
```bash
python scripts/validate_environment.py  # Checks all dependencies and paths
```

**Debug Pipeline:**
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python scripts/run_etl_pipeline.py --execution-mode synthetic  # Safe test mode
```