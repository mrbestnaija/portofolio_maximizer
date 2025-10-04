# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a portfolio management system focused on quantitative trading strategies with academic rigor. The project follows a phased development approach with strict mathematical foundations, vectorized implementations, and **configuration-driven, platform-agnostic architecture**.

**Current Status** (v4.0 - 2025-10-04):
- **Phase 4.5 Complete**: k-fold time series cross-validation with 5.5x coverage improvement
- **Test Coverage**: 85/85 tests passing (100%)
- **Configuration**: Modular, extensible, platform-agnostic (~40 KB YAML)
- **Data Sources**: yfinance (active), Alpha Vantage & Finnhub (configured, ready to implement)

## Architecture

The project is structured as a **configuration-driven, platform-agnostic** ETL portfolio management system:

### Core Modules (`etl/`)
- **Data Extraction**: Platform-agnostic adapters for multiple sources
  - `yfinance_extractor.py`: Yahoo Finance (active, 24h cache, 100% hit rate)
  - `ucl_extractor.py`: UCL database integration (future)
  - `alpha_vantage_extractor.py`: Alpha Vantage (future, configured)
  - `finnhub_extractor.py`: Finnhub (future, configured)

- **Data Processing**: Vectorized transformations
  - `data_validator.py`: Quality validation (prices, volumes, outliers)
  - `preprocessor.py`: Missing data + normalization (μ=0, σ²=1)
  - `data_storage.py`: Parquet storage + train/val/test splitting
  - `time_series_cv.py`: k-fold cross-validation (5.5x coverage improvement)

- **Analysis & Visualization**:
  - `time_series_analyzer.py`: ADF, ACF/PACF, stationarity tests
  - `visualizer.py`: 7 publication-quality plot types

### Configuration (`config/`)
**Modular, platform-agnostic design** - 8 comprehensive YAML files (~40 KB):

1. **pipeline_config.yml** (6.5 KB): Unified orchestration
   - 4-stage pipeline (extraction → validation → preprocessing → storage)
   - Data split strategies (simple 70/15/15 or k-fold CV)
   - Error handling, checkpoints, logging

2. **data_sources_config.yml**: Platform-agnostic data source registry
   - Provider abstraction layer (yfinance, Alpha Vantage, Finnhub)
   - Failover configuration, health monitoring
   - Extensibility guidelines for adding new sources

3. **Source-Specific Configs**:
   - `yfinance_config.yml` (2.6 KB): Cache, retry, rate limiting
   - `alpha_vantage_config.yml`: 5 req/min free tier, column mapping
   - `finnhub_config.yml`: 60 req/min free tier, Unix timestamp handling

4. **Processing Configs**:
   - `validation_config.yml` (7.7 KB): Price/volume checks, outlier detection
   - `preprocessing_config.yml` (4.8 KB): Missing data strategies, normalization
   - `storage_config.yml` (5.9 KB): Parquet settings, CV/simple split
   - `analysis_config.yml`: MIT statistical standards, ADF parameters

### Orchestration (`scripts/`)
- **run_etl_pipeline.py** (256 lines): **Config-driven, platform-agnostic orchestrator**
  - Data source selection: `--data-source yfinance|alpha_vantage|finnhub`
  - Split strategy: `--use-cv` for k-fold CV (default: simple split)
  - Verbose logging: `--verbose` for DEBUG mode
  - Automatic failover and health checks

### Data Storage (`data/`)
- Organized by ETL stage: raw → processed → training/validation/testing
- Parquet format (10x faster than CSV, snappy compression)
- Cache-first strategy (24h validity, 100% hit rate)

### Testing (`tests/`)
- 85 comprehensive tests (100% passing)
- Unit, integration, and performance tests
- CV-specific tests: temporal ordering, coverage improvement, backward compatibility

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source simpleTrader_env/bin/activate  # Linux/Mac
# or
simpleTrader_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/etl/test_yfinance_extractor.py

# Run with coverage
pytest tests/ --cov=etl --cov-report=html
```

### ETL Pipeline Operations
```bash
# Run full ETL pipeline
python scripts/run_etl_pipeline.py

# Monitor data quality
python scripts/data_quality_monitor.py
```

### Data Validation
```bash
# Validate workflow configurations
python -c "import yaml; yaml.safe_load(open('workflows/etl_pipeline.yml'))"
```

## Key Dependencies

- **Core Scientific Computing**: numpy, pandas, scipy, statsmodels
- **Data Sources**: yfinance, pandas-datareader
- **Visualization**: matplotlib
- **ML/Analysis**: scikit-learn
- **Testing**: pytest
- **Configuration**: pyyaml, python-dotenv
- **CLI**: click
- **Progress**: tqdm

## Development Philosophy

The project follows MIT academic standards with specific constraints:
- Maximum 50 lines per code implementation
- Vectorized operations only (no explicit loops)
- Mathematical justification required for all approaches
- Quantitative validation before task completion
- Explicit approval gates for progression

## Agent Instructions Integration

The project includes comprehensive agent instructions (AGENT-INSTRUCTIONS.md) that define:
- Task execution patterns with mathematical foundations
- Quality standards and validation criteria
- Phase-by-phase implementation roadmap
- Success metrics and decision trees

When implementing features, follow the established patterns of vectorized operations and quantitative validation defined in the agent instructions.

## Configuration Management

- Environment variables stored in `.env` file
- YAML configuration files in `config/` directory
- Virtual environment: `simpleTrader_env/`

## Testing Strategy

- Unit tests for each ETL component
- Integration tests for end-to-end workflows
- Data validation tests for quality assurance
- All files currently exist as stubs (empty files) - implementation needed