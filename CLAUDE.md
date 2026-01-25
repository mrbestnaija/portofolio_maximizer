# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Portfolio Maximizer is an autonomous quantitative trading system that extracts financial data, forecasts market regimes, routes trading signals, and executes trades automatically. It's a production-ready Python system with institutional-grade ETL pipelines, LLM integration, and comprehensive testing.

**Current Phase**: Phase 7.5 Complete (Regime detection integration with multi-ticker validation)
**Last Updated**: 2026-01-25

---

## Development Environment & Platform Considerations

### Python Environment
```bash
# REQUIRED: Always activate virtual environment first
source simpleTrader_env/bin/activate  # Linux/Mac
simpleTrader_env\Scripts\activate     # Windows

# Supported Python: >=3.10,<3.13
# Current packages: See requirements.txt (last updated 2026-01-23)
```

### Platform-Specific Notes

**Windows (Primary Development Platform)**:
- Use forward slashes or proper escaping in bash commands
- Unicode characters (✓, ✗) cause `UnicodeEncodeError` on Windows console
- Always use ASCII alternatives: `[OK]`, `[ERROR]`, `[SUCCESS]`
- Git bash on Windows: Use `/c/Users/...` instead of `C:\Users\...` for paths
- Background tasks: Use `./simpleTrader_env/Scripts/python.exe` not `python`

**Cross-Platform Best Practices**:
- Use `Path()` from `pathlib` for all file paths
- Test unicode output on Windows before deploying
- Provide ASCII fallbacks for all console output
- Document platform-specific requirements in migration scripts

---

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment (required for all operations)
source simpleTrader_env/bin/activate  # Linux/Mac
simpleTrader_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install GPU extras (optional, CUDA 12.x required)
pip install -r requirements-ml.txt
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

# Analyze multi-ticker results (Phase 7.4+)
python scripts/analyze_multi_ticker_results.py

# Generate visualizations
python scripts/visualize_dataset.py --dataset data/training/training_*.parquet --output-dir visualizations/

# Validate environment and dependencies
python scripts/validate_environment.py
```

### Database Management (Phase 7.4+)
```bash
# Migrate database to support ENSEMBLE model type
python scripts/migrate_add_ensemble_model_type.py

# Verify migration
sqlite3 data/portfolio_maximizer.db "SELECT DISTINCT model_type FROM time_series_forecasts;"
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

---

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
   - **SARIMAX**: Seasonal ARIMA with exogenous variables
   - **GARCH**: Volatility forecasting (Phase 7.3+ integration)
   - **SAMoSSA**: Singular Spectrum Analysis + RL
   - **MSSA-RL**: Multivariate SSA with reinforcement learning
   - **Ensemble**: Adaptive routing through `SignalRouter` (Phase 7.4+ quantile calibration)

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

---

## Configuration Management

The system uses modular YAML configuration files in `config/`:
- `pipeline_config.yml`: Main orchestration settings
- `forecasting_config.yml`: Model parameters and ensemble config
- `yfinance_config.yml`: Data extraction parameters
- `llm_config.yml`: LLM integration settings
- `quant_success_config.yml`: Trading success criteria
- `signal_routing_config.yml`: Signal routing logic

Configuration supports:
- Environment variable overrides
- Hyperparameter optimization (`.hyperopt.yml` files)
- Per-environment settings

**Important Notes**:
- Ensemble candidate weights defined in `forecasting_config.yml` lines 69-83
- Phase 7.4: Quantile-based confidence calibration enabled by default
- Regime detection available but not yet integrated (Phase 7.5)

---

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
- Database: `data/portfolio_maximizer.db` (SQLite with ENSEMBLE support)

---

## Development Patterns & Best Practices

### Error Handling
- Graceful degradation: synthetic fallback when live data fails
- Comprehensive logging with structured JSON for monitoring
- Circuit breaker patterns for external API calls
- **Platform consideration**: Windows console requires ASCII-only output

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
- **Phase 7.4**: Multi-ticker validation (AAPL, MSFT, NVDA)

### Code Organization
- Clear separation between extraction, processing, and execution
- Configuration-driven behavior to avoid hardcoded parameters
- Consistent logging and error handling patterns
- Type hints and comprehensive docstrings

### Database Best Practices (Phase 7.4+)
- **Always check model_type constraint** when adding new model types
- Run migration scripts before deploying forecast changes
- Validate migrations on test database first
- Document schema changes in migration scripts
- **Current constraint**: `model_type IN ('SARIMAX', 'GARCH', 'COMBINED', 'ENSEMBLE', 'SAMOSSA', 'MSSA_RL')`

---

## Phase 7.5 Specific Guidance (Regime Detection Integration)

### Overview

Phase 7.5 integrated RegimeDetector into TimeSeriesForecaster for adaptive model selection based on market conditions. The system now dynamically reorders ensemble candidates based on detected regime characteristics (volatility, trend strength, Hurst exponent).

**Status**: ✅ COMPLETE (validated across 3 tickers with feature flag enabled)

### Integration Fixes Applied

**Issue 1: Signal Generator Missing Config**
- **File**: `models/time_series_signal_generator.py` (lines 1487-1515)
- **Problem**: Wasn't extracting regime_detection params from forecasting_config.yml
- **Fix**: Added regime_cfg extraction and passing to TimeSeriesForecasterConfig in both CV paths

**Issue 2: Pipeline Script Missing Config**
- **Files**: `scripts/run_etl_pipeline.py` (lines 1858, 1865-1870, 1893-1897) + `config/pipeline_config.yml` (lines 323-360)
- **Problem**: Pipeline loaded from pipeline_config.yml but regime_detection only in forecasting_config.yml
- **Fix**: Added 37-line regime_detection section to pipeline_config.yml and loading logic

**Issue 3: RegimeConfig Parameter Mismatch**
- **File**: `forcester_ts/forecaster.py` (lines 118-132)
- **Problem**: regime_model_preferences passed but not in RegimeConfig dataclass signature
- **Fix**: Filter kwargs to only include valid fields: {enabled, lookback_window, vol_threshold_low, vol_threshold_high, trend_threshold_weak, trend_threshold_strong}

### Validation Results

**Single-Ticker (AAPL, 2024-07-01 to 2026-01-18)**:
- Regimes detected: MODERATE_TRENDING (1), HIGH_VOL_TRENDING (2), CRISIS (2)
- Average confidence: 68.3%
- Adaptation rate: 40% (2/5 builds switched to SAMOSSA-led)
- RMSE impact: +42% regression (1.043 → 1.483, expected for research phase)

**Multi-Ticker (AAPL, MSFT, NVDA)**:
- Total forecasts: 15 (5 per ticker)
- Distinct regimes: 4 types
- Adaptation rate: 53% (8/15 builds)
- Average confidence: 65.2%
- **Key finding**: ✅ Regime detection generalizes across tickers

**Regime Distribution**:
- **AAPL**: Mixed (20% MODERATE, 40% HIGH_VOL, 40% CRISIS)
- **MSFT**: 80% HIGH_VOL_TRENDING (sustained volatile trending)
- **NVDA**: Extreme volatility (avg 57.8%, peaks at 73%)

**Volatility Ranking** (Correct): NVDA (58%) > AAPL (42%) > MSFT (27%)

### Configuration

**Feature Flag** (config/pipeline_config.yml + config/forecasting_config.yml):
```yaml
regime_detection:
  enabled: true  # Currently enabled for validation/audit accumulation
  lookback_window: 60
  vol_threshold_low: 0.15
  vol_threshold_high: 0.30
  trend_threshold_weak: 0.30
  trend_threshold_strong: 0.60
```

**Regime Model Preferences**:
- HIGH_VOL_TRENDING → {samossa, mssa_rl, garch}
- CRISIS → {garch, sarimax} (defensive)
- MODERATE_TRENDING → {samossa, garch, sarimax}
- LIQUID_RANGEBOUND → {garch, sarimax, samossa}

### Known Limitations

1. **Multi-Ticker Pipeline**: Running `--tickers AAPL,MSFT,NVDA` concatenates data without ticker column. **Workaround**: Run separate pipelines per ticker.
2. **RMSE Regression**: +42% vs Phase 7.4 baseline (trades accuracy for robustness/diversity).
3. **Extreme Volatility (NVDA)**: 73% annualized detected - investigate data quality.

### Documentation

- [PHASE_7.5_VALIDATION.md](Documentation/PHASE_7.5_VALIDATION.md): Single-ticker validation (340 lines)
- [PHASE_7.5_MULTI_TICKER_RESULTS.md](Documentation/PHASE_7.5_MULTI_TICKER_RESULTS.md): Multi-ticker analysis (340 lines)

### Git Commits

- **1b696f5** (2026-01-24): Integration with 3 fixes
- **de443c9** (2026-01-25): Multi-ticker validation results

---

## Phase 7.4 Reference (GARCH Ensemble Integration - COMPLETE)

**Performance Metrics**:
- AAPL RMSE Ratio: 1.470 → 1.043 (29% improvement)
- GARCH Selection: 14% → 100%
- Target Achievement: 94.6%

**Key Features**:
- Quantile-based confidence calibration (prevents SAMoSSA dominance)
- Ensemble config preservation during CV
- Database schema migration (added ENSEMBLE model type)

---

## Agent Workflow Best Practices

### When Starting Work
1. **Read CLAUDE.md** (this file) for project context
2. **Check git status** to see current state
3. **Review recent commits** to understand recent changes
4. **Check Documentation/** for phase-specific context
5. **Activate virtual environment** before any Python operations

### When Modifying Code
1. **Read existing code** before suggesting changes (never propose changes to unread code)
2. **Preserve existing patterns** (logging, error handling, configuration style)
3. **Update requirements.txt** when adding packages
4. **Run migration scripts** when changing database schema
5. **Test on target platform** (Windows primary, Linux secondary)
6. **Document breaking changes** in relevant phase documentation

### When Creating New Features
1. **Check configuration files** for similar patterns
2. **Follow existing architecture** (7-layer model)
3. **Add comprehensive docstrings** with type hints
4. **Include error handling** with platform-aware output
5. **Update CLAUDE.md** with new patterns/practices
6. **Create tests** for new functionality

### When Debugging Issues
1. **Check logs first** (`logs/*.log`, sorted by timestamp)
2. **Look for platform-specific issues** (unicode, paths, etc.)
3. **Verify database schema** if forecast saves fail
4. **Check configuration loading** (ensemble_kwargs, etc.)
5. **Use grep/analyze scripts** before manual log parsing
6. **Document findings** in Documentation/ with timestamp

### Platform-Specific Development

**Windows Considerations**:
- Bash commands: Use `/c/Users/...` paths, not `C:\...`
- Unicode: Always use ASCII for console output ([OK] not ✓)
- Paths: Use `Path()` from pathlib, not string concatenation
- Background jobs: Full path to python.exe in venv
- Git: Line endings set to LF (core.autocrlf=false)

**Cross-Platform Testing**:
- Test migration scripts on Windows first (unicode issues)
- Verify file paths work on both Windows/Linux
- Check that all console output is ASCII-safe
- Test background job syntax on target platform

### Requirements Management

**When to Update requirements.txt**:
- After installing new packages (`pip install <package>`)
- When package versions change in environment
- After major Python version upgrade
- When deploying to new environment

**How to Update**:
```bash
# Freeze current environment
pip freeze > requirements_new.txt

# Manually update requirements.txt header and merge
# Header format:
# Supported Python runtime: >=3.10,<3.13
# Last updated: YYYY-MM-DD (Phase X.Y description)
```

**requirements-ml.txt**:
- Only update for GPU/CUDA package changes
- Test on GPU-enabled system before committing
- Document CUDA version compatibility

---

## Troubleshooting

### Common Issues

**Issue**: Virtual environment not activated
**Fix**: All operations require `source simpleTrader_env/bin/activate`

**Issue**: Cache corruption
**Fix**: Clear with `rm data/raw/*.parquet`

**Issue**: Test failures
**Fix**: Check Python version (3.10+ required)

**Issue**: LLM integration issues
**Fix**: Verify Ollama server: `curl http://localhost:11434/api/tags`

**Issue**: Database constraint errors (Phase 7.4+)
**Fix**: Run `python scripts/migrate_add_ensemble_model_type.py`

**Issue**: Unicode output errors on Windows
**Fix**: Replace unicode characters with ASCII equivalents

**Issue**: Ensemble config not preserved during CV
**Fix**: Verify forecasting_config.yml is loaded in TimeSeriesSignalGenerator

### Environment Validation
```bash
python scripts/validate_environment.py  # Checks all dependencies and paths
```

### Debug Pipeline
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python scripts/run_etl_pipeline.py --execution-mode synthetic  # Safe test mode
```

---

## Git & GitHub Integration

**Repository**: https://github.com/mrbestnaija/portofolio_maximizer.git
**Default Branch**: master
**Main Branch**: master (for PRs)

### Commit Message Format
```
Phase X.Y: Brief description (50 chars)

- Bullet point details of changes
- Reference issue numbers if applicable
- Note breaking changes
- Document migration requirements

Results: Key metrics or validation results
```

### Pre-Commit Checklist

- [ ] All tests passing (`pytest tests/`)
- [ ] Requirements updated if packages changed
- [ ] Database migrations run and tested
- [ ] Platform-specific code tested on Windows
- [ ] Documentation updated in relevant phase docs
- [ ] CLAUDE.md updated with new patterns (if applicable)
- [ ] No unicode characters in console output
- [ ] Git status clean or changes documented

---

## Quick Reference

### Essential Files
- `CLAUDE.md` - This file (agent guidance)
- `README.md` - User-facing project overview
- `requirements.txt` - Python dependencies (updated 2026-01-23)
- `config/pipeline_config.yml` - Main configuration
- `Documentation/RUNTIME_GUARDRAILS.md` - Python version constraints

### Key Directories
- `etl/` - Data extraction, transformation, loading
- `forcester_ts/` - Time series forecasting models
- `models/` - Signal generation and routing
- `execution/` - Order management and paper trading
- `tests/` - Test suite (141+ tests)
- `scripts/` - Utility scripts and migrations
- `config/` - YAML configuration files
- `Documentation/` - Phase-specific documentation
- `logs/` - Pipeline and application logs

---

**Remember**: Always activate virtual environment, check platform compatibility, and update documentation when making changes!

**Last Updated**: 2026-01-23 (Phase 7.4 completion)
**GitHub**: https://github.com/mrbestnaija/portofolio_maximizer.git
