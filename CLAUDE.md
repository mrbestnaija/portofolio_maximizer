# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a portfolio management system focused on quantitative trading strategies with academic rigor. The project follows a phased development approach with strict mathematical foundations and vectorized implementations.

## Architecture

The project is structured as an ETL-focused portfolio management system:

- **etl/**: Core data extraction, transformation, and loading modules
  - `yfinance_extractor.py`: Yahoo Finance data extraction
  - `ucl_extractor.py`: UCL database integration
  - `data_validator.py`: Data quality validation
  - `preprocessor.py`: Data preprocessing pipeline
  - `data_storage.py`: Data storage management

- **config/**: Configuration files for different data sources and processing
  - YAML-based configuration for yfinance, UCL, and preprocessing parameters

- **workflows/**: Pipeline orchestration
  - `etl_pipeline.yml`: Main ETL workflow
  - `data_validation.yml`: Data quality validation workflow

- **scripts/**: Executable scripts for pipeline operations
  - `run_etl_pipeline.py`: Main pipeline execution
  - `data_quality_monitor.py`: Monitoring and quality checks

- **tests/**: Comprehensive test suite with ETL and integration tests

- **data/**: Data storage organized by processing stage (raw, processed, training, validation, testing)

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