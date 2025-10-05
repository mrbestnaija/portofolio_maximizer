# Portfolio Maximizer v45

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](https://github.com/mrbestnaija/portofolio_maximizer)

> A production-ready quantitative portfolio management system with intelligent caching, comprehensive time series analysis, and publication-quality visualizations.

**Version**: 3.0
**Status**: Production Ready ✅
**Last Updated**: 2025-10-04

---

## 🎯 Overview

Portfolio Maximizer v45 is a sophisticated ETL-based portfolio management system built with academic rigor and production-grade performance. It provides a complete pipeline for extracting, validating, preprocessing, and analyzing financial time series data with intelligent caching and vectorized operations.

### Key Features

- **🚀 Intelligent Caching**: 20x speedup with cache-first strategy (24h validity)
- **📊 Advanced Analysis**: MIT-standard time series analysis (ADF, ACF/PACF, stationarity)
- **📈 Publication-Quality Visualizations**: 8 professional plots with 150 DPI quality
- **🔄 Robust ETL Pipeline**: 4-stage pipeline with comprehensive validation
- **✅ Comprehensive Testing**: 63 tests with 98.4% pass rate
- **⚡ High Performance**: Vectorized operations, Parquet format (10x faster than CSV)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🏗️ Architecture

### System Architecture (7 Layers)

```
┌─────────────────────────────────────────────────────────┐
│              Portfolio Maximizer v45                     │
│              Production-Ready System                     │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
  Layer 1:        Layer 2:         Layer 3:
  Extraction      Storage          Validation
  (yfinance       (Parquet         (Quality
   UCL)           Format)          Checks)
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  Layer 4:        Layer 5:         Layer 6:
  Preprocessing   Organization     Analysis &
  (Transform)     (Train/Val/Test) Visualization
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
                   Layer 7:
                   Output
                   (Reports, Plots)
```

### Core Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| **yfinance_extractor.py** | 327 | Yahoo Finance data extraction with intelligent caching |
| **data_validator.py** | 117 | Statistical validation and outlier detection |
| **preprocessor.py** | 101 | Missing data handling and normalization |
| **data_storage.py** | 158 | Parquet-based storage with train/val/test split |
| **portfolio_math.py** | 45 | Financial calculations (returns, volatility, Sharpe) |
| **time_series_analyzer.py** | 500+ | MIT-standard time series analysis |
| **visualizer.py** | 600+ | Publication-quality visualization engine |

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/mrbestnaija/portofolio_maximizer.git
cd portfolio_maximizer_v45

# Create virtual environment
python -m venv simpleTrader_env

# Activate virtual environment
# On Linux/Mac:
source simpleTrader_env/bin/activate
# On Windows:
simpleTrader_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# API Keys (if needed)
ALPHA_VANTAGE_API_KEY=your_key_here

# Database Configuration (if using UCL)
UCL_DB_HOST=localhost
UCL_DB_PORT=5432
UCL_DB_NAME=portfolio_db

# Cache Settings
CACHE_VALIDITY_HOURS=24
```

---

## ⚡ Quick Start

### Run the ETL Pipeline

```bash
# Activate virtual environment
source simpleTrader_env/bin/activate

# Run the complete ETL pipeline
python scripts/run_etl_pipeline.py

# Expected output:
# ✓ Extraction complete (cache hit: <0.1s)
# ✓ Validation complete (0.1s)
# ✓ Preprocessing complete (0.2s)
# ✓ Storage complete (0.1s)
# Total time: <1s (with cache)
```

### Analyze Dataset

```bash
# Run time series analysis on training data
python scripts/analyze_dataset.py \
    --dataset data/training/training_20251001_210734_20251001.parquet \
    --column Close \
    --output analysis_results.json

# Output: ADF test, ACF/PACF, statistical summary
```

### Generate Visualizations

```bash
# Create publication-quality plots
python scripts/visualize_dataset.py \
    --dataset data/training/training_20251001_210734_20251001.parquet \
    --column Close \
    --output-dir visualizations/

# Generates 8 plots:
# - Time series overview
# - Distribution analysis
# - ACF/PACF plots
# - Decomposition (trend/seasonal/residual)
# - Rolling statistics
# - Spectral density
# - Comprehensive dashboard
```

---

## 📖 Usage

### 1. Data Extraction

```python
from etl.yfinance_extractor import YFinanceExtractor

# Initialize extractor
extractor = YFinanceExtractor()

# Extract data with intelligent caching
df = extractor.extract_data(
    ticker='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Cache hit: <0.1s, Cache miss: ~20s
```

### 2. Data Validation

```python
from etl.data_validator import DataValidator

# Initialize validator
validator = DataValidator()

# Validate data quality
validation_results = validator.validate_dataframe(df)

# Check for:
# - Price positivity
# - Volume non-negativity
# - Outliers (3σ threshold)
# - Missing data percentage
```

### 3. Data Preprocessing

```python
from etl.preprocessor import Preprocessor

# Initialize preprocessor
preprocessor = Preprocessor()

# Handle missing data
df_filled = preprocessor.handle_missing_data(df, method='forward')

# Normalize data (Z-score)
df_normalized = preprocessor.normalize_data(df_filled, method='zscore')
```

### 4. Time Series Analysis

```python
from etl.time_series_analyzer import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer()

# Run comprehensive analysis
results = analyzer.analyze(
    data=df['Close'],
    column_name='Close'
)

# Results include:
# - ADF test (stationarity)
# - ACF/PACF (autocorrelation)
# - Statistical summary (μ, σ², skewness, kurtosis)
```

### 5. Visualization

```python
from etl.visualizer import Visualizer

# Initialize visualizer
viz = Visualizer()

# Create comprehensive dashboard
viz.plot_comprehensive_dashboard(
    data=df,
    column='Close',
    save_path='visualizations/dashboard.png'
)

# Creates 8-panel publication-quality plot
```

---

## 📁 Project Structure

```
portfolio_maximizer_v45/
│
├── config/                          # Configuration files (YAML)
│   ├── analysis_config.yml          # Time series analysis parameters
│   ├── preprocessing_config.yml     # Preprocessing settings
│   ├── ucl_config.yml              # UCL database config
│   └── yfinance_config.yml         # Yahoo Finance settings
│
├── data/                            # Data storage (organized by ETL stage)
│   ├── raw/                         # Original extracted data + cache
│   ├── processed/                   # Cleaned and transformed data
│   ├── training/                    # Training set (70%)
│   ├── validation/                  # Validation set (15%)
│   └── testing/                     # Test set (15%)
│
├── Documentation/                   # Comprehensive documentation
│   ├── arch_tree.md                # Architecture documentation
│   ├── CACHING_IMPLEMENTATION.md   # Caching mechanism guide
│   ├── GIT_WORKFLOW.md             # Git workflow (local-first)
│   ├── implementation_checkpoint.md # Implementation status
│   └── CLAUDE.md                   # Development guide
│
├── etl/                             # ETL pipeline modules
│   ├── yfinance_extractor.py       # Yahoo Finance extraction
│   ├── ucl_extractor.py            # UCL database extraction
│   ├── data_validator.py           # Data quality validation
│   ├── preprocessor.py             # Data preprocessing
│   ├── data_storage.py             # Data persistence
│   ├── portfolio_math.py           # Financial calculations
│   ├── time_series_analyzer.py     # Time series analysis
│   └── visualizer.py               # Visualization engine
│
├── scripts/                         # Executable scripts
│   ├── run_etl_pipeline.py         # Main ETL orchestration
│   ├── analyze_dataset.py          # Analysis CLI
│   ├── visualize_dataset.py        # Visualization CLI
│   ├── data_quality_monitor.py     # Quality monitoring
│   └── validate_environment.py     # Environment validation
│
├── tests/                           # Test suite (63 tests)
│   ├── etl/                        # ETL module tests
│   │   ├── test_yfinance_extractor.py
│   │   ├── test_yfinance_cache.py
│   │   ├── test_data_validator.py
│   │   ├── test_preprocessor.py
│   │   ├── test_data_storage.py
│   │   ├── test_portfolio_math.py
│   │   └── test_time_series_analyzer.py
│   └── integration/                # Integration tests
│
├── visualizations/                  # Generated visualizations
│   └── training/                    # Training data plots
│
├── workflows/                       # Pipeline orchestration (YAML)
│   ├── etl_pipeline.yml            # Main ETL workflow
│   └── data_validation.yml         # Validation workflow
│
├── .gitignore                       # Git ignore rules
├── pytest.ini                       # Pytest configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚡ Performance

### Benchmarks (AAPL Dataset: 1,006 rows)

| Operation | Time | Notes |
|-----------|------|-------|
| **Cache Hit** | <0.1s | Instant retrieval from local parquet |
| **Cache Miss** | ~20s | Network fetch + validation + save |
| **Validation** | <0.1s | Vectorized quality checks |
| **Preprocessing** | <0.2s | Missing data + normalization |
| **Train/Val/Test Split** | <0.1s | Chronological slicing |
| **Full Analysis** | 1.2s | ADF + ACF + statistics (704 rows) |
| **Single Visualization** | 0.3s | One plot at 150 DPI |
| **All Visualizations** | 2.5s | 8 plots at publication quality |
| **Full ETL Pipeline (cached)** | <1s | All 4 stages with 100% cache hit |
| **Full ETL Pipeline (no cache)** | ~25s | First run with network fetch |

### Cache Performance

- **Hit Rate**: 100% (after first run)
- **Speedup**: 20x compared to network fetch
- **Storage**: 54 KB per ticker (Parquet compressed)
- **Validity**: 24 hours (configurable)
- **Network Savings**: 100% on cache hit

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=etl --cov-report=html

# Run specific test module
pytest tests/etl/test_yfinance_cache.py -v

# Run with verbose output
pytest tests/ -v --tb=short
```

### Test Coverage Summary

| Test Suite | Tests | Passing | Coverage |
|------------|-------|---------|----------|
| **Cache Tests** | 10 | 10 (100%) | Comprehensive |
| **ETL Tests** | 27 | 26 (96.3%) | 1 network timeout |
| **Analysis Tests** | 17 | 17 (100%) | Full coverage |
| **Math Tests** | 5 | 5 (100%) | All scenarios |
| **Storage Tests** | 6 | 6 (100%) | I/O operations |
| **Preprocessing Tests** | 8 | 8 (100%) | Edge cases |
| **Total** | 63 | 62 (98.4%) | Production ready |

---

## 📚 Documentation

Comprehensive documentation is available in the `Documentation/` directory:

- **[Architecture Tree](Documentation/arch_tree.md)**: Complete architecture overview
- **[Caching Implementation](Documentation/CACHING_IMPLEMENTATION.md)**: Intelligent caching guide
- **[Git Workflow](Documentation/GIT_WORKFLOW.md)**: Local-first git workflow
- **[Implementation Checkpoint](Documentation/implementation_checkpoint.md)**: Development status
- **[Development Guide](Documentation/CLAUDE.md)**: Development standards and practices

---

## 🛣️ Roadmap

### Phase 5: Portfolio Optimization (Next)
- Mean-variance optimization (Markowitz)
- Risk parity portfolio
- Black-Litterman model
- Constraint handling (long-only, sector limits)

### Phase 6: Risk Modeling
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Expected Shortfall (CVaR)
- Maximum Drawdown analysis
- Stress testing framework

### Phase 7: Backtesting Engine
- Vectorized backtest engine
- Transaction cost modeling
- Slippage simulation
- Performance attribution

### Caching Enhancements
- Smart cache invalidation (market close triggers)
- Distributed caching (Redis/Memcached)
- Cache analytics (hit rate tracking)
- Incremental updates (partial refresh)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Follow code standards**:
   - Vectorized operations only
   - Type hints required
   - Comprehensive docstrings
   - MIT statistical standards
4. **Write tests** (maintain >95% coverage)
5. **Commit changes** (`git commit -m 'feat: Add amazing feature'`)
6. **Push to branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/portofolio_maximizer.git

# Add upstream remote
git remote add upstream https://github.com/mrbestnaija/portofolio_maximizer.git

# Create feature branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests before committing
pytest tests/ --cov=etl
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Bestman Ezekwu Enock**

- GitHub: [@mrbestnaija](https://github.com/mrbestnaija)
- Email: mrbestnaija@example.com

---

## 🙏 Acknowledgments

- **MIT OpenCourseWare**: Statistical learning standards
- **Tufte Principles**: Visualization design guidelines
- **Yahoo Finance**: Market data API
- **Claude Code**: AI-assisted development

---

## 📊 Project Statistics

- **Total Lines of Code**: 3,567 (production)
- **Test Lines**: 1,068
- **Documentation**: 8 comprehensive files
- **Test Coverage**: 98.4%
- **Performance**: 20x speedup with caching
- **Data Quality**: 0% missing data (after preprocessing)

---

## 🔧 Troubleshooting

### Common Issues

**1. Cache not working**
```bash
# Check cache directory permissions
ls -la data/raw/

# Clear cache if corrupted
rm data/raw/*.parquet

# Verify cache configuration
cat config/yfinance_config.yml
```

**2. Import errors**
```bash
# Ensure virtual environment is activated
source simpleTrader_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**3. Test failures**
```bash
# Run specific failing test with verbose output
pytest tests/etl/test_yfinance_extractor.py::test_name -v --tb=short

# Check Python version
python --version  # Should be 3.12+
```

---

## 📞 Support

For questions or issues:

1. **Check Documentation**: `Documentation/` directory
2. **Search Issues**: [GitHub Issues](https://github.com/mrbestnaija/portofolio_maximizer/issues)
3. **Open New Issue**: Provide reproducible example
4. **Email**: csgtmalice@protonmail.ch
5. **Phone**: +2348061573767(Whatsapp Only)
6. **Discord**: https://discord.gg/FVzV66Hb

---

**Built with ❤️ for Linda-Best**

**Status**: Production Ready ✅
**Version**: 3.0
**Last Updated**: 2025-10-04

