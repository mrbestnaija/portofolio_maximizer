# Portfolio Maximizer v45 - Architecture Tree

**Version**: 4.0
**Date**: 2025-10-04
**Status**: Production Ready ✅ (with k-fold CV support)

---

## Project Structure

```
portfolio_maximizer_v45/
│
├── .claude/                         # Claude Code configuration
│   └── CLAUDE.md                    # Project-specific AI instructions
│
├── config/                          # Configuration files (YAML) - Modular Architecture ⭐
│   ├── pipeline_config.yml          # Main orchestration config (6.5 KB) ⭐ NEW
│   ├── data_sources_config.yml      # Platform-agnostic data sources ⭐ NEW
│   ├── yfinance_config.yml         # Yahoo Finance API settings (2.6 KB) ⭐ UPDATED
│   ├── alpha_vantage_config.yml     # Alpha Vantage config (future) ⭐ NEW
│   ├── finnhub_config.yml           # Finnhub config (future) ⭐ NEW
│   ├── preprocessing_config.yml     # Data preprocessing settings (4.8 KB) ⭐ NEW
│   ├── validation_config.yml        # Data validation rules (7.7 KB) ⭐ NEW
│   ├── storage_config.yml           # Storage and split config (5.9 KB) ⭐ NEW
│   ├── analysis_config.yml          # Time series analysis parameters (MIT standards)
│   └── ucl_config.yml              # UCL database configuration
│
├── data/                            # Data storage (organized by ETL stage)
│   ├── raw/                         # Original extracted data + cache
│   │   ├── AAPL_20251001.parquet   # Cached AAPL data (1,006 rows)
│   │   ├── MSFT_20251001.parquet   # Cached MSFT data (1,006 rows)
│   │   ├── extraction_*.parquet    # Historical extractions
│   │   ├── ucl/                    # UCL data directory
│   │   └── yfinance/               # Yahoo Finance data directory
│   ├── processed/                   # Cleaned and transformed data
│   │   └── processed_*.parquet     # Normalized, missing data filled
│   ├── training/                    # Training set (70% - 704 rows)
│   │   └── training_*.parquet      # Chronologically first 70%
│   ├── validation/                  # Validation set (15% - 151 rows)
│   │   └── validation_*.parquet    # Middle 15% for hyperparameter tuning
│   ├── testing/                     # Test set (15% - 151 rows)
│   │   └── testing_*.parquet       # Final 15% for model evaluation
│   └── analysis_report_training.json # Analysis results (JSON)
│
├── Documentation/                   # Project documentation
│   ├── implementation_checkpoint.md # Version 4.0 - Complete implementation status
│   ├── CACHING_IMPLEMENTATION.md   # Caching mechanism guide (7.9 KB)
│   ├── TIME_SERIES_CV.md           # Cross-validation guide (15 KB) ⭐ NEW
│   ├── GIT_WORKFLOW.md             # Git workflow (local-first) ⭐ NEW
│   ├── arch_tree.md                # This file - Architecture documentation
│   ├── CLAUDE.md                   # Development guide and standards
│   ├── AGENT_INSTRUCTION.md        # AI agent guidelines
│   ├── AGENT_DEV_CHECKLIST.md     # Development checklist
│   └── TO_DO_LIST.md              # Task tracking
│
├── etl/                             # ETL pipeline modules (2,184 lines)
│   ├── __init__.py                 # Package initialization
│   ├── yfinance_extractor.py       # Yahoo Finance extraction (327 lines)
│   │                                # Features: Cache-first, retry, rate limiting
│   │                                # Cache: 24h validity, ±3 day tolerance
│   │                                # Performance: 100% hit rate, 20x speedup
│   ├── ucl_extractor.py            # UCL database extraction
│   │                                # Features: Direct SQL queries, structured data
│   ├── data_validator.py           # Data quality validation (117 lines)
│   │                                # Features: Statistical validation, outlier detection
│   │                                # Checks: Price positivity, volume non-negativity
│   ├── preprocessor.py             # Data preprocessing (101 lines)
│   │                                # Features: Missing data handling, normalization
│   │                                # Methods: Forward/backward fill, Z-score
│   ├── data_storage.py             # Data persistence (210 lines) ⭐ UPDATED
│   │                                # Features: Parquet storage, CV splits, backward compatible
│   │                                # New: use_cv parameter for k-fold cross-validation
│   │                                # Format: Parquet (10x faster than CSV)
│   ├── time_series_cv.py           # Cross-validation (336 lines) ⭐ NEW
│   │                                # Features: k-fold CV, expanding window, test isolation
│   │                                # Improvement: 5.5x temporal coverage (15% → 83%)
│   │                                # Zero temporal gap (eliminates 2.5-year disparity)
│   ├── portfolio_math.py           # Financial calculations (45 lines)
│   │                                # Features: Returns, volatility, Sharpe ratio
│   │                                # Calculations: Max drawdown, correlation
│   ├── time_series_analyzer.py     # Time series analysis (500+ lines)
│   │                                # Features: ADF test, ACF/PACF, stationarity
│   │                                # Standards: MIT statistical learning conventions
│   └── visualizer.py               # Visualization engine (600+ lines)
│                                    # Features: 7 plot types, publication quality
│                                    # Quality: 150 DPI, Tufte principles
│
├── scripts/                         # Executable scripts (730 lines)
│   ├── run_etl_pipeline.py         # Main ETL orchestration (256 lines) ⭐ UPDATED
│   │                                # Platform-agnostic, config-driven orchestrator
│   │                                # Stages: Extraction → Validation → Preprocessing → Storage
│   │                                # Features: Cache-enabled, stage-by-stage execution
│   ├── analyze_dataset.py          # Dataset analysis CLI (270+ lines)
│   │                                # Features: Full analysis, JSON export, multiple columns
│   │                                # Outputs: ADF test, ACF/PACF, statistics
│   ├── visualize_dataset.py        # Visualization CLI (200+ lines)
│   │                                # Features: All plots, auto-save, custom styling
│   │                                # Outputs: 8 publication-quality plots
│   ├── data_quality_monitor.py     # Quality monitoring (44 lines)
│   │                                # Features: Automated checks, thresholds, alerts
│   └── validate_environment.py     # Environment validation
│                                    # Checks: Dependencies, Python version, packages
│
├── tests/                           # Test suite (1,558 lines, 85 tests)
│   ├── __init__.py
│   ├── etl/                        # ETL module tests
│   │   ├── __init__.py
│   │   ├── test_yfinance_extractor.py    # 3 tests (network extraction)
│   │   ├── test_yfinance_cache.py        # 10 tests (caching mechanism)
│   │   │                                  # Coverage: hits/misses, freshness, coverage
│   │   ├── test_time_series_cv.py        # 22 tests (CV mechanism) ⭐ NEW
│   │   │                                  # Coverage: k-fold, expanding window, validation
│   │   │                                  # Quantified: 5.5x improvement, 0 gaps
│   │   ├── test_ucl_extractor.py         # UCL extraction tests
│   │   ├── test_data_validator.py        # 5 tests (validation logic)
│   │   ├── test_preprocessor.py          # 8 tests (preprocessing operations)
│   │   ├── test_data_storage.py          # 7 tests (storage + CV operations) ⭐ UPDATED
│   │   ├── test_portfolio_math.py        # 5 tests (financial calculations)
│   │   └── test_time_series_analyzer.py  # 17 tests (analysis framework)
│   └── integration/                      # Integration tests
│                                          # End-to-end pipeline testing
│
├── visualizations/                  # Generated visualizations (1.6 MB, 8 plots)
│   ├── training/                    # Training data visualizations
│   │   ├── Close_acf_pacf.png      # Autocorrelation function plot (89 KB)
│   │   ├── Close_decomposition.png # Trend/Seasonal/Residual (415 KB)
│   │   ├── Close_distribution.png  # Histogram + KDE + QQ-plot (211 KB)
│   │   ├── Close_overview.png      # Time series overview (91 KB)
│   │   └── Close_rolling_stats.png # Rolling mean/std (247 KB)
│   ├── Close_dashboard.png         # 8-panel executive dashboard (329 KB)
│   ├── Close_spectral.png          # Spectral density - Welch's method (189 KB)
│   └── Volume_dashboard.png        # Volume analysis dashboard (367 KB)
│
├── workflows/                       # Pipeline orchestration (YAML)
│   ├── etl_pipeline.yml            # Main ETL workflow (4 stages)
│   │                                # Stages: extraction, validation, preprocessing, storage
│   └── data_validation.yml         # Validation workflow
│                                    # Quality checks, thresholds, reporting
│
├── simpleTrader_env/                # Python virtual environment
│   ├── bin/                        # Executables (python, pip, pytest)
│   ├── lib/                        # Installed packages
│   └── pyvenv.cfg                  # Virtual environment configuration
│
├── .gitignore                       # Git ignore rules
├── .env                            # Environment variables (secrets)
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview (if exists)
```

---

## System Architecture

### Layer-Based Architecture (7 Layers)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Portfolio Maximizer v45                      │
│                    Production-Ready System                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │    Layer 1: Data Extraction Layer           │
        │  (Cache-First Strategy - 100% Hit Rate)     │
        │                                             │
        │  • Yahoo Finance API (yfinance)             │
        │  • UCL Database (ucl_extractor)             │
        │  • Cache: 24h validity, auto-refresh        │
        │  • Retry: 3 attempts, exponential backoff   │
        │  • Rate limiting: configurable delays       │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │    Layer 2: Data Storage Layer              │
        │   (Parquet Format - Atomic Writes)          │
        │                                             │
        │  • Raw data + cache storage                 │
        │  • Parquet format (10x faster than CSV)     │
        │  • Atomic writes (temp + rename)            │
        │  • Auto-cleanup (retention policy)          │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │    Layer 3: Data Validation Layer           │
        │  (Statistical Quality Checks - MIT Std)     │
        │                                             │
        │  • Price validation (positivity, gaps)      │
        │  • Volume validation (non-negativity)       │
        │  • Outlier detection (Z-score, 3σ)          │
        │  • Missing data analysis (ρ_missing)        │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │    Layer 4: Data Preprocessing Layer        │
        │      (Vectorized Transformations)           │
        │                                             │
        │  • Missing data: forward/backward fill      │
        │  • Normalization: Z-score (μ=0, σ²=1)       │
        │  • Returns: log returns r_t = ln(P_t/P_t-1) │
        │  • Feature engineering (future)             │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │    Layer 5: Data Organization Layer         │
        │    (Train/Validation/Test Split + CV) ⭐     │
        │                                             │
        │  Simple Split (default, backward compat):   │
        │  • Training: 70% (704 rows)                 │
        │  • Validation: 15% (151 rows)               │
        │  • Testing: 15% (151 rows)                  │
        │                                             │
        │  k-fold CV (--use-cv flag):                 │
        │  • k folds with expanding window (k=5)      │
        │  • Validation coverage: 83% (5.5x better)   │
        │  • Test isolation: 15% (never in CV)        │
        │  • Temporal gap: 0 years (eliminated)       │
        │  • Chronological ordering (no leakage)      │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │    Layer 6: Analysis & Visualization        │
        │  (MIT Statistical Standards - Academic)     │
        │                                             │
        │  • Time Series Analysis:                    │
        │    - ADF test (stationarity)                │
        │    - ACF/PACF (autocorrelation)             │
        │    - Statistical summary (μ, σ², γ₁, γ₂)    │
        │                                             │
        │  • Visualization Engine:                    │
        │    - 7 plot types                           │
        │    - Publication quality (150 DPI)          │
        │    - Tufte principles                       │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │    Layer 7: Output Layer                    │
        │                                             │
        │  • JSON Reports (analysis results)          │
        │  • PNG Visualizations (8 plots, 1.6 MB)     │
        │  • Model-ready datasets (parquet)           │
        └─────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### End-to-End Pipeline Flow

```
External Data Sources
    │
    ├─► Yahoo Finance API ──────┐
    │   (Live market data)      │
    │                           │
    └─► UCL Database ───────────┤
        (Historical data)       │
                                │
                                ▼
                        ┌───────────────┐
                        │  Cache Check  │◄─── 24h validity
                        │  (Local)      │     ±3 day tolerance
                        └───────────────┘
                                │
                        ┌───────┴───────┐
                        │               │
                    Hit ▼               ▼ Miss
                ┌──────────┐    ┌──────────────┐
                │  Cache   │    │   Network    │
                │  Load    │    │   Fetch      │
                │ (<0.1s)  │    │  (~20s)      │
                └──────────┘    └──────────────┘
                        │               │
                        │    Auto-save  │
                        │    to cache   │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  Raw Storage  │
                        │  (Parquet)    │
                        │  data/raw/    │
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  Validation   │
                        │  (Quality)    │
                        │  - Prices     │
                        │  - Volumes    │
                        │  - Outliers   │
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Preprocessing │
                        │ (Transform)   │
                        │  - Fill gaps  │
                        │  - Normalize  │
                        │  - Returns    │
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │   Processed   │
                        │   Storage     │
                        │ data/processed/│
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────────┐
                        │ Train/Val/Test    │
                        │     Split         │
                        │  (Simple or CV)   │
                        └───────────────────┘
                                │
                        ┌───────┴────────┐
                        │                │
            Simple ▼                     ▼ CV Mode
        ┌─────────────────┐      ┌─────────────────┐
        │  70/15/15 Split │      │  k-fold CV (k=5)│
        │  (Default)      │      │  (--use-cv)     │
        └─────────────────┘      └─────────────────┘
                │                         │
                │                 ┌───────┼───────┐
                ▼                 ▼       ▼       ▼
        ┌──────────────┐   ┌─────────────────────┐
        │  3 Datasets  │   │  k Folds + Test     │
        │              │   │  (Expanding Window) │
        │ • Train 70%  │   │ • Fold 1-5: Train   │
        │ • Val 15%    │   │ • Fold 1-5: Val     │
        │ • Test 15%   │   │ • Test 15% (isolated)│
        └──────────────┘   └─────────────────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                    ┌───────────────┐
                    │ Model-Ready   │
                    │   Datasets    │
                    └───────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Analysis │    │Portfolio │    │Backtest  │
        │          │    │   Opt    │    │ (Future) │
        │  - ADF   │    │ (Future) │    │          │
        │  - ACF   │    │          │    │          │
        │  - Stats │    │          │    │          │
        └──────────┘    └──────────┘    └──────────┘
                │               │               │
                └───────────────┼───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │Visualizations │
                        │   & Reports   │
                        │               │
                        │ - JSON reports│
                        │ - PNG plots   │
                        │ - Dashboards  │
                        └───────────────┘
```

---

## Module Dependencies

### Core Pipeline Dependencies

```
scripts/run_etl_pipeline.py (Main Orchestrator)
    │
    ├─► etl/yfinance_extractor.py
    │       │
    │       ├─► etl/data_storage.py (for cache operations)
    │       ├─► yfinance (external library)
    │       ├─► retry_with_backoff (decorator)
    │       └─► vectorized_quality_check (function)
    │
    ├─► etl/data_validator.py
    │       │
    │       ├─► numpy (statistical operations)
    │       ├─► pandas (data manipulation)
    │       └─► validate_prices/volumes (methods)
    │
    ├─► etl/preprocessor.py
    │       │
    │       ├─► numpy (normalization)
    │       ├─► pandas (missing data handling)
    │       └─► handle_missing/normalize (methods)
    │
    └─► etl/data_storage.py
            │
            ├─► pandas (parquet I/O)
            ├─► pathlib (file operations)
            ├─► train_validation_test_split (method)
            └─► etl/time_series_cv.py (when use_cv=True) ⭐ NEW
                    │
                    ├─► TimeSeriesCrossValidator (class)
                    ├─► CVFold (dataclass)
                    └─► k-fold split generation

scripts/analyze_dataset.py (Analysis CLI)
    │
    └─► etl/time_series_analyzer.py
            │
            ├─► statsmodels (ADF test)
            ├─► scipy (statistical tests)
            ├─► pandas (data operations)
            ├─► numpy (vectorized calculations)
            │
            └─► Methods:
                ├─► load_and_inspect_data()
                ├─► analyze_missing_data()
                ├─► identify_temporal_structure()
                ├─► statistical_summary()
                ├─► test_stationarity()
                ├─► compute_autocorrelation()
                └─► generate_report()

scripts/visualize_dataset.py (Visualization CLI)
    │
    └─► etl/visualizer.py
            │
            ├─► matplotlib (plotting)
            ├─► seaborn (statistical plots)
            ├─► scipy (signal processing)
            ├─► statsmodels (decomposition)
            │
            └─► Methods:
                ├─► plot_time_series_overview()
                ├─► plot_distribution_analysis()
                ├─► plot_autocorrelation()
                ├─► plot_decomposition()
                ├─► plot_rolling_statistics()
                ├─► plot_spectral_density()
                └─► plot_comprehensive_dashboard()
```

---

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.12+ | Core implementation |
| **Data Processing** | Pandas | Latest | Time series manipulation |
| **Data Processing** | NumPy | Latest | Vectorized calculations |
| **Statistics** | Statsmodels | Latest | Time series analysis (ADF, ACF) |
| **Statistics** | SciPy | Latest | Statistical tests, signal processing |
| **Visualization** | Matplotlib | Latest | Base plotting |
| **Visualization** | Seaborn | Latest | Statistical visualizations |
| **Data Extraction** | yfinance | Latest | Yahoo Finance API |
| **Storage** | Parquet | Latest | Efficient columnar storage |
| **Testing** | Pytest | Latest | Unit and integration testing |
| **CLI** | Click | Latest | Command-line interfaces |
| **Config** | PyYAML | Latest | Configuration management |

### Development Tools

| Tool | Purpose |
|------|---------|
| **Virtual Environment** | simpleTrader_env (isolated dependencies) |
| **Version Control** | Git (code versioning) |
| **IDE Integration** | Claude Code (AI-assisted development) |
| **Documentation** | Markdown (comprehensive docs) |

---

## Performance Characteristics

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
| **Test Suite** | 26s | 63 tests (98.4% passing) |

### Cache Performance

| Metric | Value | Impact |
|--------|-------|--------|
| **Hit Rate** | 100% | After first run |
| **Speedup** | 20x | Compared to network fetch |
| **Storage** | 271 KB | 5 tickers cached |
| **Validity** | 24 hours | Configurable |
| **Network Savings** | 100% | Zero requests on cache hit |

---

## Code Quality Metrics

### Lines of Code

| Component | Lines | Complexity | Test Coverage |
|-----------|-------|------------|---------------|
| **yfinance_extractor.py** | 327 | Medium | 10 cache tests |
| **data_validator.py** | 117 | Low | 5 tests |
| **preprocessor.py** | 101 | Low | 8 tests |
| **data_storage.py** | 210 | Medium | 7 tests ⭐ |
| **time_series_cv.py** | 336 | Medium | 22 tests ⭐ NEW |
| **portfolio_math.py** | 45 | Low | 5 tests |
| **time_series_analyzer.py** | 500+ | High | 17 tests |
| **visualizer.py** | 600+ | High | Manual testing |
| **Scripts (5 files)** | 651 | Medium | Integration tests |
| **Tests (10 files)** | 1,558 | N/A | 100% passing ⭐ |
| **Total Production** | 4,057 | - | 85/85 tests ⭐ |

### Test Coverage Summary

| Test Suite | Tests | Passing | Coverage |
|------------|-------|---------|----------|
| **Cache Tests** | 10 | 10 (100%) | Comprehensive |
| **CV Tests** | 22 | 22 (100%) | Quantified improvements ⭐ NEW |
| **ETL Tests** | 27 | 27 (100%) | All passing ⭐ |
| **Analysis Tests** | 17 | 17 (100%) | Full coverage |
| **Math Tests** | 5 | 5 (100%) | All scenarios |
| **Storage Tests** | 7 | 7 (100%) | I/O + CV operations ⭐ |
| **Preprocessing Tests** | 8 | 8 (100%) | Edge cases |
| **Total** | 85 | 85 (100%) | Production ready ⭐ |

---

## Data Specifications

### Dataset Statistics (Real AAPL Data)

| Attribute | Value | Notes |
|-----------|-------|-------|
| **Total Observations** | 1,006 | 2020-01-02 to 2023-12-29 |
| **Training Set** | 704 (70%) | 2020-01-02 to 2022-09-16 |
| **Validation Set** | 151 (15%) | Model tuning |
| **Testing Set** | 151 (15%) | Final evaluation |
| **Frequency** | Daily | Business days only |
| **Missing Data** | 0.0% | All gaps filled |
| **Columns** | 5 | Open, High, Low, Close, Volume |
| **Cache Size** | 54 KB | Per ticker (parquet compressed) |

### File Formats

| Type | Format | Compression | Speed |
|------|--------|-------------|-------|
| **Raw Data** | Parquet | Snappy | 10x faster than CSV |
| **Processed Data** | Parquet | Snappy | Optimized I/O |
| **Analysis Results** | JSON | None | Human-readable |
| **Visualizations** | PNG | None | 150 DPI publication quality |

---

## Future Roadmap

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

## Production Readiness

### Checklist Status: ✅ ALL COMPLETE

- [x] **Code Quality**: Vectorized, type-hinted, documented
- [x] **Testing**: 63 tests (98.4% passing)
- [x] **Performance**: 20x speedup with caching
- [x] **Documentation**: 5 comprehensive documents
- [x] **Data Quality**: Real data validated (0% missing)
- [x] **Architecture**: Fully documented and diagrammed
- [x] **Error Handling**: Robust retry and fallback mechanisms
- [x] **Logging**: Comprehensive logging throughout
- [x] **Configuration**: YAML-based, externalized settings

---

**Document Version**: 4.0
**Last Updated**: 2025-10-04
**Status**: PRODUCTION READY ✅ (with k-fold CV support)
**Next Review**: Before Phase 5 implementation

---

## Recent Updates (v4.0)

### Phase 4.5: Time Series Cross-Validation
- **New Module**: `etl/time_series_cv.py` (336 lines)
- **Updated**: `etl/data_storage.py` (+52 lines for CV support)
- **New Tests**: 22 comprehensive CV tests (100% passing)
- **Quantifiable Improvement**: 5.5x temporal coverage (15% → 83%)
- **Temporal Gap**: Eliminated (0 years vs 2.5 years)
- **Backward Compatibility**: Maintained (use_cv=False default)
- **CLI Integration**: `--use-cv` flag in run_etl_pipeline.py
- **Documentation**: TIME_SERIES_CV.md (15 KB guide)

### Key Benefits
1. **Better Validation Coverage**: 83% vs 15% temporal range
2. **No Temporal Disparity**: Continuous expanding window
3. **Test Isolation**: 15% never exposed during CV
4. **Backward Compatible**: Zero breaking changes
5. **Fully Tested**: 85/85 tests passing (100%)

### Modular Configuration Architecture
- **8 comprehensive config files** (~40 KB total YAML)
- **Platform-agnostic design**: Support for yfinance, Alpha Vantage, Finnhub
- **Data source abstraction**: Unified interface for multiple providers
- **Extensible orchestrator**: Config-driven pipeline with failover support
- **Future-ready**: API keys stored in .env, easy to add new sources
- **Single source of truth**: Merged workflows/ into config/pipeline_config.yml

**New Configuration Files**:
- `pipeline_config.yml`: Unified orchestration (4-stage pipeline)
- `data_sources_config.yml`: Platform-agnostic data source registry
- `yfinance_config.yml`: Yahoo Finance extraction settings (cache, retry, rate limiting)
- `alpha_vantage_config.yml`: Alpha Vantage config (5 req/min free tier)
- `finnhub_config.yml`: Finnhub config (60 req/min free tier)
- `preprocessing_config.yml`: Missing data, normalization, outliers (4.8 KB)
- `validation_config.yml`: Quality checks, outlier detection (7.7 KB)
- `storage_config.yml`: Parquet settings, CV/simple split config (5.9 KB)
