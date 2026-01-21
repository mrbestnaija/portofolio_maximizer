# Portfolio Maximizer – Autonomous Profit Engine

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](https://github.com/mrbestnaija/portofolio_maximizer)

> End-to-end quantitative automation that ingests data, forecasts regimes, routes signals, and executes trades hands-free with profit as the north star.

**Version**: 3.0  
**Status**: Phase 7.3 ensemble hardening (GARCH confidence/DB migration in place; RMSE still above target)  
**Last Updated**: 2026-01-21

---

## 🎯 Overview

Portfolio Maximizer is a self-directed trading stack that marries institutional-grade ETL with autonomous execution. It continuously extracts, validates, preprocesses, forecasts, and trades financial time series so profit-focused decisions are generated without human babysitting.

### Current Phase & Scope (Jan 2026)
- Phase 7.3 ensemble fix: ENSEMBLE DB CHECK updated, model keys canonicalized, confidence scoring de-saturated, GARCH now present in confidence/weights. RMSE ratios improved but remain above the <1.1x target.
- Next up (Phase 8 canary): Neural forecasters (PatchTST/NHITS) and skforecast+XGBoost GPU for directional edge; Chronos-Bolt as a benchmark; GARCH retained for volatility sizing/stops. Horizon: 1-hour; training cadence: real-time plus daily batch; GPU: RTX 4060 Ti (CUDA 12.9).
- Guardrails: WSL + `simpleTrader_env/bin/python` only (see Documentation/RUNTIME_GUARDRAILS.md). LLM paths are optional/feature-flagged and being replaced by TS/GPU models.

### Key Features

- **🚀 Intelligent Caching**: 20x speedup with cache-first strategy (24h validity)
- **📊 Advanced Analysis**: MIT-standard time series analysis (ADF, ACF/PACF, stationarity)
- **📈 Publication-Quality Visualizations**: 8 professional plots with 150 DPI quality
- **🔄 Robust ETL Pipeline**: 4-stage pipeline with comprehensive validation
- **✅ Comprehensive Testing**: 141+ tests with high coverage across ETL, LLM, and integration modules
- **⚡ High Performance**: Vectorized operations, Parquet format (10x faster than CSV)
- **🧠 Modular Orchestration**: Dataclass-driven pipeline runner coordinating CV splits, neural/TS stages, and ticker discovery with auditable logging
- **🔐 Resilient Data Access**: Hardened Yahoo Finance extraction with pooling to reduce transient failures
- **🤖 Autonomous Profit Engine**: `scripts/run_auto_trader.py` keeps the signal router + trading engine firing so positions are sized and executed automatically

---

### ♻️ Latest Enhancements (Jan 2026)

- ENSEMBLE DB migration: CHECK now allows `ENSEMBLE`; added busy_timeout for contention-resistant writes.
- Ensemble stability: canonicalized model keys, added confidence instrumentation, de-saturated scoring to avoid winner-takes-all, and ensured GARCH stays present with conservative fallback.
- Tests: added confidence monotonicity coverage (`tests/test_ensemble_confidence.py`).
- GPU/Neural forecaster plan: 1-hour horizon, real-time + daily retrain, GPU-enabled PatchTST/NHITS and skforecast+XGBoost under canary flags; Chronos-Bolt as zero-shot benchmark (see Documentation/MODEL_SIGNAL_REFACTOR_PLAN.md and Documentation/GPU_PARALLEL_RUNNER_CHECKLIST.md).

## Academic Rigor & Reproducibility (MIT-style)

- **Traceable artifacts**: Log config + commit hashes alongside experiment IDs; keep hashes for data snapshots and generated plots (`logs/artifacts_manifest.jsonl` when present).
- **Deterministic runs**: Set and record seeds (`PYTHONHASHSEED`, RNG, hyper-opt samplers, RL) for every reported experiment; prefer config overrides over ad hoc flags.
- **Executable evidence**: Each figure/table used for publication should have a runnable script/notebook (target: `reproducibility/` folder) that regenerates it from logged artifacts.
- **Transparency**: Document MTM assumptions, cost models, and cron wiring in experiment notes; link back to `Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md` for the publication plan and replication checklist.
- **Archiving plan**: Package replication bundles (configs, logs, plots, minimal sample data) for Zenodo/Dataverse deposit before submitting any paper/thesis.

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
│              Portfolio Maximizer                          │
│              Production-Ready System                     │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
  Layer 1:        Layer 2:         Layer 3:
  Extraction      Storage          Validation
  (yfinance &     (Parquet         (Quality
   multi-source)  Format)          Checks)
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

- Python 3.10-3.12
- pip package manager
- Virtual environment (recommended)
- **Ollama** (optional, for LLM features): Local LLM server for market analysis and signal generation
  - Installation: `curl -s https://raw.githubusercontent.com/ollama/ollama/main/install.sh | sh`
  - Start server: `ollama serve`
  - Pull models: `ollama pull deepseek-coder:6.7b-instruct-q4_K_M`
  - See [LLM Configuration](#llm-integration) for details

### Setup

```bash
# Clone the repository
git clone https://github.com/mrbestnaija/portofolio_maximizer.git
cd portofolio_maximizer

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

# Cache Settings
CACHE_VALIDITY_HOURS=24
```

### LLM Integration (Optional and being phased out)

LLM paths remain available but are optional; the roadmap prioritizes TS/GPU forecasters. If you still need LLM-powered analysis, ensure Ollama is running locally; otherwise keep `--enable-llm` off for speed/energy savings.

---

## ⚡ Quick Start

### Run the ETL Pipeline

```bash
# Activate virtual environment
source simpleTrader_env/bin/activate

# Recommended: live run with automatic synthetic fallback
python scripts/run_etl_pipeline.py \
  --tickers AAPL,MSFT \
  --include-frontier-tickers \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --execution-mode auto \
  --enable-llm

# Force live-only execution (fails fast on network/API issues)
python scripts/run_etl_pipeline.py --execution-mode live

# Offline validation (synthetic data, no network)
python scripts/run_etl_pipeline.py --execution-mode synthetic --enable-llm --include-frontier-tickers

# Expected output:
# ✓ Extraction complete (cache hit: <0.1s)
# ✓ Validation complete (0.1s)
# ✓ Preprocessing complete (0.2s)
# ✓ Storage complete (0.1s)
# Total time: varies with mode (synthetic ≈ 1s, live depends on APIs)

# Shortcut runner (auto mode with logs):
./bash/run_pipeline_live.sh
```

`--include-frontier-tickers` automatically adds the Nigeria → Bulgaria frontier symbols
curated in `etl/frontier_markets.py` (see `Documentation/arch_tree.md`) so every multi-ticker
training or validation run exercises less-liquid market scenarios. Synthetic mode is
recommended until provider-specific ticker mappings are finalized.

### Launch The Autonomous Trading Loop

```bash
python scripts/run_auto_trader.py \
  --tickers AAPL,MSFT,NVDA \
  --include-frontier-tickers \
  --lookback-days 365 \
  --forecast-horizon 30 \
  --initial-capital 25000 \
  --cycles 5 \
  --sleep-seconds 900
```

Add `--enable-llm` to activate the Ollama-backed fallback router whenever the ensemble hesitates. Each cycle:

1. Streams fresh OHLCV windows via `DataSourceManager` with cache-first failover.
2. Validates, imputes, and feeds the data into the SARIMAX/SAMOSSA/GARCH/MSSA-RL ensemble.
3. Routes the highest-confidence trade and executes it through `PaperTradingEngine`, tracking cash, PnL, and open positions in real time.

### Higher‑Order Hyper‑Parameter Optimization (Default Orchestration Mode)

For post‑implementation evaluation and regime‑aware tuning, the project includes a higher‑order
hyper‑parameter driver that wraps ETL → auto‑trader → strategy optimization in a stochastic loop.
This driver treats configuration knobs such as:

- Time window (`START` / `END` evaluation dates),
- Quant success `min_expected_profit`,
- Time Series `time_series.min_expected_return`

as higher‑order hyper‑parameters and searches over them non‑convexly using a bandit‑style
explore/exploit policy (30% explore / 70% exploit by default, dynamically adjusted).

The canonical entrypoint is:

```bash
# Run a 5‑round higher‑order hyper‑parameter search
HYPEROPT_ROUNDS=5 bash/bash/run_post_eval.sh
```

Each round:
- Generates temporary override configs (`config/quant_success_config.hyperopt.yml`,
  `config/signal_routing_config.hyperopt.yml`),
- Runs `scripts/run_etl_pipeline.py`, `scripts/run_auto_trader.py`,
  and `scripts/run_strategy_optimization.py` against a dedicated DB,
- Scores the run by realized `total_profit` over a short evaluation window,
- Logs trial parameters and scores to `logs/hyperopt/hyperopt_<RUN_ID>.log`,
- Maintains a 30/70 explore/exploit policy that slowly shifts toward exploitation
  as better configurations are discovered.

The best configuration is re‑run as `<RUN_ID>_best` and surfaced in
`visualizations/dashboard_data.json` so dashboards and downstream tools can treat it
as the current regime‑specific optimum (without hardcoding it in code).

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
portfolio_maximizer/
│
├── config/                          # Configuration files (YAML)
│   ├── analysis_config.yml          # Time series analysis parameters
│   ├── preprocessing_config.yml     # Preprocessing settings
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
│   └── implementation_checkpoint.md # Implementation status
│
├── .local_automation/              # Local automation helpers (kept private)
│   ├── developer_notes.md          # Automation & tooling playbook
│   └── settings.local.json         # Local agent settings
│
├── etl/                             # ETL pipeline modules
│   ├── yfinance_extractor.py       # Yahoo Finance extraction
│   ├── data_validator.py           # Data quality validation
│   ├── preprocessor.py             # Data preprocessing
│   ├── data_storage.py             # Data persistence
│   ├── portfolio_math.py           # Financial calculations
│   ├── time_series_analyzer.py     # Time series analysis
│   └── visualizer.py               # Visualization engine
│
├── scripts/                         # Executable scripts
│   ├── run_etl_pipeline.py         # Main ETL orchestration
│   ├── run_auto_trader.py          # Autonomous profit loop
│   ├── analyze_dataset.py          # Analysis CLI
│   ├── visualize_dataset.py        # Visualization CLI
│   └── validate_environment.py     # Environment validation
│
├── tests/                           # Test suite (141+ tests)
│   ├── etl/                        # ETL module tests
│   │   ├── test_yfinance_cache.py
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

| Test Suite | Tests | Status | Coverage |
|------------|-------|--------|----------|
| **ETL Tests** | 73+ | Passing | Core pipeline functions |
| **LLM Integration Tests** | 20+ | Passing | Market analysis, signals, risk |
| **Integration Tests** | 19+ | Passing | End-to-end workflows |
| **Profit-Critical Tests** | 12+ | Passing | Financial calculations |
| **Security Tests** | 10+ | Passing | Data protection |
| **Total** | 141+ | Production ready | Comprehensive coverage |

---

## 📚 Documentation

Comprehensive documentation is available in the `Documentation/` directory:

- **[Core Documentation (Institutional)](Documentation/CORE_PROJECT_DOCUMENTATION.md)**: Canonical docs, evidence standards, and verification ladder
- **[Metrics & Evaluation](Documentation/METRICS_AND_EVALUATION.md)**: Unambiguous metric definitions (PF/WR/Sharpe/DM-style tests)
- **[Architecture Tree](Documentation/arch_tree.md)**: Complete architecture overview
- **[Caching Implementation](Documentation/CACHING_IMPLEMENTATION.md)**: Intelligent caching guide
- **[Git Workflow](Documentation/GIT_WORKFLOW.md)**: Local-first git workflow
- **[Project Status](Documentation/PROJECT_STATUS.md)**: Current verified snapshot + reproducible commands
- **[Implementation Checkpoint](Documentation/implementation_checkpoint.md)**: Development status
- **[Research Plan](Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md)**: Research questions, protocols, and reproducibility checklist
- **[Cron Automation](Documentation/CRON_AUTOMATION.md)**: Production-style scheduling + evidence freshness wiring

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

### Client-only sync (GitHub as source of truth)

For follower/client PCs where GitHub `master` is the definitive source and local changes should never push upstream, use the guarded sync helper:

```bash
# Sync current branch from GitHub (auto-stash dirty worktrees)
bash/git_syn_to_local.sh

# Sync a specific branch from GitHub
bash/git_syn_to_local.sh master
```

- Lives at `bash/git_syn_to_local.sh` (run from repo root).
- Auto-stashes uncommitted work, fetches, rebases, and restores the stash when safe.
- Never pushes; warns if local-only commits exist so you can reconcile from the master PC.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Bestman Ezekwu Enock**

- GitHub: [@mrbestnaija](https://github.com/mrbestnaija)
- Public contact: See Support section

---

## 🙏 Acknowledgments

- **MIT OpenCourseWare**: Micro Masters in Statistics and Data Science (MMSDS)
- **Wife**: Linda Bestman
- **Yahoo Finance**: Market data API

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

# Check Python version
python --version  # Should be 3.10+
```

---

## 📞 Support

For questions or issues:

1. **Check Documentation**: `Documentation/` directory
2. **Search Issues**: [GitHub Issues](https://github.com/mrbestnaija/portofolio_maximizer/issues)
3. **Open New Issue**: Provide reproducible example
4. **Public contact email**: csgtmalice@protonmail.ch

---

**Built with ❤️ using Python, NumPy and Pandas**

**Status**: Production Ready ✅
**Version**: 3.0
**Last Updated**: 2025-12-04

