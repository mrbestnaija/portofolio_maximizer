# Portfolio Maximizer – Autonomous Profit Engine

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase 7.9 In Progress](https://img.shields.io/badge/Phase%207.9-In%20Progress-blue.svg)](Documentation/EXIT_ELIGIBILITY_AND_PROOF_MODE.md)
[![Tests: 731](https://img.shields.io/badge/tests-731%20(718%20passing)-success.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-informational.svg)](Documentation/)
[![Research Ready](https://img.shields.io/badge/research-reproducible-purple.svg)](#-research--reproducibility)

> End-to-end quantitative automation that ingests data, forecasts regimes, routes signals, and executes trades hands-free with profit as the north star.

**Version**: 4.2
**Status**: Phase 7.9 In Progress - Cross-session persistence, proof-mode validation, UTC normalization
**Last Updated**: 2026-02-09

---

## 🎯 Overview

Portfolio Maximizer is a self-directed trading stack that marries institutional-grade ETL with autonomous execution. It continuously extracts, validates, preprocesses, forecasts, and trades financial time series so profit-focused decisions are generated without human babysitting.

### Current Phase & Scope (Jan 2026)

**Phase 7.8 Complete** - All-Regime Weight Optimization:

- **3/6 regimes optimized** with SAMOSSA-dominant weights:
  - **CRISIS**: 60.69% RMSE improvement (17.15 → 6.74), 72% SAMOSSA
  - **MODERATE_MIXED**: 6.30% improvement (17.63 → 16.52), 73% SAMOSSA
  - **MODERATE_TRENDING**: 65.07% improvement (20.86 → 7.29), 90% SAMOSSA
- **Key Finding**: SAMOSSA dominates ALL regimes (72-90%), contradicting initial GARCH hypothesis
- **Method**: Rolling cross-validation with scipy.optimize.minimize (3+ years of AAPL data)
- **Validation**: 2/20 holdout audits complete

**Phase 7.9 In Progress** - Holdout Audit Accumulation:

- Current: 2/20 audits complete
- Target: 20 audits for production deployment decision
- 3 regimes not optimized (insufficient samples): HIGH_VOL_TRENDING, MODERATE_RANGEBOUND, LIQUID_RANGEBOUND

**System Architecture**:
- Regime-aware ensemble routing with adaptive model selection
- 4 forecasting models: SARIMAX, GARCH, SAMOSSA, MSSA-RL
- Quantile-based confidence calibration (Phase 7.4)
- Rolling cross-validation optimization framework
- Comprehensive logging with phase-organized structure

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

### Latest Enhancements (Jan 2026)

**Phase 7.8 Achievements**:

- All-regime weight optimization (3/6 regimes) with ~60-65% RMSE improvement for CRISIS/MODERATE_TRENDING and +6.30% for MODERATE_MIXED
- SAMOSSA dominance finding: 72-90% across ALL optimized regimes
- CRISIS regime optimization contradicts initial GARCH hypothesis
- Updated configuration files with data-driven weights
- Comprehensive documentation: [PHASE_7.8_RESULTS.md](Documentation/PHASE_7.8_RESULTS.md)

**Phase 7.7 Achievements**:

- Per-regime weight optimization framework established
- Organized log directory structure with phase-specific subdirectories
- Automated log organization script ([bash/organize_logs.sh](bash/organize_logs.sh))

**Infrastructure Improvements**:
- ENSEMBLE DB migration: CHECK constraint updated, busy_timeout for write resilience
- Enhanced confidence scoring with model key canonicalization
- SQLite read-only connections with immutable URI mode (WSL/DrvFS robustness)
- Position-based forecast alignment fallback for calendar vs business day handling
- Regime detection feature flag with instant enable/disable capability

## Academic Rigor & Reproducibility (MIT-style)

- **Traceable artifacts**: Log config + commit hashes alongside experiment IDs; keep hashes for data snapshots and generated plots (`logs/artifacts_manifest.jsonl` when present).
- **Deterministic runs**: Set and record seeds (`PYTHONHASHSEED`, RNG, hyper-opt samplers, RL) for every reported experiment; prefer config overrides over ad hoc flags.
- **Executable evidence**: Each figure/table used for publication should have a runnable script/notebook (target: `reproducibility/` folder) that regenerates it from logged artifacts.
- **Transparency**: Document MTM assumptions, cost models, and cron wiring in experiment notes; link back to `Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md` for the publication plan and replication checklist.
- **Archiving plan**: Package replication bundles (configs, logs, plots, minimal sample data) for Zenodo/Dataverse deposit before submitting any paper/thesis.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Phase 7.8 Results](#-phase-78-results-all-regime-optimization)
- [Phase 7.9 Status](#-phase-79-cross-session-persistence--proof-mode)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Research & Reproducibility](#-research--reproducibility)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎖️ Phase 7.8 Results: All-Regime Optimization

### Key Results

**3/6 Regimes Optimized** with SAMOSSA-dominant weights:

| Regime | Samples | Folds | RMSE Before | RMSE After | Improvement | Optimal Weights |
|--------|---------|-------|-------------|------------|-------------|-----------------|
| **CRISIS** | 25 | 5 | 17.15 | 6.74 | **+60.69%** | 72% SAMOSSA, 23% SARIMAX, 5% MSSA-RL |
| **MODERATE_MIXED** | 20 | 4 | 17.63 | 16.52 | +6.30% | 73% SAMOSSA, 22% MSSA-RL, 5% SARIMAX |
| **MODERATE_TRENDING** | 50 | 10 | 20.86 | 7.29 | **+65.07%** | 90% SAMOSSA, 5% SARIMAX, 5% MSSA-RL |

### Major Finding: SAMOSSA Dominance

**SAMOSSA dominates ALL optimized regimes (72-90%)**, contradicting initial hypothesis that GARCH would be optimal for CRISIS regime.

- Pattern recognition outperforms volatility modeling across all market conditions
- CRISIS regime: SAMOSSA (72%) + SARIMAX (23%) provides best defensive configuration
- MODERATE_TRENDING: Confirms Phase 7.7 results with 2x sample size validation

### Configuration Updates

```yaml
# config/forecasting_config.yml (lines 98-115)
regime_candidate_weights:
  CRISIS:
    - {sarimax: 0.23, samossa: 0.72, mssa_rl: 0.05}
  MODERATE_MIXED:
    - {sarimax: 0.05, samossa: 0.73, mssa_rl: 0.22}
  MODERATE_TRENDING:
    - {sarimax: 0.05, samossa: 0.90, mssa_rl: 0.05}
```

### Regimes Not Optimized (Insufficient Samples)

| Regime | Reason | Recommendation |
|--------|--------|----------------|
| **HIGH_VOL_TRENDING** | Rare in AAPL 2024-2026 data | Test with NVDA (higher volatility) |
| **MODERATE_RANGEBOUND** | Rare in trending market | Use default weights |
| **LIQUID_RANGEBOUND** | Very rare (stable markets) | Use default weights |

**Full Results**: [Documentation/PHASE_7.8_RESULTS.md](Documentation/PHASE_7.8_RESULTS.md)

---

## 🚀 Phase 7.9: Cross-Session Persistence & Proof Mode

### Objective

Establish reliable round-trip trade execution with cross-session position persistence, enabling profitability validation and holdout audit accumulation.

### Current Status

- **Closed trades**: 30 validated (proof-mode TIME_EXIT)
- **Holdout audits**: 9/20 (forecast audit gate active at 25% max violation rate)
- **UTC normalization**: Complete across execution and persistence layers
- **Frequency compatibility**: Deprecated pandas aliases (`'H'` -> `'h'`) resolved

### Key Components

- **Cross-session persistence**: `portfolio_state` + `portfolio_cash_state` tables via `--resume`
- **Proof mode** (`--proof-mode`): Tight max_holding (5d/6h), ATR stops/targets, flatten-before-reverse
- **Audit sprint**: `bash/run_20_audit_sprint.sh` with gate enforcement (forecast, quant health, dashboard)
- **UTC timestamps**: `etl/timestamp_utils.py` (`ensure_utc()`, `utc_now()`, `ensure_utc_index()`)

### Validation Commands

```bash
# Run proof-mode audit sprint
PROOF_MODE=1 RISK_MODE=research_production bash bash/run_20_audit_sprint.sh

# Check closed trades
python -c "
import sqlite3
conn = sqlite3.connect('data/portfolio_maximizer.db')
closed = conn.execute('SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NOT NULL').fetchone()[0]
print(f'Closed trades with realized PnL: {closed}')
conn.close()
"
```

### Success Criteria

- [x] Cross-session position persistence working
- [x] Proof mode creates guaranteed round trips
- [x] UTC-aware timestamps across all layers
- [ ] 20/20 holdout audits accumulated
- [ ] Forecast audit gate violation rate < 25%

### Phase 7.10: Production Deployment (Future)

Prerequisites:

- 20/20 audits passed
- All 3 optimized regimes show consistent improvement
- Overall RMSE regression confirmed <25%

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

LLM/Ollama integration is disabled by default to avoid unnecessary startup delays when Ollama is not running. The roadmap prioritizes TS/GPU forecasters; if you still need the legacy LLM path for experiments, set `PM_ENABLE_OLLAMA=1` and use `--enable-llm` where supported.

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
   --execution-mode auto

# Force live-only execution (fails fast on network/API issues)
python scripts/run_etl_pipeline.py --execution-mode live

# Offline validation (synthetic data, no network)
python scripts/run_etl_pipeline.py --execution-mode synthetic --include-frontier-tickers

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

Auto-trader **resumes persisted positions by default** (`--resume` is on). Use `--no-resume` to start fresh from `--initial-capital`, or run `bash/reset_portfolio.sh` to clear the saved state. Existing databases should run the one-time migration: `python scripts/migrate_add_portfolio_state.py`.

For scheduled daily+intraday passes, use `bash/run_daily_trader.sh` (WSL/Linux) or `run_daily_trader.bat` (Windows Task Scheduler); both runs keep positions via `--resume`.

Add `--enable-llm` (plus `PM_ENABLE_OLLAMA=1`) to activate the legacy Ollama-backed fallback router whenever the ensemble hesitates. Each cycle:

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

### cTrader Credentials Precedence (Demo/Live)

The cTrader client resolves credentials in this order:

1. **Environment‑specific keys** (`CTRADER_DEMO_*` or `CTRADER_LIVE_*`)
2. **Generic keys** (`USERNAME_CTRADER` / `CTRADER_USERNAME`, `PASSWORD_CTRADER` / `CTRADER_PASSWORD`, `APPLICATION_NAME_CTRADER` / `CTRADER_APPLICATION_ID`)
3. **Email fallback** (`EMAIL_CTRADER` / `CTRADER_EMAIL`) if username is missing

This allows demo + live to run side‑by‑side without cross‑env leakage.

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
├── tests/                           # Test suite (731 tests)
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
| **ETL Tests** | 300+ | Passing | Core pipeline, caching, checkpoints |
| **LLM Integration Tests** | 30+ | Passing | Market analysis, signals, risk |
| **Forecaster Tests** | 150+ | Passing | SARIMAX, GARCH, SAMOSSA, MSSA-RL, ensemble |
| **Integration Tests** | 100+ | Passing | End-to-end workflows |
| **Execution Tests** | 80+ | Passing | Order management, paper trading |
| **Security Tests** | 20+ | Passing | Data protection, credentials |
| **Total** | 731 | 718 passing, 6 skipped, 7 xfailed | Comprehensive coverage |

---

## 📚 Documentation

### Core Documentation

- **[Core Project Documentation](Documentation/CORE_PROJECT_DOCUMENTATION.md)**: Canonical docs, evidence standards, and verification ladder
- **[Metrics & Evaluation](Documentation/METRICS_AND_EVALUATION.md)**: Unambiguous metric definitions (PF/WR/Sharpe/DM-style tests)
- **[Architecture Tree](Documentation/arch_tree.md)**: Complete architecture overview
- **[Git Workflow](Documentation/GIT_WORKFLOW.md)**: Local-first git workflow
- **[Project Status](Documentation/PROJECT_STATUS.md)**: Current verified snapshot + reproducible commands
- **[OpenClaw Integration](Documentation/OPENCLAW_INTEGRATION.md)**: OpenClaw workspace skill + optional notifications (https://openclaw.ai)
- **[Pluggable Feature Engineering Pipeline](Documentation/PLUGGABLE_FEATURE_ENGINEERING_PIPELINE.md)**: Add-ons policy, gating, compute budgets
- **[Sentiment Feature Add-on](Documentation/SENTIMENT_FEATURE_ADDON.md)**: Sentiment feature spec (profit-gated, compute-aware)
- **[Feature Add-on Promotion Checklist](Documentation/FEATURE_ADDON_PROMOTION_CHECKLIST.md)**: Evidence template + promotion bar

### Phase 7 Documentation (Regime Detection & Optimization)

**Phase 7.7 - Per-Regime Optimization**:
- **[PHASE_7.7_WEIGHT_OPTIMIZATION.md](Documentation/PHASE_7.7_WEIGHT_OPTIMIZATION.md)** (380 lines): Complete optimization analysis and results
- **[PHASE_7.7_FINAL_SUMMARY.md](Documentation/PHASE_7.7_FINAL_SUMMARY.md)** (403 lines): Handoff summary and system state
- **[LOG_ORGANIZATION_SUMMARY.md](Documentation/LOG_ORGANIZATION_SUMMARY.md)**: Log structure and best practices

**Phase 7.8 - All-Regime Optimization (Complete)**:
- **[PHASE_7.8_RESULTS.md](Documentation/PHASE_7.8_RESULTS.md)**: Final results, weights, and validation plan
- **[PHASE_7.8_MANUAL_RUN_GUIDE.md](Documentation/PHASE_7.8_MANUAL_RUN_GUIDE.md)**: Reproduction/manual execution guide

**Phase 7.5 - Regime Detection Integration**:
- **[PHASE_7.5_VALIDATION.md](Documentation/PHASE_7.5_VALIDATION.md)** (340 lines): Single-ticker validation results
- **[PHASE_7.5_MULTI_TICKER_RESULTS.md](Documentation/PHASE_7.5_MULTI_TICKER_RESULTS.md)** (340 lines): Multi-ticker analysis

**Phase 7.6 - Threshold Tuning**:
- **[PHASE_7.6_THRESHOLD_TUNING.md](Documentation/PHASE_7.6_THRESHOLD_TUNING.md)**: Threshold optimization experiment

### Operational Documentation

- **[logs/README.md](logs/README.md)** (380 lines): Log structure, search patterns, retention policies
- **[Caching Implementation](Documentation/CACHING_IMPLEMENTATION.md)**: Intelligent caching guide
- **[Cron Automation](Documentation/CRON_AUTOMATION.md)**: Production-style scheduling + evidence freshness wiring
- **[Production Security + Profitability Runbook](Documentation/PRODUCTION_SECURITY_AND_PROFITABILITY_RUNBOOK.md)**: strict CVE defaults, temporary overrides, and gate-clearance workflow
- **[Implementation Checkpoint](Documentation/implementation_checkpoint.md)**: Development status

---

## 🔬 Research & Reproducibility

### For Researchers and Academics

This project follows MIT-standard statistical rigor and reproducibility practices:

**Reproducibility Standards**:
- All experiments logged with commit hashes and configuration snapshots
- Deterministic runs with documented seed values (PYTHONHASHSEED, RNG, hyper-opt samplers)
- Traceable artifacts in `logs/artifacts_manifest.jsonl` (when present)
- Executable evidence: Every figure/table has a runnable script for regeneration

**Research Documentation**:
- **[Research Progress and Publication Plan](Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md)**: Research questions, protocols, and replication checklist
- **[Agent Development Checklist](Documentation/AGENT_DEV_CHECKLIST.md)**: Overall project status and audit trail

### Citation

If you use this work in academic research, please cite:

```bibtex
@software{bestman2026portfolio,
  title={Portfolio Maximizer: Autonomous Quantitative Trading with Regime-Adaptive Ensemble},
  author={Bestman, Ezekwu Enock},
  year={2026},
  version={4.2},
  url={https://github.com/mrbestnaija/portofolio_maximizer},
  note={Phase 7.9: Cross-session persistence, proof-mode validation, UTC normalization}
}
```

### Key Research Results

**Phase 7.7 Optimization** (January 2026):
- **Method**: Rolling cross-validation with scipy.optimize.minimize
- **Dataset**: AAPL (2023-01-01 to 2026-01-18, 3+ years)
- **Results**: 65% RMSE reduction for MODERATE_TRENDING regime (19.26 → 6.74)
- **Optimal Configuration**: 90% SAMOSSA, 5% SARIMAX, 5% MSSA-RL
- **Validation**: Multi-ticker testing (AAPL, MSFT, NVDA) with 53% adaptation rate

**Phase 7.5 Regime Detection** (January 2026):
- **Regimes Identified**: 6 market regimes based on volatility, trend strength, Hurst exponent
- **Adaptation Performance**: 53% of forecasts switched to regime-specific weights
- **Cross-Ticker Validation**: Consistent regime detection across 3 tickers
- **Historical note**: early regime-weight experiments showed RMSE regressions on some windows; current governance is driven by the forecast-audit gate (see `scripts/check_forecast_audits.py` and `Documentation/ENSEMBLE_MODEL_STATUS.md`).

### Replication Instructions

**Full Replication Package**:

1. **Environment Setup**:
   ```bash
   git clone https://github.com/mrbestnaija/portofolio_maximizer.git
   cd portofolio_maximizer
   python -m venv simpleTrader_env
   source simpleTrader_env/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Reproduce Phase 7.7 Results**:
   ```bash
   # Run optimization (matches reported results)
   python scripts/optimize_ensemble_weights.py \
       --source rolling_cv \
       --tickers AAPL \
       --start-date 2023-01-01 \
       --end-date 2026-01-18 \
       --horizon 5 \
       --min-train-size 180 \
       --step-size 20 \
       --max-folds 10 \
       --min-samples-per-regime 25 \
       --output data/phase7.7_replication.json

   # Validate results
   python scripts/run_etl_pipeline.py \
       --tickers AAPL \
       --start 2024-07-01 \
       --end 2026-01-18 \
       --execution-mode auto
   ```

3. **Access Artifacts**:
   - Configuration: `config/forecasting_config.yml` (lines 87, 98-109)
   - Results: `data/phase7.7_optimized_weights.json`
   - Logs: `logs/phase7.7/phase7.7_weight_optimization.log`
   - Documentation: `Documentation/PHASE_7.7_*.md`

**Data Availability**:
- Market data: Yahoo Finance (publicly available via yfinance library)
- Synthetic data generator: `scripts/generate_synthetic_dataset.py`
- Database schema: SQLite with documented migrations in `scripts/migrate_*.py`

### Transparency & Assumptions

**Mark-to-Market Assumptions**:
- Transaction costs: 0.8 bps for liquid US stocks (configurable in `config/execution_cost_model.yml`)
- Slippage: Market impact model with square-root scaling
- Position sizing: Risk-managed via `risk/barbell_policy.py`

**Model Assumptions**:
- Stationarity: ADF test validation before forecasting
- Seasonality: Automatic detection and detrending
- Volatility clustering: GARCH modeling for time-varying volatility

**Known Limitations**:
- Only 3/6 regimes optimized (remaining regimes lack samples in the current AAPL window)
- Ensemble governance labels can mark individual windows `RESEARCH_ONLY` when the 2% promotion margin is not met; this is a monitoring/promotion label, not proof the ensemble forecast is unused (see `Documentation/ENSEMBLE_MODEL_STATUS.md`).
- Limited to US equity markets (frontier markets in synthetic mode)
- No live broker integration (paper trading engine)

---

## 🛣️ Roadmap

### Phase 7: Regime Detection & Ensemble Optimization (In Progress)

**Completed**:
- ✅ Phase 7.3: GARCH ensemble integration with confidence calibration
- ✅ Phase 7.4: Quantile-based confidence calibration (29% RMSE improvement)
- ✅ Phase 7.5: Regime detection integration (6 market regimes, multi-ticker validation)
- ✅ Phase 7.6: Threshold tuning experiments
- ✅ Phase 7.7: Per-regime weight optimization (65% RMSE reduction for MODERATE_TRENDING)
- ✅ Phase 7.8: All-regime optimization (3/6 regimes optimized; SAMOSSA dominance confirmed)

**Current**:
- Phase 7.9: Forecast-audit accumulation (holding period met)
  - As of 2026-02-04: `scripts/check_forecast_audits.py` reports **25** effective audits with RMSE and **Decision: KEEP**
  - Evidence + interpretation: `Documentation/ENSEMBLE_MODEL_STATUS.md`

**Upcoming**:
- Phase 7.10: Production deployment with full regime coverage

### Phase 8: Neural Forecasters & GPU Acceleration (Planned)

- PatchTST/NHITS integration with 1-hour horizon
- skforecast + XGBoost GPU for directional edge
- Chronos-Bolt as zero-shot benchmark
- Real-time + daily batch training cadence
- RTX 4060 Ti (CUDA 12.9) utilization

### Phase 9: Portfolio Optimization (Future)

- Mean-variance optimization (Markowitz)
- Risk parity portfolio
- Black-Litterman model
- Constraint handling (long-only, sector limits)

### Phase 10: Advanced Risk Modeling (Future)

- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Expected Shortfall (CVaR)
- Maximum Drawdown analysis with regime conditioning
- Stress testing framework

### Infrastructure Enhancements

**Caching**:
- ✅ 20x speedup with intelligent caching (24h validity)
- 🔄 Smart cache invalidation (market close triggers)
- Distributed caching (Redis/Memcached) for multi-node
- Cache analytics dashboard

**Monitoring & Observability**:
- Enhanced log organization with phase-specific directories
- Grafana/Loki integration (documented in logs/README.md)
- Real-time model health monitoring
- Automated performance degradation alerts

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

### Code Metrics

- **Total Production Code**: 10,000+ lines
- **Test Code**: 5,000+ lines
- **Test Suite**: 731 tests (718 passing, 6 skipped, 7 xfailed)
- **Test Coverage**: Comprehensive across all modules
- **Documentation**: 90+ comprehensive files

### Performance Metrics

- **Cache Performance**: 20x speedup with intelligent caching
- **Data Quality**: 0% missing data (after preprocessing)
- **Optimization Results**: 65% RMSE reduction (Phase 7.7, MODERATE_TRENDING)
- **Regime Detection**: 53% adaptation rate across multi-ticker validation
- **Model Ensemble**: 4 forecasters (SARIMAX, GARCH, SAMOSSA, MSSA-RL)

### Phase 7 Progress

- **Phases Completed**: 9 (7.0 - 7.8)
- **Current Phase**: 7.9 (Cross-session persistence, proof-mode validation)
- **Regimes Optimized**: 3/6 (CRISIS, MODERATE_MIXED, MODERATE_TRENDING)
- **Closed Trades**: 30 validated (proof-mode TIME_EXIT)
- **Audit Progress**: 9/20 (production deployment gate)

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

---

## 🎯 Current Status Summary

**Phase 7.7**: ✅ Complete (Per-Regime Optimization)
**Phase 7.8**: ✅ Complete (All-Regime Optimization)
**Phase 7.9**: 🔄 In Progress (Cross-session persistence, proof-mode, 9/20 audits)
**Production Status**: Research Phase (awaiting audit gate - 9/20 complete)

**Latest Achievements**:
- 3/6 regimes optimized with data-driven weights (SAMOSSA dominance confirmed)
- CRISIS and MODERATE_TRENDING regimes: ~60-65% RMSE reduction (MODERATE_MIXED: +6.30%)
- Cross-session position persistence via portfolio_state tables
- Proof-mode validation with 30 closed trades
- UTC-aware timestamps across all system layers
- SARIMAX disabled by default (15x single-forecast speedup)
- 731 tests collected (718 passing)

**Next Steps**:
1. Accumulate remaining holdout audits (target: 20/20, currently 9/20)
2. Validate RMSE regression targets and per-regime stability
3. Expand optimization coverage with multi-ticker data (for rare regimes)
4. Production deployment decision gate

---

**Built with Python, NumPy, Pandas, and SciPy**

**Version**: 4.2
**Status**: Phase 7.9 In Progress - Cross-session persistence & proof-mode validation
**Last Updated**: 2026-02-09
