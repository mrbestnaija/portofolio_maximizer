# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Portfolio Maximizer is an autonomous quantitative trading system that extracts financial data, forecasts market regimes, routes trading signals, and executes trades automatically. It's a production-ready Python system with institutional-grade ETL pipelines, LLM integration, and comprehensive testing.

**Current Phase**: Phase 7.9 Complete (PnL integrity enforcement, adversarial audit, OpenClaw automation, Interactions API)
**Last Updated**: 2026-02-17

---

## Development Environment & Platform Considerations

### Python Environment
```bash
# REQUIRED: Always activate virtual environment first
source simpleTrader_env/bin/activate  # Linux/Mac
simpleTrader_env\Scripts\activate     # Windows

# Supported Python: >=3.10,<3.13
# Current packages: See requirements.txt (last updated 2026-01-31)
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

# Migrate portfolio state tables (Phase 7.9: cross-session persistence)
python scripts/migrate_add_portfolio_state.py

# Verify migration
sqlite3 data/portfolio_maximizer.db "SELECT DISTINCT model_type FROM time_series_forecasts;"
```

### OpenBB Multi-Provider Extraction
```bash
# Extract data via OpenBB (yfinance -> polygon -> alpha_vantage -> finnhub fallback)
python etl/openbb_extractor.py --tickers AAPL,MSFT --start 2024-01-01 --end 2024-06-01

# Skip cache (force fresh fetch)
python etl/openbb_extractor.py --tickers AAPL --start 2024-01-01 --end 2024-06-01 --no-cache

# Run OpenBB extractor tests
pytest tests/etl/test_openbb_extractor.py -v
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
   - `openbb_extractor.py`: Multi-provider via OpenBB SDK (yfinance/polygon/alpha_vantage/finnhub fallback chain)
   - `data_source_manager.py`: Multi-source coordination with failover
   - `synthetic_extractor.py`: Synthetic data generation for testing

2. **Storage Layer** (`etl/data_storage.py`): Parquet-based storage with train/val/test splits

3. **Validation Layer** (`etl/data_validator.py`): Statistical validation and outlier detection

4. **Preprocessing Layer** (`etl/preprocessor.py`): Missing data handling and normalization

5. **Analysis Layer** (`etl/time_series_analyzer.py`): MIT-standard time series analysis

6. **Forecasting Layer** (`forcester_ts/`, `models/`): Multiple forecasting models
   - **SARIMAX**: Seasonal ARIMA with exogenous variables (disabled by default; re-enable via config or `--enable-sarimax`)
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
- 731 tests across ETL, LLM, forecaster, integration, and security modules
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
- Regime detection integrated with feature flag (Phase 7.5+)

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
- **SARIMAX off by default**: 15x single-forecast speedup (0.18s vs 2.74s); re-enable with `sarimax: enabled: true` in config

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

## Phase 7.9 Specific Guidance (Cross-Session Persistence & Proof Mode)

Phase 7.9 adds cross-session position persistence, proof-mode validation, UTC-aware timestamps, and pandas frequency compatibility.

**Status**: In progress (20 closed trades validated, PnL integrity framework deployed)

### Phase 7.9 Key Features

- **Cross-session persistence**: `portfolio_state` + `portfolio_cash_state` tables persist positions/cash across auto-trader sessions via `--resume`
- **Proof mode** (`--proof-mode`): Testing harness with tight max_holding (5 daily / 6 intraday), ATR stops/targets, flatten-before-reverse to force round trips
- **UTC-aware timestamps**: `etl/timestamp_utils.py` provides `ensure_utc()`, `utc_now()`, `ensure_utc_index()` used at all system boundaries
- **Frequency compatibility**: `forcester_ts/_freq_compat.py` normalizes deprecated pandas aliases (`'H'` -> `'h'`, `'T'` -> `'min'`)
- **Audit sprint runbook**: `bash/run_20_audit_sprint.sh` runs 20 daily+intraday passes with gate enforcement

### Database Migrations

```bash
# Required for cross-session persistence (safe to run multiple times)
python scripts/migrate_add_portfolio_state.py
```

### Phase 7.9 Configuration

- `RISK_MODE=research_production` for balanced risk filtering
- `PROOF_MODE=1` in audit sprint to enable `--proof-mode` args
- `ENABLE_DATA_CACHE=0` to ensure fresh market data per run
- Forecast audit gate: `config/forecaster_monitoring.yml` (20 audits required, 25% max violation rate)
- `INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS=60` for holdout sprints (3 for live mode)
- `INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS=66,75` for resume-originated closes

### Audit Sprint Results (2026-02-14)

**Sprint completion**: 10 holdout runs (AS_OF 2026-01-22 to 2026-02-09, step 2 days) + 3 production live runs
- Forecast gate: **PASS** (21.4% violation rate, 28 effective audits)
- Preselection gate: correctly blocks ensemble (RMSE ratio > 1.0)
- All 6 violations have mssa_rl as best single model beating ensemble blend
- Ensemble tuning applied: `diversity_tolerance` lowered 0.35 -> 0.15, added mssa_rl-dominant candidates

**PnL regression diagnosis** (sprint trades, $-230.78):
- Stop-loss exits dominate: 5 stops = -$233.49 (MSFT -$95, NVDA -$58, GS -$44, GS -$34)
- Time-exit winners too small: 8 exits = +$2.71 avg
- Root cause: proof-mode `max_holding=5` clips gains before they develop
- Recommendation: widen max_holding to 8-10 daily bars for better risk/reward

**Concurrent process guard** (bash/run_20_audit_sprint.sh):
- Lockfile (`data/.sprint.lock`) with PID-based stale detection
- Rogue `run_auto_trader.py` process detection and kill
- Sprint lockfile warning in `scripts/run_auto_trader.py`

**Adversarial test isolation** (scripts/adversarial_integrity_test.py):
- `_IsolatedConnection` wrapper: intercepts commit/BEGIN, always rolls back
- Context manager support for automatic cleanup
- Verified: 9/10 attacks blocked, 0 artifacts persist in production DB

### Phase 7.9 Documentation

- [EXIT_ELIGIBILITY_AND_PROOF_MODE.md](Documentation/EXIT_ELIGIBILITY_AND_PROOF_MODE.md): Exit diagnosis + proof-mode spec

---

## PnL Integrity Enforcement Framework (Phase 7.9+)

**Status**: ✅ DEPLOYED (2026-02-11)

The PnL Integrity Enforcement Framework provides structural prevention of double-counting, orphaned positions, diagnostic contamination, and artificial trade legs through database-level constraints and canonical metric views.

### Core Invariants

1. **Opening legs** (is_close=0) MUST have realized_pnl IS NULL. Only closing legs carry PnL.
2. **Closing legs** (is_close=1) MUST link to their opening leg via entry_trade_id (round-trip audit trail).
3. **Diagnostic trades** (is_diagnostic=1) MUST be excluded from production metrics. Cannot appear in execution_mode='live'.
4. **Synthetic trades** (is_synthetic=1) MUST be excluded from production metrics. Cannot appear in execution_mode='live'.

### New Database Columns (trade_executions)

```sql
is_diagnostic INTEGER DEFAULT 0           -- trade executed under DIAGNOSTIC_MODE
is_synthetic INTEGER DEFAULT 0            -- trade from synthetic data source
confidence_calibrated REAL                -- calibrated confidence (future use)
entry_trade_id INTEGER                    -- links closing leg to opening leg
bar_open REAL                             -- OHLC of bar used for fill price
bar_high REAL
bar_low REAL
bar_close REAL
```

### Canonical Views

**production_closed_trades**: Single source of truth for PnL reporting
```sql
SELECT * FROM trade_executions
WHERE is_close = 1
  AND COALESCE(is_diagnostic, 0) = 0
  AND COALESCE(is_synthetic, 0) = 0
```

**round_trips**: Closing legs joined to opening legs via entry_trade_id
```sql
SELECT c.id AS close_id, o.id AS open_id, c.ticker,
       o.trade_date AS entry_date, c.trade_date AS exit_date,
       o.price AS entry_price, c.exit_price,
       c.realized_pnl, c.holding_period_days, c.exit_reason
FROM trade_executions c
LEFT JOIN trade_executions o ON c.entry_trade_id = o.id
WHERE c.is_close = 1
```

### Key Module: `integrity/pnl_integrity_enforcer.py`

**PnLIntegrityEnforcer class** provides:
- `get_canonical_metrics()` -- single source of truth for PnL reporting (uses production_closed_trades view)
- `run_full_integrity_audit()` -- 6 integrity checks with severity levels
- `fix_opening_legs_pnl(dry_run=True)` -- NULL out realized_pnl on opening legs
- `backfill_entry_trade_ids(dry_run=True)` -- link closing legs to opening legs
- `backfill_diagnostic_flag(dry_run=True)` -- flag diagnostic-mode trades
- `print_report()` -- comprehensive integrity report

**Usage**:
```python
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer

with PnLIntegrityEnforcer('data/portfolio_maximizer.db') as enforcer:
    metrics = enforcer.get_canonical_metrics()
    print(f"Total PnL: ${metrics.total_realized_pnl:+,.2f}")
    print(f"Win rate: {metrics.win_rate:.1%}")

    violations = enforcer.run_full_integrity_audit()
    if violations:
        print(f"Found {len(violations)} violations")
```

**CLI**:
```bash
# Report only
python -m integrity.pnl_integrity_enforcer --db data/portfolio_maximizer.db

# Fix double-counting (dry run)
python -m integrity.pnl_integrity_enforcer --fix-opening-pnl

# Apply all fixes
python -m integrity.pnl_integrity_enforcer --fix-all --apply
```

### Database Migration

```bash
# Add integrity columns to existing databases (safe to run multiple times)
python scripts/migrate_add_integrity_columns.py

# Outputs:
#   - Added 8 new columns to trade_executions
#   - Created production_closed_trades view
#   - Created round_trips view
```

### Integration with PaperTradingEngine

**Automatic tagging** (execution/paper_trading_engine.py lines 574-580, 1255-1264):
- `is_diagnostic` set from EXECUTION_DIAGNOSTIC_MODE env var
- `is_synthetic` set from data_source field (if contains "synthetic")
- Opening legs automatically get realized_pnl=NULL (enforcement guard)
- Closing legs populated with realized_pnl, entry_trade_id

**Trade dataclass** (lines 94-98):
```python
@dataclass
class Trade:
    # ... existing fields ...
    is_diagnostic: int = 0
    is_synthetic: int = 0
```

### CI Gate

**scripts/ci_integrity_gate.py** -- fails CI if CRITICAL/HIGH violations found

```bash
# Run integrity checks as CI gate
python scripts/ci_integrity_gate.py

# Exit codes:
#   0 = all checks passed
#   1 = CRITICAL or HIGH violations found
#   2 = database error

# Strict mode (also fails on MEDIUM)
python scripts/ci_integrity_gate.py --strict
```

**Integration with bash/run_20_audit_sprint.sh**:
```bash
# After audit sprint completes, run integrity gate
python scripts/ci_integrity_gate.py || {
    echo "[ERROR] PnL integrity violations detected"
    exit 1
}
```

### Integrity Checks (6 checks, 4 severity levels)

1. **OPENING_LEG_HAS_PNL** (CRITICAL): Opening legs must not carry realized_pnl
2. **ORPHANED_POSITION** (HIGH): BUY rows with no SELL close and no entry_trade_id linkage
3. **DIAGNOSTIC_NOT_FLAGGED** (HIGH): execution_mode contains 'diagnostic' but is_diagnostic=0
4. **DUPLICATE_CLOSE_FOR_ENTRY** (HIGH): Opening leg closed multiple times (PnL duplication)
5. **CLOSE_WITHOUT_ENTRY_LINK** (MEDIUM): Closing leg has no entry_trade_id
6. **PNL_ARITHMETIC_MISMATCH** (MEDIUM): realized_pnl doesn't match (exit - entry) * size - commission

### Corrected Baseline Metrics (Post-Fix)

**Before enforcement** (with double-counting):
- 44 "closed trades" (24 BUY + 20 SELL)
- $1,345.87 total PnL (inflated by backfilled BUY PnL)

**After enforcement** (production_closed_trades only):
- 20 round-trips (is_close=1 rows only)
- $909.18 total PnL
- 60% win rate (12W/8L)
- Profit factor: 2.78
- Largest win: $497.83
- Avg holding days: 0.0 (intraday)

**Current metrics (Post-Sprint, 2026-02-14)**:
- 37 round-trips
- $673.22 total PnL
- 43.2% win rate (16W/21L)
- Profit factor: 1.85
- Integrity: ALL PASSED (0 violations with whitelist)
- Forecast gate: PASS (21.4% violation rate, threshold 25%, 28 effective audits)

### Dashboard Integration

**Always use `PnLIntegrityEnforcer.get_canonical_metrics()`** for PnL reporting:
- DO NOT query trade_executions directly for metrics
- DO NOT use is_close=0 rows in PnL calculations
- DO NOT mix diagnostic and production trades

**Example**:
```python
# WRONG
total_pnl = db.execute(
    "SELECT SUM(realized_pnl) FROM trade_executions WHERE realized_pnl IS NOT NULL"
).fetchone()[0]

# CORRECT
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
with PnLIntegrityEnforcer('data/portfolio_maximizer.db') as enforcer:
    metrics = enforcer.get_canonical_metrics()
    total_pnl = metrics.total_realized_pnl
```

### Files Added/Modified

**New files**:
- `integrity/__init__.py` -- module init
- `integrity/pnl_integrity_enforcer.py` -- 600+ lines, enforcer class + CLI
- `scripts/migrate_add_integrity_columns.py` -- migration script
- `scripts/ci_integrity_gate.py` -- CI integrity gate

**Modified files**:
- `etl/database_manager.py` -- added 8 integrity columns to schema + save_trade_execution signature
- `execution/paper_trading_engine.py` -- Trade dataclass + is_diagnostic/is_synthetic tagging + enforcement guard

### Testing Strategy

**Zero regression requirement**:
- All 731 existing tests must pass
- New tests validate integrity enforcement
- CI gate runs on all future commits

**Manual validation**:
```bash
# Run enforcer on existing database
python -m integrity.pnl_integrity_enforcer

# Verify corrected metrics match expected baseline
# Expected: 20 round-trips, $909.18 PnL, 60% WR
```

---

## OpenClaw Integration & Cron Automation (Phase 7.9+)

**Status**: DEPLOYED (2026-02-17)

OpenClaw provides autonomous monitoring via 9 audit-aligned cron jobs running real PMX scripts. The local LLM (qwen3:8b) executes commands via `exec` tool in `agentTurn` mode and only announces anomalies.

### Cron Jobs

| Job | Schedule | Script | Announce When |
|-----|----------|--------|---------------|
| [P0] PnL Integrity Audit | Every 4h | `python -m integrity.pnl_integrity_enforcer` | CRITICAL/HIGH violations |
| [P0] Production Gate Check | Daily 7 AM | `python scripts/production_audit_gate.py` | Gate FAIL or RED |
| [P0] Quant Validation Health | Daily 7:30 AM | Inline Python (quant_validation.jsonl) | FAIL rate >= 90% |
| [P1] Signal Linkage Monitor | Daily 8 AM | Inline Python (DB query) | Orphan opens/unlinked closes |
| [P1] Ticker Health Monitor | Daily 8:30 AM | Inline Python (DB query) | 3+ consecutive losses or PnL < -$300 |
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | Inline Python (forecast audits) | Unit-root rate >= 35% |
| [P2] Overnight Hold Monitor | Weekly Fri 9 AM | Inline Python (DB query) | Overnight drag > 25% |
| System Health Check | Every 6h | `llm_multi_model_orchestrator.py status` | Model offline or errors |
| Weekly Session Cleanup | Sunday 3 AM | Session file cleanup | Never (silent) |

### 3-Model Local LLM Strategy

| Model | Role | Use Case |
|-------|------|----------|
| deepseek-r1:8b | Fast reasoning | Market analysis, signal generation, regime detection |
| deepseek-r1:32b | Heavy reasoning | Portfolio optimization, adversarial audits |
| qwen3:8b | Tool orchestrator | Function-calling, API orchestration, social media |

### Key Commands

```bash
# Check OpenClaw cron status
openclaw cron list

# Force-run a cron job
openclaw cron run <job-id> --timeout 120000

# Check model status
python scripts/openclaw_models.py status --list-ollama-models

# Apply model config
python scripts/openclaw_models.py apply --strategy local-first --restart-gateway

# Check LLM health
python scripts/llm_multi_model_orchestrator.py status
```

### Configuration Files

- `config/llm_config.yml` -- LLM model selection (3-model strategy)
- `~/.openclaw/cron/jobs.json` -- Cron job definitions (not in repo)
- `AGENTS.md` -- Agent guardrails + cron notification rules
- `Documentation/OPENCLAW_INTEGRATION.md` -- Full integration docs

---

## Interactions API (Phase 7.9+)

**Status**: DEPLOYED (2026-02-17)

The Interactions API (`scripts/pmx_interactions_api.py`) provides a FastAPI HTTP surface for local testing and external integrations (e.g., via ngrok).

### Auth Modes

Controlled by `INTERACTIONS_AUTH_MODE` env var:

| Mode | Accepts API Key | Accepts JWT | Use Case |
|------|-----------------|-------------|----------|
| `any` (default) | Yes | Yes | Development |
| `jwt-only` | No | Yes | Production with Auth0 |
| `api-key-only` | Yes | No | Simple deployments |

### Key Environment Variables

```bash
INTERACTIONS_API_KEY=...             # Required for API key auth
INTERACTIONS_AUTH_MODE=any           # any | jwt-only | api-key-only
INTERACTIONS_MIN_KEY_LENGTH=16       # Minimum API key length (floor 16)
INTERACTIONS_BIND_HOST=127.0.0.1     # Bind address
INTERACTIONS_PORT=8000               # Listen port
INTERACTIONS_RATE_LIMIT_PER_MINUTE=60
INTERACTIONS_CORS_ORIGINS=...        # Comma-separated origins (omit to disable)
AUTH0_DOMAIN=...                     # Required for JWT auth
AUTH0_AUDIENCE=...                   # Required for JWT auth
```

### Launch

```bash
# Direct
python scripts/pmx_interactions_api.py

# With ngrok tunnel (PowerShell)
./scripts/start_ngrok_interactions.ps1
```

---

## Adversarial Audit Findings (2026-02-16)

**Status**: 10 findings documented in `Documentation/ADVERSARIAL_AUDIT_20260216.md`

Key findings for agent awareness:
- 94.2% quant FAIL rate (0.8% from RED gate) -- P0
- Ensemble worse than best single model 92% of the time -- P0
- Directional accuracy below coin-flip for all models (41% WR) -- P0
- Confidence calibration broken: 0.9+ confidence yields 41% win rate -- P1
- signal_id NULL for all trades (no model attribution) -- P2
- System survives on magnitude asymmetry (avg win $91.59 vs avg loss $34.54 = 2.65x)

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

## SARIMAX-Off Default (Phase 7.9 -- Fast-Only Inference)

SARIMAX is disabled by default (mirrors LLM's off-by-default pattern). Only fast forecasters (GARCH, SAMoSSA, MSSA-RL) run unless SARIMAX is explicitly re-enabled.

**Benchmark Results** (2026-02-09, 731 tests, Windows):

| Metric | SARIMAX Off | SARIMAX On | Delta |
|--------|------------|-----------|-------|
| Single forecast (AAPL, 60-day) | 0.18 s | 2.74 s | **15x speedup** |
| Full test suite | 456.53 s (7:36) | 464.34 s (7:44) | ~8 s / 1.7% |

The modest project-level delta reflects that most tests explicitly set their own `sarimax_enabled` flag. The 15x single-forecast speedup is the meaningful production metric.

**How to re-enable SARIMAX**:
- Config: set `forecasting.sarimax.enabled: true` in `config/forecasting_config.yml`
- Dataclass: `TimeSeriesForecasterConfig(sarimax_enabled=True)`
- Tests that require SARIMAX already pass `sarimax_enabled=True` explicitly

**Files changed** (commit `1c538de`):
- `forcester_ts/forecaster.py`: Dataclass default + `_build_config_from_kwargs` fallback
- `config/forecasting_config.yml`: SARIMAX disabled, ensemble weights updated
- `config/pipeline_config.yml`: Mirror of forecasting config
- `scripts/run_etl_pipeline.py`: Fallback default aligned
- `tests/etl/test_time_series_forecaster_slow.py`: Explicit `sarimax_enabled=True`

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
- `AGENTS.md` - Agent guardrails + cron notification rules
- `README.md` - User-facing project overview
- `QUICK_REFERENCE.md` - Quick command reference card
- `requirements.txt` - Python dependencies
- `config/pipeline_config.yml` - Main configuration
- `config/llm_config.yml` - LLM model selection
- `.env.template` - Environment variable template

### Key Directories
- `etl/` - Data extraction, transformation, loading
- `forcester_ts/` - Time series forecasting models
- `models/` - Signal generation and routing
- `execution/` - Order management and paper trading
- `integrity/` - PnL integrity enforcement (Phase 7.9)
- `ai_llm/` - LLM integration (Ollama client, market analyzer)
- `tests/` - Test suite (731 tests)
- `scripts/` - Utility scripts, migrations, APIs
- `tools/` - Development tools (secrets guard, git askpass)
- `config/` - YAML configuration files
- `Documentation/` - Phase-specific documentation (174 files)
- `logs/` - Pipeline and application logs

---

**Remember**: Always activate virtual environment, check platform compatibility, and update documentation when making changes!

**Last Updated**: 2026-02-17 (Phase 7.9 Complete: PnL integrity enforcement, adversarial audit, OpenClaw cron automation, Interactions API, 3-model LLM strategy, secrets leak guard)
**GitHub**: https://github.com/mrbestnaija/portofolio_maximizer.git
