# Apache Airflow Integration TODO - Portfolio Maximizer

**Created**: 2026-01-08  
**Status**: Planning Phase  
**Scope**: Migrate from bash/cron orchestration to Apache Airflow following industrial standards

## Overview

This document outlines the comprehensive plan for integrating Apache Airflow into the Portfolio Maximizer project to replace the current bash script and cron-based orchestration system. The migration will follow industry best practices for workflow orchestration, monitoring, and scalability.

## Current State Analysis

**Last Updated**: 2026-01-08  
**Project Status**: Engineering unblocked, 529 tests passing (see `Documentation/PROJECT_STATUS.md`)  
**Architecture**: Time Series PRIMARY, LLM FALLBACK (see `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`)

### Existing Orchestration Components

1. **Bash Scripts** (from `bash/production_cron.sh`):
   - `bash/production_cron.sh` - Main cron multiplexer (11+ tasks, routes to Python entrypoints)
   - `bash/run_end_to_end.sh` - End-to-end pipeline execution
   - `bash/run_pipeline_live.sh` - Live data pipeline with synthetic fallback
   - `bash/run_pipeline_dry_run.sh` - Synthetic/no-network pipeline exerciser
   - `bash/comprehensive_brutal_test.sh` - Full test suite orchestration
   - `bash/run_cv_validation.sh` - Cross-validation validation suite
   - `bash/run_post_eval.sh` - Higher-order hyperparameter optimization driver
   - `bash/auto_rebuild_and_sweep.sh` - Trade history rebuild and sweep automation
   - `bash/run_gpu_parallel.sh` - GPU-parallel execution runner
   - Multiple supporting scripts for testing and validation

2. **Cron Jobs** (from `bash/production_cron.sh` and `Documentation/CRON_AUTOMATION.md`):
   - `daily_etl` - Pre-market ETL refresh (5:15 AM weekdays) → `scripts/run_etl_pipeline.py`
   - `auto_trader` - High-frequency trading (every 30 min, 7-20 weekdays) → `scripts/run_auto_trader.py`
   - `auto_trader_core` - Core ticker accumulation (hourly, 7-20 weekdays) → `scripts/run_auto_trader.py --core-mode`
   - `monitoring` - Hourly health checks → `scripts/monitor_llm_system.py`
   - `env_sanity` - Pre-market validation (5:00 AM weekdays) → `scripts/validate_environment.py`
   - `nightly_backfill` - Signal validation (2:05 AM daily) → `scripts/backfill_signal_validation.py`
   - `ts_threshold_sweep` - Weekly threshold optimization (4:00 AM Mondays) → `scripts/sweep_ts_thresholds.py`
   - `transaction_costs` - Monthly cost estimation (4:15 AM 1st of month) → `scripts/estimate_transaction_costs.py`
   - `weekly_sleeve_maintenance` - Weekly sleeve management (5:00 AM Mondays) → `bash/weekly_sleeve_maintenance.sh`
   - `synthetic_refresh` - Weekly synthetic data (1:00 AM Mondays) → `scripts/generate_synthetic_dataset.py`
   - `sanitize_caches` - Cache/log cleanup (daily) → `scripts/sanitize_cache_and_logs.py`

3. **Python Entry Points** (Production-Ready):
   - `scripts/run_etl_pipeline.py` - Main ETL orchestration (1,900+ lines, config-driven, TS-first)
   - `scripts/run_auto_trader.py` - Autonomous trading engine (bar-aware, GPU-parallel support)
   - `scripts/monitor_llm_system.py` - LLM monitoring (418 lines, latency/backtest reporting)
   - `scripts/validate_environment.py` - Environment checks (pre-market validation)
   - `scripts/backfill_signal_validation.py` - Signal backfill (nightly validation)
   - `scripts/sweep_ts_thresholds.py` - TS threshold optimization (weekly)
   - `scripts/estimate_transaction_costs.py` - Transaction cost estimation (monthly)
   - `scripts/generate_synthetic_dataset.py` - Synthetic data generation (weekly)
   - `scripts/run_strategy_optimization.py` - Strategy optimization (stochastic, regime-aware)
   - `scripts/run_ts_model_search.py` - TS model candidate search
   - `scripts/summarize_quant_validation.py` - Quant validation summarization
   - `scripts/check_quant_validation_health.py` - Global quant health classification
   - Multiple analysis and optimization scripts

4. **Key Architecture Notes**:
   - **Signal Routing**: Time Series is PRIMARY, LLM is FALLBACK (see `models/signal_router.py`)
   - **Forecasting**: SARIMAX, GARCH, SAMOSSA, MSSA-RL ensemble (see `forcester_ts/`)
   - **Data Sources**: Multi-source with failover (yfinance, Alpha Vantage, Finnhub)
   - **GPU Support**: CUDA auto-detection with CPU fallback (`PIPELINE_DEVICE`)
   - **Checkpointing**: 7-day retention policy (`etl/checkpoint_manager.py`)
   - **Logging**: Structured JSON events (`etl/pipeline_logger.py`)
   - **Database**: SQLite with recovery mechanisms (`etl/database_manager.py`)

## Migration Strategy

### Principles

1. **Incremental Migration**: Migrate tasks incrementally, maintaining backward compatibility
2. **Zero Downtime**: Run Airflow and cron in parallel during transition
3. **Feature Parity**: All existing functionality must be preserved
4. **Industrial Standards**: Follow Airflow best practices (DAG design, error handling, monitoring)
5. **Configuration Management**: Centralize configs using Airflow Variables and Connections
6. **Observability**: Enhanced logging, monitoring, and alerting

### Existing Infrastructure to Leverage

The project already has production-ready infrastructure that Airflow can leverage:

1. **Checkpointing System** (`etl/checkpoint_manager.py`):
   - Atomic checkpoint operations with SHA256 validation
   - 7-day retention policy
   - Pipeline progress tracking
   - Can be used for Airflow task failure recovery

2. **Structured Logging** (`etl/pipeline_logger.py`):
   - JSON event logging with rotation
   - Multiple log streams (pipeline, events, errors)
   - 7-day automatic cleanup
   - Can integrate with Airflow logs

3. **Error Monitoring** (`scripts/error_monitor.py`):
   - Real-time error tracking with thresholds
   - Automated alerting (email/Slack)
   - Historical analysis
   - Can be triggered from Airflow task failures

4. **Performance Monitoring** (`monitoring/performance_dashboard.py`):
   - Real-time metrics generation
   - JSON/CSV export
   - Alert generation
   - Can be scheduled via Airflow

5. **Database Recovery** (`etl/database_manager.py`):
   - Automatic SQLite corruption recovery
   - Backup and restore mechanisms
   - Can handle Airflow-triggered database operations

6. **Multi-Source Data Extraction** (`etl/data_source_manager.py`):
   - Automatic failover between data sources
   - 99.99% reliability with 3 sources
   - Cache-first strategy (100% hit rate)
   - No changes needed for Airflow integration

7. **GPU Resource Management**:
   - Auto-detection via `PIPELINE_DEVICE` environment variable
   - GPU-parallel processing support
   - Can be managed via Airflow resource pools

### Phases

#### Phase 1: Infrastructure Setup (Foundation)
- Airflow installation and configuration
- Directory structure and project organization
- Connection and variable setup
- Custom operator development

#### Phase 2: Core Task Migration (Critical Path)
- Daily ETL pipeline
- Auto trader loop
- Monitoring and health checks
- Environment validation

#### Phase 3: Orchestration & Dependencies (Integration)
- Master orchestration DAG
- Task dependencies and scheduling
- Market hours sensors
- Data quality checks

#### Phase 4: Extended Tasks (Completeness)
- Weekly maintenance tasks
- Monthly reporting
- Synthetic data generation

#### Phase 5: Advanced Features (Optimization)
- Dynamic task mapping
- XCom data passing
- Resource pools and concurrency

#### Phase 6: Configuration Management (Standardization)
- Airflow Variables migration
- Secrets management integration
- Config-driven DAGs

#### Phase 7: Observability (Monitoring)
- Logging integration
- Alerting setup
- Dashboard integration

#### Phase 8: Reliability (Production Readiness)
- Retry strategies
- SLA management
- Versioning

#### Phase 9: Platform Support (Compatibility)
- Windows/WSL support
- Documentation
- Migration tooling

#### Phase 10: Testing & Rollout (Validation)
- Testing framework
- Performance benchmarking
- Gradual rollout

#### Phase 11: Advanced Capabilities (Enhancement)
- Advanced workflow patterns
- Data lineage
- Cost optimization

## Detailed Task Breakdown

### Phase 1: Infrastructure Setup

#### Task 1.1: Airflow Installation
- Install Apache Airflow 2.7+ (or latest stable)
- Choose executor: LocalExecutor (dev) or CeleryExecutor (production)
- Set up PostgreSQL as metadata database (replace SQLite)
- Configure Redis for Celery broker (if using CeleryExecutor)
- Create Docker Compose setup for local development

**Deliverables**:
- `docker-compose.yml` for Airflow services
- `requirements-airflow.txt` with Airflow dependencies
- Installation documentation

#### Task 1.2: Directory Structure
```
airflow/
├── dags/
│   ├── __init__.py
│   ├── etl/
│   ├── trading/
│   ├── monitoring/
│   └── maintenance/
├── plugins/
│   ├── __init__.py
│   ├── operators/
│   ├── sensors/
│   └── hooks/
├── config/
│   └── airflow.cfg (custom settings)
├── logs/
└── tests/
```

**Deliverables**:
- Complete directory structure
- `__init__.py` files
- `.gitignore` updates

#### Task 1.3: Airflow Connections
- Database connection (SQLite/PostgreSQL)
- Alpha Vantage API connection
- Finnhub API connection
- Custom connections for external services

**Deliverables**:
- Connection configuration file
- Documentation of connection requirements

#### Task 1.4: Custom Operators
Create reusable operators that wrap existing Python entrypoints:
- `PortfolioETLOperator` - Wraps `scripts/run_etl_pipeline.py` (TS-first, multi-source, GPU support)
- `AutoTraderOperator` - Wraps `scripts/run_auto_trader.py` (bar-aware, GPU-parallel, run-local metrics)
- `MonitoringOperator` - Wraps `scripts/monitor_llm_system.py` (latency, backtest, quant health)
- `BackfillOperator` - Wraps `scripts/backfill_signal_validation.py` (nightly signal validation)
- `SyntheticDataOperator` - Wraps `scripts/generate_synthetic_dataset.py` (weekly synthetic refresh)
- `TSThresholdSweepOperator` - Wraps `scripts/sweep_ts_thresholds.py` (weekly threshold optimization)
- `TransactionCostOperator` - Wraps `scripts/estimate_transaction_costs.py` (monthly cost estimation)
- `StrategyOptimizationOperator` - Wraps `scripts/run_strategy_optimization.py` (stochastic optimization)
- `QuantValidationOperator` - Wraps `scripts/check_quant_validation_health.py` (quant health checks)

**Operator Requirements**:
- Support existing environment variables (`CRON_*`, `PIPELINE_DEVICE`, `ENABLE_GPU_PARALLEL`)
- Integrate with existing logging (`logs/cron/`, `logs/pipeline.log`)
- Leverage existing checkpointing for failure recovery
- Support GPU resource allocation via Airflow resource pools
- Preserve existing error handling and retry logic

**Deliverables**:
- `airflow/plugins/operators/portfolio_operators.py`
- Unit tests for operators (integrate with existing test patterns)
- Documentation (reference existing script documentation)

### Phase 2: Core Task Migration

#### Task 2.1: Daily ETL DAG
**Current**: `bash/production_cron.sh daily_etl` → `scripts/run_etl_pipeline.py`  
**Target**: `airflow/dags/etl/daily_etl_dag.py`

**Current Implementation Notes**:
- Pipeline is TS-first (Time Series forecasting before LLM stages)
- Supports multi-source data extraction (yfinance, Alpha Vantage, Finnhub)
- GPU auto-detection via `PIPELINE_DEVICE` environment variable
- Checkpointing and structured logging already implemented
- Config-driven via `config/pipeline_config.yml`

**Requirements**:
- Schedule: `15 5 * * 1-5` (5:15 AM weekdays, matching current cron)
- Task: Run `scripts/run_etl_pipeline.py` with configurable parameters
- Environment variables: `CRON_TICKERS`, `CRON_START_DATE`, `CRON_END_DATE`, `CRON_EXEC_MODE`
- GPU support: Pass `PIPELINE_DEVICE` (cuda/cpu) from Airflow Variables
- Logging: Integrate with existing `logs/cron/` and `logs/pipeline.log` structure
- Error handling: Retry 3 times with exponential backoff
- Notifications: Email/Slack on failure
- Checkpoint recovery: Leverage existing `etl/checkpoint_manager.py` for failure recovery

**Implementation**:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.plugins.operators.portfolio_operators import PortfolioETLOperator

default_args = {
    'owner': 'portfolio-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_etl',
    default_args=default_args,
    description='Daily ETL pipeline refresh (TS-first architecture)',
    schedule_interval='15 5 * * 1-5',  # 5:15 AM weekdays (matches cron)
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'daily', 'production', 'ts-first'],
)

etl_task = PortfolioETLOperator(
    task_id='run_daily_etl',
    dag=dag,
    tickers="{{ var.value.get('CRON_TICKERS', 'AAPL,MSFT,GOOGL') }}",
    start_date="{{ var.value.get('CRON_START_DATE', '2020-01-01') }}",
    end_date="{{ ds }}",  # Airflow execution date
    execution_mode="{{ var.value.get('CRON_EXEC_MODE', 'live') }}",
    python_callable_path='scripts/run_etl_pipeline.py',
    env_vars={
        'PIPELINE_DEVICE': "{{ var.value.get('PIPELINE_DEVICE', 'cpu') }}",
        'ENABLE_GPU_PARALLEL': "{{ var.value.get('ENABLE_GPU_PARALLEL', '0') }}",
    },
)
```

#### Task 2.2: Auto Trader DAG
**Current**: `bash/production_cron.sh auto_trader` → `scripts/run_auto_trader.py` (every 30 min during market hours)  
**Target**: `airflow/dags/trading/auto_trader_dag.py`

**Current Implementation Notes**:
- Bar-aware trading loop (skips repeated cycles on same bar)
- GPU-parallel support for candidate prep + forecasts (`ENABLE_GPU_PARALLEL=1`)
- Time Series signals are PRIMARY, LLM is FALLBACK
- Supports `--core-mode` for `auto_trader_core` (hourly, 7-20 weekdays)
- Run-local reporting with `run_id` scoping
- Forecast persistence with lagged regression backfill

**Requirements**:
- Schedule: `*/30 7-20 * * 1-5` (every 30 min, 7 AM - 8 PM weekdays)
- Market hours sensor: Only run during trading hours (NYSE, NASDAQ)
- Dynamic ticker processing: Use Airflow dynamic task mapping for parallel ticker execution
- Trade count gating: Check database before execution (like `auto_trader_core`)
- Resource pool: Limit concurrent trading tasks
- GPU support: Pass `PIPELINE_DEVICE` and `ENABLE_GPU_PARALLEL` from Airflow Variables
- Bar state persistence: Leverage existing bar-aware logic in `scripts/run_auto_trader.py`

**Implementation Notes**:
- Use `MarketHoursSensor` (custom sensor) for NYSE/NASDAQ market hours
- Dynamic task mapping for parallel ticker processing (GPU-first when available)
- Task pool: `trading_pool` with concurrency limit
- Integrate with existing `run_id` tracking for run-local metrics
- Support both `auto_trader` (30-min) and `auto_trader_core` (hourly) schedules

#### Task 2.3: Monitoring DAG
**Current**: `bash/production_cron.sh monitoring` → `scripts/monitor_llm_system.py` (hourly)  
**Target**: `airflow/dags/monitoring/health_monitoring_dag.py`

**Current Implementation Notes**:
- `scripts/monitor_llm_system.py` (418 lines) provides comprehensive LLM monitoring
- Latency benchmarks logged to `logs/latency_benchmark.json`
- LLM signal backtest summaries and JSON reports
- Quant validation health checks via `scripts/check_quant_validation_health.py`
- Performance dashboard snapshot generation (`monitoring/performance_dashboard.py`)

**Requirements**:
- Schedule: `5 * * * *` (5 minutes past every hour, matching current cron)
- Multiple monitoring tasks:
  - LLM system monitoring (latency, token rate, model health)
  - Pipeline health checks (checkpoint status, log rotation)
  - Database connectivity (SQLite integrity checks)
  - API availability (data source health)
  - Quant validation health (GREEN/YELLOW/RED classification)
  - Performance metrics (equity curve, win rate, profit factor)
- Alerting: Slack/email on degradation (integrate with existing `scripts/error_monitor.py`)
- Dashboard updates: Trigger `monitoring/performance_dashboard.py` snapshot generation

#### Task 2.4: Environment Sanity DAG
**Current**: `bash/production_cron.sh env_sanity` (5:00 AM weekdays)  
**Target**: `airflow/dags/monitoring/env_sanity_dag.py`

**Requirements**:
- Schedule: `0 5 * * 1-5`
- Validation checks:
  - Python version
  - Virtual environment
  - Configuration files
  - Database connectivity
  - Secret loading
- Branching: Skip trading tasks if validation fails

### Phase 3: Orchestration & Dependencies

#### Task 3.1: Master Orchestration DAG
Create a master DAG that coordinates:
1. `env_sanity` → validates environment
2. `daily_etl` → refreshes data (if env_sanity passes)
3. `auto_trader` → runs trading loop (depends on daily_etl)
4. `monitoring` → continuous monitoring

**Dependencies**:
```
env_sanity → daily_etl → auto_trader
                ↓
           monitoring (parallel)
```

#### Task 3.2: Market Hours Sensor
Create custom sensor:
```python
from airflow.sensors.base import BaseSensorOperator

class MarketHoursSensor(BaseSensorOperator):
    """
    Sensor that waits until market is open.
    Supports NYSE, NASDAQ with timezone awareness.
    """
    def poke(self, context):
        # Check if current time is within market hours
        # Account for holidays, early closes, etc.
        pass
```

#### Task 3.3: Data Quality Checks
Integrate data quality validation:
- Use Great Expectations or custom validators
- Check OHLCV data completeness
- Validate data ranges and distributions
- Generate data quality reports

### Phase 4: Extended Tasks

#### Task 4.1: Weekly Maintenance DAG
Migrate weekly tasks from `bash/production_cron.sh`:
- `weekly_sleeve_maintenance` - Weekly sleeve management (5:00 AM Mondays) → `bash/weekly_sleeve_maintenance.sh`
- `ts_threshold_sweep` - Weekly threshold optimization (4:00 AM Mondays) → `scripts/sweep_ts_thresholds.py`
- `synthetic_refresh` - Weekly synthetic data (1:00 AM Mondays) → `scripts/generate_synthetic_dataset.py`

**Current Implementation Notes**:
- `scripts/sweep_ts_thresholds.py` sweeps `(confidence_threshold, min_expected_return)` and writes to `logs/automation/ts_threshold_sweep.json`
- `scripts/generate_synthetic_dataset.py` generates synthetic datasets with profiles, copula shocks, and microstructure
- Synthetic datasets use `SYNTHETIC_DATASET_ID=latest` pointer system

**Schedule**: Weekly (Mondays at various times, matching current cron)

#### Task 4.2: Monthly Tasks DAG
Migrate monthly tasks:
- `transaction_costs` - Monthly cost estimation (4:15 AM 1st of month) → `scripts/estimate_transaction_costs.py`

**Current Implementation Notes**:
- `scripts/estimate_transaction_costs.py` estimates commission/transaction costs by ticker/asset class
- Writes results to `logs/automation/transaction_costs.json`
- Used by `scripts/generate_config_proposals.py` for cost-aware threshold adjustments

### Phase 5: Advanced Features

#### Task 5.1: Dynamic Task Mapping
Use Airflow 2.5+ dynamic task mapping for:
- Parallel ticker processing in ETL
- Parallel ticker processing in auto_trader
- Configurable parallelism based on resources

#### Task 5.2: XCom for Data Passing
- Pass ticker lists between tasks
- Share execution dates
- Propagate execution results
- Store intermediate data

#### Task 5.3: Resource Pools
Configure pools:
- `gpu_pool` - GPU-intensive tasks (limited to available GPUs)
- `database_pool` - Database operations (prevent connection exhaustion)
- `api_pool` - External API calls (rate limiting)

### Phase 6: Configuration Management

#### Task 6.1: Airflow Variables
Migrate environment variables to Airflow Variables:
- `CRON_TICKERS` → `portfolio.default_tickers`
- `CRON_START_DATE` → `portfolio.default_start_date`
- `CRON_END_DATE` → `portfolio.default_end_date`
- `CRON_EXEC_MODE` → `portfolio.execution_mode` (live/synthetic)
- `PIPELINE_DEVICE` → `portfolio.pipeline_device` (cuda/cpu)
- `ENABLE_GPU_PARALLEL` → `portfolio.enable_gpu_parallel` (0/1)
- `ENABLE_PARALLEL_TICKER_PROCESSING` → `portfolio.enable_parallel_tickers` (0/1)
- `ENABLE_PARALLEL_FORECASTS` → `portfolio.enable_parallel_forecasts` (0/1)
- `SYNTHETIC_DATASET_ID` → `portfolio.synthetic_dataset_id` (for synthetic mode)
- `PORTFOLIO_DB_PATH` → `portfolio.database_path` (default: `data/portfolio_maximizer.db`)

#### Task 6.2: Secrets Management
- Integrate with existing `etl/secret_loader.py` (loads from `.env` file)
- Or migrate to Airflow Secrets Backend (recommended for production)
- Support for multiple environments (dev/staging/prod)
- API keys: `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY` (already in `.env`)
- Broker credentials: `USERNAME_CTRADER`, `PASSWORD_CTRADER` (if using cTrader)

#### Task 6.3: Config-Driven DAGs
Create DAG factory that reads from existing YAML configs:
- `config/pipeline_config.yml` - Main pipeline configuration (stage ordering, CV settings)
- `config/forecasting_config.yml` - Time Series forecasting parameters (SARIMAX, GARCH, SAMOSSA, MSSA-RL)
- `config/signal_routing_config.yml` - Signal routing (TS-primary, LLM-fallback thresholds)
- `config/quant_success_config.yml` - Quant validation thresholds (Sharpe, Sortino, min_expected_profit)
- `config/forecaster_monitoring.yml` - Quant monitoring tiers (GREEN/YELLOW/RED)
- `config/data_sources_config.yml` - Multi-source configuration (yfinance, Alpha Vantage, Finnhub)
- `config/yfinance_config.yml` - Yahoo Finance specific settings
- `config/llm_config.yml` - LLM configuration (Ollama models, latency guards)
- `config/ctrader_config.yml` - Broker configuration (demo/live endpoints, risk caps)

### Phase 7: Observability

#### Task 7.1: Logging Integration
- Integrate Airflow logs with `logs/cron/` structure
- Preserve existing log formats
- Add Airflow context to logs

#### Task 7.2: Alerting
- Email notifications for failures
- Slack integration for critical alerts
- PagerDuty for production incidents
- Custom alerting rules per task type

#### Task 7.3: Dashboard Integration
- Add Airflow metrics to `monitoring/performance_dashboard.py`
- Create Airflow-specific views
- Track DAG run statistics

### Phase 8: Reliability

#### Task 8.1: Retry Strategies
- Configure retries per task type
- Exponential backoff
- Dead letter queue for persistent failures
- Manual intervention workflows

#### Task 8.2: SLA Management
Define SLAs:
- `daily_etl`: Must complete by 6:00 AM (1 hour SLA)
- `auto_trader`: Max latency 5 minutes per cycle
- `monitoring`: Must complete within 10 minutes

#### Task 8.3: Versioning
- DAG versioning strategy
- Backward compatibility
- Migration scripts for config changes

### Phase 9: Platform Support

#### Task 9.1: Windows/WSL Support
- Ensure operators work on Windows
- Document WSL deployment
- Windows Task Scheduler integration (if needed)

#### Task 9.2: Documentation
Create comprehensive documentation:
- `Documentation/AIRFLOW_MIGRATION.md` - Migration guide
- `Documentation/AIRFLOW_DAG_REFERENCE.md` - DAG catalog
- `Documentation/AIRFLOW_OPERATIONS.md` - Operational procedures

#### Task 9.3: Migration Tooling
- Scripts to validate migration
- Comparison tools (cron vs Airflow outputs)
- Rollback procedures

### Phase 10: Testing & Rollout

#### Task 10.1: Testing Framework
- Unit tests for all operators
- Integration tests for DAGs
- End-to-end tests
- Performance tests

#### Task 10.2: Performance Benchmarking
- Compare Airflow vs cron performance
- Measure overhead
- Optimize task execution
- Tune Airflow scheduler

#### Task 10.3: Gradual Rollout
1. **Phase A**: Run Airflow in parallel with cron (validation)
2. **Phase B**: Migrate non-critical tasks first
3. **Phase C**: Migrate critical tasks with monitoring
4. **Phase D**: Decommission cron jobs

### Phase 11: Advanced Capabilities

#### Task 11.1: Advanced Workflow Patterns
- Conditional task execution
- Dynamic DAG generation
- Custom trigger rules
- Task groups for organization

#### Task 11.2: Data Lineage
- Integrate with existing provenance tracking
- Track data flow through pipeline
- Visualize dependencies
- Audit trail

#### Task 11.3: Cost Optimization
- Track task execution costs
- Resource usage monitoring
- Optimization recommendations
- Cost allocation by DAG/task

## Implementation Guidelines

### Code Standards

1. **DAG Design**:
   - Idempotent tasks
   - Clear task naming
   - Proper use of task groups
   - Descriptive docstrings

2. **Error Handling**:
   - Comprehensive try/except blocks
   - Meaningful error messages
   - Proper logging
   - Retry strategies

3. **Testing**:
   - Unit tests for all operators
   - Integration tests for DAGs
   - Mock external dependencies
   - Test failure scenarios

4. **Documentation**:
   - Inline comments
   - DAG descriptions
   - Task documentation
   - Operational runbooks

### Migration Checklist

For each task migration:
- [ ] Analyze current bash/cron implementation
- [ ] Design Airflow DAG structure
- [ ] Create custom operators (if needed)
- [ ] Implement DAG with proper scheduling
- [ ] Add error handling and retries
- [ ] Configure logging and monitoring
- [ ] Write unit tests
- [ ] Test in development environment
- [ ] Validate output matches cron version
- [ ] Update documentation
- [ ] Deploy to staging
- [ ] Run in parallel with cron
- [ ] Monitor and compare
- [ ] Cutover to Airflow
- [ ] Decommission cron job

## Success Criteria

1. **Functional Parity**: All existing cron jobs successfully migrated
2. **Performance**: No degradation in task execution time
3. **Reliability**: Improved error handling and recovery
4. **Observability**: Enhanced monitoring and alerting
5. **Maintainability**: Easier to modify and extend workflows
6. **Scalability**: Support for increased workload and complexity

## Risks & Mitigation

1. **Risk**: Airflow overhead may slow down tasks
   - **Mitigation**: Benchmark early, optimize scheduler settings

2. **Risk**: Learning curve for team
   - **Mitigation**: Training sessions, comprehensive documentation

3. **Risk**: Migration complexity
   - **Mitigation**: Incremental approach, parallel running

4. **Risk**: Breaking changes in Airflow
   - **Mitigation**: Pin Airflow version, test upgrades carefully

## Timeline Estimate

**Note**: Many infrastructure components are already implemented (checkpointing, logging, monitoring, error handling), which should accelerate the migration.

- **Phase 1-2**: 2-3 weeks (Infrastructure + Core tasks)
  - Airflow setup: 1 week
  - Custom operators: 1 week
  - Core DAGs (ETL, auto_trader, monitoring): 1 week
- **Phase 3-4**: 2-3 weeks (Orchestration + Extended tasks)
  - Master DAG: 1 week
  - Market hours sensors: 3 days
  - Weekly/monthly tasks: 1 week
- **Phase 5-6**: 1-2 weeks (Advanced features + Config)
  - Dynamic task mapping: 3 days
  - Config migration: 3 days
  - Secrets management: 2 days
- **Phase 7-8**: 1-2 weeks (Observability + Reliability)
  - Logging integration: 3 days (leverage existing `pipeline_logger.py`)
  - Alerting setup: 2 days (leverage existing `error_monitor.py`)
  - Retry strategies: 2 days
- **Phase 9-10**: 2 weeks (Platform support + Testing)
  - Windows/WSL support: 1 week
  - Testing framework: 1 week
- **Phase 11**: Ongoing (Advanced capabilities)

**Total**: 8-12 weeks for complete migration (reduced from 10-13 weeks due to existing infrastructure)

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Airflow Operators Guide](https://airflow.apache.org/docs/apache-airflow/stable/concepts/operators.html)
- [Airflow DAG Writing Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html#writing-dags)

## What's Already Implemented vs What Needs Airflow

### Already Implemented (No Airflow Needed)
- ✅ **Checkpointing System**: `etl/checkpoint_manager.py` - Atomic checkpoints with SHA256 validation
- ✅ **Structured Logging**: `etl/pipeline_logger.py` - JSON event logging with rotation
- ✅ **Error Monitoring**: `scripts/error_monitor.py` - Real-time error tracking and alerting
- ✅ **Performance Dashboard**: `monitoring/performance_dashboard.py` - Metrics generation (snapshot mode)
- ✅ **Database Recovery**: `etl/database_manager.py` - Automatic SQLite corruption recovery
- ✅ **Multi-Source Data**: `etl/data_source_manager.py` - Automatic failover (99.99% reliability)
- ✅ **GPU Support**: Auto-detection and parallel processing
- ✅ **Time Series Forecasting**: Complete implementation (SARIMAX, GARCH, SAMOSSA, MSSA-RL)
- ✅ **Signal Routing**: TS-primary, LLM-fallback architecture
- ✅ **Quant Validation**: Health classification (GREEN/YELLOW/RED)
- ✅ **All Python Scripts**: Production-ready entrypoints

### What Airflow Adds
- 🔄 **Workflow Orchestration**: DAG-based task scheduling and dependencies
- 🔄 **Market Hours Sensors**: Automatic gating based on trading hours
- 🔄 **Dynamic Task Mapping**: Parallel ticker processing with resource management
- 🔄 **Centralized Configuration**: Airflow Variables and Connections
- 🔄 **Enhanced Monitoring**: Airflow UI for task status, logs, and metrics
- 🔄 **SLA Management**: Automatic alerts for missed deadlines
- 🔄 **Task Retries**: Built-in retry logic with exponential backoff
- 🔄 **Resource Pools**: GPU and API rate limiting via Airflow pools
- 🔄 **Dependency Management**: Visual DAG representation of task relationships
- 🔄 **Historical Tracking**: Complete execution history and audit trail

## Next Steps

1. Review and approve this TODO list
2. Set up Airflow development environment (Phase 1)
3. Begin with daily_etl migration (Phase 2.1) - leverage existing checkpointing/logging
4. Iterate and refine based on learnings
5. Run Airflow and cron in parallel during transition period

---

**Last Updated**: 2026-01-08  
**Owner**: Engineering Team  
**Status**: Planning  
**Project Status Reference**: See `Documentation/PROJECT_STATUS.md` for current system state
