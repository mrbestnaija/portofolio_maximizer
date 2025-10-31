#!/usr/bin/env python3
"""ETL Pipeline Orchestration Script - Modular Configuration-Driven Design with LLM Integration.

This orchestrator loads configuration from modular YAML files and executes
the ETL pipeline with proper error handling and progress tracking.

Configuration files:
- config/pipeline_config.yml: Main pipeline orchestration config
- config/yfinance_config.yml: Data extraction configuration
- config/validation_config.yml: Data validation rules
- config/preprocessing_config.yml: Preprocessing parameters
- config/storage_config.yml: Storage and splitting configuration
- config/llm_config.yml: LLM integration settings (Phase 5.2)

Pipeline Flow:
1. Data Extraction (multi-source with failover)
2. Data Validation (quality checks)
3. Data Preprocessing (normalization, missing data)
4. LLM Market Analysis (optional - if --enable-llm flag set)
5. LLM Signal Generation (optional - if --enable-llm flag set)
6. LLM Risk Assessment (optional - if --enable-llm flag set)
7. Data Storage (train/val/test split with optional CV)
"""
import sys
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import click
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Sequence, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()

from etl.data_source_manager import DataSourceManager
from etl.data_validator import DataValidator
from etl.preprocessor import Preprocessor
from etl.data_storage import DataStorage
from etl.checkpoint_manager import CheckpointManager
from etl.pipeline_logger import PipelineLogger
from etl.ticker_discovery import (
    AlphaVantageTickerLoader,
    TickerUniverseManager,
    TickerValidator,
)
from etl.portfolio_math import (
    calculate_enhanced_portfolio_metrics,
    optimize_portfolio_markowitz,
)

# LLM Integration (Phase 5.2)
from ai_llm.ollama_client import OllamaClient, OllamaConnectionError
from ai_llm.market_analyzer import LLMMarketAnalyzer
from ai_llm.signal_generator import LLMSignalGenerator
from ai_llm.risk_assessor import LLMRiskAssessor
from ai_llm.performance_optimizer import LLMPerformanceOptimizer
from ai_llm.signal_quality_validator import SignalQualityValidator, Signal, SignalDirection

# Database Integration (Phase 5.2+)
from etl.database_manager import DatabaseManager

import time

# Supported execution modes for data extraction
EXECUTION_MODES = ('auto', 'live', 'synthetic')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate pipeline configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration: {e}")
        raise


def generate_synthetic_ohlcv(tickers: List[str], start_date: str,
                             end_date: str, seed: int = 123) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for offline validation."""
    date_index = pd.date_range(start=pd.Timestamp(start_date),
                               end=pd.Timestamp(end_date), freq='B')
    frames = []
    rng = np.random.default_rng(seed)
    for ticker in tickers:
        n = len(date_index)
        base = 100.0 * (1 + rng.normal(0, 0.01))
        rets = rng.normal(0.0005, 0.01, size=n)
        prices = base * np.cumprod(1 + rets)
        close = prices
        open_ = close * (1 + rng.normal(0, 0.002, size=n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=n)))
        volume = (1_000_000 * (1 + rng.normal(0, 0.05, size=n))).astype(int)
        df_t = pd.DataFrame({
            'Open': open_,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
            'Adj Close': close,
            'ticker': ticker,
        }, index=date_index)
        df_t.index.name = 'Date'
        frames.append(df_t)
    return pd.concat(frames).sort_index()


def _extract_price_matrix(data: pd.DataFrame, price_field: str, tickers: Sequence[str]) -> pd.DataFrame:
    """Return a wide price matrix (date index, ticker columns)."""
    if data is None or data.empty:
        return pd.DataFrame()

    matrix = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        try:
            matrix = data.xs(price_field, axis=1, level=-1)
        except (KeyError, ValueError):
            matrix = pd.DataFrame()
    elif price_field in data.columns and "ticker" in data.columns:
        matrix = (
            data.pivot_table(index=data.index, columns="ticker", values=price_field, aggfunc="first")
            .sort_index()
        )
    elif price_field in data.columns:
        matrix = data[[price_field]].copy()
        column_name = tickers[0] if tickers else price_field
        matrix.columns = [column_name]
    else:
        candidates = [col for col in data.columns if str(col).lower().endswith(price_field.lower())]
        if candidates:
            matrix = data[candidates].copy()

    if not matrix.empty:
        matrix = matrix.sort_index().dropna(how="all")
    return matrix


def _slice_ticker_dataframe(data: Optional[pd.DataFrame], ticker: str) -> pd.DataFrame:
    """Return data subset for a given ticker."""
    if data is None or data.empty:
        return pd.DataFrame()

    if 'ticker' in data.columns:
        return data[data['ticker'] == ticker].copy()

    if isinstance(data.index, pd.MultiIndex):
        for level in range(data.index.nlevels):
            try:
                return data.xs(ticker, level=level).copy()
            except (KeyError, ValueError):
                continue

    return data.copy()


def _prepare_validation_frame(raw_ticker_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare market data snapshot for signal validation."""
    if raw_ticker_data is None or raw_ticker_data.empty:
        return pd.DataFrame()

    validation_data = raw_ticker_data.copy()
    validation_data.columns = [str(col).lower() for col in validation_data.columns]
    return validation_data


def compute_portfolio_metrics(
    raw_data: pd.DataFrame,
    tickers: Sequence[str],
    optimizer_cfg: Dict[str, Any],
    pipeline_log: PipelineLogger,
    pipeline_id: str,
    stage_name: str,
) -> None:
    """Run portfolio optimisation and log metrics when enabled."""
    if not optimizer_cfg.get("enabled", False):
        return

    price_field = optimizer_cfg.get("price_field", "Close")
    price_matrix = _extract_price_matrix(raw_data, price_field=price_field, tickers=tickers)
    if price_matrix.empty or price_matrix.shape[0] < 2:
        logger.warning("Portfolio optimizer skipped; insufficient price data for '%s'.", price_field)
        return

    returns = price_matrix.pct_change().dropna(how="any")
    if returns.empty:
        logger.warning("Portfolio optimizer skipped; unable to compute returns.")
        return

    try:
        weights, optimisation_meta = optimize_portfolio_markowitz(
            returns.values,
            risk_aversion=optimizer_cfg.get("risk_aversion", 1.0),
            constraints=optimizer_cfg.get("constraints"),
        )
        if isinstance(optimisation_meta, dict) and not optimisation_meta.get("success", True):
            logger.warning(
                "Markowitz optimisation reported issues: %s",
                optimisation_meta.get("message", "unknown error"),
            )
    except Exception as exc:
        logger.warning("Markowitz optimisation failed (%s). Falling back to equal weights.", exc)
        weights = np.repeat(1.0 / returns.shape[1], returns.shape[1])

    metrics = calculate_enhanced_portfolio_metrics(
        returns.values,
        weights,
    )
    pipeline_log.log_event(
        "portfolio_metrics",
        pipeline_id,
        stage=stage_name,
        metadata=metrics,
    )
    logger.info(
        "Portfolio metrics | Sharpe=%.2f Sortino=%.2f MaxDD=%.2f%%",
        metrics.get("sharpe_ratio", 0.0),
        metrics.get("sortino_ratio", 0.0),
        metrics.get("max_drawdown", 0.0) * 100,
    )


@dataclass
class CVSettings:
    use_cv: bool
    n_splits: int
    test_size: float
    gap: int
    expanding_window: bool
    window_strategy: str
    train_ratio: float
    val_ratio: float
    default_strategy: str
    expected_coverage: float
    chronological_split: bool


@dataclass
class LLMComponents:
    enabled: bool = False
    client: Optional[OllamaClient] = None
    market_analyzer: Optional[LLMMarketAnalyzer] = None
    signal_generator: Optional[LLMSignalGenerator] = None
    risk_assessor: Optional[LLMRiskAssessor] = None
    signal_validator: Optional[SignalQualityValidator] = None
    optimizer: Optional[LLMPerformanceOptimizer] = None
    validator_version: str = "v1"
    llm_config: Optional[Dict[str, Any]] = None


def _load_pipeline_config_safe(config_path: str) -> Dict[str, Any]:
    try:
        return load_config(config_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load pipeline config: %s", exc)
        logger.info("Using fallback configuration...")
        return {'pipeline': {'stages': []}}


def _resolve_cv_settings(
    use_cv_flag: Optional[bool],
    n_splits_flag: Optional[int],
    test_size_flag: Optional[float],
    gap_flag: Optional[int],
    data_split_cfg: Dict[str, Any],
) -> CVSettings:
    cv_config = data_split_cfg.get('cross_validation', {})
    simple_config = data_split_cfg.get('simple_split', {})
    default_strategy = data_split_cfg.get('default_strategy', 'simple')

    use_cv = use_cv_flag if use_cv_flag is not None else (default_strategy == 'cv')
    n_splits = n_splits_flag or cv_config.get('n_splits', 5)
    test_size = test_size_flag or cv_config.get('test_size', 0.15)
    gap = gap_flag or cv_config.get('gap', 0)

    expanding_window = cv_config.get('expanding_window')
    if expanding_window is None:
        window_strategy_cfg = str(cv_config.get('window_strategy', 'expanding')).lower()
        expanding_window = window_strategy_cfg != 'sliding'
    elif isinstance(expanding_window, str):
        expanding_window = expanding_window.lower() != 'false'
    window_strategy = 'expanding' if expanding_window else 'sliding'

    train_ratio = simple_config.get('train_ratio', 0.7)
    val_ratio = simple_config.get('validation_ratio', 0.15)
    expected_coverage = cv_config.get('expected_coverage', 0.83)
    chronological = simple_config.get('chronological', True)

    return CVSettings(
        use_cv=use_cv,
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        expanding_window=expanding_window,
        window_strategy=window_strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        default_strategy=default_strategy,
        expected_coverage=expected_coverage,
        chronological_split=bool(chronological),
    )


def _load_llm_config_data() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    try:
        llm_config_data = load_config('config/llm_config.yml')
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to load LLM config: %s", exc)
        return {}, {}, {}, 'v1'

    llm_cfg = llm_config_data.get('llm', {})
    signal_validation_cfg = llm_cfg.get('signal_generator', {}).get('validation', {}) or {}
    validator_version = signal_validation_cfg.get('version', 'v1')
    return llm_config_data, llm_cfg, signal_validation_cfg, validator_version


def _initialize_llm_components(
    enable_llm: bool,
    llm_model: str,
    llm_cfg: Dict[str, Any],
    signal_validation_cfg: Dict[str, Any],
) -> LLMComponents:
    components = LLMComponents(
        enabled=False,
        validator_version=signal_validation_cfg.get('version', 'v1'),
        llm_config=llm_cfg,
    )

    if not enable_llm:
        return components

    try:
        server_cfg = llm_cfg.get('server', {})
        performance_cfg = llm_cfg.get('performance', {})
        generation_cfg = llm_cfg.get('generation', {})
        signal_cfg = llm_cfg.get('signal_generator', {})
        risk_cfg = llm_cfg.get('risk_assessor', {})

        host = server_cfg.get('host', "http://localhost:11434")
        timeout_seconds = int(server_cfg.get('timeout_seconds', 120))

        cache_enabled = bool(performance_cfg.get('track_cache_usage', True))
        cache_max_size_cfg = performance_cfg.get('cache_max_size', 32)
        try:
            cache_max_size = int(cache_max_size_cfg) if cache_max_size_cfg is not None else 32
        except (TypeError, ValueError):
            cache_max_size = 32

        optimizer = LLMPerformanceOptimizer()
        model_to_use = llm_model or llm_cfg.get('active_model')

        llm_client = OllamaClient(
            host=host,
            model=model_to_use,
            timeout=timeout_seconds,
            optimizer=optimizer,
            optimize_use_case=performance_cfg.get('default_use_case', 'balanced'),
            enable_cache=cache_enabled,
            cache_max_size=cache_max_size,
            generation_options=generation_cfg,
        )

        if not llm_client.health_check():
            raise OllamaConnectionError("Ollama health check failed")

        signal_validator = SignalQualityValidator(
            min_confidence_threshold=float(signal_validation_cfg.get('min_confidence_for_action', 0.6)),
            max_risk_threshold=float(signal_validation_cfg.get('max_risk_threshold', 0.15)),
            min_expected_return=float(signal_validation_cfg.get('min_expected_return', 0.02)),
        )

        components.enabled = True
        components.client = llm_client
        components.market_analyzer = LLMMarketAnalyzer(llm_client)
        components.signal_generator = LLMSignalGenerator(
            llm_client,
            system_prompt=signal_cfg.get('system_prompt'),
            temperature=signal_cfg.get('temperature', 0.05),
            validation_rules=signal_validation_cfg,
        )
        components.risk_assessor = LLMRiskAssessor(
            llm_client,
            system_prompt=risk_cfg.get('system_prompt'),
            temperature=risk_cfg.get('temperature', 0.1),
        )
        components.signal_validator = signal_validator
        components.optimizer = optimizer

        logger.info("✓ LLM initialized: %s", llm_client.model)
    except OllamaConnectionError as exc:
        logger.error("✗ LLM initialization failed: %s", exc)
        logger.error("  Fix: Ensure Ollama is running: 'ollama serve'")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("✗ LLM initialization error: %s", exc)

    return components


def _prepare_ticker_list(
    tickers_argument: str,
    discovery_cfg: Dict[str, Any],
    use_ticker_discovery: bool,
    refresh_ticker_universe: bool,
) -> List[str]:
    manual_tickers = [t.strip() for t in tickers_argument.split(',') if t.strip()]
    discovery_enabled = discovery_cfg.get('enabled', False) or use_ticker_discovery

    if not discovery_enabled:
        return manual_tickers

    loader_type = discovery_cfg.get('loader', 'alpha_vantage').lower()
    if loader_type != 'alpha_vantage':
        raise ValueError(f"Unsupported ticker discovery loader: {loader_type}")

    loader = AlphaVantageTickerLoader(
        api_key=discovery_cfg.get('api_key'),
        cache_dir=discovery_cfg.get('cache_dir'),
    )
    validator = TickerValidator()
    universe_manager = TickerUniverseManager(
        loader=loader,
        validator=validator,
        universe_path=discovery_cfg.get('universe_path'),
    )
    fallback_csv = discovery_cfg.get('fallback_csv')
    fallback_path = Path(fallback_csv) if fallback_csv else None

    try:
        if refresh_ticker_universe or discovery_cfg.get('auto_refresh', False):
            universe = universe_manager.refresh_universe(
                force_download=True,
                fallback_csv=fallback_path,
            )
        else:
            universe = universe_manager.load_universe()
            if not universe.tickers:
                universe = universe_manager.refresh_universe(
                    force_download=False,
                    fallback_csv=fallback_path,
                )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Ticker discovery failed (%s); falling back to CLI tickers.", exc)
        return manual_tickers

    if not universe.tickers:
        logger.warning("Ticker discovery produced an empty universe; using CLI tickers instead.")
        return manual_tickers

    logger.info("Using %s tickers from discovery universe.", len(universe.tickers))
    return universe.tickers


def _log_split_strategy(settings: CVSettings) -> None:
    if settings.use_cv:
        logger.info("✓ Using k-fold cross-validation (k=%d)", settings.n_splits)
        logger.info("  - Test size: %d%%", int(settings.test_size * 100))
        logger.info("  - Gap between train/val: %d periods", settings.gap)
        logger.info("  - Window strategy: %s", settings.window_strategy)
        logger.info("  - Expected validation coverage: %d%%", int(settings.expected_coverage * 100))
    else:
        test_ratio = max(0.0, 1 - settings.train_ratio - settings.val_ratio)
        logger.info(
            "Using simple chronological split (%d/%d/%d)",
            int(settings.train_ratio * 100),
            int(settings.val_ratio * 100),
            int(test_ratio * 100),
        )
        logger.info("  - Strategy: %s", settings.default_strategy)
        logger.info("  - Chronological: %s", 'yes' if settings.chronological_split else 'no')

@click.command()
@click.option('--config', default='config/pipeline_config.yml',
              help='Path to pipeline configuration (default: config/pipeline_config.yml)')
@click.option('--data-source', default=None,
              help='Data source to use (options: yfinance, alpha_vantage, finnhub). Defaults to config value.')
@click.option('--tickers', default='AAPL,MSFT',
              help='Comma-separated ticker symbols (default: AAPL,MSFT)')
@click.option('--start', default='2020-01-01',
              help='Start date YYYY-MM-DD (default: 2020-01-01)')
@click.option('--end', default='2024-01-01',
              help='End date YYYY-MM-DD (default: 2024-01-01)')
@click.option('--use-cv', is_flag=True, default=None,
              help='Use k-fold cross-validation. If not set, uses config default_strategy.')
@click.option('--n-splits', default=None, type=int,
              help='Number of CV folds. If not set, uses config value.')
@click.option('--test-size', default=None, type=float,
              help='Test set size (0.0-1.0). If not set, uses config value.')
@click.option('--gap', default=None, type=int,
              help='Gap between train/validation periods. If not set, uses config value.')
@click.option('--verbose', is_flag=True, default=False,
              help='Enable verbose logging (DEBUG level)')
@click.option('--enable-llm', is_flag=True, default=False,
              help='Enable LLM integration for market analysis and signal generation')
@click.option('--llm-model', default= 'qwen:14b-chat-q4_K_M',
              help='LLM model to use (default: from config). Options: deepseek-coder:6.7b-instruct-q4_K_M, codellama:13b-instruct-q4_K_M, qwen:14b-chat-q4_K_M')
@click.option('--dry-run', is_flag=True, default=False,
              help='Generate synthetic OHLCV data in-process (no network) to exercise stages')
@click.option('--execution-mode', default='auto',
              type=click.Choice(EXECUTION_MODES, case_sensitive=False),
              help='Data extraction mode: live (network), synthetic (offline), or auto (try live, fallback to synthetic)')
@click.option('--use-ticker-discovery', is_flag=True, default=False,
              help='Load tickers from the configured ticker discovery universe.')
@click.option('--refresh-ticker-universe', is_flag=True, default=False,
              help='Force refresh of the ticker discovery universe before running the pipeline.')
def run_pipeline(config: str, data_source: str, tickers: str, start: str, end: str,
                use_cv: bool, n_splits: int, test_size: float, gap: int, verbose: bool,
                enable_llm: bool, llm_model: str, dry_run: bool,
                execution_mode: str, use_ticker_discovery: bool,
                refresh_ticker_universe: bool) -> None:
    """Execute ETL pipeline with modular configuration-driven orchestration.

    Data Splitting Strategy:
    - Default (--use-cv=False): Simple 70/15/15 chronological split (backward compatible)
    - Recommended (--use-cv): k-fold cross-validation with expanding window
      * 5.5x better temporal coverage (15% → 83%)
      * Eliminates temporal gap (0 years vs 2.5 years)
      * Strict test isolation (15% never exposed during CV)

    Examples:
        # Simple split (backward compatible)
        python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --start 2020-01-01

        # k-fold CV (recommended for production)
        python scripts/run_etl_pipeline.py --tickers AAPL --use-cv --n-splits 5

        # Verbose logging
        python scripts/run_etl_pipeline.py --tickers GOOGL --use-cv --verbose

        # Live run with automatic synthetic fallback
        python scripts/run_etl_pipeline.py --tickers NVDA --execution-mode auto --enable-llm

        # Dry run (no network) with synthetic data
        python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --start 2024-01-02 --end 2024-01-19 --dry-run
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Load pipeline configuration and derived settings
    pipeline_config = _load_pipeline_config_safe(config)
    pipeline_cfg = pipeline_config.get('pipeline', {})
    stages_cfg = pipeline_cfg.get('stages', [])
    data_split_cfg = pipeline_cfg.get('data_split', {})
    discovery_cfg = pipeline_cfg.get('ticker_discovery', {})
    portfolio_optimizer_cfg = pipeline_cfg.get('portfolio_optimizer', {})

    cv_settings = _resolve_cv_settings(use_cv, n_splits, test_size, gap, data_split_cfg)
    use_cv = cv_settings.use_cv
    n_splits = cv_settings.n_splits
    test_size = cv_settings.test_size
    gap = cv_settings.gap
    expanding_window = cv_settings.expanding_window
    train_ratio = cv_settings.train_ratio
    val_ratio = cv_settings.val_ratio

    ticker_list = _prepare_ticker_list(
        tickers_argument=tickers,
        discovery_cfg=discovery_cfg,
        use_ticker_discovery=use_ticker_discovery,
        refresh_ticker_universe=refresh_ticker_universe,
    )

    llm_config_data: Dict[str, Any] = {}
    llm_cfg: Dict[str, Any] = {}
    signal_validation_cfg: Dict[str, Any] = {}
    if enable_llm:
        logger.info("Initializing LLM components...")
        llm_config_data, llm_cfg, signal_validation_cfg, validator_version = _load_llm_config_data()
    else:
        validator_version = 'v1'

    llm_components = _initialize_llm_components(
        enable_llm=enable_llm,
        llm_model=llm_model,
        llm_cfg=llm_cfg,
        signal_validation_cfg=signal_validation_cfg,
    )
    enable_llm = llm_components.enabled
    validator_version = llm_components.validator_version

    logger.info("Pipeline: Portfolio Maximizer v45 (Phase 5.2)")
    logger.info("Data Source: %s", data_source if data_source else 'from config')
    logger.info("Tickers: %s", ', '.join(ticker_list) if ticker_list else '(none)')
    logger.info("Date range: %s to %s", start, end)
    logger.info("LLM Integration: %s", 'ENABLED' if enable_llm else 'DISABLED')

    execution_mode = execution_mode.lower()
    if dry_run:
        execution_mode = 'synthetic'
        logger.info("Execution mode overridden to SYNTHETIC via --dry-run flag")
    elif execution_mode not in EXECUTION_MODES:
        raise click.BadParameter(f"Invalid execution mode '{execution_mode}'. "
                                 f"Choose from {', '.join(EXECUTION_MODES)}.")
    logger.info(f"Execution mode: {execution_mode.upper()}")
    allow_live_fallback = (execution_mode == 'auto')
    if allow_live_fallback:
        logger.info("Auto mode: attempting live extraction first, synthetic fallback enabled on failure")
    
    # Initialize LLM component handles
    llm_client = llm_components.client
    market_analyzer = llm_components.market_analyzer
    signal_generator = llm_components.signal_generator
    risk_assessor = llm_components.risk_assessor
    signal_validator = llm_components.signal_validator

    # Initialize storage
    storage = DataStorage()

    # Initialize database for persistent storage
    db_manager = DatabaseManager(db_path="data/portfolio_maximizer.db")
    logger.info("✓ Database manager initialized")

    # Initialize checkpoint manager and pipeline logger
    pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_manager = CheckpointManager(checkpoint_dir="data/checkpoints")
    pipeline_log = PipelineLogger(log_dir="logs", retention_days=7)

    logger.info(f"✓ Pipeline ID: {pipeline_id}")
    pipeline_log.log_event('pipeline_start', pipeline_id, metadata={
        'tickers': ticker_list,
        'start_date': start,
        'end_date': end,
        'use_cv': use_cv
    })

    # Initialize data source manager (platform-agnostic)
    try:
        data_source_manager = DataSourceManager(
            config_path='config/data_sources_config.yml',
            storage=storage
        )
        logger.info(f"✓ Data source manager initialized")
        logger.info(f"  Available sources: {', '.join(data_source_manager.get_available_sources())}")
        logger.info(f"  Active source: {data_source_manager.get_active_source()}")
    except Exception as e:
        logger.error(f"Failed to initialize data source manager: {e}")
        logger.error("Pipeline cannot continue without data source")
        raise

    # Determine split strategy and log configuration
    _log_split_strategy(cv_settings)

    # Define stage names (in execution order)
    stage_names = ['data_extraction', 'data_validation', 'data_preprocessing']
    
    # Add LLM stages if enabled
    if enable_llm:
        stage_names.extend(['llm_market_analysis', 'llm_signal_generation', 'llm_risk_assessment'])
    
    stage_names.append('data_storage')

    # Prepare synthetic data up-front when explicitly requested
    precomputed_raw_data: Optional[pd.DataFrame] = None
    if execution_mode == 'synthetic':
        logger.info("Synthetic mode: precomputing deterministic OHLCV data (offline)")
        precomputed_raw_data = generate_synthetic_ohlcv(ticker_list, start, end)

    # Execute pipeline stages
    logger.info("=" * 70)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 70)

    for stage_name in tqdm(stage_names, desc='Pipeline Progress'):
        logger.info(f"\n[Stage: {stage_name}]")
        stage_start_time = time.time()
        pipeline_log.log_stage_start(pipeline_id, stage_name)

        try:
            if stage_name == 'data_extraction':
                # Stage 1: Data Extraction (platform-agnostic)
                logger.info(f"Extracting OHLCV data using data source manager...")

                raw_data = None
                extraction_source = None
                synthetic_fallback = False

                if precomputed_raw_data is not None:
                    raw_data = precomputed_raw_data.copy()
                    extraction_source = 'synthetic'
                    logger.info("Using synthetic OHLCV data (precomputed)")
                else:
                    try:
                        # Use data source manager for extraction (automatic failover support)
                        raw_data = data_source_manager.extract_ohlcv(
                            tickers=ticker_list,
                            start_date=start,
                            end_date=end,
                            prefer_source=data_source if data_source not in (None, 'yfinance') else None
                        )
                        extraction_source = data_source_manager.get_active_source()
                    except Exception as extraction_error:
                        if allow_live_fallback:
                            logger.warning(f"Live extraction failed: {extraction_error}")
                            logger.warning("Falling back to synthetic OHLCV data to keep pipeline operational")
                            raw_data = generate_synthetic_ohlcv(ticker_list, start, end)
                            extraction_source = 'synthetic(auto-fallback)'
                            synthetic_fallback = True
                        else:
                            raise

                if raw_data is None or raw_data.empty:
                    if allow_live_fallback and extraction_source and not extraction_source.lower().startswith('synthetic'):
                        logger.warning("Live extraction returned empty dataset; generating synthetic fallback")
                        raw_data = generate_synthetic_ohlcv(ticker_list, start, end)
                        extraction_source = 'synthetic(auto-fallback)'
                        synthetic_fallback = True
                    else:
                        raise RuntimeError("Data extraction failed - empty dataset")

                logger.info(f"✓ Extracted {len(raw_data)} rows from {len(ticker_list)} ticker(s)")
                logger.info(f"  Source: {extraction_source}")

                if synthetic_fallback:
                    pipeline_log.log_event(
                        'auto_fallback_engaged',
                        pipeline_id,
                        stage=stage_name,
                        metadata={'reason': 'live_extraction_failure'}
                    )

                # Save to database
                rows_saved = db_manager.save_ohlcv_data(
                    raw_data,
                    source=extraction_source
                )
                logger.info(f"✓ Saved {rows_saved} rows to database")

                # Log cache statistics
                if extraction_source and not extraction_source.lower().startswith('synthetic'):
                    cache_stats = data_source_manager.get_cache_statistics()
                    for source_name, stats in cache_stats.items():
                        if stats['total_requests'] > 0:
                            partial_note = ""
                            partial_hits = stats.get('cache_partial_hits', 0)
                            if partial_hits:
                                partial_note = f", {partial_hits} partial"
                            logger.info(
                                f"  {source_name} cache: {stats['cache_hits']}/{stats['total_requests']} hits "
                                f"({stats['hit_rate']:.1%}){partial_note}"
                            )

                # Save checkpoint
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=raw_data,
                    metadata={'tickers': ticker_list, 'rows': len(raw_data)}
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'data_validation':
                # Stage 2: Data Validation
                logger.info("Validating data quality...")
                validator = DataValidator()
                report = validator.validate_ohlcv(raw_data)

                if not report['passed']:
                    logger.warning(f"⚠ Validation warnings detected")
                    logger.warning(f"  Errors: {len(report.get('errors', []))}")
                    logger.warning(f"  Warnings: {len(report.get('warnings', []))}")
                    if verbose:
                        logger.debug(f"Validation report: {report}")
                else:
                    logger.info("✓ Data validation passed")

            elif stage_name == 'data_preprocessing':
                # Stage 3: Data Preprocessing
                logger.info("Preprocessing data (missing data + normalization)...")
                processor = Preprocessor()

                # Handle missing values
                filled = processor.handle_missing(raw_data)
                logger.debug(f"  Missing data handled")

                # Normalize (returns tuple: data, stats)
                normalized, stats = processor.normalize(filled)
                processed = normalized
                logger.debug(f"  Normalization complete (μ=0, σ²=1)")

                # Save processed data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                storage.save(processed, 'processed', f'processed_{timestamp}')
                logger.info(f"✓ Preprocessed {len(processed)} rows")

            elif stage_name == 'llm_market_analysis':
                # Stage 4 (LLM): Market Analysis
                logger.info("Analyzing market data with LLM...")
                llm_analyses = {}
                
                for ticker in ticker_list:
                    try:
                        # Get ticker-specific data
                        ticker_data = processed.xs(ticker, level=0) if isinstance(processed.index, pd.MultiIndex) else processed
                        
                        # Run LLM analysis with timing
                        start_time = time.time()
                        analysis = market_analyzer.analyze_ohlcv(ticker_data, ticker)
                        latency = time.time() - start_time
                        
                        llm_analyses[ticker] = analysis
                        
                        # Save to database
                        db_manager.save_llm_analysis(
                            ticker=ticker,
                            date=datetime.now().strftime('%Y-%m-%d'),
                            analysis=analysis,
                            model_name=llm_client.model,
                            latency=latency
                        )
                        
                        logger.info(f"  ✓ {ticker}: Trend={analysis['trend']}, Strength={analysis['strength']}/10 ({latency:.1f}s)")
                        if verbose:
                            logger.debug(f"    Summary: {analysis['summary']}")
                    except Exception as e:
                        logger.warning(f"  ⚠ {ticker} analysis failed: {e}")
                        llm_analyses[ticker] = {'trend': 'neutral', 'strength': 5, 'error': str(e)}
                
                logger.info(f"✓ Analyzed {len(llm_analyses)} ticker(s) with LLM")
                
                # Save checkpoint with LLM analysis
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=processed,
                    metadata={'analyses': llm_analyses, 'tickers': ticker_list}
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'llm_signal_generation':
                # Stage 5 (LLM): Signal Generation
                logger.info("Generating trading signals with LLM...")
                llm_signals = {}
                llm_signal_validations = {}
                
                for ticker in ticker_list:
                    try:
                        # Get ticker-specific data and analysis
                        ticker_data = processed.xs(ticker, level=0) if isinstance(processed.index, pd.MultiIndex) else processed
                        market_analysis = llm_analyses.get(ticker, {})
                        
                        # Generate signal with timing
                        start_time = time.time()
                        signal = signal_generator.generate_signal(ticker_data, ticker, market_analysis)
                        latency = time.time() - start_time
                        validation_status = 'pending'
                        validation_entry: Dict[str, Any] = {}

                        ticker_raw = _slice_ticker_dataframe(raw_data, ticker)
                        validation_data = _prepare_validation_frame(ticker_raw)
                        if not validation_data.empty and 'close' in validation_data.columns:
                            try:
                                price_at_signal = float(validation_data['close'].iloc[-1])
                            except Exception:
                                price_at_signal = 0.0
                        else:
                            price_at_signal = 0.0

                        signal['entry_price'] = price_at_signal

                        if signal_validator is not None:
                            try:
                                direction_value = str(signal.get('action', 'HOLD')).upper()
                                try:
                                    direction_enum = SignalDirection(direction_value)
                                except ValueError:
                                    direction_enum = SignalDirection.HOLD

                                validator_signal = Signal(
                                    ticker=ticker,
                                    direction=direction_enum,
                                    confidence=float(signal.get('confidence', 0.0)),
                                    reasoning=signal.get('reasoning', ''),
                                    timestamp=datetime.now(),
                                    price_at_signal=price_at_signal or 0.0,
                                    expected_return=signal.get('expected_return'),
                                    risk_estimate=signal.get('risk_estimate'),
                                )

                                if validation_data.empty or 'close' not in validation_data.columns:
                                    raise ValueError("insufficient market data for validation")

                                validation_result = signal_validator.validate_signal(validator_signal, validation_data)
                                validation_status = 'validated' if validation_result.is_valid else 'failed'
                                validation_entry = {
                                    'validator_version': validator_version,
                                    'confidence_score': float(validation_result.confidence_score),
                                    'recommendation': validation_result.recommendation,
                                    'warnings': validation_result.warnings,
                                    'quality_metrics': validation_result.quality_metrics,
                                }

                                if not validation_result.is_valid or validation_result.recommendation == 'HOLD':
                                    if signal.get('action') != 'HOLD':
                                        logger.debug("Validator adjusted %s signal to HOLD", ticker)
                                        signal['action'] = 'HOLD'
                                    if validation_result.warnings:
                                        signal['reasoning'] = (
                                            f"{signal.get('reasoning', '')} "
                                            f"[Validator: {validation_result.warnings[0]}]"
                                        ).strip()
                            except Exception as validation_error:
                                validation_status = 'failed'
                                validation_entry = {
                                    'validator_version': validator_version,
                                    'confidence_score': 0.0,
                                    'recommendation': 'HOLD',
                                    'warnings': [f'validator_error: {validation_error}'],
                                    'quality_metrics': {},
                                }
                                if signal.get('action') != 'HOLD':
                                    signal['action'] = 'HOLD'
                                    signal['reasoning'] = (
                                        f"{signal.get('reasoning', '')} [Validator error: {validation_error}]"
                                    ).strip()

                        signal['validation'] = validation_entry
                        
                        llm_signals[ticker] = signal
                        if validation_entry:
                            llm_signal_validations[ticker] = validation_entry
                        
                        # Save to database
                        signal_id = db_manager.save_llm_signal(
                            ticker=ticker,
                            date=datetime.now().strftime('%Y-%m-%d'),
                            signal=signal,
                            model_name=llm_client.model,
                            latency=latency,
                            validation_status=validation_status
                        )
                        if signal_id != -1 and validation_entry:
                            db_manager.save_signal_validation(signal_id, validation_entry)
                        
                        confidence_pct = float(signal.get('confidence', 0.0)) * 100
                        logger.info(
                            "  ✓ %s: Action=%s, Confidence=%.1f%% (%0.1fs) Validation=%s",
                            ticker,
                            signal['action'],
                            confidence_pct,
                            latency,
                            validation_status.upper(),
                        )
                        if verbose:
                            logger.debug(f"    Reasoning: {signal['reasoning']}")
                    except Exception as e:
                        logger.warning(f"  ⚠ {ticker} signal generation failed: {e}")
                        llm_signals[ticker] = {'action': 'HOLD', 'confidence': 0.5, 'error': str(e)}
                
                logger.info(f"✓ Generated {len(llm_signals)} signal(s) with LLM")
                logger.warning("⚠ ADVISORY ONLY: LLM signals require 30-day validation before trading")
                
                # Save checkpoint with signals
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=processed,
                    metadata={'signals': llm_signals, 'analyses': llm_analyses, 'validations': llm_signal_validations}
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'llm_risk_assessment':
                # Stage 6 (LLM): Risk Assessment
                logger.info("Assessing portfolio risk with LLM...")
                portfolio_weight = 1.0 / len(ticker_list)  # Equal weight assumption
                llm_risks = {}
                
                for ticker in ticker_list:
                    try:
                        # Get ticker-specific data
                        ticker_data = processed.xs(ticker, level=0) if isinstance(processed.index, pd.MultiIndex) else processed
                        
                        # Assess risk with timing
                        start_time = time.time()
                        risk = risk_assessor.assess_risk(ticker_data, ticker, portfolio_weight)
                        latency = time.time() - start_time
                        
                        llm_risks[ticker] = risk
                        
                        # Save to database
                        risk['portfolio_weight'] = portfolio_weight
                        db_manager.save_llm_risk(
                            ticker=ticker,
                            date=datetime.now().strftime('%Y-%m-%d'),
                            risk=risk,
                            model_name=llm_client.model,
                            latency=latency
                        )
                        
                        logger.info(f"  ✓ {ticker}: Risk Level={risk['risk_level']}, Score={risk['risk_score']}/100 ({latency:.1f}s)")
                        if verbose and risk.get('concerns'):
                            logger.debug(f"    Concerns: {', '.join(risk['concerns'][:2])}")
                    except Exception as e:
                        logger.warning(f"  ⚠ {ticker} risk assessment failed: {e}")
                        llm_risks[ticker] = {'risk_level': 'medium', 'risk_score': 50, 'error': str(e)}
                
                logger.info(f"✓ Assessed {len(llm_risks)} ticker(s) risk with LLM")
                
                # Save checkpoint with risk assessment
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=processed,
                    metadata={'risks': llm_risks, 'signals': llm_signals, 'analyses': llm_analyses}
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'data_storage':
                # Stage 4: Data Storage (Split + Save)
                logger.info("Splitting and saving datasets...")
                logger.info(f"  Configuration source: pipeline_config.yml")
                logger.info(f"  Split strategy: {'CV' if use_cv else 'Simple'}")

                # Perform split using configuration-driven parameters
                if use_cv:
                    splits = storage.train_validation_test_split(
                        processed,
                        use_cv=True,
                        n_splits=n_splits,
                        test_size=test_size,
                        gap=gap,
                        expanding_window=expanding_window
                    )
                else:
                    splits = storage.train_validation_test_split(
                        processed,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        use_cv=False
                    )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if use_cv:
                    # Save CV folds
                    for fold in splits['cv_folds']:
                        fold_id = fold['fold_id']
                        storage.save(fold['train'], 'training',
                                   f'fold{fold_id}_train_{timestamp}')
                        storage.save(fold['validation'], 'validation',
                                   f'fold{fold_id}_val_{timestamp}')

                    # Save isolated test set
                    storage.save(splits['testing'], 'testing', f'test_{timestamp}')

                    # Calculate summary statistics
                    avg_train_size = sum(len(f['train']) for f in splits['cv_folds']) / len(splits['cv_folds'])
                    avg_val_size = sum(len(f['validation']) for f in splits['cv_folds']) / len(splits['cv_folds'])

                    logger.info(f"✓ Saved {len(splits['cv_folds'])} CV folds + 1 test set")
                    logger.info(f"  - Train size (avg): {avg_train_size:.0f} rows")
                    logger.info(f"  - Val size (avg): {avg_val_size:.0f} rows")
                    logger.info(f"  - Test size: {len(splits['testing'])} rows")
                else:
                    # Simple split (backward compatible)
                    for split_name, split_data in splits.items():
                        storage.save(split_data, split_name,
                                   f'{split_name}_{timestamp}')

                    logger.info(f"✓ Saved simple split:")
                    logger.info(f"  - Training: {len(splits['training'])} rows (70%)")
                    logger.info(f"  - Validation: {len(splits['validation'])} rows (15%)")
                    logger.info(f"  - Testing: {len(splits['testing'])} rows (15%)")

                compute_portfolio_metrics(
                    raw_data=raw_data,
                    tickers=ticker_list,
                    optimizer_cfg=portfolio_optimizer_cfg,
                    pipeline_log=pipeline_log,
                    pipeline_id=pipeline_id,
                    stage_name=stage_name,
                )

            # Log stage completion
            stage_duration = time.time() - stage_start_time
            pipeline_log.log_stage_complete(pipeline_id, stage_name, metadata={
                'duration_seconds': stage_duration
            })
            pipeline_log.log_performance(pipeline_id, stage_name, stage_duration)

        except Exception as e:
            logger.error(f"✗ Stage '{stage_name}' failed: {e}")
            pipeline_log.log_stage_error(pipeline_id, stage_name, e)

            if verbose:
                import traceback
                logger.debug(traceback.format_exc())
            raise

    # Pipeline completion
    logger.info("=" * 70)
    logger.info("✓ Pipeline completed successfully")
    logger.info("=" * 70)

    # Log pipeline completion
    pipeline_log.log_event('pipeline_complete', pipeline_id, status='success')

    # Cleanup old logs and checkpoints
    pipeline_log.cleanup_old_logs()
    checkpoint_manager.cleanup_old_checkpoints(retention_days=7)
    
    # Close database connection
    db_manager.close()


if __name__ == '__main__':
    run_pipeline()
