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
from __future__ import annotations

import sys
import yaml
import logging
import warnings
import os
import hashlib
import json
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
from etl.split_diagnostics import summarize_returns, drift_metrics, validate_non_overlap
from etl.frontier_markets import (
    FRONTIER_MARKET_TICKERS_BY_REGION,
    merge_frontier_tickers,
)

# LLM Integration (Phase 5.2)
from ai_llm.ollama_client import OllamaClient, OllamaConnectionError
from ai_llm.market_analyzer import LLMMarketAnalyzer
from ai_llm.signal_generator import LLMSignalGenerator
from ai_llm.risk_assessor import LLMRiskAssessor
from ai_llm.performance_optimizer import LLMPerformanceOptimizer
from ai_llm.signal_quality_validator import SignalQualityValidator, Signal, SignalDirection
from ai_llm.signal_validator import SignalValidator as AdvancedSignalValidator

# Database Integration (Phase 5.2+)
from etl.database_manager import DatabaseManager
from scripts.track_llm_signals import LLMSignalTracker

import time

# Supported execution modes for data extraction
EXECUTION_MODES = ('auto', 'live', 'synthetic')


def _normalize_change_points(raw_change_points: Any) -> List[str]:
    """Convert MSSA change point payloads into serialisable ISO strings."""
    if raw_change_points is None:
        return []
    if isinstance(raw_change_points, (pd.Index, pd.Series, np.ndarray)):
        iterable = list(raw_change_points)
    elif isinstance(raw_change_points, (list, tuple, set)):
        iterable = list(raw_change_points)
    else:
        iterable = [raw_change_points]

    normalized: List[str] = []
    for cp in iterable:
        if cp is None:
            continue
        if isinstance(cp, (float, np.floating)) and np.isnan(cp):
            continue
        if isinstance(cp, (pd.Timestamp, datetime, np.datetime64)):
            normalized.append(pd.Timestamp(cp).isoformat())
        elif hasattr(cp, 'isoformat'):
            normalized.append(cp.isoformat())  # type: ignore[call-arg]
        else:
            normalized.append(str(cp))
    return normalized


def _setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for pipeline execution.
    
    This function is called only when the script is run as main,
    preventing logging side effects when importing the module.
    
    Args:
        verbose: If True, set logging level to DEBUG
        
    Returns:
        Configured logger instance
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure file handler
    file_handler = logging.FileHandler(
        str(logs_dir / "pipeline_run.log"),
        mode="a"
    )
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Set root logger level
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.captureWarnings(True)
    
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def _detect_device(prefer_gpu: bool = True) -> str:
    """Detect best available device (cuda/mps/cpu) with optional GPU preference."""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


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
        alternatives = []
        if not config_file.is_absolute():
            alternatives.append(Path("config") / config_file.name)
        if config_file.name in {"config.yml", "config.yaml"}:
            alternatives.append(Path("config") / "pipeline_config.yml")

        for candidate in alternatives:
            if candidate.exists():
                logger.warning(
                    "Configuration file %s not found; using %s instead",
                    config_path,
                    candidate,
                )
                config_file = candidate
                break
        else:
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"OK Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration: {e}")
        raise


def _generate_visual_dashboards(
    pipeline_cfg: Dict[str, Any],
    db_manager: DatabaseManager,
    tickers: Sequence[str],
) -> None:
    """Automatically generate visualization dashboards from persisted outputs."""
    visualization_cfg = pipeline_cfg.get('visualization', {})
    if not visualization_cfg.get('auto_dashboard'):
        return

    try:
        from etl.dashboard_loader import DashboardDataLoader
        from etl.visualizer import TimeSeriesVisualizer
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Visualization modules unavailable: %s", exc)
        return

    output_dir = Path(visualization_cfg.get('output_dir', 'visualizations'))
    output_dir.mkdir(parents=True, exist_ok=True)
    lookback_days = visualization_cfg.get('lookback_days', 180)
    generate_forecast = visualization_cfg.get('generate_forecast_dashboard', True)
    generate_signal = visualization_cfg.get('generate_signal_dashboard', True)

    loader = DashboardDataLoader(db_manager=db_manager)
    visualizer = TimeSeriesVisualizer()
    generated = 0

    for ticker in tickers:
        price_df = loader.get_price_history(ticker, lookback_days=lookback_days)
        if price_df is None or price_df.empty:
            continue
        price_df = price_df.sort_index()
        price_df = price_df.loc[~price_df.index.duplicated(keep="last")]

        if generate_forecast:
            forecast_bundle = loader.get_forecast_bundle(ticker)
            forecasts_available = {
                model: payload
                for model, payload in forecast_bundle.items()
                if isinstance(payload, dict) and payload.get("forecast") is not None
            }
            if forecasts_available:
                ensemble_payload = forecasts_available.get("COMBINED") or forecasts_available.get("ENSEMBLE")
                weights = None
                if isinstance(ensemble_payload, dict):
                    weights = ensemble_payload.get("weights")
                fig = visualizer.plot_forecast_dashboard(
                    price_df.iloc[:, 0],
                    forecasts_available,
                    title=f"{ticker} Forecast Dashboard",
                    weights=weights,
                )
                save_path = output_dir / f"{ticker}_forecast_dashboard.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                generated += 1
                logger.info("Forecast dashboard saved for %s -> %s", ticker, save_path)

        if generate_signal:
            signal_df = loader.get_signal_backtests(ticker)
            if not signal_df.empty:
                signal_df = signal_df.sort_index()
                signal_df = signal_df.loc[~signal_df.index.duplicated(keep="last")]
                fig = visualizer.plot_signal_performance(signal_df, ticker=ticker)
                save_path = output_dir / f"{ticker}_signal_dashboard.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                generated += 1
                logger.info("Signal dashboard saved for %s -> %s", ticker, save_path)

    if generated:
        logger.info("Visualization dashboards generated: %s", generated)
    else:
        logger.info("Visualization dashboard generation skipped (no data available).")


def _emit_split_drift_json(path: Path, run_id: str, records: List[Dict[str, Any]]) -> None:
    """Persist drift diagnostics to JSON for dashboards/audits."""
    if not records:
        return
    payload = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat(),
        "drift": records,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Split drift diagnostics emitted to %s", path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to emit split drift JSON: %s", exc)


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
    data = pd.concat(frames).sort_index()
    payload = {
        "tickers": tickers,
        "start": str(start_date),
        "end": str(end_date),
        "seed": seed,
        "generator_version": "v0",
    }
    dataset_id = f"syn_{hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]}"  # nosec B303
    data.attrs["dataset_id"] = dataset_id
    data.attrs["generator_version"] = "v0"
    data.attrs["source"] = "synthetic"
    return data


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
    normalised_columns = [str(col).lower() for col in validation_data.columns]
    validation_data.columns = normalised_columns

    # Provide canonical title-case aliases for validators expecting original names
    alias_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'adj close': 'Adj Close',
    }
    for lower_name, alias_name in alias_map.items():
        if lower_name in validation_data.columns and alias_name not in validation_data.columns:
            validation_data[alias_name] = validation_data[lower_name]

    return validation_data


def _compute_forward_return(
    price_series: pd.Series,
    signal_timestamp: datetime,
    horizon_days: int = 5,
) -> Optional[float]:
    """Compute forward return over a given horizon if price data permits."""
    if price_series.empty:
        return None

    try:
        indexed_prices = price_series.copy()
        if not isinstance(indexed_prices.index, pd.DatetimeIndex):
            indexed_prices.index = pd.to_datetime(indexed_prices.index)
    except Exception:
        return None

    signal_ts = pd.to_datetime(signal_timestamp)

    if signal_ts in indexed_prices.index:
        entry_price = float(indexed_prices.loc[signal_ts])
    else:
        prior = indexed_prices.loc[:signal_ts]
        if prior.empty:
            return None
        entry_price = float(prior.iloc[-1])

    if entry_price == 0:
        return None

    future_window = indexed_prices.loc[
        signal_ts + pd.Timedelta(days=1): signal_ts + pd.Timedelta(days=horizon_days)
    ]
    if future_window.empty:
        return None

    horizon_index = min(len(future_window) - 1, horizon_days - 1)
    exit_price = float(future_window.iloc[horizon_index])
    return (exit_price / entry_price) - 1


def _format_signals_for_backtest(signal_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert persisted signal rows into the schema expected by the validator backtest."""
    formatted: List[Dict[str, Any]] = []
    for row in signal_rows:
        timestamp_raw = row.get("signal_timestamp") or f"{row.get('signal_date')}T00:00:00"
        try:
            ts = pd.to_datetime(timestamp_raw)
        except Exception:
            continue
        formatted.append(
            {
                "ticker": row.get("ticker"),
                "action": row.get("action", "HOLD"),
                "confidence": row.get("confidence", 0.5),
                "signal_timestamp": ts.isoformat(),
            }
        )
    return formatted


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
        from etl.security_utils import sanitize_error
        safe_error = sanitize_error(exc)
        logger.warning("Markowitz optimisation failed (%s). Falling back to equal weights.", safe_error)
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
    signal_validator: Optional[Any] = None
    optimizer: Optional[LLMPerformanceOptimizer] = None
    validator_version: str = "v1"
    llm_config: Optional[Dict[str, Any]] = None


def _load_pipeline_config_safe(config_path: str) -> Dict[str, Any]:
    try:
        return load_config(config_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        from etl.security_utils import sanitize_error
        safe_error = sanitize_error(exc)
        logger.error("Failed to load pipeline config: %s", safe_error)
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
        window_strategy_cfg = str(cv_config.get('window_strategy', 'sliding')).lower()
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


def _build_stage_execution_order(
    stages_cfg: List[Dict[str, Any]],
    enable_llm: bool,
    logger: logging.Logger
) -> List[str]:
    """
    Build stage execution order from config, respecting dependencies and enabled flags.
    
    This function implements config-driven orchestration:
    1. Filters stages by enabled flag
    2. Resolves dependencies to determine execution order
    3. Includes LLM stages conditionally
    4. Maintains backward compatibility with hardcoded stages
    
    Args:
        stages_cfg: List of stage configurations from pipeline_config.yml
        enable_llm: Whether LLM integration is enabled
        logger: Logger instance
        
    Returns:
        List of stage names in execution order
    """
    # Core stages that must always run (backward compatibility)
    # Time Series-first architecture expects storage to precede forecasting.
    core_stages = ['data_extraction', 'data_validation', 'data_preprocessing', 'data_storage']
    
    # Build stage map from config
    stage_map = {}
    for stage in stages_cfg:
        stage_name = stage.get('name')
        if stage_name:
            stage_map[stage_name] = {
                'enabled': stage.get('enabled', True),
                'required': stage.get('required', False),
                'depends_on': stage.get('depends_on', []),
                'config': stage
            }
    
    # Start with core stages (always included)
    execution_order = core_stages.copy()
    
    # Add config-driven stages (Time Series forecasting, signal generation, routing)
    # These stages respect enabled flag and dependencies
    config_driven_stages = []
    for stage_name, stage_info in stage_map.items():
        # Skip core stages (already added)
        if stage_name in core_stages or stage_name == 'data_storage':
            continue
            
        # Skip LLM stages (handled separately above)
        if stage_name.startswith('llm_'):
            continue
            
        # Only include if enabled
        if not stage_info.get('enabled', True):
            logger.debug(f"Skipping disabled stage: {stage_name}")
            continue
            
        config_driven_stages.append((stage_name, stage_info))
    
    # Sort config-driven stages by dependencies (topological sort)
    # Simple approach: insert stages after their dependencies
    for stage_name, stage_info in config_driven_stages:
        depends_on = stage_info.get('depends_on', [])
        
        if not depends_on:
            # No dependencies - add after data_storage
            storage_idx = execution_order.index('data_storage')
            execution_order.insert(storage_idx + 1, stage_name)
        else:
            # Has dependencies - insert after the last dependency
            max_dep_idx = -1
            for dep in depends_on:
                if dep in execution_order:
                    dep_idx = execution_order.index(dep)
                    max_dep_idx = max(max_dep_idx, dep_idx)
            
            if max_dep_idx >= 0:
                # Insert after last dependency
                execution_order.insert(max_dep_idx + 1, stage_name)
            else:
                # Dependencies not found - add at end (shouldn't happen, but safe fallback)
                logger.warning(f"Stage {stage_name} has dependencies {depends_on} not found in execution order. Adding at end.")
                execution_order.append(stage_name)
    
    # Add LLM stages after Time Series routing (Time Series-first architecture)
    if enable_llm:
        llm_stages = ['llm_market_analysis', 'llm_signal_generation', 'llm_risk_assessment']
        anchor_candidates = [
            'signal_router',
            'time_series_signal_generation',
            'time_series_forecasting',
            'data_storage',
        ]
        anchor_stage = next((stage for stage in anchor_candidates if stage in execution_order), 'data_storage')
        anchor_idx = execution_order.index(anchor_stage) + 1
        for llm_stage in llm_stages:
            execution_order.insert(anchor_idx, llm_stage)
            anchor_idx += 1

    # Remove duplicates while preserving order
    seen = set()
    unique_order = []
    for stage in execution_order:
        if stage not in seen:
            seen.add(stage)
            unique_order.append(stage)
    
    return unique_order


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
    validator_version_cfg = str(signal_validation_cfg.get('version', 'v1')).lower()
    components = LLMComponents(
        enabled=False,
        validator_version=validator_version_cfg,
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

        cache_enabled_cfg = performance_cfg.get('enable_cache', None)
        if cache_enabled_cfg is None:
            cache_enabled = bool(performance_cfg.get('track_cache_usage', True))
        else:
            cache_enabled = bool(cache_enabled_cfg)
        cache_max_size_cfg = performance_cfg.get('cache_max_size', 32)
        try:
            cache_max_size = int(cache_max_size_cfg) if cache_max_size_cfg is not None else 32
        except (TypeError, ValueError):
            cache_max_size = 32

        cache_ttl_cfg = performance_cfg.get('cache_ttl_seconds', 600)
        try:
            cache_ttl_seconds = None if cache_ttl_cfg in (None, "disabled") else int(cache_ttl_cfg)
        except (TypeError, ValueError):
            cache_ttl_seconds = 600

        latency_failover_cfg = performance_cfg.get('latency_failover_threshold', 12.0)
        try:
            latency_failover_threshold = float(latency_failover_cfg)
        except (TypeError, ValueError):
            latency_failover_threshold = 12.0

        token_rate_cfg = performance_cfg.get('token_rate_failover_threshold', 10.0)
        try:
            token_rate_failover_threshold = float(token_rate_cfg)
        except (TypeError, ValueError):
            token_rate_failover_threshold = 10.0

        model_catalog = llm_cfg.get('models', {})
        fallback_models: list[str] = []
        try:
            ranked_models = []
            for entry in model_catalog.values():
                if isinstance(entry, dict) and entry.get('name'):
                    ranked_models.append(
                        (
                            int(entry.get('priority', 999)),
                            str(entry.get('name')),
                        )
                    )
            ranked_models.sort(key=lambda x: x[0])
            fallback_models = [
                name for _, name in ranked_models if name and name != (model_to_use or llm_cfg.get('active_model'))
            ]
        except Exception:  # pragma: no cover - defensive fallback
            fallback_models = []

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
            cache_ttl_seconds=cache_ttl_seconds,
            latency_failover_threshold=latency_failover_threshold,
            token_rate_failover_threshold=token_rate_failover_threshold,
            fallback_models=fallback_models,
        )

        if not llm_client.health_check():
            logger.warning("Ollama health check failed - LLM features will be disabled")
            logger.warning("  To enable LLM features, ensure Ollama is running: 'ollama serve'")
            raise OllamaConnectionError("Ollama health check failed")

        if validator_version_cfg == 'v2':
            signal_validator = AdvancedSignalValidator(
                min_confidence=float(signal_validation_cfg.get('min_confidence_for_action', 0.55)),
                max_volatility_percentile=float(signal_validation_cfg.get('max_volatility_percentile', 0.95)),
                max_position_size=float(signal_validation_cfg.get('max_position_size', 0.02)),
                transaction_cost=float(signal_validation_cfg.get('transaction_cost', 0.001)),
            )
        else:
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

        logger.info("OK LLM initialized: %s", llm_client.model)
    except OllamaConnectionError as exc:
        logger.warning("WARN LLM initialization failed: %s", exc)
        logger.warning("  LLM features will be disabled. Pipeline will continue without LLM.")
        logger.warning("  To enable LLM features, ensure Ollama is running: 'ollama serve'")
        # Return components with enabled=False (graceful degradation)
        components.enabled = False
    except Exception as exc:  # pragma: no cover - defensive logging
        from etl.security_utils import sanitize_error
        safe_error = sanitize_error(exc)
        logger.warning("WARN LLM initialization error: %s", safe_error)
        logger.warning("  LLM features will be disabled. Pipeline will continue without LLM.")
        components.enabled = False

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
        logger.info("OK Using k-fold cross-validation (k=%d)", settings.n_splits)
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

def execute_pipeline(
    config: str = 'config/pipeline_config.yml',
    data_source: Optional[str] = None,
    tickers: str = 'AAPL,MSFT',
    start: str = '2020-01-01',
    end: str = '2024-01-01',
    use_cv: Optional[bool] = None,
    n_splits: Optional[int] = None,
    test_size: Optional[float] = None,
    gap: Optional[int] = None,
    verbose: bool = False,
    enable_llm: bool = False,
    llm_model: str = '',
    dry_run: bool = False,
    execution_mode: str = 'auto',
    use_ticker_discovery: bool = False,
    refresh_ticker_universe: bool = False,
    include_frontier_tickers: bool = False,
    db_path: Optional[str] = None,
    synthetic_dataset_id: Optional[str] = None,
    synthetic_dataset_path: Optional[str] = None,
    synthetic_config: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    prefer_gpu: bool = True,
) -> None:
    """Execute ETL pipeline with modular configuration-driven orchestration.
    
    This is the core pipeline execution function that can be called directly
    from tests or other Python code. The Click command wrapper converts CLI
    arguments and calls this function.
    
    Args:
        config: Path to pipeline configuration file
        data_source: Data source to use (yfinance, alpha_vantage, finnhub)
        tickers: Comma-separated ticker symbols
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        use_cv: Use k-fold cross-validation (None = use config default)
        n_splits: Number of CV folds (None = use config value)
        test_size: Test set size 0.0-1.0 (None = use config value)
        gap: Gap between train/validation periods (None = use config value)
        verbose: Enable verbose logging (DEBUG level)
        enable_llm: Enable LLM integration
        llm_model: LLM model override (empty = use config active_model)
        dry_run: Generate synthetic data (no network)
        execution_mode: Data extraction mode (auto, live, synthetic)
        use_ticker_discovery: Load tickers from discovery universe
        refresh_ticker_universe: Force refresh of ticker universe
        include_frontier_tickers: Append curated frontier markets to multi-ticker runs
        logger_instance: Optional logger instance (for testing)
    """
    # Use provided logger or setup new one (only when run as main)
    if logger_instance is None:
        logger = _setup_logging(verbose=verbose)
    else:
        logger = logger_instance
    if verbose:
        logger.debug("Verbose logging enabled")

    device = _detect_device(prefer_gpu=prefer_gpu)
    os.environ["PIPELINE_DEVICE"] = device
    logger.info("Device selection: %s (prefer_gpu=%s)", device, prefer_gpu)

    # Apply synthetic overrides (env bridge)
    if synthetic_dataset_id:
        os.environ["SYNTHETIC_DATASET_ID"] = synthetic_dataset_id
    if synthetic_dataset_path:
        os.environ["SYNTHETIC_DATASET_PATH"] = synthetic_dataset_path
    if synthetic_config:
        os.environ["SYNTHETIC_CONFIG_PATH"] = synthetic_config

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

    if include_frontier_tickers:
        original_count = len(ticker_list)
        ticker_list = merge_frontier_tickers(ticker_list, include_frontier=True)
        appended = len(ticker_list) - original_count
        regions = ", ".join(FRONTIER_MARKET_TICKERS_BY_REGION.keys())
        logger.info(
            "Frontier market coverage enabled (+%d tickers across %s)",
            appended,
            regions,
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
    llm_signal_generator = llm_components.signal_generator
    risk_assessor = llm_components.risk_assessor
    signal_validator = llm_components.signal_validator

    # Initialize storage
    storage = DataStorage()

    # Initialize database for persistent storage
    # Allow overrides for synthetic/brutal runs so they don't contend with the primary DB.
    resolved_db_path = db_path or os.environ.get("PORTFOLIO_DB_PATH")
    if not resolved_db_path:
        if execution_mode.lower() == "synthetic":
            resolved_db_path = "data/test_database.db"
        else:
            resolved_db_path = "data/portfolio_maximizer.db"
    db_manager = DatabaseManager(db_path=resolved_db_path)
    signal_tracker = LLMSignalTracker()
    logger.info("OK Database manager initialized")

    # Initialize checkpoint manager and pipeline logger
    pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_manager = CheckpointManager(checkpoint_dir="data/checkpoints")
    pipeline_log = PipelineLogger(log_dir="logs", retention_days=7)

    logger.info(f"OK Pipeline ID: {pipeline_id}")
    pipeline_log.log_event('pipeline_start', pipeline_id, metadata={
        'tickers': ticker_list,
        'start_date': start,
        'end_date': end,
        'use_cv': use_cv,
        'execution_mode': execution_mode,
        'synthetic_dataset_env': os.getenv("SYNTHETIC_DATASET_ID") or os.getenv("SYNTHETIC_DATASET_PATH"),
    })

    # Initialize data source manager (platform-agnostic)
    try:
        data_source_manager = DataSourceManager(
            config_path='config/data_sources_config.yml',
            storage=storage
        )
        logger.info(f"OK Data source manager initialized")
        logger.info(f"  Available sources: {', '.join(data_source_manager.get_available_sources())}")
        logger.info(f"  Active source: {data_source_manager.get_active_source()}")
    except Exception as e:
        logger.error(f"Failed to initialize data source manager: {e}")
        logger.error("Pipeline cannot continue without data source")
        raise

    # Determine split strategy and log configuration
    _log_split_strategy(cv_settings)

    # Build stage execution order from config (config-driven orchestration)
    stage_names = _build_stage_execution_order(
        stages_cfg=stages_cfg,
        enable_llm=enable_llm,
        logger=logger
    )
    
    logger.info(f"OK Stage execution order (config-driven): {', '.join(stage_names)}")

    # Prepare synthetic data up-front when explicitly requested
    precomputed_raw_data: Optional[pd.DataFrame] = None
    synthetic_dataset_id: Optional[str] = None
    synthetic_generator_version: Optional[str] = None
    dataset_id_current: Optional[str] = None
    generator_version_current: Optional[str] = None
    if execution_mode == 'synthetic':
        syn_dataset_path = os.getenv("SYNTHETIC_DATASET_PATH")
        syn_dataset_id = os.getenv("SYNTHETIC_DATASET_ID")
        if syn_dataset_path or syn_dataset_id:
            try:
                from etl.synthetic_extractor import SyntheticExtractor

                syn_extractor = SyntheticExtractor()
                precomputed_raw_data = syn_extractor.extract_ohlcv(ticker_list, start, end)
                synthetic_dataset_id = precomputed_raw_data.attrs.get("dataset_id")
                synthetic_generator_version = precomputed_raw_data.attrs.get("generator_version")
                dataset_id_current = synthetic_dataset_id
                generator_version_current = synthetic_generator_version
                logger.info("Synthetic mode: loaded persisted synthetic dataset (%s)", synthetic_dataset_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load persisted synthetic dataset (%s); falling back to generator", exc)
                precomputed_raw_data = None
        if precomputed_raw_data is None:
            logger.info("Synthetic mode: precomputing deterministic OHLCV data (offline)")
            precomputed_raw_data = generate_synthetic_ohlcv(ticker_list, start, end)
            synthetic_dataset_id = precomputed_raw_data.attrs.get("dataset_id")
            synthetic_generator_version = precomputed_raw_data.attrs.get("generator_version")
            dataset_id_current = synthetic_dataset_id
            generator_version_current = synthetic_generator_version

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
                        # Use data source manager for extraction (automatic failover support).
                        # If a specific --data-source is provided, prefer that adapter;
                        # otherwise, defer to the manager's configured primary (now cTrader
                        # for live/auto runs, with yfinance/others as fallbacks).
                        prefer_source = data_source or None
                        raw_data = data_source_manager.extract_ohlcv(
                            tickers=ticker_list,
                            start_date=start,
                            end_date=end,
                            prefer_source=prefer_source,
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

                logger.info(f"OK Extracted {len(raw_data)} rows from {len(ticker_list)} ticker(s)")
                logger.info(f"  Source: {extraction_source}")
                dataset_id = raw_data.attrs.get("dataset_id")
                generator_version = raw_data.attrs.get("generator_version")
                if dataset_id:
                    logger.info("  Synthetic dataset_id: %s (generator_version=%s)", dataset_id, generator_version or "n/a")
                    dataset_id_current = dataset_id
                    generator_version_current = generator_version
                pipeline_log.log_event(
                    'data_extraction',
                    pipeline_id,
                    metadata={
                        "source": extraction_source,
                        "rows": len(raw_data),
                        "tickers": ticker_list,
                        "dataset_id": dataset_id,
                        "generator_version": generator_version,
                        "execution_mode": execution_mode,
                    },
                )

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
                logger.info("Saved {rows_saved} rows to database")

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
                    metadata={
                        'tickers': ticker_list,
                        'rows': len(raw_data),
                        'dataset_id': dataset_id_current,
                        'generator_version': generator_version_current,
                    }
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'data_validation':
                # Stage 2: Data Validation
                logger.info("Validating data quality...")
                validator = DataValidator()
                report = validator.validate_ohlcv(raw_data)

                if not report['passed']:
                    logger.warning(f"WARN Validation warnings detected")
                    logger.warning(f"  Errors: {len(report.get('errors', []))}")
                    logger.warning(f"  Warnings: {len(report.get('warnings', []))}")
                    if verbose:
                        logger.debug(f"Validation report: {report}")
                else:
                    logger.info("OK Data validation passed")

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
                logger.debug(f"  Normalization complete (mu=0, sigma^2=1)")

                # Save processed data with run metadata
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                storage.save(
                    processed, 
                    'processed', 
                    f'processed_{timestamp}',
                    metadata={
                        'data_source': extraction_source,
                        'execution_mode': execution_mode,
                        'pipeline_id': pipeline_id,
                    },
                    run_id=pipeline_id
                )
                logger.info(f"OK Preprocessed {len(processed)} rows")

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
                        
                        logger.info(f"  OK {ticker}: Trend={analysis['trend']}, Strength={analysis['strength']}/10 ({latency:.1f}s)")
                        if verbose:
                            logger.debug(f"    Summary: {analysis['summary']}")
                    except Exception as e:
                        logger.warning(f"  WARN {ticker} analysis failed: {e}")
                        llm_analyses[ticker] = {'trend': 'neutral', 'strength': 5, 'error': str(e)}
                
                logger.info(f"OK Analyzed {len(llm_analyses)} ticker(s) with LLM")
                
                # Save checkpoint with LLM analysis
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=processed,
                    metadata={
                        'analyses': llm_analyses,
                        'tickers': ticker_list,
                        'dataset_id': dataset_id_current,
                        'generator_version': generator_version_current,
                    }
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'llm_signal_generation':
                # Stage 5 (LLM): Signal Generation
                logger.info("Generating trading signals with LLM...")
                llm_signals = {}
                llm_signal_validations = {}
                portfolio_notional = 10000.0
                signal_log_date = datetime.now().strftime('%Y-%m-%d')
                if signal_validation_cfg:
                    notional_cfg = signal_validation_cfg.get('portfolio_notional', signal_validation_cfg.get('portfolio_value', 10000.0))
                    try:
                        portfolio_notional = float(notional_cfg)
                    except (TypeError, ValueError):
                        portfolio_notional = 10000.0
                
                for ticker in ticker_list:
                    try:
                        # Get ticker-specific data and analysis
                        ticker_data = processed.xs(ticker, level=0) if isinstance(processed.index, pd.MultiIndex) else processed
                        market_analysis = llm_analyses.get(ticker, {})
                        
                        # Generate signal with timing
                        start_time = time.time()
                        if llm_signal_generator is None:
                            raise RuntimeError("LLM signal generator unavailable; check LLM initialization")
                        signal = llm_signal_generator.generate_signal(ticker_data, ticker, market_analysis)
                        latency = time.time() - start_time
                        validation_status = 'pending'
                        validation_entry: Dict[str, Any] = {}

                        signal_timestamp_raw = signal.get('signal_timestamp')
                        try:
                            if signal_timestamp_raw:
                                signal_timestamp_dt = datetime.fromisoformat(
                                    str(signal_timestamp_raw).replace('Z', '+00:00')
                                )
                            else:
                                signal_timestamp_dt = datetime.utcnow()
                        except ValueError:
                            signal_timestamp_dt = datetime.utcnow()

                        ticker_raw = _slice_ticker_dataframe(raw_data, ticker)
                        validation_data = _prepare_validation_frame(ticker_raw)
                        price_at_signal = 0.0
                        if not validation_data.empty:
                            price_column = 'Close' if 'Close' in validation_data.columns else 'close'
                            if price_column in validation_data.columns:
                                try:
                                    price_at_signal = float(validation_data[price_column].iloc[-1])
                                except Exception:
                                    price_at_signal = 0.0

                        signal['entry_price'] = price_at_signal
                        forward_horizon = int(signal_validation_cfg.get('forward_return_days', 5))
                        actual_return = None
                        if (
                            forward_horizon > 0
                            and not validation_data.empty
                            and 'Close' in validation_data.columns
                        ):
                            actual_return = _compute_forward_return(
                                validation_data['Close'],
                                signal_timestamp_dt,
                                horizon_days=forward_horizon,
                            )
                        signal['actual_return'] = actual_return

                        if signal_validator is not None:
                            try:
                                if validator_version == 'v2' and hasattr(signal_validator, 'validate_llm_signal'):
                                    if validation_data.empty or 'Close' not in validation_data.columns:
                                        raise ValueError("insufficient market data for advanced validation")

                                    validation_result = signal_validator.validate_llm_signal(
                                        signal,
                                        validation_data,
                                        portfolio_value=portfolio_notional,
                                    )
                                    validation_status = 'validated' if validation_result.is_valid else 'failed'
                                    validation_entry = {
                                        'validator_version': validator_version,
                                        'confidence_score': float(validation_result.confidence_score),
                                        'recommendation': validation_result.recommendation,
                                        'warnings': validation_result.warnings,
                                        'quality_metrics': getattr(validation_result, 'layer_results', {}),
                                    }

                                    if (not validation_result.is_valid
                                            or validation_result.recommendation.upper() in {'HOLD', 'REJECT'}):
                                        if signal.get('action') != 'HOLD':
                                            logger.debug("Advanced validator adjusted %s signal to HOLD", ticker)
                                            signal['action'] = 'HOLD'
                                        if validation_result.warnings:
                                            existing_reasoning = signal.get('reasoning') or ''
                                            signal['reasoning'] = (
                                                f"{existing_reasoning} [Validator: {validation_result.warnings[0]}]"
                                            ).strip()
                                else:
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

                        tracker_signal_id = None
                        if signal_tracker:
                            tracker_signal_id = signal_tracker.register_signal(
                                ticker,
                                signal_log_date,
                                {
                                    'action': signal.get('action', 'HOLD'),
                                    'confidence': float(signal.get('confidence', 0.0)),
                                    'reasoning': signal.get('reasoning', '')
                                }
                            )
                        
                        llm_signals[ticker] = signal
                        if validation_entry:
                            llm_signal_validations[ticker] = validation_entry
                        
                        # Save to database
                        signal_id = db_manager.save_llm_signal(
                            ticker=ticker,
                            date=signal_log_date,
                            signal=signal,
                            model_name=llm_client.model,
                            latency=latency,
                            validation_status=validation_status
                        )
                        if signal_id != -1 and validation_entry:
                            db_manager.save_signal_validation(signal_id, validation_entry)

                        lookback_days = int(signal_validation_cfg.get('backtest_lookback_days', 30))
                        history_limit = int(signal_validation_cfg.get('backtest_signal_limit', 250))
                        if (
                            signal_id > 0
                            and signal_validator is not None
                            and hasattr(signal_validator, 'backtest_signal_quality')
                        ):
                            try:
                                recent_rows = db_manager.fetch_recent_signals(
                                    ticker,
                                    reference_timestamp=signal_timestamp_dt,
                                    lookback_days=lookback_days,
                                    limit=history_limit,
                                )
                                recent_signals = _format_signals_for_backtest(recent_rows)
                                if recent_signals:
                                    report = signal_validator.backtest_signal_quality(
                                        signals=recent_signals,
                                        actual_prices=validation_data[['Close']],
                                        lookback_days=lookback_days,
                                    )
                                    db_manager.update_signal_performance(
                                        signal_id,
                                        {
                                            'actual_return': actual_return,
                                            'annual_return': getattr(report, 'annual_return', None),
                                            'sharpe_ratio': report.sharpe_ratio,
                                            'information_ratio': report.information_ratio,
                                            'hit_rate': report.hit_rate,
                                            'profit_factor': report.profit_factor,
                                        },
                                    )
                                    db_manager.save_signal_backtest_summary(ticker, lookback_days, report)
                                    if verbose:
                                        logger.debug(
                                            "    Backtest metrics | hit_rate=%.2f profit_factor=%.2f sharpe=%.2f",
                                            report.hit_rate,
                                            report.profit_factor,
                                            report.sharpe_ratio,
                                        )
                                elif actual_return is not None:
                                    db_manager.update_signal_performance(
                                        signal_id, {'actual_return': actual_return}
                                    )
                            except Exception as metrics_exc:
                                logger.warning(
                                    "  WARN %s: Unable to compute signal backtest metrics (%s)",
                                    ticker,
                                    metrics_exc,
                                )
                        elif signal_id > 0 and actual_return is not None:
                            db_manager.update_signal_performance(
                                signal_id, {'actual_return': actual_return}
                            )

                        if signal_tracker and tracker_signal_id:
                            tracker_payload = validation_entry or {
                                'validator_version': validator_version,
                                'confidence_score': float(signal.get('confidence', 0.0)),
                                'recommendation': validation_status,
                                'warnings': [],
                                'quality_metrics': {},
                            }
                            tracker_payload['latency_seconds'] = latency
                            signal_tracker.record_validator_result(
                                tracker_signal_id,
                                tracker_payload,
                                validation_status
                            )
                        
                        confidence_pct = float(signal.get('confidence', 0.0)) * 100
                        logger.info(
                            "  OK %s: Action=%s, Confidence=%.1f%% (%0.1fs) Validation=%s",
                            ticker,
                            signal['action'],
                            confidence_pct,
                            latency,
                            validation_status.upper(),
                        )
                        if verbose:
                            logger.debug(f"    Reasoning: {signal['reasoning']}")
                    except Exception as e:
                        logger.warning(f"  WARN {ticker} signal generation failed: {e}")
                        llm_signals[ticker] = {'action': 'HOLD', 'confidence': 0.5, 'error': str(e)}
                
                logger.info(f"OK Generated {len(llm_signals)} signal(s) with LLM")
                logger.warning("WARN ADVISORY ONLY: LLM signals require 30-day validation before trading")
                
                # Save checkpoint with signals
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=processed,
                    metadata={
                        'signals': llm_signals,
                        'analyses': llm_analyses,
                        'validations': llm_signal_validations,
                        'dataset_id': dataset_id_current,
                        'generator_version': generator_version_current,
                    }
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)
                if signal_tracker:
                    signal_tracker.flush()

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
                        
                        logger.info(f"  OK {ticker}: Risk Level={risk['risk_level']}, Score={risk['risk_score']}/100 ({latency:.1f}s)")
                        if verbose and risk.get('concerns'):
                            logger.debug(f"    Concerns: {', '.join(risk['concerns'][:2])}")
                    except Exception as e:
                        logger.warning(f"  WARN {ticker} risk assessment failed: {e}")
                        llm_risks[ticker] = {'risk_level': 'medium', 'risk_score': 50, 'error': str(e)}
                
                logger.info(f"OK Assessed {len(llm_risks)} ticker(s) risk with LLM")
                
                # Save checkpoint with risk assessment
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    stage=stage_name,
                    data=processed,
                    metadata={
                        'risks': llm_risks,
                        'signals': llm_signals,
                        'analyses': llm_analyses,
                        'dataset_id': dataset_id_current,
                        'generator_version': generator_version_current,
                    }
                )
                pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)

            elif stage_name == 'time_series_forecasting':
                # Stage 7: Time Series Forecasting (SARIMAX/GARCH)
                logger.info("Generating time series forecasts...")
                
                try:
                    from etl.time_series_forecaster import (
                        TimeSeriesForecaster,
                        TimeSeriesForecasterConfig,
                        RollingWindowValidator,
                        RollingWindowCVConfig,
                    )
                    # Load forecasting config
                    forecasting_cfg = pipeline_cfg.get('forecasting', {})
                    if not forecasting_cfg.get('enabled', True):
                        logger.info("Forecasting disabled in config, skipping...")
                        continue
                    
                    forecast_horizon = forecasting_cfg.get('default_forecast_horizon', 30)
                    min_history_required = int(forecasting_cfg.get('minimum_history_required', 90))
                    min_history_strict = int(forecasting_cfg.get('minimum_history_strict', 30))
                    sarimax_cfg = forecasting_cfg.get('sarimax', {})
                    garch_cfg = forecasting_cfg.get('garch', {})
                    samossa_cfg = forecasting_cfg.get('samossa', {})
                    mssa_rl_cfg = forecasting_cfg.get('mssa_rl', {})
                    ensemble_cfg = forecasting_cfg.get('ensemble', {})
                    rolling_cv_cfg = forecasting_cfg.get('rolling_cv', {})
                    
                    forecasts = {}

                    def _build_model_config(target_horizon: int) -> TimeSeriesForecasterConfig:
                        return TimeSeriesForecasterConfig(
                            forecast_horizon=int(target_horizon),
                            sarimax_enabled=sarimax_cfg.get('enabled', True),
                            garch_enabled=garch_cfg.get('enabled', True),
                            samossa_enabled=samossa_cfg.get('enabled', False),
                            mssa_rl_enabled=mssa_rl_cfg.get('enabled', True),
                            ensemble_enabled=ensemble_cfg.get('enabled', True),
                            sarimax_kwargs={
                                k: v for k, v in sarimax_cfg.items() if k != 'enabled'
                            },
                            garch_kwargs={
                                k: v for k, v in garch_cfg.items() if k != 'enabled'
                            },
                            samossa_kwargs={
                                k: v for k, v in samossa_cfg.items() if k != 'enabled'
                            },
                            mssa_rl_kwargs={
                                k: v for k, v in mssa_rl_cfg.items() if k != 'enabled'
                            },
                            ensemble_kwargs={
                                k: v for k, v in ensemble_cfg.items() if k != 'enabled'
                            },
                        )

                    for ticker in ticker_list:
                        try:
                            # Get ticker-specific data
                            if isinstance(processed.index, pd.MultiIndex):
                                ticker_data = processed.xs(ticker, level=0)
                            else:
                                ticker_data = processed
                                if 'ticker' in ticker_data.columns:
                                    ticker_mask = ticker_data['ticker'].astype(str).str.upper() == ticker.upper()
                                    ticker_data = ticker_data.loc[ticker_mask]

                            if ticker_data.empty:
                                logger.warning(f"  WARN {ticker}: No processed rows available after filtering")
                                continue
                            
                            # Extract Close price series
                            if 'Close' in ticker_data.columns:
                                price_series = ticker_data['Close'].dropna()
                            elif 'close' in ticker_data.columns:
                                price_series = ticker_data['close'].dropna()
                            else:
                                logger.warning(f"  WARN {ticker}: No Close price data available")
                                continue
                            
                            series_length = len(price_series)
                            if series_length < min_history_strict:
                                logger.warning(
                                    "  WARN %s: Insufficient data for forecasting (need >= %s, have %s)",
                                    ticker,
                                    min_history_strict,
                                    series_length,
                                )
                                continue

                            train_series = price_series
                            holdout_series = None

                            if series_length < min_history_required:
                                logger.warning(
                                    "  WARN %s: Limited history (<%s observations). Using entire series; metrics disabled.",
                                    ticker,
                                    min_history_required,
                                )
                            elif series_length >= forecast_horizon * 2:
                                train_series = price_series.iloc[:-forecast_horizon]
                                holdout_series = price_series.iloc[-forecast_horizon:]
                            else:
                                logger.warning(
                                    "  WARN %s: Not enough history for walk-forward validation (required >= %s observations). "
                                    "Using entire series for training; metrics will be skipped.",
                                    ticker,
                                    forecast_horizon * 2,
                                )

                            forecaster = TimeSeriesForecaster(
                                config=_build_model_config(forecast_horizon)
                            )

                            train_returns = train_series.pct_change().dropna()
                            forecaster.fit(train_series, returns_series=train_returns)

                            forecast_result = forecaster.forecast(
                                steps=forecast_horizon,
                                alpha=0.05,
                            )

                            metrics_map: Dict[str, Dict[str, float]] = {}
                            if holdout_series is not None and len(holdout_series.dropna()) >= 2:
                                try:
                                    metrics_map = forecaster.evaluate(holdout_series)
                                except Exception as exc:  # pragma: no cover - metrics optional
                                    logger.warning("  WARN %s: Unable to compute regression metrics: %s", ticker, exc)
                            forecast_result["regression_metrics"] = metrics_map

                            cv_results = None
                            if rolling_cv_cfg.get('enabled', False):
                                try:
                                    cv_min_train = int(rolling_cv_cfg.get('min_train_size', min_history_required))
                                    cv_horizon = int(rolling_cv_cfg.get('horizon', max(1, min(forecast_horizon, 5))))
                                    cv_step = int(rolling_cv_cfg.get('step_size', cv_horizon))
                                    max_folds_raw = rolling_cv_cfg.get('max_folds')
                                    cv_max_folds = int(max_folds_raw) if max_folds_raw not in (None, "", False) else None
                                    if series_length >= cv_min_train + cv_horizon:
                                        cv_validator = RollingWindowValidator(
                                            forecaster_config=_build_model_config(cv_horizon),
                                            cv_config=RollingWindowCVConfig(
                                                min_train_size=cv_min_train,
                                                horizon=cv_horizon,
                                                step_size=max(1, cv_step),
                                                max_folds=cv_max_folds,
                                            ),
                                        )
                                        returns_for_cv = price_series.pct_change().dropna()
                                        cv_results = cv_validator.run(
                                            price_series=price_series,
                                            returns_series=returns_for_cv,
                                        )
                                        aggregate = cv_results.get("aggregate_metrics", {})
                                        sarimax_metrics = aggregate.get("sarimax", {})
                                        rmse_val = sarimax_metrics.get("rmse")
                                        logger.info(
                                            "  -> %s rolling CV (%s folds, horizon=%s, sarimax_rmse=%s)",
                                            ticker,
                                            cv_results.get("fold_count"),
                                            cv_results.get("horizon"),
                                            f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else "n/a",
                                        )
                                    else:
                                        logger.info(
                                            "  -> %s rolling CV skipped (need >= %s observations, have %s)",
                                            ticker,
                                            cv_min_train + cv_horizon,
                                            series_length,
                                        )
                                except Exception as cv_exc:
                                    logger.warning("  WARN %s: Rolling CV failed (%s)", ticker, cv_exc)

                            if cv_results:
                                forecast_result["cross_validation"] = cv_results

                            forecasts[ticker] = forecast_result

                            # Save forecasts to database
                            forecast_date = datetime.now().strftime('%Y-%m-%d')

                            # Save SARIMAX forecast if available
                            if forecast_result.get('sarimax_forecast'):
                                sarimax_result = forecast_result['sarimax_forecast']
                                forecast_series = sarimax_result.get('forecast', pd.Series())
                                for step in range(min(forecast_horizon, len(forecast_series))):
                                    if step < len(forecast_series):
                                        forecast_data = {
                                            'model_type': 'SARIMAX',
                                            'forecast_horizon': step + 1,
                                            'forecast_value': float(forecast_series.iloc[step]),
                                            'lower_ci': float(sarimax_result['lower_ci'].iloc[step]) if 'lower_ci' in sarimax_result and step < len(sarimax_result['lower_ci']) else None,
                                            'upper_ci': float(sarimax_result['upper_ci'].iloc[step]) if 'upper_ci' in sarimax_result and step < len(sarimax_result['upper_ci']) else None,
                                            'model_order': sarimax_result.get('model_order'),
                                            'aic': sarimax_result.get('aic'),
                                            'bic': sarimax_result.get('bic'),
                                            'diagnostics': sarimax_result.get('diagnostics'),
                                            'regression_metrics': metrics_map.get('sarimax'),
                                        }
                                        db_manager.save_forecast(
                                            ticker=ticker,
                                            forecast_date=forecast_date,
                                            forecast_data=forecast_data,
                                        )
                            
                            # Save GARCH forecast if available
                            if forecast_result.get('volatility_forecast'):
                                garch_result = forecast_result['volatility_forecast']
                                vol_series = garch_result.get('volatility', pd.Series())
                                for step in range(min(forecast_horizon, len(vol_series))):
                                    if step < len(vol_series):
                                        forecast_data = {
                                            'model_type': 'GARCH',
                                            'forecast_horizon': step + 1,
                                            'forecast_value': float(vol_series.iloc[step]),
                                            'volatility': float(vol_series.iloc[step]),
                                            'model_order': garch_result.get('model_order'),
                                            'aic': garch_result.get('aic'),
                                            'bic': garch_result.get('bic'),
                                            'regression_metrics': metrics_map.get('garch'),
                                        }
                                        db_manager.save_forecast(
                                            ticker=ticker,
                                            forecast_date=forecast_date,
                                            forecast_data=forecast_data,
                                        )
                            
                            # Save SAMOSSA forecast if available
                            if forecast_result.get('samossa_forecast'):
                                samossa_result = forecast_result['samossa_forecast']
                                samossa_series = samossa_result.get('forecast', pd.Series())
                                lower_ci = samossa_result.get('lower_ci')
                                upper_ci = samossa_result.get('upper_ci')
                                for step in range(min(forecast_horizon, len(samossa_series))):
                                    if step < len(samossa_series):
                                        forecast_data = {
                                            'model_type': 'SAMOSSA',
                                            'forecast_horizon': step + 1,
                                            'forecast_value': float(samossa_series.iloc[step]),
                                            'lower_ci': float(lower_ci.iloc[step]) if isinstance(lower_ci, pd.Series) and step < len(lower_ci) else None,
                                            'upper_ci': float(upper_ci.iloc[step]) if isinstance(upper_ci, pd.Series) and step < len(upper_ci) else None,
                                            'model_order': {
                                                'window_length': samossa_result.get('window_length'),
                                                'n_components': samossa_result.get('n_components'),
                                            },
                                            'diagnostics': {
                                                'explained_variance_ratio': samossa_result.get('explained_variance_ratio'),
                                            },
                                            'regression_metrics': metrics_map.get('samossa'),
                                        }
                                        db_manager.save_forecast(
                                            ticker=ticker,
                                            forecast_date=forecast_date,
                                            forecast_data=forecast_data,
                                        )

                            if forecast_result.get('mssa_rl_forecast'):
                                mssa_result = forecast_result['mssa_rl_forecast']
                                mssa_series = mssa_result.get('forecast', pd.Series())
                                lower_ci = mssa_result.get('lower_ci')
                                upper_ci = mssa_result.get('upper_ci')
                                change_points = _normalize_change_points(
                                    mssa_result.get('change_points'),
                                )
                                diagnostics = {
                                    'baseline_variance': mssa_result.get('baseline_variance'),
                                    'q_table_size': mssa_result.get('q_table_size'),
                                    'change_points': change_points,
                                }
                                for step in range(min(forecast_horizon, len(mssa_series))):
                                    if step < len(mssa_series):
                                        forecast_data = {
                                            'model_type': 'MSSA_RL',
                                            'forecast_horizon': step + 1,
                                            'forecast_value': float(mssa_series.iloc[step]),
                                            'lower_ci': float(lower_ci.iloc[step]) if isinstance(lower_ci, pd.Series) and step < len(lower_ci) else None,
                                            'upper_ci': float(upper_ci.iloc[step]) if isinstance(upper_ci, pd.Series) and step < len(upper_ci) else None,
                                            'diagnostics': diagnostics,
                                            'regression_metrics': metrics_map.get('mssa_rl'),
                                        }
                                        db_manager.save_forecast(
                                            ticker=ticker,
                                            forecast_date=forecast_date,
                                            forecast_data=forecast_data,
                                        )

                            if forecast_result.get('ensemble_forecast'):
                                ensemble_result = forecast_result['ensemble_forecast']
                                ensemble_series = ensemble_result.get('forecast', pd.Series())
                                lower_ci = ensemble_result.get('lower_ci')
                                upper_ci = ensemble_result.get('upper_ci')
                                diagnostics = {
                                    'weights': ensemble_result.get('weights'),
                                    'confidence': ensemble_result.get('confidence'),
                                    'selection_score': ensemble_result.get('selection_score'),
                                    'regression_metrics': metrics_map.get('ensemble'),
                                }
                                for step in range(min(forecast_horizon, len(ensemble_series))):
                                    if step < len(ensemble_series):
                                        forecast_data = {
                                            'model_type': 'COMBINED',
                                            'forecast_horizon': step + 1,
                                            'forecast_value': float(ensemble_series.iloc[step]),
                                            'lower_ci': float(lower_ci.iloc[step]) if isinstance(lower_ci, pd.Series) and step < len(lower_ci) else None,
                                            'upper_ci': float(upper_ci.iloc[step]) if isinstance(upper_ci, pd.Series) and step < len(upper_ci) else None,
                                            'diagnostics': diagnostics,
                                            'regression_metrics': metrics_map.get('ensemble'),
                                        }
                                        db_manager.save_forecast(
                                            ticker=ticker,
                                            forecast_date=forecast_date,
                                            forecast_data=forecast_data,
                                        )
                            
                            logger.info(f"  OK {ticker}: Generated {forecast_horizon}-step forecast")
                            
                        except Exception as e:
                            logger.warning(f"  WARN {ticker} forecasting failed: {e}")
                            forecasts[ticker] = {'error': str(e)}
                    
                    successful_forecasts = len([f for f in forecasts.values() if 'error' not in f])
                    logger.info(f"OK Generated forecasts for {successful_forecasts}/{len(ticker_list)} ticker(s)")
                    if successful_forecasts < len(ticker_list):
                        missing = [t for t in ticker_list if t not in forecasts or 'error' in forecasts[t]]
                        logger.warning("  WARN Forecasting skipped for: %s", ", ".join(missing))

                    try:
                        _generate_visual_dashboards(pipeline_cfg, db_manager, ticker_list)
                    except Exception as viz_exc:  # pragma: no cover - visualization optional
                        logger.warning("Time Series visualization generation failed: %s", viz_exc)
                    
                    # Save checkpoint
                    checkpoint_id = checkpoint_manager.save_checkpoint(
                        pipeline_id=pipeline_id,
                        stage=stage_name,
                        data=processed,
                        metadata={
                            'forecasts': forecasts,
                            'dataset_id': dataset_id_current,
                            'generator_version': generator_version_current,
                        }
                    )
                    pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)
                    
                    # Store forecasts for signal generation stage
                    globals()['_ts_forecasts'] = forecasts or {}
                        
                except Exception as e:
                    from etl.security_utils import sanitize_error
                    safe_error = sanitize_error(e)
                    logger.error(f"Time Series forecasting stage failed: {safe_error}")

            elif stage_name == 'time_series_signal_generation':
                # Stage 8: Time Series Signal Generation (NEW - DEFAULT SIGNAL SOURCE)
                logger.info("Generating trading signals from Time Series forecasts...")
                
                try:
                    from models.time_series_signal_generator import TimeSeriesSignalGenerator
                    from models.signal_adapter import SignalAdapter
                    
                    # Load signal routing config (pipeline override or shared YAML)
                    signal_routing_cfg = pipeline_cfg.get('signal_routing', {}) or {}
                    ts_signal_cfg = signal_routing_cfg.get('time_series', {}) or {}
                    if not ts_signal_cfg:
                        cfg_path = Path("config") / "signal_routing_config.yml"
                        if cfg_path.exists():
                            try:
                                with cfg_path.open("r", encoding="utf-8") as handle:
                                    raw = yaml.safe_load(handle) or {}
                                shared_cfg = raw.get("signal_routing") or {}
                                ts_signal_cfg = shared_cfg.get("time_series", {}) or {}
                            except Exception as exc:  # pragma: no cover - defensive
                                logger.warning("Failed to load signal routing config: %s", exc)
                    
                    # Initialize signal generator
                    ts_signal_generator = TimeSeriesSignalGenerator(
                        confidence_threshold=float(ts_signal_cfg.get('confidence_threshold', 0.55)),
                        min_expected_return=float(ts_signal_cfg.get('min_expected_return', 0.003)),
                        max_risk_score=float(ts_signal_cfg.get('max_risk_score', 0.7)),
                        use_volatility_filter=bool(ts_signal_cfg.get('use_volatility_filter', True)),
                    )
                    
                    # Get forecasts from previous stage
                    forecasts = globals().get('_ts_forecasts', {})
                    if not forecasts:
                        logger.warning("No Time Series forecasts available, skipping signal generation")
                        continue
                    
                    ts_signals = {}
                    current_prices = {}
                    
                    for ticker in ticker_list:
                        try:
                            # Get ticker-specific data for current price
                            if isinstance(processed.index, pd.MultiIndex):
                                ticker_data = processed.xs(ticker, level=0)
                            else:
                                ticker_data = processed
                            
                            # Extract current price
                            if 'Close' in ticker_data.columns:
                                current_price = float(ticker_data['Close'].iloc[-1])
                            elif 'close' in ticker_data.columns:
                                current_price = float(ticker_data['close'].iloc[-1])
                            else:
                                logger.warning(f"  WARN {ticker}: No Close price data available")
                                continue
                            
                            current_prices[ticker] = current_price
                            
                            # Get forecast bundle
                            forecast_bundle = forecasts.get(ticker)
                            if not forecast_bundle or 'error' in forecast_bundle:
                                logger.warning(f"  WARN {ticker}: No valid forecast available")
                                continue
                            
                            # Generate signal
                            signal = ts_signal_generator.generate_signal(
                                forecast_bundle=forecast_bundle,
                                current_price=current_price,
                                ticker=ticker,
                                market_data=ticker_data
                            )
                            
                            # Convert to unified format
                            unified_signal = SignalAdapter.from_time_series_signal(signal)
                            signal_dict = SignalAdapter.to_legacy_dict(unified_signal)
                            ts_signals[ticker] = signal_dict
                            
                            # Save to unified trading_signals table
                            signal_date = datetime.now().strftime('%Y-%m-%d')
                            signal_id = db_manager.save_trading_signal(
                                ticker=ticker,
                                date=signal_date,
                                signal=signal_dict,
                                source='TIME_SERIES',
                                model_type=signal.model_type,
                                validation_status='pending',
                                latency=0.0  # Time Series signals are fast
                            )
                            
                            logger.info(
                                f"  OK {ticker}: {signal.action} signal "
                                f"(confidence={signal.confidence:.2f}, "
                                f"expected_return={signal.expected_return:.2%}, "
                                f"risk={signal.risk_score:.2f})"
                            )
                            
                        except Exception as e:
                            logger.warning(f"  WARN {ticker} Time Series signal generation failed: {e}")
                            ts_signals[ticker] = {'action': 'HOLD', 'confidence': 0.0, 'error': str(e)}
                    
                    logger.info(f"OK Generated {len([s for s in ts_signals.values() if s.get('action') != 'HOLD'])} Time Series signal(s)")
                    
                    # Store signals for routing stage
                    globals()['_ts_signals'] = ts_signals
                    globals()['_current_prices'] = current_prices
                    
                    # Save checkpoint
                    checkpoint_id = checkpoint_manager.save_checkpoint(
                        pipeline_id=pipeline_id,
                        stage=stage_name,
                        data=processed,
                        metadata={
                            'signals': ts_signals,
                            'forecasts': forecasts,
                            'dataset_id': dataset_id_current,
                            'generator_version': generator_version_current,
                        }
                    )
                    pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)
                    
                except ImportError as e:
                    logger.warning(f"Time Series signal generation modules not available: {e}")
                    logger.warning("Install required packages and ensure models package is available")
                except Exception as e:
                    from etl.security_utils import sanitize_error
                    safe_error = sanitize_error(e)
                    logger.error(f"Time Series signal generation stage failed: {safe_error}")

            elif stage_name == 'signal_router':
                # Stage 9: Signal Router (NEW - Routes TS primary + LLM fallback)
                logger.info("Routing signals (Time Series primary, LLM fallback)...")
                
                try:
                    from models.signal_router import SignalRouter
                    from models.signal_adapter import SignalAdapter
                    
                    # Load signal routing config
                    signal_routing_cfg = pipeline_cfg.get('signal_routing', {})
                    
                    # Initialize router
                    router = SignalRouter(
                        config=signal_routing_cfg,
                        time_series_generator=None,  # Already generated signals
                        llm_generator=llm_signal_generator if enable_llm and llm_signal_generator is not None else None
                    )
                    
                    # Get Time Series signals from previous stage
                    ts_signals = globals().get('_ts_signals', {})
                    forecasts = globals().get('_ts_forecasts', {})
                    current_prices = globals().get('_current_prices', {})
                    
                    # Get LLM signals if available
                    llm_signals = globals().get('llm_signals', {})
                    
                    routed_bundles = {}
                    
                    for ticker in ticker_list:
                        try:
                            # Get ticker-specific data
                            if isinstance(processed.index, pd.MultiIndex):
                                ticker_data = processed.xs(ticker, level=0)
                            else:
                                ticker_data = processed
                            
                            current_price = current_prices.get(ticker, 0.0)
                            if current_price == 0.0:
                                # Try to extract from data
                                if 'Close' in ticker_data.columns:
                                    current_price = float(ticker_data['Close'].iloc[-1])
                                elif 'close' in ticker_data.columns:
                                    current_price = float(ticker_data['close'].iloc[-1])
                            
                            forecast_bundle = forecasts.get(ticker)
                            llm_signal = llm_signals.get(ticker) if llm_signals else None
                            
                            # Route signal
                            bundle = router.route_signal(
                                ticker=ticker,
                                forecast_bundle=forecast_bundle,
                                current_price=current_price,
                                market_data=ticker_data,
                                llm_signal=llm_signal
                            )
                            
                            routed_bundles[ticker] = bundle
                            
                            # Log routing result
                            primary_action = bundle.primary_signal.get('action', 'HOLD') if bundle.primary_signal else 'HOLD'
                            primary_source = bundle.primary_signal.get('source', 'UNKNOWN') if bundle.primary_signal else 'UNKNOWN'
                            
                            logger.info(
                                f"  OK {ticker}: Routed {primary_action} signal "
                                f"(source={primary_source}, "
                                f"fallback={'yes' if bundle.fallback_signal else 'no'})"
                            )
                            
                        except Exception as e:
                            logger.warning(f"  WARN {ticker} signal routing failed: {e}")
                            routed_bundles[ticker] = None
                    
                    logger.info(f"OK Routed signals for {len([b for b in routed_bundles.values() if b])} ticker(s)")
                    
                    # Store routing stats
                    routing_stats = router.get_routing_stats()
                    logger.info(f"Routing statistics: {routing_stats['stats']}")
                    
                    # Save checkpoint
                    checkpoint_id = checkpoint_manager.save_checkpoint(
                        pipeline_id=pipeline_id,
                        stage=stage_name,
                        data=processed,
                        metadata={
                            'routed_bundles': {k: {
                                'primary': v.primary_signal if v and v.primary_signal else None,
                                'fallback': v.fallback_signal if v and v.fallback_signal else None
                            } for k, v in routed_bundles.items() if v},
                            'routing_stats': routing_stats,
                            'dataset_id': dataset_id_current,
                            'generator_version': generator_version_current,
                        }
                    )
                    pipeline_log.log_checkpoint(pipeline_id, stage_name, checkpoint_id)
                    
                except ImportError as e:
                    logger.warning(f"Signal router modules not available: {e}")
                except Exception as e:
                    from etl.security_utils import sanitize_error
                    safe_error = sanitize_error(e)
                    logger.error(f"Signal routing stage failed: {safe_error}")

            elif stage_name == 'data_storage':
                # Stage 4: Data Storage (Split + Save)
                logger.info("Splitting and saving datasets...")
                logger.info(f"  Configuration source: pipeline_config.yml")
                logger.info(f"  Split strategy: {'CV' if use_cv else 'Simple'}")
                drift_records: List[Dict[str, Any]] = []

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

                # Split diagnostics and drift checks
                def _log_split_stats(name: str, frame: pd.DataFrame) -> None:
                    summary = summarize_returns(name, frame)
                    logger.info(
                        "Split %s: len=%d start=%s end=%s mean=%.4f std=%.4f skew=%.4f kurt=%.4f",
                        summary.name,
                        summary.length,
                        summary.start,
                        summary.end,
                        summary.mean,
                        summary.std,
                        summary.skew,
                        summary.kurtosis,
                    )

                if use_cv and splits.get('cv_folds'):
                    for fold in splits['cv_folds']:
                        tr = fold['train']
                        va = fold['validation']
                        if not validate_non_overlap(tr.index, va.index):
                            logger.warning("Overlap detected in CV fold %s", fold['fold_id'])
                    _log_split_stats(f"cv{fold['fold_id']}_train", tr)
                    _log_split_stats(f"cv{fold['fold_id']}_val", va)
                    drift = drift_metrics(tr, va)
                    drift_records.append(
                        {
                            "split": f"cv{fold['fold_id']}_train_val",
                            "psi": drift["psi"],
                            "mean_delta": drift["mean_delta"],
                            "std_delta": drift["std_delta"],
                            "vol_psi": drift["vol_psi"],
                            "vol_delta": drift["vol_delta"],
                            "volatility_ratio": drift.get("volatility_ratio"),
                        }
                    )
                    try:
                        db_manager.save_split_drift(
                            run_id=pipeline_id,
                            ticker=None,
                            split_name=f"cv_fold_{fold['fold_id']}",
                            metrics=drift,
                        )
                    except Exception:
                        logger.debug("Skipping drift persistence for CV fold %s", fold['fold_id'])
                    if drift["psi"] > 0.2 or drift["vol_psi"] > 0.2:
                        logger.warning(
                            "Drift detected in CV fold %s (psi=%.3f vol_psi=%.3f)",
                            fold['fold_id'],
                            drift["psi"],
                            drift["vol_psi"],
                        )
                    else:
                        logger.info(
                            "CV fold %s drift psi=%.3f vol_psi=%.3f (OK)",
                            fold['fold_id'],
                            drift["psi"],
                            drift["vol_psi"],
                        )
                    try:
                        db_manager.save_latency_metrics(
                            ticker="CV",
                            run_id=pipeline_id,
                            stage=f"cv_fold_{fold['fold_id']}",
                            ts_ms=None,
                            llm_ms=None,
                        )
                    except Exception:
                        logger.debug("Skipping latency metrics persistence for CV fold %s", fold['fold_id'])
                test_df = splits.get('testing')
                if test_df is not None and not test_df.empty:
                    _log_split_stats("test", test_df)
            else:
                tr = splits.get('training', pd.DataFrame())
                va = splits.get('validation', pd.DataFrame())
                te = splits.get('testing', pd.DataFrame())
                if not validate_non_overlap(tr.index, va.index):
                    logger.warning("Overlap detected between train and val")
                if not validate_non_overlap(tr.index, te.index):
                    logger.warning("Overlap detected between train and test")
                _log_split_stats("train", tr)
                _log_split_stats("val", va)
                _log_split_stats("test", te)
                drift_tv = drift_metrics(tr, va)
                drift_tt = drift_metrics(tr, te)
                drift_records.extend(
                    [
                        {
                            "split": "train_val",
                            "psi": drift_tv["psi"],
                            "mean_delta": drift_tv["mean_delta"],
                            "std_delta": drift_tv["std_delta"],
                            "vol_psi": drift_tv["vol_psi"],
                            "vol_delta": drift_tv["vol_delta"],
                            "volatility_ratio": drift_tv.get("volatility_ratio"),
                        },
                        {
                            "split": "train_test",
                            "psi": drift_tt["psi"],
                            "mean_delta": drift_tt["mean_delta"],
                            "std_delta": drift_tt["std_delta"],
                            "vol_psi": drift_tt["vol_psi"],
                            "vol_delta": drift_tt["vol_delta"],
                            "volatility_ratio": drift_tt.get("volatility_ratio"),
                        },
                    ]
                )
                try:
                    db_manager.save_split_drift(
                        run_id=pipeline_id,
                        ticker=None,
                        split_name="train_val",
                        metrics=drift_tv,
                    )
                    db_manager.save_split_drift(
                        run_id=pipeline_id,
                        ticker=None,
                        split_name="train_test",
                        metrics=drift_tt,
                    )
                except Exception:
                    logger.debug("Skipping drift persistence for holdout splits")
                if drift_tv["psi"] > 0.2 or drift_tv["vol_psi"] > 0.2:
                    logger.warning("Train/Val drift psi=%.3f vol_psi=%.3f", drift_tv["psi"], drift_tv["vol_psi"])
                if drift_tt["psi"] > 0.2 or drift_tt["vol_psi"] > 0.2:
                    logger.warning("Train/Test drift psi=%.3f vol_psi=%.3f", drift_tt["psi"], drift_tt["vol_psi"])

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if use_cv:
                    # Save CV folds with run metadata
                    for fold in splits['cv_folds']:
                        fold_id = fold['fold_id']
                        fold_metadata = {
                            'data_source': extraction_source,
                            'execution_mode': execution_mode,
                            'pipeline_id': pipeline_id,
                            'fold_id': fold_id,
                            'split_strategy': 'cross_validation',
                        }
                        storage.save(
                            fold['train'], 
                            'training',
                            f'fold{fold_id}_train_{timestamp}',
                            metadata=fold_metadata,
                            run_id=pipeline_id
                        )
                        storage.save(
                            fold['validation'], 
                            'validation',
                            f'fold{fold_id}_val_{timestamp}',
                            metadata=fold_metadata,
                            run_id=pipeline_id
                        )

                    # Save isolated test set
                    storage.save(
                        splits['testing'], 
                        'testing', 
                        f'test_{timestamp}',
                        metadata={
                            'data_source': extraction_source,
                            'execution_mode': execution_mode,
                            'pipeline_id': pipeline_id,
                            'split_strategy': 'cross_validation',
                            'dataset_id': dataset_id_current,
                            'generator_version': generator_version_current,
                        },
                        run_id=pipeline_id
                    )

                    # Calculate summary statistics
                    avg_train_size = sum(len(f['train']) for f in splits['cv_folds']) / len(splits['cv_folds'])
                    avg_val_size = sum(len(f['validation']) for f in splits['cv_folds']) / len(splits['cv_folds'])

                    logger.info("Saved %s CV folds + 1 test set", len(splits['cv_folds']))
                    logger.info("  - Train size (avg): %.0f rows", avg_train_size)
                    logger.info("  - Val size (avg): %.0f rows", avg_val_size)
                    logger.info("  - Test size: %s rows", len(splits['testing']))
                else:
                    # Simple split (backward compatible) with run metadata
                    for split_name, split_data in splits.items():
                        storage.save(
                            split_data, 
                            split_name,
                            f'{split_name}_{timestamp}',
                            metadata={
                                'data_source': extraction_source,
                                'execution_mode': execution_mode,
                                'pipeline_id': pipeline_id,
                                'split_strategy': 'simple',
                                'dataset_id': dataset_id_current,
                                'generator_version': generator_version_current,
                            },
                            run_id=pipeline_id
                        )

                    logger.info("Saved simple split:")
                    logger.info("  - Training: %s rows (70%)", len(splits['training']))
                    logger.info("  - Validation: %s rows (15%)", len(splits['validation']))
                    logger.info("  - Testing: %s rows (15%)", len(splits['testing']))

                _emit_split_drift_json(Path("visualizations") / "split_drift_latest.json", pipeline_id, drift_records)

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
            from etl.security_utils import sanitize_error
            safe_error = sanitize_error(e)
            logger.error(f"X Stage '{stage_name}' failed: {safe_error}")
            # Log original error internally for debugging
            pipeline_log.log_stage_error(pipeline_id, stage_name, e)

            if verbose:
                import traceback
                logger.debug(traceback.format_exc())
            raise

    # Pipeline completion
    logger.info("=" * 70)
    logger.info("OK Pipeline completed successfully")
    logger.info("=" * 70)

    # Log pipeline completion
    pipeline_log.log_event('pipeline_complete', pipeline_id, status='success', metadata={
        "execution_mode": execution_mode,
        "dataset_id": dataset_id_current,
        "generator_version": generator_version_current,
    })

    # Cleanup old logs and checkpoints
    pipeline_log.cleanup_old_logs()
    checkpoint_manager.cleanup_old_checkpoints(retention_days=7)
    
    # Close database connection
    db_manager.close()


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
@click.option('--llm-model', default='',
              help='LLM model override. Leave blank to use config active_model (recommended). Options include deepseek-coder:6.7b-instruct-q4_K_M, codellama:13b-instruct-q4_K_M, qwen:14b-chat-q4_K_M')
@click.option('--dry-run', is_flag=True, default=False,
              help='Generate synthetic OHLCV data in-process (no network) to exercise stages')
@click.option('--execution-mode', default='auto',
              type=click.Choice(EXECUTION_MODES, case_sensitive=False),
              help='Data extraction mode: live (network), synthetic (offline), or auto (try live, fallback to synthetic)')
@click.option(
    '--db-path',
    default=None,
    help='Override SQLite database path (default: data/portfolio_maximizer.db; synthetic runs default to data/test_database.db)',
)
@click.option('--use-ticker-discovery', is_flag=True, default=False,
              help='Load tickers from the configured ticker discovery universe.')
@click.option('--refresh-ticker-universe', is_flag=True, default=False,
              help='Force refresh of the ticker discovery universe before running the pipeline.')
@click.option(
    '--include-frontier-tickers',
    is_flag=True,
    default=False,
    help=(
        'Append curated frontier market tickers (Nigeria, Kenya, Vietnam, '
        'Bangladesh, Sri Lanka, Pakistan, Kuwait, Qatar, Romania, Bulgaria) '
        'to multi-ticker runs.'
    ),
)
@click.option(
    '--synthetic-dataset-id',
    default=None,
    help='Optional synthetic dataset_id to load (persists across runs). Sets SYNTHETIC_DATASET_ID.',
)
@click.option(
    '--synthetic-dataset-path',
    default=None,
    help='Optional synthetic dataset path to load (persists across runs). Sets SYNTHETIC_DATASET_PATH.',
)
@click.option(
    '--synthetic-config',
    default=None,
    help='Optional synthetic config path override (passed to SyntheticExtractor). Sets SYNTHETIC_CONFIG_PATH.',
)
@click.option(
    '--prefer-gpu/--no-prefer-gpu',
    default=True,
    help='Attempt to use GPU (cuda/mps) when available; fallback to CPU automatically.',
)
def run_pipeline(config: str, data_source: str, tickers: str, start: str, end: str,
                use_cv: bool, n_splits: int, test_size: float, gap: int, verbose: bool,
                enable_llm: bool, llm_model: str, dry_run: bool, prefer_gpu: bool,
                execution_mode: str, db_path: Optional[str], use_ticker_discovery: bool,
                refresh_ticker_universe: bool, include_frontier_tickers: bool,
                synthetic_dataset_id: Optional[str], synthetic_dataset_path: Optional[str],
                synthetic_config: Optional[str]) -> None:
    """Execute ETL pipeline with modular configuration-driven orchestration.

    Data Splitting Strategy:
    - Default (--use-cv=False): Simple 70/15/15 chronological split (backward compatible)
    - Recommended (--use-cv): k-fold cross-validation with expanding window
      * 5.5x better temporal coverage (15% -> 83%)
      * Eliminates temporal gap (0 years vs 2.5 years)
      * Strict test isolation (15% never exposed during CV)

    Examples:
        # Simple split (backward compatible)
        python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --include-frontier-tickers --start 2020-01-01

        # k-fold CV (recommended for production)
        python scripts/run_etl_pipeline.py --tickers AAPL --use-cv --n-splits 5

        # Verbose logging
        python scripts/run_etl_pipeline.py --tickers GOOGL --use-cv --verbose

        # Live run with automatic synthetic fallback
        python scripts/run_etl_pipeline.py --tickers NVDA --execution-mode auto --enable-llm

        # Dry run (no network) with synthetic data
        python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --include-frontier-tickers --start 2024-01-02 --end 2024-01-19 --dry-run
    """
    # Convert Click arguments to execute_pipeline call
    execute_pipeline(
        config=config,
        data_source=data_source,
        tickers=tickers,
        start=start,
        end=end,
        use_cv=use_cv,
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        verbose=verbose,
        enable_llm=enable_llm,
        llm_model=llm_model,
        dry_run=dry_run,
        execution_mode=execution_mode,
        prefer_gpu=prefer_gpu,
        db_path=db_path,
        use_ticker_discovery=use_ticker_discovery,
        refresh_ticker_universe=refresh_ticker_universe,
        include_frontier_tickers=include_frontier_tickers,
        synthetic_dataset_id=synthetic_dataset_id,
        synthetic_dataset_path=synthetic_dataset_path,
        synthetic_config=synthetic_config,
    )


if __name__ == '__main__':
    run_pipeline()
