#!/usr/bin/env python3
"""Run ensemble diagnostics on recent pipeline forecasts.

This script extracts forecast data from the database and runs comprehensive
ensemble error tracking diagnostics to identify why ensemble RMSE > best single model.

Usage:
    python scripts/run_ensemble_diagnostics.py --pipeline-id pipeline_20260120_021448
    python scripts/run_ensemble_diagnostics.py --recent  # Use most recent pipeline
    python scripts/run_ensemble_diagnostics.py --ticker AAPL --days 30
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forcester_ts.ensemble_diagnostics import (
    EnsembleDiagnostics,
    ModelPerformance,
    EnsemblePerformance
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_forecast_data_from_db(
    ticker: str,
    days: Optional[int] = None,
    pipeline_id: Optional[str] = None
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """Extract forecast data from database for a specific ticker.

    Args:
        ticker: Ticker symbol to analyze
        days: Number of recent days to include (optional)
        pipeline_id: Specific pipeline ID to analyze (optional)

    Returns:
        Tuple of (model_forecasts dict, actual_values series)
    """
    conn = sqlite3.connect('data/portfolio_maximizer.db')

    # Build query (note: column is model_type not model_name in schema)
    query = f"""
        SELECT
            model_type as model_name,
            forecast_date,
            forecast_value,
            lower_ci,
            upper_ci,
            volatility,
            created_at
        FROM time_series_forecasts
        WHERE ticker = '{ticker}'
    """

    if pipeline_id:
        query += f" AND created_at LIKE '{pipeline_id.split('_')[-2]}%'"

    if days:
        query += f" AND date(forecast_date) >= date('now', '-{days} days')"

    query += " ORDER BY forecast_date, model_name"

    logger.info(f"Querying forecasts for {ticker}...")
    df = pd.read_sql_query(query, conn)

    if df.empty:
        conn.close()
        raise ValueError(f"No forecast data found for {ticker}")

    logger.info(f"Found {len(df)} forecast records")

    # Pivot by model
    model_forecasts = {}
    actual_values = None

    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name].sort_values('forecast_date')

        # Compute confidence from CI width (narrower CI = higher confidence)
        # Normalize to [0, 1] scale
        ci_width = (model_df['upper_ci'] - model_df['lower_ci']).abs()
        if ci_width.max() > 0:
            # Inverse: narrow CI = high confidence
            confidence = 1.0 - (ci_width / ci_width.max())
            # Scale to [0.5, 1.0] range for more realistic values
            confidence = 0.5 + 0.5 * confidence
            confidence = confidence.values  # Convert Series to array
        else:
            confidence = np.ones(len(model_df)) * 0.75  # Default

        model_forecasts[model_name] = pd.DataFrame({
            'date': pd.to_datetime(model_df['forecast_date']),
            'forecast': model_df['forecast_value'].values,
            'confidence': confidence,  # Already numpy array
            'actual': np.nan  # Will need to fetch actuals separately
        })

    # Fetch actual prices from OHLCV data
    # Try to get actuals from the database
    actual_query = f"""
        SELECT date, close
        FROM ohlcv_data
        WHERE ticker = '{ticker}'
        AND date >= (SELECT MIN(forecast_date) FROM time_series_forecasts WHERE ticker = '{ticker}')
        ORDER BY date
    """

    try:
        actuals_df = pd.read_sql_query(actual_query, conn)
        if not actuals_df.empty:
            actual_values = actuals_df['close'].values
            logger.info(f"Fetched {len(actual_values)} actual price values")
        else:
            logger.warning("No actual price data found in OHLCV table")
            actual_values = None
    except Exception as e:
        logger.warning(f"Could not fetch actuals: {e}")
        actual_values = None

    conn.close()

    if actual_values is None or len(actual_values) == 0:
        raise ValueError(f"No actual values found for {ticker} - cannot compute errors")

    return model_forecasts, actual_values


def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, float]:
    """Compute standard forecast error metrics.

    Args:
        predictions: Model predictions
        actuals: Actual values

    Returns:
        Dictionary of metrics (RMSE, MAE, MAPE, directional_accuracy)
    """
    errors = predictions - actuals
    squared_errors = errors ** 2

    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(np.abs(errors))

    # MAPE (avoid division by zero)
    mask = actuals != 0
    mape = np.mean(np.abs(errors[mask] / actuals[mask])) * 100 if mask.sum() > 0 else 0

    # Directional accuracy (% of times forecast direction matches actual direction)
    pred_direction = np.diff(predictions)
    actual_direction = np.diff(actuals)
    correct_direction = (pred_direction * actual_direction) > 0
    directional_accuracy = correct_direction.sum() / len(correct_direction) if len(correct_direction) > 0 else 0

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }


def extract_ensemble_weights_from_logs(
    pipeline_id: Optional[str] = None
) -> Dict[str, float]:
    """Extract ensemble weights from pipeline logs.

    Args:
        pipeline_id: Pipeline ID to extract weights for

    Returns:
        Dictionary of model weights
    """
    # Try to extract from logs
    log_file = Path('logs/pipeline_run.log')

    if not log_file.exists():
        logger.warning("Pipeline log not found, using uniform weights")
        return {}

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Look for ensemble weight logs
        for line in reversed(lines):
            if 'ENSEMBLE build_complete' in line and 'weights=' in line:
                # Extract weights from log line
                # Format: weights={'sarimax': 0.6, 'samossa': 0.4}
                import re
                match = re.search(r"weights=\{([^}]+)\}", line)
                if match:
                    weights_str = match.group(1)
                    weights = {}
                    for pair in weights_str.split(','):
                        if ':' in pair:
                            key, val = pair.split(':')
                            key = key.strip().strip("'\"")
                            val = float(val.strip())
                            weights[key] = val
                    logger.info(f"Extracted ensemble weights: {weights}")
                    return weights
    except Exception as e:
        logger.warning(f"Failed to extract weights from logs: {e}")

    return {}


def run_diagnostics_for_ticker(
    ticker: str,
    days: Optional[int] = None,
    pipeline_id: Optional[str] = None,
    output_dir: str = "visualizations/ensemble_diagnostics"
) -> None:
    """Run full ensemble diagnostics for a ticker.

    Args:
        ticker: Ticker symbol to analyze
        days: Number of recent days to include
        pipeline_id: Specific pipeline ID to analyze
        output_dir: Output directory for visualizations
    """
    logger.info(f"Starting ensemble diagnostics for {ticker}")

    # Create diagnostics instance
    diagnostics = EnsembleDiagnostics(output_dir=f"{output_dir}/{ticker}")

    # Extract forecast data
    try:
        model_forecasts, actuals = extract_forecast_data_from_db(ticker, days, pipeline_id)
    except Exception as e:
        logger.error(f"Failed to extract forecast data: {e}")
        return

    # Add individual model performances
    for model_name, model_df in model_forecasts.items():
        if model_name.lower() == 'ensemble':
            continue  # Skip ensemble for now, handle separately

        predictions = model_df['forecast'].values
        confidence = model_df['confidence'].values

        # Align with actuals (may have different lengths)
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals_aligned = actuals[:min_len]
        confidence = confidence[:min_len]

        metrics = compute_metrics(predictions, actuals_aligned)

        perf = ModelPerformance(
            name=model_name,
            predictions=predictions,
            actuals=actuals_aligned,
            confidence=confidence,
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            mape=metrics['mape'],
            directional_accuracy=metrics['directional_accuracy']
        )

        diagnostics.add_model_performance(perf)

    # Add ensemble performance if available (check multiple possible keys)
    ensemble_key = None
    for key in ['ensemble', 'ENSEMBLE', 'combined', 'COMBINED']:
        if key.lower() in [k.lower() for k in model_forecasts.keys()]:
            ensemble_key = next(k for k in model_forecasts.keys() if k.lower() == key.lower())
            break

    if ensemble_key:
        ensemble_df = model_forecasts[ensemble_key]

        predictions = ensemble_df['forecast'].values
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals_aligned = actuals[:min_len]

        errors = predictions - actuals_aligned
        rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)
        variance = np.var(errors)

        # Extract weights from logs or use default
        weights = extract_ensemble_weights_from_logs(pipeline_id)
        if not weights:
            # Default: equal weights among available models
            n_models = len([k for k in model_forecasts.keys() if k.lower() != 'ensemble'])
            weights = {name: 1.0/n_models for name in model_forecasts.keys() if name.lower() != 'ensemble'}

        ens_perf = EnsemblePerformance(
            predictions=predictions,
            actuals=actuals_aligned,
            weights=weights,
            rmse=rmse,
            bias=bias,
            variance=variance
        )

        diagnostics.add_ensemble_performance(ens_perf)
    else:
        logger.warning("No ensemble forecast data found - will analyze individual models only")

    # Run full diagnostics
    logger.info("Generating diagnostic visualizations...")
    diagnostics.run_full_diagnostics()

    logger.info(f"✓ Diagnostics complete for {ticker}. Results in {output_dir}/{ticker}/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run ensemble forecasting diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze most recent pipeline for AAPL
  python scripts/run_ensemble_diagnostics.py --ticker AAPL

  # Analyze specific pipeline
  python scripts/run_ensemble_diagnostics.py --ticker AAPL --pipeline-id pipeline_20260120_021448

  # Analyze last 30 days of forecasts
  python scripts/run_ensemble_diagnostics.py --ticker AAPL --days 30

  # Analyze multiple tickers
  python scripts/run_ensemble_diagnostics.py --ticker AAPL,MSFT,NVDA
        """
    )

    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Ticker symbol(s) to analyze (comma-separated)')
    parser.add_argument('--days', type=int, default=None,
                       help='Number of recent days to analyze')
    parser.add_argument('--pipeline-id', type=str, default=None,
                       help='Specific pipeline ID to analyze')
    parser.add_argument('--output-dir', type=str, default='visualizations/ensemble_diagnostics',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Parse tickers
    tickers = [t.strip().upper() for t in args.ticker.split(',')]

    # Run diagnostics for each ticker
    for ticker in tickers:
        try:
            run_diagnostics_for_ticker(
                ticker=ticker,
                days=args.days,
                pipeline_id=args.pipeline_id,
                output_dir=args.output_dir
            )
        except Exception as e:
            logger.error(f"Failed to run diagnostics for {ticker}: {e}", exc_info=True)
            continue

    logger.info("All diagnostics complete!")


if __name__ == '__main__':
    main()
