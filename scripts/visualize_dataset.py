#!/usr/bin/env python3
"""Comprehensive Time Series Visualization Script.

Usage:
    python scripts/visualize_dataset.py --data data/training/*.parquet --column Close
    python scripts/visualize_dataset.py --data data/*.parquet --column Close --all-plots
    python scripts/visualize_dataset.py --data data/*.parquet --column Close --dashboard
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import click
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.visualizer import TimeSeriesVisualizer
from forcester_ts.instrumentation import describe_dataframe
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _sanitize_token(token: Optional[str]) -> Optional[str]:
    """Normalize user-provided names (ticker / column) for filesystem safety."""
    if not token:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in token.strip())
    cleaned = cleaned.strip("_")
    return cleaned or None


def _build_save_path(output_dir: str, suffix: str, column: str, ticker: Optional[str] = None) -> Path:
    """Compose a deterministic filename that includes the ticker when available."""
    parts = []
    ticker_token = _sanitize_token(ticker)
    column_token = _sanitize_token(column)
    if ticker_token:
        parts.append(ticker_token)
    if column_token:
        parts.append(column_token)
    parts.append(suffix)
    filename = "_".join(parts)
    return Path(output_dir) / f"{filename}.png"


@click.command()
@click.option('--data', required=False, help='Path to dataset file (parquet or csv)')
@click.option('--from-db', is_flag=True, help='Load data directly from the SQLite database')
@click.option('--db-path', default='data/portfolio_maximizer.db', help='Path to SQLite database')
@click.option('--ticker', help='Ticker symbol when loading from database')
@click.option('--lookback-days', default=180, help='Lookback window (days) when loading from database')
@click.option('--column', default='Close', help='Column to visualize (default: Close)')
@click.option('--overview', is_flag=True, help='Plot time series overview')
@click.option('--distribution', is_flag=True, help='Plot distribution analysis')
@click.option('--acf', is_flag=True, help='Plot ACF/PACF')
@click.option('--decomposition', is_flag=True, help='Plot trend decomposition')
@click.option('--rolling', is_flag=True, help='Plot rolling statistics')
@click.option('--spectral', is_flag=True, help='Plot spectral density')
@click.option('--dashboard', is_flag=True, help='Create comprehensive dashboard')
@click.option('--forecast-dashboard', is_flag=True, help='Create forecast performance dashboard (database only)')
@click.option('--signal-dashboard', is_flag=True, help='Create signal performance dashboard (database only)')
@click.option('--context-columns', default='', help='Comma-separated context columns (market/commodity indices)')
@click.option('--all-plots', is_flag=True, help='Generate all visualization types')
@click.option('--output-dir', default='visualizations', help='Output directory for plots')
@click.option('--save', is_flag=True, help='Save plots to files')
@click.option('--show', is_flag=True, default=True, help='Display plots interactively')
@click.option('--window', default=30, help='Rolling window size (default: 30)')
@click.option('--nlags', default=40, help='Number of lags for ACF/PACF (default: 40)')
def visualize_dataset(data: Optional[str], from_db: bool, db_path: str, ticker: Optional[str],
                     lookback_days: int, column: str, overview: bool, distribution: bool,
                     acf: bool, decomposition: bool, rolling: bool, spectral: bool,
                     dashboard: bool, forecast_dashboard: bool, signal_dashboard: bool,
                     context_columns: str, all_plots: bool, output_dir: str,
                     save: bool, show: bool, window: int, nlags: int):
    """Comprehensive time series visualization following MIT standards."""

    logger.info("=" * 80)
    logger.info("TIME SERIES VISUALIZATION - Statistical Graphics")
    logger.info("=" * 80)

    # Load data
    forecast_bundle: Dict[str, Dict[str, Optional[pd.Series]]] = {}
    signal_metrics = pd.DataFrame()

    if from_db:
        if not ticker:
            logger.error("--ticker is required when using --from-db")
            sys.exit(1)
        logger.info(f"\n📊 Loading {ticker} from database: {db_path}")
        try:
            from etl.dashboard_loader import DashboardDataLoader
        except ImportError as exc:
            logger.error(f"Unable to import dashboard loader: {exc}")
            sys.exit(1)

        loader = DashboardDataLoader.from_path(db_path)
        price_df = loader.get_price_history(ticker, lookback_days=lookback_days, columns=[column])
        if price_df is None or price_df.empty:
            logger.error("No price history available for ticker %s", ticker)
            sys.exit(1)
        df = price_df
        forecast_bundle = loader.get_forecast_bundle(ticker)
        signal_metrics = loader.get_signal_backtests(ticker)
        logger.info(f"✓ Loaded {len(df)} rows from database for {ticker}")
    else:
        if not data:
            logger.error("Either --data or --from-db must be provided.")
            sys.exit(1)
        logger.info(f"\n📊 Loading data from: {data}")
        try:
            if data.endswith('.parquet'):
                df = pd.read_parquet(data)
            else:
                df = pd.read_csv(data)

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("No datetime index found. Using integer index.")

            logger.info(f"✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)

    # Validate column
    if column not in df.columns:
        logger.error(f"Column '{column}' not found. Available: {df.columns.tolist()}")
        sys.exit(1)

    context_cols = [c.strip() for c in context_columns.split(",") if c.strip()]
    metadata_columns = [column] + [col for col in context_cols if col in df.columns]
    dataset_metadata = describe_dataframe(df, columns=metadata_columns)

    # Initialize visualizer
    visualizer = TimeSeriesVisualizer(figsize=(16, 12))

    # Determine which plots to create
    if all_plots:
        overview = distribution = acf = decomposition = rolling = spectral = dashboard = True
        forecast_dashboard = signal_dashboard = True

    if not any([overview, distribution, acf, decomposition, rolling, spectral, dashboard,
                forecast_dashboard, signal_dashboard]):
        # Default: show overview if nothing specified
        overview = True

    # Create output directory if saving
    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"\n📁 Output directory: {output_dir}")

    # Generate plots
    logger.info(f"\n🎨 Generating visualizations for column: {column}")
    logger.info("-" * 80)

    plots_created = []

    # 1. Time Series Overview
    if overview:
        logger.info("Creating time series overview...")
        fig = visualizer.plot_time_series_overview(df, columns=[column],
                                                   title=f"Time Series Overview: {column}")
        plots_created.append(('overview', fig))
        if save:
            save_path = _build_save_path(output_dir, "overview", column, ticker)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 2. Distribution Analysis
    if distribution:
        logger.info("Creating distribution analysis...")
        fig = visualizer.plot_distribution_analysis(df, column)
        plots_created.append(('distribution', fig))
        if save:
            save_path = _build_save_path(output_dir, "distribution", column, ticker)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 3. ACF/PACF
    if acf:
        logger.info(f"Creating ACF/PACF plots (nlags={nlags})...")
        fig = visualizer.plot_autocorrelation(df, column, nlags=nlags)
        plots_created.append(('acf', fig))
        if save:
            save_path = _build_save_path(output_dir, "acf_pacf", column, ticker)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 4. Decomposition
    if decomposition:
        logger.info("Creating trend-seasonal decomposition...")
        try:
            fig = visualizer.plot_decomposition(df, column)
            plots_created.append(('decomposition', fig))
            if save:
                save_path = _build_save_path(output_dir, "decomposition", column, ticker)
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"  ✓ Saved to {save_path}")
        except Exception as e:
            logger.warning(f"  ⚠️  Decomposition failed: {e}")

    # 5. Rolling Statistics
    if rolling:
        logger.info(f"Creating rolling statistics (window={window})...")
        fig = visualizer.plot_rolling_statistics(df, column, window=window)
        plots_created.append(('rolling', fig))
        if save:
            save_path = _build_save_path(output_dir, "rolling_stats", column, ticker)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 6. Spectral Density
    if spectral:
        logger.info("Creating spectral density plot...")
        fig = visualizer.plot_spectral_density(df, column)
        plots_created.append(('spectral', fig))
        if save:
            save_path = _build_save_path(output_dir, "spectral", column, ticker)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 7. Comprehensive Dashboard
    if dashboard:
        logger.info("Creating comprehensive dashboard...")
        save_path = _build_save_path(output_dir, "dashboard", column, ticker) if save else None
        fig = visualizer.plot_comprehensive_dashboard(
            df,
            column,
            save_path=str(save_path) if save_path else None,
            market_columns=context_cols or None,
            metadata=dataset_metadata,
        )
        plots_created.append(('dashboard', fig))
        if save:
            logger.info(f"  ✓ Saved to {save_path}")

    if forecast_dashboard:
        if not from_db:
            logger.warning("Forecast dashboard requires --from-db; skipping.")
        else:
            logger.info("Creating forecast dashboard...")
            forecasts_available = {
                model: payload
                for model, payload in forecast_bundle.items()
                if isinstance(payload, dict) and payload.get("forecast") is not None
            }
            if not forecasts_available:
                logger.warning("No forecasts available in the database; skipping forecast dashboard.")
            else:
                ensemble_payload = forecasts_available.get("ENSEMBLE") or forecasts_available.get("COMBINED")
                weights = None
                if isinstance(ensemble_payload, dict):
                    weights = ensemble_payload.get("weights")
                fig = visualizer.plot_forecast_dashboard(
                    df[column],
                    forecasts_available,
                    title=f"{ticker or column} Forecast Dashboard",
                    weights=weights,
                )
                plots_created.append(('forecast_dashboard', fig))
                if save:
                    save_path = _build_save_path(output_dir, "forecast_dashboard", column, ticker)
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    logger.info(f"  ✓ Saved to {save_path}")

    if signal_dashboard:
        if not from_db:
            logger.warning("Signal dashboard requires --from-db; skipping.")
        else:
            logger.info("Creating signal performance dashboard...")
            if signal_metrics.empty:
                logger.warning("No LLM signal backtest data available; skipping signal dashboard.")
            else:
                fig = visualizer.plot_signal_performance(
                    signal_metrics,
                    ticker=ticker,
                )
                plots_created.append(('signal_dashboard', fig))
                if save:
                    save_path = _build_save_path(output_dir, "signal_dashboard", column, ticker)
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    logger.info(f"  ✓ Saved to {save_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    if from_db and ticker:
        logger.info(f"Ticker analyzed: {ticker}")
    logger.info(f"Column analyzed: {column}")
    logger.info(f"Plots created: {len(plots_created)}")
    for plot_name, _ in plots_created:
        logger.info(f"  ✓ {plot_name}")

    if save:
        logger.info(f"\n📁 All plots saved to: {output_dir}/")

    # Display plots interactively
    if show and not save:
        logger.info("\n👁️  Displaying plots interactively...")
        plt.show()
    elif show and save:
        logger.info("\n👁️  Close plot windows to exit...")
        plt.show()

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    visualize_dataset()
