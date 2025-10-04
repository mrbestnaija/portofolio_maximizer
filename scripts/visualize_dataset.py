#!/usr/bin/env python3
"""Comprehensive Time Series Visualization Script.

Usage:
    python scripts/visualize_dataset.py --data data/training/*.parquet --column Close
    python scripts/visualize_dataset.py --data data/*.parquet --column Close --all-plots
    python scripts/visualize_dataset.py --data data/*.parquet --column Close --dashboard
"""

import sys
from pathlib import Path
import click
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.visualizer import TimeSeriesVisualizer
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--data', required=True, help='Path to dataset file (parquet or csv)')
@click.option('--column', required=True, help='Column to visualize')
@click.option('--overview', is_flag=True, help='Plot time series overview')
@click.option('--distribution', is_flag=True, help='Plot distribution analysis')
@click.option('--acf', is_flag=True, help='Plot ACF/PACF')
@click.option('--decomposition', is_flag=True, help='Plot trend decomposition')
@click.option('--rolling', is_flag=True, help='Plot rolling statistics')
@click.option('--spectral', is_flag=True, help='Plot spectral density')
@click.option('--dashboard', is_flag=True, help='Create comprehensive dashboard')
@click.option('--all-plots', is_flag=True, help='Generate all visualization types')
@click.option('--output-dir', default='visualizations', help='Output directory for plots')
@click.option('--save', is_flag=True, help='Save plots to files')
@click.option('--show', is_flag=True, default=True, help='Display plots interactively')
@click.option('--window', default=30, help='Rolling window size (default: 30)')
@click.option('--nlags', default=40, help='Number of lags for ACF/PACF (default: 40)')
def visualize_dataset(data: str, column: str, overview: bool, distribution: bool,
                     acf: bool, decomposition: bool, rolling: bool, spectral: bool,
                     dashboard: bool, all_plots: bool, output_dir: str,
                     save: bool, show: bool, window: int, nlags: int):
    """Comprehensive time series visualization following MIT standards."""

    logger.info("=" * 80)
    logger.info("TIME SERIES VISUALIZATION - Statistical Graphics")
    logger.info("=" * 80)

    # Load data
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

    # Initialize visualizer
    visualizer = TimeSeriesVisualizer(figsize=(16, 12))

    # Determine which plots to create
    if all_plots:
        overview = distribution = acf = decomposition = rolling = spectral = dashboard = True

    if not any([overview, distribution, acf, decomposition, rolling, spectral, dashboard]):
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
            save_path = Path(output_dir) / f"{column}_overview.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 2. Distribution Analysis
    if distribution:
        logger.info("Creating distribution analysis...")
        fig = visualizer.plot_distribution_analysis(df, column)
        plots_created.append(('distribution', fig))
        if save:
            save_path = Path(output_dir) / f"{column}_distribution.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 3. ACF/PACF
    if acf:
        logger.info(f"Creating ACF/PACF plots (nlags={nlags})...")
        fig = visualizer.plot_autocorrelation(df, column, nlags=nlags)
        plots_created.append(('acf', fig))
        if save:
            save_path = Path(output_dir) / f"{column}_acf_pacf.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 4. Decomposition
    if decomposition:
        logger.info("Creating trend-seasonal decomposition...")
        try:
            fig = visualizer.plot_decomposition(df, column)
            plots_created.append(('decomposition', fig))
            if save:
                save_path = Path(output_dir) / f"{column}_decomposition.png"
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
            save_path = Path(output_dir) / f"{column}_rolling_stats.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 6. Spectral Density
    if spectral:
        logger.info("Creating spectral density plot...")
        fig = visualizer.plot_spectral_density(df, column)
        plots_created.append(('spectral', fig))
        if save:
            save_path = Path(output_dir) / f"{column}_spectral.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {save_path}")

    # 7. Comprehensive Dashboard
    if dashboard:
        logger.info("Creating comprehensive dashboard...")
        save_path = Path(output_dir) / f"{column}_dashboard.png" if save else None
        fig = visualizer.plot_comprehensive_dashboard(df, column, save_path=str(save_path) if save_path else None)
        plots_created.append(('dashboard', fig))
        if save:
            logger.info(f"  ✓ Saved to {save_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"VISUALIZATION COMPLETE")
    logger.info("=" * 80)
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
