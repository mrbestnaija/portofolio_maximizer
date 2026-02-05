#!/usr/bin/env python3
"""Comprehensive Time Series Dataset Analysis Script.

Usage:
    python scripts/analyze_dataset.py --data data/raw/extraction_*.parquet
    python scripts/analyze_dataset.py --data data/training/train_*.parquet --stationarity
    python scripts/analyze_dataset.py --data data/*.parquet --full-analysis --visualize
"""

import sys
from pathlib import Path
import click
import logging
import json
import os
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MPLBACKEND", "Agg")

from etl.time_series_analyzer import TimeSeriesDatasetAnalyzer
from etl.visualizer import TimeSeriesVisualizer
import matplotlib.pyplot as plt
import seaborn as sns

logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(logs_dir / "analyze_dataset.log"),
    filemode="a",
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)


@click.command()
@click.option('--data', required=True, help='Path to dataset file (parquet or csv)')
@click.option('--name', default=None, help='Dataset name (auto-detected if not provided)')
@click.option('--description', default='', help='Dataset description')
@click.option('--date-column', default=None, help='Name of date column (auto-detected if not provided)')
@click.option('--stationarity', is_flag=True, help='Perform stationarity tests on numeric columns')
@click.option('--autocorr', is_flag=True, help='Compute autocorrelation functions')
@click.option('--nlags', default=40, help='Number of lags for autocorrelation (default: 40)')
@click.option('--full-analysis', is_flag=True, help='Perform all available analyses')
@click.option('--visualize', is_flag=True, help='Generate visualization plots')
@click.option('--output', default=None, help='Output path for JSON report')
def analyze_dataset(data: str, name: Optional[str], description: str,
                   date_column: Optional[str], stationarity: bool,
                   autocorr: bool, nlags: int, full_analysis: bool,
                   visualize: bool, output: Optional[str]):
    """Comprehensive time series dataset analysis following MIT standards."""

    logger.info("=" * 80)
    logger.info("TIME SERIES DATASET ANALYZER - MIT Statistical Learning Standards")
    logger.info("=" * 80)

    # Auto-detect dataset name from file
    if name is None:
        name = Path(data).stem

    # Initialize analyzer
    analyzer = TimeSeriesDatasetAnalyzer(dataset_name=name, description=description)

    # 1. Load and inspect data
    logger.info("\n📊 PHASE 1: DATA LOADING AND INSPECTION")
    logger.info("-" * 80)
    inspection = analyzer.load_and_inspect_data(data)
    if inspection is None:
        logger.error("Failed to load dataset. Exiting.")
        sys.exit(1)

    print(f"\n✓ Dataset loaded successfully")
    print(f"  • Dimensions: {inspection['total_rows']} rows × {inspection['total_columns']} columns")
    print(f"  • Memory: {inspection['memory_usage_mb']:.2f} MB")
    print(f"  • Numeric columns: {len(inspection['numeric_columns'])}")
    print(f"  • Categorical columns: {len(inspection['categorical_columns'])}")

    # 2. Missing data analysis
    logger.info("\n🔍 PHASE 2: MISSING DATA ANALYSIS")
    logger.info("-" * 80)
    missing = analyzer.analyze_missing_data()

    print(f"\n✓ Missing data analysis complete")
    print(f"  • Overall missing rate: {missing['overall_missing_rate']:.4f}%")
    print(f"  • Category: {missing['missing_category']}")
    print(f"  • Severity: {missing['severity']}")
    print(f"  • Pattern entropy: {missing['pattern_entropy']:.4f}")

    if missing['columns_with_missing']:
        print(f"  • Columns with missing data:")
        for col in missing['columns_with_missing'][:5]:
            pct = missing['missing_percentage_by_column'][col]
            print(f"    - {col}: {pct:.2f}%")

    # 3. Temporal structure analysis
    logger.info("\n🕒 PHASE 3: TEMPORAL STRUCTURE IDENTIFICATION")
    logger.info("-" * 80)
    temporal = analyzer.identify_temporal_structure(date_column=date_column)

    if temporal and temporal.get('is_time_series'):
        print(f"\n✓ Time series structure identified")
        print(f"  • Sampling frequency: {temporal['sampling_frequency']}")
        print(f"  • Sampling period: {temporal['sampling_period_days']:.2f} days")
        print(f"  • Nyquist frequency: {temporal['nyquist_frequency']:.6f} cycles/day")
        print(f"  • Time span: {temporal['time_span']['start']} to {temporal['time_span']['end']}")
        print(f"  • Total periods: {temporal['time_span']['total_periods']}")
        print(f"  • Duration: {temporal['time_span']['duration_days']} days")
        print(f"  • Temporal regularity: {temporal['temporal_regularity']:.4f}")
        print(f"  • Gaps detected: {temporal['temporal_gaps_detected']}")
    else:
        print(f"\n⚠️  No clear time series structure detected")

    # 4. Statistical summary
    logger.info("\n📈 PHASE 4: STATISTICAL SUMMARY")
    logger.info("-" * 80)
    stats_summary = analyzer.statistical_summary()

    print(f"\n✓ Statistical summary computed")
    numeric_cols = list(stats_summary['mean'].keys())
    if numeric_cols:
        print(f"  • Numeric columns analyzed: {len(numeric_cols)}")
        print(f"\n  First numeric column statistics ({numeric_cols[0]}):")
        col = numeric_cols[0]
        print(f"    - Mean (μ): {stats_summary['mean'][col]:.6f}")
        print(f"    - Std (σ): {stats_summary['std'][col]:.6f}")
        print(f"    - Skewness (γ₁): {stats_summary['skewness'][col]:.6f}")
        print(f"    - Kurtosis (γ₂): {stats_summary['kurtosis'][col]:.6f}")
        if col in stats_summary['normality_tests']:
            norm_test = stats_summary['normality_tests'][col]
            print(f"    - Jarque-Bera p-value: {norm_test['p_value']:.6f}")
            print(f"    - Normal distribution: {norm_test['is_normal']}")

    # 5. Stationarity tests (if requested or full analysis)
    if stationarity or full_analysis:
        logger.info("\n🔬 PHASE 5: STATIONARITY TESTING (ADF)")
        logger.info("-" * 80)

        numeric_cols = inspection['numeric_columns'][:3]  # Limit to first 3 for efficiency
        print(f"\n✓ Testing stationarity for {len(numeric_cols)} columns")

        for col in numeric_cols:
            result = analyzer.test_stationarity(col)
            if result:
                print(f"\n  {col}:")
                print(f"    - ADF statistic: {result['adf_statistic']:.6f}")
                print(f"    - p-value: {result['p_value']:.6f}")
                print(f"    - Conclusion: {result['conclusion']}")
                print(f"    - Critical values:")
                for level, cv in result['critical_values'].items():
                    print(f"      • {level}: {cv:.6f}")

    # 6. Autocorrelation analysis (if requested or full analysis)
    if autocorr or full_analysis:
        logger.info("\n📊 PHASE 6: AUTOCORRELATION ANALYSIS")
        logger.info("-" * 80)

        numeric_cols = inspection['numeric_columns'][:2]  # Limit to first 2
        print(f"\n✓ Computing ACF/PACF for {len(numeric_cols)} columns")

        for col in numeric_cols:
            result = analyzer.compute_autocorrelation(col, nlags=nlags)
            if result:
                print(f"\n  {col}:")
                print(f"    - Significant ACF lags: {result['significant_lags_acf'][:10]}")
                print(f"    - Significant PACF lags: {result['significant_lags_pacf'][:10]}")
                print(f"    - 95% confidence interval: ±{result['confidence_interval']:.6f}")

    # 7. Generate comprehensive report
    logger.info("\n📋 PHASE 7: REPORT GENERATION")
    logger.info("-" * 80)
    report = analyzer.generate_report()

    # Save report to JSON if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✓ Report saved to: {output_path}")
        logger.info(f"Report saved to: {output_path}")
    else:
        print(f"\n✓ Report generated (use --output to save JSON)")

    # 8. Visualization (if requested)
    if visualize:
        logger.info("\n📊 PHASE 8: VISUALIZATION GENERATION")
        logger.info("-" * 80)

        create_analysis_plots(analyzer)
        print(f"\n✓ Visualizations created")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {name}")
    print(f"Rows: {inspection['total_rows']:,}")
    print(f"Columns: {inspection['total_columns']}")
    print(f"Missing data: {missing['overall_missing_rate']:.4f}% ({missing['severity']})")
    if temporal and temporal.get('is_time_series'):
        print(f"Temporal structure: {temporal['sampling_frequency']}")
    print(f"{'='*80}")


def create_analysis_plots(analyzer: TimeSeriesDatasetAnalyzer):
    """Generate comprehensive visualization plots using TimeSeriesVisualizer."""

    data = analyzer.data
    results = analyzer.analysis_results

    # Initialize visualizer
    visualizer = TimeSeriesVisualizer(figsize=(18, 12))

    # Get first numeric column
    numeric_cols = results['inspection']['numeric_columns']
    if not numeric_cols:
        print("⚠️  No numeric columns to visualize")
        return

    column = numeric_cols[0]
    print(f"\nGenerating visualizations for: {column}")

    # Create comprehensive dashboard
    fig = visualizer.plot_comprehensive_dashboard(data, column)

    # Keep the old basic visualization for comparison
    fig_basic = plt.figure(figsize=(18, 12))

    # Plot 1: Missing data heatmap
    plt.subplot(3, 3, 1)
    if data.isnull().sum().sum() > 0:
        sns.heatmap(data.isnull(), cbar=True, yticklabels=False, cmap='YlOrRd')
        plt.title('Missing Data Pattern', fontsize=12, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Missing Data\n✓', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=16, fontweight='bold', color='green')
        plt.title('Missing Data Status', fontsize=12, fontweight='bold')
        plt.axis('off')

    # Plot 2: Data type distribution
    plt.subplot(3, 3, 2)
    numeric_cols = len(results['inspection']['numeric_columns'])
    categorical_cols = len(results['inspection']['categorical_columns'])
    plt.pie([numeric_cols, categorical_cols],
           labels=['Numeric', 'Categorical'],
           autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
    plt.title('Column Type Distribution', fontsize=12, fontweight='bold')

    # Plot 3: Sample size by column
    plt.subplot(3, 3, 3)
    valid_counts = data.count()
    plt.barh(range(len(valid_counts)), valid_counts.values, color='steelblue')
    plt.xlabel('Valid Observations', fontsize=10)
    plt.ylabel('Column Index', fontsize=10)
    plt.title('Data Completeness by Column', fontsize=12, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # Plot 4-6: Time series of first 3 numeric columns
    numeric_cols = results['inspection']['numeric_columns'][:3]
    for idx, col in enumerate(numeric_cols, start=4):
        plt.subplot(3, 3, idx)
        data[col].plot(color='navy', linewidth=1.5, alpha=0.7)
        plt.title(f'Time Series: {col}', fontsize=10, fontweight='bold')
        plt.xlabel('Time Index', fontsize=9)
        plt.ylabel('Value', fontsize=9)
        plt.grid(alpha=0.3)

    # Plot 7-9: Distributions of first 3 numeric columns
    for idx, col in enumerate(numeric_cols, start=7):
        plt.subplot(3, 3, idx)
        data[col].hist(bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
        plt.title(f'Distribution: {col}', fontsize=10, fontweight='bold')
        plt.xlabel('Value', fontsize=9)
        plt.ylabel('Frequency', fontsize=9)
        plt.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Comprehensive Analysis: {analyzer.dataset_name}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


if __name__ == '__main__':
    analyze_dataset()
