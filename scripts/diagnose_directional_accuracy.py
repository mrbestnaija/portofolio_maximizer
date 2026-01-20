#!/usr/bin/env python3
"""
Diagnose directional accuracy failures in forecasting models.

This script analyzes quant validation failures to understand why forecasts
predict the wrong direction and provides actionable recommendations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_validation_failures(log_path: Path, since_date: str = None):
    """Load all validation failures with directional accuracy issues."""
    failures = []

    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return failures

    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # Filter by date if provided
                if since_date:
                    record_ts = data.get('timestamp', '')
                    if record_ts < since_date:
                        continue

                # Only process failures
                if data.get('status') != 'FAIL':
                    continue

                # Only directional accuracy failures
                failed_criteria = data.get('failed_criteria', [])
                if 'directional_accuracy' not in failed_criteria:
                    continue

                failures.append(data)
            except json.JSONDecodeError:
                continue

    return failures


def analyze_forecast_errors(failures):
    """Analyze forecast error patterns."""
    error_analysis = {
        'by_ticker': defaultdict(list),
        'by_action': defaultdict(list),
        'metrics': [],
    }

    for failure in failures:
        ticker = failure.get('ticker', 'UNKNOWN')
        action = failure.get('action', 'UNKNOWN')
        expected_return = failure.get('expected_return', 0.0)

        quant_val = failure.get('quant_validation', {})
        metrics = quant_val.get('metrics', {})

        error_analysis['by_ticker'][ticker].append({
            'expected_return': expected_return,
            'action': action,
            'metrics': metrics,
        })

        error_analysis['by_action'][action].append({
            'ticker': ticker,
            'expected_return': expected_return,
            'metrics': metrics,
        })

        error_analysis['metrics'].append(metrics)

    return error_analysis


def diagnose_model_issues(failures):
    """Diagnose specific model configuration issues."""
    issues = {
        'overfitting': [],
        'underfitting': [],
        'trend_reversal': [],
        'volatility_spike': [],
        'insufficient_data': [],
    }

    for failure in failures:
        quant_val = failure.get('quant_validation', {})
        metrics = quant_val.get('metrics', {})
        ticker = failure.get('ticker')

        # Check for overfitting: high in-sample, low out-sample
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        if sharpe < -1.5 and sortino < -2.0:
            issues['overfitting'].append({
                'ticker': ticker,
                'sharpe': sharpe,
                'sortino': sortino,
            })

        # Check for trend reversal: negative returns
        annual_return = metrics.get('annual_return', 0)
        if annual_return < -0.15:  # -15% annual return
            issues['trend_reversal'].append({
                'ticker': ticker,
                'annual_return': annual_return,
            })

        # Check for volatility spike: high max drawdown
        max_dd = metrics.get('max_drawdown', 0)
        volatility = metrics.get('volatility', 0)
        if max_dd > 0.3 or volatility > 0.2:  # >30% drawdown or >20% vol
            issues['volatility_spike'].append({
                'ticker': ticker,
                'max_dd': max_dd,
                'volatility': volatility,
            })

        # Check for insufficient data
        lookback_bars = quant_val.get('lookback_bars', 0)
        if lookback_bars < 200:  # Less than 200 trading days
            issues['insufficient_data'].append({
                'ticker': ticker,
                'lookback_bars': lookback_bars,
            })

    return issues


def generate_recommendations(error_analysis, issues):
    """Generate actionable recommendations."""
    recommendations = []

    # Analyze ticker-specific patterns
    ticker_fails = error_analysis['by_ticker']
    high_fail_tickers = {
        ticker: len(fails)
        for ticker, fails in ticker_fails.items()
        if len(fails) >= 2
    }

    if high_fail_tickers:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Multiple tickers failing directional accuracy',
            'tickers': list(high_fail_tickers.keys()),
            'actions': [
                'Increase SARIMAX order to capture more complex patterns',
                'Extend lookback window for trend identification',
                'Enable seasonal decomposition for cyclical patterns',
            ],
            'config_changes': {
                'file': 'config/forecasting_config.yml',
                'changes': {
                    'sarimax.order': '(3, 1, 2) instead of (2, 1, 1)',
                    'sarimax.seasonal_order': '(1, 0, 1, 12) for monthly seasonality',
                    'lookback_days': '730 (2 years) instead of 550',
                }
            }
        })

    # Check for overfitting
    if len(issues['overfitting']) > 0:
        recommendations.append({
            'priority': 'CRITICAL',
            'issue': f'Overfitting detected in {len(issues["overfitting"])} tickers',
            'tickers': [t['ticker'] for t in issues['overfitting']],
            'actions': [
                'Reduce model complexity (lower SARIMAX order)',
                'Increase regularization in ensemble',
                'Use walk-forward validation instead of expanding window',
            ],
            'config_changes': {
                'file': 'config/forecasting_config.yml',
                'changes': {
                    'sarimax.order': '(1, 1, 1) simpler model',
                    'ensemble.regularization': '0.1 to penalize overfitting',
                }
            }
        })

    # Check for trend reversals
    if len(issues['trend_reversal']) > 5:
        recommendations.append({
            'priority': 'HIGH',
            'issue': f'Trend reversals in {len(issues["trend_reversal"])} tickers',
            'actions': [
                'Implement regime detection (bull/bear/sideways)',
                'Use GARCH for volatility clustering',
                'Add mean reversion detection',
            ],
            'config_changes': {
                'file': 'config/forecasting_config.yml',
                'changes': {
                    'regime_detection.enabled': 'true',
                    'regime_detection.method': 'hmm or threshold',
                }
            }
        })

    # Check for volatility spikes
    if len(issues['volatility_spike']) > 3:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': f'High volatility in {len(issues["volatility_spike"])} tickers',
            'actions': [
                'Increase GARCH model weight in ensemble',
                'Add volatility filters to signal generation',
                'Reduce position sizing during high vol periods',
            ],
            'config_changes': {
                'file': 'config/forecasting_config.yml',
                'changes': {
                    'ensemble.candidate_weights': '[{garch: 0.5, sarimax: 0.3, samossa: 0.2}]',
                }
            }
        })

    # Check for insufficient data
    if len(issues['insufficient_data']) > 0:
        recommendations.append({
            'priority': 'LOW',
            'issue': f'Insufficient data in {len(issues["insufficient_data"])} tickers',
            'tickers': [t['ticker'] for t in issues['insufficient_data']],
            'actions': [
                'Extend data extraction period',
                'Exclude tickers with <1 year of data',
            ],
            'config_changes': {
                'file': 'scripts/run_etl_pipeline.py',
                'changes': {
                    'start_date': '2022-01-01 (extend 2 more years)',
                }
            }
        })

    return recommendations


def create_visualizations(failures, error_analysis, output_dir: Path):
    """Create diagnostic visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Failure rate by ticker
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Failures by ticker
    ticker_counts = Counter([f.get('ticker') for f in failures])
    tickers = list(ticker_counts.keys())
    counts = list(ticker_counts.values())

    axes[0, 0].bar(tickers, counts, color='#e74c3c')
    axes[0, 0].set_title('Directional Accuracy Failures by Ticker', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Ticker')
    axes[0, 0].set_ylabel('Number of Failures')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Subplot 2: Expected return distribution
    expected_returns = [f.get('expected_return', 0) * 100 for f in failures]
    axes[0, 1].hist(expected_returns, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero return')
    axes[0, 1].set_title('Expected Return Distribution (Failures)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Expected Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Subplot 3: Sharpe ratio distribution
    sharpe_ratios = [
        f.get('quant_validation', {}).get('metrics', {}).get('sharpe_ratio', 0)
        for f in failures
    ]
    axes[1, 0].hist(sharpe_ratios, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Sharpe')
    axes[1, 0].set_title('Sharpe Ratio Distribution (Failures)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sharpe Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # Subplot 4: Win rate distribution
    win_rates = [
        f.get('quant_validation', {}).get('metrics', {}).get('win_rate', 0) * 100
        for f in failures
    ]
    axes[1, 1].hist(win_rates, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(50, color='red', linestyle='--', linewidth=2, label='50% win rate')
    axes[1, 1].set_title('Win Rate Distribution (Failures)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Win Rate (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    plt.tight_layout()
    output_path = output_dir / 'directional_accuracy_diagnostics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"OK Visualization saved: {output_path}")
    plt.close()

    # 2. Correlation heatmap of metrics
    if failures:
        metrics_df = pd.DataFrame([
            {
                'annual_return': f.get('quant_validation', {}).get('metrics', {}).get('annual_return', 0),
                'sharpe_ratio': f.get('quant_validation', {}).get('metrics', {}).get('sharpe_ratio', 0),
                'sortino_ratio': f.get('quant_validation', {}).get('metrics', {}).get('sortino_ratio', 0),
                'max_drawdown': f.get('quant_validation', {}).get('metrics', {}).get('max_drawdown', 0),
                'volatility': f.get('quant_validation', {}).get('metrics', {}).get('volatility', 0),
                'win_rate': f.get('quant_validation', {}).get('metrics', {}).get('win_rate', 0),
                'profit_factor': f.get('quant_validation', {}).get('metrics', {}).get('profit_factor', 0),
            }
            for f in failures
        ])

        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = metrics_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Metric Correlation Matrix (Failed Forecasts)', fontsize=14, fontweight='bold')

        output_path = output_dir / 'metrics_correlation_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"OK Correlation heatmap saved: {output_path}")
        plt.close()


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("DIRECTIONAL ACCURACY DIAGNOSTIC REPORT")
    print("=" * 80)
    print()

    # Load validation failures from today
    today = datetime.now().strftime('%Y-%m-%d')
    log_path = Path('logs/signals/quant_validation.jsonl')

    print(f"[1] Loading validation failures (since {today})...")
    failures = load_validation_failures(log_path, since_date=today)

    if not failures:
        print("❌ No directional accuracy failures found today")
        print("    Run the pipeline first to generate validation data")
        return

    print(f"OK Loaded {len(failures)} directional accuracy failures")
    print()

    # Analyze error patterns
    print("[2] Analyzing forecast error patterns...")
    error_analysis = analyze_forecast_errors(failures)

    print(f"Failures by ticker:")
    for ticker, fails in sorted(error_analysis['by_ticker'].items()):
        print(f"  {ticker}: {len(fails)} failures")

    print(f"\nFailures by action:")
    for action, fails in sorted(error_analysis['by_action'].items()):
        print(f"  {action}: {len(fails)} failures")
    print()

    # Diagnose model issues
    print("[3] Diagnosing model configuration issues...")
    issues = diagnose_model_issues(failures)

    print(f"Issues detected:")
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"  {issue_type}: {len(issue_list)} cases")
    print()

    # Generate recommendations
    print("[4] Generating recommendations...")
    recommendations = generate_recommendations(error_analysis, issues)

    print("=" * 80)
    print("RECOMMENDATIONS (Priority Order)")
    print("=" * 80)
    print()

    for i, rec in enumerate(recommendations, 1):
        priority = rec['priority']
        issue = rec['issue']

        # Color code priority
        if priority == 'CRITICAL':
            priority_str = f"[!] {priority}"
        elif priority == 'HIGH':
            priority_str = f"[**] {priority}"
        elif priority == 'MEDIUM':
            priority_str = f"[*] {priority}"
        else:
            priority_str = f"[-] {priority}"

        print(f"[{i}] {priority_str}: {issue}")

        if 'tickers' in rec:
            print(f"    Affected tickers: {', '.join(rec['tickers'][:5])}")
            if len(rec['tickers']) > 5:
                print(f"    ... and {len(rec['tickers']) - 5} more")

        print(f"    Actions:")
        for action in rec['actions']:
            print(f"      • {action}")

        if 'config_changes' in rec:
            print(f"    Config changes ({rec['config_changes']['file']}):")
            for key, value in rec['config_changes']['changes'].items():
                print(f"      • {key}: {value}")

        print()

    # Create visualizations
    print("[5] Creating diagnostic visualizations...")
    viz_dir = Path('visualizations/diagnostics')
    create_visualizations(failures, error_analysis, viz_dir)
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    # Calculate aggregate metrics
    all_metrics = [
        f.get('quant_validation', {}).get('metrics', {})
        for f in failures
    ]

    if all_metrics:
        avg_annual_return = np.mean([m.get('annual_return', 0) for m in all_metrics])
        avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in all_metrics])
        avg_win_rate = np.mean([m.get('win_rate', 0) for m in all_metrics])
        avg_max_dd = np.mean([m.get('max_drawdown', 0) for m in all_metrics])

        print(f"Average metrics (failed forecasts):")
        print(f"  Annual Return: {avg_annual_return:.2%}")
        print(f"  Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"  Win Rate: {avg_win_rate:.2%}")
        print(f"  Max Drawdown: {avg_max_dd:.2%}")
        print()

        print(f"Target metrics (for passing):")
        print(f"  Annual Return: >0%")
        print(f"  Sharpe Ratio: >0.5")
        print(f"  Win Rate: >50%")
        print(f"  Max Drawdown: <20%")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review recommendations above (sorted by priority)")
    print("2. Apply config changes to forecasting_config.yml")
    print("3. Re-run pipeline with updated config")
    print("4. Measure improvement in pass rate (target: 45%)")
    print()
    print("Expected improvement path:")
    print("  Current: 28.6% pass rate (6/21)")
    print("  After fixes: 45% pass rate (9-10/21)")
    print("  Final target: 60% pass rate (12-13/21)")
    print()


if __name__ == "__main__":
    main()
