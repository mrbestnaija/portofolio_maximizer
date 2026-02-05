#!/usr/bin/env python3
"""
Analyze multi-ticker ensemble validation results.
Extracts GARCH weights, RMSE ratios, and model performance across tickers.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_log_file(log_path):
    """Parse pipeline log and extract ensemble metrics."""
    results = {
        'tickers': defaultdict(lambda: {
            'ensemble_builds': [],
            'garch_weights': [],
            'rmse_ratios': [],
            'confidence_scores': [],
        }),
        'summary': {
            'total_ensembles': 0,
            'garch_dominant': 0,
            'samossa_only': 0,
            'mixed': 0,
        }
    }

    current_ticker = None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Extract ticker from context
            ticker_match = re.search(r'Saved forecast for (\w+)', line)
            if ticker_match:
                current_ticker = ticker_match.group(1)

            # Extract ensemble weights
            ensemble_match = re.search(
                r"ENSEMBLE build_complete :: weights=({.*?}), confidence=({.*?})",
                line
            )
            if ensemble_match and current_ticker:
                weights_str = ensemble_match.group(1)
                confidence_str = ensemble_match.group(2)

                # Parse weights
                garch_weight = 0.0
                if "'garch':" in weights_str:
                    garch_match = re.search(r"'garch':\s*([\d.]+)", weights_str)
                    if garch_match:
                        garch_weight = float(garch_match.group(1))

                results['tickers'][current_ticker]['ensemble_builds'].append(weights_str)
                results['tickers'][current_ticker]['garch_weights'].append(garch_weight)
                results['tickers'][current_ticker]['confidence_scores'].append(confidence_str)

                results['summary']['total_ensembles'] += 1
                if garch_weight >= 0.5:
                    results['summary']['garch_dominant'] += 1
                elif garch_weight == 0:
                    results['summary']['samossa_only'] += 1
                else:
                    results['summary']['mixed'] += 1

            # Extract RMSE ratio
            rmse_match = re.search(r'ratio=([\d.]+)', line)
            if rmse_match and current_ticker:
                ratio = float(rmse_match.group(1))
                if ratio > 1.0:  # Only track ratios > 1.0 (ensemble worse than best)
                    results['tickers'][current_ticker]['rmse_ratios'].append(ratio)

    return results


def print_summary(results):
    """Print formatted summary of multi-ticker validation."""
    print("\n" + "="*80)
    print("MULTI-TICKER VALIDATION SUMMARY (Phase 7.3)")
    print("="*80)

    print("\n## Overall Metrics")
    print(f"Total Ensembles Built: {results['summary']['total_ensembles']}")
    print(f"GARCH-Dominant (>=50% weight): {results['summary']['garch_dominant']} "
          f"({100*results['summary']['garch_dominant']/max(results['summary']['total_ensembles'],1):.1f}%)")
    print(f"SAMoSSA-Only (0% GARCH): {results['summary']['samossa_only']} "
          f"({100*results['summary']['samossa_only']/max(results['summary']['total_ensembles'],1):.1f}%)")
    print(f"Mixed Ensemble: {results['summary']['mixed']} "
          f"({100*results['summary']['mixed']/max(results['summary']['total_ensembles'],1):.1f}%)")

    print("\n## Per-Ticker Breakdown")
    print("-" * 80)

    for ticker in sorted(results['tickers'].keys()):
        data = results['tickers'][ticker]

        print(f"\n### {ticker}")
        print(f"Ensemble Builds: {len(data['ensemble_builds'])}")

        if data['garch_weights']:
            avg_garch = sum(data['garch_weights']) / len(data['garch_weights'])
            max_garch = max(data['garch_weights'])
            min_garch = min(data['garch_weights'])
            print(f"GARCH Weight: avg={avg_garch:.2%}, max={max_garch:.2%}, min={min_garch:.2%}")

        if data['rmse_ratios']:
            avg_ratio = sum(data['rmse_ratios']) / len(data['rmse_ratios'])
            min_ratio = min(data['rmse_ratios'])
            max_ratio = max(data['rmse_ratios'])
            print(f"RMSE Ratio: avg={avg_ratio:.3f}, best={min_ratio:.3f}, worst={max_ratio:.3f}")

            # Check if target achieved
            target = 1.100
            if avg_ratio < target:
                print(f"  [OK] Target achieved! ({avg_ratio:.3f} < {target})")
            else:
                gap = avg_ratio - target
                improvement = ((1.682 - avg_ratio) / (1.682 - target)) * 100
                print(f"  [!] {gap:.3f} from target (reached {improvement:.1f}% of goal)")

    print("\n" + "="*80)
    print("## Validation Status")

    garch_success = results['summary']['garch_dominant'] > 0
    print(f"[OK] GARCH Integration: {'SUCCESS' if garch_success else 'FAILED'}")
    print(f"   - GARCH appearing in {results['summary']['garch_dominant']} ensembles")

    # Check if RMSE improved across tickers
    all_ratios = []
    for data in results['tickers'].values():
        all_ratios.extend(data['rmse_ratios'])

    if all_ratios:
        overall_avg = sum(all_ratios) / len(all_ratios)
        baseline = 1.682  # From Phase 7.3 diagnostics
        improvement = ((baseline - overall_avg) / baseline) * 100
        print(f"\n[OK] RMSE Improvement: {improvement:.1f}%")
        print(f"   - Baseline: {baseline:.3f}")
        print(f"   - Current: {overall_avg:.3f}")
        print(f"   - Target: 1.100")

    print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        log_path = Path("logs/phase7.3_multi_ticker_validation.log")
    else:
        log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Analyzing: {log_path}")
    results = parse_log_file(log_path)
    print_summary(results)


if __name__ == "__main__":
    main()
