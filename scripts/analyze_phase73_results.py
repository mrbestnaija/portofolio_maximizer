#!/usr/bin/env python3
"""
Analyze Phase 7.3 model improvement results.
"""
import sqlite3
import json
from pathlib import Path

def analyze_forecast_audits():
    """Check forecast RMSE ratios from today's run."""
    conn = sqlite3.connect('data/portfolio_maximizer.db')
    cursor = conn.cursor()

    # Get today's forecast audits
    cursor.execute('''
        SELECT ticker, ensemble_rmse, baseline_rmse, rmse_ratio,
               baseline_model, date
        FROM forecast_audits
        WHERE date >= '2026-01-19'
        ORDER BY ticker, date
    ''')

    audits = cursor.fetchall()
    conn.close()

    if not audits:
        print("No forecast audits found for 2026-01-19")
        return

    print("\nForecast RMSE Analysis (Phase 7.3):")
    print("=" * 70)

    # Group by ticker
    ticker_data = {}
    for ticker, ens_rmse, base_rmse, ratio, base_model, date in audits:
        if ticker not in ticker_data:
            ticker_data[ticker] = []
        ticker_data[ticker].append({
            'ens_rmse': ens_rmse,
            'base_rmse': base_rmse,
            'ratio': ratio,
            'base_model': base_model,
            'date': date
        })

    # Analyze each ticker
    for ticker, audits_list in sorted(ticker_data.items()):
        avg_ratio = sum(a['ratio'] for a in audits_list) / len(audits_list)
        latest = audits_list[-1]

        status = "PASS" if avg_ratio <= 1.1 else "FAIL"
        improvement = (1.68 - avg_ratio) / 1.68 * 100  # vs previous 1.68x

        print(f"\n{ticker}:")
        print(f"  Latest RMSE ratio: {latest['ratio']:.3f}x  ({status})")
        print(f"  Average RMSE ratio: {avg_ratio:.3f}x  ({len(audits_list)} audits)")
        print(f"  Improvement vs 1.68x: {improvement:.1f}%")
        print(f"  Ensemble RMSE: {latest['ens_rmse']:.4f}")
        print(f"  Baseline ({latest['base_model']}): {latest['base_rmse']:.4f}")

    # Overall summary
    all_ratios = [a['ratio'] for audits_list in ticker_data.values() for a in audits_list]
    overall_avg = sum(all_ratios) / len(all_ratios)
    passing = sum(1 for r in all_ratios if r <= 1.1)

    print("\n" + "=" * 70)
    print(f"OVERALL: {passing}/{len(all_ratios)} audits pass (<= 1.1x threshold)")
    print(f"Average RMSE ratio: {overall_avg:.3f}x")
    print(f"Improvement vs 1.68x baseline: {(1.68 - overall_avg) / 1.68 * 100:.1f}%")

    if overall_avg <= 1.1:
        print("\nVERDICT: PHASE 7.3 SUCCESS - RMSE target achieved!")
    elif overall_avg <= 1.3:
        print("\nVERDICT: SIGNIFICANT IMPROVEMENT - Close to target")
    else:
        print("\nVERDICT: NEEDS MORE WORK - Target not yet achieved")

def analyze_quant_validation():
    """Check quant validation pass rate."""
    log_file = Path("logs/signals/quant_validation.jsonl")

    if not log_file.exists():
        print("\nNo quant validation log found")
        return

    # Read last 100 lines
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Parse recent entries
    recent = []
    for line in lines[-100:]:
        try:
            record = json.loads(line)
            if record.get('timestamp', '').startswith('2026-01-19'):
                recent.append(record)
        except:
            continue

    if not recent:
        print("\nNo recent quant validation records found")
        return

    passed = sum(1 for r in recent if r.get('status') == 'PASS')
    total = len(recent)
    pass_rate = passed / total * 100 if total > 0 else 0

    print("\nQuant Validation Analysis:")
    print("=" * 70)
    print(f"Pass rate: {passed}/{total} = {pass_rate:.1f}%")

    # Count failure reasons
    failure_counts = {}
    for r in recent:
        if r.get('status') == 'FAIL':
            for criterion in r.get('failed_criteria', []):
                failure_counts[criterion] = failure_counts.get(criterion, 0) + 1

    if failure_counts:
        print("\nFailure reasons:")
        for criterion, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {criterion}: {count} times ({count/(total-passed)*100:.1f}% of failures)")

    # Show recent examples
    print("\nRecent signals:")
    for r in recent[-10:]:
        status = r.get('status', 'UNKNOWN')
        ticker = r.get('ticker', 'UNKNOWN')
        action = r.get('action', 'UNKNOWN')
        conf = r.get('confidence', 0) * 100
        criteria = ', '.join(r.get('failed_criteria', [])[:2])
        print(f"  {ticker:6s} {action:4s} {status:4s} (conf={conf:.0f}%) - {criteria}")

if __name__ == '__main__':
    analyze_forecast_audits()
    analyze_quant_validation()
