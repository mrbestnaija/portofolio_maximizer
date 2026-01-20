#!/usr/bin/env python3
"""
Analyze the most recent pipeline run to diagnose signal pass rates and blockers.
"""
import json
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

def analyze_quant_validation(log_path: Path, since_timestamp: str = None):
    """Analyze quant validation log file."""
    if not log_path.exists():
        print(f"⚠️  Quant validation log not found: {log_path}")
        return None

    results = {
        'total': 0,
        'pass': 0,
        'fail': 0,
        'by_ticker': defaultdict(lambda: {'pass': 0, 'fail': 0}),
        'failure_reasons': Counter(),
        'all_records': []
    }

    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # Filter by timestamp if provided
                if since_timestamp:
                    record_ts = data.get('timestamp', '')
                    if record_ts < since_timestamp:
                        continue

                status = data.get('status', 'UNKNOWN')
                ticker = data.get('ticker', 'UNKNOWN')

                results['total'] += 1
                results['all_records'].append(data)

                if status == 'PASS':
                    results['pass'] += 1
                    results['by_ticker'][ticker]['pass'] += 1
                elif status == 'FAIL':
                    results['fail'] += 1
                    results['by_ticker'][ticker]['fail'] += 1

                    # Collect failure reasons
                    failed_criteria = data.get('failed_criteria', [])
                    for criterion in failed_criteria:
                        results['failure_reasons'][criterion] += 1
            except json.JSONDecodeError:
                continue

    return results

def analyze_pipeline_log(log_path: Path):
    """Analyze pipeline output log for blockers."""
    if not log_path.exists():
        print(f"⚠️  Pipeline log not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Count barbell policy decisions
    policy_counts = Counter()
    for line in content.split('\n'):
        if 'ENSEMBLE policy_decision' in line:
            if 'RESEARCH_ONLY' in line:
                policy_counts['RESEARCH_ONLY'] += 1
            elif 'DISABLE_DEFAULT' in line:
                policy_counts['DISABLE_DEFAULT'] += 1
            elif 'APPROVED' in line:
                policy_counts['APPROVED'] += 1

    # Check for LLM signals
    llm_signals = []
    for line in content.split('\n'):
        if 'Signal generated for' in line:
            llm_signals.append(line)

    return {
        'policy_counts': policy_counts,
        'llm_signals': llm_signals,
    }

def check_database_trades(db_path: Path, since_date: str = None):
    """Check if any trades were recorded."""
    if not db_path.exists():
        print(f"⚠️  Database not found: {db_path}")
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM trade_executions"
    if since_date:
        query += f" WHERE timestamp >= '{since_date}'"

    cursor.execute(query)
    total_trades = cursor.fetchone()[0]

    # Check production trades if view exists
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='production_trades'")
        if cursor.fetchone():
            query_prod = "SELECT COUNT(*) FROM production_trades"
            if since_date:
                query_prod += f" WHERE timestamp >= '{since_date}'"
            cursor.execute(query_prod)
            production_trades = cursor.fetchone()[0]
        else:
            production_trades = None
    except Exception:
        production_trades = None

    conn.close()

    return {
        'total_trades': total_trades,
        'production_trades': production_trades,
    }

def main():
    """Main analysis function."""
    print("=" * 80)
    print("PIPELINE RUN ANALYSIS")
    print("=" * 80)
    print()

    # Analyze pipeline log
    pipeline_log = Path("pipeline_test_output.log")
    print("[1] PIPELINE LOG ANALYSIS")
    print("-" * 80)

    pipeline_data = analyze_pipeline_log(pipeline_log)
    if pipeline_data:
        print(f"Ensemble Policy Decisions:")
        for status, count in pipeline_data['policy_counts'].items():
            print(f"  {status}: {count}")
        print()
        print(f"LLM Signals Generated: {len(pipeline_data['llm_signals'])}")
        for signal in pipeline_data['llm_signals']:
            print(f"  {signal.split(' - ')[-1] if ' - ' in signal else signal.strip()}")
    print()

    # Analyze quant validation (filter to today)
    today = datetime.now().strftime('%Y-%m-%d')
    quant_log = Path("logs/signals/quant_validation.jsonl")
    print("[2] QUANT VALIDATION ANALYSIS (Today Only)")
    print("-" * 80)

    quant_data = analyze_quant_validation(quant_log, since_timestamp=today)
    if quant_data and quant_data['total'] > 0:
        total = quant_data['total']
        passed = quant_data['pass']
        failed = quant_data['fail']
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"Total Signals Validated: {total}")
        print(f"PASS: {passed} ({pass_rate:.1f}%)")
        print(f"FAIL: {failed} ({100 - pass_rate:.1f}%)")
        print()

        print("Per-Ticker Results:")
        for ticker, counts in sorted(quant_data['by_ticker'].items()):
            t_total = counts['pass'] + counts['fail']
            t_rate = (counts['pass'] / t_total * 100) if t_total > 0 else 0
            print(f"  {ticker}: {counts['pass']}/{t_total} ({t_rate:.1f}%)")
        print()

        if quant_data['failure_reasons']:
            print("Top 10 Failure Reasons:")
            for reason, count in quant_data['failure_reasons'].most_common(10):
                pct = (count / failed * 100) if failed > 0 else 0
                print(f"  {reason}: {count} ({pct:.1f}%)")
    else:
        print("❌ NO SIGNALS VALIDATED TODAY")
        print()
        print("Possible reasons:")
        print("  1. Ensemble forecasts blocked by barbell policy (RESEARCH_ONLY/DISABLE)")
        print("  2. LLM signals did not trigger quant validation")
        print("  3. Signal generation stage was skipped")
    print()

    # Check database
    db_path = Path("data/portfolio_maximizer.db")
    print("[3] DATABASE ANALYSIS (Today Only)")
    print("-" * 80)

    db_data = check_database_trades(db_path, since_date=today)
    if db_data:
        print(f"Total Trades Today: {db_data['total_trades']}")
        if db_data['production_trades'] is not None:
            print(f"Production Trades Today: {db_data['production_trades']}")
    print()

    # Summary and diagnosis
    print("=" * 80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)

    if pipeline_data:
        blocked = pipeline_data['policy_counts'].get('RESEARCH_ONLY', 0) + pipeline_data['policy_counts'].get('DISABLE_DEFAULT', 0)
        approved = pipeline_data['policy_counts'].get('APPROVED', 0)

        if blocked > 0 and approved == 0:
            print("⚠️  CRITICAL: All ensemble forecasts blocked by barbell policy")
            print()
            print("ROOT CAUSE:")
            print("  - Barbell policy requires 2% margin lift over baseline")
            print("  - Current ensemble provides insufficient improvement")
            print("  - RMSE regression detected (error > 1.1x baseline)")
            print()
            print("SOLUTIONS:")
            print("  1. Disable barbell policy temporarily to test signal quality")
            print("  2. Improve model hyperparameters (SAMoSSA window size, SARIMAX order)")
            print("  3. Increase lookback data (current: 18 months → try 24 months)")
            print("  4. Use ensemble override to force-enable promising candidates")
            print()

    if quant_data and quant_data['total'] == 0:
        if pipeline_data and len(pipeline_data['llm_signals']) > 0:
            print("⚠️  WARNING: LLM signals generated but NOT validated")
            print()
            print("ROOT CAUSE:")
            print("  - LLM signals might bypass quant validation in current config")
            print("  - Check if 'quant_validation' is enabled for LLM signals")
            print()
            print("SOLUTION:")
            print("  - Review config/llm_config.yml: ensure quant_validation is enabled")
            print("  - Check logs/signals/ for LLM-specific signal logs")
        else:
            print("⚠️  WARNING: No signals generated at all")
            print()
            print("ROOT CAUSE:")
            print("  - Both ensemble (blocked by policy) and LLM failed to generate")
            print("  - Or signal generation stage was skipped")
            print()
            print("SOLUTION:")
            print("  - Check pipeline execution order in logs")
            print("  - Verify LLM server is running: curl http://localhost:11434/api/tags")

    print()

if __name__ == "__main__":
    main()
