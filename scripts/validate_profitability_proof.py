#!/usr/bin/env python3
"""
Profitability Proof Validator

Rigorous validation that performance metrics represent REAL profitability.
Part of the Critical Profitability Analysis & Remediation Plan (Phase 6).

Usage:
    python scripts/validate_profitability_proof.py [--db data/portfolio_maximizer.db]
"""

import argparse
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import yaml


def load_requirements(config_path: str = "config/profitability_proof_requirements.yml") -> Dict[str, Any]:
    """Load profitability proof requirements from config."""
    default_requirements = {
        "data_quality": {
            "min_data_source_coverage": 1.0,
            "max_synthetic_ticker_pct": 0.0,
            "allowed_execution_modes": ["live", "paper"]
        },
        "statistical_significance": {
            "min_closed_trades": 30,
            "min_trading_days": 21,
            "max_win_rate": 0.85,
            "min_win_rate": 0.35
        },
        "performance": {
            "min_profit_factor": 1.1,
            "max_drawdown": 0.30,
            "min_sharpe_ratio": 0.3
        },
        "audit_trail": {
            "require_pipeline_id": False,
            "require_run_id": False,
            "require_entry_exit_matching": True
        }
    }

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f).get("profitability_proof_requirements", default_requirements)
    return default_requirements


def check_null_data_sources(cursor) -> float:
    """Return percentage of trades with NULL data_source."""
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN data_source IS NULL OR data_source = '' THEN 1 ELSE 0 END) as null_count
        FROM trade_executions
    """)
    row = cursor.fetchone()
    if row[0] == 0:
        return 0.0
    return row[1] / row[0]


def count_synthetic_tickers(cursor) -> int:
    """Count trades with synthetic test tickers (SYN0, SYN1, etc.)."""
    cursor.execute("""
        SELECT COUNT(*) FROM trade_executions
        WHERE ticker LIKE 'SYN%' AND ticker GLOB 'SYN[0-9]*'
    """)
    return cursor.fetchone()[0]


def calculate_win_rate(cursor, production_only: bool = True) -> Optional[float]:
    """Calculate win rate from closed positions."""
    # Note: is_test_data column may not exist in older schemas
    query = """
        SELECT
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses
        FROM trade_executions
        WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
    """
    # Skip is_test_data filter since column may not exist
    # if production_only:
    #     query += " AND (is_test_data = 0 OR is_test_data IS NULL)"

    cursor.execute(query)
    row = cursor.fetchone()

    if row[0] is None or row[1] is None:
        return None

    total = (row[0] or 0) + (row[1] or 0)
    if total == 0:
        return None
    return row[0] / total


def count_actions(cursor) -> Dict[str, int]:
    """Count BUY, SELL, and HOLD actions."""
    cursor.execute("""
        SELECT action, COUNT(*) FROM trade_executions GROUP BY action
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_trade_stats(cursor) -> Dict[str, Any]:
    """Get comprehensive trade statistics."""
    cursor.execute("""
        SELECT
            COUNT(*) as total_trades,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT DATE(trade_date)) as trading_days,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(CASE WHEN realized_pnl IS NULL OR realized_pnl = 0 THEN 1 ELSE 0 END) as no_pnl_trades,
            COALESCE(SUM(realized_pnl), 0) as total_pnl,
            COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END), 0) as gross_profit,
            COALESCE(SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END), 0) as gross_loss
        FROM trade_executions
    """)
    row = cursor.fetchone()

    return {
        "total_trades": row[0],
        "unique_tickers": row[1],
        "trading_days": row[2],
        "winning_trades": row[3] or 0,
        "losing_trades": row[4] or 0,
        "no_pnl_trades": row[5] or 0,
        "total_pnl": row[6],
        "gross_profit": row[7],
        "gross_loss": row[8],
        "profit_factor": row[7] / row[8] if row[8] and row[8] > 0 else None
    }


def validate_profitability_proof(db_path: str) -> Dict[str, Any]:
    """
    Validate whether performance metrics represent REAL profitability.

    Returns:
        {
            "is_profitable": bool,
            "is_proof_valid": bool,
            "violations": List[str],
            "warnings": List[str],
            "metrics": Dict[str, Any],
            "recommendations": List[str]
        }
    """
    requirements = load_requirements()
    violations = []
    warnings = []
    recommendations = []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get trade statistics
    stats = get_trade_stats(cursor)
    actions = count_actions(cursor)

    # Check if we have any trades
    if stats["total_trades"] == 0:
        violations.append("No trades found in database")
        recommendations.append("Run trading pipeline with --execution-mode live to generate trades")
        return {
            "is_profitable": False,
            "is_proof_valid": False,
            "violations": violations,
            "warnings": warnings,
            "metrics": {
                **stats,
                "buy_count": 0,
                "sell_count": 0,
                "hold_count": 0,
                "win_rate": None,
                "null_data_source_pct": 0,
                "synthetic_ticker_count": 0
            },
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }

    # 1. Data Source Coverage
    null_source_pct = check_null_data_sources(cursor)
    min_coverage = requirements["data_quality"]["min_data_source_coverage"]
    if null_source_pct > (1 - min_coverage):
        violations.append(f"Data source NULL for {null_source_pct:.1%} of trades (max allowed: {(1-min_coverage):.1%})")

    # 2. Synthetic Contamination
    synthetic_count = count_synthetic_tickers(cursor)
    max_synthetic = requirements["data_quality"]["max_synthetic_ticker_pct"]
    if synthetic_count > 0 and max_synthetic == 0:
        violations.append(f"Found {synthetic_count} synthetic ticker trades (SYN*)")
        recommendations.append("Run: DELETE FROM trade_executions WHERE ticker LIKE 'SYN%'")

    # 3. Win Rate Reality Check
    win_rate = calculate_win_rate(cursor)
    if win_rate is not None:
        max_win_rate = requirements["statistical_significance"]["max_win_rate"]
        min_win_rate = requirements["statistical_significance"]["min_win_rate"]

        if win_rate > max_win_rate:
            violations.append(f"Win rate {win_rate:.1%} exceeds {max_win_rate:.1%} - statistically suspicious")
            recommendations.append("Investigate data quality - 100% win rate is impossible in real trading")
        elif win_rate < min_win_rate:
            warnings.append(f"Win rate {win_rate:.1%} below {min_win_rate:.1%} - strategy may be unprofitable")

    # 4. Position Lifecycle Completeness
    buy_count = actions.get("BUY", 0)
    sell_count = actions.get("SELL", 0)

    if buy_count > 0 and sell_count == 0:
        violations.append(f"Found {buy_count} BUY actions but 0 SELL actions - positions never closed")
        recommendations.append("Implement proper exit tracking in paper_trading_engine.py")
    elif buy_count > sell_count * 5:
        warnings.append(f"{buy_count} BUY vs {sell_count} SELL - positions may not be closing properly")

    # 5. Minimum Closed Trades
    closed_trades = stats["winning_trades"] + stats["losing_trades"]
    min_trades = requirements["statistical_significance"]["min_closed_trades"]
    if closed_trades < min_trades:
        violations.append(f"Only {closed_trades} closed trades (need {min_trades} for statistical significance)")
        recommendations.append(f"Continue trading to accumulate {min_trades - closed_trades} more closed positions")

    # 6. Minimum Trading Days
    min_days = requirements["statistical_significance"]["min_trading_days"]
    if stats["trading_days"] and stats["trading_days"] < min_days:
        warnings.append(f"Only {stats['trading_days']} trading days (need {min_days} for validation)")

    # 7. Profit Factor
    if stats["profit_factor"] is not None:
        min_pf = requirements["performance"]["min_profit_factor"]
        if stats["profit_factor"] < min_pf:
            warnings.append(f"Profit factor {stats['profit_factor']:.2f} below {min_pf} threshold")

    # Determine overall validity
    is_proof_valid = len(violations) == 0
    is_profitable = stats["total_pnl"] > 0 and is_proof_valid

    conn.close()

    return {
        "is_profitable": is_profitable,
        "is_proof_valid": is_proof_valid,
        "violations": violations,
        "warnings": warnings,
        "metrics": {
            **stats,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": actions.get("HOLD", 0),
            "win_rate": win_rate,
            "null_data_source_pct": null_source_pct,
            "synthetic_ticker_count": synthetic_count
        },
        "recommendations": recommendations,
        "timestamp": datetime.utcnow().isoformat()
    }


def print_report(result: Dict[str, Any]) -> None:
    """Print formatted validation report."""
    print("\n" + "=" * 60)
    print("PROFITABILITY PROOF VALIDATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {result['timestamp']}")
    print()

    # Overall Status
    if result["is_proof_valid"]:
        status = "[PASS] Profitability proof is VALID"
    else:
        status = "[FAIL] Profitability proof is INVALID"
    print(status)
    print()

    # Metrics Summary
    metrics = result["metrics"]
    print("--- METRICS ---")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Unique Tickers: {metrics['unique_tickers']}")
    print(f"Trading Days: {metrics['trading_days']}")
    print(f"Actions: BUY={metrics['buy_count']}, SELL={metrics['sell_count']}, HOLD={metrics['hold_count']}")
    print(f"Wins/Losses: {metrics['winning_trades']}/{metrics['losing_trades']} (No P&L: {metrics['no_pnl_trades']})")
    if metrics["win_rate"] is not None:
        print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    if metrics["profit_factor"] is not None:
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print()

    # Data Quality
    print("--- DATA QUALITY ---")
    print(f"NULL Data Source: {metrics['null_data_source_pct']:.1%}")
    print(f"Synthetic Tickers: {metrics['synthetic_ticker_count']}")
    print()

    # Violations
    if result["violations"]:
        print("--- VIOLATIONS ---")
        for v in result["violations"]:
            print(f"  [X] {v}")
        print()

    # Warnings
    if result["warnings"]:
        print("--- WARNINGS ---")
        for w in result["warnings"]:
            print(f"  [!] {w}")
        print()

    # Recommendations
    if result["recommendations"]:
        print("--- RECOMMENDATIONS ---")
        for r in result["recommendations"]:
            print(f"  -> {r}")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate profitability proof")
    parser.add_argument(
        "--db",
        default="data/portfolio_maximizer.db",
        help="Path to database file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted report"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1

    result = validate_profitability_proof(str(db_path))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)

    return 0 if result["is_proof_valid"] else 1


if __name__ == "__main__":
    exit(main())
