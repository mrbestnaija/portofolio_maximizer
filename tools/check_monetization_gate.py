"""
Monetization gate CLI.

Classifies readiness as BLOCKED | EXPERIMENTAL | READY_FOR_PUBLIC based on recent
performance metrics. Intended to run before any alert/report/monetization flow.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from etl.database_manager import DatabaseManager

# Thresholds from REWARD_TO_EFFORT_INTEGRATION_PLAN.md / MONITIZATION.md
ANNUAL_RETURN_MIN = 0.10
SHARPE_MIN = 1.0
MAX_DRAWDOWN_MAX = 0.25
MIN_TRADES = 30

logger = logging.getLogger(__name__)


@dataclass
class GateMetrics:
    annual_return: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    trade_count: int


@dataclass
class GateResult:
    status: str
    reasons: List[str]
    metrics: GateMetrics


def _fetch_performance_metrics(db: DatabaseManager, start_date: str) -> GateMetrics:
    """
    Pull the most recent performance_metrics row on/after start_date and trade count.

    Falls back to trade_executions summary for trade_count only when perf metrics
    are missing.
    """
    db.conn.row_factory = sqlite3.Row
    row = db.conn.execute(
        """
        SELECT metric_date, total_return_pct, sharpe_ratio, max_drawdown, num_trades
        FROM performance_metrics
        WHERE metric_date >= ?
        ORDER BY metric_date DESC
        LIMIT 1
        """,
        (start_date,),
    ).fetchone()

    perf_annual_return: Optional[float] = None
    perf_sharpe: Optional[float] = None
    perf_drawdown: Optional[float] = None
    perf_trades: int = 0

    if row:
        perf_annual_return = row["total_return_pct"]
        perf_sharpe = row["sharpe_ratio"]
        perf_drawdown = row["max_drawdown"]
        perf_trades = row["num_trades"] or 0

    # Always fetch trade count as a sanity check / fallback.
    trade_summary = db.get_performance_summary(start_date=start_date)
    trade_count = int(trade_summary.get("total_trades") or 0)
    if perf_trades == 0:
        perf_trades = trade_count

    return GateMetrics(
        annual_return=perf_annual_return,
        sharpe_ratio=perf_sharpe,
        max_drawdown=perf_drawdown,
        trade_count=perf_trades,
    )


def evaluate_gate(metrics: GateMetrics) -> GateResult:
    reasons: List[str] = []

    if metrics.annual_return is None:
        reasons.append("missing_annual_return")
    elif metrics.annual_return < ANNUAL_RETURN_MIN:
        reasons.append("annual_return_below_threshold")

    if metrics.sharpe_ratio is None:
        reasons.append("missing_sharpe_ratio")
    elif metrics.sharpe_ratio < SHARPE_MIN:
        reasons.append("sharpe_ratio_below_threshold")

    if metrics.max_drawdown is None:
        reasons.append("missing_max_drawdown")
    elif metrics.max_drawdown > MAX_DRAWDOWN_MAX:
        reasons.append("max_drawdown_above_threshold")

    if metrics.trade_count < MIN_TRADES:
        reasons.append("insufficient_trade_count")

    status = "READY_FOR_PUBLIC" if not reasons else "BLOCKED"
    return GateResult(status=status, reasons=reasons, metrics=metrics)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check monetization readiness gate.")
    parser.add_argument(
        "--db-path",
        default="data/portfolio_maximizer.db",
        help="Path to SQLite DB (default: data/portfolio_maximizer.db)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=365,
        help="Lookback window in days for performance metrics (default: 365).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        raise SystemExit(1)

    start_date = (date.today() - timedelta(days=args.window)).isoformat()

    with DatabaseManager(str(db_path)) as db:
        metrics = _fetch_performance_metrics(db, start_date=start_date)

    result = evaluate_gate(metrics)

    print("=== Monetization Gate ===")
    print(f"Window (days)     : {args.window}")
    print(f"Trades analyzed   : {metrics.trade_count}")
    print(f"Annual return     : {metrics.annual_return}")
    print(f"Sharpe ratio      : {metrics.sharpe_ratio}")
    print(f"Max drawdown      : {metrics.max_drawdown}")
    print(f"Thresholds        : return>={ANNUAL_RETURN_MIN}, sharpe>={SHARPE_MIN}, drawdown<={MAX_DRAWDOWN_MAX}, trades>={MIN_TRADES}")
    print(f"Gate status       : {result.status}")
    if result.reasons:
        print("Reasons:")
        for reason in result.reasons:
            print(f"  - {reason}")

    if result.status != "READY_FOR_PUBLIC":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
