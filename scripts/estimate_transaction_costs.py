#!/usr/bin/env python3
"""
estimate_transaction_costs.py
-----------------------------

Estimate empirical transaction costs (slippage + commission) from realised
trades and emit grouped statistics for use in config tuning.

Design goals:
- Read-only helper over trade_executions; does not modify schema or configs.
- Group results by ticker or simple derived groups (e.g. prefix-based
  asset buckets) so callers can map them onto execution / TS configs.
- Produce JSON outputs that automation or notebooks can consume.

NOTE:
- The current schema does not store explicit mid-price-at-fill, so this
  scaffold treats `commission` as the direct transaction cost signal and
  reports realised_pnl distributions separately. A future refinement can
  extend trade_executions with mid-price snapshots to measure slippage.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in __import__("sys").modules["sys"].path:
    __import__("sys").modules["sys"].path.insert(0, str(ROOT_PATH))

from etl.database_manager import DatabaseManager  # noqa: E402

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class CostStats:
    group: str
    trades: int
    commission_median: float
    commission_mean: float
    commission_p95: float
    pnl_median: float
    pnl_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "trades": self.trades,
            "commission_median": self.commission_median,
            "commission_mean": self.commission_mean,
            "commission_p95": self.commission_p95,
            "pnl_median": self.pnl_median,
            "pnl_mean": self.pnl_mean,
        }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    q = max(0.0, min(100.0, q))
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    weight = pos - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


def _group_key(ticker: str, mode: str) -> str:
    ticker = (ticker or "").upper()
    if mode == "ticker":
        return ticker or "UNKNOWN"
    # Simple prefix-based asset buckets; callers can override by
    # post-processing the JSON if they prefer more nuanced classes.
    if mode == "asset_class":
        if ticker.endswith("=X"):
            return "FX"
        if ticker.endswith("-USD") or ticker in {"BTC", "ETH"}:
            return "CRYPTO"
        if "^" in ticker:
            return "INDEX"
        if any(ticker.endswith(suffix) for suffix in (".NS", ".TW", ".L")):
            return "INTL_EQUITY"
        return "US_EQUITY"
    return "UNKNOWN"


def _load_trades(
    db: DatabaseManager,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    query = """
        SELECT ticker, trade_date, commission, realized_pnl
        FROM trade_executions
        WHERE realized_pnl IS NOT NULL
    """
    params: List[Any] = []
    if start_date:
        query += " AND trade_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND trade_date <= ?"
        params.append(end_date)
    query += " ORDER BY trade_date ASC, id ASC"

    db.cursor.execute(query, params)
    rows = db.cursor.fetchall()
    return [dict(row) for row in rows]


def _compute_group_stats(records: Dict[str, List[Dict[str, Any]]]) -> List[CostStats]:
    stats: List[CostStats] = []
    for group, rows in records.items():
        commissions: List[float] = []
        pnls: List[float] = []
        for row in rows:
            commission = float(row.get("commission") or 0.0)
            pnl = float(row.get("realized_pnl") or 0.0)
            commissions.append(commission)
            pnls.append(pnl)

        trades = len(rows)
        if trades == 0:
            continue
        commission_mean = sum(commissions) / trades if trades else 0.0
        pnl_mean = sum(pnls) / trades if trades else 0.0
        stats.append(
            CostStats(
                group=group,
                trades=trades,
                commission_median=_percentile(commissions, 50.0),
                commission_mean=commission_mean,
                commission_p95=_percentile(commissions, 95.0),
                pnl_median=_percentile(pnls, 50.0),
                pnl_mean=pnl_mean,
            )
        )
    return stats


@click.command()
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite database path used by the trading engine.",
)
@click.option(
    "--lookback-days",
    default=365,
    show_default=True,
    help="Number of days to look back from --as-of for trade samples.",
)
@click.option(
    "--as-of",
    default=None,
    help="Reference date in YYYY-MM-DD (default: today UTC).",
)
@click.option(
    "--grouping",
    type=click.Choice(["ticker", "asset_class"], case_sensitive=False),
    default="asset_class",
    show_default=True,
    help="Grouping mode for cost statistics.",
)
@click.option(
    "--min-trades",
    default=5,
    show_default=True,
    help="Minimum trades per group required to emit stats.",
)
@click.option(
    "--output",
    default="logs/automation/transaction_costs.json",
    show_default=True,
    help="Path to write JSON summary (directories are created as needed).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    db_path: str,
    lookback_days: int,
    as_of: Optional[str],
    grouping: str,
    min_trades: int,
    output: str,
    verbose: bool,
) -> None:
    """
    Estimate transaction cost statistics from realised trades.

    The output is a JSON document with per-group commission statistics and
    realised PnL distributions. Callers can use this to calibrate
    friction buffers and minimum expected returns in configs.
    """
    _configure_logging(verbose)

    if as_of:
        ref_date = datetime.fromisoformat(as_of).date()
    else:
        ref_date = datetime.utcnow().date()
    start = ref_date - timedelta(days=int(lookback_days))
    start_iso = start.isoformat()
    end_iso = ref_date.isoformat()

    with DatabaseManager(db_path=db_path) as db:
        rows = _load_trades(db, start_iso, end_iso)
        if not rows:
            logger.info("No realised trades found in window %s -> %s", start_iso, end_iso)
            return

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = _group_key(str(row.get("ticker") or ""), grouping.lower())
            grouped[key].append(row)

        # Filter by minimum trade count and compute stats.
        filtered = {g: recs for g, recs in grouped.items() if len(recs) >= int(min_trades)}
        if not filtered:
            logger.info(
                "All groups have fewer than %s trades in window %s -> %s; nothing to report.",
                min_trades,
                start_iso,
                end_iso,
            )
            return

        stats = _compute_group_stats(filtered)

    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "db_path": db_path,
            "as_of": ref_date.isoformat(),
            "lookback_days": lookback_days,
            "grouping": grouping,
            "min_trades": min_trades,
        },
        "groups": [s.to_dict() for s in stats],
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Transaction cost summary written to %s (%s groups)",
        out_path,
        len(stats),
    )


if __name__ == "__main__":
    main()

