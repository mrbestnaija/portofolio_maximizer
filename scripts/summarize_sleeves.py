#!/usr/bin/env python3
"""
summarize_sleeves.py
--------------------

Per-sleeve PnL summary helper.

Computes rolling profitability metrics (total trades, win rate, profit factor,
total profit, average trade PnL) grouped by logical "sleeves" derived from the
canonical `production_closed_trades` view and a configurable lookback window.

- Barbell buckets (safe/core/speculative) in config/barbell.yml, and
- asset_class tags on trade_executions (e.g. crypto vs equity).

This is a read-only tool used for:
- Quant automation (Phase 4 in QUANT_VALIDATION_AUTOMATION_TODO.md),
- Barbell/NAV diagnostics (NAV_RISK_BUDGET_ARCH.md, NAV_BAR_BELL_TODO.md),
- Research summaries (RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from risk.barbell_policy import BarbellConfig

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from integrity.sqlite_guardrails import guarded_sqlite_connect


@dataclass
class SleeveMetrics:
    sleeve: str
    ticker: str
    trades: int
    wins: int
    losses: int
    total_profit: float
    gross_profit: float
    gross_loss: float

    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    def profit_factor(self) -> float:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return float("inf") if self.gross_profit > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sleeve": self.sleeve,
            "ticker": self.ticker,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate(),
            "total_profit": self.total_profit,
            "profit_factor": self.profit_factor(),
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
        }


def _row_value(row: sqlite3.Row | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def _connect(db_path: str) -> sqlite3.Connection:
    conn = guarded_sqlite_connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_trades(
    conn: sqlite3.Connection,
    *,
    lookback_days: Optional[int],
    as_of_date: Optional[str],
) -> List[sqlite3.Row]:
    cur = conn.cursor()
    end_date = date.fromisoformat(as_of_date) if as_of_date else date.today()
    start_date = None
    if lookback_days is not None:
        start_date = end_date - timedelta(days=max(int(lookback_days), 1) - 1)
    params: list[Any] = []
    where_clauses: list[str] = []
    if start_date is not None:
        where_clauses.append("DATE(trade_date) >= DATE(?)")
        params.append(start_date.isoformat())
        where_clauses.append("DATE(trade_date) <= DATE(?)")
        params.append(end_date.isoformat())
    cur.execute(
        """
        SELECT
            ticker,
            trade_date,
            asset_class,
            instrument_type,
            realized_pnl
        FROM production_closed_trades
        {where_clause}
        ORDER BY ticker, id
        """.format(
            where_clause=("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        ),
        params,
    )
    return cur.fetchall()


def _build_sleeve_classifier(cfg: BarbellConfig):
    safe = set(cfg.safe_symbols)
    core = set(cfg.core_symbols)
    spec = set(cfg.speculative_symbols)

    def classify(ticker: str, asset_class: Optional[str]) -> str:
        sym = (ticker or "").upper()
        if sym in safe:
            return "safe"
        if sym in core:
            return "core"
        if sym in spec:
            return "speculative"
        ac = (asset_class or "").lower()
        if ac == "crypto":
            return "crypto"
        return "other"

    return classify


def _aggregate_by_sleeve(
    rows: List[sqlite3.Row],
    cfg: BarbellConfig,
    min_trades: int,
) -> List[SleeveMetrics]:
    classify = _build_sleeve_classifier(cfg)
    buckets: Dict[Tuple[str, str], SleeveMetrics] = {}

    for row in rows:
        ticker = str(_row_value(row, "ticker", ""))
        ac = _row_value(row, "asset_class")
        instrument_type = str(_row_value(row, "instrument_type", "") or "").lower()
        if not ac and instrument_type == "crypto":
            ac = "crypto"
        sleeve = classify(ticker, ac)
        key = (sleeve, ticker)
        pnl = float(_row_value(row, "realized_pnl", 0.0) or 0.0)

        if key not in buckets:
            buckets[key] = SleeveMetrics(
                sleeve=sleeve,
                ticker=ticker,
                trades=0,
                wins=0,
                losses=0,
                total_profit=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
            )
        m = buckets[key]
        m.trades += 1
        m.total_profit += pnl
        if pnl > 0:
            m.wins += 1
            m.gross_profit += pnl
        elif pnl < 0:
            m.losses += 1
            m.gross_loss += abs(pnl)

    return [
        m
        for m in buckets.values()
        if m.trades >= min_trades
    ]


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
    help="Rolling lookback window in days. Use 365 for weekly maintenance.",
)
@click.option(
    "--as-of-date",
    default=None,
    help="Optional ISO date used as the window end date (defaults to today).",
)
@click.option(
    "--min-trades",
    default=5,
    show_default=True,
    help="Minimum trades per (sleeve, ticker) required to report metrics.",
)
@click.option(
    "--output",
    default=None,
    help="Optional path to write JSON summary (if omitted, prints table only).",
)
def main(db_path: str, lookback_days: int, as_of_date: Optional[str], min_trades: int, output: Optional[str]) -> None:
    """
    Summarise PnL metrics per sleeve (safe/core/speculative/crypto/other).
    """
    cfg = BarbellConfig.from_yaml()
    conn = _connect(db_path)
    try:
        rows = _load_trades(conn, lookback_days=lookback_days, as_of_date=as_of_date)
    finally:
        conn.close()

    if not rows:
        print("No realised trades found; nothing to summarise.")
        return

    metrics = _aggregate_by_sleeve(rows, cfg, min_trades=min_trades)
    if not metrics:
        print(f"No sleeves with at least {min_trades} trades; nothing to summarise.")
        return

    # Simple table to stdout.
    header = f"{'Sleeve':<12} {'Ticker':<10} {'Trades':>6} {'Win%':>7} {'PF':>7} {'TotalPnL':>12}"
    print(header)
    print("-" * len(header))
    for m in sorted(metrics, key=lambda x: (x.sleeve, x.ticker)):
        print(
            f"{m.sleeve:<12} {m.ticker:<10} {m.trades:6d} "
            f"{m.win_rate()*100:7.1f} {m.profit_factor():7.2f} {m.total_profit:12.2f}"
        )

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        end_date = date.fromisoformat(as_of_date) if as_of_date else date.today()
        start_date = end_date - timedelta(days=max(int(lookback_days), 1) - 1)
        payload = {
            "generated_at": date.today().isoformat(),
            "db_path": db_path,
            "lookback_days": int(lookback_days),
            "as_of_date": end_date.isoformat(),
            "window": {
                "lookback_days": int(lookback_days),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "state": "rolling_window",
                "source_view": "production_closed_trades",
            },
            "source_view": "production_closed_trades",
            "min_trades": min_trades,
            "sleeves": [m.to_dict() for m in metrics],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON summary written to {out_path}")


if __name__ == "__main__":
    main()
