#!/usr/bin/env python3
"""
summarize_sleeves.py
--------------------

Per-sleeve PnL summary helper.

Computes basic profitability metrics (total trades, win rate, profit factor,
total profit, average trade PnL) grouped by logical "sleeves" derived from:

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
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from risk.barbell_policy import BarbellConfig

ROOT_PATH = Path(__file__).resolve().parent.parent


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


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_trades(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            ticker,
            trade_date,
            asset_class,
            instrument_type,
            realized_pnl
        FROM trade_executions
        WHERE realized_pnl IS NOT NULL
        """
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
        ticker = str(row["ticker"])
        ac = row.get("asset_class")
        sleeve = classify(ticker, ac)
        key = (sleeve, ticker)
        pnl = float(row["realized_pnl"] or 0.0)

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
def main(db_path: str, min_trades: int, output: Optional[str]) -> None:
    """
    Summarise PnL metrics per sleeve (safe/core/speculative/crypto/other).
    """
    cfg = BarbellConfig.from_yaml()
    conn = _connect(db_path)
    try:
        rows = _load_trades(conn)
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
        payload = {
            "generated_at": date.today().isoformat(),
            "db_path": db_path,
            "min_trades": min_trades,
            "sleeves": [m.to_dict() for m in metrics],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON summary written to {out_path}")


if __name__ == "__main__":
    main()
