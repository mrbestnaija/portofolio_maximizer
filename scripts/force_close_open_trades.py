#!/usr/bin/env python3
"""
force_close_open_trades.py
--------------------------

Diagnostic helper to mark all trades with NULL realized_pnl as closed so that
downstream threshold sweeps can proceed. This is a simulation-only tool and
sets realized_pnl and realized_pnl_pct to 0.0 with a holding_period_days of 0.

WARNING:
- This does NOT compute mark-to-market PnL. It simply finalizes open trades
  with zero PnL for evidence-collection purposes. Use only in diagnostic runs.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrity.sqlite_guardrails import guarded_sqlite_connect


@click.command()
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite database path used by the trading engine.",
)
def main(db_path: str) -> None:
    path = Path(db_path)
    if not path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = guarded_sqlite_connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE trade_executions
        SET realized_pnl = 0.0,
            realized_pnl_pct = 0.0,
            holding_period_days = 0
        WHERE realized_pnl IS NULL
        """
    )
    updated = cur.rowcount
    conn.commit()
    conn.close()
    click.echo(f"Force-closed {updated} open trades (set realized_pnl=0).")


if __name__ == "__main__":
    main()
