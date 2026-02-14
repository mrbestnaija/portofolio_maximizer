#!/usr/bin/env python3
"""Migration: Add PnL integrity enforcement columns to trade_executions.

New columns:
  - is_diagnostic  INTEGER DEFAULT 0  -- trade executed under DIAGNOSTIC_MODE
  - is_synthetic   INTEGER DEFAULT 0  -- trade from synthetic data source
  - confidence_calibrated REAL        -- calibrated confidence (future use)
  - entry_trade_id INTEGER            -- links closing leg to its opening leg
  - bar_open       REAL               -- OHLC of the bar used for fill price
  - bar_high       REAL
  - bar_low        REAL
  - bar_close      REAL

Also creates canonical views:
  - production_closed_trades  -- is_close=1, not diagnostic, not synthetic
  - round_trips               -- closing legs joined to opening legs

Safe to run multiple times (idempotent).
"""

import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "portfolio_maximizer.db")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from integrity.sqlite_guardrails import guarded_sqlite_connect


def migrate(db_path: str = DB_PATH) -> None:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        sys.exit(1)

    conn = guarded_sqlite_connect(
        db_path,
        allow_schema_changes=True,
    )
    conn.row_factory = sqlite3.Row
    cur = conn.execute("PRAGMA table_info(trade_executions)")
    existing = {row["name"] for row in cur.fetchall()}

    new_columns = {
        "is_diagnostic": "INTEGER DEFAULT 0",
        "is_synthetic": "INTEGER DEFAULT 0",
        "confidence_calibrated": "REAL",
        "entry_trade_id": "INTEGER",
        "bar_open": "REAL",
        "bar_high": "REAL",
        "bar_low": "REAL",
        "bar_close": "REAL",
    }

    added = []
    for col, typedef in new_columns.items():
        if col not in existing:
            conn.execute(
                f"ALTER TABLE trade_executions ADD COLUMN {col} {typedef}"
            )
            added.append(col)
            print(f"  [OK] Added column: {col} {typedef}")
        else:
            print(f"  [SKIP] Column already exists: {col}")

    # Create canonical views
    conn.execute("DROP VIEW IF EXISTS production_closed_trades")
    conn.execute("""
        CREATE VIEW IF NOT EXISTS production_closed_trades AS
        SELECT *
        FROM   trade_executions
        WHERE  is_close = 1
          AND  COALESCE(is_diagnostic, 0) = 0
          AND  COALESCE(is_synthetic, 0)  = 0
    """)
    print("  [OK] Created view: production_closed_trades")

    conn.execute("DROP VIEW IF EXISTS round_trips")
    conn.execute("""
        CREATE VIEW IF NOT EXISTS round_trips AS
        SELECT
            c.id            AS close_id,
            c.ticker,
            o.id            AS open_id,
            o.trade_date    AS entry_date,
            c.trade_date    AS exit_date,
            o.price         AS entry_price,
            c.exit_price    AS exit_price,
            c.shares,
            c.realized_pnl,
            c.realized_pnl_pct,
            c.holding_period_days,
            c.exit_reason,
            c.execution_mode,
            COALESCE(c.is_diagnostic, 0) AS is_diagnostic,
            COALESCE(c.is_synthetic, 0)  AS is_synthetic
        FROM   trade_executions c
        LEFT JOIN trade_executions o ON c.entry_trade_id = o.id
        WHERE  c.is_close = 1
    """)
    print("  [OK] Created view: round_trips")

    conn.commit()
    conn.close()

    if added:
        print(f"\n[DONE] Added {len(added)} columns: {', '.join(added)}")
    else:
        print("\n[DONE] All columns already exist. Views refreshed.")


if __name__ == "__main__":
    print("=== PnL Integrity Migration ===")
    migrate()
