#!/usr/bin/env python3
"""
migrate_add_portfolio_state.py
------------------------------
Add portfolio_state and portfolio_cash_state tables for cross-session
position persistence.

Safe to run multiple times (CREATE TABLE IF NOT EXISTS).

Usage:
    python scripts/migrate_add_portfolio_state.py
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio_maximizer.db"


def migrate(db_path: Path = DB_PATH):
    if not db_path.exists():
        print(f"[ERROR] Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            shares INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            entry_timestamp TEXT,
            stop_loss REAL,
            target_price REAL,
            max_holding_days INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_cash_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL NOT NULL,
            initial_capital REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("[OK] portfolio_state and portfolio_cash_state tables created/verified")


if __name__ == "__main__":
    migrate()
