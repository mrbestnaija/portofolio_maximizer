#!/usr/bin/env python3
"""Migration: Relax immutable ledger trigger to allow entry_trade_id backfill.

Current trigger blocks ALL updates to closed trades. This prevents backfilling
the entry_trade_id audit linkage for historical trades.

New trigger allows entry_trade_id updates (NULL -> value) but blocks changes to:
- realized_pnl, realized_pnl_pct (PnL integrity)
- shares, price, total_value (execution details)
- ticker, trade_date, action (trade identity)

Safe to run multiple times (idempotent).
"""

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "portfolio_maximizer.db"


def migrate(db_path: Path, dry_run: bool = True):
    """Replace immutable ledger trigger with backfill-aware version."""
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    print("=" * 70)
    print("RELAX IMMUTABLE LEDGER TRIGGER")
    print("=" * 70)
    print()

    # Check current trigger
    cur = conn.execute("""
        SELECT sql FROM sqlite_master
        WHERE type = 'trigger'
          AND name = 'enforce_immutable_closed_trades'
    """)
    row = cur.fetchone()

    if row:
        print("Current trigger found:")
        print(row[0])
        print()
    else:
        print("[WARNING] Trigger not found - may already be migrated")
        print()

    if dry_run:
        print("[DRY RUN] Would replace trigger with:")
        print()
        print("""
CREATE TRIGGER enforce_immutable_closed_trades
BEFORE UPDATE ON trade_executions
WHEN OLD.is_close = 1
BEGIN
    -- Allow entry_trade_id backfill (NULL -> value)
    SELECT CASE
        WHEN OLD.entry_trade_id IS NULL AND NEW.entry_trade_id IS NOT NULL
            THEN 0  -- Allow backfill
        WHEN NEW.realized_pnl != OLD.realized_pnl
            OR NEW.realized_pnl_pct != OLD.realized_pnl_pct
            OR NEW.shares != OLD.shares
            OR NEW.price != OLD.price
            OR NEW.total_value != OLD.total_value
            OR NEW.ticker != OLD.ticker
            OR NEW.trade_date != OLD.trade_date
            OR NEW.action != OLD.action
            THEN RAISE(ABORT, 'Cannot modify core fields of closed trades')
        ELSE 0  -- Allow other audit field updates
    END;
END;
        """)
        print()
        print("Re-run with --apply to execute")
        conn.close()
        return 0

    # Apply migration
    print("Dropping old trigger...")
    conn.execute("DROP TRIGGER IF EXISTS enforce_immutable_closed_trades")

    print("Creating relaxed trigger...")
    conn.execute("""
        CREATE TRIGGER enforce_immutable_closed_trades
        BEFORE UPDATE ON trade_executions
        WHEN OLD.is_close = 1
        BEGIN
            -- Allow entry_trade_id backfill (NULL -> value)
            SELECT CASE
                WHEN OLD.entry_trade_id IS NULL AND NEW.entry_trade_id IS NOT NULL
                    THEN 0  -- Allow backfill
                WHEN (NEW.realized_pnl IS NOT NULL AND OLD.realized_pnl IS NOT NULL
                      AND NEW.realized_pnl != OLD.realized_pnl)
                    OR (NEW.realized_pnl_pct IS NOT NULL AND OLD.realized_pnl_pct IS NOT NULL
                        AND NEW.realized_pnl_pct != OLD.realized_pnl_pct)
                    OR NEW.shares != OLD.shares
                    OR NEW.price != OLD.price
                    OR NEW.total_value != OLD.total_value
                    OR NEW.ticker != OLD.ticker
                    OR NEW.trade_date != OLD.trade_date
                    OR NEW.action != OLD.action
                    THEN RAISE(ABORT, 'Cannot modify core fields of closed trades')
                ELSE 0  -- Allow other audit field updates
            END;
        END
    """)

    conn.commit()
    conn.close()

    print("[SUCCESS] Trigger updated to allow entry_trade_id backfill")
    print()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--apply", action="store_true", help="Apply migration (default is dry-run)")

    args = parser.parse_args()

    sys.exit(migrate(args.db, dry_run=not args.apply))
