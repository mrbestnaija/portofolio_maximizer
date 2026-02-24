"""
migrate_add_ts_signal_id.py — Phase 7.13-A2 migration.

Adds the `ts_signal_id TEXT` column to trade_executions (if missing), creates a
covering index, and backfills legacy rows (ts_signal_id IS NULL) with synthetic
IDs of the form  legacy_{trade_date}_{id}  so attribution is explicit rather than
null.

Safe to run multiple times (idempotent).

Usage:
    python scripts/migrate_add_ts_signal_id.py [--db PATH] [--dry-run]

Exit codes:
    0  success
    1  DB not found or schema error
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------------------

def migrate(db_path: Path, dry_run: bool = False) -> None:
    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # ── 1. Inspect existing columns ──────────────────────────────────
        cur.execute("PRAGMA table_info(trade_executions)")
        cols = {row["name"] for row in cur.fetchall()}

        # ── 2. Add column if missing ──────────────────────────────────────
        if "ts_signal_id" not in cols:
            print("[migrate] Adding ts_signal_id TEXT column to trade_executions...")
            if not dry_run:
                cur.execute(
                    "ALTER TABLE trade_executions ADD COLUMN ts_signal_id TEXT"
                )
            else:
                print("[dry-run] Would ALTER TABLE trade_executions ADD COLUMN ts_signal_id TEXT")
        else:
            print("[migrate] ts_signal_id column already present — skipping ALTER TABLE.")

        # ── 3. Create index if missing ────────────────────────────────────
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_trade_executions_ts_signal_id'"
        )
        if not cur.fetchone():
            print("[migrate] Creating index idx_trade_executions_ts_signal_id...")
            if not dry_run:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trade_executions_ts_signal_id "
                    "ON trade_executions(ts_signal_id)"
                )
            else:
                print("[dry-run] Would CREATE INDEX idx_trade_executions_ts_signal_id")
        else:
            print("[migrate] Index already exists — skipping CREATE INDEX.")

        # ── 4. Backfill legacy NULL rows ──────────────────────────────────
        # Skip count if we're dry-running and the column didn't exist yet
        col_exists_now = (not dry_run) or ("ts_signal_id" in cols)
        if not col_exists_now:
            print("[dry-run] Skipping backfill count (column not yet present in dry-run mode).")
            null_count = 0
        else:
            cur.execute(
                "SELECT COUNT(*) FROM trade_executions WHERE ts_signal_id IS NULL"
            )
            null_count = cur.fetchone()[0]
        print(f"[migrate] Rows with ts_signal_id IS NULL: {null_count}")

        if null_count > 0:
            backfill_sql = """
                UPDATE trade_executions
                SET ts_signal_id = 'legacy_' || COALESCE(trade_date, 'unknown') || '_' || CAST(id AS TEXT)
                WHERE ts_signal_id IS NULL
            """
            if not dry_run:
                cur.execute(backfill_sql)
                print(f"[migrate] Backfilled {cur.rowcount} legacy rows.")
            else:
                print(f"[dry-run] Would backfill {null_count} legacy rows with synthetic IDs.")

        # ── 5. Verify ─────────────────────────────────────────────────────
        if not dry_run:
            cur.execute(
                "SELECT COUNT(*) FROM trade_executions WHERE ts_signal_id IS NULL"
            )
            remaining = cur.fetchone()[0]
            if remaining > 0:
                print(f"[WARNING] {remaining} rows still have ts_signal_id IS NULL after migration.")
            else:
                print("[migrate] All rows now have ts_signal_id set. Migration complete.")

            conn.commit()
        else:
            print("[dry-run] No changes written.")

    except sqlite3.Error as exc:
        print(f"[ERROR] SQLite error during migration: {exc}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 7.13-A2: Add ts_signal_id TEXT column to trade_executions."
    )
    p.add_argument(
        "--db",
        type=Path,
        default=Path("data/portfolio_maximizer.db"),
        help="Path to SQLite database (default: data/portfolio_maximizer.db)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying the database.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    migrate(db_path=args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
