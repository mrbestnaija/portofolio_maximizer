"""
Phase 7.14-D: Add detected_regime and regime_confidence columns to time_series_forecasts.

Idempotent: safe to run multiple times. Checks PRAGMA table_info before any ALTER TABLE.

Usage:
    python scripts/migrate_add_regime_to_forecasts.py
    python scripts/migrate_add_regime_to_forecasts.py --db path/to/other.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def run_migration(db_path: Path) -> None:
    print(f"Connecting to: {db_path}")
    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        # Check current columns
        cur.execute("PRAGMA table_info(time_series_forecasts)")
        existing = {row["name"] for row in cur.fetchall()}
        if not existing:
            print("[ERROR] time_series_forecasts table not found. Run pipeline first.")
            sys.exit(1)

        added = []

        if "detected_regime" not in existing:
            cur.execute(
                "ALTER TABLE time_series_forecasts ADD COLUMN detected_regime TEXT"
            )
            added.append("detected_regime TEXT")
            print("[OK] Added: detected_regime TEXT")
        else:
            print("[SKIP] detected_regime already exists")

        if "regime_confidence" not in existing:
            cur.execute(
                "ALTER TABLE time_series_forecasts ADD COLUMN regime_confidence REAL"
            )
            added.append("regime_confidence REAL")
            print("[OK] Added: regime_confidence REAL")
        else:
            print("[SKIP] regime_confidence already exists")

        if added:
            # Index for regime-based queries
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tsf_regime "
                "ON time_series_forecasts(detected_regime)"
            )
            print("[OK] Index idx_tsf_regime ensured")
            conn.commit()
            print(f"[DONE] Migration applied: {', '.join(added)}")
        else:
            print("[DONE] No changes needed (already up to date)")

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7.14-D: Add regime columns to forecasts")
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite DB (default: data/portfolio_maximizer.db)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    db_path = Path(args.db) if args.db else root / "data" / "portfolio_maximizer.db"
    run_migration(db_path)


if __name__ == "__main__":
    main()
