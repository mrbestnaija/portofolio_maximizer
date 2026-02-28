#!/usr/bin/env python3
"""
migrate_add_model_order_stats.py
---------------------------------
Create model_order_stats table for the auto-learning order cache and
backfill from existing time_series_forecasts rows.

Safe to run multiple times (CREATE TABLE IF NOT EXISTS + INSERT OR IGNORE).

Usage:
    python scripts/migrate_add_model_order_stats.py
"""

import json
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio_maximizer.db"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrity.sqlite_guardrails import guarded_sqlite_connect


DDL = """
CREATE TABLE IF NOT EXISTS model_order_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL,
    model_type   TEXT    NOT NULL,
    regime       TEXT,
    order_params TEXT    NOT NULL,
    n_fits       INTEGER DEFAULT 0,
    aic_sum      REAL    DEFAULT 0.0,
    bic_sum      REAL    DEFAULT 0.0,
    best_aic     REAL,
    last_used    DATE,
    first_seen   DATE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, model_type, regime, order_params)
)
"""

BACKFILL_SQL = """
INSERT OR IGNORE INTO model_order_stats
    (ticker, model_type, regime, order_params,
     n_fits, aic_sum, bic_sum, best_aic, last_used, first_seen)
SELECT
    ticker,
    model_type,
    COALESCE(detected_regime, 'UNKNOWN') AS regime,
    model_order                          AS order_params,
    COUNT(*)                             AS n_fits,
    COALESCE(SUM(CASE WHEN aic IS NOT NULL THEN aic ELSE 0 END), 0.0) AS aic_sum,
    COALESCE(SUM(CASE WHEN bic IS NOT NULL THEN bic ELSE 0 END), 0.0) AS bic_sum,
    MIN(aic)                             AS best_aic,
    MAX(DATE(created_at))                AS last_used,
    MIN(DATE(created_at))                AS first_seen
FROM time_series_forecasts
WHERE model_order IS NOT NULL
  AND model_type IN ('GARCH', 'SARIMAX', 'SAMOSSA')
GROUP BY ticker, model_type, COALESCE(detected_regime, 'UNKNOWN'), model_order
"""


def migrate(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        print(f"[ERROR] Database not found at {db_path}")
        sys.exit(1)

    conn = guarded_sqlite_connect(str(db_path), allow_schema_changes=True)
    cur = conn.cursor()

    # Create table
    cur.execute(DDL)
    conn.commit()
    print("[OK] model_order_stats table created/verified")

    # Check whether time_series_forecasts has the source columns
    cur.execute("PRAGMA table_info(time_series_forecasts)")
    cols = {row[1] for row in cur.fetchall()}
    if "model_order" not in cols:
        print("[WARN] time_series_forecasts.model_order column missing — skipping backfill")
        conn.close()
        return

    # Backfill
    cur.execute(BACKFILL_SQL)
    backfilled = cur.rowcount
    conn.commit()

    # Report
    cur.execute("SELECT COUNT(*) FROM model_order_stats")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT model_type, COALESCE(regime,'NULL'), COUNT(*) "
        "FROM model_order_stats GROUP BY 1, 2 ORDER BY 1, 2"
    )
    rows = cur.fetchall()

    conn.close()

    print(f"[OK] Backfilled {backfilled} rows into model_order_stats (total: {total})")
    print("\nSummary by (model_type, regime):")
    for model_type, regime, count in rows:
        print(f"  {model_type:12s}  {regime:24s}  {count} entries")


if __name__ == "__main__":
    migrate()
