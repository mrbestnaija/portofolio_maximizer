#!/usr/bin/env python3
"""ONE-TIME historical migration: tag trades 252 and 255 as is_contaminated=1.

IMPORTANT — This script is NOT the general contamination model.

Current contamination architecture (2026-04-11):
  - PaperTradingEngine (paper_trading_engine.py) auto-tags is_contaminated=1 at
    write time whenever the closing leg's opener is_synthetic=1 (INT-05 guard,
    lines ~1379-1397). No manual intervention required for future trades.
  - The integrity check (_check_cross_mode_contamination) flags only UNTAGGED
    closes (is_contaminated=0 with a synthetic opener). Tagged closes are already
    excluded from production_closed_trades and require no whitelist.
  - DO NOT add new trade IDs to _CONTAMINATED_CLOSE_IDS. If a trade is
    correctly tagged is_contaminated=1 in the DB, PTE handled it. If it was
    missed (is_contaminated=0 with synthetic opener), the integrity check will
    detect it and prompt running this migration.

Historical context for trades 252 and 255 (identified 2026-03-25):
  Both closing legs were live-mode executions whose PnL was driven by synthetic
  opening prices, predating the INT-05 auto-tag guard.

  Trade 252 (MSFT, -$1,027.31):
    entry_trade_id=244, opener is_synthetic=1 (exec_mode=synthetic, price=$64.37)
    Live closer at $405.44 — 530% phantom stop-distance, pure noise.

  Trade 255 (TSLA, -$629.77):
    entry_trade_id=NULL, entry_price=$82.13 from portfolio_state contamination.
    A prior synthetic run wrote TSLA=-2 to portfolio_state (no is_synthetic guard
    at the time). Live --resume read it as a live position and closed it at $397.

Safe to run multiple times (idempotent). Does not affect any other trades.
"""

import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "portfolio_maximizer.db")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Closing-leg trade IDs that are cross-mode contamination
_CONTAMINATED_CLOSE_IDS = (252, 255)

_VIEW_PRODUCTION_CLOSED_TRADES = """\
CREATE VIEW production_closed_trades AS
SELECT t.*
FROM   trade_executions t
WHERE  t.is_close = 1
  AND  t.is_diagnostic = 0
  AND  t.is_synthetic = 0
  AND  t.is_contaminated = 0
  AND  NOT EXISTS (
       SELECT 1
       FROM   trade_executions o
       WHERE  o.id = t.entry_trade_id
         AND  o.is_synthetic = 1
  )"""


def _connect(db_path: str) -> sqlite3.Connection:
    try:
        from integrity.sqlite_guardrails import guarded_sqlite_connect
        return guarded_sqlite_connect(db_path, allow_schema_changes=True)
    except ImportError:
        return sqlite3.connect(db_path)


def run(db_path: str = DB_PATH, dry_run: bool = False) -> None:
    conn = _connect(db_path)
    try:
        # ------------------------------------------------------------------
        # Step 1: add is_contaminated column if missing
        # ------------------------------------------------------------------
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trade_executions)")}
        if "is_contaminated" not in cols:
            if dry_run:
                print("[DRY RUN] Would add is_contaminated INTEGER DEFAULT 0 column")
            else:
                conn.execute(
                    "ALTER TABLE trade_executions "
                    "ADD COLUMN is_contaminated INTEGER DEFAULT 0"
                )
                print("[OK] Added is_contaminated column to trade_executions")
        else:
            print("[OK] is_contaminated column already exists")

        # ------------------------------------------------------------------
        # Step 2: verify the contaminated trades exist
        # ------------------------------------------------------------------
        cur = conn.execute(
            "SELECT id, ticker, realized_pnl, is_contaminated, entry_trade_id "
            "FROM trade_executions WHERE id IN (?,?)",
            _CONTAMINATED_CLOSE_IDS,
        )
        rows = cur.fetchall()
        if not rows:
            print("[OK] Contaminated trades not found — nothing to fix.")
            return

        print("Contaminated closing legs:")
        all_fixed = True
        for r in rows:
            status = "[already fixed]" if r[3] == 1 else "[NEEDS FIX]"
            print(
                f"  Trade {r[0]} ({r[1]}): pnl=${r[2]:+.2f}, "
                f"is_contaminated={r[3]}, entry_id={r[4]}  {status}"
            )
            if r[3] != 1:
                all_fixed = False

        # ------------------------------------------------------------------
        # Step 3: mark contaminated closing legs
        # ------------------------------------------------------------------
        if all_fixed:
            print("[OK] Both trades already marked is_contaminated=1.")
        elif dry_run:
            print(f"[DRY RUN] Would mark trades {_CONTAMINATED_CLOSE_IDS} as is_contaminated=1")
        else:
            conn.execute(
                "UPDATE trade_executions SET is_contaminated=1 WHERE id IN (?,?)",
                _CONTAMINATED_CLOSE_IDS,
            )
            print(f"[OK] Marked trades {_CONTAMINATED_CLOSE_IDS} as is_contaminated=1")

        # ------------------------------------------------------------------
        # Step 4: recreate production_closed_trades view
        # ------------------------------------------------------------------
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' "
            "AND name='production_closed_trades'"
        ).fetchone()

        target_sql = _VIEW_PRODUCTION_CLOSED_TRADES.strip()
        needs_update = existing is None or existing[0].strip() != target_sql

        if not needs_update:
            print("[OK] production_closed_trades view already up-to-date.")
        elif dry_run:
            print("[DRY RUN] Would recreate production_closed_trades view.")
        else:
            conn.execute("DROP VIEW IF EXISTS production_closed_trades")
            conn.execute(target_sql)
            print("[OK] Recreated production_closed_trades view with opener-join guard.")

        # ------------------------------------------------------------------
        # Step 5: commit and show corrected metrics
        # ------------------------------------------------------------------
        if not dry_run:
            conn.commit()

        cur = conn.execute(
            "SELECT COUNT(*) as n, "
            "SUM(realized_pnl) as total_pnl, "
            "AVG(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0.0 END) as wr, "
            "SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_win, "
            "SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) as gross_loss "
            "FROM production_closed_trades"
        )
        r = cur.fetchone()
        n, total_pnl, wr, gross_win, gross_loss = r
        pf = (gross_win / gross_loss) if gross_loss else float("inf")

        prefix = "[DRY RUN] " if dry_run else ""
        print()
        print(f"{prefix}=== Corrected production metrics ===")
        print(f"  Round-trips : {n}")
        print(f"  Total PnL   : ${total_pnl:+,.2f}")
        print(f"  Win rate    : {wr:.1%}")
        print(f"  Profit factor: {pf:.2f}")

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    args = parser.parse_args()
    run(db_path=args.db, dry_run=args.dry_run)
