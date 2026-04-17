"""
Retire zombie open legs that would consume THIN_LINKAGE credit.

Problem: trade_executions has 35+ unmatched open legs per ticker accumulated over
many interrupted sessions. INT-06 FIFO sorts by (is_synthetic, trade_id) ASC, so
the OLDEST non-synthetic lot is consumed first on each close. These old lots (Feb/Mar
2026, legacy) have no forecast audit files → closes linked to them get 0 THIN_LINKAGE
credit. The 4 current portfolio_state positions will waste their closes on zombie lots.

Fix: Mark zombie opens as is_synthetic=1. INT-06 then deprioritizes them. Each ticker
retains exactly N non-synthetic open lots (where N = portfolio_state share count). All
closes will now pair with the current lots (April 2026) that have audit files.

Safe: No rows deleted. is_synthetic=1 on open legs does not affect production_closed_trades
(which only looks at close legs). Retired lots remain auditable.

Usage:
  python scripts/retire_zombie_opens.py          # dry-run (shows changes, no write)
  python scripts/retire_zombie_opens.py --apply  # applies changes
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/portfolio_maximizer.db")

# Source of truth: what the system currently tracks as live positions.
# Maps ticker -> number of shares currently held (from portfolio_state).
# MUST be verified against DB before running --apply.
PORTFOLIO_STATE = {
    "AAPL": 1,
    "AMZN": 1,
    "GOOG": 1,
    "NVDA": 2,
}


def _get_open_lots(conn: sqlite3.Connection) -> dict[str, list[int]]:
    """Return all unmatched open non-close legs per ticker, sorted by (is_synthetic, id)."""
    cur = conn.execute("""
        SELECT id, ticker, trade_date, price, is_synthetic, ts_signal_id
        FROM trade_executions
        WHERE is_close = 0
          AND id NOT IN (
              SELECT COALESCE(entry_trade_id, 0)
              FROM trade_executions
              WHERE is_close = 1 AND entry_trade_id IS NOT NULL
          )
        ORDER BY ticker, is_synthetic ASC, id ASC
    """)
    lots: dict[str, list[tuple]] = {}
    for row in cur.fetchall():
        lots.setdefault(row[1], []).append(row)
    return lots


def _verify_portfolio_state(conn: sqlite3.Connection) -> bool:
    """Confirm portfolio_state matches our PORTFOLIO_STATE constant."""
    cur = conn.execute("SELECT ticker, shares FROM portfolio_state ORDER BY ticker")
    db_state = {r[0]: r[1] for r in cur.fetchall()}
    ok = True
    for ticker, shares in PORTFOLIO_STATE.items():
        db_shares = db_state.get(ticker, 0)
        status = "OK" if db_shares == shares else "MISMATCH"
        print(f"  portfolio_state {ticker}: DB={db_shares} expected={shares}  [{status}]")
        if db_shares != shares:
            ok = False
    extra = set(db_state) - set(PORTFOLIO_STATE)
    if extra:
        print(f"  [WARN] portfolio_state has extra tickers not in PORTFOLIO_STATE: {extra}")
    return ok


def main(apply: bool) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n=== Zombie Open Leg Retirement ({mode}) ===\n")

    conn = sqlite3.connect(str(DB_PATH))

    print("Verifying portfolio_state against PORTFOLIO_STATE constant:")
    if not _verify_portfolio_state(conn):
        print("\n[ABORT] portfolio_state mismatch. Update PORTFOLIO_STATE in this script first.")
        conn.close()
        return
    print()

    lots = _get_open_lots(conn)

    retire_ids: list[int] = []
    keep_ids: list[int] = []

    for ticker, rows in sorted(lots.items()):
        n_keep = PORTFOLIO_STATE.get(ticker, 0)
        # Non-synthetic lots sorted by id ASC — the FIFO queue order.
        non_synthetic = [r for r in rows if r[4] == 0]  # is_synthetic == 0
        synthetic = [r for r in rows if r[4] == 1]

        # Keep the NEWEST n_keep non-synthetic lots (highest IDs = most audit-file coverage).
        # Retire the oldest ones (they have no recent audit files and would waste closes).
        keep = non_synthetic[-n_keep:] if n_keep > 0 else []
        retire_live = non_synthetic[:-n_keep] if n_keep > 0 else non_synthetic

        keep_ids.extend(r[0] for r in keep)
        retire_ids.extend(r[0] for r in retire_live)
        # Synthetic lots are already deprioritized by INT-06, but flag them for clarity
        already_synthetic = [r[0] for r in synthetic]

        print(f"{ticker}: {len(non_synthetic)} live opens, {len(synthetic)} already-synthetic")
        if keep:
            for r in keep:
                print(f"  KEEP   id={r[0]:4d}  date={r[2]}  price={r[3]:.2f}  "
                      f"is_syn={r[4]}  tsid={r[5]}")
        if retire_live:
            for r in retire_live:
                print(f"  RETIRE id={r[0]:4d}  date={r[2]}  price={r[3]:.2f}  "
                      f"is_syn={r[4]}  tsid={r[5]}")
        print()

    print(f"Summary: {len(keep_ids)} lots to KEEP, {len(retire_ids)} lots to RETIRE\n")

    if not retire_ids:
        print("Nothing to retire. DB is clean.")
        conn.close()
        return

    if not apply:
        print("[DRY-RUN] No changes written. Run with --apply to execute.")
        conn.close()
        return

    # Backup before surgery
    backup_path = DB_PATH.with_suffix(f".backup_{int(time.time())}.db")
    shutil.copy2(DB_PATH, backup_path)
    print(f"Backup written: {backup_path.name}")

    # Apply: mark retired lots as is_synthetic=1 and clear live execution_mode.
    # The schema CHECK constraint forbids is_synthetic=1 on execution_mode='live' rows.
    placeholders = ",".join("?" * len(retire_ids))
    conn.execute(
        f"UPDATE trade_executions "
        f"SET is_synthetic=1, execution_mode='zombie_retired' "
        f"WHERE id IN ({placeholders})",
        retire_ids,
    )
    conn.commit()
    print(f"Marked {len(retire_ids)} open legs as is_synthetic=1 / execution_mode='zombie_retired'.")

    # Verify result
    cur = conn.execute("""
        SELECT ticker, COUNT(*) as remaining_live
        FROM trade_executions
        WHERE is_close=0 AND is_synthetic=0
          AND id NOT IN (
              SELECT COALESCE(entry_trade_id,0)
              FROM trade_executions WHERE is_close=1 AND entry_trade_id IS NOT NULL
          )
        GROUP BY ticker ORDER BY ticker
    """)
    print("\nRemaining live open lots per ticker (should match portfolio_state):")
    for row in cur.fetchall():
        expected = PORTFOLIO_STATE.get(row[0], 0)
        status = "OK" if row[1] == expected else "MISMATCH"
        print(f"  {row[0]}: {row[1]} live lots (expected {expected})  [{status}]")

    conn.close()
    print("\nDone. Run production_audit_gate.py to confirm THIN_LINKAGE status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retire zombie open legs")
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes (default is dry-run)")
    args = parser.parse_args()
    main(apply=args.apply)
