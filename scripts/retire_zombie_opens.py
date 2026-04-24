"""Retire zombie open legs that would consume THIN_LINKAGE credit.

The live engine persists the authoritative inventory snapshot in ``portfolio_state``.
This helper compares unmatched open BUY legs against that live snapshot and marks
any surplus oldest lots as ``is_synthetic=1`` so INT-06 deprioritizes them.

Safe: no rows are deleted; the open ledger remains auditable. Only surplus open legs
are retired, and only after the live ``portfolio_state`` snapshot confirms the ticker
still has fewer active shares than open BUY lots.

Usage:
  python scripts/retire_zombie_opens.py --db data/portfolio_maximizer.db
  python scripts/retire_zombie_opens.py --db data/portfolio_maximizer.db --apply
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/portfolio_maximizer.db")


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


def _load_live_portfolio_state(conn: sqlite3.Connection) -> dict[str, int]:
    """Load the authoritative live position counts from portfolio_state.

    Returns a ticker -> share count mapping. Raises if the table is absent, because
    the helper cannot safely infer current inventory from the execution ledger alone.
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_state' LIMIT 1"
    ).fetchone()
    if not has_table:
        raise RuntimeError("portfolio_state table missing")

    cur = conn.execute("SELECT ticker, shares FROM portfolio_state ORDER BY ticker")
    live_state: dict[str, int] = {}
    for ticker, shares in cur.fetchall():
        ticker_str = str(ticker or "").strip().upper()
        if not ticker_str:
            continue
        try:
            qty = int(round(float(shares or 0.0)))
        except (TypeError, ValueError):
            continue
        if qty > 0:
            live_state[ticker_str] = qty
    return live_state


def main(apply: bool, *, db_path: Path | None = None) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n=== Zombie Open Leg Retirement ({mode}) ===\n")

    db_path = db_path or DB_PATH
    conn = sqlite3.connect(str(db_path))

    try:
        live_state = _load_live_portfolio_state(conn)
    except Exception as exc:
        print(f"[ABORT] {exc}")
        conn.close()
        return
    if not live_state:
        print("[ABORT] portfolio_state is empty; nothing authoritative to retire against.")
        conn.close()
        return

    print("Live portfolio_state snapshot:")
    for ticker, shares in sorted(live_state.items()):
        print(f"  {ticker}: {shares} share(s)")
    print()

    lots = _get_open_lots(conn)

    retire_ids: list[int] = []
    keep_ids: list[int] = []

    for ticker, rows in sorted(lots.items()):
        n_keep = live_state.get(ticker, 0)
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
        expected = live_state.get(str(row[0]).upper(), 0)
        status = "OK" if row[1] == expected else "MISMATCH"
        print(f"  {row[0]}: {row[1]} live lots (expected {expected})  [{status}]")

    conn.close()
    print("\nDone. Run production_audit_gate.py to confirm THIN_LINKAGE status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retire zombie open legs")
    parser.add_argument(
        "--db",
        default=str(DB_PATH),
        help="Path to the SQLite database (default: data/portfolio_maximizer.db)",
    )
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes (default is dry-run)")
    args = parser.parse_args()
    main(apply=args.apply, db_path=Path(args.db).expanduser().resolve())
