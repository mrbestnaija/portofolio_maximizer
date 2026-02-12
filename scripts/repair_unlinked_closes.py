#!/usr/bin/env python3
"""Repair script: Backfill entry_trade_id for unlinked closing legs.

Context:
  4 closing legs (IDs 9, 10, 15, 23) have realized_pnl but missing entry_trade_id.
  These are from 2026-02-10 live trading runs.

  The orphaned BUY entries exist (IDs 5,6,7,8,11,13,18,21) but were never linked.

Strategy:
  Match unlinked SELLs to orphaned BUYs by:
  1. Same ticker
  2. SELL shares <= accumulated BUY shares (FIFO matching)
  3. Closest timestamp match within same run_id context

Safe to run multiple times (checks if already repaired).
"""

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "portfolio_maximizer.db"


def find_unlinked_closes(conn: sqlite3.Connection) -> list:
    """Find closing legs with PnL but no entry_trade_id."""
    cur = conn.execute("""
        SELECT id, ticker, trade_date, shares, price, realized_pnl,
               bar_timestamp, run_id
        FROM trade_executions
        WHERE is_close = 1
          AND entry_trade_id IS NULL
          AND realized_pnl IS NOT NULL
        ORDER BY trade_date, id
    """)
    return cur.fetchall()


def find_orphaned_entries(conn: sqlite3.Connection, ticker: str) -> list:
    """Find orphaned BUY entries for a ticker (no SELL linkage)."""
    cur = conn.execute("""
        SELECT id, ticker, trade_date, shares, price, bar_timestamp,
               run_id, position_after
        FROM trade_executions
        WHERE ticker = ?
          AND action = 'BUY'
          AND is_close = 0
          AND id NOT IN (
              SELECT DISTINCT entry_trade_id
              FROM trade_executions
              WHERE entry_trade_id IS NOT NULL
          )
        ORDER BY trade_date, id
    """, (ticker,))
    return cur.fetchall()


def match_fifo(unlinked_sell, orphaned_buys):
    """FIFO matching: match SELL to earliest unmatched BUY(s) with sufficient shares.

    Args:
        unlinked_sell: (id, ticker, date, shares, price, pnl, bar, run_id)
        orphaned_buys: List of (id, ticker, date, shares, price, bar, run_id, pos_after)

    Returns:
        Best matching BUY id or None
    """
    sell_id, ticker, sell_date, sell_shares, sell_price, pnl, sell_bar, sell_run = unlinked_sell

    # Strategy: Match to BUY where:
    # 1. Same run_id context (within a few runs of each other)
    # 2. BUY timestamp <= SELL timestamp
    # 3. Shares match (accounting for potential partial closes)

    candidates = []
    for buy in orphaned_buys:
        buy_id, buy_ticker, buy_date, buy_shares, buy_price, buy_bar, buy_run, pos_after = buy

        # Must be before or same time
        if buy_date > sell_date:
            continue

        # Prefer same-day or close run_id
        run_distance = abs(int(sell_run.split('_')[1]) - int(buy_run.split('_')[1]))

        # Score: closer run_id + exact share match preferred
        share_match = (sell_shares == buy_shares)
        score = -run_distance + (100 if share_match else 0)

        candidates.append((score, buy_id, buy_shares, run_distance, share_match))

    if not candidates:
        return None

    # Sort by score descending, return best match
    candidates.sort(reverse=True)
    best_score, best_buy_id, best_shares, dist, exact = candidates[0]

    return best_buy_id


def repair_linkage(db_path: Path, dry_run: bool = True):
    """Backfill entry_trade_id for unlinked closes."""
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    print("=" * 70)
    print("REPAIR UNLINKED CLOSES")
    print("=" * 70)
    print()

    unlinked = find_unlinked_closes(conn)
    if not unlinked:
        print("[OK] No unlinked closes found")
        conn.close()
        return 0

    print(f"Found {len(unlinked)} unlinked closing legs")
    print()

    repairs = []

    for sell in unlinked:
        sell_id, ticker, sell_date, shares, price, pnl, bar, run_id = sell
        print(f"Unlinked SELL ID {sell_id}: {ticker} {sell_date} - {shares} shares @ {price:.2f}")
        print(f"  PnL: ${pnl:.2f}, Run: {run_id}")

        # Find orphaned BUYs for this ticker
        orphans = find_orphaned_entries(conn, ticker)
        if not orphans:
            print(f"  [WARNING] No orphaned BUYs found for {ticker}")
            print()
            continue

        # Match FIFO
        matched_buy_id = match_fifo(sell, orphans)
        if matched_buy_id:
            # Get BUY details
            buy = next(b for b in orphans if b[0] == matched_buy_id)
            buy_id, _, buy_date, buy_shares, buy_price, buy_bar, buy_run, _ = buy

            print(f"  [MATCH] BUY ID {buy_id}: {buy_date} - {buy_shares} shares @ {buy_price:.2f}")
            print(f"          Run: {buy_run}")

            repairs.append((sell_id, buy_id, ticker, shares))
        else:
            print(f"  [WARNING] No suitable BUY match found")

        print()

    if not repairs:
        print("[WARNING] No repairs identified")
        conn.close()
        return 0

    print(f"Identified {len(repairs)} repairs")
    print()

    if dry_run:
        print("[DRY RUN] Would apply the following repairs:")
        for sell_id, buy_id, ticker, shares in repairs:
            print(f"  UPDATE trade_executions SET entry_trade_id = {buy_id} WHERE id = {sell_id}")
        print()
        print("Re-run with --apply to execute")
        conn.close()
        return 0

    # Apply repairs
    print("Applying repairs...")
    try:
        for sell_id, buy_id, ticker, shares in repairs:
            conn.execute(
                "UPDATE trade_executions SET entry_trade_id = ? WHERE id = ?",
                (buy_id, sell_id)
            )
            print(f"  [OK] Linked SELL {sell_id} -> BUY {buy_id} ({ticker})")

        conn.commit()
        print()
        print("[SUCCESS] All repairs applied")

    except sqlite3.Error as e:
        print(f"[ERROR] Repair failed: {e}")
        conn.rollback()
        conn.close()
        return 1

    conn.close()

    # Verify repairs
    print()
    print("Verifying repairs...")
    conn = sqlite3.connect(db_path)
    remaining = find_unlinked_closes(conn)
    conn.close()

    print(f"Remaining unlinked closes: {len(remaining)}")

    if remaining:
        print("[WARNING] Some closes still unlinked:")
        for row in remaining:
            print(f"  ID {row[0]}: {row[1]} {row[2]}")
        return 1
    else:
        print("[OK] All closes now linked")
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--apply", action="store_true", help="Apply repairs (default is dry-run)")

    args = parser.parse_args()

    sys.exit(repair_linkage(args.db, dry_run=not args.apply))
