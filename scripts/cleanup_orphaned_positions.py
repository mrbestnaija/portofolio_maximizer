#!/usr/bin/env python3
"""Cleanup orphaned open positions in trade_executions.

Phase 7.9 audit sprint opened positions across multiple as-of-date passes,
but portfolio_state only persisted the last pass's positions.  This left
BUY rows with realized_pnl IS NULL that can never be closed by the engine.

Two categories:
1. *Paired BUYs* -- opening leg whose matching SELL (is_close=1) already
   exists.  Backfill the BUY's realized_pnl from the SELL to keep the
   ledger consistent.
2. *Truly orphaned BUYs* -- no matching SELL.  Replay the ATR-adaptive
   exit logic against actual subsequent market data (yfinance) to find
   the realistic exit price, reason, and date.

Safe to run multiple times (idempotent).
"""

import argparse
import sqlite3
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "portfolio_maximizer.db")


def _resolve_as_of_date(ticker: str, entry_price: float, hist_close: pd.Series) -> str:
    """Find the trading date whose close price best matches the entry price."""
    diffs = abs(hist_close - entry_price)
    best_idx = diffs.idxmin()
    return best_idx.strftime("%Y-%m-%d") if hasattr(best_idx, "strftime") else str(best_idx)[:10]


def _replay_atr_exit(
    hist: pd.DataFrame, entry_price: float, shares: float, as_of: str
) -> dict:
    """Replay proof-mode ATR-adaptive exit logic on real market data.

    Returns dict with exit_price, exit_reason, pnl, pnl_pct, bars_held, exit_date.
    """
    close = hist["Close"].squeeze()
    atr_series = (hist["High"] - hist["Low"]).squeeze().rolling(14).mean()

    entry_date = pd.Timestamp(as_of)
    entry_loc = hist.index.get_indexer([entry_date], method="pad")[0]
    if entry_loc < 14:
        entry_loc = 14

    atr = float(atr_series.iloc[entry_loc])
    atr_pct = atr / entry_price if entry_price > 0 else 0

    # ATR-adaptive holding (mirrors run_auto_trader.py proof-mode logic)
    default_horizon = 5
    if atr_pct > 0.03:
        proof_horizon = 3
        stop_mult, target_mult = 1.0, 1.5
    elif atr_pct > 0.015:
        proof_horizon = default_horizon
        stop_mult, target_mult = 1.25, 1.75
    else:
        proof_horizon = default_horizon + 2
        stop_mult, target_mult = 1.25, 1.75

    stop_loss = entry_price - stop_mult * atr
    target_price = entry_price + target_mult * atr

    # Evaluate exit bar by bar
    exit_price = None
    exit_reason = None
    exit_date = None
    bars_held = 0

    for i in range(entry_loc + 1, min(entry_loc + proof_horizon + 5, len(close))):
        bars_held += 1
        px = float(close.iloc[i])

        if px <= stop_loss:
            exit_price, exit_reason = px, "STOP_LOSS"
            exit_date = close.index[i]
            break
        elif px >= target_price:
            exit_price, exit_reason = px, "TAKE_PROFIT"
            exit_date = close.index[i]
            break
        elif bars_held >= proof_horizon:
            exit_price, exit_reason = px, "TIME_EXIT"
            exit_date = close.index[i]
            break

    if exit_price is None:
        exit_price = float(close.iloc[-1])
        exit_reason = "TIME_EXIT"
        exit_date = close.index[-1]
        bars_held = len(close) - entry_loc - 1

    pnl = (exit_price - entry_price) * shares
    pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
    exit_date_str = exit_date.strftime("%Y-%m-%d") if hasattr(exit_date, "strftime") else str(exit_date)[:10]

    return {
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "bars_held": bars_held,
        "exit_date": exit_date_str,
    }


def main(dry_run: bool = True) -> None:
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ---- Gather orphan BUYs (realized_pnl IS NULL) ----
    orphan_buys = conn.execute(
        "SELECT id, ticker, shares, price FROM trade_executions "
        "WHERE realized_pnl IS NULL ORDER BY id"
    ).fetchall()

    if not orphan_buys:
        print("[OK] No orphaned positions found.")
        conn.close()
        return

    # ---- Gather closed SELLs ----
    closed_sells = conn.execute(
        "SELECT id, ticker, shares, entry_price, realized_pnl, realized_pnl_pct, "
        "       exit_price, holding_period_days, exit_reason "
        "FROM trade_executions "
        "WHERE realized_pnl IS NOT NULL AND is_close = 1 "
        "ORDER BY id"
    ).fetchall()

    # ---- Match paired BUYs ----
    paired_buy_ids = {}  # buy_id -> sell_row
    used_sell_ids = set()
    for sell in closed_sells:
        for buy in orphan_buys:
            if (
                buy["id"] not in paired_buy_ids
                and sell["id"] not in used_sell_ids
                and buy["ticker"] == sell["ticker"]
                and abs(buy["price"] - sell["entry_price"]) < 0.01
                and abs(buy["shares"] - sell["shares"]) < 0.01
            ):
                paired_buy_ids[buy["id"]] = sell
                used_sell_ids.add(sell["id"])
                break

    truly_orphaned = [b for b in orphan_buys if b["id"] not in paired_buy_ids]

    print(f"Orphan BUYs:      {len(orphan_buys)}")
    print(f"  Paired (have matching SELL): {len(paired_buy_ids)}")
    print(f"  Truly orphaned (no SELL):    {len(truly_orphaned)}")
    print()

    # ---- Fix 1: Backfill paired BUYs from their matching SELL ----
    print("=== Paired BUYs: backfill realized_pnl from matching SELL ===")
    for buy_id, sell in sorted(paired_buy_ids.items()):
        print(
            f"  id={buy_id} {sell['ticker']} "
            f"pnl={sell['realized_pnl']:+.2f} exit={sell['exit_reason']}"
        )
        if not dry_run:
            conn.execute(
                "UPDATE trade_executions SET "
                "  realized_pnl = ?, realized_pnl_pct = ?, "
                "  entry_price = price, "
                "  exit_price = ?, holding_period_days = ?, "
                "  exit_reason = ?, is_close = 0 "
                "WHERE id = ?",
                (
                    sell["realized_pnl"],
                    sell["realized_pnl_pct"],
                    sell["exit_price"],
                    sell["holding_period_days"],
                    sell["exit_reason"],
                    buy_id,
                ),
            )

    # ---- Fix 2: Replay ATR exit on real market data for truly orphaned BUYs ----
    if truly_orphaned:
        print()
        print("=== Truly orphaned BUYs: replay ATR exit on real market data ===")
        print(
            f"{'id':>4} {'Ticker':<6} {'Entry':>8} {'Exit':>8} "
            f"{'PnL':>8} {'PnL%':>7} {'Bars':>4} {'Reason':<14} {'ExitDate'}"
        )
        print("-" * 82)

        # Cache historical data per ticker
        hist_cache = {}
        for buy in truly_orphaned:
            ticker = buy["ticker"]
            if ticker not in hist_cache:
                hist_cache[ticker] = yf.download(
                    ticker, start="2024-09-01", end="2026-02-10", progress=False
                )

        for buy in truly_orphaned:
            ticker = buy["ticker"]
            entry_price = buy["price"]
            shares = buy["shares"]
            hist = hist_cache[ticker]

            if hist.empty:
                print(f"{buy['id']:>4} {ticker:<6} NO DATA -- skipping")
                continue

            close_series = hist["Close"].squeeze()
            as_of = _resolve_as_of_date(ticker, entry_price, close_series)
            result = _replay_atr_exit(hist, entry_price, shares, as_of)

            print(
                f"{buy['id']:>4} {ticker:<6} {entry_price:>8.2f} "
                f"{result['exit_price']:>8.2f} {result['pnl']:>+8.2f} "
                f"{result['pnl_pct']:>+6.2%} {result['bars_held']:>4} "
                f"{result['exit_reason']:<14} {result['exit_date']}"
            )

            if not dry_run:
                conn.execute(
                    "UPDATE trade_executions SET "
                    "  realized_pnl = ?, realized_pnl_pct = ?, "
                    "  entry_price = price, "
                    "  exit_price = ?, holding_period_days = ?, "
                    "  exit_reason = ?, is_close = 0 "
                    "WHERE id = ?",
                    (
                        result["pnl"],
                        result["pnl_pct"],
                        result["exit_price"],
                        result["bars_held"],
                        result["exit_reason"],
                        buy["id"],
                    ),
                )

    if dry_run:
        print()
        print("[DRY RUN] No changes made. Re-run with --apply to execute.")
    else:
        conn.commit()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NULL"
        ).fetchone()[0]
        total_closed = conn.execute(
            "SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NOT NULL"
        ).fetchone()[0]
        total_pnl = conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM trade_executions "
            "WHERE realized_pnl IS NOT NULL"
        ).fetchone()[0]
        print()
        print(f"[APPLIED] Remaining open: {remaining}, Total closed: {total_closed}")
        print(f"[APPLIED] Total realized PnL: ${total_pnl:.2f}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry-run)",
    )
    args = parser.parse_args()
    main(dry_run=not args.apply)
