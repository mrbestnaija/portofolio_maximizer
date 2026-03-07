"""
Audit exit quality to explain the forecast DA vs trade win-rate gap.

Reads production_closed_trades (is_close=1, non-diagnostic, non-synthetic)
and decomposes win/loss by exit_reason (stop_loss, time_exit, signal_exit,
flatten_before_reverse, NULL).

Key diagnostics:
- Per-reason win rate and mean/median PnL
- Median R-multiple (realized PnL / risk-unit proxy)
- "Correct direction, negative PnL" count — forecast right but exit behaviour lost
- Gap interpretation: stop_too_tight | holding_too_short | mix

Usage:
    python scripts/exit_quality_audit.py [--db data/portfolio_maximizer.db] [--tail-n 100]

Exit codes:
    0 = audit complete (non-blocking diagnostic)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "portfolio_maximizer.db"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Exit reasons we expect; NULL maps to "unknown"
KNOWN_REASONS = {"stop_loss", "time_exit", "signal_exit", "flatten_before_reverse"}


def load_production_trades(db_path: Path, tail_n: int | None = None) -> pd.DataFrame:
    """Load production closed trades from the canonical view.

    Uses production_closed_trades (is_close=1, non-diagnostic, non-synthetic).
    Columns returned: ticker, trade_date, action, exit_reason, realized_pnl,
    entry_price, exit_price, bar_high, bar_low, holding_period_days.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        return pd.DataFrame()

    query = """
        SELECT
            ticker,
            trade_date,
            action,
            COALESCE(exit_reason, 'unknown') AS exit_reason,
            realized_pnl,
            entry_price,
            exit_price,
            bar_high,
            bar_low,
            COALESCE(holding_period_days, 0) AS holding_period_days
        FROM production_closed_trades
        ORDER BY trade_date DESC
    """
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as exc:
        log.error("Failed to query database: %s", exc)
        return pd.DataFrame()

    if df.empty:
        log.info("No production closed trades found.")
        return df

    if tail_n is not None and tail_n > 0:
        df = df.head(tail_n)  # already sorted DESC, head = most recent N

    # Normalize numerics first to avoid dtype instability on sparse/legacy rows.
    realized_pnl = pd.to_numeric(df["realized_pnl"], errors="coerce")
    entry_price = pd.to_numeric(df["entry_price"], errors="coerce")
    exit_price = pd.to_numeric(df["exit_price"], errors="coerce")
    bar_high = pd.to_numeric(df["bar_high"], errors="coerce")
    bar_low = pd.to_numeric(df["bar_low"], errors="coerce")

    # Derived: is_winner, atr_proxy, r_multiple
    df["is_winner"] = (realized_pnl.fillna(0.0) > 0).astype(int)

    # ATR proxy = bar_high - bar_low (single-bar range); fallback to 1.5% of entry.
    has_atr = bar_high.notna() & bar_low.notna() & (entry_price > 0)
    df["atr_proxy"] = np.nan
    if bool(has_atr.any()):
        atr_values = (bar_high - bar_low).where(has_atr)
        df.loc[has_atr, "atr_proxy"] = atr_values.loc[has_atr].astype(float)

    # Fallback: 1.5% of entry_price
    no_atr = df["atr_proxy"].isna() & entry_price.notna() & (entry_price > 0)
    if bool(no_atr.any()):
        df.loc[no_atr, "atr_proxy"] = (entry_price.loc[no_atr] * 0.015).astype(float)

    risk_unit = (df["atr_proxy"] * 1.5).replace(0.0, np.nan)
    df["r_multiple"] = realized_pnl.fillna(0.0) / risk_unit

    # Correct direction but negative PnL:
    # BUY: exit_price > entry_price AND realized_pnl < 0
    # SELL: exit_price < entry_price AND realized_pnl < 0
    ep = entry_price.fillna(0.0)
    xp = exit_price.fillna(0.0)
    pnl = realized_pnl.fillna(0.0)
    buy_right_lost = (df["action"] == "BUY") & (xp > ep) & (pnl < 0)
    sell_right_lost = (df["action"] == "SELL") & (xp < ep) & (pnl < 0)
    df["correct_dir_neg_pnl"] = (buy_right_lost | sell_right_lost).astype(int)

    return df


def compute_exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Group by exit_reason and compute per-group statistics.

    Returns a DataFrame with columns:
    exit_reason, count, pct_of_total, win_rate, mean_pnl, median_pnl, median_r_multiple
    """
    if trades.empty:
        return pd.DataFrame(
            columns=["exit_reason", "count", "pct_of_total", "win_rate",
                     "mean_pnl", "median_pnl", "median_r_multiple"]
        )

    total = len(trades)
    records = []
    for reason, grp in trades.groupby("exit_reason"):
        records.append({
            "exit_reason": reason,
            "count": len(grp),
            "pct_of_total": len(grp) / total,
            "win_rate": grp["is_winner"].mean(),
            "mean_pnl": grp["realized_pnl"].mean(),
            "median_pnl": grp["realized_pnl"].median(),
            "median_r_multiple": grp["r_multiple"].median(),
        })

    df_out = pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)
    return df_out


def diagnose_direction_gap(trades: pd.DataFrame) -> dict:
    """Explain the forecast DA → trade win-rate gap.

    Returns a dict with key metrics and an interpretation label.
    """
    if trades.empty:
        return {
            "total_trades": 0,
            "overall_win_rate": None,
            "stop_loss_pct": 0.0,
            "time_exit_pct": 0.0,
            "signal_exit_pct": 0.0,
            "stop_loss_win_rate": None,
            "time_exit_win_rate": None,
            "signal_exit_win_rate": None,
            "correct_direction_negative_pnl": 0,
            "pct_correct_dir_neg_pnl": 0.0,
            "mean_holding_days_winners": None,
            "mean_holding_days_losers": None,
            "median_r_multiple_by_reason": {},
            "interpretation": "no_data",
        }

    n = len(trades)
    overall_wr = trades["is_winner"].mean()

    def _pct(reason: str) -> float:
        return (trades["exit_reason"] == reason).sum() / n

    def _wr(reason: str) -> float | None:
        sub = trades[trades["exit_reason"] == reason]
        return float(sub["is_winner"].mean()) if not sub.empty else None

    def _med_r(reason: str) -> float | None:
        sub = trades[trades["exit_reason"] == reason]
        return float(sub["r_multiple"].median()) if not sub.empty and sub["r_multiple"].notna().any() else None

    stop_pct = _pct("stop_loss")
    time_pct = _pct("time_exit")
    signal_pct = _pct("signal_exit")

    winners = trades[trades["is_winner"] == 1]
    losers = trades[trades["is_winner"] == 0]

    correct_dir_neg = int(trades["correct_dir_neg_pnl"].sum())

    reasons = trades["exit_reason"].unique()
    med_r_by_reason = {r: _med_r(r) for r in reasons}

    # Interpretation
    _time_wr = _wr("time_exit")
    if stop_pct > 0.40:
        interpretation = "stop_too_tight"
    elif time_pct > 0.40 and (_time_wr if _time_wr is not None else 0.5) < 0.45:
        interpretation = "holding_too_short"
    else:
        interpretation = "mix"

    return {
        "total_trades": n,
        "overall_win_rate": float(overall_wr),
        "stop_loss_pct": float(stop_pct),
        "time_exit_pct": float(time_pct),
        "signal_exit_pct": float(signal_pct),
        "stop_loss_win_rate": _wr("stop_loss"),
        "time_exit_win_rate": _wr("time_exit"),
        "signal_exit_win_rate": _wr("signal_exit"),
        "correct_direction_negative_pnl": correct_dir_neg,
        "pct_correct_dir_neg_pnl": float(correct_dir_neg / n),
        "mean_holding_days_winners": float(winners["holding_period_days"].mean()) if not winners.empty else None,
        "mean_holding_days_losers": float(losers["holding_period_days"].mean()) if not losers.empty else None,
        "median_r_multiple_by_reason": med_r_by_reason,
        "interpretation": interpretation,
    }


def _print_report(breakdown: pd.DataFrame, gap: dict) -> None:
    """Print a readable summary to stdout."""
    log.info("=== Exit Quality Audit ===")
    log.info("Total production closed trades: %d", gap["total_trades"])
    if gap["total_trades"] == 0:
        log.info("[OK] No trades to audit.")
        return

    log.info("Overall win rate: %.1f%%", (gap["overall_win_rate"] or 0) * 100)
    log.info("")
    log.info("--- Exit reason breakdown ---")
    for _, row in breakdown.iterrows():
        r_str = (
            f"R={row['median_r_multiple']:.2f}"
            if pd.notna(row["median_r_multiple"])
            else "R=n/a"
        )
        log.info(
            "  %-28s  count=%3d  pct=%4.1f%%  win=%.0f%%  mean_pnl=$%+.2f  %s",
            row["exit_reason"],
            row["count"],
            row["pct_of_total"] * 100,
            (row["win_rate"] or 0) * 100,
            row["mean_pnl"] or 0,
            r_str,
        )
    log.info("")
    log.info("--- Gap diagnosis ---")
    log.info("  Stop-loss pct: %.1f%%", gap["stop_loss_pct"] * 100)
    log.info("  Time-exit pct: %.1f%%  win-rate: %s",
             gap["time_exit_pct"] * 100,
             f"{gap['time_exit_win_rate']:.0%}" if gap["time_exit_win_rate"] is not None else "n/a")
    log.info("  Correct direction but negative PnL: %d (%.1f%% of trades)",
             gap["correct_direction_negative_pnl"],
             gap["pct_correct_dir_neg_pnl"] * 100)
    log.info("  Mean holding days (winners): %s",
             f"{gap['mean_holding_days_winners']:.1f}" if gap["mean_holding_days_winners"] is not None else "n/a")
    log.info("  Mean holding days (losers):  %s",
             f"{gap['mean_holding_days_losers']:.1f}" if gap["mean_holding_days_losers"] is not None else "n/a")
    log.info("")
    log.info("  Interpretation: [%s]", gap["interpretation"].upper())
    if gap["interpretation"] == "stop_too_tight":
        log.info("    > Stop-loss exits dominate (>40%%). "
                 "Stops placed too close — consider wider ATR multiplier.")
    elif gap["interpretation"] == "holding_too_short":
        log.info("    > Time-exits dominate AND win-rate <45%%. "
                 "Positions closed before gains develop — consider wider max_holding.")
    else:
        log.info("    > Mixed causes. Inspect stop-loss AND time-exit populations.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit exit quality vs forecast quality gap.")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Path to portfolio_maximizer.db",
    )
    parser.add_argument(
        "--tail-n",
        type=int,
        default=None,
        help="Limit to most recent N trades (default: all)",
    )
    args = parser.parse_args(argv)

    trades = load_production_trades(Path(args.db), tail_n=args.tail_n)
    breakdown = compute_exit_reason_breakdown(trades)
    gap = diagnose_direction_gap(trades)
    _print_report(breakdown, gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
