#!/usr/bin/env python3
"""compute_capital_utilization.py

Time-weighted capital utilization KPI for the NGN hurdle plan.

Formula (exact, pinned for regression):
    deployment_fraction = sum(notional_i * hold_days_i) / (capital * total_days)

Where:
    notional_i     = entry_price * shares  for each closed round-trip
    hold_days_i    = holding_period_days   for that round-trip (from production_closed_trades)
    capital        = portfolio_cash_state.initial_capital
    total_days     = calendar days from first open to last close

This is a time-weighted measure of how much of the capital base was deployed per day
on average. It is the correct denominator for reasoning about capital efficiency and
trade-frequency leverage. It is NOT the same as:
    - avg notional per trade / capital  (overstates deployment by ~3.4x at 1.4d avg hold)
    - total notional / capital          (cumulative, not time-weighted)

Outputs a JSON artifact to logs/capital_utilization_latest.json and optionally prints
a human-readable summary.

CLI:
    python scripts/compute_capital_utilization.py [--db PATH] [--output PATH] [--json]
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_OUTPUT = ROOT / "logs" / "capital_utilization_latest.json"


def compute_utilization(
    db_path: Path,
    capital: Optional[float] = None,
) -> dict:
    """Compute time-weighted capital utilization from production_closed_trades.

    Returns a dict with all intermediate values so callers can verify the formula.
    Raises ValueError if no closed trades exist.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        # Confirmed capital base
        if capital is None:
            row = cur.execute(
                "SELECT initial_capital FROM portfolio_cash_state ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("portfolio_cash_state is empty; cannot determine capital base")
            capital = float(row[0])

        # Sum(notional * hold_days) from production_closed_trades
        # production_closed_trades excludes synthetic and diagnostic trades
        cur.execute("""
            SELECT
                COUNT(*)                                          AS n_trips,
                SUM(o.price * o.shares * c.holding_period_days)  AS notional_days,
                SUM(o.price * o.shares)                          AS total_notional,
                AVG(o.price * o.shares)                          AS avg_notional,
                AVG(c.holding_period_days)                       AS avg_hold_days,
                SUM(c.realized_pnl)                              AS total_pnl,
                SUM(CASE WHEN c.realized_pnl > 0 THEN 1 ELSE 0 END) AS n_wins,
                MIN(o.trade_date)                                AS first_open,
                MAX(c.trade_date)                                AS last_close
            FROM production_closed_trades c
            JOIN trade_executions o ON c.entry_trade_id = o.id
        """)
        row = cur.fetchone()
        if row is None or row[0] == 0:
            raise ValueError("No closed round-trips found in production_closed_trades")

        (
            n_trips, notional_days, total_notional, avg_notional,
            avg_hold_days, total_pnl, n_wins, first_open, last_close,
        ) = row

        if notional_days is None or notional_days == 0:
            raise ValueError(
                "notional_days is zero — holding_period_days may be NULL for all rows. "
                "Cannot compute time-weighted utilization."
            )

        import datetime as _dt
        d1 = _dt.datetime.fromisoformat(first_open[:10])
        d2 = _dt.datetime.fromisoformat(last_close[:10])
        total_days = max((d2 - d1).days, 1)

        # Core formula — pinned
        twc_per_day = notional_days / total_days
        deployment_fraction = twc_per_day / capital
        notional_overstatement = avg_notional / twc_per_day if twc_per_day > 0 else None

        # ROI metrics
        win_rate = n_wins / n_trips
        roi_cum = total_pnl / capital
        roi_ann = roi_cum * 365 / total_days

        # Trades/day
        trades_per_day = n_trips / total_days

        return {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "capital": capital,
            "n_trips": n_trips,
            "total_days": total_days,
            "first_open": first_open[:10],
            "last_close": last_close[:10],
            # --- Core KPI ---
            "notional_days": round(notional_days, 2),
            "twc_per_day": round(twc_per_day, 2),
            "deployment_fraction": round(deployment_fraction, 6),
            "deployment_pct": round(deployment_fraction * 100, 2),
            # --- Formula audit ---
            "formula": "notional_days / (capital * total_days)",
            "avg_notional_per_trade": round(avg_notional, 2),
            "avg_hold_days": round(avg_hold_days or 0, 2),
            "avg_notional_overstatement_factor": (
                round(notional_overstatement, 2) if notional_overstatement else None
            ),
            # --- Edge metrics ---
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 4),
            "roi_cum_pct": round(roi_cum * 100, 4),
            "roi_ann_pct": round(roi_ann * 100, 2),
            "trades_per_day": round(trades_per_day, 3),
            # --- Utilization scenarios ---
            "scenarios": {
                "current": {
                    "trades_per_day": round(trades_per_day, 3),
                    "proj_pnl": round(total_pnl, 2),
                    "roi_ann_pct": round(roi_ann * 100, 2),
                },
                "partial_unblock_0_95": _project(
                    total_pnl, n_trips, total_days, capital, 0.95
                ),
                "target_1_40": _project(
                    total_pnl, n_trips, total_days, capital, 1.40
                ),
            },
        }
    finally:
        conn.close()


def _project(
    base_pnl: float,
    base_trips: int,
    total_days: int,
    capital: float,
    target_trades_per_day: float,
) -> dict:
    scale = target_trades_per_day / (base_trips / total_days)
    proj_pnl = base_pnl * scale
    roi_ann = (proj_pnl / capital) * 365 / total_days * 100
    return {
        "trades_per_day": target_trades_per_day,
        "scale_factor": round(scale, 2),
        "proj_pnl": round(proj_pnl, 2),
        "roi_ann_pct": round(roi_ann, 1),
        "note": "assumes identical per-trade edge distribution; no slippage adjustment",
    }


@click.command()
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="SQLite DB path")
@click.option(
    "--output", default=str(DEFAULT_OUTPUT), show_default=True, help="Output JSON artifact"
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON to stdout only")
def main(db: str, output: str, as_json: bool) -> None:
    result = compute_utilization(Path(db))

    if as_json:
        print(json.dumps(result, indent=2))
        return

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Capital utilization KPI")
    print(f"  Capital base       : ${result['capital']:,.0f}")
    print(f"  Trades             : {result['n_trips']}  over {result['total_days']} days")
    print(f"  Trades/day         : {result['trades_per_day']:.2f}")
    print(f"  Time-weighted dep. : ${result['twc_per_day']:.0f}/day"
          f"  = {result['deployment_pct']:.1f}% of capital")
    print(f"  Avg notional       : ${result['avg_notional_per_trade']:.0f}  "
          f"(overstates by {result['avg_notional_overstatement_factor']:.1f}x vs time-weighted)")
    print(f"  Current ann ROI    : {result['roi_ann_pct']:.1f}%")
    print(f"  Gap to 28% ann     :",
          f"{28 / result['roi_ann_pct']:.2f}x"
          if result['roi_ann_pct'] > 0 else "N/A")
    print()
    print("  Scenarios (same per-trade edge, no slippage adjustment):")
    for name, s in result["scenarios"].items():
        hurdle = " ✓ NGN" if s["roi_ann_pct"] >= 28 else ""
        print(f"    {s['trades_per_day']:.2f}/day -> ${s['proj_pnl']:,.0f}"
              f"  ({s['roi_ann_pct']:.1f}% ann){hurdle}")
    print(f"\nArtifact: {output}")


if __name__ == "__main__":
    main()
