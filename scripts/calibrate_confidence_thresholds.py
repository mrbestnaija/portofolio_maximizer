"""
P3-A: Calibrate confidence thresholds from realized trades.

Bins closed round-trip trades by confidence quintile and reports win-rate,
trade count, and realized PnL per bin. Output written to
logs/confidence_calibration.json for downstream threshold calibration, with
timestamped archive copies under logs/confidence_calibration_history/.

Usage:
    python scripts/calibrate_confidence_thresholds.py [--db PATH] [--bins N]

Exit codes:
    0 — success (calibration JSON written)
    1 — error (DB not found or insufficient data)
    2 — cold-start (fewer than MIN_TRADES usable trades)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH))

from utils.evidence_io import write_versioned_json_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate_confidence_thresholds")

DEFAULT_DB = ROOT_PATH / "data" / "portfolio_maximizer.db"
OUTPUT_PATH = ROOT_PATH / "logs" / "confidence_calibration.json"
ARCHIVE_DIR = ROOT_PATH / "logs" / "confidence_calibration_history"
MIN_TRADES = 20  # below this threshold, calibration is unreliable


_QUERY = """
SELECT
    COALESCE(o.confidence_calibrated, o.effective_confidence) AS conf,
    c.realized_pnl,
    CASE WHEN c.realized_pnl > 0 THEN 1 ELSE 0 END AS win,
    c.ticker,
    c.exit_reason
FROM trade_executions c
JOIN trade_executions o ON c.entry_trade_id = o.id
WHERE
    c.is_close = 1
    AND COALESCE(c.is_diagnostic, 0) = 0
    AND COALESCE(c.is_synthetic, 0) = 0
    AND COALESCE(o.confidence_calibrated, o.effective_confidence) IS NOT NULL
    AND c.realized_pnl IS NOT NULL
ORDER BY conf ASC
"""

# Mechanical exits are directionally uninformative — exclude from calibration
_MECHANICAL_EXIT_REASONS = frozenset(
    {"stop_loss", "max_holding", "time_exit", "forced_exit", "flatten"}
)


def _load_trades(db_path: Path) -> list[dict]:
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(_QUERY)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def _bin_trades(rows: list[dict], n_bins: int) -> list[dict]:
    confs = np.array([r["conf"] for r in rows], dtype=float)
    wins = np.array([r["win"] for r in rows], dtype=int)
    pnls = np.array([r["realized_pnl"] for r in rows], dtype=float)

    edges = np.quantile(confs, np.linspace(0.0, 1.0, n_bins + 1))
    # Deduplicate edges to handle clustered confidence values
    edges = np.unique(edges)

    bins: list[dict] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bin_wins = wins[mask]
        bin_pnls = pnls[mask]
        bins.append(
            {
                "conf_low": round(lo, 4),
                "conf_high": round(hi, 4),
                "n_trades": n,
                "win_rate": round(float(bin_wins.mean()), 4),
                "mean_pnl": round(float(bin_pnls.mean()), 4),
                "total_pnl": round(float(bin_pnls.sum()), 4),
            }
        )
    return bins


def run(db_path: Path, n_bins: int) -> dict:
    rows = _load_trades(db_path)
    # Filter mechanical exits — directionally uninformative
    filtered = [
        r for r in rows
        if str(r.get("exit_reason") or "").lower() not in _MECHANICAL_EXIT_REASONS
    ]
    n_total = len(rows)
    n_filtered = n_total - len(filtered)
    logger.info(
        "Loaded %d round-trips; excluded %d mechanical exits; %d for calibration",
        n_total, n_filtered, len(filtered),
    )

    if len(filtered) < MIN_TRADES:
        logger.warning(
            "Only %d directional trades — cold-start (need %d). Writing partial output.",
            len(filtered), MIN_TRADES,
        )
        status = "cold_start"
    else:
        status = "ok"

    bins = _bin_trades(filtered, n_bins) if filtered else []

    # Breakeven win-rate: profit_factor (mean_win / mean_loss) implies
    # breakeven_wr = 1 / (1 + profit_factor)
    # We compute it from the data.
    wins_pnl = [r["realized_pnl"] for r in filtered if r["win"]]
    losses_pnl = [r["realized_pnl"] for r in filtered if not r["win"]]
    mean_win = float(np.mean(wins_pnl)) if wins_pnl else None
    mean_loss = float(np.mean(losses_pnl)) if losses_pnl else None
    if mean_win is not None and mean_loss is not None and mean_loss < 0:
        profit_factor = mean_win / abs(mean_loss)
        breakeven_wr = 1.0 / (1.0 + profit_factor)
    else:
        profit_factor = None
        breakeven_wr = None

    payload = {
        "status": status,
        "n_total_closed_trades": n_total,
        "n_mechanical_exits_excluded": n_filtered,
        "n_directional_trades": len(filtered),
        "n_bins_requested": n_bins,
        "n_bins_produced": len(bins),
        "mean_win_pnl": round(mean_win, 4) if mean_win is not None else None,
        "mean_loss_pnl": round(mean_loss, 4) if mean_loss is not None else None,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "breakeven_win_rate": round(breakeven_wr, 4) if breakeven_wr is not None else None,
        "bins": bins,
        "interpretation": (
            "win_rate should be monotonically non-decreasing with conf_low. "
            "If win_rate is flat or decreasing, confidence scores are not well-calibrated. "
            "Compare each bin's win_rate to breakeven_win_rate to identify the minimum "
            "confidence threshold above which execution generates positive expected value."
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate confidence thresholds from realized trades")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to portfolio_maximizer.db")
    parser.add_argument("--bins", type=int, default=5, help="Number of quantile bins (default 5)")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    try:
        payload = run(args.db, args.bins)
    except FileNotFoundError:
        return 1

    status = payload["status"]

    write_result = write_versioned_json_artifact(
        latest_path=OUTPUT_PATH,
        payload=payload,
        archive_root=ARCHIVE_DIR,
        archive_name="confidence_calibration",
    )
    logger.info(
        "Calibration written to %s (archive=%s)",
        OUTPUT_PATH,
        write_result.get("archive_path"),
    )

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\nConfidence Calibration — {payload['n_directional_trades']} directional trades")
        print(f"Profit factor: {payload['profit_factor']}  Breakeven WR: {payload['breakeven_win_rate']}")
        print(f"{'Bin':>20s}  {'N':>5s}  {'Win%':>6s}  {'Mean PnL':>10s}")
        print("-" * 50)
        for b in payload["bins"]:
            label = f"[{b['conf_low']:.3f}, {b['conf_high']:.3f}]"
            print(f"{label:>20s}  {b['n_trades']:>5d}  {b['win_rate']*100:>5.1f}%  {b['mean_pnl']:>10.2f}")

    return 2 if status == "cold_start" else 0


if __name__ == "__main__":
    sys.exit(main())
