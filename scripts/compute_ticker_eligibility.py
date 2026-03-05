"""
Read-only per-ticker eligibility classification.

Public status enum is intentionally strict:
  - HEALTHY
  - WEAK
  - LAB_ONLY

This script never modifies routing config or any gate threshold. It only
computes recommendations from existing trade outcomes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality_pipeline_common import append_threshold_hash_change_warning, connect_ro
from scripts.robustness_thresholds import (
    R3_MIN_PROFIT_FACTOR,
    R3_MIN_TRADES,
    R3_MIN_WIN_RATE,
    WEAK_MAX_PROFIT_FACTOR,
    WEAK_MAX_WIN_RATE,
    WEAK_MIN_TRADES,
    threshold_map,
)

log = logging.getLogger(__name__)

DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_OUTPUT = ROOT / "logs" / "ticker_eligibility.json"

HEALTHY_MIN_WIN_RATE = R3_MIN_WIN_RATE
HEALTHY_MIN_PROFIT_FACTOR = R3_MIN_PROFIT_FACTOR
HEALTHY_MIN_TRADES = R3_MIN_TRADES
ALL_STATUSES = ("HEALTHY", "WEAK", "LAB_ONLY")


def _query_per_ticker(db_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return per-ticker stats from production_closed_trades."""
    if not db_path.exists():
        return [], ["db_missing"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        conn = connect_ro(db_path)
        try:
            raw = conn.execute(
                """
                SELECT ticker,
                       COUNT(*) AS n,
                       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) AS gross_win,
                       SUM(CASE WHEN realized_pnl <= 0 THEN ABS(realized_pnl) ELSE 0 END) AS gross_loss,
                       SUM(realized_pnl) AS total_pnl
                FROM production_closed_trades
                GROUP BY ticker
                ORDER BY ticker
                """
            ).fetchall()
        finally:
            conn.close()
        for r in raw:
            n = int(r["n"] or 0)
            wins = int(r["wins"] or 0)
            gross_win = float(r["gross_win"] or 0.0)
            gross_loss = float(r["gross_loss"] or 0.0)
            pf = min(gross_win / gross_loss, 99.0) if gross_loss > 1e-9 else (99.0 if gross_win > 0.0 else 0.0)
            rows.append(
                {
                    "ticker": str(r["ticker"] or "").upper(),
                    "n_trades": n,
                    "win_rate": (wins / n) if n else 0.0,
                    "profit_factor": pf,
                    "total_pnl": round(float(r["total_pnl"] or 0.0), 2),
                }
            )
    except Exception as exc:
        log.warning("Per-ticker DB query failed: %s", exc)
        errors.append("per_ticker_query_failed")
    return rows, errors


def classify_ticker(row: dict[str, Any], lab_only_set: set[str]) -> str:
    status, _ = classify_ticker_details(row, lab_only_set)
    return status


def classify_ticker_details(
    row: dict[str, Any],
    lab_only_set: set[str],
) -> tuple[str, list[str]]:
    ticker = str(row["ticker"]).upper()
    n = int(row["n_trades"] or 0)
    wr = float(row["win_rate"] or 0.0)
    pf = float(row["profit_factor"] or 0.0)
    reasons: list[str] = []

    if ticker in lab_only_set:
        reasons.append("explicit_lab_only_override")
        return "LAB_ONLY", reasons

    if n >= HEALTHY_MIN_TRADES and wr >= HEALTHY_MIN_WIN_RATE and pf >= HEALTHY_MIN_PROFIT_FACTOR:
        reasons.append(
            f"meets_r3_thresholds(n>={HEALTHY_MIN_TRADES}, wr>={HEALTHY_MIN_WIN_RATE:.2f}, pf>={HEALTHY_MIN_PROFIT_FACTOR:.2f})"
        )
        return "HEALTHY", reasons

    weak_reasons: list[str] = []
    if wr < WEAK_MAX_WIN_RATE:
        weak_reasons.append(f"win_rate_below_weak_floor({wr:.2f}<{WEAK_MAX_WIN_RATE:.2f})")
    if pf < WEAK_MAX_PROFIT_FACTOR:
        weak_reasons.append(f"profit_factor_below_break_even({pf:.2f}<{WEAK_MAX_PROFIT_FACTOR:.2f})")
    if weak_reasons and n >= WEAK_MIN_TRADES:
        reasons.extend(weak_reasons)
        reasons.append(f"sufficient_weak_evidence(n>={WEAK_MIN_TRADES})")
        return "WEAK", reasons

    if n < HEALTHY_MIN_TRADES:
        reasons.append(f"insufficient_trade_count({n}<{HEALTHY_MIN_TRADES})")
    if wr < HEALTHY_MIN_WIN_RATE:
        reasons.append(f"below_healthy_win_rate({wr:.2f}<{HEALTHY_MIN_WIN_RATE:.2f})")
    if pf < HEALTHY_MIN_PROFIT_FACTOR:
        reasons.append(f"below_healthy_profit_factor({pf:.2f}<{HEALTHY_MIN_PROFIT_FACTOR:.2f})")
    if not reasons:
        reasons.append("manual_research_only")
    return "LAB_ONLY", reasons


def compute_eligibility(
    db_path: Path = DEFAULT_DB,
    lab_only_tickers: list[str] | None = None,
) -> dict[str, Any]:
    lab_only_set: set[str] = {str(t).upper() for t in (lab_only_tickers or [])}
    rows, errors = _query_per_ticker(db_path)

    ticker_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        status, reasons = classify_ticker_details(row, lab_only_set)
        ticker_map[row["ticker"]] = {
            "status": status,
            "n_trades": row["n_trades"],
            "win_rate": round(float(row["win_rate"] or 0.0), 4),
            "profit_factor": round(float(row["profit_factor"] or 0.0), 4),
            "total_pnl": row["total_pnl"],
            "reasons": reasons,
        }

    summary = {s: 0 for s in ALL_STATUSES}
    for info in ticker_map.values():
        summary[info["status"]] += 1

    healthy_names = [t for t, v in ticker_map.items() if v["status"] == "HEALTHY"]
    weak_names = [t for t, v in ticker_map.items() if v["status"] == "WEAK"]
    lab_names = [t for t, v in ticker_map.items() if v["status"] == "LAB_ONLY"]
    routing_note = (
        f"HEALTHY tickers for manual routing consideration: {', '.join(healthy_names) or 'none'}. "
        f"WEAK tickers should stay constrained in signal_routing_config.yml via manual review: "
        f"{', '.join(weak_names) or 'none'}. "
        f"LAB_ONLY tickers remain research-only until more evidence accumulates: {', '.join(lab_names) or 'none'}. "
        "This output is recommendation-only and NEVER changes thresholds or routing config."
    )

    thresholds = threshold_map()
    warnings: list[str] = []
    if summary["HEALTHY"] == 0 and ticker_map:
        warnings.append("zero_healthy_tickers")
    if errors:
        warnings.append("eligibility_query_error")
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "db_path": str(db_path),
        "n_tickers": len(ticker_map),
        "tickers": ticker_map,
        "summary": summary,
        "routing_note": routing_note,
        "thresholds": thresholds,
        "source_thresholds": thresholds,
        "thresholds_used": thresholds,
        "warnings": warnings,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-ticker eligibility (HEALTHY/WEAK/LAB_ONLY). "
            "Read-only: never modifies routing config or gate thresholds."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to portfolio_maximizer.db")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path (default: logs/ticker_eligibility.json)",
    )
    parser.add_argument("--lab-only", type=str, default="", help="Comma-separated tickers to force into LAB_ONLY")
    parser.add_argument("--json", action="store_true", dest="emit_json", help="Also print result to stdout as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    lab_only: list[str] = [t.strip().upper() for t in args.lab_only.split(",") if t.strip()]
    result = compute_eligibility(db_path=args.db, lab_only_tickers=lab_only)
    append_threshold_hash_change_warning(args.output, result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Ticker eligibility written to {args.output}")
        summary = result["summary"]
        print(
            f"  HEALTHY={summary['HEALTHY']} "
            f"WEAK={summary['WEAK']} "
            f"LAB_ONLY={summary['LAB_ONLY']}"
        )
        for ticker, info in sorted(result["tickers"].items()):
            wr_pct = f"{info['win_rate']:.0%}"
            print(
                f"  {ticker:6s} [{info['status']:<8}] n={info['n_trades']:3d} "
                f"WR={wr_pct} PF={info['profit_factor']:.2f} PnL=${info['total_pnl']:+.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
