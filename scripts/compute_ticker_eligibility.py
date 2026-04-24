"""
Read-only per-ticker eligibility classification.

Public status enum is intentionally strict:
  - HEALTHY
  - WEAK
  - LAB_ONLY

This script never modifies routing config or any gate threshold. It only
computes rolling recommendations from existing `production_closed_trades`
outcomes so weak names can be re-admitted when fresh evidence recovers.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.domain_objective import (
    MIN_OMEGA_VS_HURDLE,
    MIN_TAKE_PROFIT_FREQUENCY,
    SYSTEM_OBJECTIVE,
    TARGET_AMPLITUDE_MULTIPLIER,
    TAKE_PROFIT_FILTER_THRESHOLD_FALLBACK,
)
from etl.portfolio_math import portfolio_metrics_ngn
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
DEFAULT_LOOKBACK_DAYS = 365

HEALTHY_MIN_WIN_RATE = R3_MIN_WIN_RATE
HEALTHY_MIN_PROFIT_FACTOR = R3_MIN_PROFIT_FACTOR
HEALTHY_MIN_TRADES = R3_MIN_TRADES
ALL_STATUSES = ("HEALTHY", "WEAK", "LAB_ONLY")


def _parse_as_of_date(as_of_date: str | None) -> datetime.date:
    if not as_of_date:
        return datetime.date.today()
    return datetime.date.fromisoformat(as_of_date)


def _query_per_ticker(
    db_path: Path,
    *,
    lookback_days: int | None,
    as_of_date: str | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Return per-ticker stats from production_closed_trades over a rolling window."""
    if not db_path.exists():
        return [], ["db_missing"], {}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    end_date = _parse_as_of_date(as_of_date)
    start_date = None
    if lookback_days is not None:
        start_date = end_date - datetime.timedelta(days=max(int(lookback_days), 1) - 1)
    try:
        conn = connect_ro(db_path)
        try:
            params: list[Any] = []
            where_clauses: list[str] = []
            if start_date is not None:
                where_clauses.append("DATE(trade_date) >= DATE(?)")
                params.append(start_date.isoformat())
                where_clauses.append("DATE(trade_date) <= DATE(?)")
                params.append(end_date.isoformat())
            raw = conn.execute(
                """
                SELECT ticker,
                       trade_date,
                       realized_pnl,
                       entry_price,
                       shares,
                       COALESCE(close_size, shares, 0) AS close_size,
                       exit_reason,
                       holding_period_days
                FROM production_closed_trades
                {where_clause}
                ORDER BY ticker, id
                """.format(
                    where_clause=("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                ),
                params,
            ).fetchall()
        finally:
            conn.close()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in raw:
            ticker = str(r["ticker"] or "").upper()
            if not ticker:
                continue
            grouped.setdefault(ticker, []).append(r)

        for ticker in sorted(grouped):
            trade_rows = grouped[ticker]
            n = len(trade_rows)
            wins = 0
            gross_win = 0.0
            gross_loss = 0.0
            total_pnl = 0.0
            tp_count = 0
            stop_count = 0
            time_exit_count = 0
            returns: list[float] = []
            for r in trade_rows:
                realized_pnl = float(r["realized_pnl"] or 0.0)
                entry_price = float(r["entry_price"] or 0.0)
                qty = abs(float(r["close_size"] or r["shares"] or 0.0))
                pnl = realized_pnl
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                    gross_win += pnl
                elif pnl <= 0:
                    gross_loss += abs(pnl)
                exit_reason = str(r["exit_reason"] or "").strip().upper()
                if exit_reason == "TAKE_PROFIT":
                    tp_count += 1
                elif exit_reason == "STOP_LOSS":
                    stop_count += 1
                elif exit_reason == "TIME_EXIT":
                    time_exit_count += 1
                capital_at_risk = abs(entry_price) * qty if entry_price and qty else 0.0
                if capital_at_risk > 1e-9:
                    returns.append(pnl / capital_at_risk)
            pf = min(gross_win / gross_loss, 99.0) if gross_loss > 1e-9 else (99.0 if gross_win > 0.0 else 0.0)
            series = pd.Series(returns, dtype="float64") if returns else pd.Series(dtype="float64")
            ngn_metrics = portfolio_metrics_ngn(series) if not series.empty else {}
            omega_ratio = float(ngn_metrics.get("omega_ratio")) if ngn_metrics.get("omega_ratio") is not None else 0.0
            payoff_asymmetry = float(ngn_metrics.get("payoff_asymmetry_effective")) if ngn_metrics.get("payoff_asymmetry_effective") is not None else 0.0
            take_profit_frequency = (tp_count / n) if n else 0.0
            rows.append(
                {
                    "ticker": ticker,
                    "n_trades": n,
                    "win_rate": (wins / n) if n else 0.0,
                    "profit_factor": pf,
                    "total_pnl": round(total_pnl, 2),
                    "omega_ratio": omega_ratio,
                    "payoff_asymmetry_effective": payoff_asymmetry,
                    "take_profit_count": tp_count,
                    "take_profit_frequency": take_profit_frequency,
                    "stop_loss_count": stop_count,
                    "time_exit_count": time_exit_count,
                    "ngn_annual_hurdle_pct": ngn_metrics.get("ngn_annual_hurdle_pct"),
                    "beats_ngn_hurdle": bool(ngn_metrics.get("beats_ngn_hurdle")),
                }
            )
    except Exception as exc:
        log.warning("Per-ticker DB query failed: %s", exc)
        errors.append("per_ticker_query_failed")
    window = {
        "lookback_days": int(lookback_days) if lookback_days is not None else None,
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat(),
        "state": "rolling_window" if lookback_days is not None else "lifetime",
        "source_view": "production_closed_trades",
    }
    return rows, errors, window


def classify_ticker(row: dict[str, Any], lab_only_set: set[str], thresholds: dict[str, Any] | None = None) -> str:
    status, _ = classify_ticker_details(row, lab_only_set, thresholds)
    return status


def classify_ticker_details(
    row: dict[str, Any],
    lab_only_set: set[str],
    thresholds: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    thresholds = thresholds or threshold_map()
    ticker = str(row["ticker"]).upper()
    n = int(row["n_trades"] or 0)
    wr = float(row["win_rate"] or 0.0)
    pf = float(row["profit_factor"] or 0.0)
    omega = float(row.get("omega_ratio") or 0.0)
    payoff = float(row.get("payoff_asymmetry_effective") or 0.0)
    tp_freq = float(row.get("take_profit_frequency") or 0.0)
    omega_floor = float(thresholds.get("min_omega_ratio") or MIN_OMEGA_VS_HURDLE)
    payoff_floor = float(thresholds.get("min_payoff_asymmetry") or TARGET_AMPLITUDE_MULTIPLIER)
    tp_floor = float(thresholds.get("min_take_profit_frequency_live") or 0.05)
    reasons: list[str] = []

    if ticker in lab_only_set:
        reasons.append("explicit_lab_only_override")
        return "LAB_ONLY", reasons

    meets_primary = (
        n >= HEALTHY_MIN_TRADES
        and omega >= omega_floor
        and payoff >= payoff_floor
        and tp_freq >= tp_floor
    )
    if meets_primary:
        reasons.append(
            "meets_take_profit_policy("
            f"n>={HEALTHY_MIN_TRADES}, omega>={omega_floor:.2f}, "
            f"payoff>={payoff_floor:.2f}, tp_freq>={tp_floor:.3f})"
        )
        return "HEALTHY", reasons

    weak_reasons: list[str] = []
    if omega < omega_floor:
        weak_reasons.append(f"omega_below_hurdle({omega:.2f}<{omega_floor:.2f})")
    if payoff < payoff_floor:
        weak_reasons.append(f"payoff_below_target({payoff:.2f}<{payoff_floor:.2f})")
    if tp_freq < tp_floor:
        weak_reasons.append(f"take_profit_frequency_below_floor({tp_freq:.3f}<{tp_floor:.3f})")
    if n >= WEAK_MIN_TRADES and (omega >= omega_floor or payoff >= payoff_floor or tp_freq >= tp_floor):
        reasons.extend(weak_reasons)
        reasons.append(f"sufficient_weak_evidence(n>={WEAK_MIN_TRADES})")
        return "WEAK", reasons

    if n < HEALTHY_MIN_TRADES:
        reasons.append(f"insufficient_trade_count({n}<{HEALTHY_MIN_TRADES})")
    if wr < HEALTHY_MIN_WIN_RATE:
        reasons.append(f"diagnostic_win_rate_below_r3_floor({wr:.2f}<{HEALTHY_MIN_WIN_RATE:.2f})")
    if pf < HEALTHY_MIN_PROFIT_FACTOR:
        reasons.append(f"diagnostic_profit_factor_below_r3_floor({pf:.2f}<{HEALTHY_MIN_PROFIT_FACTOR:.2f})")
    if not weak_reasons:
        reasons.append("primary_metrics_below_floor")
    if not reasons:
        reasons.append("manual_research_only")
    return "LAB_ONLY", reasons


def compute_eligibility(
    db_path: Path = DEFAULT_DB,
    lab_only_tickers: list[str] | None = None,
    *,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    lab_only_set: set[str] = {str(t).upper() for t in (lab_only_tickers or [])}
    thresholds = threshold_map()
    rows, errors, window = _query_per_ticker(
        db_path,
        lookback_days=lookback_days,
        as_of_date=as_of_date,
    )

    ticker_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        status, reasons = classify_ticker_details(row, lab_only_set, thresholds)
        ticker_map[row["ticker"]] = {
            "status": status,
            "n_trades": row["n_trades"],
            "win_rate": round(float(row["win_rate"] or 0.0), 4),
            "profit_factor": round(float(row["profit_factor"] or 0.0), 4),
            "total_pnl": row["total_pnl"],
            "omega_ratio": round(float(row.get("omega_ratio") or 0.0), 4),
            "payoff_asymmetry_effective": round(float(row.get("payoff_asymmetry_effective") or 0.0), 4),
            "take_profit_count": int(row.get("take_profit_count") or 0),
            "take_profit_frequency": round(float(row.get("take_profit_frequency") or 0.0), 4),
            "stop_loss_count": int(row.get("stop_loss_count") or 0),
            "time_exit_count": int(row.get("time_exit_count") or 0),
            "beats_ngn_hurdle": bool(row.get("beats_ngn_hurdle")),
            "reasons": reasons,
        }

    summary = {s: 0 for s in ALL_STATUSES}
    for info in ticker_map.values():
        summary[info["status"]] += 1

    healthy_names = [t for t, v in ticker_map.items() if v["status"] == "HEALTHY"]
    weak_names = [t for t, v in ticker_map.items() if v["status"] == "WEAK"]
    lab_names = [t for t, v in ticker_map.items() if v["status"] == "LAB_ONLY"]
    routing_note = (
        f"Objective={SYSTEM_OBJECTIVE}. "
        f"Window={window.get('start_date') or 'full_history'}..{window.get('end_date') or 'unknown'}. "
        f"HEALTHY tickers for manual routing consideration: {', '.join(healthy_names) or 'none'}. "
        f"WEAK tickers should stay constrained in signal_routing_config.yml via manual review: "
        f"{', '.join(weak_names) or 'none'}. "
        f"LAB_ONLY tickers remain research-only until more evidence accumulates: {', '.join(lab_names) or 'none'}. "
        "This output is recommendation-only and NEVER changes thresholds or routing config."
    )

    warnings: list[str] = []
    if summary["HEALTHY"] == 0 and ticker_map:
        warnings.append("zero_healthy_tickers")
    if errors:
        warnings.append("eligibility_query_error")
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "db_path": str(db_path),
        "window": window,
        "source_view": "production_closed_trades",
        "n_tickers": len(ticker_map),
        "tickers": ticker_map,
        "summary": summary,
        "routing_note": routing_note,
        "system_objective": SYSTEM_OBJECTIVE,
        "policy_reference_take_profit_frequency": MIN_TAKE_PROFIT_FREQUENCY,
        "take_profit_filter_threshold_fallback": TAKE_PROFIT_FILTER_THRESHOLD_FALLBACK,
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
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Rolling lookback window in days (use 365 for weekly maintenance).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Optional ISO date used as the window end date (defaults to today).",
    )
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
    result = compute_eligibility(
        db_path=args.db,
        lab_only_tickers=lab_only,
        lookback_days=args.lookback_days,
        as_of_date=args.as_of_date,
    )
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
