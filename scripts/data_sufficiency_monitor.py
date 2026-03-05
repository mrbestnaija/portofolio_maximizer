"""
Read-only monitor for capital-readiness data sufficiency.

This script reports whether current data volume and quality are sufficient to
support readiness checks. It never mutates thresholds, gate files, or routing.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS_DIR = str(ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from scripts.robustness_thresholds import (
    R3_MIN_PROFIT_FACTOR,
    R3_MIN_TRADES,
    R3_MIN_WIN_RATE,
    threshold_map,
)

log = logging.getLogger(__name__)

TARGET_TRADES = 50
TARGET_COVERAGE_RATIO = 0.20
TARGET_N_USED = 50

DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_AUDIT_DIR = ROOT / "logs" / "forecast_audits"


def _read_layer1(audit_dir: Path) -> dict[str, Any]:
    try:
        from check_model_improvement import run_layer1_forecast_quality

        result = run_layer1_forecast_quality(audit_dir=audit_dir)
        return {"metrics": result.metrics or {}, "error": None}
    except Exception as exc:
        log.warning("Layer 1 read failed: %s", exc)
        return {"metrics": {}, "error": f"layer1_read_failed: {exc}"}


def _read_layer3(db_path: Path) -> dict[str, Any]:
    try:
        from check_model_improvement import run_layer3_trade_quality

        result = run_layer3_trade_quality(db_path)
        return {"metrics": result.metrics or {}, "error": None}
    except Exception as exc:
        log.warning("Layer 3 read failed: %s", exc)
        return {"metrics": {}, "error": f"layer3_read_failed: {exc}"}


def _unpack_layer_payload(payload: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(payload, dict) and ("metrics" in payload or "error" in payload):
        metrics = payload.get("metrics")
        return (metrics if isinstance(metrics, dict) else {}), payload.get("error")
    if isinstance(payload, dict):
        return payload, None
    return {}, "invalid_layer_payload"


def _finite_float(value: Any, *, field: str, errors: list[str], default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        errors.append(f"non_numeric_{field}")
        return default
    if not math.isfinite(parsed):
        errors.append(f"non_finite_{field}")
        return default
    return parsed


def _read_per_ticker(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            raw = conn.execute(
                """
                SELECT ticker,
                       COUNT(*) AS n,
                       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(realized_pnl) AS total_pnl
                FROM production_closed_trades
                GROUP BY ticker
                ORDER BY total_pnl ASC
                """
            ).fetchall()
        finally:
            conn.close()
        for row in raw:
            n = int(row["n"] or 0)
            wins = int(row["wins"] or 0)
            rows.append(
                {
                    "ticker": str(row["ticker"] or "").upper(),
                    "n_trades": n,
                    "win_rate": round(wins / n, 4) if n else 0.0,
                    "total_pnl": round(float(row["total_pnl"] or 0.0), 2),
                    "wins": wins,
                }
            )
    except Exception as exc:
        log.warning("Per-ticker query failed: %s", exc)
    return rows


def run_data_sufficiency(
    db_path: Path = DEFAULT_DB,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> dict[str, Any]:
    recommendations: list[str] = []
    metrics: dict[str, Any] = {}
    layer_errors: list[str] = []

    l3_raw = _read_layer3(db_path)
    l3, l3_error = _unpack_layer_payload(l3_raw)
    n_trades = int(l3.get("n_trades") or 0)
    win_rate = _finite_float(l3.get("win_rate"), field="win_rate", errors=layer_errors, default=0.0)
    profit_factor = _finite_float(
        l3.get("profit_factor"),
        field="profit_factor",
        errors=layer_errors,
        default=0.0,
    )
    metrics.update(
        {
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "target_trades": TARGET_TRADES,
            "thresholds": threshold_map(),
        }
    )

    if n_trades < R3_MIN_TRADES:
        recommendations.append(
            f"TRADE_COUNT: {n_trades} trades < {R3_MIN_TRADES} minimum (R3 hard gate)."
        )
    elif n_trades < TARGET_TRADES:
        recommendations.append(
            f"TRADE_COUNT: {n_trades} trades < {TARGET_TRADES} target for statistical stability."
        )

    if win_rate < R3_MIN_WIN_RATE:
        recommendations.append(
            f"WIN_RATE: {win_rate:.1%} < {R3_MIN_WIN_RATE:.0%} (R3 hard gate)."
        )

    if profit_factor < R3_MIN_PROFIT_FACTOR:
        recommendations.append(
            f"PROFIT_FACTOR: {profit_factor:.2f} < {R3_MIN_PROFIT_FACTOR:.2f} (R3 hard gate)."
        )

    l1_raw = _read_layer1(audit_dir)
    l1, l1_error = _unpack_layer_payload(l1_raw)
    n_used = int(l1.get("n_used_windows") or l1.get("n_used") or 0)
    coverage_ratio = _finite_float(
        l1.get("coverage_ratio"),
        field="coverage_ratio",
        errors=layer_errors,
        default=0.0,
    )
    lift_recent = _finite_float(
        l1.get("lift_fraction_recent"),
        field="lift_fraction_recent",
        errors=layer_errors,
        default=0.0,
    )
    lift_global = _finite_float(
        l1.get("lift_fraction_global"),
        field="lift_fraction_global",
        errors=layer_errors,
        default=0.0,
    )
    lift_ci_low = l1.get("lift_ci_low")
    if lift_ci_low is not None:
        lift_ci_low = _finite_float(lift_ci_low, field="lift_ci_low", errors=layer_errors, default=0.0)
    metrics.update(
        {
            "n_used_audit_windows": n_used,
            "coverage_ratio": round(coverage_ratio, 4),
            "lift_fraction_global": round(lift_global, 4),
            "lift_fraction_recent": round(lift_recent, 4),
            "lift_ci_low": lift_ci_low,
            "target_coverage_ratio": TARGET_COVERAGE_RATIO,
            "target_n_used": TARGET_N_USED,
        }
    )

    if coverage_ratio < TARGET_COVERAGE_RATIO or n_used < TARGET_N_USED:
        recommendations.append(
            f"AUDIT_COVERAGE: {n_used} usable windows, coverage_ratio={coverage_ratio:.1%} "
            f"(target: {TARGET_N_USED}+ windows, {TARGET_COVERAGE_RATIO:.0%} ratio)."
        )

    if lift_ci_low is not None and float(lift_ci_low) <= 0.0:
        recommendations.append(
            f"LIFT_ADVISORY: lift CI low={float(lift_ci_low):.4f} <= 0. "
            "Advisory only; continue accumulating audit windows."
        )

    layer_errors.extend([err for err in (l1_error, l3_error) if err])
    if layer_errors:
        return {
            "status": "DATA_ERROR",
            "sufficient": False,
            "recommendations": layer_errors,
            "warnings": [],
            "metrics": metrics,
            "per_ticker": [],
        }

    per_ticker = _read_per_ticker(db_path)
    weak_tickers = [row for row in per_ticker if row["n_trades"] >= 3 and row["win_rate"] < 0.30]
    if weak_tickers:
        names = ", ".join(
            f"{row['ticker']} (WR={row['win_rate']:.0%}, n={row['n_trades']})" for row in weak_tickers
        )
        recommendations.append(f"WEAK_TICKERS: {names}")

    sufficient = len(recommendations) == 0
    return {
        "status": "SUFFICIENT" if sufficient else "INSUFFICIENT",
        "sufficient": sufficient,
        "recommendations": recommendations,
        "warnings": [],
        "metrics": metrics,
        "per_ticker": per_ticker,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Data Sufficiency Monitor. Read-only: reports data gaps without modifying gates."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not args.db.exists():
        payload = {"error": f"DB not found: {args.db}", "sufficient": False, "status": "DATA_ERROR"}
        if args.emit_json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"[ERROR] DB not found: {args.db}")
        return 2

    result = run_data_sufficiency(db_path=args.db, audit_dir=args.audit_dir)

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
        if result["status"] == "DATA_ERROR":
            return 2
        return 0 if result["sufficient"] else 1

    if result["status"] == "DATA_ERROR":
        label = "[ERROR] DATA_ERROR"
    elif result["sufficient"]:
        label = "[OK] SUFFICIENT"
    else:
        label = "[WARN] INSUFFICIENT"
    print(f"\nData Sufficiency: {label}")
    print(f"  Status: {result['status']}")
    if result["recommendations"]:
        for index, rec in enumerate(result["recommendations"], 1):
            print(f"  {index}. {rec}")
    else:
        print("  All data sufficiency targets met.")
    if result["status"] == "DATA_ERROR":
        return 2
    return 0 if result["sufficient"] else 1


if __name__ == "__main__":
    sys.exit(main())
