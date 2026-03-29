"""
Read-only context quality analytics (regime + confidence bins).

This script is schema-aware. Optional columns may be absent on older DBs; the
script must continue producing partial data instead of collapsing to an empty
result due to one missing optional column.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality_pipeline_common import (
    append_threshold_hash_change_warning,
    coalesce_expr,
    connect_ro,
    first_existing_columns,
    has_production_closed_trades_view,
    production_closed_trades_sql,
    sqlite_master_names,
    table_columns,
)
from scripts.robustness_thresholds import threshold_map

log = logging.getLogger(__name__)

DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_OUTPUT = ROOT / "logs" / "context_quality_latest.json"

CONF_BIN_EDGES: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
CONF_BINS: list[tuple[float, float, str]] = [
    (0.50, 0.60, "0.50-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 0.80, "0.70-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 1.01, "0.90-1.00"),
]

_UNKNOWN_REGIME = "UNKNOWN"


def _safe_pf(gross_win: float, gross_loss: float) -> float:
    if gross_loss < 1e-9:
        return 99.0 if gross_win > 0.0 else 0.0
    return min(gross_win / gross_loss, 99.0)


def _conf_bin_label(conf: Optional[float]) -> Optional[str]:
    if conf is None or not math.isfinite(conf):
        return None
    if conf < 0.0 or conf > 1.0:
        return None
    for lo, hi, label in CONF_BINS:
        if lo <= conf < hi:
            return label
    return None


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0, "total_pnl": 0.0}
    wins = sum(1 for row in rows if row["realized_pnl"] > 0)
    gross_win = sum(row["realized_pnl"] for row in rows if row["realized_pnl"] > 0)
    gross_loss = sum(abs(row["realized_pnl"]) for row in rows if row["realized_pnl"] <= 0)
    total_pnl = sum(row["realized_pnl"] for row in rows)
    return {
        "n": n,
        "win_rate": round(wins / n, 4),
        "profit_factor": round(_safe_pf(gross_win, gross_loss), 4),
        "total_pnl": round(total_pnl, 2),
    }


def _load_context_rows(db_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if not db_path.exists():
        return [], {"db_missing": True}, ["db_missing"]

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    schema_used: dict[str, Any] = {
        "db_missing": False,
        "used_production_closed_trades_view": False,
        "confidence_columns": [],
        "regime_source": _UNKNOWN_REGIME,
        "partial_data": False,
    }

    try:
        conn = connect_ro(db_path)
    except Exception as exc:
        log.warning("Could not open DB %s: %s", db_path, exc)
        return [], {"db_error": str(exc)}, ["db_open_failed"]

    try:
        te_cols = table_columns(conn, "trade_executions")
        tsf_exists = "time_series_forecasts" in sqlite_master_names(conn, "table")
        tsf_cols = table_columns(conn, "time_series_forecasts") if tsf_exists else set()

        conf_candidates = first_existing_columns(
            te_cols,
            ("base_confidence", "confidence_calibrated", "effective_confidence"),
        )
        schema_used["confidence_columns"] = conf_candidates
        if len(conf_candidates) < 3:
            schema_used["partial_data"] = True
            warnings.append("missing_optional_confidence_columns")
        conf_expr = coalesce_expr("te", conf_candidates)

        has_regime_col = "detected_regime" in tsf_cols
        has_ts_signal_id = "ts_signal_id" in te_cols
        has_tsf_ts_signal_id = "ts_signal_id" in tsf_cols
        if tsf_exists and has_regime_col and has_ts_signal_id and has_tsf_ts_signal_id:
            regime_expr = "COALESCE(tsf.detected_regime, ?)"
            join_clause = (
                "LEFT JOIN time_series_forecasts tsf "
                "ON te.ts_signal_id = tsf.ts_signal_id "
                "AND tsf.ts_signal_id IS NOT NULL"
            )
            schema_used["regime_source"] = "time_series_forecasts.detected_regime"
        else:
            regime_expr = "?"
            join_clause = ""
            schema_used["partial_data"] = True
            if not tsf_exists:
                warnings.append("time_series_forecasts_missing")
            elif not has_regime_col:
                warnings.append("detected_regime_missing")
            elif not has_ts_signal_id:
                warnings.append("ts_signal_id_missing")
            elif not has_tsf_ts_signal_id:
                warnings.append("forecast_ts_signal_id_missing")

        if has_production_closed_trades_view(conn):
            schema_used["used_production_closed_trades_view"] = True
            query = f"""
                SELECT
                    p.ticker AS ticker,
                    p.realized_pnl AS realized_pnl,
                    {regime_expr} AS detected_regime,
                    {conf_expr} AS confidence
                FROM production_closed_trades p
                LEFT JOIN trade_executions te ON p.id = te.id
                {join_clause}
            """
            raw = conn.execute(query, (_UNKNOWN_REGIME,)).fetchall()
        else:
            schema_used["partial_data"] = True
            warnings.append("production_closed_trades_view_missing")
            query = f"""
                SELECT
                    te.ticker AS ticker,
                    te.realized_pnl AS realized_pnl,
                    {regime_expr} AS detected_regime,
                    {conf_expr} AS confidence
                {production_closed_trades_sql("te", te_cols, join_clause)}
            """
            raw = conn.execute(query, (_UNKNOWN_REGIME,)).fetchall()

        for r in raw:
            conf_val = r["confidence"]
            conf_num = None
            if conf_val is not None:
                try:
                    parsed = float(conf_val)
                    conf_num = parsed if math.isfinite(parsed) else None
                except Exception:
                    conf_num = None
            rows.append(
                {
                    "ticker": str(r["ticker"] or "").upper(),
                    "realized_pnl": float(r["realized_pnl"] or 0.0),
                    "detected_regime": str(r["detected_regime"] or _UNKNOWN_REGIME),
                    "confidence": conf_num,
                }
            )
    except Exception as exc:
        log.warning("Context rows load failed: %s", exc)
        return [], {"db_error": str(exc), "partial_data": True}, ["query_failed"]
    finally:
        conn.close()

    return rows, schema_used, warnings


def compute_context_quality(db_path: Path = DEFAULT_DB) -> dict[str, Any]:
    rows, schema_used, warnings = _load_context_rows(db_path)
    n_total = len(rows)

    regime_groups: dict[str, list[dict[str, Any]]] = {}
    bin_groups: dict[str, list[dict[str, Any]]] = {}
    ticker_regime: dict[str, dict[str, list[dict[str, Any]]]] = {}
    n_no_conf = 0
    n_out_of_range = 0

    for row in rows:
        regime = row["detected_regime"] or _UNKNOWN_REGIME
        regime_groups.setdefault(regime, []).append(row)
        ticker_regime.setdefault(row["ticker"], {}).setdefault(regime, []).append(row)
        conf_val = row["confidence"]
        if conf_val is not None and math.isfinite(conf_val) and not (0.0 <= conf_val <= 1.0):
            n_out_of_range += 1
        label = _conf_bin_label(row["confidence"])
        if label is None:
            n_no_conf += 1
        else:
            bin_groups.setdefault(label, []).append(row)

    regime_quality = {regime: _summarise(group) for regime, group in sorted(regime_groups.items())}
    conf_bin_quality = {
        label: _summarise(bin_groups[label])
        for _, _, label in CONF_BINS
        if label in bin_groups
    }
    ticker_regime_quality = {
        ticker: {regime: _summarise(group) for regime, group in sorted(regimes.items())}
        for ticker, regimes in sorted(ticker_regime.items())
    }

    worst_regime = min(
        (r for r in regime_quality if r != _UNKNOWN_REGIME and regime_quality[r]["n"] >= 3),
        key=lambda r: regime_quality[r]["win_rate"],
        default=None,
    )
    best_regime = max(
        (r for r in regime_quality if r != _UNKNOWN_REGIME and regime_quality[r]["n"] >= 3),
        key=lambda r: regime_quality[r]["win_rate"],
        default=None,
    )
    worst_bin = min(
        (b for b in conf_bin_quality if conf_bin_quality[b]["n"] >= 3),
        key=lambda b: conf_bin_quality[b]["win_rate"],
        default=None,
    )

    advisory: list[str] = []
    if worst_regime:
        advisory.append(
            f"Worst regime WR: {worst_regime} ({regime_quality[worst_regime]['win_rate']:.0%})"
        )
    if best_regime and best_regime != worst_regime:
        advisory.append(
            f"Best regime WR: {best_regime} ({regime_quality[best_regime]['win_rate']:.0%})"
        )
    if worst_bin:
        advisory.append(
            f"Worst confidence bin WR: {worst_bin} ({conf_bin_quality[worst_bin]['win_rate']:.0%})"
        )
    if n_no_conf:
        advisory.append(f"{n_no_conf}/{n_total} trades have no usable confidence value")
    if n_out_of_range:
        advisory.append(f"{n_out_of_range}/{n_total} trades have out-of-range confidence values")
        warnings.append("confidence_out_of_range")

    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "db_path": str(db_path),
        "n_total_trades": n_total,
        "n_trades_no_confidence": n_no_conf,
        "n_confidence_out_of_range": n_out_of_range,
        "confidence_bin_edges": list(CONF_BIN_EDGES),
        "regime_quality": regime_quality,
        "confidence_bin_quality": conf_bin_quality,
        "ticker_regime_quality": ticker_regime_quality,
        "schema_used": schema_used,
        "thresholds": threshold_map(),
        "partial_data": bool(schema_used.get("partial_data")),
        "warnings": warnings,
        "advisory": advisory,
        "note": (
            "Informational only. Tune confidence_threshold and regime_filters manually "
            "in signal_routing_config.yml. This script never modifies config or gate thresholds."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-regime and per-confidence-bin win-rate/PF statistics. "
            "Read-only: never modifies routing config or gate thresholds."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to portfolio_maximizer.db")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path (default: logs/context_quality_latest.json)",
    )
    parser.add_argument("--json", action="store_true", dest="emit_json", help="Also print result to stdout as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    result = compute_context_quality(db_path=args.db)
    append_threshold_hash_change_warning(args.output, result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Context quality written to {args.output}")
        print(f"  Trades analysed: {result['n_total_trades']}")
        if result["warnings"]:
            print(f"  Warnings: {', '.join(result['warnings'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
