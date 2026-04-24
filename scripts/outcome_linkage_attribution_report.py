#!/usr/bin/env python3
"""
Outcome linkage attribution report.

Builds a closed-trade attribution view with forecast linkage via ts_signal_id.
Primary purpose: quantify stop-loss toxicity and "direction-right but negative PnL"
on outcome-linked evidence before changing trading mechanics.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from statistics import NormalDist, median
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional scientific dependency
    from scipy.stats import beta as beta_dist
except Exception:  # pragma: no cover - keep script usable without scipy
    beta_dist = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
# Prefer the production-specific subdirectory when it exists; fall back to root.
# Without this, the report defaults to the combined (research + production) directory,
# which inflates attribution denominators with non-trade research audits.
_production_audit_dir = ROOT / "logs" / "forecast_audits" / "production"
DEFAULT_AUDIT_DIR = _production_audit_dir if _production_audit_dir.exists() else ROOT / "logs" / "forecast_audits"
DEFAULT_OUTPUT = ROOT / "logs" / "automation" / "tp_contingency_latest.json"
_SNR_TERCILE_LOW_SAMPLE_SUPPORT = 5


def _load_audit_index(audit_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if not audit_dir.exists():
        return index
    files = sorted(
        audit_dir.glob("forecast_audit_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        signal_context = payload.get("signal_context")
        if not isinstance(signal_context, dict):
            continue
        ts_signal_id = str(signal_context.get("ts_signal_id") or "").strip()
        if not ts_signal_id or ts_signal_id in index:
            continue
        dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
        index[ts_signal_id] = {
            "audit_file": path.name,
            "entry_ts": signal_context.get("entry_ts"),
            "forecast_horizon": signal_context.get("forecast_horizon"),
            "dataset_end": dataset.get("end") if isinstance(dataset, dict) else None,
            "regime": dataset.get("detected_regime") if isinstance(dataset, dict) else signal_context.get("regime"),
            "snr": signal_context.get("snr"),
            "entry_price": signal_context.get("entry_price"),
            "stop_loss": signal_context.get("stop_loss"),
            "target_price": signal_context.get("target_price"),
            "expected_return": signal_context.get("expected_return"),
            "expected_return_net": signal_context.get("expected_return_net"),
            "effective_horizon": signal_context.get("effective_horizon"),
            "exit_reason": signal_context.get("exit_reason"),
        }
    return index


def _safe_float(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def _realized_direction(entry_price: Optional[float], exit_price: Optional[float]) -> str:
    if entry_price is None or exit_price is None:
        return "UNKNOWN"
    delta = exit_price - entry_price
    if abs(delta) < 1e-9:
        return "FLAT"
    return "UP" if delta > 0 else "DOWN"


def _direction_match(forecast_direction: str, realized_direction: str) -> Optional[bool]:
    fd = str(forecast_direction or "").strip().upper()
    rd = str(realized_direction or "").strip().upper()
    if fd not in {"BUY", "SELL"}:
        return None
    if rd not in {"UP", "DOWN", "FLAT"}:
        return None
    if rd == "FLAT":
        return False
    return (fd == "BUY" and rd == "UP") or (fd == "SELL" and rd == "DOWN")


def _is_ts_trade_signal_id(ts_signal_id: Any) -> bool:
    sid = str(ts_signal_id or "").strip()
    return sid.startswith("ts_")


def _finite_float(raw: Any) -> Optional[float]:
    try:
        if raw in (None, "", [], {}):
            return None
        value = float(raw)
    except Exception:
        return None
    if value != value or abs(value) == float("inf"):
        return None
    return value


def _beta_binomial_posterior(successes: int, failures: int) -> Dict[str, Any]:
    alpha = max(int(successes), 0) + 1
    beta = max(int(failures), 0) + 1
    mean = alpha / (alpha + beta)
    if beta_dist is not None:
        lower = float(beta_dist.ppf(0.025, alpha, beta))
        upper = float(beta_dist.ppf(0.975, alpha, beta))
    else:  # pragma: no cover - only exercised when scipy is unavailable
        variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
        sd = variance ** 0.5
        z = NormalDist().inv_cdf(0.975)
        lower = max(0.0, mean - z * sd)
        upper = min(1.0, mean + z * sd)
    return {
        "successes": int(successes),
        "failures": int(failures),
        "posterior_alpha": alpha,
        "posterior_beta": beta,
        "posterior_mean": mean,
        "posterior_interval_95": [lower, upper],
    }


def _assign_equal_frequency_bins(records: List[Dict[str, Any]], key: str, bins: int = 3) -> Dict[str, Dict[str, Any]]:
    usable = [
        record
        for record in records
        if _finite_float(record.get(key)) is not None
    ]
    usable.sort(key=lambda record: (_finite_float(record.get(key)) or 0.0, str(record.get("ts_signal_id") or "")))
    if not usable:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    n = len(usable)
    for idx, record in enumerate(usable):
        bin_index = min(bins - 1, int(idx * bins / n))
        label = f"{key}_tercile_{bin_index + 1}"
        result.setdefault(label, {"records": []})["records"].append(record)
    return result


def _load_closed_trades(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            c.id AS close_id,
            c.ticker,
            c.trade_date AS close_date,
            c.bar_timestamp AS close_ts,
            c.realized_pnl,
            c.exit_reason,
            c.ts_signal_id,
            c.holding_period_days,
            c.entry_trade_id,
            c.entry_price AS close_entry_price,
            c.exit_price AS close_exit_price,
            c.price AS close_leg_price,
            e.trade_date AS entry_date,
            e.bar_timestamp AS entry_ts,
            e.price AS open_leg_price,
            e.action AS entry_action
        FROM production_closed_trades c
        LEFT JOIN trade_executions e ON c.entry_trade_id = e.id
        ORDER BY c.trade_date DESC, c.id DESC
        """
    )
    cols = [c[0] for c in cur.description]
    rows = []
    for raw in cur.fetchall():
        rows.append(dict(zip(cols, raw)))
    return rows


def build_report(db_path: Path, audit_dir: Path, limit: int) -> Dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    audit_index = _load_audit_index(audit_dir)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        closed_rows = _load_closed_trades(conn)
    finally:
        conn.close()

    records: List[Dict[str, Any]] = []
    close_before_entry_count = 0
    closed_missing_exit_reason_count = 0
    for row in closed_rows:
        ts_signal_id = str(row.get("ts_signal_id") or "").strip()
        audit_meta = audit_index.get(ts_signal_id)
        outcome_linked = bool(audit_meta)

        entry_price = _safe_float(row.get("close_entry_price"))
        if entry_price is None:
            entry_price = _safe_float(row.get("open_leg_price"))
        exit_price = _safe_float(row.get("close_exit_price"))
        if exit_price is None:
            exit_price = _safe_float(row.get("close_leg_price"))

        pnl = _safe_float(row.get("realized_pnl"))
        exit_reason = str(row.get("exit_reason") or "").strip().upper() or None
        holding_period_days = _safe_float(row.get("holding_period_days"))
        holding_period_int = int(holding_period_days) if holding_period_days is not None else None
        entry_ts_raw = row.get("entry_ts") or row.get("entry_date")
        close_ts_raw = row.get("close_ts") or row.get("close_date")
        entry_ts = _parse_utc_datetime(entry_ts_raw)
        close_ts = _parse_utc_datetime(close_ts_raw)
        integrity_reasons: List[str] = []
        if entry_ts is not None and close_ts is not None and close_ts < entry_ts:
            integrity_reasons.append("CAUSALITY_VIOLATION")
            close_before_entry_count += 1
        if str(exit_reason or "").strip() == "":
            integrity_reasons.append("MISSING_EXIT_REASON")
            closed_missing_exit_reason_count += 1

        forecast_direction = str(row.get("entry_action") or "").strip().upper() or "UNKNOWN"
        realized_direction = _realized_direction(entry_price, exit_price)
        direction_match = _direction_match(forecast_direction, realized_direction)
        correct_direction_negative_pnl = bool(direction_match is True and (pnl or 0.0) < 0.0)

        snr = _safe_float(audit_meta.get("snr")) if audit_meta else None
        target_price = _safe_float(audit_meta.get("target_price")) if audit_meta else None
        stop_loss = _safe_float(audit_meta.get("stop_loss")) if audit_meta else None
        expected_return = _safe_float(audit_meta.get("expected_return_net")) if audit_meta else None
        if expected_return is None and audit_meta:
            expected_return = _safe_float(audit_meta.get("expected_return"))
        effective_horizon = (
            int(audit_meta.get("effective_horizon"))
            if audit_meta and audit_meta.get("effective_horizon") is not None
            else _safe_float(row.get("effective_horizon"))
        )
        regime = None
        if audit_meta:
            regime = str(audit_meta.get("regime") or "").strip().upper() or None
        rr_ratio = None
        if entry_price is not None and target_price is not None and stop_loss is not None:
            stop_distance = abs(entry_price - stop_loss)
            target_distance = abs(target_price - entry_price)
            if stop_distance > 1e-9:
                rr_ratio = target_distance / stop_distance

        take_profit_hit = exit_reason == "TAKE_PROFIT"
        target_amplitude_fraction = None
        target_amplitude_hit = None
        if entry_price is not None and target_price is not None and abs(entry_price) > 1e-9:
            target_distance = abs(target_price - entry_price)
            target_amplitude_fraction = target_distance / abs(entry_price)
            if expected_return is not None:
                # Terminal-return proxy: this is intentionally not the full path maximum.
                target_amplitude_hit = abs(expected_return) >= target_amplitude_fraction

        integrity_status = "HIGH" if integrity_reasons else "OK"
        record = {
            "close_id": row.get("close_id"),
            "ts_signal_id": ts_signal_id or None,
            "ticker": row.get("ticker"),
            "regime": regime,
            "entry_ts": entry_ts_raw,
            "close_ts": close_ts_raw,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "holding_period_days": holding_period_int,
            "holding_period_at_exit": holding_period_int,
            "forecast_direction": forecast_direction,
            "realized_direction": realized_direction,
            "direction_match": direction_match,
            "correct_direction_negative_pnl": correct_direction_negative_pnl,
            "outcome_linked": outcome_linked,
            "audit_file": audit_meta.get("audit_file") if audit_meta else None,
            "forecast_horizon": audit_meta.get("forecast_horizon") if audit_meta else None,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "expected_return": expected_return,
            "effective_horizon": effective_horizon,
            "rr_ratio": rr_ratio,
            "snr": snr,
            "take_profit_hit": take_profit_hit,
            "target_amplitude_fraction": target_amplitude_fraction,
            "target_amplitude_hit": target_amplitude_hit,
            "excursion_min_pct": None,
            "excursion_max_pct": None,
            "integrity_status": integrity_status,
            "integrity_blocking": bool(integrity_reasons),
            "integrity_reasons": integrity_reasons,
            "counts_toward_readiness_denominator": not bool(integrity_reasons),
            "counts_toward_linkage_denominator": bool(outcome_linked and not integrity_reasons),
        }
        records.append(record)

    linked_records = [r for r in records if r["outcome_linked"]]
    analysis_records = [r for r in linked_records if not r.get("integrity_blocking")]
    tp_records = [r for r in analysis_records if r.get("take_profit_hit")]
    stop_loss_linked = [
        r for r in analysis_records if str(r.get("exit_reason") or "").lower().startswith("stop")
    ]
    fast_tp_median = None
    tp_holding_values = [
        int(r["holding_period_at_exit"])
        for r in tp_records
        if r.get("holding_period_at_exit") is not None
    ]
    if tp_holding_values:
        fast_tp_median = median(tp_holding_values)
    fast_take_profit_median_reliable = bool(len(tp_holding_values) >= 10)
    for record in analysis_records:
        if fast_tp_median is None or record.get("holding_period_at_exit") is None:
            record["fast_take_profit_hit"] = False
        else:
            record["fast_take_profit_hit"] = bool(
                record.get("take_profit_hit")
                and int(record["holding_period_at_exit"]) <= int(fast_tp_median)
            )

    snr_bins = _assign_equal_frequency_bins(analysis_records, "snr", bins=3)
    snr_terciles: List[Dict[str, Any]] = []
    for bin_label, payload in sorted(snr_bins.items()):
        bin_records = payload.get("records") or []
        bin_support = len(bin_records)
        successes = sum(1 for r in bin_records if bool(r.get("take_profit_hit")))
        failures = len(bin_records) - successes
        fast_successes = sum(1 for r in bin_records if bool(r.get("fast_take_profit_hit")))
        fast_failures = len(bin_records) - fast_successes
        beta_tp = _beta_binomial_posterior(successes, failures)
        beta_fast = _beta_binomial_posterior(fast_successes, fast_failures)
        snr_values = [float(r["snr"]) for r in bin_records if r.get("snr") is not None]
        snr_terciles.append(
            {
                "bin": bin_label,
                "n": len(bin_records),
                "snr_min": min(snr_values) if snr_values else None,
                "snr_max": max(snr_values) if snr_values else None,
                "reliability": "low_sample" if bin_support < _SNR_TERCILE_LOW_SAMPLE_SUPPORT else "supported",
                "reliability_support_threshold": _SNR_TERCILE_LOW_SAMPLE_SUPPORT,
                "take_profit": {
                    "successes": successes,
                    "failures": failures,
                    "posterior_mean": beta_tp["posterior_mean"],
                    "posterior_interval_95": beta_tp["posterior_interval_95"],
                },
                "fast_take_profit": {
                    "successes": fast_successes,
                    "failures": fast_failures,
                    "posterior_mean": beta_fast["posterior_mean"],
                    "posterior_interval_95": beta_fast["posterior_interval_95"],
                },
            }
        )

    snr_values_linked = sorted(float(r["snr"]) for r in analysis_records if r.get("snr") is not None)
    snr_median = median(snr_values_linked) if snr_values_linked else None
    rr_high_threshold = 2.0
    tp_high_count = sum(1 for r in tp_records if r.get("rr_ratio") is not None and float(r["rr_ratio"]) >= rr_high_threshold)
    tp_low_count = len(tp_records) - tp_high_count
    multiway_table = None
    multiway_table_status = "HIDDEN_UNTIL_SUPPORT"
    multiway_table_tp_needed = max(0, 30 - len(tp_records))
    take_profit_rate = (len(tp_records) / len(analysis_records)) if analysis_records else 0.0
    multiway_table_estimated_trading_days_at_current_rate = None
    if take_profit_rate > 0:
        multiway_table_estimated_trading_days_at_current_rate = round(
            multiway_table_tp_needed / take_profit_rate,
            2,
        )
    if len(tp_records) >= 30:
        multiway_table = {}
        multiway_table_status = "VISIBLE"
        for record in analysis_records:
            if record.get("snr") is None or record.get("rr_ratio") is None:
                continue
            snr_side = "high" if snr_median is not None and float(record["snr"]) >= float(snr_median) else "low"
            rr_side = "high" if float(record["rr_ratio"]) >= rr_high_threshold else "low"
            regime = str(record.get("regime") or "UNKNOWN").upper()
            key = (snr_side, rr_side, regime)
            cell = multiway_table.setdefault(key, {"n": 0, "take_profit": 0, "fast_take_profit": 0})
            cell["n"] += 1
            if record.get("take_profit_hit"):
                cell["take_profit"] += 1
            if record.get("fast_take_profit_hit"):
                cell["fast_take_profit"] += 1
        multiway_table = [
            {
                "snr_side": key[0],
                "rr_side": key[1],
                "regime": key[2],
                **value,
                "take_profit_rate": (value["take_profit"] / value["n"]) if value["n"] else None,
                "fast_take_profit_rate": (value["fast_take_profit"] / value["n"]) if value["n"] else None,
            }
            for key, value in sorted(multiway_table.items())
        ]

    tp_pnl_values = [float(r["pnl"]) for r in tp_records if r.get("pnl") is not None and float(r["pnl"]) > 0]
    stop_pnl_values = [abs(float(r["pnl"])) for r in stop_loss_linked if r.get("pnl") is not None]
    tp_avg_pnl = (sum(tp_pnl_values) / len(tp_pnl_values)) if tp_pnl_values else None
    stop_avg_pnl = (sum(stop_pnl_values) / len(stop_pnl_values)) if stop_pnl_values else None
    threshold_support = {
        "take_profit_count": len(tp_pnl_values),
        "stop_loss_count": len(stop_pnl_values),
    }
    if tp_avg_pnl is not None and stop_avg_pnl is not None and len(tp_pnl_values) >= 5 and len(stop_pnl_values) >= 5:
        take_profit_filter_threshold = stop_avg_pnl / (tp_avg_pnl + stop_avg_pnl) if (tp_avg_pnl + stop_avg_pnl) > 0 else 0.15
        take_profit_filter_threshold_source = "observed"
    else:
        take_profit_filter_threshold = 0.15
        take_profit_filter_threshold_source = "fallback_0.15"

    linked_tp_count = len(tp_records)
    fast_tp_count = sum(1 for r in analysis_records if bool(r.get("fast_take_profit_hit")))
    total_ts_trades = len([r for r in records if _is_ts_trade_signal_id(r.get("ts_signal_id"))])
    linked_ts_trades = len([r for r in linked_records if _is_ts_trade_signal_id(r.get("ts_signal_id"))])
    ts_trade_coverage = (linked_ts_trades / total_ts_trades) if total_ts_trades else 0.0
    summary = {
        "db_path": str(db_path),
        "audit_dir": str(audit_dir),
        "target_amplitude_hit_definition": "terminal_return_proxy",
        "total_closed_trades": len(records),
        "linked_closed_trades": len(linked_records),
        "linked_trade_ratio": (len(linked_records) / len(records)) if records else 0.0,
        "analysis_closed_trades": len(analysis_records),
        "analysis_trade_ratio": (len(analysis_records) / len(records)) if records else 0.0,
        "total_ts_trades": total_ts_trades,
        "linked_ts_trades": linked_ts_trades,
        "linked_ts_trade_ratio": ts_trade_coverage,
        "ts_trade_coverage": ts_trade_coverage,
        "take_profit_count": linked_tp_count,
        "take_profit_rate": take_profit_rate,
        "fast_take_profit_count": fast_tp_count,
        "fast_take_profit_rate": (fast_tp_count / len(analysis_records)) if analysis_records else 0.0,
        "median_tp_holding": fast_tp_median,
        "fast_take_profit_median_reliable": fast_take_profit_median_reliable,
        "fast_take_profit_median_support": len(tp_holding_values),
        "tp_avg_pnl": tp_avg_pnl,
        "stop_avg_pnl": stop_avg_pnl,
        "take_profit_filter_threshold": take_profit_filter_threshold,
        "take_profit_filter_threshold_source": take_profit_filter_threshold_source,
        "take_profit_filter_threshold_support": threshold_support,
        "snr_tercile_support_threshold": _SNR_TERCILE_LOW_SAMPLE_SUPPORT,
        "snr_terciles": snr_terciles,
        "multiway_table_status": multiway_table_status,
        "multiway_table_tp_needed": multiway_table_tp_needed,
        "multiway_table_estimated_trading_days_at_current_rate": multiway_table_estimated_trading_days_at_current_rate,
        "multiway_table": multiway_table,
        "all_stop_loss_count": len(stop_loss_linked),
        "all_stop_loss_rate": (len(stop_loss_linked) / len(records)) if records else 0.0,
        "close_before_entry_count": close_before_entry_count,
        "closed_missing_exit_reason_count": closed_missing_exit_reason_count,
        "high_integrity_violation_count": close_before_entry_count + closed_missing_exit_reason_count,
        "readiness_denominator_exclusion_count": sum(
            1 for r in records if not bool(r.get("counts_toward_readiness_denominator"))
        ),
    }

    return {
        "summary": summary,
        "records": records[: max(int(limit), 0)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a closed-trade attribution report for outcome-linked evidence."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path.")
    parser.add_argument(
        "--audit-dir",
        default=str(DEFAULT_AUDIT_DIR),
        help="Directory containing forecast_audit_*.json files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of records in output payload (default: 50).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Optional JSON output file path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    payload = build_report(
        db_path=Path(args.db),
        audit_dir=Path(args.audit_dir),
        limit=int(args.limit),
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        summary = payload.get("summary", {})
        print("=== Outcome Linkage Attribution ===")
        print(f"DB: {summary.get('db_path')}")
        print(f"Audit dir: {summary.get('audit_dir')}")
        print(
            "Closed trades: "
            f"{summary.get('total_closed_trades', 0)} | "
            f"Linked: {summary.get('linked_closed_trades', 0)} "
            f"({float(summary.get('linked_trade_ratio') or 0.0):.1%})"
        )
        print(
            "TS trades: "
            f"{summary.get('total_ts_trades', 0)} "
            f"({float(summary.get('ts_trade_coverage') or 0.0):.1%} coverage) | "
            f"Linked TS: {summary.get('linked_ts_trades', 0)} "
            f"({float(summary.get('linked_ts_trade_ratio') or 0.0):.1%})"
        )
        print(
            "Take-profit: "
            f"{summary.get('take_profit_count', 0)} "
            f"({float(summary.get('take_profit_rate') or 0.0):.1%}) | "
            f"Fast TP: {summary.get('fast_take_profit_count', 0)} "
            f"({float(summary.get('fast_take_profit_rate') or 0.0):.1%})"
        )
        print(
            "TP threshold: "
            f"{float(summary.get('take_profit_filter_threshold') or 0.0):.3f} "
            f"({summary.get('take_profit_filter_threshold_source')})"
        )
        print(
            "Target amplitude: "
            f"{summary.get('target_amplitude_hit_definition')}"
        )
        print(
            "Median TP hold: "
            f"{summary.get('median_tp_holding')}"
        )
        print(
            "Fast TP median: "
            f"reliable={summary.get('fast_take_profit_median_reliable')} "
            f"support={summary.get('fast_take_profit_median_support')}"
        )
        print(
            "Multi-way table: "
            f"{summary.get('multiway_table_status')} "
            f"tp_needed={summary.get('multiway_table_tp_needed')} "
            f"est_days={summary.get('multiway_table_estimated_trading_days_at_current_rate')}"
        )
        print(
            "All stop-loss rate: "
            f"{float(summary.get('all_stop_loss_rate') or 0.0):.1%} "
            f"({summary.get('all_stop_loss_count', 0)})"
        )
        print(
            "All correct-direction-negative rate: "
            f"{float(summary.get('all_correct_direction_negative_rate') or 0.0):.1%} "
            f"({summary.get('all_correct_direction_negative_count', 0)})"
        )
        print(
            "Linked stop-loss rate: "
            f"{float(summary.get('linked_stop_loss_rate') or 0.0):.1%} "
            f"({summary.get('linked_stop_loss_count', 0)})"
        )
        print(
            "Linked correct-direction-negative rate: "
            f"{float(summary.get('linked_correct_direction_negative_rate') or 0.0):.1%} "
            f"({summary.get('linked_correct_direction_negative_count', 0)})"
        )
        print(
            "Integrity high: "
            f"{summary.get('high_integrity_violation_count', 0)} "
            f"(close_before_entry={summary.get('close_before_entry_count', 0)}, "
            f"missing_exit_reason={summary.get('closed_missing_exit_reason_count', 0)})"
        )
        print(f"Rows emitted: {len(payload.get('records', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
