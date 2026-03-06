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
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_AUDIT_DIR = ROOT / "logs" / "forecast_audits"


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

        forecast_direction = str(row.get("entry_action") or "").strip().upper() or "UNKNOWN"
        realized_direction = _realized_direction(entry_price, exit_price)
        direction_match = _direction_match(forecast_direction, realized_direction)
        pnl = _safe_float(row.get("realized_pnl"))
        correct_direction_negative_pnl = bool(direction_match is True and (pnl or 0.0) < 0.0)
        entry_ts_raw = row.get("entry_ts") or row.get("entry_date")
        close_ts_raw = row.get("close_ts") or row.get("close_date")
        entry_ts = _parse_utc_datetime(entry_ts_raw)
        close_ts = _parse_utc_datetime(close_ts_raw)
        integrity_reasons: List[str] = []
        if entry_ts is not None and close_ts is not None and close_ts < entry_ts:
            integrity_reasons.append("CAUSALITY_VIOLATION")
            close_before_entry_count += 1
        exit_reason = row.get("exit_reason")
        if str(exit_reason or "").strip() == "":
            integrity_reasons.append("MISSING_EXIT_REASON")
            closed_missing_exit_reason_count += 1
        integrity_status = "HIGH" if integrity_reasons else "OK"

        record = {
            "close_id": row.get("close_id"),
            "ts_signal_id": ts_signal_id or None,
            "ticker": row.get("ticker"),
            "entry_ts": entry_ts_raw,
            "close_ts": close_ts_raw,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "holding_period_days": row.get("holding_period_days"),
            "forecast_direction": forecast_direction,
            "realized_direction": realized_direction,
            "direction_match": direction_match,
            "correct_direction_negative_pnl": correct_direction_negative_pnl,
            "outcome_linked": outcome_linked,
            "audit_file": audit_meta.get("audit_file") if audit_meta else None,
            "forecast_horizon": audit_meta.get("forecast_horizon") if audit_meta else None,
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
    ts_records = [r for r in records if _is_ts_trade_signal_id(r.get("ts_signal_id"))]
    linked_ts_records = [r for r in ts_records if r["outcome_linked"]]
    stop_loss_all = [r for r in records if "stop" in str(r.get("exit_reason") or "").lower()]
    right_dir_negative_all = [
        r for r in records if bool(r.get("correct_direction_negative_pnl"))
    ]
    stop_loss_linked = [
        r for r in linked_records if "stop" in str(r.get("exit_reason") or "").lower()
    ]
    right_dir_negative_linked = [
        r for r in linked_records if bool(r.get("correct_direction_negative_pnl"))
    ]

    summary = {
        "db_path": str(db_path),
        "audit_dir": str(audit_dir),
        "total_closed_trades": len(records),
        "linked_closed_trades": len(linked_records),
        "linked_trade_ratio": (len(linked_records) / len(records)) if records else 0.0,
        "total_ts_trades": len(ts_records),
        "linked_ts_trades": len(linked_ts_records),
        "linked_ts_trade_ratio": (len(linked_ts_records) / len(ts_records)) if ts_records else 0.0,
        "ts_trade_coverage": (len(ts_records) / len(records)) if records else 0.0,
        "all_stop_loss_count": len(stop_loss_all),
        "all_stop_loss_rate": (len(stop_loss_all) / len(records)) if records else 0.0,
        "all_correct_direction_negative_count": len(right_dir_negative_all),
        "all_correct_direction_negative_rate": (
            len(right_dir_negative_all) / len(records) if records else 0.0
        ),
        "linked_stop_loss_count": len(stop_loss_linked),
        "linked_stop_loss_rate": (
            len(stop_loss_linked) / len(linked_records) if linked_records else 0.0
        ),
        "linked_correct_direction_negative_count": len(right_dir_negative_linked),
        "linked_correct_direction_negative_rate": (
            len(right_dir_negative_linked) / len(linked_records) if linked_records else 0.0
        ),
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
        default="",
        help="Optional JSON output file path.",
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
