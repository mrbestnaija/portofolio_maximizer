#!/usr/bin/env python3
"""Append-only family calibration writer for post-DCR measurement windows.

The writer collects a single window summary into
``logs/automation/family_calibration.jsonl``. Each row is independently
parseable so malformed tails can be skipped without losing the rest of the
history.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_AUDIT_DIR = ROOT / "logs" / "forecast_audits" / "production"
if not DEFAULT_AUDIT_DIR.exists():
    DEFAULT_AUDIT_DIR = ROOT / "logs" / "forecast_audits"
DEFAULT_OUTPUT = ROOT / "logs" / "automation" / "family_calibration.jsonl"

SCHEMA_VERSION = 1
MIN_ANALYSIS_CYCLES = 20
MIN_ANALYSIS_REGIMES = 2
MIN_ANALYSIS_DAYS = 10

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _parse_date(value: Any) -> Optional[datetime.date]:
    ts = _parse_utc(value)
    if ts is None:
        return None
    return ts.date()


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    if pct <= 0:
        return float(ordered[0])
    if pct >= 100:
        return float(ordered[-1])
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _json_load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_audit_files(audit_dir: Path) -> Iterable[Path]:
    if not audit_dir.exists():
        return []
    return sorted(
        (p for p in audit_dir.rglob("forecast_audit_*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _extract_audit_row(path: Path) -> Optional[dict[str, Any]]:
    payload = _json_load(path)
    if not payload:
        return None
    signal_context = payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
    routed = payload.get("routed_signal_snapshot") if isinstance(payload.get("routed_signal_snapshot"), dict) else {}
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}

    ts_signal_id = str(
        signal_context.get("ts_signal_id")
        or routed.get("ts_signal_id")
        or payload.get("ts_signal_id")
        or ""
    ).strip()
    if not ts_signal_id:
        return None

    model_type = str(
        signal_context.get("model_type")
        or routed.get("model_type")
        or payload.get("model_type")
        or ""
    ).strip().upper() or None
    snr = _safe_float(
        signal_context.get("snr")
        or routed.get("snr")
        or payload.get("snr")
        or signal_context.get("decision_context_snr")
        or routed.get("decision_context_snr")
    )
    blocked_by_snr = bool(signal_context.get("snr_gate_blocked")) or "SNR" in str(
        signal_context.get("hold_reason") or routed.get("hold_reason") or signal_context.get("routing_reason") or routed.get("routing_reason") or ""
    ).upper()
    blocked_by_evidence = bool(signal_context.get("execution_policy_blocked")) or any(
        marker in str(
            signal_context.get("hold_reason")
            or routed.get("hold_reason")
            or signal_context.get("routing_reason")
            or routed.get("routing_reason")
            or payload.get("evidence_context")
            or ""
        ).upper()
        for marker in ("EVIDENCE", "VALIDATION", "HYGIENE", "FRESHNESS")
    )
    execution_mode = str(
        signal_context.get("execution_mode")
        or routed.get("execution_mode")
        or payload.get("execution_mode")
        or ""
    ).strip().lower() or None
    ticker = str(
        dataset.get("ticker")
        or signal_context.get("ticker")
        or routed.get("ticker")
        or payload.get("ticker")
        or ""
    ).strip().upper() or None
    regime = str(
        dataset.get("detected_regime")
        or routed.get("detected_regime")
        or signal_context.get("detected_regime")
        or payload.get("detected_regime")
        or ""
    ).strip().upper() or None
    bar_timestamp = signal_context.get("bar_timestamp") or routed.get("bar_timestamp") or dataset.get("end")
    signal_timestamp = signal_context.get("signal_timestamp") or routed.get("signal_timestamp") or payload.get("signal_timestamp")
    forecast_horizon = signal_context.get("forecast_horizon") or routed.get("forecast_horizon") or dataset.get("forecast_horizon")
    try:
        forecast_horizon_int = int(forecast_horizon) if forecast_horizon is not None else None
    except Exception:
        forecast_horizon_int = None

    return {
        "ts_signal_id": ts_signal_id,
        "model_type": model_type,
        "ticker": ticker,
        "regime": regime,
        "snr": snr,
        "blocked_by_snr": blocked_by_snr,
        "blocked_by_evidence": blocked_by_evidence,
        "execution_mode": execution_mode,
        "bar_timestamp": bar_timestamp,
        "signal_timestamp": signal_timestamp,
        "forecast_horizon": forecast_horizon_int,
        "bar_open": _safe_float(signal_context.get("bar_open") or routed.get("bar_open") or payload.get("bar_open")),
        "bar_high": _safe_float(signal_context.get("bar_high") or routed.get("bar_high") or payload.get("bar_high")),
        "bar_low": _safe_float(signal_context.get("bar_low") or routed.get("bar_low") or payload.get("bar_low")),
        "bar_close": _safe_float(signal_context.get("bar_close") or routed.get("bar_close") or payload.get("bar_close")),
        "bar_position_proxy": _safe_float(
            signal_context.get("bar_position_proxy")
            or routed.get("bar_position_proxy")
            or payload.get("bar_position_proxy")
        ),
        "source_file": path.name,
        "raw": payload,
    }


def _load_audit_rows(audit_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_audit_files(audit_dir):
        row = _extract_audit_row(path)
        if row is not None:
            rows.append(row)
    return rows


def _load_closed_trades(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    def _table_columns(table: str) -> set[str]:
        try:
            cur = conn.execute(f"PRAGMA table_info({table})")
        except Exception:
            return set()
        return {str(row[1] or "") for row in cur.fetchall()}

    desired_columns = [
        "ts_signal_id",
        "ticker",
        "realized_pnl",
        "holding_period_days",
        "trade_date",
        "exit_reason",
        "id",
        "execution_mode",
        "bar_timestamp",
        "bar_open",
        "bar_high",
        "bar_low",
        "bar_close",
    ]

    for table in ("production_closed_trades", "trade_executions"):
        columns = _table_columns(table)
        if not columns:
            continue
        select_exprs = [col if col in columns else f"NULL AS {col}" for col in desired_columns]
        if table == "production_closed_trades":
            where_clause = "WHERE ts_signal_id IS NOT NULL AND TRIM(ts_signal_id) <> ''"
        else:
            where_clause = (
                "WHERE is_close = 1 "
                "AND COALESCE(is_diagnostic, 0) = 0 "
                "AND COALESCE(is_synthetic, 0) = 0 "
                "AND ts_signal_id IS NOT NULL AND TRIM(ts_signal_id) <> ''"
            )
        query = f"SELECT {', '.join(select_exprs)} FROM {table} {where_clause}"
        try:
            cur = conn.execute(query)
            cols = [col[0] for col in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            continue
    return []


def _load_price_history(conn: sqlite3.Connection, ticker: str) -> list[tuple[datetime.date, float]]:
    try:
        cur = conn.execute(
            """
            SELECT date, COALESCE(adj_close, close) AS close
            FROM ohlcv_data
            WHERE UPPER(ticker) = UPPER(?)
              AND COALESCE(adj_close, close) IS NOT NULL
            ORDER BY date ASC
            """,
            (ticker,),
        )
    except Exception:
        return []
    history: list[tuple[datetime.date, float]] = []
    for date_raw, close_raw in cur.fetchall():
        date_obj = _parse_date(date_raw)
        close = _safe_float(close_raw)
        if date_obj is None or close is None:
            continue
        history.append((date_obj, close))
    return history


def _forward_return_pct(history: list[tuple[datetime.date, float]], anchor_date: Optional[datetime.date], horizon: int | None) -> Optional[float]:
    if not history or anchor_date is None or horizon is None or horizon <= 0:
        return None
    index_map = {date: idx for idx, (date, _) in enumerate(history)}
    idx = index_map.get(anchor_date)
    if idx is None:
        return None
    future_idx = idx + int(horizon)
    if future_idx >= len(history):
        return None
    entry_close = history[idx][1]
    future_close = history[future_idx][1]
    if entry_close == 0:
        return None
    return round(((future_close - entry_close) / entry_close) * 100.0, 6)


def _bar_position_proxy(signal_timestamp: Any, bar_timestamp: Any) -> Optional[float]:
    signal_dt = _parse_utc(signal_timestamp)
    bar_dt = _parse_utc(bar_timestamp)
    if signal_dt is None or bar_dt is None:
        return None
    lag_seconds = max(0.0, (signal_dt - bar_dt).total_seconds())
    # Normalize against a one-hour bar proxy so late-in-bar signals approach 0
    # and at-close signals approach 1. The value is a relative fill-quality
    # proxy only, not a literal market microstructure measurement.
    return round(max(0.0, min(1.0, 1.0 - (lag_seconds / 3600.0))), 6)


def _bar_range_fraction(bar_high: Any, bar_low: Any, bar_close: Any) -> Optional[float]:
    high = _safe_float(bar_high)
    low = _safe_float(bar_low)
    close = _safe_float(bar_close)
    if high is None or low is None or close is None or close <= 0:
        return None
    return round(max(0.0, high - low) / close, 6)


def build_family_calibration_row(
    *,
    db_path: Path,
    audit_dir: Path,
    window_cycles: Optional[int] = None,
    window_start_utc: Any = None,
    window_end_utc: Any = None,
) -> dict[str, Any]:
    audit_rows = _load_audit_rows(audit_dir)
    closed_rows: list[dict[str, Any]] = []
    conn: Optional[sqlite3.Connection] = None
    try:
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            closed_rows = _load_closed_trades(conn)
    finally:
        if conn is not None:
            conn.close()

    family_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "signals_seen": 0,
            "blocked_by_snr": 0,
            "blocked_by_evidence": 0,
            "_snr_values": [],
            "_blocked_forward_returns": [],
        }
    )
    regime_distribution: Counter[str] = Counter()

    for row in audit_rows:
        family = row.get("model_type") or "UNKNOWN"
        stats = family_stats[str(family)]
        stats["signals_seen"] += 1
        if row.get("blocked_by_snr"):
            stats["blocked_by_snr"] += 1
        if row.get("blocked_by_evidence"):
            stats["blocked_by_evidence"] += 1
        if row.get("snr") is not None:
            stats["_snr_values"].append(float(row["snr"]))
        regime = row.get("regime")
        if regime:
            regime_distribution[str(regime)] += 1

    price_cache: dict[str, list[tuple[datetime.date, float]]] = {}
    for row in audit_rows:
        if not (row.get("blocked_by_snr") or row.get("blocked_by_evidence")):
            continue
        ticker = row.get("ticker")
        if not ticker:
            continue
        if ticker not in price_cache and conn is not None:
            price_cache[ticker] = _load_price_history(conn, ticker)
        history = price_cache.get(ticker, [])
        anchor = _parse_date(row.get("bar_timestamp") or row.get("signal_timestamp"))
        ret = _forward_return_pct(history, anchor, row.get("forecast_horizon"))
        if ret is None:
            continue
        family = row.get("model_type") or "UNKNOWN"
        family_stats[str(family)]["_blocked_forward_returns"].append(ret)

    shadow_rows = [row for row in audit_rows if str(row.get("execution_mode") or "").strip().lower() == "synthetic"]
    shadow_position_proxies: list[float] = []
    shadow_range_fractions: list[float] = []
    for row in shadow_rows:
        position_proxy = row.get("bar_position_proxy")
        if position_proxy is None:
            position_proxy = _bar_position_proxy(row.get("signal_timestamp"), row.get("bar_timestamp"))
        if position_proxy is not None:
            shadow_position_proxies.append(float(position_proxy))
        range_fraction = _bar_range_fraction(row.get("bar_high"), row.get("bar_low"), row.get("bar_close"))
        if range_fraction is not None:
            shadow_range_fractions.append(float(range_fraction))

    closed_by_family: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_pnl": 0.0, "_wins": 0}
    )
    attribution_available = False
    audit_index: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        tsid = str(row.get("ts_signal_id") or "").strip()
        if tsid and tsid not in audit_index:
            audit_index[tsid] = row
    for row in closed_rows:
        family_row = audit_index.get(str(row.get("ts_signal_id") or "").strip())
        family = str(family_row.get("model_type") or "").strip().upper() if family_row else ""
        if not family:
            continue
        stats = closed_by_family[family]
        pnl = _safe_float(row.get("realized_pnl")) or 0.0
        stats["count"] += 1
        stats["total_pnl"] += pnl
        stats["_wins"] += 1 if pnl > 0 else 0
        attribution_available = True

    if window_cycles is None:
        window_cycles = len(audit_rows)

    window_start_dt = _parse_utc(window_start_utc)
    window_end_dt = _parse_utc(window_end_utc)
    if window_start_dt is None or window_end_dt is None:
        timestamps = [
            ts
            for ts in (_parse_utc(row.get("signal_timestamp")) or _parse_utc(row.get("bar_timestamp")) for row in audit_rows)
            if ts is not None
        ]
        if timestamps:
            if window_start_dt is None:
                window_start_dt = min(timestamps)
            if window_end_dt is None:
                window_end_dt = max(timestamps)
    now_utc = datetime.now(timezone.utc)
    if window_start_dt is None:
        window_start_dt = now_utc
    if window_end_dt is None:
        window_end_dt = now_utc

    days_span = max(0.0, (window_end_dt - window_start_dt).total_seconds() / 86400.0)
    regime_count = sum(1 for _regime, count in regime_distribution.items() if count > 0)
    analysis_gate_reasons: list[str] = []
    if int(window_cycles) < MIN_ANALYSIS_CYCLES:
        analysis_gate_reasons.append("window_cycles_below_min")
    if regime_count < MIN_ANALYSIS_REGIMES:
        analysis_gate_reasons.append("regime_diversity_insufficient")
    if days_span < float(MIN_ANALYSIS_DAYS):
        analysis_gate_reasons.append("window_span_days_below_min")

    family_payload: dict[str, Any] = {}
    for family, stats in sorted(family_stats.items()):
        snr_values = [float(value) for value in stats.pop("_snr_values", [])]
        blocked_forward_returns = [float(value) for value in stats.pop("_blocked_forward_returns", [])]
        family_payload[family] = {
            "signals_seen": int(stats["signals_seen"]),
            "blocked_by_snr": int(stats["blocked_by_snr"]),
            "blocked_by_evidence": int(stats["blocked_by_evidence"]),
            "observed_snr_summary": {
                "p5": _percentile(snr_values, 5),
                "p25": _percentile(snr_values, 25),
                "p50": _percentile(snr_values, 50),
                "p75": _percentile(snr_values, 75),
                "p95": _percentile(snr_values, 95),
            },
            "simulated_forward_return_if_unblocked": statistics.median(blocked_forward_returns) if blocked_forward_returns else None,
        }

    closed_family_payload: dict[str, Any] = {}
    for family, stats in sorted(closed_by_family.items()):
        count = int(stats["count"])
        closed_family_payload[family] = {
            "count": count,
            "total_pnl": round(float(stats["total_pnl"]), 6),
            "win_rate": round(float(stats["_wins"]) / count, 6) if count else None,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "window_cycles": int(window_cycles),
        "window_start_utc": window_start_dt.isoformat().replace("+00:00", "Z"),
        "window_end_utc": window_end_dt.isoformat().replace("+00:00", "Z"),
        "window_span_days": round(days_span, 6),
        "regime_distribution": dict(sorted(regime_distribution.items())),
        "family_stats": family_payload,
        "closed_trades_by_model_family": closed_family_payload,
        "attribution_available": bool(attribution_available),
        "analysis_gate_passed": not analysis_gate_reasons,
        "analysis_gate_reasons": analysis_gate_reasons,
        "shadow_trade_metrics": {
            "count": len(shadow_rows),
            "bar_position_proxy": {
                "p25": _percentile(shadow_position_proxies, 25),
                "p50": _percentile(shadow_position_proxies, 50),
                "p75": _percentile(shadow_position_proxies, 75),
            },
            "bar_range_fraction": {
                "p25": _percentile(shadow_range_fractions, 25),
                "p50": _percentile(shadow_range_fractions, 50),
                "p75": _percentile(shadow_range_fractions, 75),
            },
            "spread_proxy_note": "range-based, upward-biased",
        },
        "window_source": str(audit_dir),
    }


def _append_jsonl_row(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_family_calibration_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                LOGGER.warning("Skipping malformed family calibration row %s:%s (%s)", path, lineno, exc)
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Append a family calibration measurement row.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path.")
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR), help="Forecast audit directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSONL output path.")
    parser.add_argument("--window-cycles", type=int, default=None, help="Override window cycle count.")
    parser.add_argument("--window-start-utc", default=None, help="Override window start UTC.")
    parser.add_argument("--window-end-utc", default=None, help="Override window end UTC.")
    parser.add_argument("--json", action="store_true", help="Print the row to stdout as JSON.")
    args = parser.parse_args(argv)

    row = build_family_calibration_row(
        db_path=Path(args.db),
        audit_dir=Path(args.audit_dir),
        window_cycles=args.window_cycles,
        window_start_utc=args.window_start_utc,
        window_end_utc=args.window_end_utc,
    )
    _append_jsonl_row(Path(args.output), row)
    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        print(f"[OK] appended family calibration row to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
