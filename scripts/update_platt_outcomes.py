"""
update_platt_outcomes.py - Platt scaling outcome reconciliation.

Reads logs/signals/quant_validation.jsonl, queries trade_executions for closed
trades matching each entry's signal_id, and writes an 'outcome' field back into
the JSONL. The file is rewritten atomically (temp file + os.replace).

Usage:
    python scripts/update_platt_outcomes.py [--db PATH] [--log PATH] [--dry-run]

Exit codes:
    0  success (even if 0 entries were updated)
    1  DB not found or JSONL parse error
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Phase 7.13-C1: central path constants (fallback gracefully if etl not in sys.path)
try:
    from etl.paths import DB_PATH as _DEFAULT_DB_PATH, QUANT_VALIDATION_JSONL as _DEFAULT_JSONL_PATH
except ImportError:
    _DEFAULT_DB_PATH = Path("data/portfolio_maximizer.db")
    _DEFAULT_JSONL_PATH = Path("logs/signals/quant_validation.jsonl")

logger = logging.getLogger(__name__)
MATCH_DIAGNOSTIC_SAMPLE_LIMIT = 5
MATCH_TIME_TOLERANCE_DAYS = 1
MATCH_TIME_TOLERANCE_MINUTES = 90
_DEFAULT_EXECUTION_LOG_PATH = Path("logs/automation/execution_log.jsonl")
_DEFAULT_DATE_FALLBACK_HISTORY_PATH = Path("logs/audit_gate/date_fallback_rate_history.jsonl")
ELIGIBILITY_BUFFER = timedelta(minutes=5)
EXECUTION_LOG_MALFORMED_SAMPLE_LIMIT = 3
DATE_FALLBACK_WINDOW_RUNS_DEFAULT = 30
DATE_FALLBACK_SLO_MAX_RATE_DEFAULT = 0.05
DATE_FALLBACK_HISTORY_RETENTION_RUNS = 300


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_rate(value: Any) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 0.0
    if rate < 0.0:
        return 0.0
    if rate > 1.0:
        return 1.0
    return rate


def _load_date_fallback_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_date_fallback_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    if payload:
        payload += "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _evaluate_date_fallback_slo(
    *,
    date_fallback_rate: float,
    matched_new: int,
    timestamp_match_rate: float,
    history_path: Optional[Path],
    window_runs: int,
    max_rate: float,
    persist_history: bool,
) -> Dict[str, Any]:
    window = max(int(window_runs), 1)
    slo_max_rate = _coerce_rate(max_rate)
    run_entry = {
        "generated_utc": now_utc().isoformat(),
        "date_fallback_rate": _coerce_rate(date_fallback_rate),
        "matched_new": int(matched_new),
        "timestamp_match_rate": _coerce_rate(timestamp_match_rate),
    }
    if history_path is None:
        tail = [run_entry]
        rolling_rate = _coerce_rate(date_fallback_rate)
        return {
            "pass": rolling_rate <= slo_max_rate,
            "rolling_rate": rolling_rate,
            "records_considered": 1,
            "window_runs": window,
            "slo_max_rate": slo_max_rate,
            "history_path": None,
            "history_persisted": False,
        }

    history = _load_date_fallback_history(history_path)
    history.append(run_entry)
    if len(history) > DATE_FALLBACK_HISTORY_RETENTION_RUNS:
        history = history[-DATE_FALLBACK_HISTORY_RETENTION_RUNS :]
    if persist_history:
        _write_date_fallback_history(history_path, history)
    tail = history[-window:]
    rates = [_coerce_rate(item.get("date_fallback_rate")) for item in tail]
    rolling_rate = (sum(rates) / len(rates)) if rates else 0.0
    return {
        "pass": rolling_rate <= slo_max_rate,
        "rolling_rate": rolling_rate,
        "records_considered": len(rates),
        "window_runs": window,
        "slo_max_rate": slo_max_rate,
        "history_path": str(history_path),
        "history_persisted": bool(persist_history),
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), timeout=5.0)


def _trade_execution_columns(conn: sqlite3.Connection) -> set[str]:
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(trade_executions)")
        return {str(row[1]) for row in cur.fetchall() if len(row) > 1}
    except Exception:
        return set()


def _normalize_symbol(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.endswith(".US"):
        text = text[:-3]
    return text or None


def _parse_forecast_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_trade_date(value: Any) -> Optional[datetime.date]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 10:
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return None
    parsed = _parse_forecast_time(text)
    return parsed.date() if parsed is not None else None


def _coerce_horizon(entry: Dict[str, Any]) -> Optional[int]:
    qv = entry.get("quant_validation") if isinstance(entry.get("quant_validation"), dict) else {}
    forecast_edge = qv.get("forecast_edge") if isinstance(qv.get("forecast_edge"), dict) else {}
    for raw in (
        entry.get("forecast_horizon"),
        qv.get("forecast_horizon"),
        entry.get("holding_period_days"),
        qv.get("holding_period_days"),
        entry.get("forecast_edge_horizon"),
        forecast_edge.get("horizon"),
    ):
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _build_match_record(entry_index: int, entry: Dict[str, Any]) -> Dict[str, Any]:
    signal_id = entry.get("signal_id")
    signal_id_text = str(signal_id).strip() if signal_id is not None else ""
    explicit_forecast_time_raw = entry.get("forecast_time") or entry.get("signal_timestamp")
    forecast_time_raw = explicit_forecast_time_raw or entry.get("timestamp")
    forecast_time = _parse_forecast_time(forecast_time_raw)
    forecast_horizon = _coerce_horizon(entry)
    symbol = _normalize_symbol(entry.get("ticker") or entry.get("symbol"))
    expected_close_ts = None
    expected_close_date = None
    if forecast_time is not None and forecast_horizon is not None:
        expected_close_ts = forecast_time + timedelta(days=forecast_horizon)
        expected_close_date = expected_close_ts.date()
    stable_key = (
        symbol or "",
        forecast_time.isoformat() if forecast_time is not None else "",
        "" if forecast_horizon is None else str(forecast_horizon),
        signal_id_text,
    )
    return {
        "entry_index": entry_index,
        "entry": entry,
        "symbol": symbol,
        "forecast_time_raw": forecast_time_raw,
        "forecast_time": forecast_time,
        "forecast_horizon": forecast_horizon,
        "expected_close_ts": expected_close_ts,
        "expected_close_date": expected_close_date,
        "ts_signal_id": signal_id_text or None,
        "stable_key": stable_key,
        "fingerprint": stable_key[:3],
        "failure_reason": None,
        "query": {},
        "match_source": None,
    }


def _eligible_for_outcome_match(record: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    expected_close_ts = record.get("expected_close_ts")
    if expected_close_ts is None:
        return True, None
    if expected_close_ts + ELIGIBILITY_BUFFER > now_utc():
        return False, "expected_close_ts_in_future"
    return True, None


def _summarize_not_yet_eligible(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    expected_close_timestamps = [
        item.get("expected_close_ts")
        for item in records
        if item.get("failure_reason") == "NOT_YET_ELIGIBLE" and item.get("expected_close_ts") is not None
    ]
    if not expected_close_timestamps:
        return None

    today = now_utc().date()
    expected_close_dates = [item.date() for item in expected_close_timestamps]
    counts = Counter(expected_close_dates)
    earliest = min(expected_close_timestamps)
    latest = max(expected_close_timestamps)
    upcoming = [
        {"expected_close_date": due.isoformat(), "count": counts[due]}
        for due in sorted(counts.keys())[:5]
    ]
    return {
        "count": len(expected_close_timestamps),
        "earliest_expected_close_ts": earliest.isoformat(),
        "latest_expected_close_ts": latest.isoformat(),
        "earliest_expected_close_date": earliest.date().isoformat(),
        "latest_expected_close_date": latest.date().isoformat(),
        "next_due_in_days": max((earliest.date() - today).days, 0),
        "upcoming_due_counts": upcoming,
    }


def _candidate_row(
    *,
    ticker: Any,
    trade_date: Any,
    close_ts: Any,
    pnl: Any,
    pnl_pct: Any,
    close_ts_signal_id: Any,
    source: str,
) -> Optional[Dict[str, Any]]:
    try:
        pnl_f = float(pnl)
    except (TypeError, ValueError):
        return None
    try:
        pnl_pct_f = round(float(pnl_pct), 6) if pnl_pct is not None else None
    except (TypeError, ValueError):
        pnl_pct_f = None
    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "close_ts": close_ts,
        "outcome": {
            "win": pnl_f > 0,
            "pnl": round(pnl_f, 4),
            "pnl_pct": pnl_pct_f,
        },
        "close_ts_signal_id": str(close_ts_signal_id).strip() if close_ts_signal_id is not None else None,
        "match_source": source,
    }


def _append_candidate_rows(
    bucket: Dict[str, List[Dict[str, Any]]],
    rows: list,
    *,
    source: str,
) -> None:
    for match_id, ticker, trade_date, close_ts, pnl, pnl_pct, close_ts_signal_id in rows:
        if match_id is None:
            continue
        candidate = _candidate_row(
            ticker=ticker,
            trade_date=trade_date,
            close_ts=close_ts,
            pnl=pnl,
            pnl_pct=pnl_pct,
            close_ts_signal_id=close_ts_signal_id,
            source=source,
        )
        if candidate is None:
            continue
        bucket.setdefault(str(match_id), []).append(candidate)


def _fetch_outcome_candidates_for_signals(
    conn: sqlite3.Connection,
    signal_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {open_ts_signal_id: [candidate_rows...]} for closed trades matching any ids."""
    if not signal_ids:
        return {}

    columns = _trade_execution_columns(conn)
    if "ts_signal_id" not in columns:
        logger.warning("trade_executions.ts_signal_id missing; ts_signal_id match path disabled")
        return {}

    placeholders = ",".join("?" * len(signal_ids))
    result: Dict[str, List[Dict[str, Any]]] = {}
    cur = conn.cursor()
    ticker_expr = "ticker AS ticker" if "ticker" in columns else "NULL AS ticker"
    trade_date_expr = "trade_date AS trade_date" if "trade_date" in columns else "NULL AS trade_date"
    close_ts_expr = (
        "bar_timestamp AS close_ts"
        if "bar_timestamp" in columns
        else (
            "created_at AS close_ts"
            if "created_at" in columns
            else ("trade_date AS close_ts" if "trade_date" in columns else "NULL AS close_ts")
        )
    )

    try:
        cur.execute(
            f"""
            SELECT
                ts_signal_id AS match_id,
                {ticker_expr},
                {trade_date_expr},
                {close_ts_expr},
                realized_pnl,
                realized_pnl_pct,
                ts_signal_id AS close_ts_signal_id
            FROM trade_executions
            WHERE ts_signal_id IN ({placeholders})
              AND is_close = 1
              AND realized_pnl IS NOT NULL
            ORDER BY ABS(realized_pnl) DESC
            """,
            signal_ids,
        )
        _append_candidate_rows(result, cur.fetchall(), source="direct_ts_signal_id")
    except Exception as exc:
        logger.error("DB query (pass 1) failed: %s", exc)

    if "entry_trade_id" in columns and "id" in columns:
        close_ticker_expr = "c.ticker AS ticker" if "ticker" in columns else "NULL AS ticker"
        close_trade_date_expr = "c.trade_date AS trade_date" if "trade_date" in columns else "NULL AS trade_date"
        close_ts_expr_joined = (
            "c.bar_timestamp AS close_ts"
            if "bar_timestamp" in columns
            else (
                "c.created_at AS close_ts"
                if "created_at" in columns
                else ("c.trade_date AS close_ts" if "trade_date" in columns else "NULL AS close_ts")
            )
        )
        try:
            cur.execute(
                f"""
                SELECT
                    o.ts_signal_id AS match_id,
                    {close_ticker_expr},
                    {close_trade_date_expr},
                    {close_ts_expr_joined},
                    c.realized_pnl,
                    c.realized_pnl_pct,
                    c.ts_signal_id AS close_ts_signal_id
                FROM trade_executions c
                JOIN trade_executions o ON c.entry_trade_id = o.id
                WHERE o.ts_signal_id IN ({placeholders})
                  AND c.is_close = 1
                  AND c.realized_pnl IS NOT NULL
                ORDER BY ABS(c.realized_pnl) DESC
                """,
                signal_ids,
            )
            _append_candidate_rows(result, cur.fetchall(), source="entry_trade_open_signal")
        except Exception as exc:
            logger.error("DB query (pass 2) failed: %s", exc)

    return result


def _fetch_symbol_time_candidates(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    expected_close_ts: Optional[datetime],
    expected_close_date: Optional[datetime.date],
) -> List[Dict[str, Any]]:
    columns = _trade_execution_columns(conn)
    if "ticker" not in columns or "trade_date" not in columns:
        return []

    cur = conn.cursor()
    anchor_date: Optional[datetime.date] = expected_close_ts.date() if expected_close_ts else expected_close_date
    if anchor_date is None:
        return []
    start_date = (anchor_date - timedelta(days=MATCH_TIME_TOLERANCE_DAYS)).isoformat()
    end_date = (anchor_date + timedelta(days=MATCH_TIME_TOLERANCE_DAYS)).isoformat()
    close_ts_expr = (
        "bar_timestamp AS close_ts"
        if "bar_timestamp" in columns
        else (
            "created_at AS close_ts"
            if "created_at" in columns
            else ("trade_date AS close_ts" if "trade_date" in columns else "NULL AS close_ts")
        )
    )
    try:
        cur.execute(
            f"""
            SELECT ticker, trade_date, {close_ts_expr}, realized_pnl, realized_pnl_pct, ts_signal_id
            FROM trade_executions
            WHERE is_close = 1
              AND realized_pnl IS NOT NULL
              AND trade_date BETWEEN ? AND ?
            ORDER BY ABS(realized_pnl) DESC
            """,
            (start_date, end_date),
        )
    except Exception as exc:
        logger.error("DB query (symbol/time fallback) failed: %s", exc)
        return []

    matches: List[Dict[str, Any]] = []
    for ticker, trade_date, close_ts, pnl, pnl_pct, close_ts_signal_id in cur.fetchall():
        if _normalize_symbol(ticker) != symbol:
            continue
        candidate = _candidate_row(
            ticker=ticker,
            trade_date=trade_date,
            close_ts=close_ts,
            pnl=pnl,
            pnl_pct=pnl_pct,
            close_ts_signal_id=close_ts_signal_id,
            source="symbol_time_fallback",
        )
        if candidate is not None:
            matches.append(candidate)
    return matches


def _fetch_open_signal_ids(
    conn: sqlite3.Connection,
    signal_ids: List[str],
) -> set[str]:
    if not signal_ids:
        return set()
    columns = _trade_execution_columns(conn)
    if "ts_signal_id" not in columns:
        return set()

    placeholders = ",".join("?" * len(signal_ids))
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT DISTINCT ts_signal_id
            FROM trade_executions
            WHERE ts_signal_id IN ({placeholders})
              AND is_close = 0
            """,
            signal_ids,
        )
    except Exception as exc:
        logger.error("DB query (open-leg presence) failed: %s", exc)
        return set()
    return {
        str(row[0]).strip()
        for row in cur.fetchall()
        if row and row[0] is not None and str(row[0]).strip()
    }


def _select_candidate_for_record(
    record: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    query_mode: str,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    expected_close_ts = record.get("expected_close_ts")
    expected_close_date = record.get("expected_close_date")
    query: Dict[str, Any] = {
        "mode": query_mode,
        "filters": {
            "symbol": record.get("symbol"),
            "forecast_time": record.get("forecast_time").isoformat() if record.get("forecast_time") else None,
            "forecast_horizon": record.get("forecast_horizon"),
            "ts_signal_id": record.get("ts_signal_id"),
            "expected_close_ts": (
                expected_close_ts.isoformat() if isinstance(expected_close_ts, datetime) else None
            ),
            "expected_close_date": record.get("expected_close_date").isoformat()
            if record.get("expected_close_date") is not None
            else None,
            "time_tolerance_days": MATCH_TIME_TOLERANCE_DAYS,
            "time_tolerance_minutes": MATCH_TIME_TOLERANCE_MINUTES,
        },
        "candidate_count": len(candidates),
        "candidate_sources": sorted(
            {c.get("match_source") for c in candidates if c.get("match_source")}
        ),
    }
    if not candidates:
        return None, "NO_ROW", query

    same_symbol_candidates: List[Dict[str, Any]] = []
    symbol_mismatch_count = 0
    symbol_unknown_count = 0
    for candidate in candidates:
        candidate_symbol = _normalize_symbol(candidate.get("ticker"))
        if record.get("symbol") and candidate_symbol:
            if candidate_symbol != record["symbol"]:
                symbol_mismatch_count += 1
                continue
        elif record.get("symbol") and candidate_symbol is None:
            symbol_unknown_count += 1
        same_symbol_candidates.append(candidate)

    if not same_symbol_candidates and symbol_mismatch_count > 0:
        query["symbol_mismatch_count"] = symbol_mismatch_count
        return None, "SYMBOL_MISMATCH", query

    valid_candidates: List[Dict[str, Any]] = []
    time_mismatch_count = 0
    time_unvalidated_count = 0
    date_fallback_count = 0
    time_tolerance = timedelta(minutes=MATCH_TIME_TOLERANCE_MINUTES)
    for candidate in same_symbol_candidates:
        raw_close_ts = candidate.get("close_ts")
        candidate_close_ts = _parse_forecast_time(raw_close_ts)
        close_ts_text = str(raw_close_ts).strip() if raw_close_ts is not None else ""
        if len(close_ts_text) == 10:
            # Date-only close timestamps are coarse; treat as fallback-grade evidence.
            candidate_close_ts = None
        candidate_trade_date = _parse_trade_date(candidate.get("trade_date"))
        if expected_close_ts is not None and candidate_close_ts is not None:
            if abs(candidate_close_ts - expected_close_ts) > time_tolerance:
                time_mismatch_count += 1
                continue
            candidate_copy = dict(candidate)
            candidate_copy["match_anchor"] = "timestamp"
            valid_candidates.append(candidate_copy)
            continue

        if expected_close_date is not None and candidate_trade_date is not None:
            if abs((candidate_trade_date - expected_close_date).days) > MATCH_TIME_TOLERANCE_DAYS:
                time_mismatch_count += 1
                continue
        elif expected_close_date is not None and candidate_trade_date is None:
            time_unvalidated_count += 1
        candidate_copy = dict(candidate)
        candidate_copy["match_anchor"] = "date_fallback"
        date_fallback_count += 1
        valid_candidates.append(candidate_copy)

    query["symbol_unknown_count"] = symbol_unknown_count
    query["time_unvalidated_count"] = time_unvalidated_count
    query["date_fallback_candidates"] = date_fallback_count
    if not valid_candidates and time_mismatch_count > 0:
        query["time_mismatch_count"] = time_mismatch_count
        return None, "TIME_MISMATCH", query
    if not valid_candidates:
        return None, "NO_ROW", query
    if len(valid_candidates) > 1:
        if record.get("symbol") is None and expected_close_date is None:
            query["matching_rows"] = len(valid_candidates)
            query["ambiguity_resolution"] = "largest_abs_pnl"
            return valid_candidates[0], "MATCHED", query
        query["matching_rows"] = len(valid_candidates)
        return None, "MULTIPLE_ROWS", query
    selected = valid_candidates[0]
    if selected.get("match_anchor") == "date_fallback":
        query["reason_code"] = "DATE_FALLBACK_USED"
    return selected, "MATCHED", query


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
    return entries


def _normalize_exec_status(value: Any, executed_flag: Any) -> Optional[str]:
    if isinstance(executed_flag, bool):
        return "EXECUTED" if executed_flag else "NOT_EXECUTED"
    text = str(value).strip().upper() if value is not None else ""
    return text or None


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_execution_events(
    path: Path,
) -> Dict[str, Any]:
    exists = path.exists()
    by_ts_signal_id: Dict[str, List[Dict[str, Any]]] = {}
    by_run_ticker_time: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    by_run_ticker: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    result: Dict[str, Any] = {
        "exists": exists,
        "lines_total": 0,
        "lines_parsed": 0,
        "lines_bad": 0,
        "first_malformed_line_numbers": [],
        "loaded_ok": False,
        "fatal": False,
        "integrity": "MISSING",
        "by_ts_signal_id": by_ts_signal_id,
        "by_run_ticker_time": by_run_ticker_time,
        "by_run_ticker": by_run_ticker,
    }
    if not exists:
        return result

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.warning("Failed to read execution log %s: %s", path, exc)
        result["fatal"] = True
        result["integrity"] = "ERROR_READ_FAILED"
        return result

    malformed_sample: List[int] = []
    for lineno, raw_line in enumerate(raw_lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        result["lines_total"] += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            result["lines_bad"] += 1
            if len(malformed_sample) < EXECUTION_LOG_MALFORMED_SAMPLE_LIMIT:
                malformed_sample.append(lineno)
            logger.warning("Malformed execution log line %d in %s: %s", lineno, path, exc)
            continue
        if not isinstance(item, dict):
            result["lines_bad"] += 1
            if len(malformed_sample) < EXECUTION_LOG_MALFORMED_SAMPLE_LIMIT:
                malformed_sample.append(lineno)
            logger.warning("Malformed execution log line %d in %s: expected JSON object", lineno, path)
            continue

        result["lines_parsed"] += 1
        status = _normalize_exec_status(item.get("status"), item.get("executed"))
        if status is None:
            continue
        ts_signal_id = _coerce_text(item.get("ts_signal_id"))
        run_id = _coerce_text(item.get("run_id"))
        ticker = _normalize_symbol(item.get("ticker"))
        signal_timestamp = _coerce_text(item.get("signal_timestamp"))
        event = {
            "status": status,
            "reason": _coerce_text(item.get("reason")),
            "ts_signal_id": ts_signal_id,
            "run_id": run_id,
            "ticker": ticker,
            "signal_timestamp": signal_timestamp,
        }
        if ts_signal_id:
            by_ts_signal_id.setdefault(ts_signal_id, []).append(event)
        if run_id and ticker:
            by_run_ticker.setdefault((run_id, ticker), []).append(event)
        if run_id and ticker and signal_timestamp:
            by_run_ticker_time.setdefault((run_id, ticker, signal_timestamp), []).append(event)

    result["first_malformed_line_numbers"] = malformed_sample
    result["loaded_ok"] = bool(result["lines_parsed"] > 0)
    if result["lines_total"] > 0 and result["lines_parsed"] == 0:
        result["fatal"] = True
        result["integrity"] = "ERROR_NO_VALID_EVENTS"
    elif result["loaded_ok"] and result["lines_bad"] > 0:
        result["integrity"] = "WARN_PARTIAL_PARSE"
    elif result["loaded_ok"]:
        result["integrity"] = "OK"
    elif result["lines_total"] == 0:
        result["integrity"] = "EMPTY"
    else:
        result["integrity"] = "MISSING"
    return result


def _attach_execution_state(
    record: Dict[str, Any],
    *,
    execution_log_loaded: bool,
    execution_log_integrity: str,
    execution_log_state: Dict[str, Any],
    by_ts_signal_id: Dict[str, List[Dict[str, Any]]],
    by_run_ticker_time: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    by_run_ticker: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    if not execution_log_loaded:
        if execution_log_integrity.startswith("ERROR"):
            record["failure_reason"] = "EXECUTION_LOG_CORRUPT"
            record["query"] = {
                "mode": "execution_gate",
                "reason": "execution log integrity failure",
                "execution_log_integrity": execution_log_integrity,
                "lines_total": int(execution_log_state.get("lines_total", 0)),
                "lines_parsed": int(execution_log_state.get("lines_parsed", 0)),
                "lines_bad": int(execution_log_state.get("lines_bad", 0)),
                "first_malformed_line_numbers": execution_log_state.get("first_malformed_line_numbers", []),
            }
            return record
        if bool(execution_log_state.get("exists")):
            record["failure_reason"] = "NO_EXECUTION_RECORD"
            record["query"] = {
                "mode": "execution_gate",
                "reason": "execution log has no parsed execution events",
                "execution_log_integrity": execution_log_integrity,
                "lines_total": int(execution_log_state.get("lines_total", 0)),
                "lines_parsed": int(execution_log_state.get("lines_parsed", 0)),
            }
            return record
        record["execution_gate_mode"] = "legacy_no_execution_log"
        return record

    event = None
    if record.get("ts_signal_id"):
        events = by_ts_signal_id.get(record["ts_signal_id"], [])
        if events:
            event = events[-1]
            record["execution_gate_mode"] = "ts_signal_id"
            record["execution_event_count"] = len(events)
    if event is None:
        run_id = _coerce_text(record["entry"].get("run_id"))
        forecast_time_raw = _coerce_text(record.get("forecast_time_raw"))
        if run_id and record.get("symbol") and forecast_time_raw:
            key = (run_id, record["symbol"], forecast_time_raw)
            events = by_run_ticker_time.get(key, [])
            if len(events) == 1:
                event = events[0]
                record["execution_gate_mode"] = "run_ticker_time"
                record["execution_event_count"] = 1
            elif len(events) > 1:
                record["failure_reason"] = "MULTIPLE_EXECUTION_ROWS"
                record["query"] = {
                    "mode": "execution_gate",
                    "reason": "multiple execution rows matched run+ticker+signal_timestamp",
                    "run_id": run_id,
                    "symbol": record.get("symbol"),
                    "signal_timestamp": forecast_time_raw,
                    "matching_rows": len(events),
                }
                return record
        if event is None and run_id and record.get("symbol"):
            events = by_run_ticker.get((run_id, record["symbol"]), [])
            if len(events) == 1:
                event = events[0]
                record["execution_gate_mode"] = "run_ticker"
                record["execution_event_count"] = 1

    if event is None:
        record["failure_reason"] = "NO_EXECUTION_RECORD"
        record["query"] = {
            "mode": "execution_gate",
            "reason": "no execution event matched this signal",
        }
        return record

    record["execution_status"] = event.get("status")
    if event.get("status") != "EXECUTED":
        record["failure_reason"] = "NOT_EXECUTED"
        record["query"] = {
            "mode": "execution_gate",
            "execution_status": event.get("status"),
            "reason": event.get("reason"),
            "match_mode": record.get("execution_gate_mode"),
        }
        return record

    record["query"] = {
        "mode": "execution_gate",
        "execution_status": "EXECUTED",
        "match_mode": record.get("execution_gate_mode"),
    }
    return record


def _write_jsonl_atomic(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write entries to a temp file then atomically replace path."""
    parent = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".platt_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, default=_json_default) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


# ---------------------------------------------------------------------------
# Main reconciliation logic
# ---------------------------------------------------------------------------

def reconcile(
    *,
    db_path: Path,
    log_path: Path,
    execution_log_path: Optional[Path] = None,
    dry_run: bool = False,
    date_fallback_history_path: Optional[Path] = _DEFAULT_DATE_FALLBACK_HISTORY_PATH,
    date_fallback_window_runs: int = DATE_FALLBACK_WINDOW_RUNS_DEFAULT,
    date_fallback_slo_max_rate: float = DATE_FALLBACK_SLO_MAX_RATE_DEFAULT,
    enforce_date_fallback_slo: bool = False,
) -> Tuple[int, int, int]:
    """Run the reconciliation.

    Returns (total_entries, updated, already_had_outcome).
    """
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        sys.exit(1)

    if not log_path.exists():
        print(f"[INFO] JSONL log not found: {log_path} - nothing to reconcile.")
        return 0, 0, 0

    entries = _load_jsonl(log_path)
    total = len(entries)
    execution_gate_enabled = execution_log_path is not None
    execution_log_state: Dict[str, Any]
    if execution_gate_enabled and execution_log_path is not None:
        execution_log_state = _load_execution_events(execution_log_path)
    else:
        execution_log_state = {
            "exists": False,
            "lines_total": 0,
            "lines_parsed": 0,
            "lines_bad": 0,
            "first_malformed_line_numbers": [],
            "loaded_ok": False,
            "fatal": False,
            "integrity": "DISABLED",
            "by_ts_signal_id": {},
            "by_run_ticker_time": {},
            "by_run_ticker": {},
        }
    execution_log_loaded = bool(execution_log_state.get("loaded_ok"))
    execution_log_integrity = str(execution_log_state.get("integrity", "MISSING"))
    if not execution_gate_enabled:
        execution_evidence_status = "DISABLED"
    elif execution_log_integrity.startswith("ERROR"):
        execution_evidence_status = "ERROR"
    elif execution_log_integrity == "OK":
        execution_evidence_status = "OK"
    else:
        execution_evidence_status = "WARN"
    execution_log_metrics = {
        "exists": bool(execution_log_state.get("exists")),
        "lines_total": int(execution_log_state.get("lines_total", 0)),
        "lines_parsed": int(execution_log_state.get("lines_parsed", 0)),
        "lines_bad": int(execution_log_state.get("lines_bad", 0)),
        "first_malformed_line_numbers": execution_log_state.get("first_malformed_line_numbers", []),
        "loaded_ok": execution_log_loaded,
        "integrity": execution_log_integrity,
        "fatal": bool(execution_log_state.get("fatal", False)),
    }
    if execution_gate_enabled and not execution_log_metrics["exists"]:
        print(f"[update_platt_outcomes] execution_log_missing={execution_log_path}")
    if execution_log_integrity.startswith("ERROR"):
        logger.error(
            "Execution log integrity failure (%s): total=%d parsed=%d bad=%d",
            execution_log_integrity,
            execution_log_metrics["lines_total"],
            execution_log_metrics["lines_parsed"],
            execution_log_metrics["lines_bad"],
        )

    already_done = 0
    hold_skipped = 0
    other_no_sid = 0
    not_yet_eligible = 0
    pending_records: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for entry_index, entry in enumerate(entries):
        if "outcome" in entry:
            already_done += 1
            continue
        action = str(entry.get("action", "")).upper()
        if action == "HOLD":
            hold_skipped += 1
            continue
        record = _build_match_record(entry_index, entry)
        has_fallback_key = (
            record.get("symbol") is not None
            and record.get("forecast_time") is not None
            and record.get("forecast_horizon") is not None
        )
        if not record.get("ts_signal_id") and not has_fallback_key:
            other_no_sid += 1
            record["failure_reason"] = "MISSING_KEY_FIELDS"
            record["query"] = {"mode": "unmatchable", "reason": "missing_signal_id_and_incomplete_stable_key"}
            diagnostics.append(record)
            continue
        eligible, eligibility_reason = _eligible_for_outcome_match(record)
        if not eligible:
            not_yet_eligible += 1
            record["failure_reason"] = "NOT_YET_ELIGIBLE"
            record["query"] = {
                "mode": "eligibility",
                "reason": eligibility_reason,
                "expected_close_ts": (
                    record["expected_close_ts"].isoformat()
                    if record.get("expected_close_ts") is not None
                    else None
                ),
                "expected_close_date": (
                    record["expected_close_date"].isoformat()
                    if record.get("expected_close_date") is not None
                    else None
                ),
                "eligibility_buffer_minutes": int(ELIGIBILITY_BUFFER.total_seconds() // 60),
            }
            diagnostics.append(record)
            continue
        pending_records.append(record)

    if not pending_records:
        failure_counts = Counter(
            str(item.get("failure_reason"))
            for item in diagnostics
            if item.get("failure_reason")
        )
        eligibility_summary = _summarize_not_yet_eligible(diagnostics)
        matched_new = 0
        matched_total = already_done + matched_new
        date_fallback_slo = _evaluate_date_fallback_slo(
            date_fallback_rate=0.0,
            matched_new=matched_new,
            timestamp_match_rate=0.0,
            history_path=date_fallback_history_path,
            window_runs=date_fallback_window_runs,
            max_rate=date_fallback_slo_max_rate,
            persist_history=not dry_run,
        )
        print(
            f"[update_platt_outcomes] total={total} pending=0 "
            f"matched={matched_total} matched_new={matched_new} "
            f"already_done={already_done} "
            f"still_pending=0 "
            f"hold_skipped={hold_skipped} no_sid={other_no_sid} "
            f"not_yet_eligible={not_yet_eligible} "
            f"timestamp_match_rate={0.0:.2%} "
            f"date_fallback_rate={0.0:.2%} "
            f"date_fallback_slo_pass={int(bool(date_fallback_slo['pass']))} "
            f"date_fallback_slo_rolling_rate={date_fallback_slo['rolling_rate']:.2%} "
            f"date_fallback_slo_window={int(date_fallback_slo['records_considered'])} "
            f"execution_log_loaded={int(execution_log_loaded)} "
            f"execution_log_integrity={execution_log_integrity} "
            f"evidence_status={execution_evidence_status}"
        )
        if execution_gate_enabled:
            print(f"[update_platt_outcomes] execution_log_stats={json.dumps(execution_log_metrics, sort_keys=True)}")
        print(f"[update_platt_outcomes] date_fallback_slo={json.dumps(date_fallback_slo, sort_keys=True)}")
        if failure_counts:
            print(f"[update_platt_outcomes] failure_reasons={json.dumps(dict(sorted(failure_counts.items())))}")
        if eligibility_summary:
            print(f"[update_platt_outcomes] eligibility_window={json.dumps(eligibility_summary, sort_keys=True)}")
        for sample in diagnostics[:MATCH_DIAGNOSTIC_SAMPLE_LIMIT]:
            payload = {
                "symbol": sample.get("symbol"),
                "forecast_time": sample.get("forecast_time_raw"),
                "forecast_horizon": sample.get("forecast_horizon"),
                "expected_close_ts": (
                    sample["expected_close_ts"].isoformat()
                    if sample.get("expected_close_ts") is not None
                    else None
                ),
                "ts_signal_id": sample.get("ts_signal_id"),
                "execution_status": sample.get("execution_status"),
                "match_source": sample.get("match_source"),
                "match_anchor": sample.get("match_anchor"),
                "reason_code": sample.get("reason_code"),
                "failure_reason": sample.get("failure_reason"),
                "query": sample.get("query"),
            }
            print(f"[update_platt_outcomes][sample] {json.dumps(payload, default=_json_default, sort_keys=True)}")
        if enforce_date_fallback_slo and not bool(date_fallback_slo["pass"]):
            logger.error(
                "Date fallback SLO violated (rolling_rate=%.4f > max_rate=%.4f over %d run(s)).",
                float(date_fallback_slo["rolling_rate"]),
                float(date_fallback_slo["slo_max_rate"]),
                int(date_fallback_slo["records_considered"]),
            )
            raise SystemExit(1)
        return total, 0, already_done

    seen_keys: set[Tuple[str, str, str, str]] = set()
    ts_fingerprints: Dict[str, Tuple[str, str, str]] = {}
    conflicting_ids: set[str] = set()
    for record in pending_records:
        stable_key = record["stable_key"]
        if stable_key in seen_keys:
            record["failure_reason"] = "DUPLICATE_KEY"
            record["query"] = {"mode": "dedupe", "stable_key": stable_key}
            diagnostics.append(record)
            continue
        seen_keys.add(stable_key)
        ts_signal_id = record.get("ts_signal_id")
        if ts_signal_id:
            fingerprint = record["fingerprint"]
            prior = ts_fingerprints.get(ts_signal_id)
            if prior is None:
                ts_fingerprints[ts_signal_id] = fingerprint
            elif prior != fingerprint:
                conflicting_ids.add(ts_signal_id)

    for record in pending_records:
        if record.get("failure_reason") is None and record.get("ts_signal_id") in conflicting_ids:
            record["failure_reason"] = "KEY_CONFLICT"
            record["query"] = {
                "mode": "ts_signal_id",
                "ts_signal_id": record.get("ts_signal_id"),
                "reason": "same ts_signal_id observed with multiple stable key fingerprints",
            }
            diagnostics.append(record)

    execution_by_ts_signal_id: Dict[str, List[Dict[str, Any]]] = execution_log_state.get("by_ts_signal_id", {})
    execution_by_run_ticker_time: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = execution_log_state.get(
        "by_run_ticker_time",
        {},
    )
    execution_by_run_ticker: Dict[Tuple[str, str], List[Dict[str, Any]]] = execution_log_state.get(
        "by_run_ticker",
        {},
    )

    active_records: List[Dict[str, Any]] = []
    for record in pending_records:
        if record.get("failure_reason") is not None:
            continue
        if not execution_gate_enabled:
            active_records.append(record)
            continue
        _attach_execution_state(
            record,
            execution_log_loaded=execution_log_loaded,
            execution_log_integrity=execution_log_integrity,
            execution_log_state=execution_log_state,
            by_ts_signal_id=execution_by_ts_signal_id,
            by_run_ticker_time=execution_by_run_ticker_time,
            by_run_ticker=execution_by_run_ticker,
        )
        if record.get("failure_reason") is not None:
            diagnostics.append(record)
            continue
        active_records.append(record)

    conn = _connect(db_path)
    try:
        signal_ids = sorted({r["ts_signal_id"] for r in active_records if r.get("ts_signal_id")})
        candidate_map = _fetch_outcome_candidates_for_signals(conn, signal_ids)
        open_signal_ids = _fetch_open_signal_ids(conn, signal_ids)

        updated = 0
        for record in active_records:
            if record.get("ts_signal_id"):
                candidates = candidate_map.get(record["ts_signal_id"], [])
                candidate, status, query = _select_candidate_for_record(
                    record,
                    candidates,
                    query_mode="ts_signal_id",
                )
            else:
                fallback_candidates = _fetch_symbol_time_candidates(
                    conn,
                    symbol=record["symbol"],
                    expected_close_ts=record.get("expected_close_ts"),
                    expected_close_date=record["expected_close_date"],
                )
                candidate, status, query = _select_candidate_for_record(
                    record,
                    fallback_candidates,
                    query_mode="symbol_time_fallback",
                )

            record["query"] = query
            if candidate is not None:
                record["entry"]["outcome"] = candidate["outcome"]
                record["match_source"] = candidate.get("match_source")
                record["match_anchor"] = candidate.get("match_anchor") or "timestamp"
                if record["match_anchor"] == "date_fallback":
                    record["reason_code"] = "DATE_FALLBACK_USED"
                updated += 1
            else:
                if (
                    status == "NO_ROW"
                    and record.get("ts_signal_id")
                    and record["ts_signal_id"] in open_signal_ids
                ):
                    record["failure_reason"] = "OPEN_ONLY_LIFECYCLE_LAG"
                else:
                    record["failure_reason"] = status
            diagnostics.append(record)
    finally:
        conn.close()

    pending_count = len(pending_records)
    matched_new = sum(1 for item in diagnostics if item.get("match_source"))
    matched_count = already_done + matched_new
    still_pending = pending_count - matched_new
    timestamp_matched = sum(
        1
        for item in diagnostics
        if item.get("match_source") and str(item.get("match_anchor") or "").lower() == "timestamp"
    )
    date_fallback_matched = sum(
        1
        for item in diagnostics
        if item.get("match_source") and str(item.get("match_anchor") or "").lower() == "date_fallback"
    )
    timestamp_match_rate = (timestamp_matched / matched_new) if matched_new else 0.0
    date_fallback_rate = (date_fallback_matched / matched_new) if matched_new else 0.0
    date_fallback_slo = _evaluate_date_fallback_slo(
        date_fallback_rate=date_fallback_rate,
        matched_new=matched_new,
        timestamp_match_rate=timestamp_match_rate,
        history_path=date_fallback_history_path,
        window_runs=date_fallback_window_runs,
        max_rate=date_fallback_slo_max_rate,
        persist_history=not dry_run,
    )
    failure_counts = Counter(
        str(item.get("failure_reason"))
        for item in diagnostics
        if item.get("failure_reason")
    )
    eligibility_summary = _summarize_not_yet_eligible(diagnostics)

    print(
        f"[update_platt_outcomes] total={total} pending={pending_count} "
        f"matched={matched_count} matched_new={matched_new} "
        f"already_done={already_done} "
        f"still_pending={still_pending} "
        f"hold_skipped={hold_skipped} no_sid={other_no_sid} "
        f"not_yet_eligible={not_yet_eligible} "
        f"timestamp_match_rate={timestamp_match_rate:.2%} "
        f"date_fallback_rate={date_fallback_rate:.2%} "
        f"date_fallback_slo_pass={int(bool(date_fallback_slo['pass']))} "
        f"date_fallback_slo_rolling_rate={date_fallback_slo['rolling_rate']:.2%} "
        f"date_fallback_slo_window={int(date_fallback_slo['records_considered'])} "
        f"execution_log_loaded={int(execution_log_loaded)} "
        f"execution_log_integrity={execution_log_integrity} "
        f"evidence_status={execution_evidence_status}"
    )
    if execution_gate_enabled:
        print(f"[update_platt_outcomes] execution_log_stats={json.dumps(execution_log_metrics, sort_keys=True)}")
    print(f"[update_platt_outcomes] date_fallback_slo={json.dumps(date_fallback_slo, sort_keys=True)}")
    if failure_counts:
        print(f"[update_platt_outcomes] failure_reasons={json.dumps(dict(sorted(failure_counts.items())))}")
    if eligibility_summary:
        print(f"[update_platt_outcomes] eligibility_window={json.dumps(eligibility_summary, sort_keys=True)}")

    for sample in diagnostics[:MATCH_DIAGNOSTIC_SAMPLE_LIMIT]:
        payload = {
            "symbol": sample.get("symbol"),
            "forecast_time": sample.get("forecast_time_raw"),
            "forecast_horizon": sample.get("forecast_horizon"),
            "expected_close_ts": (
                sample["expected_close_ts"].isoformat()
                if sample.get("expected_close_ts") is not None
                else None
            ),
            "ts_signal_id": sample.get("ts_signal_id"),
            "execution_status": sample.get("execution_status"),
            "match_source": sample.get("match_source"),
            "match_anchor": sample.get("match_anchor"),
            "reason_code": sample.get("reason_code"),
            "failure_reason": sample.get("failure_reason"),
            "query": sample.get("query"),
        }
        print(f"[update_platt_outcomes][sample] {json.dumps(payload, default=_json_default, sort_keys=True)}")

    if updated > 0 and not dry_run:
        _write_jsonl_atomic(log_path, entries)
        print(f"[update_platt_outcomes] Wrote {total} entries back to {log_path}")
    elif dry_run and updated > 0:
        print(f"[update_platt_outcomes] dry-run: would update {updated} entries (no file written)")

    if enforce_date_fallback_slo and not bool(date_fallback_slo["pass"]):
        logger.error(
            "Date fallback SLO violated (rolling_rate=%.4f > max_rate=%.4f over %d run(s)).",
            float(date_fallback_slo["rolling_rate"]),
            float(date_fallback_slo["slo_max_rate"]),
            int(date_fallback_slo["records_considered"]),
        )
        raise SystemExit(1)

    return total, updated, already_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconcile quant_validation.jsonl with trade_executions outcomes for Platt scaling."
    )
    p.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="Path to portfolio_maximizer.db (default: PORTFOLIO_DB_PATH env or data/portfolio_maximizer.db)",
    )
    p.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="Path to quant_validation.jsonl (default: logs/signals/quant_validation.jsonl)",
    )
    p.add_argument(
        "--execution-log",
        default=None,
        metavar="PATH",
        help="Path to execution_log.jsonl (default: logs/automation/execution_log.jsonl)",
    )
    p.add_argument(
        "--date-fallback-history",
        default=str(_DEFAULT_DATE_FALLBACK_HISTORY_PATH),
        metavar="PATH",
        help=(
            "Path to rolling date-fallback history JSONL "
            "(default: logs/audit_gate/date_fallback_rate_history.jsonl)."
        ),
    )
    p.add_argument(
        "--date-fallback-window-runs",
        type=int,
        default=DATE_FALLBACK_WINDOW_RUNS_DEFAULT,
        metavar="N",
        help=f"Rolling run window for date fallback SLO (default: {DATE_FALLBACK_WINDOW_RUNS_DEFAULT}).",
    )
    p.add_argument(
        "--date-fallback-slo-max-rate",
        type=float,
        default=DATE_FALLBACK_SLO_MAX_RATE_DEFAULT,
        metavar="RATIO",
        help=f"Maximum allowed rolling date_fallback_rate (default: {DATE_FALLBACK_SLO_MAX_RATE_DEFAULT:.2f}).",
    )
    p.add_argument(
        "--enforce-date-fallback-slo",
        action="store_true",
        help="Fail with exit code 1 when rolling date fallback rate exceeds the configured SLO.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing anything.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.db) if args.db else _DEFAULT_DB_PATH
    log_path = Path(args.log) if args.log else _DEFAULT_JSONL_PATH
    execution_log_path = Path(args.execution_log) if args.execution_log else _DEFAULT_EXECUTION_LOG_PATH
    date_fallback_history_path = (
        Path(args.date_fallback_history) if args.date_fallback_history else None
    )

    reconcile(
        db_path=db_path,
        log_path=log_path,
        execution_log_path=execution_log_path,
        dry_run=args.dry_run,
        date_fallback_history_path=date_fallback_history_path,
        date_fallback_window_runs=args.date_fallback_window_runs,
        date_fallback_slo_max_rate=args.date_fallback_slo_max_rate,
        enforce_date_fallback_slo=args.enforce_date_fallback_slo,
    )


if __name__ == "__main__":
    main()
