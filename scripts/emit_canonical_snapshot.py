#!/usr/bin/env python3
"""emit_canonical_snapshot.py

Single source of truth for all PMX performance reporting.
Writes logs/canonical_snapshot_latest.json and prints a human-readable summary.

This is the ONLY file any plan, gate, or documentation is allowed to quote for:
  - Closed PnL / WR / PF          → production_closed_trades view
  - Capital base                   → portfolio_cash_state.initial_capital
  - Time-weighted deployment KPI   → scripts/compute_capital_utilization.py
  - Open risk / open lots          → trade_executions WHERE is_close=0
  - Gate readiness                 → logs/production_gate_latest.json (if present)
  - Unattended readiness           → scripts/institutional_unattended_gate.py --json (if present)

SCHEMA_VERSION = 4. Any consumer must assert schema_version >= 4.
metrics_summary.json is deprecated as a source-of-truth — it is a UI artifact only.

CLI:
    python scripts/emit_canonical_snapshot.py [--db PATH] [--output PATH] [--json]
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

try:  # pragma: no cover - optional dependency is now pinned in requirements.txt
    import pandas as pd
except Exception:  # pragma: no cover - keep emitter usable in degraded bootstrap
    pd = None  # type: ignore[assignment]

try:  # pragma: no cover - imported for schedule-aware freshness
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - dependency may be absent in pre-install envs
    mcal = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.domain_objective import DOMAIN_OBJECTIVE_VERSION

DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_OUTPUT = ROOT / "logs" / "canonical_snapshot_latest.json"
SCHEMA_VERSION = 4
CANONICAL_SOURCE_REGISTRY_PATH = ROOT / "config" / "canonical_source_registry.yml"
FORECAST_AUDIT_SUMMARY_PATH = ROOT / "logs" / "forecast_audits_cache" / "latest_summary.json"
WARMUP_DEADLINE_UTC = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
WARMUP_MATCH_THRESHOLD = 10
WARMUP_TRAJECTORY_ALERT_DAYS = 3
# Two-level coverage alarm: warn when projected closes < 100% of needed (off-track),
# critical when < 25% (severely off-track). A single 0.15 threshold only fires when
# the situation is already hopeless; these thresholds give actionable early warning.
WARMUP_COVERAGE_RATIO_WARN_THRESHOLD = 1.0
WARMUP_COVERAGE_RATIO_CRITICAL_THRESHOLD = 0.25


def _gate_artifact_candidates() -> tuple[Path, Path]:
    return (
        ROOT / "logs" / "audit_gate" / "production_gate_latest.json",
        ROOT / "logs" / "production_gate_latest.json",
    )


def _ui_metrics_summary_path() -> Path:
    return ROOT / "visualizations" / "performance" / "metrics_summary.json"


def _load_yaml_dict(path: Path) -> tuple[dict[str, Any], Optional[str]]:
    if not path.exists():
        return {}, f"missing:{path.name}"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {}, f"unreadable:{path.name}:{exc}"
    if not isinstance(payload, dict):
        return {}, f"invalid:{path.name}:root_not_object"
    return payload, None


def _utc_timestamp(value: Any) -> Optional[datetime]:
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


def _to_iso_utc(value: Any) -> Optional[str]:
    ts = _utc_timestamp(value)
    return ts.isoformat().replace("+00:00", "Z") if ts else None


def _finite_number(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        value = float(raw)
    except Exception:
        return None
    if value != value or abs(value) == float("inf"):
        return None
    return value


def _market_schedule_closes(reference_dt: datetime, *, lookback_days: int = 10, lookahead_days: int = 10) -> tuple[Optional[datetime], Optional[datetime]]:
    if pd is None or mcal is None:
        return None, None
    try:
        calendar = mcal.get_calendar("NYSE")
        start = (reference_dt - timedelta(days=int(lookback_days))).date()
        end = (reference_dt + timedelta(days=int(lookahead_days))).date()
        schedule = calendar.schedule(start_date=start, end_date=end)
    except Exception:
        return None, None
    if schedule.empty or "market_close" not in schedule:
        return None, None

    closes = schedule["market_close"]
    try:
        closes = closes.dt.tz_convert(timezone.utc)
    except Exception:
        try:
            closes = closes.dt.tz_localize(timezone.utc)
        except Exception:
            return None, None

    ref_ts = pd.Timestamp(reference_dt)
    if ref_ts.tzinfo is None:
        ref_ts = ref_ts.tz_localize(timezone.utc)
    else:
        ref_ts = ref_ts.tz_convert(timezone.utc)

    prior = closes[closes <= ref_ts]
    if prior.empty:
        return None, None
    last_expected = prior.iloc[-1].to_pydatetime()

    last_expected_ts = pd.Timestamp(last_expected)
    if last_expected_ts.tzinfo is None:
        last_expected_ts = last_expected_ts.tz_localize(timezone.utc)
    else:
        last_expected_ts = last_expected_ts.tz_convert(timezone.utc)
    later = closes[closes > last_expected_ts]
    next_expected = later.iloc[0].to_pydatetime() if not later.empty else None
    return last_expected, next_expected


def _normalize_registry_entry(entry: Any) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not isinstance(entry, dict):
        return None, {"file": "config/canonical_source_registry.yml", "line": 0, "pattern": "entry", "reason": "entry_not_object"}
    metric = str(entry.get("metric") or "").strip()
    source_file = str(entry.get("source_file") or "").strip()
    query_or_key = str(entry.get("query_or_key") or "").strip()
    if not metric or not source_file or not query_or_key:
        return None, {
            "file": "config/canonical_source_registry.yml",
            "line": 0,
            "pattern": "canonical_sources",
            "reason": "missing_metric_source_file_or_query_or_key",
        }
    return {
        "metric": metric,
        "source_file": source_file,
        "query_or_key": query_or_key,
    }, None


def _load_source_contract() -> dict[str, Any]:
    payload, err = _load_yaml_dict(CANONICAL_SOURCE_REGISTRY_PATH)
    violations: list[dict[str, Any]] = []
    if err:
        violations.append(
            {
                "file": str(CANONICAL_SOURCE_REGISTRY_PATH),
                "line": 0,
                "pattern": "canonical_source_registry.yml",
                "reason": err,
            }
        )
        return {
            "status": "violation",
            "canonical_sources": [],
            "allowlisted_readers": [],
            "violations_found": violations,
            "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "registry_path": str(CANONICAL_SOURCE_REGISTRY_PATH),
            "canonical": {},
            "ui_only": {"metrics_summary": str(_ui_metrics_summary_path())},
        }

    canonical_sources_raw = payload.get("canonical_sources") or []
    allowlisted_readers_raw = payload.get("allowlisted_readers") or []
    canonical_sources: list[dict[str, Any]] = []
    for idx, entry in enumerate(canonical_sources_raw):
        normalized, violation = _normalize_registry_entry(entry)
        if normalized is not None:
            canonical_sources.append(normalized)
        elif violation is not None:
            violation = dict(violation)
            violation["line"] = idx + 2
            violations.append(violation)

    allowlisted_readers = [
        str(item).strip()
        for item in allowlisted_readers_raw
        if str(item).strip()
    ]
    if not canonical_sources:
        violations.append(
            {
                "file": str(CANONICAL_SOURCE_REGISTRY_PATH),
                "line": 0,
                "pattern": "canonical_sources",
                "reason": "registry_missing_canonical_sources",
            }
        )

    status = "clean" if not violations else "violation"
    legacy_canonical = {entry["metric"]: entry["query_or_key"] for entry in canonical_sources}
    return {
        "status": status,
        "canonical_sources": canonical_sources,
        "allowlisted_readers": allowlisted_readers,
        "violations_found": violations,
        "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(CANONICAL_SOURCE_REGISTRY_PATH),
        "canonical": legacy_canonical,
        "ui_only": {"metrics_summary": str(_ui_metrics_summary_path())},
    }


def _startup_registry_error() -> Optional[str]:
    if CANONICAL_SOURCE_REGISTRY_PATH.exists():
        return None
    return f"canonical_source_registry.yml not found at {CANONICAL_SOURCE_REGISTRY_PATH}"


def _latest_close_price(conn: sqlite3.Connection, ticker: str) -> Optional[float]:
    for table, date_col, price_col in (
        ("ohlcv_data", "date", "close"),
        ("price_history", "date", "close"),
        ("market_data", "date", "close"),
    ):
        try:
            row = conn.execute(
                f"""
                SELECT {price_col}
                FROM {table}
                WHERE UPPER(ticker) = UPPER(?)
                ORDER BY {date_col} DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
        except sqlite3.Error:
            continue
        if row and row[0] is not None:
            try:
                return float(row[0])
            except Exception:
                continue
    return None


def _exit_proximity_fraction(current_price: Optional[float], stop_loss: Optional[float], target_price: Optional[float]) -> Optional[float]:
    if current_price is None or stop_loss is None or target_price is None:
        return None
    try:
        stop = float(stop_loss)
        target = float(target_price)
        price = float(current_price)
    except Exception:
        return None
    denom = abs(target - stop)
    if denom <= 0:
        return None
    return abs(price - stop) / denom


def _query_close_rates(conn: sqlite3.Connection, audit_root: Path, reference_dt: datetime) -> dict[str, Any]:
    cutoff = (reference_dt - timedelta(days=14)).date().isoformat()
    try:
        closed_rows = conn.execute(
            """
            SELECT trade_date, ticker, ts_signal_id
            FROM production_closed_trades
            WHERE trade_date >= ?
            """,
            (cutoff,),
        ).fetchall()
    except sqlite3.Error:
        closed_rows = []

    audit_index = _scan_audit_coverage(audit_root)
    production_tsids = audit_index.get("production_tsids") or set()
    covered_closed = 0
    total_closed = 0
    for row in closed_rows:
        total_closed += 1
        tsid = str(row[2] or "").strip()
        if tsid and tsid in production_tsids:
            covered_closed += 1
    covered_daily = covered_closed / 14.0
    total_daily = total_closed / 14.0
    new_round_trip_daily = max(0.0, total_daily - covered_daily)
    return {
        "covered_lot_daily_close_rate": round(covered_daily, 6),
        "new_round_trip_daily_rate": round(new_round_trip_daily, 6),
        "covered_closed_14d": int(covered_closed),
        "total_closed_14d": int(total_closed),
    }


def _query_closed_pnl(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Closed PnL metrics from production_closed_trades (canonical view)."""
    row = conn.execute("""
        SELECT
            COUNT(*)                                          AS n_trips,
            SUM(realized_pnl)                                AS total_pnl,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS n_wins,
            SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) AS gross_profit,
            SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) AS gross_loss,
            MIN(trade_date) AS first_close,
            MAX(trade_date) AS last_close
        FROM production_closed_trades
    """).fetchone()
    if row is None or row[0] == 0:
        return {"n_trips": 0, "total_pnl": 0.0, "win_rate": None, "profit_factor": None,
                "first_close": None, "last_close": None, "source": "production_closed_trades"}
    n, pnl, wins, gp, gl, fc, lc = row
    win_rate = wins / n if n else None
    pf = (gp / gl) if gl and gl > 0 else None
    return {
        "n_trips": n,
        "total_pnl": round(float(pnl or 0), 2),
        "n_wins": wins,
        "gross_profit": round(float(gp or 0), 2),
        "gross_loss": round(float(gl or 0), 2),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_factor": round(pf, 3) if pf is not None else None,
        "first_close": fc,
        "last_close": lc,
        "source": "production_closed_trades",
    }


def _query_capital(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Capital base from portfolio_cash_state.initial_capital (single authoritative source)."""
    row = conn.execute(
        "SELECT initial_capital, cash FROM portfolio_cash_state ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {"capital": None, "cash": None, "source": "portfolio_cash_state"}
    return {
        "capital": float(row[0]),
        "cash": round(float(row[1]), 2),
        "source": "portfolio_cash_state",
    }


def _query_open_risk(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Open positions risk from trade_executions WHERE is_close=0."""
    rows = conn.execute("""
        SELECT ticker, COUNT(*) AS lots, SUM(shares * price) AS notional
        FROM trade_executions
        WHERE is_close = 0 AND COALESCE(is_synthetic, 0) = 0
        GROUP BY ticker
        ORDER BY notional DESC
    """).fetchall()
    positions = {r[0]: {"lots": r[1], "notional": round(float(r[2] or 0), 2)} for r in rows}
    total_notional = sum(p["notional"] for p in positions.values())
    return {
        "open_positions": positions,
        "total_open_notional": round(total_notional, 2),
        "source": "trade_executions_is_close_0",
    }


_FORECAST_AUDIT_ROOT = ROOT / "logs" / "forecast_audits"
_THIN_LINKAGE_THRESHOLD = 10
_WARMUP_DEADLINE = "2026-04-24"


def _scan_audit_coverage(audit_root: Path) -> Dict[str, Any]:
    """Scan the forecast-audit tree and separate production coverage from misroutes."""
    production_tsids: set[str] = set()
    tsids_by_subdir: Dict[str, set[str]] = {}
    scan_errors: list[str] = []
    non_production_parse_errors = 0
    non_production_parse_error_sample: list[str] = []
    excluded_corrupted_legacy_files = 0
    excluded_corrupted_legacy_sample: list[str] = []
    corrupted_legacy_dir_exists = (audit_root / "corrupted_legacy").exists()

    if not audit_root.exists():
        return {
            "production_tsids": production_tsids,
            "tsids_by_subdir": tsids_by_subdir,
            "scan_errors": scan_errors,
            "non_production_parse_errors": non_production_parse_errors,
            "non_production_parse_error_sample": non_production_parse_error_sample,
            "excluded_corrupted_legacy_files": excluded_corrupted_legacy_files,
            "excluded_corrupted_legacy_sample": excluded_corrupted_legacy_sample,
            "corrupted_legacy_dir_exists": corrupted_legacy_dir_exists,
            "audit_root_missing": True,
        }

    # corrupted_legacy/ holds pre-routing malformed files — skip intentionally
    _EXCLUDED_SUBDIRS = {"corrupted_legacy"}

    for f in audit_root.rglob("forecast_audit_*.json"):
        try:
            rel_parts = f.relative_to(audit_root).parts
        except Exception:
            rel_parts = ()
        if rel_parts and rel_parts[0] in _EXCLUDED_SUBDIRS:
            excluded_corrupted_legacy_files += 1
            if len(excluded_corrupted_legacy_sample) < 3:
                excluded_corrupted_legacy_sample.append(f.relative_to(audit_root).as_posix())
            continue

        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            sc = d.get("signal_context") or {}
            tsid = str(sc.get("ts_signal_id") or "").strip()
        except Exception as exc:
            try:
                rel = f.relative_to(audit_root).as_posix()
                rel_parent_for_err = f.parent.relative_to(audit_root).as_posix()
            except Exception:
                rel = f.as_posix()
                rel_parent_for_err = "unknown"
            # Only report parse errors for files in production root — errors in
            # research/ or quarantine/ don't affect THIN_LINKAGE coverage.
            if rel_parent_for_err == "production":
                scan_errors.append(f"{rel}: {exc.__class__.__name__}: {exc}")
            else:
                non_production_parse_errors += 1
                if len(non_production_parse_error_sample) < 3:
                    non_production_parse_error_sample.append(f"{rel}: {exc.__class__.__name__}: {exc}")
            continue

        if not tsid:
            continue

        try:
            rel_parent = f.parent.relative_to(audit_root).as_posix()
        except Exception:
            rel_parent = f.parent.as_posix()

        if rel_parent == "production":
            production_tsids.add(tsid)
        else:
            tsids_by_subdir.setdefault(rel_parent, set()).add(tsid)

    return {
        "production_tsids": production_tsids,
        "tsids_by_subdir": tsids_by_subdir,
        "scan_errors": scan_errors,
        "non_production_parse_errors": non_production_parse_errors,
        "non_production_parse_error_sample": non_production_parse_error_sample,
        "excluded_corrupted_legacy_files": excluded_corrupted_legacy_files,
        "excluded_corrupted_legacy_sample": excluded_corrupted_legacy_sample,
        "corrupted_legacy_dir_exists": corrupted_legacy_dir_exists,
        "audit_root_missing": False,
    }


def _query_thin_linkage(
    conn: sqlite3.Connection,
    gate_summary: Optional[Dict[str, Any]],
    audit_root: Path = _FORECAST_AUDIT_ROOT,
    generated_dt: Optional[datetime] = None,
) -> Dict[str, Any]:
    """THIN_LINKAGE countdown: classify open lots by audit file coverage.

    Each covered lot that closes gives +1 to gate matched.
    Legacy-tsid lots give 0 credit even when they close.
    """
    if generated_dt is None:
        generated_dt = datetime.now(timezone.utc)
    matched = int((gate_summary or {}).get("matched") or 0)
    base = {
        "matched_current": matched,
        "matched_threshold": _THIN_LINKAGE_THRESHOLD,
        "matched_needed": max(0, _THIN_LINKAGE_THRESHOLD - matched),
        "warmup_deadline": _WARMUP_DEADLINE,
        "status": "ok",
        "query_error": None,
        "audit_scan_errors": 0,
        "audit_scan_error_sample": [],
    }

    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(trade_executions)").fetchall()
        }
    except sqlite3.Error as exc:
        return {
            **base,
            "status": "query_error",
            "query_error": f"PRAGMA table_info(trade_executions) failed: {exc}",
            "open_lots_total": None,
            "open_lots_with_audit_coverage": None,
            "open_lots_legacy_no_coverage": None,
            "open_lots_other_no_coverage": None,
            "covered_lots_by_ticker": {},
            "pipeline_defects": {
                "canonical_tsid_lots_without_production_audit": None,
                "misrouted_audit_lots_by_subdir": {},
                "missing_audit_lots": None,
                "action_required": True,
            },
            "note": (
                "Thin-linkage countdown unavailable: could not inspect trade_executions schema. "
                "No threshold dodge — fix the schema/query path before using this snapshot."
            ),
        }

    if "ts_signal_id" not in columns:
        return {
            **base,
            "status": "schema_minimal",
            "query_error": "trade_executions.ts_signal_id column missing",
            "open_lots_total": None,
            "open_lots_with_audit_coverage": None,
            "open_lots_legacy_no_coverage": None,
            "open_lots_other_no_coverage": None,
            "covered_lots_by_ticker": {},
            "pipeline_defects": {
                "canonical_tsid_lots_without_production_audit": None,
                "misrouted_audit_lots_by_subdir": {},
                "missing_audit_lots": None,
                "action_required": True,
            },
            "note": (
                "Thin-linkage countdown unavailable: trade_executions.ts_signal_id is missing. "
                "This is explicit schema-minimal status, not an empty-linkage pass."
            ),
        }

    select_cols = ["id", "ticker", "ts_signal_id"]
    select_cols.append("stop_loss" if "stop_loss" in columns else "NULL AS stop_loss")
    select_cols.append("target_price" if "target_price" in columns else "NULL AS target_price")
    diagnostic_filter = " AND COALESCE(is_diagnostic, 0) = 0" if "is_diagnostic" in columns else ""

    try:
        rows = conn.execute(f"""
            SELECT {", ".join(select_cols)} FROM trade_executions
            WHERE is_close = 0 AND COALESCE(is_synthetic, 0) = 0
              {diagnostic_filter}
              AND ts_signal_id IS NOT NULL
        """).fetchall()
    except sqlite3.Error as exc:
        return {
            **base,
            "status": "query_error",
            "query_error": f"trade_executions open-lot query failed: {exc}",
            "open_lots_total": None,
            "open_lots_with_audit_coverage": None,
            "open_lots_legacy_no_coverage": None,
            "open_lots_other_no_coverage": None,
            "covered_lots_by_ticker": {},
            "pipeline_defects": {
                "canonical_tsid_lots_without_production_audit": None,
                "misrouted_audit_lots_by_subdir": {},
                "missing_audit_lots": None,
                "action_required": True,
            },
            "note": (
                "Thin-linkage countdown unavailable: could not read open lots. "
                "No threshold dodge — fix the query path before using this snapshot."
            ),
        }

    audit_index = _scan_audit_coverage(audit_root)
    production_tsids = audit_index["production_tsids"]
    tsids_by_subdir = audit_index["tsids_by_subdir"]
    scan_errors = audit_index["scan_errors"]
    non_production_parse_errors = audit_index["non_production_parse_errors"]
    non_production_parse_error_sample = audit_index["non_production_parse_error_sample"]
    excluded_corrupted_legacy_files = audit_index["excluded_corrupted_legacy_files"]
    excluded_corrupted_legacy_sample = audit_index["excluded_corrupted_legacy_sample"]
    corrupted_legacy_dir_exists = audit_index["corrupted_legacy_dir_exists"]
    audit_root_missing = bool(audit_index.get("audit_root_missing"))

    current_price_by_ticker: Dict[str, Optional[float]] = {}
    covered: Dict[str, Dict[str, Any]] = {}
    legacy_n = 0
    other_n = 0
    missing_n = 0
    misrouted_by_subdir: Dict[str, int] = {}
    misrouted_examples: Dict[str, Dict[str, Any]] = {}
    for row_id, ticker, tsid, stop_loss, target_price in rows:
        tsid_s = str(tsid or "").strip()
        ticker_u = str(ticker or "").upper()
        if ticker_u not in current_price_by_ticker:
            current_price_by_ticker[ticker_u] = _latest_close_price(conn, ticker_u)
        if tsid_s in production_tsids:
            bucket = covered.setdefault(
                ticker_u,
                {"count": 0, "lots_within_atr_fraction_of_exit": 0},
            )
            bucket["count"] = int(bucket.get("count") or 0) + 1
            fraction = _exit_proximity_fraction(
                current_price_by_ticker[ticker_u],
                float(stop_loss) if stop_loss is not None else None,
                float(target_price) if target_price is not None else None,
            )
            if fraction is not None and fraction <= 0.10:
                bucket["lots_within_atr_fraction_of_exit"] = int(bucket.get("lots_within_atr_fraction_of_exit") or 0) + 1
        elif tsid_s.startswith("legacy_"):
            legacy_n += 1
        else:
            other_n += 1
            found_anywhere = False
            for subdir, tsids in tsids_by_subdir.items():
                if tsid_s in tsids:
                    misrouted_by_subdir[subdir] = misrouted_by_subdir.get(subdir, 0) + 1
                    misrouted_examples.setdefault(
                        subdir,
                        {"id": row_id, "ticker": ticker_u, "ts_signal_id": tsid_s},
                    )
                    found_anywhere = True
            if not found_anywhere:
                missing_n += 1

    status = "audit_root_missing" if audit_root_missing else ("audit_scan_error" if scan_errors else "ok")
    if audit_root_missing:
        query_error = f"forecast audit root missing: {audit_root}"
    else:
        query_error = f"{len(scan_errors)} audit file parse error(s)" if scan_errors else None
    hygiene_status = (
        "missing"
        if audit_root_missing
        else "degraded"
        if excluded_corrupted_legacy_files or non_production_parse_errors or scan_errors
        else "clean"
    )
    quarantined_n = int(misrouted_by_subdir.get("production/quarantine") or 0)
    blocking_misrouted_n = sum(
        int(count or 0)
        for subdir, count in misrouted_by_subdir.items()
        if str(subdir).strip().lower() != "production/quarantine"
    )

    remediation_steps: list[str] = []
    recoverable_n = blocking_misrouted_n + missing_n
    if recoverable_n > 0:
        remediation_steps.append(
            f"Investigate {recoverable_n} canonical-tsid lot(s) without production audits before clearing THIN_LINKAGE"
        )
    elif quarantined_n > 0:
        remediation_steps.append(
            f"{quarantined_n} canonical-tsid lot(s) are quarantined; keep them out of THIN_LINKAGE and review separately"
        )
    quarantine_example = misrouted_examples.get("production/quarantine")
    if quarantine_example:
        remediation_steps.append(
            f"Verify {quarantine_example['ticker']} id={quarantine_example['id']} quarantine reason before restoring to production/"
        )
    if misrouted_examples.get("research"):
        remediation_steps.append(
            "Fix ETL routing: live-mode runs should write to production/, not research/"
        )
    if audit_root_missing:
        remediation_steps.append(
            f"Restore the forecast audit root at {audit_root} before treating thin linkage as valid"
        )

    open_lots_with_audit_coverage = sum(int(entry.get("count") or 0) for entry in covered.values())
    covered_lots_remaining = open_lots_with_audit_coverage
    rates = _query_close_rates(conn, audit_root, generated_dt)
    covered_daily_close_rate = float(rates.get("covered_lot_daily_close_rate") or 0.0)
    new_round_trip_daily_rate = float(rates.get("new_round_trip_daily_rate") or 0.0)
    days_to_deadline = max(
        0.0,
        (WARMUP_DEADLINE_UTC - generated_dt).total_seconds() / 86400.0,
    )
    expected_closes_remaining = round(covered_daily_close_rate * days_to_deadline, 6)
    shortfall = round(max(0.0, float(base["matched_needed"]) - expected_closes_remaining), 6)
    coverage_ratio = None
    if float(base["matched_needed"]) > 0:
        coverage_ratio = round(expected_closes_remaining / float(base["matched_needed"]), 6)
    # Two-level severity: warn fires early (off-track), critical fires when severely off-track.
    # advisory-only — does not block unattended gate (matched<10 already does that).
    if coverage_ratio is None:
        _cov_severity = "no_data"
    elif coverage_ratio >= float(WARMUP_COVERAGE_RATIO_WARN_THRESHOLD):
        _cov_severity = "cleared"
    elif coverage_ratio >= float(WARMUP_COVERAGE_RATIO_CRITICAL_THRESHOLD):
        _cov_severity = "warning"
    else:
        _cov_severity = "critical"
    coverage_ratio_alarm_active = _cov_severity in ("warning", "critical")
    trajectory_alarm_active = bool(days_to_deadline <= float(WARMUP_TRAJECTORY_ALERT_DAYS) and shortfall > 0)
    covered_term_days = None
    new_round_trip_term_days = None
    estimate_status = "inactive"
    if base["matched_current"] < WARMUP_MATCH_THRESHOLD:
        estimate_status = "unavailable"
        if base["matched_current"] < WARMUP_MATCH_THRESHOLD and days_to_deadline > 0:
            estimate_status = "active"
            covered_term_matches = min(float(base["matched_needed"]), float(covered_lots_remaining))
            if covered_daily_close_rate > 0:
                covered_term_days = round(covered_term_matches / covered_daily_close_rate, 6)
            if new_round_trip_daily_rate > 0:
                remaining_matches = max(0.0, float(base["matched_needed"]) - float(covered_lots_remaining))
                new_round_trip_term_days = round(remaining_matches / new_round_trip_daily_rate, 6)
    estimate_days = None
    if covered_term_days is not None or new_round_trip_term_days is not None:
        estimate_days = round(float(covered_term_days or 0.0) + float(new_round_trip_term_days or 0.0), 6)

    return {
        **base,
        "status": status,
        "query_error": query_error,
        "audit_scan_errors": len(scan_errors),
        "audit_scan_error_sample": scan_errors[:3],
        "audit_hygiene": {
            "status": hygiene_status,
            "corrupted_legacy_dir_exists": corrupted_legacy_dir_exists,
            "excluded_corrupted_legacy_files": excluded_corrupted_legacy_files,
            "excluded_corrupted_legacy_sample": excluded_corrupted_legacy_sample,
            "non_production_parse_errors": non_production_parse_errors,
            "non_production_parse_error_sample": non_production_parse_error_sample,
        },
        "open_lots_total": len(rows),
        "open_lots_with_audit_coverage": open_lots_with_audit_coverage,
        "open_lots_legacy_no_coverage": legacy_n,
        "open_lots_other_no_coverage": other_n,
        "covered_lots_by_ticker": covered,
        "covered_lots_remaining": covered_lots_remaining,
        "covered_lot_daily_close_rate": round(covered_daily_close_rate, 6),
        "new_round_trip_daily_rate": round(new_round_trip_daily_rate, 6),
        "days_to_deadline": round(days_to_deadline, 6),
        "trajectory_alarm": {
            "active": trajectory_alarm_active,
            "days_to_deadline": round(days_to_deadline, 6),
            "matched_needed": base["matched_needed"],
            "expected_closes_remaining": expected_closes_remaining,
            "shortfall": shortfall,
        },
        "coverage_ratio_alarm": {
            "active": coverage_ratio_alarm_active,
            "severity": _cov_severity,
            "ratio": coverage_ratio,
            "warn_threshold": WARMUP_COVERAGE_RATIO_WARN_THRESHOLD,
            "critical_threshold": WARMUP_COVERAGE_RATIO_CRITICAL_THRESHOLD,
            "expected_closes_remaining": expected_closes_remaining,
            "matched_needed": base["matched_needed"],
            "shortfall": shortfall,
        },
        "post_deadline_time_to_10_estimate": {
            "status": estimate_status,
            "estimated_days": estimate_days,
            "covered_lot_term_days": covered_term_days,
            "new_round_trip_term_days": new_round_trip_term_days,
            "covered_lots_remaining": covered_lots_remaining,
            "matched_needed": base["matched_needed"],
            "covered_lot_daily_close_rate": round(covered_daily_close_rate, 6),
            "new_round_trip_daily_rate": round(new_round_trip_daily_rate, 6),
        },
        # Lots with canonical tsids but no production audit file are pipeline defects.
        # Common causes: audit written to research/ (ETL run) or production/quarantine/ (hygiene sweep).
        "pipeline_defects": {
            "canonical_tsid_lots_without_production_audit": other_n,
            "misrouted_audit_lots_by_subdir": dict(sorted(misrouted_by_subdir.items())),
            "quarantined_audit_lots": quarantined_n,
            "remediation_steps": remediation_steps,
            "missing_audit_lots": missing_n,
            "action_required": blocking_misrouted_n > 0 or missing_n > 0 or bool(scan_errors) or audit_root_missing,
        },
        "note": (
            "Each covered-lot close increments matched by 1. "
            "Legacy lots (legacy_ prefix) give 0 credit when they close. "
            "canonical_tsid_lots_without_production_audit counts lots lacking a "
            "production-root audit file. If misrouted_audit_lots_by_subdir is non-empty, "
            "the audit exists in research/ or quarantine/ and must not count toward THIN_LINKAGE. "
            "Quarantined audits stay diagnostic-only and do not, by themselves, block readiness. "
            "audit_hygiene surfaces excluded corrupted_legacy files and non-production parse "
            "errors so they are visible even when they do not affect matched. "
            "The non-production parse count is intentionally flattened so consumers do not "
            "couple to directory names. "
            "No threshold dodge — code cannot force closes, only ensure valid closes count."
        ),
    }


def _read_gate_artifact() -> Optional[Dict[str, Any]]:
    """Read production_gate_latest.json if present (not always up-to-date)."""
    for gate_path in _gate_artifact_candidates():
        if not gate_path.exists():
            continue
        try:
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("_artifact_path", str(gate_path))
            return payload
    return None


def _run_utilization(db_path: Path, capital: float) -> Optional[Dict[str, Any]]:
    """Delegate to compute_capital_utilization.compute_utilization()."""
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.compute_capital_utilization import compute_utilization
        return compute_utilization(db_path, capital=capital)
    except Exception as exc:
        return {"error": str(exc)}


def _build_alpha_objective(util: Dict[str, Any]) -> Dict[str, Any]:
    roi_ann_pct = util.get("roi_ann_pct") if isinstance(util, dict) else None
    deployment_pct = util.get("deployment_pct") if isinstance(util, dict) else None
    try:
        roi_value = float(roi_ann_pct) if roi_ann_pct is not None else None
    except Exception:
        roi_value = None
    try:
        deployment_value = float(deployment_pct) if deployment_pct is not None else None
    except Exception:
        deployment_value = None

    objective_valid = bool(
        roi_value is not None
        and deployment_value is not None
        and roi_value > 0
        and deployment_value > 0
    )
    objective_score = round(float(roi_value) * float(deployment_value), 6) if objective_valid else None
    return {
        "roi_ann_pct": round(roi_value, 2) if roi_value is not None else None,
        "deployment_pct": round(deployment_value, 2) if deployment_value is not None else None,
        "objective_score": objective_score,
        "objective_valid": objective_valid,
    }


def _file_age_minutes(path: Path, reference_dt: datetime) -> Optional[float]:
    """Return the file age in minutes relative to reference_dt, or None when unavailable."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None
    age = (reference_dt - mtime).total_seconds() / 60.0
    return round(age, 2)


def _derive_evidence_health(
    *,
    freshness_status: Dict[str, Any],
    source_contract: Dict[str, Any],
    thin_linkage: Dict[str, Any],
    warmup_posture: str,
) -> str:
    """Evidence posture follows freshness -> integrity -> warmup -> clean.

    Diagnostic-only hygiene noise (for example, excluded legacy-corruption
    files that are already out of the countable path) must not downgrade the
    evidence posture on its own.
    """
    freshness = str(freshness_status.get("status") or "").strip().lower()
    if freshness != "fresh":
        return "degraded"

    if str(source_contract.get("status") or "").strip().lower() != "clean":
        return "degraded"

    defects = thin_linkage.get("pipeline_defects") or {}
    if bool(defects.get("action_required")):
        return "degraded"
    if warmup_posture == "active":
        return "bridge_state"
    if warmup_posture == "expired" and int(thin_linkage.get("matched_current") or 0) < WARMUP_MATCH_THRESHOLD:
        return "warmup_expired_fail"
    return "clean"


def _build_freshness_status(generated_dt: datetime, gate_path: Optional[Path]) -> Dict[str, Any]:
    if gate_path is None:
        return {
            "status": "missing",
            "age_minutes": None,
            "expected_max_age_minutes": None,
            "last_expected_emission_utc": None,
            "last_actual_emission_utc": None,
            "reason": "missing_gate_artifact",
        }

    actual_dt = None
    try:
        actual_dt = datetime.fromtimestamp(gate_path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        actual_dt = None

    last_expected, next_expected = _market_schedule_closes(generated_dt)
    if last_expected is None:
        fallback_max_age_minutes = 60.0
        age_minutes = None if actual_dt is None else round(max(0.0, (generated_dt - actual_dt).total_seconds() / 60.0), 2)
        if actual_dt is None:
            return {
                "status": "missing",
                "age_minutes": None,
                "expected_max_age_minutes": None,
                "last_expected_emission_utc": None,
                "last_actual_emission_utc": None,
                "reason": "missing_gate_artifact",
            }
        return {
            "status": "fresh" if age_minutes is not None and age_minutes <= fallback_max_age_minutes else "stale",
            "age_minutes": age_minutes,
            "expected_max_age_minutes": fallback_max_age_minutes,
            "last_expected_emission_utc": None,
            "last_actual_emission_utc": _to_iso_utc(actual_dt),
            "reason": "market_schedule_unavailable_fallback"
            if age_minutes is not None and age_minutes <= fallback_max_age_minutes
            else "market_schedule_unavailable",
        }

    if next_expected is None:
        expected_max_age_minutes = 24 * 60.0
    else:
        expected_max_age_minutes = max(0.0, (next_expected - last_expected).total_seconds() / 60.0)

    age_minutes = None
    if actual_dt is not None:
        age_minutes = round(max(0.0, (generated_dt - actual_dt).total_seconds() / 60.0), 2)

    last_expected_iso = _to_iso_utc(last_expected)
    status = "fresh"
    reason = None
    if actual_dt is None:
        status = "missing"
        reason = "gate_artifact_unreadable"
    elif age_minutes is None:
        status = "missing"
        reason = "gate_artifact_missing_mtime"
    elif age_minutes > expected_max_age_minutes:
        status = "stale"
        reason = "gate_artifact_older_than_market_cycle_gap"

    return {
        "status": status,
        "age_minutes": age_minutes,
        "expected_max_age_minutes": round(expected_max_age_minutes, 2),
        "last_expected_emission_utc": last_expected_iso,
        "last_actual_emission_utc": _to_iso_utc(actual_dt),
        "reason": reason,
    }


def emit_snapshot(db_path: Path) -> Dict[str, Any]:
    """Build and return the canonical snapshot dict."""
    generated_dt = datetime.now(timezone.utc)
    gate = _read_gate_artifact()
    gate_path: Optional[Path] = None
    gate_summary: Dict[str, Any] = {
        "phase3_ready": None,
        "posture": None,
        "phase3_reason": None,
        "matched": None,
        "eligible": None,
        "artifact_path": str(_gate_artifact_candidates()[0]),
        "gate_artifact_age_minutes": None,
    }
    if gate:
        gate_path = Path(gate.get("_artifact_path") or _gate_artifact_candidates()[0])
        gate_summary.update(
            {
                "phase3_ready": gate.get("phase3_ready"),
                "posture": gate.get("posture"),
                "phase3_reason": gate.get("phase3_reason"),
                "matched": gate.get("readiness", {}).get("outcome_matched")
                if isinstance(gate.get("readiness"), dict)
                else None,
                "eligible": gate.get("readiness", {}).get("outcome_eligible")
                if isinstance(gate.get("readiness"), dict)
                else None,
                "artifact_path": str(gate_path),
                "gate_artifact_age_minutes": _file_age_minutes(gate_path, generated_dt),
            }
        )

    conn = sqlite3.connect(str(db_path))
    try:
        pnl = _query_closed_pnl(conn)
        cap = _query_capital(conn)
        risk = _query_open_risk(conn)
        thin_linkage = _query_thin_linkage(conn, gate_summary, _FORECAST_AUDIT_ROOT, generated_dt)
    finally:
        conn.close()

    capital = cap.get("capital")
    util = _run_utilization(db_path, capital) if capital else {"error": "no capital"}
    source_contract = _load_source_contract()
    forecast_audit_summary: Dict[str, Any] = {}
    forecast_audit_summary_exists = FORECAST_AUDIT_SUMMARY_PATH.exists()
    if forecast_audit_summary_exists:
        try:
            loaded_forecast_summary = json.loads(FORECAST_AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded_forecast_summary, dict):
                forecast_audit_summary = loaded_forecast_summary
        except Exception:
            forecast_audit_summary = {}
    freshness_status = _build_freshness_status(
        generated_dt,
        gate_path if gate_path and gate_path.exists() else gate_path,
    )
    alpha_objective = _build_alpha_objective(util if isinstance(util, dict) else {})

    warmup_posture = "expired"
    if gate:
        posture_raw = str(gate.get("posture") or "").strip().upper()
        warmup_expired_flag = gate.get("warmup_expired")
        if warmup_expired_flag is False or str(warmup_expired_flag).strip().lower() in {"0", "false", "none", ""}:
            if posture_raw == "WARMUP_COVERED_PASS":
                warmup_posture = "active"

    gate_warmup_state = {
        "posture": warmup_posture,
        "deadline_utc": WARMUP_DEADLINE_UTC.isoformat().replace("+00:00", "Z"),
        "matched_needed": int(thin_linkage.get("matched_needed") or 0),
    }
    gate_summary["matched"] = gate_summary.get("matched") if gate_summary.get("matched") is not None else thin_linkage.get("matched_current")
    gate_summary["eligible"] = gate_summary.get("eligible") if gate_summary.get("eligible") is not None else thin_linkage.get("matched_threshold")
    gate_summary["freshness_status"] = freshness_status
    gate_summary["warmup_state"] = gate_warmup_state
    gate_summary["trajectory_alarm"] = thin_linkage.get("trajectory_alarm")
    gate_summary["coverage_ratio_alarm"] = thin_linkage.get("coverage_ratio_alarm")
    gate_summary["post_deadline_time_to_10_estimate"] = thin_linkage.get("post_deadline_time_to_10_estimate")
    evidence_health = _derive_evidence_health(
        freshness_status=freshness_status,
        source_contract=source_contract,
        thin_linkage=thin_linkage,
        warmup_posture=warmup_posture,
    )

    ann_roi = alpha_objective.get("roi_ann_pct")
    deployment_pct = alpha_objective.get("deployment_pct")
    ngn_gap_pp = round(28.0 - float(ann_roi), 2) if ann_roi is not None else None
    objective_valid = bool(alpha_objective.get("objective_valid"))
    objective_score = alpha_objective.get("objective_score")
    trajectory_alarm_active = bool((thin_linkage.get("trajectory_alarm") or {}).get("active"))
    coverage_ratio_alarm_active = bool((thin_linkage.get("coverage_ratio_alarm") or {}).get("active"))
    target_amplitude_hit_rate_raw = forecast_audit_summary.get("target_amplitude_hit_rate")
    target_amplitude_hit_rate_rolling_20_raw = forecast_audit_summary.get("target_amplitude_hit_rate_rolling_20")
    target_amplitude_support_raw = forecast_audit_summary.get("target_amplitude_support")
    target_amplitude_hit_count_raw = forecast_audit_summary.get("target_amplitude_hit_count")
    target_amplitude_hit_rate = _finite_number(target_amplitude_hit_rate_raw)
    target_amplitude_hit_rate_rolling_20 = _finite_number(target_amplitude_hit_rate_rolling_20_raw)
    target_amplitude_support = _finite_number(target_amplitude_support_raw)
    target_amplitude_hit_count = _finite_number(target_amplitude_hit_count_raw)
    if not forecast_audit_summary_exists:
        alpha_model_quality_status = "no_audit_cache"
    elif target_amplitude_support is not None and target_amplitude_support < 5:
        alpha_model_quality_status = "insufficient_audits"
    elif target_amplitude_hit_rate is not None:
        alpha_model_quality_status = "available"
    else:
        alpha_model_quality_status = "insufficient_audits"
    alpha_model_quality = {
        "status": alpha_model_quality_status,
        "value": target_amplitude_hit_rate,
        "support": int(target_amplitude_support) if target_amplitude_support is not None else None,
        "target_amplitude_hit_rate": target_amplitude_hit_rate,
        "target_amplitude_hit_rate_rolling_20": target_amplitude_hit_rate_rolling_20,
        "target_amplitude_hit_count": int(target_amplitude_hit_count) if target_amplitude_hit_count is not None else None,
        "target_amplitude_support": int(target_amplitude_support) if target_amplitude_support is not None else None,
        "forecast_audit_summary_path": str(FORECAST_AUDIT_SUMMARY_PATH),
        "forecast_audit_summary_generated_utc": forecast_audit_summary.get("generated_utc"),
        "domain_objective_version": DOMAIN_OBJECTIVE_VERSION,
    }
    gate_summary["objective_valid"] = objective_valid
    gate_summary["objective_score"] = objective_score
    gate_summary["alpha_model_quality_status"] = alpha_model_quality_status
    gate_summary["domain_objective_version"] = DOMAIN_OBJECTIVE_VERSION

    # Unattended gate status
    unattended_status = "NOT_CHECKED"
    unattended_path = ROOT / "scripts" / "institutional_unattended_gate.py"
    if unattended_path.exists():
        try:
            res = subprocess.run(
                [sys.executable, str(unattended_path), "--json"],
                capture_output=True, text=True, timeout=30,
                cwd=str(ROOT),
            )
            if res.returncode == 0:
                unattended_status = "PASS"
            else:
                unattended_status = "FAIL"
        except Exception:
            unattended_status = "ERROR"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_dt.isoformat(),
        "db_path": str(db_path),
        "domain_objective_version": DOMAIN_OBJECTIVE_VERSION,
        "source_contract": source_contract,
        # ── Canonical metric sections ──
        "closed_pnl": pnl,
        "capital": cap,
        "open_risk": risk,
        "utilization": util,
        "thin_linkage": thin_linkage,
        "alpha_objective": alpha_objective,
        "alpha_model_quality": alpha_model_quality,
        "gate": {
            **gate_summary,
            "freshness_status": freshness_status,
            "warmup_state": gate_warmup_state,
            "trajectory_alarm": thin_linkage.get("trajectory_alarm"),
            "coverage_ratio_alarm": thin_linkage.get("coverage_ratio_alarm"),
            "post_deadline_time_to_10_estimate": thin_linkage.get("post_deadline_time_to_10_estimate"),
            "objective_valid": objective_valid,
            "objective_score": objective_score,
            "alpha_model_quality_status": alpha_model_quality_status,
            "domain_objective_version": DOMAIN_OBJECTIVE_VERSION,
        },
        # ── Derived summary ──
        "summary": {
            "ann_roi_pct": ann_roi,
            "roi_ann_pct": ann_roi,
            "deployment_pct": deployment_pct,
            "objective_score": objective_score,
            "objective_valid": objective_valid,
            "alpha_model_quality": alpha_model_quality,
            "domain_objective_version": DOMAIN_OBJECTIVE_VERSION,
            "ngn_hurdle_pct": 28.0,
            "gap_to_hurdle_pp": ngn_gap_pp,
            "evidence_health": evidence_health,
            "coverage_ratio_alarm_active": coverage_ratio_alarm_active,
            "unattended_gate": unattended_status,
            "unattended_ready": bool(
                unattended_status == "PASS"
                and evidence_health == "clean"
                and objective_valid
                and freshness_status.get("status") == "fresh"
                and str(source_contract.get("status") or "").strip().lower() == "clean"
                and not trajectory_alarm_active
                # coverage_ratio_alarm is advisory-only; matched<10 already blocks via
                # unattended_status FAIL, so duplicating it here is redundant and confusing.
            ),
        },
        # ── Deprecation notice ──
        "_note": (
            "metrics_summary.json is deprecated as a source-of-truth (UI artifact only). "
            "All plans must reference the canonical snapshot for measured metrics."
        ),
    }


def _fmt_float(value: Any, pattern: str, fallback: str = "N/A") -> str:
    """Format numeric CLI fields defensively.

    Snapshot data can be partially unavailable in degraded states. The CLI must
    not short-circuit on None or string error payloads, because that hides the
    very hygiene warnings the snapshot is meant to surface.
    """
    try:
        if value is None:
            return fallback
        return format(float(value), pattern)
    except Exception:
        return fallback


def _fmt_currency(value: Any, pattern: str = ",.0f", fallback: str = "N/A") -> str:
    """Format currency-like fields defensively for CLI output."""
    try:
        if value is None:
            return fallback
        return format(float(value), pattern)
    except Exception:
        return fallback


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _fallback_emission_envelope(error: str, *, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION if partial else 0,
        "schema_version_attempted": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "emission_error": error,
        "domain_objective_version": DOMAIN_OBJECTIVE_VERSION,
        "source_contract": {
            "status": "violation",
            "canonical_sources": [],
            "allowlisted_readers": [],
            "violations_found": [
                {
                    "file": str(CANONICAL_SOURCE_REGISTRY_PATH),
                    "line": 0,
                    "pattern": "emitter_failure",
                    "reason": error,
                }
            ],
            "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "registry_path": str(CANONICAL_SOURCE_REGISTRY_PATH),
        },
    }
    if partial and isinstance(partial, dict):
        envelope.update(partial)
        envelope["schema_version"] = SCHEMA_VERSION
        envelope["schema_version_attempted"] = SCHEMA_VERSION
        envelope["emission_error"] = error
    return envelope


@click.command()
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="SQLite DB path")
@click.option("--output", default=str(DEFAULT_OUTPUT), show_default=True, help="Output JSON path")
@click.option("--json", "as_json", is_flag=True, help="Print JSON to stdout only")
def main(db: str, output: str, as_json: bool) -> None:
    startup_error = _startup_registry_error()
    if startup_error:
        snapshot = _fallback_emission_envelope(startup_error)
    else:
        try:
            snapshot = emit_snapshot(Path(db))
        except Exception as exc:  # pragma: no cover - defensive fallback envelope
            snapshot = _fallback_emission_envelope(str(exc))

    if as_json:
        print(json.dumps(snapshot, indent=2))
        return

    out_path = Path(output)
    _atomic_write_json(out_path, snapshot)

    if not isinstance(snapshot.get("summary"), dict):
        print("Canonical PMX Snapshot fallback written")
        if snapshot.get("emission_error"):
            print(f"  Emission error     : {snapshot.get('emission_error')}")
        return

    s = snapshot["summary"]
    pnl = snapshot["closed_pnl"]
    cap = snapshot["capital"]
    util = snapshot.get("utilization") or {}
    gate = snapshot.get("gate") or {}

    print("Canonical PMX Snapshot")
    print(f"  Schema version     : {snapshot['schema_version']}")
    print(
        "  Capital base       : $"
        f"{_fmt_currency(cap.get('capital'))}  (cash: ${_fmt_currency(cap.get('cash'))})"
    )
    print(
        f"  Closed trades      : {pnl['n_trips']}  "
        f"WR={_fmt_float(pnl.get('win_rate'), '.1%') if pnl.get('win_rate') is not None else 'N/A'}  "
        f"PF={_fmt_float(pnl.get('profit_factor'), '.2f') if pnl.get('profit_factor') is not None else 'N/A'}  "
        f"PnL=${_fmt_float(pnl.get('total_pnl'), '+,.2f') if pnl.get('total_pnl') is not None else 'N/A'}"
    )
    if isinstance(util, dict) and "ann_roi_pct" in util:
        ann_roi = _fmt_float(util.get("ann_roi_pct"), ".1f")
        trades_per_day = _fmt_float(util.get("trades_per_day"), ".2f")
        deployment_pct = _fmt_float(util.get("deployment_pct"), ".1f")
        print(
            f"  Ann ROI            : {ann_roi}%  "
            f"({trades_per_day} trades/day, {deployment_pct}% capital deployed/day)"
        )
    gap_pp = s.get("gap_to_hurdle_pp")
    gap_str = f"{gap_pp:+.1f}pp" if gap_pp is not None else "N/A"
    print(f"  NGN hurdle gap     : {gap_str}  (hurdle=28%)")
    _ev_health = s.get("evidence_health", "unknown")
    print(f"  Evidence health    : {_ev_health}")
    if _ev_health == "warmup_expired_fail":
        print(
            "    [recovery]       : continue accumulating -- no operator action needed."
            " Recovery is complete when evidence_health=clean."
        )
    elif _ev_health == "clean":
        print("    [recovery]       : COMPLETE (evidence_health=clean)")
    alpha = snapshot.get("alpha_objective") or {}
    print(
        "  Alpha objective    : "
        f"valid={alpha.get('objective_valid')} "
        f"roi_ann_pct={_fmt_float(alpha.get('roi_ann_pct'), '.2f')} "
        f"deployment_pct={_fmt_float(alpha.get('deployment_pct'), '.2f')} "
        f"score={_fmt_float(alpha.get('objective_score'), '.2f') if alpha.get('objective_score') is not None else 'N/A'}"
    )
    alpha_quality = snapshot.get("alpha_model_quality") or {}
    print(
        "  Alpha quality      : "
        f"status={alpha_quality.get('status')} "
        f"hit_rate={_fmt_float(alpha_quality.get('target_amplitude_hit_rate'), '.2f')} "
        f"support={alpha_quality.get('target_amplitude_support')}"
    )
    print(f"  Unattended gate    : {s['unattended_gate']}")
    if gate:
        age = gate.get("gate_artifact_age_minutes")
        age_str = f"{age:.1f}m" if isinstance(age, (int, float)) else "N/A"
        print(
            f"  Phase3 posture     : {gate.get('posture')}  "
            f"matched={gate.get('matched')}/{gate.get('eligible')}  "
            f"gate_age={age_str}"
        )
        freshness = gate.get("freshness_status") or {}
        warmup = gate.get("warmup_state") or {}
        print(
            "  Freshness          : "
            f"{freshness.get('status')} age={freshness.get('age_minutes')}m "
            f"expected_max={freshness.get('expected_max_age_minutes')}m"
        )
        print(
            "  Warmup state       : "
            f"{warmup.get('posture')} deadline={warmup.get('deadline_utc')} "
            f"need={warmup.get('matched_needed')}"
        )
        coverage_alarm = gate.get("coverage_ratio_alarm") or {}
        print(
            "  Coverage alarm     : "
            f"severity={coverage_alarm.get('severity', 'no_data')} "
            f"ratio={coverage_alarm.get('ratio')} "
            f"warn>={coverage_alarm.get('warn_threshold')} "
            f"crit>={coverage_alarm.get('critical_threshold')}"
        )
    tl = snapshot.get("thin_linkage") or {}
    if tl:
        status = tl.get("status", "ok")
        matched_current = tl.get("matched_current")
        matched_threshold = tl.get("matched_threshold")
        matched_needed = tl.get("matched_needed")
        deadline = tl.get("warmup_deadline")
        hygiene = tl.get("audit_hygiene") or {}
        if status == "ok":
            covered_str = "  ".join(
                f"{t}x{entry.get('count')}({entry.get('lots_within_atr_fraction_of_exit')} near-exit)"
                for t, entry in sorted((tl.get("covered_lots_by_ticker") or {}).items())
                if isinstance(entry, dict)
            )
            print(
                f"  THIN_LINKAGE       : {matched_current}/{matched_threshold} matched  "
                f"need {matched_needed} more  deadline={deadline}"
            )
            print(
                f"    covered lots     : {tl['open_lots_with_audit_coverage']} ({covered_str})  "
                f"legacy={tl['open_lots_legacy_no_coverage']}  "
                f"other={tl['open_lots_other_no_coverage']}"
            )
            if tl.get("trajectory_alarm"):
                ta = tl.get("trajectory_alarm") or {}
                print(
                    "    trajectory alarm : "
                    f"active={ta.get('active')} days_to_deadline={ta.get('days_to_deadline')} "
                    f"shortfall={ta.get('shortfall')}"
                )
            if tl.get("coverage_ratio_alarm"):
                ca = tl.get("coverage_ratio_alarm") or {}
                _sev = ca.get("severity", "no_data")
                print(
                    "    coverage alarm   : "
                    f"severity={_sev} ratio={ca.get('ratio')} "
                    f"warn>={ca.get('warn_threshold')} crit>={ca.get('critical_threshold')} "
                    f"shortfall={ca.get('shortfall')}"
                )
            if tl.get("post_deadline_time_to_10_estimate"):
                est = tl.get("post_deadline_time_to_10_estimate") or {}
                print(
                    "    post-deadline    : "
                    f"status={est.get('status')} estimated_days={est.get('estimated_days')} "
                    f"covered_term={est.get('covered_lot_term_days')} new_round_trip_term={est.get('new_round_trip_term_days')}"
                )
        else:
            print(
                f"  THIN_LINKAGE       : {matched_current}/{matched_threshold} matched  "
                f"need {matched_needed} more  deadline={deadline}"
            )
            print(
                f"    status           : {status}  "
                f"{tl.get('query_error') or 'see pipeline_defects'}"
            )
            if tl.get("audit_scan_errors"):
                sample = tl.get("audit_scan_error_sample") or []
                sample_str = "; ".join(sample[:2]) if sample else "n/a"
                print(
                    f"    audit scan errs  : {tl['audit_scan_errors']}  "
                    f"sample={sample_str}"
                )
        if hygiene and hygiene.get("status") != "clean":
            excluded = hygiene.get("excluded_corrupted_legacy_files")
            non_prod_parse_errors = hygiene.get("non_production_parse_errors")
            sample = hygiene.get("non_production_parse_error_sample") or []
            sample_str = "; ".join(sample[:2]) if sample else "n/a"
            print(
                f"    audit hygiene    : {hygiene.get('status')}  "
                f"excluded_legacy={excluded}  non_prod_parse_errors={non_prod_parse_errors}  "
                f"sample={sample_str}"
            )
    print(f"\nArtifact: {output}")


if __name__ == "__main__":
    main()
