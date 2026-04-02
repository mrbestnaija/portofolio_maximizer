#!/usr/bin/env python3
"""
dashboard_db_bridge.py
----------------------

Continuously renders `visualizations/dashboard_data.json` from the project's
SQLite database so `visualizations/live_dashboard.html` can stay real-time while
remaining a static HTML page (no backend required).

Default behavior is READ-ONLY against the trading DB. Optional snapshot
persisting writes into a separate audit DB to avoid contention.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path(os.getenv("PORTFOLIO_DB_PATH") or (ROOT / "data" / "portfolio_maximizer.db"))
DEFAULT_OUTPUT_PATH = ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_AUDIT_DB_PATH = ROOT / "data" / "dashboard_audit.db"
DEFAULT_ELIGIBILITY_PATH = ROOT / "logs" / "ticker_eligibility.json"
DEFAULT_CONTEXT_QUALITY_PATH = ROOT / "logs" / "context_quality_latest.json"
DEFAULT_PERFORMANCE_METRICS_PATH = ROOT / "visualizations" / "performance" / "metrics_summary.json"
DEFAULT_FORECAST_SUMMARY_PATH = ROOT / "logs" / "forecast_audits_cache" / "latest_summary.json"
DEFAULT_LIVE_DENOMINATOR_PATH = ROOT / "logs" / "overnight_denominator" / "live_denominator_latest.json"
DEFAULT_QUANT_VALIDATION_LOG_PATH = ROOT / "logs" / "signals" / "quant_validation.jsonl"
DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH = ROOT / "logs" / "automation" / "run_auto_trader_latest.json"
DEFAULT_MONITORING_CONFIG_PATH = ROOT / "config" / "forecaster_monitoring.yml"
DEFAULT_OPENCLAW_MAINTENANCE_PATH = ROOT / "logs" / "automation" / "openclaw_maintenance_latest.json"
DEFAULT_PRODUCTION_GATE_PATH = ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_LLM_ACTIVITY_DIR = ROOT / "logs" / "llm_activity"
DEFAULT_SIDECAR_MAX_AGE_MINUTES = 120
DEFAULT_DASHBOARD_EXPECTED_REFRESH_SECONDS = 60.0
DEFAULT_POSITIONS_MAX_AGE_DAYS = 14
DEFAULT_OPERATOR_ACTIVITY_DAYS = 7
DEFAULT_OPERATOR_ACTIVITY_MAX_EVENTS = 8
DASHBOARD_PAYLOAD_SCHEMA_VERSION = 2
DASHBOARD_REQUIRED_TOP_LEVEL_KEYS = (
    "meta",
    "pnl",
    "signals",
    "trade_events",
    "price_series",
    "robustness",
    "live_denominator",
    "quant_validation",
)

try:
    from integrity.sqlite_guardrails import apply_sqlite_guardrails, guarded_sqlite_connect
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from integrity.sqlite_guardrails import apply_sqlite_guardrails, guarded_sqlite_connect
    except ModuleNotFoundError:
        guardrails_path = ROOT / "integrity" / "sqlite_guardrails.py"
        spec = importlib.util.spec_from_file_location("pmx_sqlite_guardrails", guardrails_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        apply_sqlite_guardrails = module.apply_sqlite_guardrails
        guarded_sqlite_connect = module.guarded_sqlite_connect

try:
    from scripts.production_gate_contract import (
        gate_semantics_status as _gate_semantics_status,
        legacy_phase3_ready as _legacy_phase3_ready,
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from production_gate_contract import (  # type: ignore
        gate_semantics_status as _gate_semantics_status,
        legacy_phase3_ready as _legacy_phase3_ready,
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )

def _wsl_mirror_path(db_path: Path) -> Optional[Path]:
    if os.name != "posix":
        return None
    if not db_path.as_posix().startswith("/mnt/"):
        return None
    tmp_root = Path(os.environ.get("WSL_SQLITE_TMP", "/tmp"))
    return tmp_root / f"{db_path.name}.wsl"


def _select_read_path(db_path: Path) -> Path:
    mirror = _wsl_mirror_path(db_path)
    # Prefer the canonical DB path when it exists. Mirrors are a recovery tool
    # for disk I/O issues on Windows mounts, but using them as a primary read
    # target can surface partially-written/corrupted copies after abrupt exits.
    if db_path.exists():
        return db_path
    if mirror and mirror.exists():
        return mirror
    return db_path


def _connect_ro_with_fallback(db_path: Path) -> sqlite3.Connection:
    """
    Connect read-only, preferring the canonical path and only falling back to a
    WSL mirror when the canonical path cannot be opened or is corrupt.
    """
    primary = _select_read_path(db_path)
    mirror = _wsl_mirror_path(db_path)
    tried: List[Path] = []
    for candidate in (primary, mirror):
        if candidate is None:
            continue
        candidate = Path(candidate)
        if candidate in tried:
            continue
        tried.append(candidate)
        if not candidate.exists():
            continue
        try:
            conn = _connect_ro(candidate)
            # Quick corruption signal. If the DB is corrupt, we'd rather fail
            # fast (or fall back) than crash deeper in a SELECT.
            try:
                row = conn.execute("PRAGMA quick_check(1)").fetchone()
                if row and row[0] != "ok":
                    raise sqlite3.DatabaseError(str(row[0]))
            except sqlite3.DatabaseError:
                conn.close()
                raise
            return conn
        except sqlite3.DatabaseError:
            continue
    raise sqlite3.DatabaseError(
        f"Unable to open a healthy SQLite DB for dashboard rendering. Tried: {', '.join(str(p) for p in tried)}"
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_digest(payload: Dict[str, Any]) -> str:
    cloned = json.loads(json.dumps(payload, sort_keys=True))
    meta = cloned.get("meta")
    if isinstance(meta, dict):
        meta.pop("payload_digest", None)
    raw = json.dumps(cloned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    # [SETUP-PHASE BYPASS] The guardrail authorizer is a whitelist: any PRAGMA not in
    # ALLOWED_READ_PRAGMAS is denied, including busy_timeout.  Set it before guardrails
    # lock the connection, then apply guardrails immediately after.
    conn = guarded_sqlite_connect(uri, uri=True, timeout=2.0, enable_guardrails=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=2000")
    apply_sqlite_guardrails(conn, allow_schema_changes=False)
    return conn


def _connect_rw(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # [SETUP-PHASE BYPASS] journal_mode=WAL is in BLOCKED_PRAGMAS and must be set before
    # guardrails lock the connection.  apply_sqlite_guardrails() MUST be the next call after
    # the PRAGMA block -- do not insert code between them.
    conn = guarded_sqlite_connect(str(db_path), timeout=5.0, enable_guardrails=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # blocked once guardrails apply
    conn.execute("PRAGMA busy_timeout=5000")  # not blocked; set here for co-location clarity
    apply_sqlite_guardrails(conn, allow_schema_changes=False)
    return conn


def _safe_fetchall(conn: sqlite3.Connection, query: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    last_exc: Optional[Exception] = None
    for _ in range(3):
        try:
            cur = conn.execute(query, params)
            return list(cur.fetchall() or [])
        except sqlite3.OperationalError as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "no such table" in msg:
                return []
            if "locked" in msg or "busy" in msg:
                time.sleep(0.2)
                continue
            raise
    if last_exc:
        raise last_exc
    return []


def _safe_fetchone(conn: sqlite3.Connection, query: str, params: Tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
    rows = _safe_fetchall(conn, query, params)
    return rows[0] if rows else None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    try:
        rows = _safe_fetchall(conn, f"PRAGMA table_info({table_name})")
    except sqlite3.DatabaseError:
        return set()
    cols: set = set()
    for row in rows:
        try:
            cols.add(str(row["name"]))
        except Exception:
            continue
    return cols


def _default_tickers(conn: sqlite3.Connection) -> List[str]:
    tickers: List[str] = []
    try:
        import yaml  # project dependency
    except Exception:
        yaml = None  # type: ignore[assignment]

    cfg_path = ROOT / "config" / "barbell.yml"
    if yaml and cfg_path.exists():
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            b = raw.get("barbell") or {}
            for key in ("safe_bucket", "core_bucket", "speculative_bucket"):
                sym = ((b.get(key) or {}).get("symbols") or []) if isinstance(b, dict) else []
                for s in sym:
                    if isinstance(s, str) and s.strip():
                        tickers.append(s.strip().upper())
        except Exception:
            pass

    for table, col in (
        ("portfolio_positions", "ticker"),
        ("trading_signals", "ticker"),
        ("trade_executions", "ticker"),
    ):
        try:
            rows = _safe_fetchall(conn, f"SELECT DISTINCT {col} AS t FROM {table} ORDER BY {col} LIMIT 200")
            for r in rows:
                t = str(r["t"]).strip().upper()
                if t:
                    tickers.append(t)
        except Exception:
            continue

    deduped: List[str] = []
    seen = set()
    for t in tickers:
        if t.startswith("SYN") and t[3:].isdigit():
            continue
        if t.lower() in {"demo", "sample"}:
            continue
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped[:50]

def _barbell_bucket_map() -> Dict[str, str]:
    """
    Best-effort mapping of ticker -> bucket from config/barbell.yml.

    Buckets: safe, core, speculative, other
    """
    try:
        import yaml  # project dependency
    except Exception:
        return {}

    cfg_path = ROOT / "config" / "barbell.yml"
    if not cfg_path.exists():
        return {}
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    barbell = raw.get("barbell") if isinstance(raw, dict) else None
    if not isinstance(barbell, dict):
        return {}
    safe_syms = set(((barbell.get("safe_bucket") or {}).get("symbols") or []) if isinstance(barbell.get("safe_bucket"), dict) else [])
    core_syms = set(((barbell.get("core_bucket") or {}).get("symbols") or []) if isinstance(barbell.get("core_bucket"), dict) else [])
    spec_syms = set(((barbell.get("speculative_bucket") or {}).get("symbols") or []) if isinstance(barbell.get("speculative_bucket"), dict) else [])

    out: Dict[str, str] = {}
    for s in safe_syms:
        if isinstance(s, str) and s.strip():
            out[s.strip().upper()] = "safe"
    for s in core_syms:
        if isinstance(s, str) and s.strip():
            out[s.strip().upper()] = "core"
    for s in spec_syms:
        if isinstance(s, str) and s.strip():
            out[s.strip().upper()] = "speculative"
    return out


def _latest_run_id(conn: sqlite3.Connection) -> Optional[str]:
    row = _safe_fetchone(
        conn,
        """
        SELECT run_id
        FROM trade_executions
        WHERE run_id IS NOT NULL
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
    )
    if not row:
        return None
    rid = row["run_id"]
    return str(rid) if rid else None


def _load_db_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        cur = conn.execute("SELECT value FROM db_metadata WHERE key = ? LIMIT 1", (key,))
        row = cur.fetchone()
        if not row:
            return None
        val = row["value"] if isinstance(row, sqlite3.Row) else row[0]
        return str(val) if val is not None else None
    except Exception:
        return None


def _provenance_summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    ohlcv_sources: Dict[str, int] = {}
    trade_sources: Dict[str, int] = {}
    synthetic_dataset_ids: List[str] = []

    try:
        cur = conn.execute("SELECT source, COUNT(*) AS n FROM ohlcv_data GROUP BY source")
        for row in cur.fetchall():
            src = str(row["source"] or "") if isinstance(row, sqlite3.Row) else str(row[0] or "")
            if src:
                ohlcv_sources[src] = int(row["n"] if isinstance(row, sqlite3.Row) else row[1] or 0)
    except Exception:
        ohlcv_sources = {}

    try:
        cur = conn.execute("SELECT data_source, COUNT(*) AS n FROM trade_executions GROUP BY data_source")
        for row in cur.fetchall():
            src = str(row["data_source"] or "") if isinstance(row, sqlite3.Row) else str(row[0] or "")
            if src:
                trade_sources[src] = int(row["n"] if isinstance(row, sqlite3.Row) else row[1] or 0)
    except Exception:
        trade_sources = {}

    try:
        cur = conn.execute(
            """
            SELECT DISTINCT synthetic_dataset_id
            FROM trade_executions
            WHERE synthetic_dataset_id IS NOT NULL AND synthetic_dataset_id != ''
            ORDER BY synthetic_dataset_id
            """
        )
        for row in cur.fetchall():
            val = row["synthetic_dataset_id"] if isinstance(row, sqlite3.Row) else row[0]
            if val:
                synthetic_dataset_ids.append(str(val))
    except Exception:
        synthetic_dataset_ids = []

    last_run_provenance = None
    raw = _load_db_metadata(conn, "last_run_provenance")
    if raw:
        try:
            last_run_provenance = json.loads(raw)
        except Exception:
            last_run_provenance = raw

    has_synthetic = bool(
        ohlcv_sources.get("synthetic")
        or trade_sources.get("synthetic")
        or synthetic_dataset_ids
    )
    has_non_synthetic_ohlcv = any(src for src in ohlcv_sources if src and src != "synthetic")
    has_non_synthetic_trades = any(src for src in trade_sources if src and src != "synthetic")
    origin = "synthetic" if has_synthetic else "live"
    if has_synthetic and (has_non_synthetic_ohlcv or has_non_synthetic_trades):
        origin = "mixed"

    data_source = None
    execution_mode = None
    dataset_id = None
    if isinstance(last_run_provenance, dict):
        data_source = last_run_provenance.get("data_source")
        execution_mode = last_run_provenance.get("execution_mode")
        dataset_id = last_run_provenance.get("synthetic_dataset_id") or last_run_provenance.get("dataset_id")
    if not dataset_id and synthetic_dataset_ids:
        dataset_id = synthetic_dataset_ids[-1]

    return {
        "origin": origin,
        "data_source": data_source,
        "execution_mode": execution_mode,
        "dataset_id": dataset_id,
        "synthetic_dataset_ids": synthetic_dataset_ids,
        "ohlcv_sources": ohlcv_sources,
        "trade_sources": trade_sources,
        "last_run_provenance": last_run_provenance,
    }


def _positions_max_age_days() -> float:
    try:
        value = float(os.getenv("PMX_POSITIONS_MAX_AGE_DAYS", str(DEFAULT_POSITIONS_MAX_AGE_DAYS)))
    except Exception:
        value = float(DEFAULT_POSITIONS_MAX_AGE_DAYS)
    return value if value >= 0 else float(DEFAULT_POSITIONS_MAX_AGE_DAYS)


def _positions(
    conn: sqlite3.Connection,
) -> tuple:
    row = _safe_fetchone(conn, "SELECT MAX(position_date) AS d FROM portfolio_positions")
    if not row or not row["d"]:
        return _positions_from_executions(conn), False, None, "trade_executions_fallback"
    latest = str(row["d"])
    latest_ts = _parse_utc_datetime(latest)
    max_age_days = _positions_max_age_days()
    positions_stale = False
    if latest_ts is not None:
        age_days = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 86400.0
        positions_stale = age_days > max_age_days
    if positions_stale:
        return _positions_from_executions(conn), True, latest, "trade_executions_fallback_stale"
    # Backward-compatible: some environments/tests may have a minimal schema.
    query_full = """
    SELECT ticker, shares, average_cost, current_price, unrealized_pnl, unrealized_pnl_pct, market_value
    FROM portfolio_positions
    WHERE position_date = ?
    """
    query_min = """
    SELECT ticker, shares, average_cost
    FROM portfolio_positions
    WHERE position_date = ?
    """
    try:
        rows = _safe_fetchall(conn, query_full, (latest,))
    except sqlite3.OperationalError as exc:
        if "no such column" not in str(exc).lower():
            raise
        rows = _safe_fetchall(conn, query_min, (latest,))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = str(r["ticker"]).upper()
        try:
            shares = int(round(float(r["shares"] or 0.0)))
        except Exception:
            shares = 0
        try:
            entry = float(r["average_cost"]) if r["average_cost"] is not None else None
        except Exception:
            entry = None
        try:
            current_price = (
                float(r["current_price"])
                if ("current_price" in r.keys() and r["current_price"] is not None)
                else None
            )
        except Exception:
            current_price = None
        try:
            unreal_abs = (
                float(r["unrealized_pnl"])
                if ("unrealized_pnl" in r.keys() and r["unrealized_pnl"] is not None)
                else None
            )
        except Exception:
            unreal_abs = None
        try:
            unreal_pct = (
                float(r["unrealized_pnl_pct"])
                if ("unrealized_pnl_pct" in r.keys() and r["unrealized_pnl_pct"] is not None)
                else None
            )
        except Exception:
            unreal_pct = None
        try:
            market_value = (
                float(r["market_value"])
                if ("market_value" in r.keys() and r["market_value"] is not None)
                else None
            )
        except Exception:
            market_value = None
        out[t] = {"shares": shares, "entry_price": entry}
        if current_price is not None:
            out[t]["current_price"] = current_price
        if unreal_abs is not None:
            out[t]["unrealized_pnl"] = unreal_abs
        if unreal_pct is not None:
            out[t]["unrealized_pnl_pct"] = unreal_pct
        if market_value is not None:
            out[t]["market_value"] = market_value
        out[t]["status"] = "ACTIVE" if shares else "FLAT"
    return out, False, latest, "portfolio_positions"


def _latest_close(conn: sqlite3.Connection, ticker: str) -> Optional[float]:
    row = _safe_fetchone(
        conn,
        """
        SELECT close
        FROM ohlcv_data
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (ticker,),
    )
    if not row:
        return None
    try:
        return float(row["close"])
    except Exception:
        return None


def _positions_from_executions(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """Fallback position view when portfolio_positions is empty.

    Reconstructs average cost by replaying executions in time order so partial
    closes don't distort entry price. Uses latest close from ohlcv_data as
    current price when available.
    """
    cols = _table_columns(conn, "trade_executions")
    where_parts = [
        "ticker IS NOT NULL",
        "action IS NOT NULL",
    ]
    if "is_diagnostic" in cols:
        where_parts.append("is_diagnostic = 0")
    if "is_synthetic" in cols:
        where_parts.append("is_synthetic = 0")
    if "is_contaminated" in cols:
        where_parts.append("is_contaminated = 0")
    where_sql = " AND ".join(where_parts)
    query = f"""
        SELECT ticker, action, shares, price, trade_date, created_at
        FROM trade_executions
        WHERE {where_sql}
        ORDER BY COALESCE(created_at, trade_date) ASC, id ASC
    """
    rows = _safe_fetchall(conn, query)
    positions: Dict[str, Dict[str, Any]] = {}

    def _apply_trade(state: Dict[str, Any], signed_qty: float, price: float) -> None:
        pos = float(state.get("shares", 0.0) or 0.0)
        avg = state.get("entry_price")
        if avg is None:
            avg = price

        if pos == 0:
            state["shares"] = signed_qty
            state["entry_price"] = price
            return

        same_side = (pos > 0 and signed_qty > 0) or (pos < 0 and signed_qty < 0)
        if same_side:
            total = abs(pos) + abs(signed_qty)
            if total > 0:
                state["entry_price"] = (abs(pos) * avg + abs(signed_qty) * price) / total
            state["shares"] = pos + signed_qty
            return

        if abs(signed_qty) < abs(pos):
            state["shares"] = pos + signed_qty
            return

        if abs(signed_qty) == abs(pos):
            state["shares"] = 0.0
            state["entry_price"] = None
            return

        state["shares"] = pos + signed_qty
        state["entry_price"] = price

    for r in rows:
        t = str(r["ticker"]).upper()
        if not t:
            continue
        action = str(r["action"]).strip().upper()
        if action not in {"BUY", "SELL"}:
            continue
        try:
            qty = abs(float(r["shares"] or 0.0))
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue
        try:
            price = float(r["price"] or 0.0)
        except Exception:
            price = 0.0
        if price <= 0:
            continue
        state = positions.setdefault(t, {"shares": 0.0, "entry_price": None})
        signed_qty = qty if action == "BUY" else -qty
        _apply_trade(state, signed_qty, price)

    out: Dict[str, Dict[str, Any]] = {}
    for t, state in positions.items():
        shares = float(state.get("shares") or 0.0)
        if abs(shares) < 1e-6:
            continue
        entry = state.get("entry_price")
        current_price = _latest_close(conn, t)
        market_value = current_price * shares if current_price is not None else None
        unreal_abs = None
        unreal_pct = None
        if current_price is not None and entry:
            unreal_abs = (current_price - entry) * shares
            unreal_pct = (current_price / entry - 1.0) * (1 if shares > 0 else -1)
        out[t] = {
            "shares": int(round(shares)),
            "entry_price": entry,
            "status": "ACTIVE",
        }
        if current_price is not None:
            out[t]["current_price"] = current_price
        if market_value is not None:
            out[t]["market_value"] = market_value
        if unreal_abs is not None:
            out[t]["unrealized_pnl"] = unreal_abs
        if unreal_pct is not None:
            out[t]["unrealized_pnl_pct"] = unreal_pct
    return out


def _classify_trade_event(action: str, realized_pnl: Optional[float]) -> str:
    a = str(action or "").strip().upper()
    # In this project, BUY is an entry/open event; PnL is realized on exit/close.
    if a == "BUY":
        return "ENTRY"
    if a != "SELL":
        return "ENTRY" if realized_pnl is None else "EXIT_FLAT"
    try:
        v = float(realized_pnl)
    except Exception:
        return "EXIT_FLAT"
    if v > 0:
        return "EXIT_PROFIT"
    if v < 0:
        return "EXIT_LOSS"
    return "EXIT_FLAT"


def _quality_latest(conn: sqlite3.Connection, ticker: str) -> Optional[Dict[str, Any]]:
    row = _safe_fetchone(
        conn,
        """
        SELECT quality_score, missing_pct, coverage, outlier_frac, source
        FROM data_quality_snapshots
        WHERE ticker = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (ticker,),
    )
    if not row:
        return None
    try:
        return {
            "ticker": ticker,
            "quality_score": float(row["quality_score"] or 0.0),
            "missing_pct": float(row["missing_pct"] or 0.0),
            "coverage": float(row["coverage"] or 0.0),
            "outlier_frac": float(row["outlier_frac"] or 0.0),
            "source": str(row["source"] or ""),
        }
    except Exception:
        return None


def _price_series(conn: sqlite3.Connection, ticker: str, lookback_days: int) -> List[Dict[str, Any]]:
    rows = _safe_fetchall(
        conn,
        """
        SELECT date, close
        FROM ohlcv_data
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, int(max(lookback_days, 1) * 3)),
    )
    if not rows:
        return []

    pts: List[Tuple[str, float]] = []
    for r in rows:
        d = str(r["date"])
        try:
            c = float(r["close"])
        except Exception:
            continue
        pts.append((d, c))
    pts.reverse()
    return [{"t": d, "close": c} for d, c in pts]


def _latest_signals(conn: sqlite3.Connection, tickers: Iterable[str], limit: int) -> List[Dict[str, Any]]:
    ticker_list = [str(t).upper() for t in tickers if str(t).strip()]
    if not ticker_list:
        return []
    placeholders = ",".join("?" for _ in ticker_list)
    rows = _safe_fetchall(
        conn,
        f"""
        SELECT ticker, action, confidence, expected_return, source, signal_timestamp, created_at
        FROM trading_signals
        WHERE UPPER(ticker) IN ({placeholders})
        ORDER BY COALESCE(signal_timestamp, created_at) DESC, id DESC
        LIMIT ?
        """,
        tuple(ticker_list + [int(limit)]),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        t = str(r["ticker"]).upper()
        ts = r["signal_timestamp"] or r["created_at"] or _utc_now_iso()
        try:
            confidence = float(r["confidence"] or 0.0)
        except Exception:
            confidence = 0.0
        try:
            exp_ret = float(r["expected_return"] or 0.0)
        except Exception:
            exp_ret = 0.0
        sig: Dict[str, Any] = {
            "ticker": t,
            "action": str(r["action"] or "HOLD"),
            "confidence": confidence,
            "expected_return": exp_ret,
            "source": str(r["source"] or "UNKNOWN"),
            "timestamp": str(ts),
        }
        q = _quality_latest(conn, t)
        if q:
            sig["quality"] = {"quality_score": q["quality_score"]}
        out.append(sig)
    return out


def _trade_events(conn: sqlite3.Connection, tickers: Iterable[str], limit: int) -> List[Dict[str, Any]]:
    return _trade_events_filtered(conn, tickers, limit, latest_run_only=False, latest_run_id=None)


def _trade_events_filtered(
    conn: sqlite3.Connection,
    tickers: Iterable[str],
    limit: int,
    *,
    latest_run_only: bool,
    latest_run_id: Optional[str],
) -> List[Dict[str, Any]]:
    ticker_list = [str(t).upper() for t in tickers if str(t).strip()]
    if not ticker_list:
        return []
    placeholders = ",".join("?" for _ in ticker_list)
    run_clause = ""
    params_base: List[Any] = ticker_list + [int(limit)]
    if latest_run_only and latest_run_id:
        run_clause = " AND run_id = ? "
        params_base = ticker_list + [latest_run_id, int(limit)]
    trade_cols = _table_columns(conn, "trade_executions")
    preferred_cols = [
        "ticker", "action", "shares", "price", "trade_date", "created_at",
        "realized_pnl", "realized_pnl_pct", "mid_slippage_bps",
        "exit_reason", "data_source", "execution_mode",
        "barbell_bucket", "barbell_multiplier", "base_confidence", "effective_confidence",
    ]
    select_cols = [name for name in preferred_cols if not trade_cols or name in trade_cols]
    if not select_cols:
        select_cols = ["ticker", "action", "shares", "price", "trade_date", "created_at"]
    query = f"""
    SELECT {", ".join(select_cols)}
    FROM trade_executions
    WHERE UPPER(ticker) IN ({placeholders}) {run_clause}
    ORDER BY COALESCE(created_at, trade_date) DESC, id DESC
    LIMIT ?
    """
    rows = _safe_fetchall(conn, query, tuple(params_base))
    out: List[Dict[str, Any]] = []
    for r in rows:
        t = str(r["ticker"]).upper()
        ts = r["created_at"] or r["trade_date"] or _utc_now_iso()
        try:
            shares = int(round(float(r["shares"] or 0.0)))
        except Exception:
            shares = 0
        try:
            price = float(r["price"] or 0.0)
        except Exception:
            price = 0.0
        pnl_raw = r["realized_pnl"] if "realized_pnl" in r.keys() else None
        try:
            pnl: Optional[float] = float(pnl_raw) if pnl_raw is not None else None
        except Exception:
            pnl = None
        try:
            pnl_pct = float(r["realized_pnl_pct"]) if r["realized_pnl_pct"] is not None else None
        except Exception:
            pnl_pct = None
        try:
            slip_bps = float(r["mid_slippage_bps"] or 0.0)
            slippage = slip_bps / 10000.0
        except Exception:
            slippage = 0.0
        action = str(r["action"] or "")
        event_type = _classify_trade_event(action, pnl)
        out.append(
            {
                "ticker": t,
                "action": action,
                "event_type": event_type,
                "shares": shares,
                "price": price,
                "timestamp": str(ts),
                "bar_timestamp": str(r["trade_date"]) if r["trade_date"] else None,
                "realized_pnl": pnl,
                "realized_pnl_pct": pnl_pct,
                "exit_reason": (
                    str(r["exit_reason"])
                    if ("exit_reason" in r.keys() and r["exit_reason"] is not None)
                    else None
                ),
                "slippage": float(slippage),
                "data_source": str(r["data_source"] or "") if ("data_source" in r.keys()) else "",
                "execution_mode": str(r["execution_mode"] or "") if ("execution_mode" in r.keys()) else "",
                "barbell_bucket": str(r["barbell_bucket"] or "") if ("barbell_bucket" in r.keys()) else "",
                "barbell_multiplier": float(r["barbell_multiplier"]) if ("barbell_multiplier" in r.keys() and r["barbell_multiplier"] is not None) else None,
                "base_confidence": float(r["base_confidence"]) if ("base_confidence" in r.keys() and r["base_confidence"] is not None) else None,
                "effective_confidence": float(r["effective_confidence"]) if ("effective_confidence" in r.keys() and r["effective_confidence"] is not None) else None,
            }
        )
    out.reverse()
    return out


def _latest_performance(conn: sqlite3.Connection) -> Dict[str, Any]:
    # Phase 7.9: Prefer canonical metrics from PnL integrity enforcer (CHECK constraints)
    # Falls back to performance_metrics table if integrity module unavailable
    try:
        # Get DB path from connection
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        canonical = _canonical_metrics_pnl_integrity(db_path)
        if canonical:
            canonical["performance_unknown"] = False
            canonical["performance_source"] = "pnl_integrity_enforcer"
            return canonical
    except Exception:
        pass  # Fall back to performance_metrics table

    try:
        row = _safe_fetchone(
            conn,
            """
            SELECT total_return, total_return_pct, win_rate, profit_factor, num_trades,
                   sharpe_ratio, sortino_ratio, max_drawdown,
                   avg_win, avg_loss, largest_win, largest_loss
            FROM performance_metrics
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
        )
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            row = None
        elif "no such column" in str(exc).lower():
            row = _safe_fetchone(
                conn,
                """
                SELECT total_return, total_return_pct, win_rate, profit_factor, num_trades
                FROM performance_metrics
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
            )
        else:
            raise
    if not row:
        return {
            "pnl_abs": None,
            "pnl_pct": None,
            "win_rate": None,
            "profit_factor": None,
            "trade_count": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "avg_win": None,
            "avg_loss": None,
            "largest_win": None,
            "largest_loss": None,
            "performance_unknown": True,
            "performance_source": "performance_metrics_missing",
        }
    def _f(key: str) -> Optional[float]:
        try:
            value = row[key]
            if value is None:
                return None
            return float(value)
        except Exception:
            return None
    def _i(key: str) -> Optional[int]:
        try:
            value = row[key]
            if value is None:
                return None
            return int(value)
        except Exception:
            return None
    pnl_abs = _f("total_return")
    pnl_pct = _f("total_return_pct")
    win_rate = _f("win_rate")
    profit_factor = _f("profit_factor")
    trade_count = _i("num_trades")
    return {
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trade_count": trade_count,
        "sharpe": _f("sharpe_ratio"),
        "sortino": _f("sortino_ratio"),
        "max_drawdown": _f("max_drawdown"),
        "avg_win": _f("avg_win"),
        "avg_loss": _f("avg_loss"),
        "largest_win": _f("largest_win"),
        "largest_loss": _f("largest_loss"),
        "performance_unknown": False,
        "performance_source": "performance_metrics",
    }


def _canonical_metrics_pnl_integrity(db_path: str) -> Dict[str, Any]:
    """
    Get canonical PnL metrics using PnLIntegrityEnforcer (Phase 7.9).

    Uses production_closed_trades view (is_close=1, not diagnostic, not synthetic).
    This is the SINGLE SOURCE OF TRUTH for PnL reporting with database-level
    CHECK constraints enforcing invariants.

    Returns dict compatible with _latest_performance output format.
    """
    try:
        from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer

        with PnLIntegrityEnforcer(db_path, auto_create_views=False) as enforcer:
            m = enforcer.get_canonical_metrics()
            return {
                "pnl_abs": m.total_realized_pnl,
                "pnl_pct": 0.0,  # Need initial capital to compute
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "trade_count": m.total_round_trips,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "avg_win": m.avg_win,
                "avg_loss": m.avg_loss,
                "largest_win": m.largest_win,
                "largest_loss": m.largest_loss,
                # Additional integrity metrics
                "diagnostic_excluded": m.diagnostic_trades_excluded,
                "synthetic_excluded": m.synthetic_trades_excluded,
                "double_count_violations": m.opening_legs_with_pnl,
            }
    except ImportError:
        # Fallback if integrity module not available
        return {}
    except Exception:
        return {}


def _quant_validation_payload() -> Dict[str, Any]:
    path = DEFAULT_QUANT_VALIDATION_LOG_PATH
    if not path.exists():
        return {
            "status": "MISSING",
            "path": str(path),
            "warnings": ["quant_validation_missing"],
            "total": 0,
            "pass_count": 0,
            "fail_count": 0,
            "fail_fraction": 0.0,
            "negative_expected_profit_count": 0,
            "negative_expected_profit_fraction": 0.0,
            "age_minutes": None,
        }

    try:
        from scripts.check_quant_validation_health import (
            _load_entries,
            _load_monitoring_cfg,
            _summarize_global,
        )

        entries = _load_entries(path)
        monitoring_cfg = _load_monitoring_cfg(DEFAULT_MONITORING_CONFIG_PATH)
        qv_cfg = monitoring_cfg.get("quant_validation") or {}
        max_fail_frac = float(qv_cfg.get("max_fail_fraction", 0.85))  # Phase 7.14: 0.95→0.85
        max_neg_exp_frac = float(qv_cfg.get("max_negative_expected_profit_fraction", 0.50))
        warn_fail_frac = float(qv_cfg.get("warn_fail_fraction", max_fail_frac))
        warn_neg_exp_frac = float(
            qv_cfg.get("warn_negative_expected_profit_fraction", max_neg_exp_frac)
        )

        summary = _summarize_global(entries)
        violations: List[str] = []
        if summary.fail_fraction > max_fail_frac:
            violations.append("FAIL_fraction_exceeds_max")
        if summary.negative_expected_profit_fraction > max_neg_exp_frac:
            violations.append("negative_expected_profit_fraction_exceeds_max")

        if violations:
            status = "RED"
        elif (
            summary.fail_fraction > warn_fail_frac
            or summary.negative_expected_profit_fraction > warn_neg_exp_frac
        ):
            status = "YELLOW"
        else:
            status = "GREEN"

        age = _sidecar_age_minutes(path, {})
        return {
            "status": status,
            "path": str(path),
            "warnings": violations,
            "total": int(summary.total),
            "pass_count": int(summary.pass_count),
            "fail_count": int(summary.fail_count),
            "fail_fraction": float(summary.fail_fraction),
            "negative_expected_profit_count": int(summary.negative_expected_profit_count),
            "negative_expected_profit_fraction": float(summary.negative_expected_profit_fraction),
            "age_minutes": round(age, 2) if age is not None else None,
        }
    except SystemExit as exc:
        return {
            "status": "ERROR",
            "path": str(path),
            "warnings": [f"quant_validation_error:{exc}"],
            "total": 0,
            "pass_count": 0,
            "fail_count": 0,
            "fail_fraction": 0.0,
            "negative_expected_profit_count": 0,
            "negative_expected_profit_fraction": 0.0,
            "age_minutes": None,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "path": str(path),
            "warnings": [f"quant_validation_error:{exc}"],
            "total": 0,
            "pass_count": 0,
            "fail_count": 0,
            "fail_fraction": 0.0,
            "negative_expected_profit_count": 0,
            "negative_expected_profit_fraction": 0.0,
            "age_minutes": None,
        }


def _operator_issue(
    title: str,
    detail: str,
    severity: str,
    focus: str,
    *,
    meta: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "title": str(title or "").strip(),
        "detail": str(detail or "").strip(),
        "severity": str(severity or "INFO").upper(),
        "focus": str(focus or "runtime").strip().lower(),
        "meta": str(meta or "").strip(),
    }


def _iter_recent_activity_files(activity_dir: Path, days: int) -> List[Path]:
    if not activity_dir.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(days)))
    files: List[Path] = []
    for path in sorted(activity_dir.glob("*.jsonl")):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            continue
        if modified >= cutoff:
            files.append(path)
    return files


def _recent_llm_activity_payload() -> Dict[str, Any]:
    activity_dir = DEFAULT_LLM_ACTIVITY_DIR
    days = int(os.getenv("PMX_OPERATOR_ACTIVITY_DAYS", str(DEFAULT_OPERATOR_ACTIVITY_DAYS)) or DEFAULT_OPERATOR_ACTIVITY_DAYS)
    max_events = int(
        os.getenv("PMX_OPERATOR_ACTIVITY_MAX_EVENTS", str(DEFAULT_OPERATOR_ACTIVITY_MAX_EVENTS))
        or DEFAULT_OPERATOR_ACTIVITY_MAX_EVENTS
    )
    files = _iter_recent_activity_files(activity_dir, days=max(1, days))
    if not files:
        return {
            "status": "MISSING",
            "path": str(activity_dir),
            "days": max(1, days),
            "files_scanned": 0,
            "total_events": 0,
            "parse_errors": 0,
            "counts_by_type": {},
            "counts_by_channel": {},
            "counts_by_event_type": {},
            "short_circuit_events": 0,
            "tool_calls": 0,
            "recent_events": [],
        }

    by_type: Counter[str] = Counter()
    by_channel: Counter[str] = Counter()
    by_event_type: Counter[str] = Counter()
    recent_events: List[Dict[str, Any]] = []
    parse_errors = 0

    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            parse_errors += 1
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                parse_errors += 1
                continue
            if not isinstance(payload, dict):
                continue
            entry_type = str(payload.get("type") or "unknown").strip() or "unknown"
            by_type[entry_type] += 1
            channel = str(payload.get("channel") or "").strip().lower()
            if channel:
                by_channel[channel] += 1
            event_type = str(payload.get("event_type") or "").strip()
            if event_type:
                by_event_type[event_type] += 1
            recent_events.append(
                {
                    "timestamp": str(payload.get("timestamp") or "").strip(),
                    "type": entry_type,
                    "channel": channel or None,
                    "event_type": event_type or None,
                    "latency_ms": payload.get("latency_ms"),
                    "session_id": payload.get("session_id"),
                }
            )

    def _event_sort_key(item: Dict[str, Any]) -> tuple[float, str]:
        parsed = _parse_utc_datetime(item.get("timestamp"))
        return ((parsed.timestamp() if parsed else 0.0), str(item.get("timestamp") or ""))

    recent_events.sort(key=_event_sort_key, reverse=True)
    short_circuit_events = sum(
        count for name, count in by_event_type.items() if "fast_path" in name or "short_circuit" in name
    )
    tool_calls = int(by_type.get("tool_call", 0))
    status = "OK" if recent_events else ("WARN" if files else "MISSING")
    return {
        "status": status,
        "path": str(activity_dir),
        "days": max(1, days),
        "files_scanned": len(files),
        "total_events": sum(by_type.values()),
        "parse_errors": int(parse_errors),
        "counts_by_type": dict(sorted(by_type.items())),
        "counts_by_channel": dict(sorted(by_channel.items())),
        "counts_by_event_type": dict(sorted(by_event_type.items())),
        "short_circuit_events": int(short_circuit_events),
        "tool_calls": tool_calls,
        "recent_events": recent_events[: max(1, max_events)],
    }


def _operator_console_payload() -> Dict[str, Any]:
    maintenance, maintenance_err = _load_sidecar_json(DEFAULT_OPENCLAW_MAINTENANCE_PATH)
    production_gate, production_gate_err = _load_sidecar_json(DEFAULT_PRODUCTION_GATE_PATH)
    activity = _recent_llm_activity_payload()

    maintenance_step = maintenance.get("steps", {}) if isinstance(maintenance, dict) else {}
    fast_supervisor = maintenance_step.get("fast_supervisor", {}) if isinstance(maintenance_step, dict) else {}
    session_route = maintenance_step.get("session_route_reconcile", {}) if isinstance(maintenance_step, dict) else {}
    gateway_health = maintenance_step.get("gateway_health", {}) if isinstance(maintenance_step, dict) else {}
    broken_channel_disable = maintenance_step.get("broken_channel_disable", {}) if isinstance(maintenance_step, dict) else {}
    channels_snapshot = maintenance_step.get("channels_status_snapshot", {}) if isinstance(maintenance_step, dict) else {}

    profitability = production_gate.get("profitability_proof", {}) if isinstance(production_gate, dict) else {}
    lift_gate = production_gate.get("lift_gate", {}) if isinstance(production_gate, dict) else {}
    readiness = production_gate.get("readiness", {}) if isinstance(production_gate, dict) else {}
    production_profitability_gate = (
        production_gate.get("production_profitability_gate", {})
        if isinstance(production_gate, dict)
        else {}
    )
    repo_state = production_gate.get("repo_state", {}) if isinstance(production_gate, dict) else {}
    repo_status = repo_state.get("status", {}) if isinstance(repo_state, dict) else {}

    maintenance_age = _sidecar_age_minutes(DEFAULT_OPENCLAW_MAINTENANCE_PATH, maintenance) if isinstance(maintenance, dict) else None
    gate_age = _sidecar_age_minutes(DEFAULT_PRODUCTION_GATE_PATH, production_gate) if isinstance(production_gate, dict) else None
    primary_channel_name = str(maintenance.get("primary_channel") or "whatsapp") if isinstance(maintenance, dict) else "whatsapp"
    snapshot_channels = channels_snapshot.get("channels", {}) if isinstance(channels_snapshot, dict) else {}
    primary_snapshot = snapshot_channels.get(primary_channel_name, {}) if isinstance(snapshot_channels, dict) else {}
    gateway_warnings = [str(x) for x in gateway_health.get("warnings", [])] if isinstance(gateway_health, dict) else []
    fast_supervisor_warnings = [str(x) for x in fast_supervisor.get("warnings", [])] if isinstance(fast_supervisor, dict) else []
    recovery_events: List[str] = []
    if (
        str(fast_supervisor.get("action") or "") == "soft_timeout_skip"
        or str(fast_supervisor.get("reason") or "") == "channels_status_timeout_softened"
    ):
        recovery_events.append("channels_status_timeout_softened")
    if (
        str(fast_supervisor.get("action") or "") == "gateway_restart_triggered"
        and not gateway_health.get("primary_channel_issue_final")
    ):
        recovery_events.append("gateway_restart_recovered")
    if (
        str(gateway_health.get("primary_channel_issue") or "") == "whatsapp_handshake_timeout"
        and not gateway_health.get("primary_channel_issue_final")
    ):
        recovery_events.append("whatsapp_handshake_recovered")
    if "gateway_detached_listener_conflict" in gateway_warnings:
        recovery_events.append("gateway_detached_listener_conflict")
    recovery_mode = recovery_events[0] if recovery_events else "steady_state"

    maintenance_summary = {
        "status": str(maintenance.get("status") or ("MISSING" if maintenance_err else "UNKNOWN")).upper(),
        "warning_count": len(maintenance.get("warnings", []) if isinstance(maintenance, dict) else []),
        "error_count": len(maintenance.get("errors", []) if isinstance(maintenance, dict) else []),
        "warnings": [str(x) for x in (maintenance.get("warnings", []) if isinstance(maintenance, dict) else [])],
        "errors": [str(x) for x in (maintenance.get("errors", []) if isinstance(maintenance, dict) else [])],
        "age_minutes": round(maintenance_age, 2) if maintenance_age is not None else None,
        "primary_channel": primary_channel_name,
        "bound_agent_id": str(session_route.get("bound_agent_id") or ""),
        "expected_model": str(session_route.get("expected_model") or ""),
        "duplicate_wrong_agent_keys": int(session_route.get("duplicate_wrong_agent_keys") or 0),
        "refreshed_bound_keys": int(session_route.get("refreshed_bound_keys") or 0),
        "updated_agents": [str(x) for x in session_route.get("updated_agents", [])] if isinstance(session_route, dict) else [],
        "gateway_rpc_ok": gateway_health.get("rpc_ok"),
        "gateway_service_status": gateway_health.get("service_status"),
        "gateway_listener_pid": gateway_health.get("gateway_listener_pid"),
        "fast_supervisor_action": str(fast_supervisor.get("action") or ""),
        "fast_supervisor_reason": str(fast_supervisor.get("reason") or ""),
        "fast_supervisor_warnings": fast_supervisor_warnings,
        "primary_channel_issue": str(gateway_health.get("primary_channel_issue") or ""),
        "primary_channel_issue_after_restart": str(gateway_health.get("primary_channel_issue_after_restart") or ""),
        "primary_channel_issue_final": str(gateway_health.get("primary_channel_issue_final") or ""),
        "recovery_mode": recovery_mode,
        "recovery_events": recovery_events,
        "reconnect_attempts": int(primary_snapshot.get("reconnectAttempts") or 0),
        "last_connected_at": primary_snapshot.get("lastConnectedAt"),
        "last_event_at": primary_snapshot.get("lastEventAt"),
        "last_error": str(primary_snapshot.get("lastError") or ""),
        "last_disconnect": primary_snapshot.get("lastDisconnect") if isinstance(primary_snapshot.get("lastDisconnect"), dict) else {},
        "broken_channels_disabled": [str(x) for x in broken_channel_disable.get("disabled", [])]
        if isinstance(broken_channel_disable, dict)
        else [],
    }

    evidence_progress = profitability.get("evidence_progress", {}) if isinstance(profitability, dict) else {}
    production_summary = {
        "status": str(production_profitability_gate.get("status") or ("MISSING" if production_gate_err else "UNKNOWN")).upper(),
        "proof_status": str(profitability.get("status") or "UNKNOWN").upper(),
        "lift_status": str(lift_gate.get("status") or "UNKNOWN").upper(),
        "gate_semantics_status": _gate_semantics_status(production_gate),
        "phase3_ready": bool(_phase3_strict_ready(production_gate)),
        "phase3_reason": _phase3_strict_reason(production_gate),
        "phase3_legacy_ready": bool(_legacy_phase3_ready(production_gate)),
        "remaining_trading_days": int(evidence_progress.get("remaining_trading_days") or 0),
        "remaining_closed_trades": int(evidence_progress.get("remaining_closed_trades") or 0),
        "closed_trades": int(profitability.get("closed_trades") or 0),
        "profit_factor": profitability.get("profit_factor"),
        "lift_fraction": lift_gate.get("lift_fraction"),
        "min_lift_fraction": lift_gate.get("min_lift_fraction"),
        "warmup_expired": bool(production_gate.get("warmup_expired", False)),
        "repo_dirty_tracked": int(repo_status.get("tracked_changed") or 0),
        "repo_dirty_untracked": int(repo_status.get("untracked") or 0),
        "age_minutes": round(gate_age, 2) if gate_age is not None else None,
    }

    issues: List[Dict[str, Any]] = []
    if maintenance_err:
        issues.append(
            _operator_issue(
                "OpenClaw maintenance artifact missing",
                f"Unable to load {DEFAULT_OPENCLAW_MAINTENANCE_PATH.name} ({maintenance_err}).",
                "WARN",
                "runtime",
            )
        )
    else:
        if maintenance_summary["duplicate_wrong_agent_keys"] > 0:
            issues.append(
                _operator_issue(
                    "Session route reconciliation corrected mismatched WhatsApp wiring",
                    (
                        f"duplicate_wrong_agent_keys={maintenance_summary['duplicate_wrong_agent_keys']} "
                        f"for {maintenance_summary['primary_channel']} -> {maintenance_summary['bound_agent_id']}; "
                        f"expected_model={maintenance_summary['expected_model'] or '--'}."
                    ),
                    "WARN",
                    "wiring",
                    meta="stale direct-session ownership was reset",
                )
            )
        if maintenance_summary["refreshed_bound_keys"] > 0:
            issues.append(
                _operator_issue(
                    "Bound sessions were refreshed due to stale model metadata",
                    f"refreshed_bound_keys={maintenance_summary['refreshed_bound_keys']}.",
                    "WARN",
                    "wiring",
                )
            )
        if maintenance_summary["warnings"]:
            issues.append(
                _operator_issue(
                    "Maintenance completed with warnings",
                    ", ".join(maintenance_summary["warnings"]),
                    "WARN",
                    "runtime",
                )
            )
        if maintenance_summary["errors"]:
            issues.append(
                _operator_issue(
                    "Maintenance recorded hard errors",
                    ", ".join(maintenance_summary["errors"]),
                    "ERROR",
                    "runtime",
                )
            )
        if gateway_health.get("rpc_ok") is False:
            issues.append(
                _operator_issue(
                    "Gateway RPC is unavailable",
                    "OpenClaw gateway did not respond to the maintenance probe.",
                    "ERROR",
                    "runtime",
                )
            )
        if maintenance_summary["recovery_events"]:
            issues.append(
                _operator_issue(
                    "Control-plane recovery evidence captured",
                    (
                        f"recovery_mode={maintenance_summary['recovery_mode']} "
                        f"fast_action={maintenance_summary['fast_supervisor_action'] or '--'} "
                        f"reconnect_attempts={maintenance_summary['reconnect_attempts']}."
                    ),
                    "INFO",
                    "runtime",
                    meta=", ".join(maintenance_summary["recovery_events"]),
                )
            )
        if "gateway_detached_listener_conflict" in gateway_warnings:
            issues.append(
                _operator_issue(
                    "Gateway listener conflict was softened instead of treated as hard-down",
                    (
                        f"service_status={maintenance_summary['gateway_service_status'] or '--'} "
                        f"listener_pid={maintenance_summary['gateway_listener_pid'] or '--'}."
                    ),
                    "WARN",
                    "runtime",
                    meta="false-down probe noise should not trigger threshold dodge",
                )
            )

    if production_gate_err:
        issues.append(
            _operator_issue(
                "Production gate artifact missing",
                f"Unable to load {DEFAULT_PRODUCTION_GATE_PATH.name} ({production_gate_err}).",
                "WARN",
                "gate",
            )
        )
    else:
        if production_summary["status"] not in {"PASS", "OK"}:
            issues.append(
                _operator_issue(
                    "Production gate remains fail-closed",
                    (
                        f"gate={production_summary['status']} proof={production_summary['proof_status']} "
                        f"lift={production_summary['lift_status']} "
                        f"remaining_days={production_summary['remaining_trading_days']}."
                    ),
                    "ERROR",
                    "gate",
                    meta=f"phase3_reason={production_summary['phase3_reason'] or '--'}",
                )
            )
        elif not production_summary["phase3_ready"]:
            issues.append(
                _operator_issue(
                    "Phase 3 is not yet ready",
                    f"phase3_reason={production_summary['phase3_reason'] or '--'}.",
                    "WARN",
                    "gate",
                )
            )
        if not production_summary["warmup_expired"] and production_summary["remaining_trading_days"] > 0:
            issues.append(
                _operator_issue(
                    "Profitability proof is still in warm-up",
                    (
                        f"remaining_trading_days={production_summary['remaining_trading_days']} "
                        f"and remaining_closed_trades={production_summary['remaining_closed_trades']}."
                    ),
                    "INFO",
                    "gate",
                    meta="fail-closed until proof window matures; this prevents threshold dodge",
                )
            )
        if production_summary["repo_dirty_tracked"] or production_summary["repo_dirty_untracked"]:
            issues.append(
                _operator_issue(
                    "Repo state is dirty during operator review",
                    (
                        f"tracked_changed={production_summary['repo_dirty_tracked']} "
                        f"untracked={production_summary['repo_dirty_untracked']}."
                    ),
                    "INFO",
                    "gate",
                )
            )

    if activity["status"] == "MISSING":
        issues.append(
            _operator_issue(
                "No recent LLM activity log files were found",
                f"Expected JSONL activity under {activity['path']}.",
                "WARN",
                "activity",
            )
        )
    else:
        if activity["short_circuit_events"] > 0:
            issues.append(
                _operator_issue(
                    "Bridge fast-path short-circuit is active",
                    (
                        f"short_circuit_events={activity['short_circuit_events']} "
                        f"across total_events={activity['total_events']} in the last {activity['days']} day(s)."
                    ),
                    "INFO",
                    "activity",
                )
            )
        if activity["parse_errors"] > 0:
            issues.append(
                _operator_issue(
                    "Recent activity logs had parse errors",
                    f"parse_errors={activity['parse_errors']}.",
                    "WARN",
                    "activity",
                )
            )

    severity_rank = {"INFO": 1, "WARN": 2, "ERROR": 3}
    overall_status = "OK"
    if any(severity_rank.get(str(issue.get("severity", "")).upper(), 0) >= 3 for issue in issues):
        overall_status = "ERROR"
    elif any(severity_rank.get(str(issue.get("severity", "")).upper(), 0) >= 2 for issue in issues):
        overall_status = "WARN"
    elif maintenance_err or production_gate_err or activity["status"] != "OK":
        overall_status = "WARN"

    issues.sort(key=lambda item: severity_rank.get(str(item.get("severity", "")).upper(), 0), reverse=True)
    return {
        "status": overall_status,
        "generated_utc": _utc_now_iso(),
        "maintenance": maintenance_summary,
        "production_gate": production_summary,
        "activity": activity,
        "issues": issues[:12],
    }


def build_dashboard_payload(
    *,
    conn: sqlite3.Connection,
    tickers: List[str],
    lookback_days: int,
    max_signals: int,
    max_trades: int,
    latest_run_only: bool = True,
    db_path: Optional[Path] = None,
    read_path: Optional[Path] = None,
    mirror_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    perf = _latest_performance(conn)
    positions, positions_stale, positions_asof, positions_source = _positions(conn)
    run_id = _latest_run_id(conn) or "db_bridge"
    ts = _utc_now_iso()
    bucket_map = _barbell_bucket_map()
    model_params = _model_params(conn)
    checks = _data_checks(conn)
    if positions_stale:
        checks.append(
            f"portfolio_positions stale (as_of={positions_asof},"
            f" max_age_days={_positions_max_age_days():.0f}); using filtered trade_executions fallback."
        )
    provenance = _provenance_summary(conn)
    quant_validation = _quant_validation_payload()

    qual_records: List[Dict[str, Any]] = []
    for t in tickers:
        rec = _quality_latest(conn, t)
        if rec:
            qual_records.append(rec)
    avg_q = sum(r["quality_score"] for r in qual_records) / len(qual_records) if qual_records else 0.0
    min_q = min((r["quality_score"] for r in qual_records), default=0.0)

    def _opt_float(raw: Any) -> Optional[float]:
        try:
            if raw is None:
                return None
            value = float(raw)
            return value if value == value else None  # reject NaN
        except Exception:
            return None

    def _opt_int(raw: Any) -> Optional[int]:
        try:
            if raw is None:
                return None
            return int(raw)
        except Exception:
            return None

    payload: Dict[str, Any] = {
        # NOTE: keep payload self-contained for the static HTML dashboard.
        "meta": {
            "run_id": str(run_id),
            "ts": ts,
            "generated_utc": ts,
            "tickers": tickers,
            "cycles": None,
            "llm_enabled": None,
            "dashboard_version": "db_bridge_v1",
            "payload_schema_version": DASHBOARD_PAYLOAD_SCHEMA_VERSION,
            "payload_required_sections": list(DASHBOARD_REQUIRED_TOP_LEVEL_KEYS),
            "data_origin": provenance.get("origin"),
            "dataset_id": provenance.get("dataset_id"),
            "data_source": provenance.get("data_source"),
            "execution_mode": provenance.get("execution_mode"),
            "provenance": provenance,
            "storage": {
                "db_path": str(db_path) if db_path else None,
                "read_path": str(read_path) if read_path else None,
                "mirror_path": str(mirror_path) if mirror_path else None,
                "using_mirror": bool(read_path and mirror_path and Path(read_path) == Path(mirror_path)),
                "output_path": str(output_path) if output_path else None,
            },
            "scope": [
                "Close",
                "Entry",
                "Exit-Profit",
                "Exit-Loss",
                "Active",
                "Closed-Profit",
                "Closed-Loss",
                "Open",
            ],
            "ticker_buckets": {t: bucket_map.get(t, "other") for t in tickers},
        },
        "pnl": {
            "absolute": _opt_float(perf.get("pnl_abs")),
            "pct": _opt_float(perf.get("pnl_pct")),
        },
        "win_rate": _opt_float(perf.get("win_rate")),
        "trade_count": _opt_int(perf.get("trade_count")),
        "performance_unknown": bool(perf.get("performance_unknown", False)),
        "performance": perf,
        "positions": positions,
        "positions_stale": positions_stale,
        "positions_asof": positions_asof,
        "positions_source": positions_source,
        "latency": {"ts_ms": None, "llm_ms": None},
        "routing": {"ts_signals": 0, "llm_signals": 0, "fallback_used": 0},
        "quality": {"average": float(avg_q), "minimum": float(min_q), "records": qual_records},
        "equity": [],
        "equity_realized": [],
        "signals": _latest_signals(conn, tickers, max_signals),
        "trade_events": _trade_events_filtered(
            conn,
            tickers,
            max_trades,
            latest_run_only=latest_run_only,
            latest_run_id=run_id if run_id else None,
        ),
        "price_series": {t: _price_series(conn, t, lookback_days) for t in tickers},
        "model_params": model_params,
        "checks": checks,
        "robustness": _robustness_payload(),
        "live_denominator": _live_denominator_payload(),
        "quant_validation": quant_validation,
        "operator_console": _operator_console_payload(),
    }
    payload["meta"]["payload_digest"] = _payload_digest(payload)
    return payload


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_sidecar_json(path: Path) -> tuple[Dict[str, Any], Optional[str]]:
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "unreadable"
    if not isinstance(payload, dict):
        return {}, "invalid"
    return payload, None


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _sidecar_age_minutes(path: Path, payload: Dict[str, Any]) -> Optional[float]:
    generated = (
        payload.get("generated_utc")
        if isinstance(payload, dict)
        else None
    )
    parsed = _parse_utc_datetime(generated)
    if parsed is None:
        try:
            parsed = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None
    now = datetime.now(timezone.utc)
    age = (now - parsed).total_seconds() / 60.0
    return age if age >= 0 else 0.0


def _dashboard_snapshot_max_age_seconds() -> float:
    raw = os.getenv("PMX_DASHBOARD_EXPECTED_REFRESH_SECONDS", str(DEFAULT_DASHBOARD_EXPECTED_REFRESH_SECONDS))
    try:
        expected_refresh_seconds = float(raw)
    except Exception:
        expected_refresh_seconds = float(DEFAULT_DASHBOARD_EXPECTED_REFRESH_SECONDS)
    return max(expected_refresh_seconds * 2.0, 600.0)


def validate_dashboard_payload_contract(
    payload: Dict[str, Any],
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
    freshness_threshold_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    missing_keys = [key for key in DASHBOARD_REQUIRED_TOP_LEVEL_KEYS if key not in payload]
    if missing_keys:
        errors.append("missing_top_level_keys:" + ",".join(missing_keys))

    threshold_seconds = (
        float(freshness_threshold_seconds)
        if freshness_threshold_seconds is not None
        else _dashboard_snapshot_max_age_seconds()
    )
    if threshold_seconds < 0:
        threshold_seconds = 0.0

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    payload_schema_version = meta.get("payload_schema_version")
    if payload_schema_version is not None:
        try:
            parsed_schema_version = int(payload_schema_version)
        except Exception:
            parsed_schema_version = None
        if parsed_schema_version != DASHBOARD_PAYLOAD_SCHEMA_VERSION:
            errors.append(f"unexpected_payload_schema_version:{payload_schema_version}")

    required_sections = meta.get("payload_required_sections")
    if isinstance(required_sections, list):
        missing_declared = [
            key for key in DASHBOARD_REQUIRED_TOP_LEVEL_KEYS if key not in {str(item) for item in required_sections}
        ]
        if missing_declared:
            errors.append("meta.payload_required_sections_missing:" + ",".join(missing_declared))

    generated = _parse_utc_datetime(meta.get("generated_utc")) or _parse_utc_datetime(meta.get("ts"))
    if generated is None and path is not None and path.exists():
        try:
            generated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            generated = None
    if generated is None:
        errors.append("missing_dashboard_timestamp")

    now_utc = now or datetime.now(timezone.utc)
    age_seconds: Optional[float] = None
    if generated is not None:
        age_seconds = max(0.0, (now_utc - generated).total_seconds())
        if age_seconds > threshold_seconds:
            errors.append(
                f"stale_dashboard_payload:age_seconds={age_seconds:.1f}:threshold_seconds={threshold_seconds:.1f}"
            )

    key_type_expectations = {
        "meta": dict,
        "pnl": dict,
        "signals": list,
        "trade_events": list,
        "price_series": dict,
        "robustness": dict,
        "live_denominator": dict,
        "quant_validation": dict,
    }
    for key, expected_type in key_type_expectations.items():
        value = payload.get(key)
        if key in payload and not isinstance(value, expected_type):
            errors.append(f"invalid_type:{key}:{expected_type.__name__}")

    robustness = payload.get("robustness") if isinstance(payload.get("robustness"), dict) else {}
    overall_status = str(robustness.get("overall_status") or "").strip().upper()
    freshness_status = str(robustness.get("freshness_status") or "").strip().upper()
    if not overall_status:
        errors.append("missing_robustness.overall_status")
    if not freshness_status:
        errors.append("missing_robustness.freshness_status")
    elif freshness_status != "FRESH":
        errors.append(f"robustness_not_fresh:{freshness_status}")

    if "sidecar_age_minutes" in robustness and not isinstance(robustness.get("sidecar_age_minutes"), dict):
        errors.append("invalid_type:robustness.sidecar_age_minutes:dict")

    live_denominator = payload.get("live_denominator") if isinstance(payload.get("live_denominator"), dict) else {}
    live_denominator_status = str(live_denominator.get("status") or "").strip().upper()
    if not live_denominator_status:
        errors.append("missing_live_denominator.status")

    return {
        "ok": not errors,
        "errors": errors,
        "missing_keys": missing_keys,
        "generated_utc": generated.isoformat() if generated else None,
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "freshness_threshold_seconds": round(threshold_seconds, 3),
    }


def validate_dashboard_payload_file(
    path: Path = DEFAULT_OUTPUT_PATH,
    *,
    now: Optional[datetime] = None,
    freshness_threshold_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    payload, err = _load_sidecar_json(path)
    if err:
        return {
            "ok": False,
            "errors": [f"dashboard_payload_{err}"],
            "missing_keys": list(DASHBOARD_REQUIRED_TOP_LEVEL_KEYS),
            "generated_utc": None,
            "age_seconds": None,
            "freshness_threshold_seconds": round(
                float(freshness_threshold_seconds)
                if freshness_threshold_seconds is not None
                else _dashboard_snapshot_max_age_seconds(),
                3,
            ),
        }
    return validate_dashboard_payload_contract(
        payload,
        path=path,
        now=now,
        freshness_threshold_seconds=freshness_threshold_seconds,
    )


def _robustness_payload() -> Dict[str, Any]:
    load_warnings: List[str] = []
    semantic_warnings: List[str] = []
    optional_warnings: List[str] = []
    eligibility, elig_err = _load_sidecar_json(DEFAULT_ELIGIBILITY_PATH)
    context, ctx_err = _load_sidecar_json(DEFAULT_CONTEXT_QUALITY_PATH)
    performance, perf_err = _load_sidecar_json(DEFAULT_PERFORMANCE_METRICS_PATH)
    forecast_summary, forecast_summary_err = _load_sidecar_json(DEFAULT_FORECAST_SUMMARY_PATH)
    try:
        sidecar_max_age_minutes = float(
            os.getenv("PMX_SIDECAR_MAX_AGE_MINUTES", str(DEFAULT_SIDECAR_MAX_AGE_MINUTES))
        )
    except Exception:
        sidecar_max_age_minutes = float(DEFAULT_SIDECAR_MAX_AGE_MINUTES)
    sidecar_age_minutes: Dict[str, Optional[float]] = {}
    stale_sidecars: List[str] = []

    if elig_err:
        load_warnings.append(f"eligibility_{elig_err}")
    if ctx_err:
        load_warnings.append(f"context_quality_{ctx_err}")
    if perf_err:
        load_warnings.append(f"performance_metrics_{perf_err}")
    if forecast_summary_err and forecast_summary_err != "missing":
        optional_warnings.append(f"forecast_summary_{forecast_summary_err}")

    freshness_critical_sidecars = {"eligibility", "context_quality", "performance_metrics"}

    for name, path, payload, err in (
        ("eligibility", DEFAULT_ELIGIBILITY_PATH, eligibility, elig_err),
        ("context_quality", DEFAULT_CONTEXT_QUALITY_PATH, context, ctx_err),
        ("performance_metrics", DEFAULT_PERFORMANCE_METRICS_PATH, performance, perf_err),
        ("forecast_summary", DEFAULT_FORECAST_SUMMARY_PATH, forecast_summary, forecast_summary_err),
    ):
        if err:
            sidecar_age_minutes[name] = None
            continue
        age = _sidecar_age_minutes(path, payload)
        sidecar_age_minutes[name] = round(age, 2) if age is not None else None
        if age is not None and age > sidecar_max_age_minutes:
            if name in freshness_critical_sidecars:
                stale_sidecars.append(name)
            else:
                optional_warnings.append(f"stale_sidecar:{name}")

    tickers = eligibility.get("tickers", {}) if isinstance(eligibility, dict) else {}
    weak_tickers = sorted(
        ticker for ticker, info in tickers.items()
        if isinstance(info, dict) and info.get("status") == "WEAK"
    )
    perf_metrics = performance if isinstance(performance, dict) else {}
    sufficiency = perf_metrics.get("sufficiency", {}) if isinstance(perf_metrics.get("sufficiency"), dict) else {}
    chart_paths = perf_metrics.get("chart_paths", {}) if isinstance(perf_metrics, dict) else {}
    chart_missing: List[str] = []
    if isinstance(chart_paths, dict):
        for name, raw_path in chart_paths.items():
            chart_name = str(name).strip() or "unknown"
            if not isinstance(raw_path, str) or not raw_path.strip():
                chart_missing.append(f"chart_missing:{chart_name}")
                continue
            path_obj = Path(raw_path)
            if not path_obj.is_absolute():
                path_obj = ROOT / path_obj
            if not path_obj.exists():
                chart_missing.append(f"chart_missing:{chart_name}")
    elif perf_metrics:
        semantic_warnings.append("chart_paths_invalid")
    semantic_warnings.extend(chart_missing)
    telemetry_contract = (
        forecast_summary.get("telemetry_contract", {})
        if isinstance(forecast_summary, dict)
        else {}
    )
    telemetry_schema_version: Optional[int] = None
    if isinstance(telemetry_contract, dict) and telemetry_contract:
        raw_schema = telemetry_contract.get("schema_version")
        try:
            telemetry_schema_version = int(raw_schema) if raw_schema is not None else None
        except (TypeError, ValueError):
            telemetry_schema_version = None
        if telemetry_schema_version is None or telemetry_schema_version < 2:
            semantic_warnings.append("telemetry_contract_legacy_schema")
    elif isinstance(forecast_summary, dict) and forecast_summary:
        semantic_warnings.append("telemetry_contract_missing")

    for warning in eligibility.get("warnings", []) if isinstance(eligibility, dict) else []:
        if isinstance(warning, str):
            semantic_warnings.append(warning)
    for warning in context.get("warnings", []) if isinstance(context, dict) else []:
        if isinstance(warning, str):
            semantic_warnings.append(warning)
    for warning in perf_metrics.get("warnings", []) if isinstance(perf_metrics, dict) else []:
        if isinstance(warning, str):
            semantic_warnings.append(warning)
    if isinstance(context, dict) and context.get("partial_data"):
        semantic_warnings.append("context_partial_data")
    missing_count = sum(1 for err in (elig_err, ctx_err, perf_err) if err == "missing")
    present_count = sum(1 for payload in (eligibility, context, performance) if isinstance(payload, dict) and payload)
    all_warnings = sorted(set(load_warnings + semantic_warnings + optional_warnings))
    if stale_sidecars:
        freshness_status = "STALE"
        freshness_reason = "STALE_SIDECAR"
    elif any(value is not None for value in sidecar_age_minutes.values()):
        freshness_status = "FRESH"
        freshness_reason = None
    else:
        freshness_status = "UNKNOWN"
        freshness_reason = None

    if missing_count == 3 and present_count == 0:
        robustness_status = "MISSING"
    elif load_warnings or stale_sidecars:
        robustness_status = "STALE"
    elif (
        bool(optional_warnings)
        or
        (isinstance(perf_metrics, dict) and perf_metrics.get("status") == "WARN")
        or (isinstance(sufficiency, dict) and sufficiency.get("status") not in (None, "SUFFICIENT"))
        or bool(semantic_warnings)
    ):
        robustness_status = "WARN"
    else:
        robustness_status = "OK"

    return {
        "status": robustness_status,
        "overall_status": robustness_status,
        "freshness_status": freshness_status,
        "freshness_reason": freshness_reason,
        "sidecar_max_age_minutes": sidecar_max_age_minutes,
        "sidecar_age_minutes": sidecar_age_minutes,
        "eligibility_summary": eligibility.get("summary", {}) if isinstance(eligibility, dict) else {},
        "weak_tickers": weak_tickers,
        "telemetry_contract": telemetry_contract if isinstance(telemetry_contract, dict) else {},
        "window_counts": forecast_summary.get("window_counts", {}) if isinstance(forecast_summary, dict) else {},
        "window_diversity": forecast_summary.get("window_diversity", {}) if isinstance(forecast_summary, dict) else {},
        "cache_status": forecast_summary.get("cache_status", {}) if isinstance(forecast_summary, dict) else {},
        "context_quality_summary": {
            "n_total_trades": context.get("n_total_trades", 0),
            "n_trades_no_confidence": context.get("n_trades_no_confidence", 0),
            "partial_data": bool(context.get("partial_data", False)),
            "regime_count": len(context.get("regime_quality", {}) or {}),
            "confidence_bin_count": len(context.get("confidence_bin_quality", {}) or {}),
        },
        "sufficiency": sufficiency,
        "performance_metrics": perf_metrics,
        "chart_paths": chart_paths if isinstance(chart_paths, dict) else {},
        "warnings": all_warnings,
    }


def _live_denominator_payload() -> Dict[str, Any]:
    payload, payload_err = _load_sidecar_json(DEFAULT_LIVE_DENOMINATOR_PATH)
    if payload_err:
        return {
            "status": "MISSING",
            "warnings": [f"live_denominator_{payload_err}"],
            "run_meta": {},
            "current": {},
            "cycles_completed": 0,
        }

    run_meta = payload.get("run_meta", {}) if isinstance(payload.get("run_meta"), dict) else {}
    cycles = payload.get("cycles", []) if isinstance(payload.get("cycles"), list) else []
    current = cycles[-1] if cycles and isinstance(cycles[-1], dict) else {}
    warnings: List[str] = []

    sleep_seconds = 0
    try:
        sleep_seconds = int(run_meta.get("sleep_seconds") or 0)
    except (TypeError, ValueError):
        sleep_seconds = 0

    age = _sidecar_age_minutes(DEFAULT_LIVE_DENOMINATOR_PATH, current or run_meta)
    age_minutes = round(age, 2) if age is not None else None

    status = "WAITING"
    if current.get("progress_triggered"):
        status = "PROGRESS"
    if age is not None:
        stale_limit_minutes = max(180.0, (sleep_seconds / 60.0) * 2.0) if sleep_seconds > 0 else 180.0
        if age > stale_limit_minutes:
            status = "STALE"
            warnings.append("live_denominator_stale")

    return {
        "status": status,
        "warnings": warnings,
        "age_minutes": age_minutes,
        "run_meta": run_meta,
        "current": current,
        "cycles_completed": len(cycles),
    }


def _merge_dashboard_producer_artifact(
    fresh: Dict[str, Any],
    *,
    producer_path: Path = DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH,
) -> Dict[str, Any]:
    if not producer_path.exists():
        return fresh
    try:
        existing = json.loads(producer_path.read_text(encoding="utf-8"))
    except Exception:
        return fresh
    if not isinstance(existing, dict):
        return fresh
    producer_meta = existing.get("meta") if isinstance(existing.get("meta"), dict) else {}
    fresh_meta = fresh.get("meta") if isinstance(fresh.get("meta"), dict) else {}
    for key in ("cycles", "llm_enabled", "strategy"):
        if key in producer_meta and producer_meta.get(key) is not None:
            fresh_meta[key] = producer_meta.get(key)
    if fresh_meta:
        fresh["meta"] = fresh_meta

    # The bridge remains the canonical dashboard writer, but it still reads a
    # few runtime-only fields from the latest auto-trader snapshot.
    for key in ("latency", "routing", "equity", "equity_realized", "forecaster_health", "regime", "notes"):
        value = existing.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)) and not value:
            continue
        fresh[key] = value
    return fresh


def _persist_snapshot(audit_db: Path, payload: Dict[str, Any]) -> None:
    conn = _connect_rw(audit_db)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              run_id TEXT,
              payload_json TEXT NOT NULL
            )
            """
        )
        meta = payload.get("meta") or {}
        run_id = meta.get("run_id")
        conn.execute(
            "INSERT INTO dashboard_snapshots(created_at, run_id, payload_json) VALUES (?,?,?)",
            (_utc_now_iso(), str(run_id) if run_id else None, json.dumps(payload, sort_keys=True)),
        )
        conn.commit()
    finally:
        conn.close()

def _model_params(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    """
    Best-effort fetch of latest model parameters per ticker from time_series_forecasts.
    Handles flexible column names for params to stay backward compatible.
    """
    try:
        cols = [r[1] for r in _safe_fetchall(conn, "PRAGMA table_info(time_series_forecasts)")]
    except sqlite3.OperationalError:
        return {}
    param_cols = [c for c in cols if "param" in c.lower()]
    if not param_cols:
        return {}
    # prefer JSON-like column names
    chosen_col = param_cols[0]
    try:
        rows = _safe_fetchall(
            conn,
            f"""
            SELECT ticker, model_type, {chosen_col} AS params, created_at
            FROM time_series_forecasts
            WHERE {chosen_col} IS NOT NULL
            ORDER BY created_at DESC, id DESC
            LIMIT 400
            """
        )
    except sqlite3.OperationalError:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        t = str(r["ticker"] or "").strip().upper()
        if not t:
            continue
        models = out.setdefault(t, [])
        try:
            import json  # type: ignore
            parsed = json.loads(r["params"]) if isinstance(r["params"], str) else r["params"]
        except Exception:
            parsed = r["params"]
        models.append(
            {
                "model_type": str(r["model_type"] or "").upper(),
                "params": parsed,
                "created_at": str(r["created_at"] or ""),
            }
        )
    return out

def _data_checks(conn: sqlite3.Connection) -> List[str]:
    """Lightweight data/diagnostic checks to surface common pitfalls on the dashboard."""
    checks: List[str] = []
    # positions present?
    try:
        pos_count = _safe_fetchone(conn, "SELECT COUNT(*) AS c FROM portfolio_positions") or {"c": 0}
        if int(pos_count["c"] or 0) == 0:
            checks.append("No portfolio_positions rows found (positions table empty).")
    except Exception:
        checks.append("portfolio_positions table unavailable.")
    # trade executions mix of actions?
    try:
        actions = _safe_fetchall(conn, "SELECT action, COUNT(*) AS c FROM trade_executions GROUP BY action")
        sells = sum(int(r["c"] or 0) for r in actions if str(r["action"]).upper() in {"SELL", "CLOSE", "EXIT"})
        buys = sum(int(r["c"] or 0) for r in actions if str(r["action"]).upper() in {"BUY", "OPEN"})
        if buys > 0 and sells == 0:
            checks.append("Only BUY/OPEN trades recorded; no SELL/CLOSE exits found.")
    except Exception:
        checks.append("trade_executions table unavailable.")
    # performance metrics presence
    try:
        perf_rows = _safe_fetchone(conn, "SELECT COUNT(*) AS c FROM performance_metrics") or {"c": 0}
        if int(perf_rows["c"] or 0) == 0:
            checks.append("performance_metrics missing; run performance aggregation.")
    except Exception:
        checks.append("performance_metrics table unavailable.")
    return checks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render live_dashboard payload from SQLite DB.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Trading DB path (default: data/portfolio_maximizer.db).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path (default: visualizations/dashboard_data.json).")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers (defaults to barbell.yml + DB observed tickers).")
    parser.add_argument("--lookback-days", type=int, default=180, help="Price series lookback (days).")
    parser.add_argument("--max-signals", type=int, default=50, help="Max trading_signals rows to include.")
    parser.add_argument("--max-trades", type=int, default=200, help="Max trade_executions rows to include.")
    parser.add_argument("--interval-seconds", type=float, default=5.0, help="Refresh interval (seconds) when looping.")
    parser.add_argument("--once", action="store_true", help="Render once and exit.")
    parser.add_argument("--persist-snapshot", action="store_true", help="Persist each snapshot into an audit SQLite DB.")
    parser.add_argument("--audit-db-path", default=str(DEFAULT_AUDIT_DB_PATH), help="Audit DB path for snapshots (default: data/dashboard_audit.db).")
    runs_group = parser.add_mutually_exclusive_group()
    runs_group.add_argument(
        "--latest-run-only",
        dest="latest_run_only",
        action="store_true",
        help="Show trade events only for the latest run_id (default: on).",
    )
    runs_group.add_argument(
        "--all-runs",
        dest="latest_run_only",
        action="store_false",
        help="Include trade events from all runs (disables --latest-run-only).",
    )
    parser.set_defaults(latest_run_only=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    db_path = Path(args.db_path).expanduser()
    out_path = Path(args.output).expanduser()
    mirror_path = _wsl_mirror_path(db_path)

    def _tickers(conn: sqlite3.Connection) -> List[str]:
        raw = str(args.tickers or "").strip()
        if raw:
            return [t.strip().upper() for t in raw.split(",") if t.strip()]
        return _default_tickers(conn)

    while True:
        read_path_used = _select_read_path(db_path)
        conn = _connect_ro_with_fallback(db_path)
        try:
            try:
                tickers = _tickers(conn)
                payload = build_dashboard_payload(
                    conn=conn,
                    tickers=tickers,
                    lookback_days=int(args.lookback_days),
                    max_signals=int(args.max_signals),
                    max_trades=int(args.max_trades),
                    latest_run_only=bool(args.latest_run_only),
                    db_path=db_path,
                    read_path=read_path_used,
                    mirror_path=mirror_path,
                    output_path=out_path,
                )
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "disk i/o error" in msg and mirror_path:
                    try:
                        if db_path.exists():
                            mirror_path.parent.mkdir(parents=True, exist_ok=True)
                            import shutil
                            shutil.copy2(db_path, mirror_path)
                    except Exception:
                        pass
                    if mirror_path.exists():
                        conn.close()
                        conn = _connect_ro(mirror_path)
                        read_path_used = mirror_path
                        tickers = _tickers(conn)
                        payload = build_dashboard_payload(
                            conn=conn,
                            tickers=tickers,
                            lookback_days=int(args.lookback_days),
                            max_signals=int(args.max_signals),
                            max_trades=int(args.max_trades),
                            latest_run_only=bool(args.latest_run_only),
                            db_path=db_path,
                            read_path=read_path_used,
                            mirror_path=mirror_path,
                            output_path=out_path,
                        )
                    else:
                        raise
                else:
                    raise
        finally:
            conn.close()

        merged = _merge_dashboard_producer_artifact(payload)
        if isinstance(merged.get("meta"), dict):
            merged["meta"]["payload_digest"] = _payload_digest(merged)
        _atomic_write_json(out_path, merged)
        if args.persist_snapshot:
            _persist_snapshot(Path(args.audit_db_path).expanduser(), merged)

        if args.once:
            return
        time.sleep(max(0.5, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
