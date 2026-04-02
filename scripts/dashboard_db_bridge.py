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
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
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
DEFAULT_PRODUCTION_GATE_PATH = ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_SIDECAR_MAX_AGE_MINUTES = 120
DEFAULT_PRODUCTION_GATE_MAX_AGE_MINUTES = 30
DEFAULT_AUDIT_SNAPSHOT_MAX_AGE_MINUTES = 60

try:
    from integrity.sqlite_guardrails import apply_sqlite_guardrails, guarded_sqlite_connect
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from integrity.sqlite_guardrails import apply_sqlite_guardrails, guarded_sqlite_connect

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


def _relative_repo_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _freshness_from_age(
    age_minutes: Optional[float],
    *,
    max_age_minutes: float,
) -> Tuple[str, Optional[str]]:
    if age_minutes is None:
        return "UNKNOWN", None
    if age_minutes > max_age_minutes:
        return "STALE", "AGE_EXCEEDED"
    return "FRESH", None


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


def _connect_rw(db_path: Path, *, allow_schema_changes: bool = False) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # [SETUP-PHASE BYPASS] journal_mode=WAL is in BLOCKED_PRAGMAS and must be set before
    # guardrails lock the connection.  apply_sqlite_guardrails() MUST be the next call after
    # the PRAGMA block -- do not insert code between them.
    conn = guarded_sqlite_connect(str(db_path), timeout=5.0, enable_guardrails=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # blocked once guardrails apply
    conn.execute("PRAGMA busy_timeout=5000")  # not blocked; set here for co-location clarity
    apply_sqlite_guardrails(conn, allow_schema_changes=allow_schema_changes)
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
    origin = "synthetic" if has_synthetic else "live"
    if has_synthetic and any(src for src in ohlcv_sources if src and src != "synthetic"):
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


def _positions(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    row = _safe_fetchone(conn, "SELECT MAX(position_date) AS d FROM portfolio_positions")
    if not row or not row["d"]:
        return _positions_from_executions(conn)
    latest = str(row["d"])
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
    return out


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
    rows = _safe_fetchall(
        conn,
        """
        SELECT ticker, action, shares, price, trade_date, created_at
        FROM trade_executions
        WHERE ticker IS NOT NULL AND action IS NOT NULL
        ORDER BY COALESCE(created_at, trade_date) ASC, id ASC
        """,
    )
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
    query_full = f"""
    SELECT ticker, action, shares, price, trade_date, created_at, realized_pnl, realized_pnl_pct, mid_slippage_bps,
           data_source, execution_mode,
           barbell_bucket, barbell_multiplier, base_confidence, effective_confidence
    FROM trade_executions
    WHERE UPPER(ticker) IN ({placeholders}) {run_clause}
    ORDER BY COALESCE(created_at, trade_date) DESC, id DESC
    LIMIT ?
    """
    query_min = f"""
    SELECT ticker, action, shares, price, trade_date, created_at, realized_pnl, realized_pnl_pct, mid_slippage_bps
    FROM trade_executions
    WHERE UPPER(ticker) IN ({placeholders}) {run_clause}
    ORDER BY COALESCE(created_at, trade_date) DESC, id DESC
    LIMIT ?
    """
    try:
        rows = _safe_fetchall(conn, query_full, tuple(params_base))
    except sqlite3.OperationalError as exc:
        if "no such column" not in str(exc).lower():
            raise
        rows = _safe_fetchall(conn, query_min, tuple(params_base))
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
                "exit_reason": None,
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
            "pnl_abs": 0.0,
            "pnl_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }
    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(row[key] or 0.0)
        except Exception:
            return default
    def _i(key: str, default: int = 0) -> int:
        try:
            return int(row[key] or 0)
        except Exception:
            return default
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


def build_dashboard_payload(
    *,
    conn: sqlite3.Connection,
    tickers: List[str],
    lookback_days: int,
    max_signals: int,
    max_trades: int,
    latest_run_only: bool = True,
) -> Dict[str, Any]:
    perf = _latest_performance(conn)
    run_id = _latest_run_id(conn) or "db_bridge"
    ts = _utc_now_iso()
    bucket_map = _barbell_bucket_map()
    model_params = _model_params(conn)
    checks = _data_checks(conn)
    provenance = _provenance_summary(conn)
    robustness = _robustness_payload()
    evidence = {
        "canonical_view": "static_evidence_dashboard",
        "latest_run_only": bool(latest_run_only),
        "production_gate": _production_gate_summary(),
        "dashboard_audit": _dashboard_audit_summary(),
    }

    qual_records: List[Dict[str, Any]] = []
    for t in tickers:
        rec = _quality_latest(conn, t)
        if rec:
            qual_records.append(rec)
    avg_q = sum(r["quality_score"] for r in qual_records) / len(qual_records) if qual_records else 0.0
    min_q = min((r["quality_score"] for r in qual_records), default=0.0)

    payload: Dict[str, Any] = {
        # NOTE: keep payload self-contained for the static HTML dashboard.
        "meta": {
            "run_id": str(run_id),
            "ts": ts,
            "tickers": tickers,
            "cycles": None,
            "llm_enabled": None,
            "dashboard_version": "db_bridge_v1",
            "canonical_view": evidence["canonical_view"],
            "latest_run_only": bool(latest_run_only),
            "data_origin": provenance.get("origin"),
            "dataset_id": provenance.get("dataset_id"),
            "data_source": provenance.get("data_source"),
            "execution_mode": provenance.get("execution_mode"),
            "provenance": provenance,
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
        "pnl": {"absolute": float(perf["pnl_abs"]), "pct": float(perf["pnl_pct"])},
        "win_rate": float(perf["win_rate"]),
        "trade_count": int(perf["trade_count"]),
        "performance": perf,
        "positions": _positions(conn),
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
        "robustness": robustness,
        "evidence": evidence,
    }
    payload["alerts"] = _operator_alerts(
        provenance=provenance,
        evidence=evidence,
        robustness=robustness,
        signal_count=len(payload["signals"]),
        trade_count=len(payload["trade_events"]),
        price_series_count=len(payload["price_series"]),
    )
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


def _robustness_payload() -> Dict[str, Any]:
    load_warnings: List[str] = []
    semantic_warnings: List[str] = []
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
        load_warnings.append(f"forecast_summary_{forecast_summary_err}")

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
            stale_sidecars.append(name)
            semantic_warnings.append(f"stale_sidecar:{name}")

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
    all_warnings = sorted(set(load_warnings + semantic_warnings))
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


def _dashboard_audit_summary() -> Dict[str, Any]:
    path = DEFAULT_AUDIT_DB_PATH
    summary: Dict[str, Any] = {
        "available": path.exists(),
        "path": _relative_repo_path(path),
        "status": "MISSING",
        "snapshot_count": 0,
        "latest_created_at": None,
        "latest_run_id": None,
        "age_minutes": None,
        "freshness_status": "UNKNOWN",
        "freshness_reason": None,
        "error": None,
    }
    if not path.exists():
        return summary

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect_ro(path)
        count_row = _safe_fetchone(conn, "SELECT COUNT(*) AS c FROM dashboard_snapshots")
        latest_row = _safe_fetchone(
            conn,
            """
            SELECT created_at, run_id
            FROM dashboard_snapshots
            ORDER BY id DESC
            LIMIT 1
            """,
        )
        snapshot_count = int(count_row["c"] or 0) if count_row else 0
        latest_created_at = str(latest_row["created_at"]) if latest_row and latest_row["created_at"] else None
        latest_run_id = str(latest_row["run_id"]) if latest_row and latest_row["run_id"] else None
        try:
            max_age_minutes = float(
                os.getenv(
                    "PMX_AUDIT_SNAPSHOT_MAX_AGE_MINUTES",
                    str(DEFAULT_AUDIT_SNAPSHOT_MAX_AGE_MINUTES),
                )
            )
        except Exception:
            max_age_minutes = float(DEFAULT_AUDIT_SNAPSHOT_MAX_AGE_MINUTES)
        age_minutes = (
            _sidecar_age_minutes(path, {"generated_utc": latest_created_at})
            if latest_created_at
            else None
        )
        freshness_status, freshness_reason = _freshness_from_age(
            age_minutes,
            max_age_minutes=max_age_minutes,
        )
        summary.update(
            {
                "status": "OK" if snapshot_count > 0 else "EMPTY",
                "snapshot_count": snapshot_count,
                "latest_created_at": latest_created_at,
                "latest_run_id": latest_run_id,
                "age_minutes": round(age_minutes, 2) if age_minutes is not None else None,
                "freshness_status": freshness_status,
                "freshness_reason": freshness_reason,
            }
        )
        return summary
    except Exception as exc:
        summary.update(
            {
                "status": "ERROR",
                "error": str(exc),
            }
        )
        return summary
    finally:
        if conn is not None:
            conn.close()


def _production_gate_summary() -> Dict[str, Any]:
    path = DEFAULT_PRODUCTION_GATE_PATH
    payload, err = _load_sidecar_json(path)
    summary: Dict[str, Any] = {
        "available": err is None,
        "path": _relative_repo_path(path),
        "status": "MISSING" if err == "missing" else "ERROR" if err else "UNKNOWN",
        "gate_semantics_status": None,
        "phase3_ready": False,
        "phase3_reason": None,
        "artifact_binding_pass": None,
        "artifact_binding_reasons": [],
        "proof_status": None,
        "remaining_closed_trades": None,
        "remaining_trading_days": None,
        "telemetry_status": None,
        "telemetry_severity": None,
        "generated_utc": None,
        "age_minutes": None,
        "freshness_status": "UNKNOWN",
        "freshness_reason": None,
        "error": None if err in (None, "missing") else err,
    }
    if err:
        return summary

    gate = payload.get("production_profitability_gate", {}) if isinstance(payload, dict) else {}
    proof = payload.get("profitability_proof", {}) if isinstance(payload, dict) else {}
    readiness = payload.get("readiness", {}) if isinstance(payload, dict) else {}
    artifact_binding = payload.get("artifact_binding", {}) if isinstance(payload, dict) else {}
    telemetry = payload.get("telemetry_contract", {}) if isinstance(payload, dict) else {}
    evidence_progress = proof.get("evidence_progress", {}) if isinstance(proof, dict) else {}
    generated_utc = None
    if isinstance(payload, dict):
        generated_utc = payload.get("timestamp_utc")
    if generated_utc is None and isinstance(telemetry, dict):
        generated_utc = telemetry.get("generated_utc")
    try:
        max_age_minutes = float(
            os.getenv(
                "PMX_PRODUCTION_GATE_MAX_AGE_MINUTES",
                str(DEFAULT_PRODUCTION_GATE_MAX_AGE_MINUTES),
            )
        )
    except Exception:
        max_age_minutes = float(DEFAULT_PRODUCTION_GATE_MAX_AGE_MINUTES)
    age_minutes = _sidecar_age_minutes(
        path,
        {"generated_utc": generated_utc} if generated_utc else payload,
    )
    freshness_status, freshness_reason = _freshness_from_age(
        age_minutes,
        max_age_minutes=max_age_minutes,
    )
    summary.update(
        {
            "status": str(gate.get("status") or "UNKNOWN").upper(),
            "gate_semantics_status": gate.get("gate_semantics_status"),
            "phase3_ready": bool(
                payload.get("phase3_ready")
                if isinstance(payload, dict) and payload.get("phase3_ready") is not None
                else readiness.get("phase3_ready", False)
            ),
            "phase3_reason": (
                payload.get("phase3_reason")
                if isinstance(payload, dict) and payload.get("phase3_reason")
                else readiness.get("phase3_reason")
            ) or readiness.get("phase3_reason"),
            "artifact_binding_pass": (
                bool(artifact_binding.get("pass"))
                if isinstance(artifact_binding, dict) and "pass" in artifact_binding
                else None
            ),
            "artifact_binding_reasons": (
                artifact_binding.get("reason_codes", [])
                if isinstance(artifact_binding, dict)
                else []
            ),
            "proof_status": proof.get("status") if isinstance(proof, dict) else None,
            "remaining_closed_trades": (
                int(evidence_progress.get("remaining_closed_trades"))
                if isinstance(evidence_progress, dict)
                and evidence_progress.get("remaining_closed_trades") is not None
                else None
            ),
            "remaining_trading_days": (
                int(evidence_progress.get("remaining_trading_days"))
                if isinstance(evidence_progress, dict)
                and evidence_progress.get("remaining_trading_days") is not None
                else None
            ),
            "telemetry_status": telemetry.get("status") if isinstance(telemetry, dict) else None,
            "telemetry_severity": telemetry.get("severity") if isinstance(telemetry, dict) else None,
            "generated_utc": generated_utc,
            "age_minutes": round(age_minutes, 2) if age_minutes is not None else None,
            "freshness_status": freshness_status,
            "freshness_reason": freshness_reason,
        }
    )
    return summary


def _operator_alerts(
    *,
    provenance: Dict[str, Any],
    evidence: Dict[str, Any],
    robustness: Dict[str, Any],
    signal_count: int,
    trade_count: int,
    price_series_count: int,
) -> List[str]:
    alerts: List[str] = []
    origin = str(provenance.get("origin") or "").strip().lower()
    gate = evidence.get("production_gate", {}) if isinstance(evidence, dict) else {}
    audit = evidence.get("dashboard_audit", {}) if isinstance(evidence, dict) else {}
    robustness_status = str(robustness.get("overall_status") or robustness.get("status") or "UNKNOWN").upper()
    gate_state = str(gate.get("status") or "UNKNOWN").upper()
    audit_state = str(audit.get("status") or "UNKNOWN").upper()

    if origin in {"synthetic", "mixed"}:
        alerts.append("Data origin is not fully live; treat the dashboard as evidence-in-progress.")

    gate_status = gate_state
    if gate_status in {"FAIL", "INCONCLUSIVE"}:
        alerts.append(
            f"Production gate is {gate_status}; phase3_ready={int(bool(gate.get('phase3_ready')))}."
        )
    elif gate_state == "ERROR":
        alerts.append("Production gate artifact is unreadable or invalid.")
    elif gate.get("available") is False:
        alerts.append("Production gate artifact is missing.")

    if gate.get("freshness_status") == "STALE":
        alerts.append("Production gate artifact is stale relative to the alerting freshness policy.")

    if gate.get("artifact_binding_pass") is False:
        reasons = gate.get("artifact_binding_reasons") or []
        suffix = f" ({', '.join(str(r) for r in reasons[:3])})" if reasons else ""
        alerts.append(f"Artifact binding failed{suffix}.")

    remaining_closed = gate.get("remaining_closed_trades")
    remaining_days = gate.get("remaining_trading_days")
    if (remaining_closed or 0) > 0 or (remaining_days or 0) > 0:
        alerts.append(
            "Profitability proof runway incomplete: "
            f"{int(remaining_closed or 0)} closed trades and {int(remaining_days or 0)} trading days remaining."
        )

    if audit_state == "ERROR":
        alerts.append("Dashboard audit snapshot DB exists but cannot be queried cleanly.")
    elif audit.get("available") is False:
        alerts.append("Dashboard audit snapshot DB is missing.")
    elif audit_state == "EMPTY" or int(audit.get("snapshot_count") or 0) == 0:
        alerts.append("Dashboard audit snapshot DB has no persisted snapshots yet.")
    elif audit.get("freshness_status") == "STALE":
        alerts.append("Dashboard audit snapshots exist but are stale.")

    if robustness_status in {"WARN", "STALE", "MISSING"}:
        alerts.append(f"Robustness sidecars are {robustness_status}.")

    if trade_count == 0:
        alerts.append("No trade events are present in the current canonical payload.")
    if signal_count == 0:
        alerts.append("No signals are present in the current canonical payload.")
    if price_series_count == 0:
        alerts.append("No price series are present in the current canonical payload.")

    deduped: List[str] = []
    seen = set()
    for alert in alerts:
        if alert not in seen:
            seen.add(alert)
            deduped.append(alert)
    return deduped


def _maybe_merge_with_existing(path: Path, fresh: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return fresh
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fresh
    if not isinstance(existing, dict):
        return fresh
    # Preserve blocks the DB bridge doesn't own, if present.
    for key in ("forecaster_health", "regime", "notes"):
        if key in existing and key not in fresh:
            fresh[key] = existing[key]
    return fresh


def _persist_snapshot(audit_db: Path, payload: Dict[str, Any]) -> None:
    # Snapshot persistence owns its small audit DB schema and needs CREATE TABLE
    # during first-run bootstrap.
    conn = _connect_rw(audit_db, allow_schema_changes=True)
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
                        tickers = _tickers(conn)
                        payload = build_dashboard_payload(
                            conn=conn,
                            tickers=tickers,
                            lookback_days=int(args.lookback_days),
                            max_signals=int(args.max_signals),
                            max_trades=int(args.max_trades),
                        )
                    else:
                        raise
                else:
                    raise
        finally:
            conn.close()

        merged = _maybe_merge_with_existing(out_path, payload)
        _atomic_write_json(out_path, merged)
        if args.persist_snapshot:
            _persist_snapshot(Path(args.audit_db_path).expanduser(), merged)

        if args.once:
            return
        time.sleep(max(0.5, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
