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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path(os.getenv("PORTFOLIO_DB_PATH") or (ROOT / "data" / "portfolio_maximizer.db"))
DEFAULT_OUTPUT_PATH = ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_AUDIT_DB_PATH = ROOT / "data" / "dashboard_audit.db"

def _wsl_mirror_path(db_path: Path) -> Optional[Path]:
    if os.name != "posix":
        return None
    if not db_path.as_posix().startswith("/mnt/"):
        return None
    tmp_root = Path(os.environ.get("WSL_SQLITE_TMP", "/tmp"))
    return tmp_root / f"{db_path.name}.wsl"


def _select_read_path(db_path: Path) -> Path:
    mirror = _wsl_mirror_path(db_path)
    if mirror and mirror.exists():
        try:
            if not db_path.exists():
                return mirror
            if mirror.stat().st_mtime >= db_path.stat().st_mtime:
                return mirror
        except OSError:
            return mirror
    return db_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


def _connect_rw(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
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
    }
    return payload


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


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
        read_path = _select_read_path(db_path)
        conn = _connect_ro(read_path)
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
