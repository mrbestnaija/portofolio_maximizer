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


def _positions(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    row = _safe_fetchone(conn, "SELECT MAX(position_date) AS d FROM portfolio_positions")
    if not row or not row["d"]:
        return {}
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
    ticker_list = [str(t).upper() for t in tickers if str(t).strip()]
    if not ticker_list:
        return []
    placeholders = ",".join("?" for _ in ticker_list)
    query_full = f"""
    SELECT ticker, action, shares, price, trade_date, created_at, realized_pnl, realized_pnl_pct, mid_slippage_bps,
           data_source, execution_mode,
           barbell_bucket, barbell_multiplier, base_confidence, effective_confidence
    FROM trade_executions
    WHERE UPPER(ticker) IN ({placeholders})
    ORDER BY COALESCE(created_at, trade_date) DESC, id DESC
    LIMIT ?
    """
    query_min = f"""
    SELECT ticker, action, shares, price, trade_date, created_at, realized_pnl, realized_pnl_pct, mid_slippage_bps
    FROM trade_executions
    WHERE UPPER(ticker) IN ({placeholders})
    ORDER BY COALESCE(created_at, trade_date) DESC, id DESC
    LIMIT ?
    """
    try:
        rows = _safe_fetchall(conn, query_full, tuple(ticker_list + [int(limit)]))
    except sqlite3.OperationalError as exc:
        if "no such column" not in str(exc).lower():
            raise
        rows = _safe_fetchall(conn, query_min, tuple(ticker_list + [int(limit)]))
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
    row = _safe_fetchone(
        conn,
        """
        SELECT total_return, total_return_pct, win_rate, profit_factor, num_trades
        FROM performance_metrics
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
    )
    if not row:
        return {"pnl_abs": 0.0, "pnl_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "trade_count": 0}
    try:
        pnl_abs = float(row["total_return"] or 0.0)
    except Exception:
        pnl_abs = 0.0
    try:
        pnl_pct = float(row["total_return_pct"] or 0.0)
    except Exception:
        pnl_pct = 0.0
    try:
        win_rate = float(row["win_rate"] or 0.0)
    except Exception:
        win_rate = 0.0
    try:
        profit_factor = float(row["profit_factor"] or 0.0)
    except Exception:
        profit_factor = 0.0
    try:
        trade_count = int(row["num_trades"] or 0)
    except Exception:
        trade_count = 0
    return {
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trade_count": trade_count,
    }


def build_dashboard_payload(
    *,
    conn: sqlite3.Connection,
    tickers: List[str],
    lookback_days: int,
    max_signals: int,
    max_trades: int,
) -> Dict[str, Any]:
    perf = _latest_performance(conn)
    run_id = _latest_run_id(conn) or "db_bridge"
    ts = _utc_now_iso()
    bucket_map = _barbell_bucket_map()

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
        "positions": _positions(conn),
        "latency": {"ts_ms": None, "llm_ms": None},
        "routing": {"ts_signals": 0, "llm_signals": 0, "fallback_used": 0},
        "quality": {"average": float(avg_q), "minimum": float(min_q), "records": qual_records},
        "equity": [],
        "equity_realized": [],
        "signals": _latest_signals(conn, tickers, max_signals),
        "trade_events": _trade_events(conn, tickers, max_trades),
        "price_series": {t: _price_series(conn, t, lookback_days) for t in tickers},
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    db_path = Path(args.db_path).expanduser()
    out_path = Path(args.output).expanduser()

    def _tickers(conn: sqlite3.Connection) -> List[str]:
        raw = str(args.tickers or "").strip()
        if raw:
            return [t.strip().upper() for t in raw.split(",") if t.strip()]
        return _default_tickers(conn)

    while True:
        conn = _connect_ro(db_path)
        try:
            tickers = _tickers(conn)
            payload = build_dashboard_payload(
                conn=conn,
                tickers=tickers,
                lookback_days=int(args.lookback_days),
                max_signals=int(args.max_signals),
                max_trades=int(args.max_trades),
            )
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
