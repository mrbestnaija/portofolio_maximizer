#!/usr/bin/env python3
"""
audit_dashboard_payload_sources.py
---------------------------------

Audits the dashboard payload snapshot + snapshot DB to ensure:
- Sources are from an approved allowlist (no unknown providers).
- No demo/sample payloads are being injected into live dashboards.
- No synthetic tickers appear in the dashboard payload unless explicitly allowed.

This is a fast, local-only check (no network, no docs parsing).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrity.sqlite_guardrails import guarded_sqlite_connect

DEFAULT_DB = Path("data/portfolio_maximizer.db")
DEFAULT_AUDIT_DB = Path("data/dashboard_audit.db")
DEFAULT_DASH_JSON = Path("visualizations/dashboard_data.json")


def _connect(db_path: Path, *, ro: bool) -> sqlite3.Connection:
    if ro:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        con = guarded_sqlite_connect(uri, uri=True, timeout=2.0)
    else:
        con = guarded_sqlite_connect(str(db_path), timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def _fetchall(con: sqlite3.Connection, q: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    cur = con.execute(q, params)
    return list(cur.fetchall() or [])


def _fetchone(con: sqlite3.Connection, q: str, params: Tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
    rows = _fetchall(con, q, params)
    return rows[0] if rows else None


def _synthetic_ticker(t: str) -> bool:
    u = str(t or "").strip().upper()
    return u.startswith("SYN") and u[3:].isdigit()


def _load_dashboard_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"FAIL: dashboard JSON missing at {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FAIL: invalid JSON at {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"FAIL: dashboard JSON is not an object at {path}")
    return obj


def _audit_db_latest_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    con = _connect(path, ro=True)
    try:
        row = _fetchone(
            con,
            """
            SELECT created_at, run_id, payload_json
            FROM dashboard_snapshots
            ORDER BY id DESC
            LIMIT 1
            """,
        )
        if not row:
            raise SystemExit(f"FAIL: dashboard_snapshots empty in {path}")
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError as exc:
            raise SystemExit(f"FAIL: payload_json is invalid JSON in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"FAIL: payload_json is not an object in {path}")
        payload["_audit_meta"] = {"created_at": row["created_at"], "run_id": row["run_id"]}
        return payload
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"FAIL: audit DB schema missing dashboard_snapshots in {path}: {exc}") from exc
    finally:
        con.close()


def _distinct_sources(con: sqlite3.Connection, table: str, col: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        rows = _fetchall(con, f"SELECT {col} AS v, COUNT(*) AS n FROM {table} GROUP BY {col} ORDER BY n DESC")
    except sqlite3.OperationalError:
        return out
    for r in rows:
        v = r["v"]
        key = "" if v is None else str(v)
        out[key] = int(r["n"] or 0)
    return out


def _failures_to_exit_code(failures: List[str]) -> int:
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dashboard payload + audit DB sources.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB), help="Portfolio DB path (default: data/portfolio_maximizer.db)")
    parser.add_argument("--audit-db-path", default=str(DEFAULT_AUDIT_DB), help="Dashboard audit DB path (default: data/dashboard_audit.db)")
    parser.add_argument("--dashboard-json", default=str(DEFAULT_DASH_JSON), help="Dashboard JSON path (default: visualizations/dashboard_data.json)")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic SYN* tickers in payload.")
    parser.add_argument(
        "--allow-missing-audit-db",
        action="store_true",
        help="Do not fail when the audit DB is missing (dashboard_data.json is still audited).",
    )
    parser.add_argument("--fail-on-missing-trade-source", action="store_true", help="Fail if trade_executions.data_source is empty/NULL for any rows.")
    parser.add_argument(
        "--approved-ohlcv-sources",
        default="yfinance,alpha_vantage,finnhub,ctrader,polygon,iex,manual,synthetic",
        help="Comma-separated allowed values for ohlcv_data.source",
    )
    parser.add_argument(
        "--approved-trade-sources",
        default="yfinance,alpha_vantage,finnhub,ctrader,polygon,iex,manual,synthetic",
        help="Comma-separated allowed values for trade_executions.data_source",
    )
    parser.add_argument(
        "--approved-signal-sources",
        default="TIME_SERIES,LLM,RULES,UNKNOWN",
        help="Comma-separated allowed values for trading_signals.source",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()
    audit_db_path = Path(args.audit_db_path).expanduser()
    dash_json_path = Path(args.dashboard_json).expanduser()

    failures: List[str] = []
    warnings: List[str] = []

    dash = _load_dashboard_json(dash_json_path)
    snap: Optional[Dict[str, Any]] = None
    try:
        # Small retry loop to tolerate first-run races where the daemon has not
        # yet flushed its first snapshot.
        for _ in range(10):
            try:
                snap = _audit_db_latest_snapshot(audit_db_path)
                break
            except FileNotFoundError:
                snap = None
            except sqlite3.OperationalError:
                snap = None
            except SystemExit:
                # Propagate structural audit DB issues immediately.
                raise
    except FileNotFoundError:
        snap = None

    if snap is None and not args.allow_missing_audit_db:
        raise SystemExit(f"FAIL: audit DB missing at {audit_db_path} (expected snapshots persisted)")

    payloads: List[Tuple[str, Dict[str, Any]]] = [("dashboard_data.json", dash)]
    if isinstance(snap, dict):
        payloads.append(("dashboard_audit.db/latest", snap))
    for name, payload in payloads:
        meta = payload.get("meta") or {}
        rid = str(meta.get("run_id") or "")
        if "demo" in rid.lower() or "sample" in rid.lower():
            failures.append(f"{name}: run_id looks non-canonical ({rid!r})")

        if "T-3" in json.dumps(payload) or "\"demo\"" in json.dumps(payload):
            failures.append(f"{name}: appears to include demo/placeholder payload fragments")

        tickers = meta.get("tickers") or []
        if isinstance(tickers, list):
            syn = [t for t in tickers if _synthetic_ticker(str(t))]
            if syn and not args.allow_synthetic:
                failures.append(f"{name}: synthetic tickers present in payload: {syn[:10]}")

        trade_events = payload.get("trade_events") or []
        if isinstance(trade_events, list):
            for ev in trade_events[:2000]:
                if not isinstance(ev, dict):
                    continue
                action = str(ev.get("action") or "").upper()
                et = str(ev.get("event_type") or "").upper()
                if action == "BUY" and et and et != "ENTRY":
                    failures.append(f"{name}: trade_event BUY has non-ENTRY event_type ({et})")
                    break
                if action == "SELL" and et == "ENTRY":
                    failures.append(f"{name}: trade_event SELL has ENTRY event_type")
                    break

    con = _connect(db_path, ro=True)
    try:
        approved_ohlcv = {s.strip().lower() for s in str(args.approved_ohlcv_sources).split(",") if s.strip()}
        approved_trade = {s.strip().lower() for s in str(args.approved_trade_sources).split(",") if s.strip()}
        approved_signal = {s.strip().upper() for s in str(args.approved_signal_sources).split(",") if s.strip()}

        ohlcv_sources = _distinct_sources(con, "ohlcv_data", "source")
        trade_sources = _distinct_sources(con, "trade_executions", "data_source")
        signal_sources = _distinct_sources(con, "trading_signals", "source")

        unknown_ohlcv = [s for s in ohlcv_sources.keys() if s and s.strip().lower() not in approved_ohlcv]
        if unknown_ohlcv:
            failures.append(f"portfolio DB: unknown ohlcv_data.source values: {unknown_ohlcv}")

        unknown_trade = [s for s in trade_sources.keys() if s and s.strip().lower() not in approved_trade]
        if unknown_trade:
            failures.append(f"portfolio DB: unknown trade_executions.data_source values: {unknown_trade}")

        missing_trade = int(trade_sources.get("", 0))
        if missing_trade:
            msg = f"portfolio DB: trade_executions.data_source is NULL/empty for {missing_trade} rows"
            if args.fail_on_missing_trade_source:
                failures.append(msg)
            else:
                warnings.append(msg)

        unknown_signal = [s for s in signal_sources.keys() if s and s.strip().upper() not in approved_signal]
        if unknown_signal:
            failures.append(f"portfolio DB: unknown trading_signals.source values: {unknown_signal}")

        # Payload-level trade data_source checks (if present in payload).
        payload_trade_sources: Set[str] = set()
        payload_trade_missing = 0
        for ev in (dash.get("trade_events") or []):
            if not isinstance(ev, dict):
                continue
            ds = str(ev.get("data_source") or "").strip()
            if not ds:
                payload_trade_missing += 1
                continue
            payload_trade_sources.add(ds.lower())
        unknown_payload_trade = [s for s in sorted(payload_trade_sources) if s not in approved_trade]
        if unknown_payload_trade:
            failures.append(f"dashboard payload: unknown trade_events[].data_source values: {unknown_payload_trade}")
        if payload_trade_missing:
            warnings.append(f"dashboard payload: trade_events missing data_source for {payload_trade_missing} events")
    finally:
        con.close()

    print("=== Dashboard Payload Source Audit ===")
    print(f"dashboard_json: {dash_json_path}")
    print(f"audit_db_path : {audit_db_path}")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f" - {w}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f" - {f}")
    else:
        print("\nOK: no blocking audit failures detected.")

    sys.exit(_failures_to_exit_code(failures))


if __name__ == "__main__":
    main()
