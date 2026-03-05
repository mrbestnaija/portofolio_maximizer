"""
Shared helpers for the read-only quality pipeline scripts.

Library-style helper only: no CLI entrypoint.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def connect_ro(db_path: Path) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(str(db_path))
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def sqlite_master_names(conn: sqlite3.Connection, object_type: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?",
        (object_type,),
    ).fetchall()
    names: set[str] = set()
    for row in rows:
        value = row["name"] if isinstance(row, sqlite3.Row) else row[0]
        if value:
            names.add(str(value))
    return names


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    except sqlite3.DatabaseError:
        return set()
    cols: set[str] = set()
    for row in rows:
        try:
            cols.add(str(row[1]))
        except Exception:
            continue
    return cols


def first_existing_columns(columns: set[str], candidates: Iterable[str]) -> list[str]:
    return [name for name in candidates if name in columns]


def coalesce_expr(alias: str, columns: Iterable[str]) -> str:
    cols = [f"{alias}.{name}" for name in columns]
    if not cols:
        return "NULL"
    if len(cols) == 1:
        return cols[0]
    return "COALESCE(" + ", ".join(cols) + ")"


def has_production_closed_trades_view(conn: sqlite3.Connection) -> bool:
    return "production_closed_trades" in sqlite_master_names(conn, "view")


def production_closed_trades_sql(table_alias: str = "te") -> str:
    return (
        f"FROM trade_executions {table_alias} "
        "WHERE "
        f"{table_alias}.is_close = 1 "
        f"AND {table_alias}.realized_pnl IS NOT NULL "
        f"AND COALESCE({table_alias}.is_diagnostic, 0) = 0 "
        f"AND COALESCE({table_alias}.is_synthetic, 0) = 0"
    )


def load_json_dict(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    target = Path(path)
    if not target.exists():
        return None, "missing"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None, "unreadable"
    if not isinstance(payload, dict):
        return None, "invalid"
    return payload, None


def _extract_threshold_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("thresholds", "source_thresholds", "thresholds_used"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("thresholds")
        if isinstance(value, dict):
            return value
    return None


def append_threshold_hash_change_warning(output_path: Path, payload: dict[str, Any]) -> None:
    current_thresholds = _extract_threshold_block(payload)
    if not isinstance(current_thresholds, dict):
        return
    current_hashes = current_thresholds.get("source_hashes")
    if not isinstance(current_hashes, dict) or not current_hashes:
        return

    existing, error = load_json_dict(output_path)
    if error or not isinstance(existing, dict):
        return
    previous_thresholds = _extract_threshold_block(existing)
    if not isinstance(previous_thresholds, dict):
        return
    previous_hashes = previous_thresholds.get("source_hashes")
    if not isinstance(previous_hashes, dict) or not previous_hashes:
        return
    if previous_hashes == current_hashes:
        return

    warnings = payload.setdefault("warnings", [])
    if isinstance(warnings, list) and "threshold_source_hash_changed" not in warnings:
        warnings.append("threshold_source_hash_changed")
