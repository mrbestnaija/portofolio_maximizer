"""
Shared helpers for the read-only quality pipeline scripts.

Library-style helper only: no CLI entrypoint.
"""
from __future__ import annotations

import json
import logging
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


def configure_cli_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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


def _same_path(left: Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return Path(left) == Path(right)


def resolve_forecast_audit_dir(
    requested_audit_dir: Path,
    *,
    default_audit_root: Path,
    default_audit_production_dir: Path,
) -> Path:
    """
    Resolve the canonical audit directory with production-first semantics:
    use production when present, else fallback to legacy root.
    """
    requested = Path(requested_audit_dir)
    if _same_path(requested, default_audit_production_dir) and not requested.exists():
        return Path(default_audit_root)
    return requested


def resolve_forecast_audit_roots(
    audit_dir: Path,
    *,
    include_research: bool,
    default_audit_root: Path,
) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        key = Path(path)
        try:
            key = key.resolve()
        except Exception:
            pass
        if key in seen:
            return
        seen.add(key)
        roots.append(Path(path))

    audit_dir = Path(audit_dir)
    _add(audit_dir)

    if include_research:
        if audit_dir.name.lower() == "production":
            research_dir = audit_dir.parent / "research"
            if research_dir != audit_dir:
                _add(research_dir)
        elif _same_path(audit_dir, default_audit_root):
            _add(audit_dir / "research")
        else:
            sibling_research = audit_dir.parent / "research"
            if sibling_research != audit_dir:
                _add(sibling_research)

    return roots


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def compute_lifecycle_integrity_metrics(db_path: Path) -> dict[str, Any]:
    """
    Shared lifecycle integrity probe used by readiness and production gate paths.

    Returns:
      close_before_entry_count
      closed_missing_exit_reason_count
      high_integrity_violation_count
      query_error (None on success)
    """
    result: dict[str, Any] = {
        "close_before_entry_count": 0,
        "closed_missing_exit_reason_count": 0,
        "high_integrity_violation_count": 0,
        "query_error": None,
    }
    if not Path(db_path).exists():
        result["query_error"] = f"db_not_found:{db_path}"
        return result

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        close_before_entry = 0
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM production_closed_trades c
                LEFT JOIN trade_executions e ON c.entry_trade_id = e.id
                WHERE c.entry_trade_id IS NOT NULL
                  AND c.bar_timestamp IS NOT NULL
                  AND e.bar_timestamp IS NOT NULL
                  AND c.bar_timestamp < e.bar_timestamp
                  AND COALESCE(c.ts_signal_id, '') NOT LIKE 'legacy_%'
                  AND COALESCE(e.ts_signal_id, '') NOT LIKE 'legacy_%'
                """
            ).fetchone()
            close_before_entry = _safe_int(row["n"] if row else 0)
        except sqlite3.OperationalError:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM production_closed_trades c
                LEFT JOIN trade_executions e ON c.entry_trade_id = e.id
                WHERE c.entry_trade_id IS NOT NULL
                  AND c.trade_date IS NOT NULL
                  AND e.trade_date IS NOT NULL
                  AND c.trade_date < e.trade_date
                  AND COALESCE(c.ts_signal_id, '') NOT LIKE 'legacy_%'
                  AND COALESCE(e.ts_signal_id, '') NOT LIKE 'legacy_%'
                """
            ).fetchone()
            close_before_entry = _safe_int(row["n"] if row else 0)

        missing_exit_reason = 0
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM production_closed_trades
                WHERE COALESCE(TRIM(exit_reason), '') = ''
                """
            ).fetchone()
            missing_exit_reason = _safe_int(row["n"] if row else 0)
        except sqlite3.OperationalError:
            missing_exit_reason = 0

        result["close_before_entry_count"] = close_before_entry
        result["closed_missing_exit_reason_count"] = missing_exit_reason
        result["high_integrity_violation_count"] = close_before_entry + missing_exit_reason
        return result
    except Exception as exc:
        result["query_error"] = str(exc)
        return result
    finally:
        if conn is not None:
            conn.close()
