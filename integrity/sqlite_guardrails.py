"""SQLite runtime guardrails for operational connections.

Goal: reduce blast radius from arbitrary SQL execution in runtime paths by
enforcing:
1) connection defensive settings
2) authorizer-based sandboxing for dangerous actions
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Iterable, Set

# PRAGMAs that can weaken integrity guarantees or alter schema behavior.
BLOCKED_PRAGMAS: Set[str] = {
    "ignore_check_constraints",
    "writable_schema",
    "trusted_schema",
    "foreign_keys",
    "legacy_alter_table",
    "recursive_triggers",
    "schema_version",
    "user_version",
    "application_id",
    "journal_mode",
    "locking_mode",
    "synchronous",
    "wal_checkpoint",
    "temp_store",
    "mmap_size",
    "cache_size",
    "page_size",
    "secure_delete",
}

# Read-only PRAGMAs required by operational code paths.
ALLOWED_READ_PRAGMAS: Set[str] = {
    "table_info",
    "database_list",
    "quick_check",
    "integrity_check",
    "index_list",
    "index_info",
    "index_xinfo",
    "foreign_key_list",
    "compile_options",
}


def _safe_setconfig(conn: sqlite3.Connection, option: str, value: int) -> bool:
    option_value = getattr(sqlite3, option, None)
    if option_value is None:
        return False
    try:
        conn.setconfig(option_value, int(value))
        return True
    except Exception:
        return False


def _apply_defensive_db_config(conn: sqlite3.Connection) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    try:
        conn.enable_load_extension(False)
        results["enable_load_extension(false)"] = True
    except Exception:
        results["enable_load_extension(false)"] = False

    # SQLite defensive mode blocks several foot-guns at engine level.
    results["dbconfig_defensive"] = _safe_setconfig(conn, "SQLITE_DBCONFIG_DEFENSIVE", 1)
    # Avoid trusting schema-defined SQL functions.
    results["dbconfig_trusted_schema_off"] = _safe_setconfig(
        conn, "SQLITE_DBCONFIG_TRUSTED_SCHEMA", 0
    )
    # Explicitly disable extension loading in db config too.
    results["dbconfig_disable_load_extension"] = _safe_setconfig(
        conn, "SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION", 0
    )
    return results


def _action_codes(names: Iterable[str]) -> Set[int]:
    out: Set[int] = set()
    for name in names:
        value = getattr(sqlite3, name, None)
        if isinstance(value, int):
            out.add(value)
    return out


def apply_sqlite_guardrails(
    conn: sqlite3.Connection,
    *,
    allow_schema_changes: bool = False,
    extra_allowed_read_pragmas: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Apply runtime guardrails to an open SQLite connection.

    Args:
        conn: Open sqlite3 connection.
        allow_schema_changes: If True, CREATE/ALTER/DROP actions remain allowed.
        extra_allowed_read_pragmas: Additional read-only PRAGMA names to allow.
    """

    allowed_pragmas = {p.lower() for p in ALLOWED_READ_PRAGMAS}
    if extra_allowed_read_pragmas:
        allowed_pragmas.update(str(p).strip().lower() for p in extra_allowed_read_pragmas if str(p).strip())

    blocked_pragmas = {p.lower() for p in BLOCKED_PRAGMAS}

    ddl_actions = _action_codes(
        {
            "SQLITE_ALTER_TABLE",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_VIEW",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_VIEW",
        }
    )
    sandbox_actions = _action_codes({"SQLITE_ATTACH", "SQLITE_DETACH"})
    pragma_action = getattr(sqlite3, "SQLITE_PRAGMA", None)
    sqlite_ok = getattr(sqlite3, "SQLITE_OK", 0)
    sqlite_deny = getattr(sqlite3, "SQLITE_DENY", 1)

    def _authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        inner_trigger_or_view: str | None,
    ) -> int:
        del db_name, inner_trigger_or_view  # unused but required by sqlite API.

        if action == pragma_action:
            pragma_name = (arg1 or "").strip().lower()
            if pragma_name in blocked_pragmas:
                return sqlite_deny
            if pragma_name not in allowed_pragmas:
                return sqlite_deny

        if action in sandbox_actions:
            return sqlite_deny
        if not allow_schema_changes and action in ddl_actions:
            return sqlite_deny
        return sqlite_ok

    conn.set_authorizer(_authorizer)
    config_results = _apply_defensive_db_config(conn)
    return {
        "authorizer_installed": True,
        "allow_schema_changes": bool(allow_schema_changes),
        "allowed_read_pragmas": sorted(allowed_pragmas),
        "blocked_pragmas": sorted(blocked_pragmas),
        "defensive_config": config_results,
    }


def guarded_sqlite_connect(
    database: str,
    *,
    allow_schema_changes: bool = False,
    extra_allowed_read_pragmas: Iterable[str] | None = None,
    enable_guardrails: bool | None = None,
    **connect_kwargs: Any,
) -> sqlite3.Connection:
    """Open SQLite connection with guardrails applied by default.

    Set ``enable_guardrails=False`` for controlled maintenance tooling.
    """
    conn = sqlite3.connect(database, **connect_kwargs)
    use_guardrails = enable_guardrails
    if use_guardrails is None:
        use_guardrails = os.environ.get("SECURITY_SQLITE_GUARDRAILS", "1").strip() != "0"
    if use_guardrails:
        apply_sqlite_guardrails(
            conn,
            allow_schema_changes=allow_schema_changes,
            extra_allowed_read_pragmas=extra_allowed_read_pragmas,
        )
    return conn
