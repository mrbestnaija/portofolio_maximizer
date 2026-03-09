"""
Shared helpers for the read-only quality pipeline scripts.

Library-style helper only: no CLI entrypoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from forcester_ts.residual_ensemble import (
    CANONICAL_FIELDS as _RESIDUAL_EXPERIMENT_CANONICAL_FIELDS,
    inactive_artifact as _residual_inactive_artifact,
)


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


def _audit_window_fingerprint(audit: dict[str, Any]) -> str:
    ds = audit.get("dataset", {})
    summary = audit.get("summary", {})
    payload = {
        "ticker": ds.get("ticker"),
        "start": ds.get("start"),
        "end": ds.get("end"),
        "length": ds.get("length"),
        "horizon": summary.get("forecast_horizon") or ds.get("forecast_horizon"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_forecast_audit_windows(
    audit_dir: Path,
    *,
    dedupe: bool = True,
    parse_error_sample_limit: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load forecast_audit JSON files with compact parse diagnostics."""
    root = Path(audit_dir)
    files = sorted(root.glob("forecast_audit_*.json"))
    raw: list[dict[str, Any]] = []
    parse_error_files: list[str] = []

    for audit_file in files:
        try:
            payload = json.loads(audit_file.read_text(encoding="utf-8"))
        except Exception:
            parse_error_files.append(audit_file.name)
            continue
        if not isinstance(payload, dict):
            parse_error_files.append(audit_file.name)
            continue
        payload["_path"] = audit_file.name
        payload["_mtime"] = audit_file.stat().st_mtime
        raw.append(payload)

    windows: list[dict[str, Any]]
    duplicates_removed = 0
    if not dedupe:
        windows = raw
    else:
        seen: dict[str, dict[str, Any]] = {}
        for audit in sorted(raw, key=lambda a: (a.get("_mtime", 0), a.get("_path", ""))):
            seen[_audit_window_fingerprint(audit)] = audit
        windows = list(seen.values())
        duplicates_removed = max(0, len(raw) - len(windows))

    stats = {
        "audit_files_scanned": len(files),
        "audit_files_loaded": len(raw),
        "parse_error_count": len(parse_error_files),
        "parse_error_samples": parse_error_files[: max(0, int(parse_error_sample_limit))],
        "dedupe_enabled": bool(dedupe),
        "deduped_windows": len(windows),
        "duplicates_removed": duplicates_removed,
    }
    return windows, stats


# Canonical schema is owned by Agent A's residual module.
# Keep deterministic key order from inactive_artifact() while enforcing
# set-level equality with exported CANONICAL_FIELDS.
_RESIDUAL_EXPERIMENT_CANONICAL_FIELDS_ORDERED: tuple[str, ...] = tuple(
    _residual_inactive_artifact(reason="schema_probe").keys()
)
if frozenset(_RESIDUAL_EXPERIMENT_CANONICAL_FIELDS_ORDERED) != frozenset(
    _RESIDUAL_EXPERIMENT_CANONICAL_FIELDS
):
    raise RuntimeError(
        "Residual experiment canonical field mismatch: "
        "inactive_artifact() keys differ from CANONICAL_FIELDS"
    )
RESIDUAL_EXPERIMENT_CANONICAL_FIELDS: tuple[str, ...] = (
    _RESIDUAL_EXPERIMENT_CANONICAL_FIELDS_ORDERED
)
RESIDUAL_EXPERIMENT_FIELDS: tuple[str, ...] = RESIDUAL_EXPERIMENT_CANONICAL_FIELDS
RESIDUAL_EXPERIMENT_CONTRACT_VERSION = "exp-r5-001.v1"
RESIDUAL_EXPERIMENT_CONTRACT_PATH = "Documentation/EXP_R5_001_ARTIFACT_CONTRACT.md"


def residual_experiment_contract_example() -> dict[str, Any]:
    return {
        "contract_version": RESIDUAL_EXPERIMENT_CONTRACT_VERSION,
        "contract_path": RESIDUAL_EXPERIMENT_CONTRACT_PATH,
        "artifacts": {
            "residual_experiment": {
                "y_hat_anchor": [405.10, 406.20, 407.05],
                "y_hat_residual_ensemble": [404.92, 405.97, 406.88],
                "rmse_anchor": 1.8421,
                "rmse_residual_ensemble": 1.6315,
                "rmse_ratio": 0.8857,
                "da_anchor": 0.5333,
                "da_residual_ensemble": 0.6,
                "corr_anchor_residual": 0.8742,
            }
        },
    }


def _finite_float(value: Any) -> float | None:
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cast):
        return None
    return cast


def _finite_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for item in value:
        cast = _finite_float(item)
        if cast is None:
            return None
        out.append(cast)
    return out or None


def _parse_optional_float_list(
    source: dict[str, Any],
    key: str,
    errors: list[str],
) -> list[float] | None:
    if key not in source:
        return None
    raw_value = source.get(key)
    if raw_value is None:
        return None
    parsed = _finite_float_list(raw_value)
    if parsed is None:
        errors.append(f"invalid_{key}")
    return parsed


def _parse_optional_float(
    source: dict[str, Any],
    key: str,
    errors: list[str],
) -> float | None:
    if key not in source:
        return None
    raw_value = source.get(key)
    if raw_value is None:
        return None
    parsed = _finite_float(raw_value)
    if parsed is None:
        errors.append(f"invalid_{key}")
    return parsed


def _parse_optional_bool(
    source: dict[str, Any],
    key: str,
    errors: list[str],
) -> bool | None:
    if key not in source:
        return None
    raw_value = source.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    errors.append(f"invalid_{key}")
    return None


def _parse_optional_str(
    source: dict[str, Any],
    key: str,
    errors: list[str],
) -> str | None:
    if key not in source:
        return None
    raw_value = source.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return raw_value
    errors.append(f"invalid_{key}")
    return None


def _parse_optional_int(
    source: dict[str, Any],
    key: str,
    errors: list[str],
) -> int | None:
    if key not in source:
        return None
    raw_value = source.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        errors.append(f"invalid_{key}")
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        errors.append(f"invalid_{key}")
        return None
    return value


def _parse_optional_dict(
    source: dict[str, Any],
    key: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if key not in source:
        return None
    raw_value = source.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    errors.append(f"invalid_{key}")
    return None


def extract_residual_experiment_diagnostics(
    audit_payload: dict[str, Any],
    *,
    anchor_model: str = "mssa_rl",
    residual_model: str = "residual_ensemble",
) -> dict[str, Any]:
    """Extract residual metrics and classify payload state (present/missing/malformed)."""
    fields: dict[str, Any] = {key: None for key in RESIDUAL_EXPERIMENT_FIELDS}
    errors: list[str] = []
    if not isinstance(audit_payload, dict):
        return {
            "fields": fields,
            "has_experiment_signal": False,
            "malformed": True,
            "errors": ["audit_payload_not_dict"],
        }

    artifacts = audit_payload.get("artifacts")
    if artifacts is None:
        artifacts = {}
    if not isinstance(artifacts, dict):
        return {
            "fields": fields,
            "has_experiment_signal": False,
            "malformed": True,
            "errors": ["artifacts_not_object"],
        }

    residual_block_raw = artifacts.get("residual_experiment")
    has_residual_block_key = "residual_experiment" in artifacts
    residual_block: dict[str, Any] = {}
    if residual_block_raw is None:
        residual_block = {}
    elif isinstance(residual_block_raw, dict):
        residual_block = residual_block_raw
    else:
        errors.append("residual_experiment_not_object")

    fields["experiment_id"] = _parse_optional_str(residual_block, "experiment_id", errors)
    fields["anchor_model_id"] = _parse_optional_str(residual_block, "anchor_model_id", errors)
    fields["phase"] = _parse_optional_int(residual_block, "phase", errors)
    fields["residual_status"] = _parse_optional_str(residual_block, "residual_status", errors)
    fields["residual_active"] = _parse_optional_bool(residual_block, "residual_active", errors)
    fields["reason"] = _parse_optional_str(residual_block, "reason", errors)

    fields["y_hat_anchor"] = _parse_optional_float_list(residual_block, "y_hat_anchor", errors)
    fields["y_hat_residual_ensemble"] = _parse_optional_float_list(
        residual_block,
        "y_hat_residual_ensemble",
        errors,
    )
    fields["rmse_anchor"] = _parse_optional_float(residual_block, "rmse_anchor", errors)
    fields["rmse_residual_ensemble"] = _parse_optional_float(
        residual_block,
        "rmse_residual_ensemble",
        errors,
    )
    fields["rmse_ratio"] = _parse_optional_float(residual_block, "rmse_ratio", errors)
    fields["da_anchor"] = _parse_optional_float(residual_block, "da_anchor", errors)
    fields["da_residual_ensemble"] = _parse_optional_float(
        residual_block,
        "da_residual_ensemble",
        errors,
    )
    fields["corr_anchor_residual"] = _parse_optional_float(
        residual_block,
        "corr_anchor_residual",
        errors,
    )
    fields["residual_mean"] = _parse_optional_float(residual_block, "residual_mean", errors)
    fields["residual_std"] = _parse_optional_float(residual_block, "residual_std", errors)
    fields["n_corrected"] = _parse_optional_int(residual_block, "n_corrected", errors)
    fields["phi_hat"] = _parse_optional_float(residual_block, "phi_hat", errors)
    fields["intercept_hat"] = _parse_optional_float(residual_block, "intercept_hat", errors)
    fields["n_train_residuals"] = _parse_optional_int(residual_block, "n_train_residuals", errors)
    fields["oos_n_used"] = _parse_optional_int(residual_block, "oos_n_used", errors)
    fields["skip_reason"] = _parse_optional_str(residual_block, "skip_reason", errors)
    fields["promotion_contract"] = _parse_optional_dict(residual_block, "promotion_contract", errors)

    eval_metrics = artifacts.get("evaluation_metrics")
    anchor_eval: dict[str, Any] | None = None
    residual_eval: dict[str, Any] | None = None
    if isinstance(eval_metrics, dict):
        anchor_eval_raw = eval_metrics.get(anchor_model)
        if anchor_eval_raw is None:
            anchor_eval = None
        elif isinstance(anchor_eval_raw, dict):
            anchor_eval = anchor_eval_raw
        else:
            errors.append(f"{anchor_model}_metrics_not_object")

        residual_eval_raw = eval_metrics.get(residual_model)
        if residual_eval_raw is None:
            residual_eval = None
        elif isinstance(residual_eval_raw, dict):
            residual_eval = residual_eval_raw
        else:
            errors.append(f"{residual_model}_metrics_not_object")

    has_experiment_signal = has_residual_block_key or residual_eval is not None or bool(
        fields["y_hat_residual_ensemble"]
    )
    if not has_experiment_signal:
        return {
            "fields": fields,
            "has_experiment_signal": False,
            "malformed": False,
            "errors": [],
        }

    if anchor_eval is not None:
        if fields["rmse_anchor"] is None:
            fields["rmse_anchor"] = _finite_float(anchor_eval.get("rmse"))
        if fields["da_anchor"] is None:
            fields["da_anchor"] = _finite_float(
                anchor_eval.get("directional_accuracy", anchor_eval.get("da"))
            )

    if residual_eval is not None:
        if fields["rmse_residual_ensemble"] is None:
            fields["rmse_residual_ensemble"] = _finite_float(residual_eval.get("rmse"))
        if fields["da_residual_ensemble"] is None:
            fields["da_residual_ensemble"] = _finite_float(
                residual_eval.get("directional_accuracy", residual_eval.get("da"))
            )

    if fields["rmse_ratio"] is None:
        rmse_anchor = fields.get("rmse_anchor")
        rmse_residual = fields.get("rmse_residual_ensemble")
        if isinstance(rmse_anchor, float) and isinstance(rmse_residual, float) and rmse_anchor > 0:
            fields["rmse_ratio"] = _finite_float(rmse_residual / rmse_anchor)

    return {
        "fields": fields,
        "has_experiment_signal": True,
        "malformed": bool(errors),
        "errors": sorted(set(errors)),
    }


def extract_residual_experiment_fields(
    audit_payload: dict[str, Any],
    *,
    anchor_model: str = "mssa_rl",
    residual_model: str = "residual_ensemble",
) -> dict[str, Any]:
    """Extract optional EXP-R5-001 residual metrics from a forecast audit payload."""
    return extract_residual_experiment_diagnostics(
        audit_payload,
        anchor_model=anchor_model,
        residual_model=residual_model,
    )["fields"]


def residual_experiment_metrics_present(fields: dict[str, Any]) -> bool:
    if not isinstance(fields, dict):
        return False
    # RC5: only realized/scalar residual metrics count as "present";
    # structural forecast vectors never qualify as measured evidence.
    for key in (
        "rmse_anchor",
        "rmse_residual_ensemble",
        "rmse_ratio",
        "da_anchor",
        "da_residual_ensemble",
        "corr_anchor_residual",
    ):
        if fields.get(key) is not None:
            return True
    return False


def residual_experiment_window_active(fields: dict[str, Any]) -> bool:
    """Return True only when the canonical EXP-R5-001 active markers are present."""
    if not isinstance(fields, dict):
        return False
    return bool(
        fields.get("experiment_id") == "EXP-R5-001"
        and fields.get("anchor_model_id") == "mssa_rl"
        and fields.get("residual_status") == "active"
        and fields.get("residual_active") is True
    )
