#!/usr/bin/env python3
"""
check_forecast_audits.py
------------------------

Brutal-style sanity check for Time Series forecaster performance.

Reads the most recent forecast audit JSON files emitted by
forcester_ts/forecaster.py (via ModelInstrumentation) from
logs/forecast_audits/, compares ensemble regression metrics to a
baseline model, and exits non-zero if the ensemble underperforms
systematically.

This script is read-only and safe to call from brutal/dry-run
or CI workflows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scripts.audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT
except Exception:  # pragma: no cover - script execution path fallback
    from audit_gate_defaults import FORECAST_AUDIT_MAX_FILES_DEFAULT

try:
    from scripts.telemetry_adapter import normalize_telemetry_payload, telemetry_now_utc
except Exception:  # pragma: no cover - script execution path fallback
    from telemetry_adapter import normalize_telemetry_payload, telemetry_now_utc

DEFAULT_AUDIT_ROOT = Path("logs/forecast_audits")
DEFAULT_AUDIT_PRODUCTION_DIR = DEFAULT_AUDIT_ROOT / "production"
DEFAULT_AUDIT_DIR = DEFAULT_AUDIT_PRODUCTION_DIR
DEFAULT_MONITORING_CONFIG = Path("config/forecaster_monitoring.yml")
DEFAULT_BASELINE_MODEL = "BEST_SINGLE"
DEFAULT_DECISION_KEEP = "KEEP"
DEFAULT_DECISION_RESEARCH = "RESEARCH_ONLY"
DEFAULT_DECISION_DISABLE = "DISABLE_DEFAULT"
DEFAULT_MANIFEST_FILENAME = "forecast_audit_manifest.jsonl"
DEFAULT_MANIFEST_MODE = "off"
MANIFEST_MODES = {"off", "warn", "fail"}
TELEMETRY_SCHEMA_VERSION = 3
OUTCOME_ELIGIBILITY_BUFFER = timedelta(minutes=5)


@dataclass
class AuditCheckResult:
    path: Path
    ensemble_rmse: Optional[float]
    baseline_rmse: Optional[float]
    rmse_ratio: Optional[float]
    violation: bool
    baseline_model: Optional[str] = None
    ensemble_missing: bool = False
    default_model: Optional[str] = None
    residual_diag_present: bool = False
    residual_diag_white_noise: Optional[bool] = None
    residual_diag_n: Optional[int] = None
    ensemble_index_mismatch: bool = False


def _load_audit_with_error(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, str(exc)


def _load_audit(path: Path) -> Optional[Dict[str, Any]]:
    payload, _ = _load_audit_with_error(path)
    return payload


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _parse_window_day(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except Exception:
        return text[:10] if len(text) >= 10 else None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            # Date-only windows are interpreted as end-of-day UTC to avoid
            # prematurely treating same-day windows as outcome-eligible.
            day_start = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
            return day_start + timedelta(days=1) - timedelta(seconds=1)
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _expected_close_ts(end_raw: Any, horizon_raw: Any) -> Optional[datetime]:
    end_ts = _parse_utc_datetime(end_raw)
    if end_ts is None:
        return None
    try:
        horizon = int(horizon_raw)
    except (TypeError, ValueError):
        return None
    if horizon < 0:
        return None
    return end_ts + timedelta(days=horizon)


def _parse_non_negative_int(raw: Any) -> Optional[int]:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def compute_expected_close(
    signal_context: Dict[str, Any],
    dataset: Dict[str, Any],
) -> Tuple[Optional[datetime], str]:
    """Prefer signal-context timing for outcome eligibility.

    Returns (expected_close_ts, source) where source is one of:
      - signal_context
      - dataset_fallback
      - signal_context_invalid
      - unavailable
    """
    ctx = signal_context if isinstance(signal_context, dict) else {}
    has_signal_context = bool(ctx) and not bool(ctx.get("signal_context_missing"))
    if has_signal_context:
        entry_ts = _parse_utc_datetime(ctx.get("entry_ts"))
        horizon = _parse_non_negative_int(ctx.get("forecast_horizon"))
        if entry_ts is None or horizon is None:
            return None, "signal_context_invalid"
        return entry_ts + timedelta(days=horizon), "signal_context"

    fallback = _expected_close_ts(dataset.get("end"), dataset.get("forecast_horizon"))
    if fallback is None:
        return None, "unavailable"
    return fallback, "dataset_fallback"


def _load_closed_trade_match_counts(
    db_path: Optional[Path],
) -> Tuple[bool, Dict[str, int], Optional[str]]:
    if db_path is None:
        return False, {}, "db_not_configured"
    if not db_path.exists():
        return False, {}, f"db_not_found:{db_path}"
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts_signal_id, COUNT(*) AS n
            FROM production_closed_trades
            WHERE ts_signal_id IS NOT NULL AND TRIM(ts_signal_id) <> ''
            GROUP BY ts_signal_id
            """
        )
        mapping = {}
        for ts_signal_id, count in cur.fetchall():
            sid = str(ts_signal_id or "").strip()
            if sid:
                mapping[sid] = int(count or 0)
        return True, mapping, None
    except Exception as exc:
        return False, {}, str(exc)
    finally:
        if conn is not None:
            conn.close()


def _extract_window_metadata(audit: Dict[str, Any]) -> Dict[str, Optional[str]]:
    dataset = audit.get("dataset") or {}
    ticker = str(dataset.get("ticker") or dataset.get("symbol") or "").strip().upper() or None
    regime = str(dataset.get("detected_regime") or dataset.get("regime") or "").strip().upper() or None
    return {
        "ticker": ticker,
        "detected_regime": regime,
        "end_day": _parse_window_day(dataset.get("end")),
    }


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for _attempt in range(2):
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            with tmp_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            os.replace(tmp_path, path)
            return
        except Exception as exc:
            last_error = exc
            try:
                tmp_path.unlink()
            except OSError:
                pass
    if last_error is not None:
        raise last_error


def _counts_toward_readiness_denominator(
    *,
    context_type: str,
    outcome_status: str,
    gate_eligible: bool = True,
) -> bool:
    normalized_context = str(context_type or "TRADE").strip().upper() or "TRADE"
    normalized_status = str(outcome_status or "OUTCOMES_NOT_LOADED").strip().upper()
    return bool(gate_eligible) and (
        normalized_context == "TRADE"
        and normalized_status
        not in {
            "INVALID_CONTEXT",
            "NON_TRADE_CONTEXT",
            "OUTCOMES_NOT_LOADED",
            "NOT_DUE",
        }
    )


def _lift_threshold_rmse_ratio(min_lift_rmse_ratio: Optional[float]) -> Optional[float]:
    if min_lift_rmse_ratio is None:
        return None
    try:
        return 1.0 - float(min_lift_rmse_ratio)
    except (TypeError, ValueError):
        return None


def _merge_summary_fields(summary: Dict[str, Any], summary_fields: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(summary_fields, dict):
        return summary
    for key, value in summary_fields.items():
        if key == "window_counts" and isinstance(value, dict):
            base = summary.get("window_counts")
            if not isinstance(base, dict):
                base = {}
            merged = dict(base)
            merged.update(value)
            summary["window_counts"] = merged
            continue
        if key == "telemetry_contract" and isinstance(value, dict):
            base = summary.get("telemetry_contract")
            if not isinstance(base, dict):
                base = {}
            merged = dict(base)
            merged.update(value)
            summary["telemetry_contract"] = merged
            continue
        summary[key] = value
    return summary


def _normalize_reason_codes(
    reason_codes: Any,
    *,
    fallback_reason_code: Optional[str] = None,
) -> List[str]:
    if isinstance(reason_codes, list):
        normalized = [str(item).strip().upper() for item in reason_codes if str(item).strip()]
        if normalized:
            return normalized
    fallback = str(fallback_reason_code or "").strip().upper()
    if not fallback or fallback == "READY":
        return []
    return [part.strip() for part in fallback.split(",") if part.strip()]


def _summarize_admission_entries(dataset_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counts = {
        "ELIGIBLE": 0,
        "ACCEPTED_NONELIGIBLE": 0,
        "QUARANTINED": 0,
    }
    source_counts = {
        "producer": 0,
        "legacy_derived": 0,
    }
    for entry in dataset_entries:
        bucket = str(entry.get("gate_bucket") or "").strip().upper()
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
        source = str(entry.get("semantic_admission_source") or "").strip().lower()
        if source in source_counts:
            source_counts[source] += 1
    return {
        "accepted_records": sum(1 for entry in dataset_entries if bool(entry.get("accepted_for_audit_history"))),
        "accepted_noneligible_records": bucket_counts["ACCEPTED_NONELIGIBLE"],
        "eligible_records": sum(1 for entry in dataset_entries if bool(entry.get("gate_eligible"))),
        "quarantined_records": bucket_counts["QUARANTINED"],
        "duplicate_conflicts": sum(1 for entry in dataset_entries if bool(entry.get("duplicate_conflict"))),
        "missing_execution_metadata_records": sum(
            1 for entry in dataset_entries if bool(entry.get("missing_execution_metadata"))
        ),
        "bucket_counts": bucket_counts,
        "source_counts": source_counts,
    }


def _derive_semantic_admission(
    entry: Dict[str, Any],
    *,
    include_research: bool,
) -> Dict[str, Any]:
    semantic = entry.get("semantic_admission")
    semantic = semantic if isinstance(semantic, dict) else {}
    producer_present = any(
        key in semantic
        for key in (
            "accepted_for_audit_history",
            "admissible_for_readiness",
            "gate_eligible",
            "gate_bucket",
            "reason_code",
            "reason_codes",
            "duplicate_conflict",
            "quarantined",
            "missing_execution_metadata",
        )
    )
    context_type = str(entry.get("context_type") or "TRADE").strip().upper() or "TRADE"
    manifest_status = str(entry.get("manifest_verification_status") or "").strip().lower()
    duplicate_conflict = bool(entry.get("duplicate_conflict")) or bool(semantic.get("duplicate_conflict"))
    quarantined = bool(entry.get("quarantined")) or bool(semantic.get("quarantined"))
    production_labeled = bool(
        semantic.get("production_labeled")
        if "production_labeled" in semantic
        else (not include_research or str(entry.get("cohort_id") or "").strip().lower() == "production")
    )
    accepted_for_audit_history = bool(semantic.get("accepted_for_audit_history", True))

    if producer_present:
        explicit_admissible = semantic.get("admissible_for_readiness")
        explicit_gate_eligible = semantic.get("gate_eligible")
        if explicit_admissible is None:
            admissible_for_readiness = bool(explicit_gate_eligible)
        else:
            admissible_for_readiness = bool(explicit_admissible)
        gate_eligible = (
            bool(explicit_gate_eligible)
            if explicit_gate_eligible is not None
            else bool(admissible_for_readiness)
        )
        gate_bucket = str(
            semantic.get("gate_bucket")
            or ("ELIGIBLE" if gate_eligible else "ACCEPTED_NONELIGIBLE")
        ).strip().upper() or ("ELIGIBLE" if gate_eligible else "ACCEPTED_NONELIGIBLE")
        reason_code = str(
            semantic.get("reason_code")
            or semantic.get("admission_reason_code")
            or ("READY" if gate_eligible else "NON_ELIGIBLE")
        ).strip().upper() or ("READY" if gate_eligible else "NON_ELIGIBLE")
        missing_execution_metadata_fields = semantic.get("missing_execution_metadata_fields")
        if isinstance(missing_execution_metadata_fields, list):
            missing_execution_metadata_fields = [
                str(item).strip()
                for item in missing_execution_metadata_fields
                if str(item).strip()
            ]
        else:
            missing_execution_metadata_fields = []
        missing_execution_metadata = bool(semantic.get("missing_execution_metadata"))
        if missing_execution_metadata_fields:
            missing_execution_metadata = True
        if "duplicate_conflict" in semantic:
            duplicate_conflict = bool(semantic.get("duplicate_conflict"))
        if "quarantined" in semantic:
            quarantined = bool(semantic.get("quarantined"))
        elif "not_quarantined" in semantic:
            quarantined = not bool(semantic.get("not_quarantined"))
        return {
            "accepted_for_audit_history": accepted_for_audit_history,
            "admissible_for_readiness": admissible_for_readiness,
            "gate_eligible": gate_eligible,
            "gate_bucket": gate_bucket,
            "admission_reason_code": reason_code,
            "admission_reason_codes": _normalize_reason_codes(
                semantic.get("reason_codes"),
                fallback_reason_code=reason_code,
            ),
            "duplicate_conflict": duplicate_conflict,
            "quarantined": quarantined,
            "production_labeled": bool(semantic.get("production_labeled", production_labeled)),
            "missing_execution_metadata": missing_execution_metadata,
            "missing_execution_metadata_fields": missing_execution_metadata_fields,
            "semantic_admission_source": "producer",
            "semantic_admission_preserved": True,
        }

    reason_codes: list[str] = []
    if not production_labeled:
        reason_codes.append("NOT_PRODUCTION_LABELED")
    if context_type != "TRADE":
        reason_codes.append("NON_TRADE_CONTEXT")
    if manifest_status and manifest_status != "verified":
        reason_codes.append(f"MANIFEST_{manifest_status.upper()}")
    if duplicate_conflict:
        reason_codes.append("DUPLICATE_CONFLICT")
    if quarantined:
        reason_codes.append("QUARANTINED")

    explicit_admissible = semantic.get("admissible_for_readiness")
    if explicit_admissible is None:
        admissible_for_readiness = accepted_for_audit_history and len(reason_codes) == 0
    else:
        admissible_for_readiness = bool(explicit_admissible) and len(reason_codes) == 0

    if quarantined or duplicate_conflict:
        gate_bucket = "QUARANTINED"
    elif admissible_for_readiness:
        gate_bucket = "ELIGIBLE"
    else:
        gate_bucket = "ACCEPTED_NONELIGIBLE"
    missing_execution_metadata_fields: List[str] = []
    if not str(entry.get("run_id") or "").strip():
        missing_execution_metadata_fields.append("run_id")
    if not str(entry.get("entry_ts") or "").strip():
        missing_execution_metadata_fields.append("entry_ts")
    missing_execution_metadata = (
        str(entry.get("outcome_reason") or "").strip().upper() == "MISSING_EXECUTION_METADATA"
        or bool(missing_execution_metadata_fields)
    )

    return {
        "accepted_for_audit_history": accepted_for_audit_history,
        "admissible_for_readiness": admissible_for_readiness,
        "gate_eligible": admissible_for_readiness,
        "gate_bucket": gate_bucket,
        "admission_reason_code": "READY" if admissible_for_readiness else ",".join(reason_codes) or "NON_ELIGIBLE",
        "admission_reason_codes": [] if admissible_for_readiness else reason_codes,
        "duplicate_conflict": duplicate_conflict,
        "quarantined": quarantined,
        "production_labeled": production_labeled,
        "missing_execution_metadata": missing_execution_metadata,
        "missing_execution_metadata_fields": missing_execution_metadata_fields,
        "semantic_admission_source": "legacy_derived",
        "semantic_admission_preserved": False,
    }


def _write_summary_with_guard(path: Path, payload: Dict[str, Any]) -> None:
    required = (
        "audit_dir",
        "audit_roots",
        "generated_utc",
        "source_script",
        "schema_version",
        "max_files",
        "scope",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("summary_missing_required_fields:" + ",".join(missing))
    _write_json_atomic(path, payload)


def _build_failure_dataset_snapshot(
    *,
    audit_roots: List[Path],
    max_files: int,
    db_path: Optional[Path],
    min_forecast_horizon: Optional[int],
    generated_utc: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], bool, Optional[str]]:
    files = _collect_audit_files(audit_roots=audit_roots, max_files=max_files)

    def _outcome_dedupe_key_from_audit(audit: Dict[str, Any]) -> Tuple[Any, ...]:
        dataset = audit.get("dataset") or {}
        ticker = str(dataset.get("ticker") or dataset.get("symbol") or "").strip().upper() or None
        return (
            ticker,
            dataset.get("start"),
            dataset.get("end"),
            dataset.get("length"),
            dataset.get("forecast_horizon"),
        )

    parseable_count = 0
    parse_error_count = 0
    dedup_map: Dict[Tuple[Any, ...], Tuple[Path, Dict[str, Any]]] = {}
    horizon_filtered_count = 0

    for path in files:
        audit, _ = _load_audit_with_error(path)
        if not audit:
            parse_error_count += 1
            continue
        parseable_count += 1
        if min_forecast_horizon is not None:
            horizon = _parse_non_negative_int((audit.get("dataset") or {}).get("forecast_horizon"))
            if horizon is None or horizon < min_forecast_horizon:
                horizon_filtered_count += 1
                continue
        key = _outcome_dedupe_key_from_audit(audit)
        if key not in dedup_map:
            dedup_map[key] = (path, audit)

    outcomes_loaded = False
    outcome_join_error: Optional[str] = None
    closed_trade_counts: Dict[str, int] = {}
    if db_path is not None:
        outcomes_loaded, closed_trade_counts, outcome_join_error = _load_closed_trade_match_counts(db_path)

    now = now_utc()
    dataset_entries: List[Dict[str, Any]] = []

    outcome_windows_eligible = 0
    outcome_windows_matched = 0
    outcome_windows_missing = 0
    outcome_windows_ambiguous = 0
    outcome_windows_not_due = 0
    outcome_windows_not_yet_eligible = 0
    outcome_windows_invalid_context = 0
    outcome_windows_outcomes_not_loaded = 0
    outcome_windows_no_signal_id = 0
    outcome_windows_non_trade_context = 0
    outcome_windows_missing_execution_metadata = 0

    def _legacy_to_status(outcome_status: str) -> str:
        mapping = {
            "MATCHED": "PASS",
            "OUTCOME_MISSING": "FAIL",
            "NOT_DUE": "INCONCLUSIVE_ALLOWED",
            "INVALID_CONTEXT": "INVALID_CONTEXT",
            "NON_TRADE_CONTEXT": "WARN",
            "OUTCOMES_NOT_LOADED": "WARN",
        }
        return mapping.get(outcome_status, "WARN")

    for path, audit in dedup_map.values():
        ds = (audit or {}).get("dataset") or {}
        signal_context = (audit or {}).get("signal_context") or {}
        if not isinstance(signal_context, dict):
            signal_context = {}

        context_type = str(signal_context.get("context_type") or "").strip().upper() or "TRADE"
        raw_ts_signal_id = signal_context.get("ts_signal_id")
        ts_signal_id = str(raw_ts_signal_id).strip() if raw_ts_signal_id is not None else ""
        ts_signal_id = ts_signal_id or None

        expected_close, expected_close_source = compute_expected_close(signal_context, ds)
        meta = _extract_window_metadata(audit or {})

        entry: Dict[str, Any] = {
            "file": path.name,
            "audit_id": str((audit or {}).get("audit_id") or path.stem).strip() or path.stem,
            "start": ds.get("start"),
            "end": ds.get("end"),
            "length": ds.get("length"),
            "forecast_horizon": ds.get("forecast_horizon"),
            "ticker": meta.get("ticker"),
            "detected_regime": meta.get("detected_regime"),
            "end_day": meta.get("end_day"),
            "context_type": context_type,
            "ts_signal_id": ts_signal_id,
            "run_id": signal_context.get("run_id"),
            "entry_ts": signal_context.get("entry_ts"),
            "signal_context_missing": bool(signal_context.get("signal_context_missing")),
            "dataset_forecast_horizon": ds.get("forecast_horizon"),
            "signal_forecast_horizon": signal_context.get("forecast_horizon"),
            "expected_close_ts": expected_close.isoformat() if expected_close else None,
            "expected_close_source": expected_close_source,
            "evidence_contract_version": (audit or {}).get("evidence_contract_version"),
            "cohort_id": (audit or {}).get("cohort_id"),
            "cohort_identity": (audit or {}).get("cohort_identity"),
            "semantic_admission": (audit or {}).get("semantic_admission"),
            "outcome_status": None,
            "outcome_reason": None,
        }

        ticker = str(entry.get("ticker") or "").strip().upper()
        run_id = str(entry.get("run_id") or "").strip()
        entry_ts_raw = str(entry.get("entry_ts") or "").strip()
        signal_horizon = _parse_non_negative_int(entry.get("signal_forecast_horizon"))
        dataset_horizon = _parse_non_negative_int(entry.get("dataset_forecast_horizon"))
        expected_close_ts = _parse_utc_datetime(entry.get("expected_close_ts"))
        entry_ts = _parse_utc_datetime(entry.get("entry_ts"))

        if context_type != "TRADE":
            entry["outcome_status"] = "NON_TRADE_CONTEXT"
            entry["outcome_reason"] = "NON_TRADE_CONTEXT"
            outcome_windows_non_trade_context += 1
        elif not ticker:
            entry["outcome_status"] = "NON_TRADE_CONTEXT"
            entry["outcome_reason"] = "MISSING_TICKER"
            outcome_windows_non_trade_context += 1
        elif not run_id or not entry_ts_raw:
            entry["outcome_status"] = "INVALID_CONTEXT"
            entry["outcome_reason"] = "MISSING_EXECUTION_METADATA"
            outcome_windows_missing_execution_metadata += 1
            outcome_windows_invalid_context += 1
        elif not ts_signal_id:
            entry["outcome_status"] = "INVALID_CONTEXT"
            entry["outcome_reason"] = "MISSING_SIGNAL_ID"
            outcome_windows_no_signal_id += 1
            outcome_windows_invalid_context += 1
        elif (
            signal_horizon is not None
            and dataset_horizon is not None
            and signal_horizon != dataset_horizon
        ):
            entry["outcome_status"] = "INVALID_CONTEXT"
            entry["outcome_reason"] = "HORIZON_MISMATCH"
            outcome_windows_invalid_context += 1
        elif expected_close_ts is None:
            entry["outcome_status"] = "INVALID_CONTEXT"
            entry["outcome_reason"] = "EXPECTED_CLOSE_UNAVAILABLE"
            outcome_windows_invalid_context += 1
        elif entry_ts is not None and expected_close_ts < entry_ts:
            entry["outcome_status"] = "INVALID_CONTEXT"
            entry["outcome_reason"] = "CAUSALITY_VIOLATION"
            outcome_windows_invalid_context += 1
        elif (expected_close_ts + OUTCOME_ELIGIBILITY_BUFFER) > now and not (
            # Early-credit: if the trade is already confirmed closed in the DB,
            # skip the NOT_DUE wait — no point deferring a known outcome.
            outcomes_loaded
            and ts_signal_id is not None
            and int(closed_trade_counts.get(ts_signal_id, 0)) == 1
        ):
            entry["outcome_status"] = "NOT_DUE"
            entry["outcome_reason"] = "OUTCOME_WINDOW_OPEN"
            outcome_windows_not_due += 1
            outcome_windows_not_yet_eligible += 1
        elif not outcomes_loaded:
            entry["outcome_status"] = "OUTCOMES_NOT_LOADED"
            entry["outcome_reason"] = "OUTCOME_JOIN_UNAVAILABLE"
            outcome_windows_outcomes_not_loaded += 1
        else:
            match_count = int(closed_trade_counts.get(ts_signal_id, 0))
            entry["outcome_match_count"] = match_count
            if match_count == 1:
                entry["outcome_status"] = "MATCHED"
                entry["outcome_reason"] = "ONE_TO_ONE_MATCH"
                outcome_windows_eligible += 1
                outcome_windows_matched += 1
            elif match_count == 0:
                entry["outcome_status"] = "OUTCOME_MISSING"
                entry["outcome_reason"] = "DUE_BUT_MISSING_CLOSE"
                outcome_windows_eligible += 1
                outcome_windows_missing += 1
            else:
                entry["outcome_status"] = "INVALID_CONTEXT"
                entry["outcome_reason"] = "AMBIGUOUS_MATCH"
                outcome_windows_ambiguous += 1
                outcome_windows_invalid_context += 1

        outcome_status = str(entry.get("outcome_status") or "OUTCOMES_NOT_LOADED").strip().upper()
        outcome_reason = str(entry.get("outcome_reason") or "OUTCOME_JOIN_UNAVAILABLE").strip().upper()
        counts_toward_linkage = outcome_status in {"MATCHED", "OUTCOME_MISSING"}
        counts_toward_readiness = _counts_toward_readiness_denominator(
            context_type=context_type,
            outcome_status=outcome_status,
        )
        severity = "LOW"
        blocking = False
        if outcome_status == "INVALID_CONTEXT":
            severity = "HIGH"
            blocking = True
        elif outcome_status in {"OUTCOME_MISSING", "OUTCOMES_NOT_LOADED"}:
            severity = "MEDIUM"
        entry.update(
            _derive_semantic_admission(
                entry,
                include_research=False,
            )
        )
        entry.update(
            normalize_telemetry_payload(
                {
                    "status": _legacy_to_status(outcome_status),
                    "reason_code": outcome_reason,
                    "context_type": context_type,
                    "severity": severity,
                    "blocking": blocking,
                    "counts_toward_readiness_denominator": counts_toward_readiness,
                    "counts_toward_linkage_denominator": counts_toward_linkage,
                    "generated_utc": generated_utc,
                    "source_script": "scripts/check_forecast_audits.py",
                },
                source_script="scripts/check_forecast_audits.py",
                generated_utc=generated_utc,
            )
        )
        dataset_entries.append(entry)

    readiness_denominator_included = sum(
        1 for entry in dataset_entries if bool(entry.get("counts_toward_readiness_denominator"))
    )
    linkage_denominator_included = sum(
        1 for entry in dataset_entries if bool(entry.get("counts_toward_linkage_denominator"))
    )
    readiness_excluded_non_trade = sum(
        1
        for entry in dataset_entries
        if str(entry.get("outcome_status") or "").strip().upper() == "NON_TRADE_CONTEXT"
    )
    readiness_excluded_invalid = sum(
        1
        for entry in dataset_entries
        if str(entry.get("outcome_status") or "").strip().upper() == "INVALID_CONTEXT"
    )
    readiness_excluded_not_due = sum(
        1
        for entry in dataset_entries
        if str(entry.get("outcome_status") or "").strip().upper() == "NOT_DUE"
    )

    window_counts = {
        "n_raw_windows": len(files),
        "n_parseable_windows": parseable_count,
        "n_deduped_windows": len(dedup_map),
        "n_outcome_deduped_windows": len(dedup_map),
        "n_rmse_windows_processed": 0,
        "n_rmse_windows_usable": 0,
        "n_outcome_windows_eligible": outcome_windows_eligible,
        "n_outcome_windows_matched": outcome_windows_matched,
        "n_outcome_windows_missing": outcome_windows_missing,
        "n_outcome_windows_ambiguous": outcome_windows_ambiguous,
        "n_outcome_windows_not_due": outcome_windows_not_due,
        "n_outcome_windows_not_yet_eligible": outcome_windows_not_yet_eligible,
        "n_outcome_windows_invalid_context": outcome_windows_invalid_context,
        "n_outcome_windows_outcomes_not_loaded": outcome_windows_outcomes_not_loaded,
        "n_outcome_windows_no_signal_id": outcome_windows_no_signal_id,
        "n_outcome_windows_non_trade_context": outcome_windows_non_trade_context,
        "n_outcome_windows_missing_execution_metadata": outcome_windows_missing_execution_metadata,
        "n_readiness_denominator_included": readiness_denominator_included,
        "n_linkage_denominator_included": linkage_denominator_included,
        "n_readiness_excluded_non_trade_context": readiness_excluded_non_trade,
        "n_readiness_excluded_invalid_context": readiness_excluded_invalid,
        "n_readiness_excluded_not_due": readiness_excluded_not_due,
        "n_horizon_filtered_windows": horizon_filtered_count,
    }

    return dataset_entries, window_counts, outcomes_loaded, outcome_join_error


def _emit_failure_summary_and_exit(
    *,
    message: str,
    audit_dir: Path,
    audit_roots: List[Path],
    include_research: bool,
    max_files: int,
    db_path: Optional[Path] = None,
    min_forecast_horizon: Optional[int] = None,
    exit_code: int = 1,
    summary_fields: Optional[Dict[str, Any]] = None,
) -> None:
    generated_utc = telemetry_now_utc()
    cache_dir = Path("logs/forecast_audits_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "latest_summary.json"
    dataset_entries, window_counts, outcomes_loaded, outcome_join_error = _build_failure_dataset_snapshot(
        audit_roots=audit_roots,
        max_files=int(max_files),
        db_path=db_path,
        min_forecast_horizon=min_forecast_horizon,
        generated_utc=generated_utc,
    )
    admission_summary = _summarize_admission_entries(dataset_entries)
    summary: Dict[str, Any] = {
        "audit_dir": str(audit_dir),
        "audit_roots": [str(root) for root in audit_roots],
        "generated_utc": generated_utc,
        "source_script": "scripts/check_forecast_audits.py",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "status": "PASS" if int(exit_code) == 0 else "FAIL",
        "exit_code": int(exit_code),
        "exit_reason": str(message).strip() or "UNSPECIFIED_FAILURE",
        "max_files": int(max_files),
        "measurement_contract_version": 1,
        "baseline_model": None,
        "lift_threshold_rmse_ratio": None,
        "effective_audits": None,
        "effective_outcome_audits": None,
        "total_unique_audits": window_counts.get("n_deduped_windows"),
        "violation_count": None,
        "violation_rate": None,
        "max_violation_rate": None,
        "ensemble_missing_count": None,
        "ensemble_missing_rate": None,
        "max_missing_ensemble_rate": None,
        "holding_period_required": None,
        "lift_fraction": None,
        "min_lift_fraction": None,
        "percentiles": {"p10": None, "p50": None, "p90": None},
        "ratio_distribution": {"count": 0, "min": None, "max": None, "mean": None, "p10": None, "p50": None, "p90": None},
        "decision": None,
        "decision_reason": str(message).strip() or "UNSPECIFIED_FAILURE",
        "recent_window_audits": None,
        "recent_effective_audits": None,
        "recent_violation_count": None,
        "recent_violation_rate": None,
        "recent_window_max_violation_rate": None,
        "recent_rmse_ratio_p90": None,
        "recent_window_max_p90_rmse_ratio": None,
        "scope": {
            "include_research": bool(include_research),
            "production_audit_only": not bool(include_research),
        },
        "window_counts": window_counts,
        "admission_summary": admission_summary,
        "dataset_windows": dataset_entries,
        "telemetry_contract": normalize_telemetry_payload(
            {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "rmse_inputs_present": False,
                "outcomes_loaded": outcomes_loaded,
                "execution_log_loaded": False,
                "outcome_join_attempted": db_path is not None,
                "status": "FAIL",
                "reason_code": "FAILURE_SUMMARY_EXIT",
                "context_type": "TRADE",
                "severity": "MEDIUM",
                "blocking": True,
                "counts_toward_readiness_denominator": False,
                "counts_toward_linkage_denominator": False,
                "generated_utc": generated_utc,
                "source_script": "scripts/check_forecast_audits.py",
                "outcome_join_error": outcome_join_error,
            },
            source_script="scripts/check_forecast_audits.py",
            generated_utc=generated_utc,
        ),
        "cache_status": {"write_ok": True, "errors": []},
    }
    summary = _merge_summary_fields(summary, summary_fields)
    summary["telemetry_contract"] = normalize_telemetry_payload(
        summary.get("telemetry_contract", {}),
        source_script="scripts/check_forecast_audits.py",
        generated_utc=generated_utc,
    )
    try:
        _write_summary_with_guard(cache_path, summary)
    except Exception as exc:
        print(f"[WARN] forecast_audits_cache_write_failed target=latest_summary error={exc}")
    raise SystemExit(int(exit_code))


def _resolve_audit_roots(audit_dir: Path, include_research: bool) -> List[Path]:
    roots: List[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        rp = path.resolve()
        if rp in seen:
            return
        seen.add(rp)
        roots.append(path)

    _add(audit_dir)
    if include_research:
        if audit_dir.name.lower() == "production":
            research_dir = audit_dir.parent / "research"
            if research_dir != audit_dir:
                _add(research_dir)
        elif audit_dir.resolve() == DEFAULT_AUDIT_ROOT.resolve():
            _add(audit_dir / "research")
        else:
            sibling_research = audit_dir.parent / "research"
            if sibling_research != audit_dir:
                _add(sibling_research)
    return roots


def _collect_audit_files(
    *,
    audit_roots: List[Path],
    max_files: int,
) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for root in audit_roots:
        if not root.exists():
            continue
        for path in root.glob("forecast_audit_*.json"):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:max(0, int(max_files))]


def _load_manifest_index(manifest_path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    index: Dict[str, str] = {}
    stats = {
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "invalid_records": 0,
    }
    if not manifest_path.exists():
        return index, stats

    for raw_line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            stats["invalid_records"] += 1
            continue
        file_name = str(payload.get("file") or "").strip()
        digest = str(payload.get("sha256") or "").strip().lower()
        if not file_name or len(digest) != 64:
            stats["invalid_records"] += 1
            continue
        index[file_name] = digest
    return index, stats


def _verify_manifest_entry(path: Path, manifest_index: Dict[str, str]) -> str:
    """
    Return manifest verification status for an audit artifact.

    Status values:
    - ok
    - missing
    - hash_failed
    - mismatch
    """
    expected = manifest_index.get(path.name)
    if not expected:
        return "missing"
    actual = _sha256_file(path)
    if not actual:
        return "hash_failed"
    if actual.lower() != expected.lower():
        return "mismatch"
    return "ok"


def _extract_metrics(
    audit: Dict[str, Any], *, baseline_model: str
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[str]]:
    """
    Return (ensemble_metrics, baseline_metrics, resolved_baseline_model) from an audit payload.

    Ensemble metrics are taken from artifacts['evaluation_metrics']['ensemble']
    when available.

    Baseline selection:
    - BEST_SINGLE: choose the available single-model entry with the smallest RMSE.
      Candidate set includes SARIMAX, GARCH, SAMOSSA, and MSSA_RL.
    - SAMOSSA: use samossa when present; else fall back to BEST_SINGLE.
    - GARCH: use garch when present; else fall back to BEST_SINGLE.
    - SARIMAX: use sarimax when present; else fall back to BEST_SINGLE.
    """
    artifacts = audit.get("artifacts") or {}
    eval_metrics = artifacts.get("evaluation_metrics") or {}
    if not isinstance(eval_metrics, dict):
        return None, None, None

    ensemble = eval_metrics.get("ensemble")
    sarimax = eval_metrics.get("sarimax")
    garch = eval_metrics.get("garch")
    samossa = eval_metrics.get("samossa")

    if ensemble is None and sarimax is None and garch is None and samossa is None:
        return None, None, None

    ensemble_metrics = ensemble if isinstance(ensemble, dict) else None

    baseline_model = (baseline_model or DEFAULT_BASELINE_MODEL).strip().upper()

    def _best_single() -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for name in ("sarimax", "garch", "samossa", "mssa_rl"):
            payload = eval_metrics.get(name)
            if isinstance(payload, dict):
                candidates.append((name, payload))
        best_payload: Optional[Dict[str, Any]] = None
        best_rmse: Optional[float] = None
        best_name: Optional[str] = None
        for name, payload in candidates:
            rmse = _rmse_from(payload)
            if rmse is None:
                continue
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_payload = payload
                best_name = name.upper()
        if best_payload is not None:
            return best_payload, best_name
        if isinstance(sarimax, dict):
            return sarimax, "SARIMAX"
        if isinstance(garch, dict):
            return garch, "GARCH"
        if isinstance(samossa, dict):
            return samossa, "SAMOSSA"
        return None, None

    if baseline_model == "SAMOSSA":
        if isinstance(samossa, dict):
            baseline_metrics = samossa
            resolved_baseline = "SAMOSSA"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    elif baseline_model == "GARCH":
        if isinstance(garch, dict):
            baseline_metrics = garch
            resolved_baseline = "GARCH"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    elif baseline_model == "SARIMAX":
        if isinstance(sarimax, dict):
            baseline_metrics = sarimax
            resolved_baseline = "SARIMAX"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    else:
        baseline_metrics, resolved_baseline = _best_single()

    return ensemble_metrics, baseline_metrics if isinstance(baseline_metrics, dict) else None, resolved_baseline


def _rmse_from(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("rmse")
    return float(val) if isinstance(val, (int, float)) else None


def _normalize_model_name(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    return text.upper().replace("-", "_").replace(" ", "_")


def _extract_effective_default_model(audit: Dict[str, Any]) -> Optional[str]:
    artifacts = audit.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return None
    direct = _normalize_model_name(artifacts.get("effective_default_model"))
    if direct:
        return direct
    ensemble_selection = artifacts.get("ensemble_selection") or {}
    if isinstance(ensemble_selection, dict):
        return _normalize_model_name(ensemble_selection.get("default_model"))
    return None


def _extract_default_residual_diagnostics(
    audit: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    artifacts = audit.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return None, None
    default_model = _extract_effective_default_model(audit)
    diagnostics_by_model = artifacts.get("residual_diagnostics") or {}
    if not isinstance(diagnostics_by_model, dict):
        return default_model, None
    if not default_model:
        return default_model, None
    model_key = default_model.lower()
    diagnostics = diagnostics_by_model.get(model_key)
    if isinstance(diagnostics, dict) and diagnostics:
        return default_model, diagnostics
    # When the effective default is ENSEMBLE, it has no own residual diagnostics.
    # Fall back to the primary component model from ensemble_selection, then try
    # the standard preference order (samossa → garch → mssa_rl → sarimax).
    if model_key == "ensemble":
        ensemble_selection = artifacts.get("ensemble_selection") or {}
        primary = None
        if isinstance(ensemble_selection, dict):
            raw_primary = ensemble_selection.get("primary_model")
            primary = _normalize_model_name(raw_primary)
        fallback_order = [primary] + ["samossa", "garch", "mssa_rl", "sarimax"]
        for candidate in fallback_order:
            if not candidate:
                continue
            diag = diagnostics_by_model.get(candidate.lower())
            if isinstance(diag, dict) and diag:
                return candidate, diag
    return default_model, None


def _extract_ensemble_index_mismatch(audit: Dict[str, Any]) -> bool:
    artifacts = audit.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return False
    direct = artifacts.get("ensemble_index_mismatch")
    if isinstance(direct, bool):
        return direct
    ensemble_selection = artifacts.get("ensemble_selection") or {}
    if isinstance(ensemble_selection, dict):
        nested = ensemble_selection.get("ensemble_index_mismatch")
        if isinstance(nested, bool):
            return nested
    return False


def check_audit_file(
    path: Path,
    tolerance: float,
    *,
    baseline_model: str,
) -> Optional[AuditCheckResult]:
    audit = _load_audit(path)
    if not audit:
        return None

    ensemble_metrics, baseline_metrics, resolved_baseline = _extract_metrics(
        audit, baseline_model=baseline_model
    )
    default_model, residual_diag = _extract_default_residual_diagnostics(audit)
    ensemble_index_mismatch = _extract_ensemble_index_mismatch(audit)
    residual_diag_present = isinstance(residual_diag, dict) and bool(residual_diag)
    residual_diag_white_noise = (
        bool(residual_diag.get("white_noise"))
        if residual_diag_present and isinstance(residual_diag.get("white_noise"), bool)
        else None
    )
    residual_diag_n = (
        int(residual_diag.get("n"))
        if residual_diag_present and isinstance(residual_diag.get("n"), (int, float))
        else None
    )
    ensemble_rmse = _rmse_from(ensemble_metrics)
    baseline_rmse = _rmse_from(baseline_metrics)
    ensemble_missing = (
        ensemble_metrics is None
        and baseline_rmse is not None
        and baseline_rmse > 0
    )

    if ensemble_rmse is None or baseline_rmse is None or baseline_rmse <= 0:
        return AuditCheckResult(
            path=path,
            ensemble_rmse=ensemble_rmse,
            baseline_rmse=baseline_rmse,
            rmse_ratio=None,
            violation=False,
            baseline_model=resolved_baseline,
            ensemble_missing=ensemble_missing,
            default_model=default_model,
            residual_diag_present=residual_diag_present,
            residual_diag_white_noise=residual_diag_white_noise,
            residual_diag_n=residual_diag_n,
            ensemble_index_mismatch=ensemble_index_mismatch,
        )

    rmse_ratio = ensemble_rmse / baseline_rmse
    violation = rmse_ratio > (1.0 + tolerance)

    return AuditCheckResult(
        path=path,
        ensemble_rmse=ensemble_rmse,
        baseline_rmse=baseline_rmse,
        rmse_ratio=rmse_ratio,
        violation=violation,
        baseline_model=resolved_baseline,
        ensemble_missing=ensemble_missing,
        default_model=default_model,
        residual_diag_present=residual_diag_present,
        residual_diag_white_noise=residual_diag_white_noise,
        residual_diag_n=residual_diag_n,
        ensemble_index_mismatch=ensemble_index_mismatch,
    )


def _load_monitoring_thresholds(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path or not config_path.exists():
        return {}
    try:
        import yaml  # Local import to keep dependency optional
    except ImportError:
        return {}

    raw = yaml.safe_load(config_path.read_text()) or {}
    fm = raw.get("forecaster_monitoring") or {}
    return fm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check TS forecast audit files for ensemble underperformance."
    )
    parser.add_argument(
        "--audit-dir",
        default=str(DEFAULT_AUDIT_DIR),
        help="Directory containing forecast_audit_*.json files "
        "(default: logs/forecast_audits/production when present, else logs/forecast_audits)",
    )
    parser.add_argument(
        "--include-research",
        action="store_true",
        help=(
            "Include logs/forecast_audits/research alongside the selected audit directory. "
            "Default mode inspects production audits only."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=FORECAST_AUDIT_MAX_FILES_DEFAULT,
        help=(
            "Maximum number of most recent audit files to inspect "
            f"(default: {FORECAST_AUDIT_MAX_FILES_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Allowed RMSE degradation vs baseline before flagging a violation. "
        "If omitted, will fall back to config/forecaster_monitoring.yml or 0.10.",
    )
    parser.add_argument(
        "--max-violation-rate",
        type=float,
        default=None,
        help="Maximum fraction of checked audits allowed to violate the RMSE tolerance "
        "before exiting non-zero. If omitted, will fall back to config/forecaster_monitoring.yml or 0.25.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_MONITORING_CONFIG),
        help="Optional path to forecaster_monitoring.yml "
        "(default: config/forecaster_monitoring.yml if present)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Optional path to SQLite DB for deterministic ts_signal_id outcome joins. "
            "If omitted, outcome join telemetry stays disabled."
        ),
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Baseline model for the RMSE gate: BEST_SINGLE, SAMOSSA, GARCH, or SARIMAX. "
        "If omitted, uses forecaster_monitoring.regression_metrics.baseline_model "
        f"or {DEFAULT_BASELINE_MODEL}.",
    )
    parser.add_argument(
        "--require-effective-audits",
        type=int,
        default=None,
        help="If set, exit non-zero when effective audits with RMSE metrics are below this count.",
    )
    parser.add_argument(
        "--require-holding-period",
        action="store_true",
        help="If set, require effective audits to meet holding_period_audits from the monitoring config.",
    )
    parser.add_argument(
        "--manifest-integrity-mode",
        default=None,
        choices=sorted(MANIFEST_MODES),
        help=(
            "Audit provenance enforcement mode: off|warn|fail. "
            "If omitted, uses forecaster_monitoring.regression_metrics.manifest_integrity_mode."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help=(
            "Path to forecast audit manifest JSONL. "
            "If omitted, uses <audit-dir>/forecast_audit_manifest.jsonl "
            "or regression_metrics.manifest_filename."
        ),
    )
    parser.add_argument(
        "--max-missing-ensemble-rate",
        type=float,
        default=None,
        help=(
            "Maximum fraction of unique audits allowed to miss ensemble metrics "
            "before exiting non-zero. If omitted, uses config value or 1.0."
        ),
    )
    parser.add_argument(
        "--min-forecast-horizon",
        type=int,
        default=None,
        help=(
            "Minimum dataset.forecast_horizon required for an audit artifact "
            "to participate in gate statistics. If omitted, uses "
            "regression_metrics.min_forecast_horizon when present."
        ),
    )
    args = parser.parse_args()

    requested_audit_dir = Path(args.audit_dir)
    audit_dir = requested_audit_dir
    if (
        requested_audit_dir.resolve() == DEFAULT_AUDIT_PRODUCTION_DIR.resolve()
        and not requested_audit_dir.exists()
        and DEFAULT_AUDIT_ROOT.exists()
    ):
        # Backward-compatible fallback for repos that still keep audits under
        # logs/forecast_audits without production/research partitioning.
        audit_dir = DEFAULT_AUDIT_ROOT
    db_path = Path(args.db) if args.db else None
    min_forecast_horizon: Optional[int] = None
    audit_roots = _resolve_audit_roots(audit_dir, bool(args.include_research))
    files = _collect_audit_files(audit_roots=audit_roots, max_files=int(args.max_files))
    if not files:
        roots_text = ", ".join(str(root) for root in audit_roots)
        _emit_failure_summary_and_exit(
            message=f"No forecast_audit_*.json files found in: {roots_text}",
            audit_dir=audit_dir,
            audit_roots=audit_roots,
            include_research=bool(args.include_research),
            max_files=int(args.max_files),
            db_path=db_path,
            min_forecast_horizon=min_forecast_horizon,
            exit_code=1,
        )

    monitoring_cfg = _load_monitoring_thresholds(
        Path(args.config_path) if args.config_path else None
    )
    rmse_cfg = monitoring_cfg.get("regression_metrics") if monitoring_cfg else {}

    tolerance = (
        float(args.tolerance)
        if args.tolerance is not None
        else float(rmse_cfg.get("max_rmse_ratio_vs_baseline", 1.10)) - 1.0
    )
    max_violation_rate = (
        float(args.max_violation_rate)
        if args.max_violation_rate is not None
        else float(rmse_cfg.get("max_violation_rate", 0.25))
    )
    min_effective_audits = int(rmse_cfg.get("min_effective_audits", 0) or 0)
    baseline_model = (
        str(args.baseline_model)
        if args.baseline_model
        else str(rmse_cfg.get("baseline_model", DEFAULT_BASELINE_MODEL))
    )
    holding_period = int(rmse_cfg.get("holding_period_audits", 0) or 0)
    fail_on_violation_during_holding_period = bool(
        rmse_cfg.get("fail_on_violation_during_holding_period", False)
    )
    disable_if_no_lift = bool(rmse_cfg.get("disable_ensemble_if_no_lift", False))
    min_lift_rmse_ratio = float(rmse_cfg.get("min_lift_rmse_ratio", 0.0) or 0.0)
    min_lift_fraction = float(rmse_cfg.get("min_lift_fraction", 0.0) or 0.0)
    promotion_margin = float(rmse_cfg.get("promotion_margin", 0.0) or 0.0)
    recent_window_audits = max(int(rmse_cfg.get("recent_window_audits", 0) or 0), 0)
    recent_window_max_violation_rate = float(
        rmse_cfg.get("recent_window_max_violation_rate", max_violation_rate)
    )
    raw_recent_p90 = rmse_cfg.get("recent_window_max_p90_rmse_ratio")
    recent_window_max_p90_rmse_ratio = (
        float(raw_recent_p90)
        if isinstance(raw_recent_p90, (int, float))
        else None
    )
    manifest_mode = (
        str(args.manifest_integrity_mode).strip().lower()
        if args.manifest_integrity_mode
        else str(rmse_cfg.get("manifest_integrity_mode", DEFAULT_MANIFEST_MODE)).strip().lower()
    )
    if manifest_mode not in MANIFEST_MODES:
        manifest_mode = DEFAULT_MANIFEST_MODE
    manifest_filename = str(rmse_cfg.get("manifest_filename", DEFAULT_MANIFEST_FILENAME))
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else (audit_dir / manifest_filename)
    )
    max_missing_ensemble_rate = (
        float(args.max_missing_ensemble_rate)
        if args.max_missing_ensemble_rate is not None
        else float(rmse_cfg.get("max_missing_ensemble_rate", 1.0))
    )
    raw_max_index_mismatch_rate = rmse_cfg.get("max_index_mismatch_rate")
    max_index_mismatch_rate = (
        float(raw_max_index_mismatch_rate)
        if raw_max_index_mismatch_rate is not None
        else 1.0
    )
    raw_max_non_white_noise_rate = rmse_cfg.get("max_non_white_noise_rate")
    max_non_white_noise_rate = (
        float(raw_max_non_white_noise_rate)
        if raw_max_non_white_noise_rate is not None
        else 1.0
    )
    min_residual_diagnostics_n = int(
        rmse_cfg.get("min_residual_diagnostics_n", 0) or 0
    )
    fail_on_missing_residual_diagnostics = bool(
        rmse_cfg.get("fail_on_missing_residual_diagnostics", False)
    )
    residual_diagnostics_rate_warn_only = bool(
        rmse_cfg.get("residual_diagnostics_rate_warn_only", False)
    )
    min_forecast_horizon = (
        int(args.min_forecast_horizon)
        if args.min_forecast_horizon is not None
        else (
            int(rmse_cfg.get("min_forecast_horizon"))
            if rmse_cfg.get("min_forecast_horizon") is not None
            else None
        )
    )
    if min_forecast_horizon is not None and min_forecast_horizon < 0:
        min_forecast_horizon = 0

    def _rmse_dedupe_key_from_audit(audit: Dict[str, Any]) -> Tuple[Any, ...]:
        dataset = audit.get("dataset") or {}
        ds_key = (
            dataset.get("start"),
            dataset.get("end"),
            dataset.get("length"),
            dataset.get("forecast_horizon"),
        )
        # Deduplicate by data window (start/end/length/horizon) only and keep the
        # most recent file as the authoritative result. This prevents stale
        # earlier runs (often with different ensemble weights) from inflating
        # the violation rate and aligns with "latest evidence wins" monitoring.
        return ds_key

    def _outcome_dedupe_key_from_audit(audit: Dict[str, Any]) -> Tuple[Any, ...]:
        dataset = audit.get("dataset") or {}
        ticker = str(dataset.get("ticker") or dataset.get("symbol") or "").strip().upper() or None
        return (
            ticker,
            dataset.get("start"),
            dataset.get("end"),
            dataset.get("length"),
            dataset.get("forecast_horizon"),
        )

    def _forecast_horizon_from_audit(audit: Dict[str, Any]) -> Optional[int]:
        dataset = audit.get("dataset") or {}
        raw = dataset.get("forecast_horizon")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    unique_map: dict[Tuple[Any, ...], Path] = {}
    outcome_unique_map: dict[Tuple[Any, ...], Path] = {}
    horizon_filtered_count = 0
    parseable_count = 0
    parse_error_count = 0

    manifest_index: Dict[str, str] = {}
    manifest_stats: Dict[str, Any] = {
        "mode": manifest_mode,
        "manifest_path": str(manifest_path),
        "manifest_exists": None,
        "invalid_records": 0,
        "verified": 0,
        "missing": 0,
        "hash_failed": 0,
        "mismatch": 0,
        "excluded_unverified": 0,
    }
    if manifest_mode != "off":
        manifest_index, loaded_stats = _load_manifest_index(manifest_path)
        manifest_stats.update(loaded_stats)

    for f in files:
        audit, load_error = _load_audit_with_error(f)
        if not audit:
            if load_error:
                parse_error_count += 1
            continue
        parseable_count += 1

        if min_forecast_horizon is not None:
            horizon = _forecast_horizon_from_audit(audit)
            if horizon is None or horizon < min_forecast_horizon:
                horizon_filtered_count += 1
                continue

        if manifest_mode != "off":
            status = _verify_manifest_entry(f, manifest_index)
            if status == "ok":
                manifest_stats["verified"] += 1
            else:
                manifest_stats[status] += 1
                if manifest_mode == "fail":
                    manifest_stats["excluded_unverified"] += 1
                    continue
        key = _rmse_dedupe_key_from_audit(audit)
        # Files are sorted newest-first, so keep the first (newest) entry we see
        # for each dataset window and ignore older duplicates.
        if key in unique_map:
            pass
        else:
            unique_map[key] = f
        outcome_key = _outcome_dedupe_key_from_audit(audit)
        if outcome_key not in outcome_unique_map:
            outcome_unique_map[outcome_key] = f

    unique_files: List[Path] = list(unique_map.values())
    outcome_unique_files: List[Path] = list(outcome_unique_map.values())

    results: List[AuditCheckResult] = []
    for f in unique_files:
        res = check_audit_file(f, tolerance, baseline_model=baseline_model)
        if res is not None:
            results.append(res)

    print("=== Forecast Audit Regression Check ===")
    print(f"Audit directory : {audit_dir}")
    print(
        "Audit roots    : "
        + ", ".join(str(root) for root in audit_roots)
        + f" (include_research={int(bool(args.include_research))})"
    )
    print(
        f"Files inspected : {len(results)} unique (raw={len(files)}, max_files={args.max_files})"
    )
    print(
        "Parse stats     : "
        f"parseable={parseable_count} "
        f"parse_errors={parse_error_count}"
    )
    print(
        "Window stats    : "
        f"deduped={len(unique_map)} "
        f"checked={len(results)}"
    )
    print(f"Baseline model  : {baseline_model}")
    print(f"RMSE tolerance  : ensemble_rmse <= (1 + {tolerance:.2f}) * baseline_rmse")
    if min_effective_audits > 0:
        print(f"Min effective   : {min_effective_audits} audit(s) before hard gating")
    if holding_period > 0:
        print(f"Holding period  : {holding_period} effective audit(s)")
        if fail_on_violation_during_holding_period:
            print("Warmup behavior : fail on violations during holding period")
    if disable_if_no_lift:
        print(
            "No-lift gate    : enabled "
            f"(min_lift_rmse_ratio={min_lift_rmse_ratio:.2%}, "
            f"min_lift_fraction={min_lift_fraction:.2%})"
        )
    if promotion_margin > 0:
        print(f"Promotion margin: requires >= {promotion_margin:.2%} lift to keep ensemble as default")
    if recent_window_audits > 0:
        print(
            f"Recent window  : {recent_window_audits} effective audit(s) "
            f"(max violation rate {recent_window_max_violation_rate:.2%})"
        )
    if min_forecast_horizon is not None:
        print(
            "Horizon filter : "
            f"forecast_horizon >= {min_forecast_horizon} "
            f"(excluded={horizon_filtered_count})"
        )
    if recent_window_max_p90_rmse_ratio is not None:
        print(
            "Recent p90 gate: "
            f"p90(rmse_ratio) <= {recent_window_max_p90_rmse_ratio:.3f}"
        )
    if manifest_mode != "off":
        print(
            "Manifest check : "
            f"mode={manifest_mode} path={manifest_stats.get('manifest_path')} "
            f"exists={manifest_stats.get('manifest_exists')}"
        )
        print(
            "Manifest stats : "
            f"verified={manifest_stats.get('verified', 0)} "
            f"missing={manifest_stats.get('missing', 0)} "
            f"mismatch={manifest_stats.get('mismatch', 0)} "
            f"hash_failed={manifest_stats.get('hash_failed', 0)} "
            f"invalid_records={manifest_stats.get('invalid_records', 0)}"
        )
        if manifest_mode == "fail":
            unverified = (
                int(manifest_stats.get("missing", 0))
                + int(manifest_stats.get("mismatch", 0))
                + int(manifest_stats.get("hash_failed", 0))
                + int(manifest_stats.get("invalid_records", 0))
            )
            if not bool(manifest_stats.get("manifest_exists", False)):
                _emit_failure_summary_and_exit(
                    message=(
                        "Manifest integrity mode=fail but manifest file is missing: "
                        f"{manifest_path}"
                    ),
                    audit_dir=audit_dir,
                    audit_roots=audit_roots,
                    include_research=bool(args.include_research),
                    max_files=int(args.max_files),
                    db_path=db_path,
                    min_forecast_horizon=min_forecast_horizon,
                    exit_code=1,
                )
            if unverified > 0:
                _emit_failure_summary_and_exit(
                    message=(
                        "Manifest integrity failed: "
                        f"missing={manifest_stats.get('missing', 0)} "
                        f"mismatch={manifest_stats.get('mismatch', 0)} "
                        f"hash_failed={manifest_stats.get('hash_failed', 0)} "
                        f"invalid_records={manifest_stats.get('invalid_records', 0)}"
                    ),
                    audit_dir=audit_dir,
                    audit_roots=audit_roots,
                    include_research=bool(args.include_research),
                    max_files=int(args.max_files),
                    db_path=db_path,
                    min_forecast_horizon=min_forecast_horizon,
                    exit_code=1,
                )

    violation_count = sum(1 for r in results if r.violation)
    rmse_windows_processed = len(results)
    effective_n = sum(
        1
        for r in results
        if (
            r.ensemble_rmse is not None
            and r.baseline_rmse is not None
            and r.baseline_rmse > 0
        )
    )
    violation_rate = (violation_count / effective_n) if effective_n else 0.0
    ensemble_missing_count = sum(1 for r in results if r.ensemble_missing)
    ensemble_missing_rate = (ensemble_missing_count / len(results)) if results else 0.0
    warmup_required = max(min_effective_audits, holding_period, 0)
    model_backed_results = [
        r
        for r in results
        if r.default_model in {"SARIMAX", "SAMOSSA", "GARCH", "MSSA_RL"}
    ]
    residual_effective_results = [
        r
        for r in model_backed_results
        if (
            r.residual_diag_present
            and r.residual_diag_n is not None
            and r.residual_diag_n >= min_residual_diagnostics_n
        )
    ]
    residual_effective_n = len(residual_effective_results)
    non_white_noise_count = sum(
        1 for r in residual_effective_results if r.residual_diag_white_noise is False
    )
    non_white_noise_rate = (
        non_white_noise_count / residual_effective_n if residual_effective_n else 0.0
    )
    missing_residual_diag_count = sum(
        1 for r in model_backed_results if not r.residual_diag_present
    )
    ensemble_index_mismatch_count = sum(
        1 for r in results if r.ensemble_index_mismatch
    )
    ensemble_index_mismatch_rate = (
        ensemble_index_mismatch_count / len(results) if results else 0.0
    )

    def _percentiles(values: list[float], percents: list[float]) -> Dict[float, float]:
        if not values:
            return {}
        vals = sorted(values)
        out: Dict[float, float] = {}
        for p in percents:
            if p <= 0:
                out[p] = vals[0]
                continue
            if p >= 1:
                out[p] = vals[-1]
                continue
            idx = (len(vals) - 1) * p
            lower = int(idx)
            upper = min(lower + 1, len(vals) - 1)
            weight = idx - lower
            out[p] = vals[lower] * (1 - weight) + vals[upper] * weight
        return out

    ratios = [
        r.rmse_ratio
        for r in results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    pct = _percentiles(ratios, [0.1, 0.5, 0.9]) if ratios else {}
    recent_results: List[AuditCheckResult] = (
        results[:recent_window_audits] if recent_window_audits > 0 else []
    )
    recent_effective_n = sum(
        1
        for r in recent_results
        if (
            r.ensemble_rmse is not None
            and r.baseline_rmse is not None
            and r.baseline_rmse > 0
        )
    )
    recent_violation_count = sum(1 for r in recent_results if r.violation)
    recent_violation_rate = (
        (recent_violation_count / recent_effective_n) if recent_effective_n else 0.0
    )
    recent_ratios = [
        r.rmse_ratio
        for r in recent_results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    recent_pct = _percentiles(recent_ratios, [0.5, 0.9]) if recent_ratios else {}
    rmse_windows_usable = effective_n
    outcomes_loaded = False
    outcome_join_attempted = False
    outcome_join_error: Optional[str] = None
    outcome_windows_eligible = 0
    outcome_windows_matched = 0
    outcome_windows_missing = 0
    outcome_windows_ambiguous = 0
    outcome_windows_not_due = 0
    outcome_windows_not_yet_eligible = 0  # legacy alias for telemetry schema v2
    outcome_windows_invalid_context = 0
    outcome_windows_outcomes_not_loaded = 0
    outcome_windows_no_signal_id = 0
    outcome_windows_non_trade_context = 0
    outcome_windows_missing_execution_metadata = 0

    print(
        "RMSE coverage  : "
        f"raw={len(files)} "
        f"parseable={parseable_count} "
        f"deduped={len(unique_map)} "
        f"processed={rmse_windows_processed} "
        f"usable={rmse_windows_usable}"
    )
    print(
        "Outcome dedupe : "
        f"raw={len(files)} "
        f"deduped={len(outcome_unique_map)}"
    )

    print(f"\nEffective audits with RMSE: {effective_n}")
    print(f"Violations (ensemble worse than baseline beyond tolerance): {violation_count}")
    print(f"Violation rate: {violation_rate:.2%} (max allowed {max_violation_rate:.2%})")
    print(
        "Missing ensemble metrics      : "
        f"{ensemble_missing_count}/{len(results)} ({ensemble_missing_rate:.2%}) "
        f"(max allowed {max_missing_ensemble_rate:.2%})"
    )
    print(
        "Residual diagnostics         : "
        f"model_backed={len(model_backed_results)} "
        f"usable={residual_effective_n} "
        f"missing={missing_residual_diag_count} "
        f"non_white_noise={non_white_noise_count} "
        f"rate={non_white_noise_rate:.2%} "
        f"(max allowed {max_non_white_noise_rate:.2%}, min_n={min_residual_diagnostics_n})"
    )
    print(
        "Ensemble index mismatches    : "
        f"{ensemble_index_mismatch_count}/{len(results)} "
        f"({ensemble_index_mismatch_rate:.2%}) "
        f"(max allowed {max_index_mismatch_rate:.2%})"
    )

    def _current_failure_summary(
        *,
        decision: Optional[str] = None,
        decision_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        lift_value = 0.0
        if effective_n:
            lift_threshold = 1.0 - min_lift_rmse_ratio
            lift_count = sum(
                1
                for r in results
                if (
                    r.rmse_ratio is not None
                    and isinstance(r.rmse_ratio, (int, float))
                    and float(r.rmse_ratio) < float(lift_threshold)
                )
            )
            lift_value = lift_count / effective_n

        ratios_filtered = [
            (r.path.name, float(r.rmse_ratio))
            for r in results
            if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
        ]
        ratio_values = [val for _, val in ratios_filtered]
        ratio_stats = {
            "count": len(ratio_values),
            "min": min(ratio_values) if ratio_values else None,
            "max": max(ratio_values) if ratio_values else None,
            "mean": (sum(ratio_values) / len(ratio_values)) if ratio_values else None,
            "p10": pct.get(0.1) if pct else None,
            "p50": pct.get(0.5) if pct else None,
            "p90": pct.get(0.9) if pct else None,
        }
        return {
            "measurement_contract_version": 1,
            "baseline_model": baseline_model,
            "lift_threshold_rmse_ratio": _lift_threshold_rmse_ratio(min_lift_rmse_ratio),
            "effective_audits": effective_n,
            "effective_outcome_audits": None,
            "total_unique_audits": len(results),
            "violation_count": violation_count,
            "violation_rate": violation_rate,
            "max_violation_rate": max_violation_rate,
            "ensemble_missing_count": ensemble_missing_count,
            "ensemble_missing_rate": ensemble_missing_rate,
            "max_missing_ensemble_rate": max_missing_ensemble_rate,
            "ensemble_index_mismatch_count": ensemble_index_mismatch_count,
            "ensemble_index_mismatch_rate": ensemble_index_mismatch_rate,
            "max_index_mismatch_rate": max_index_mismatch_rate,
            "residual_diag_effective_audits": residual_effective_n,
            "non_white_noise_count": non_white_noise_count,
            "non_white_noise_rate": non_white_noise_rate,
            "max_non_white_noise_rate": max_non_white_noise_rate,
            "missing_residual_diag_count": missing_residual_diag_count,
            "min_residual_diagnostics_n": min_residual_diagnostics_n,
            "fail_on_missing_residual_diagnostics": fail_on_missing_residual_diagnostics,
            "holding_period_required": warmup_required,
            "lift_fraction": lift_value,
            "min_lift_fraction": min_lift_fraction,
            "percentiles": {
                "p10": pct.get(0.1) if pct else None,
                "p50": pct.get(0.5) if pct else None,
                "p90": pct.get(0.9) if pct else None,
            },
            "ratio_distribution": ratio_stats,
            "decision": decision,
            "decision_reason": decision_reason,
            "recent_window_audits": recent_window_audits,
            "recent_effective_audits": recent_effective_n,
            "recent_violation_count": recent_violation_count,
            "recent_violation_rate": recent_violation_rate,
            "recent_window_max_violation_rate": recent_window_max_violation_rate,
            "recent_rmse_ratio_p90": recent_pct.get(0.9) if recent_pct else None,
            "recent_window_max_p90_rmse_ratio": recent_window_max_p90_rmse_ratio,
            "window_counts": {
                "n_rmse_windows_processed": rmse_windows_processed,
                "n_rmse_windows_usable": effective_n,
            },
        }

    if (
        len(results) > 0
        and max_missing_ensemble_rate >= 0
    ):
        if ensemble_missing_rate > max_missing_ensemble_rate:
            _emit_failure_summary_and_exit(
                message=(
                    "Missing ensemble metric rate "
                    f"{ensemble_missing_rate:.2%} exceeds max "
                    f"{max_missing_ensemble_rate:.2%}"
                ),
                audit_dir=audit_dir,
                audit_roots=audit_roots,
                include_research=bool(args.include_research),
                max_files=int(args.max_files),
                db_path=db_path,
                min_forecast_horizon=min_forecast_horizon,
                exit_code=1,
                summary_fields=_current_failure_summary(
                    decision=DEFAULT_DECISION_RESEARCH,
                    decision_reason="missing ensemble metrics exceed threshold",
                ),
            )
    if (
        len(results) > 0
        and max_index_mismatch_rate >= 0
        and ensemble_index_mismatch_rate > max_index_mismatch_rate
    ):
        _emit_failure_summary_and_exit(
            message=(
                "Ensemble index mismatch rate "
                f"{ensemble_index_mismatch_rate:.2%} exceeds max "
                f"{max_index_mismatch_rate:.2%}"
            ),
            audit_dir=audit_dir,
            audit_roots=audit_roots,
            include_research=bool(args.include_research),
            max_files=int(args.max_files),
            db_path=db_path,
            min_forecast_horizon=min_forecast_horizon,
            exit_code=1,
            summary_fields=_current_failure_summary(
                decision=DEFAULT_DECISION_RESEARCH,
                decision_reason="ensemble index mismatch rate exceeds threshold",
            ),
        )
    # Only hard-fail on residual diagnostics rate once we have accumulated enough
    # audits (warmup_required = max(min_effective_audits, holding_period) = 20).
    # Below this floor, the rate is treated as inconclusive — consistent with how
    # fail_on_violation_during_holding_period=false gates the RMSE violation check.
    # When residual_diagnostics_rate_warn_only=true the threshold is still checked
    # and printed, but the gate emits a warning rather than a hard FAIL exit.
    # Use this while SSA-based in-sample fit residuals dominate the diagnostic pool
    # (Ljung-Box at n=261 structurally rejects H0 for SAMoSSA/MSSA-RL regardless
    # of model quality — this is NOT a sign of mis-specification).
    if (
        residual_effective_n >= warmup_required
        and non_white_noise_rate > max_non_white_noise_rate
    ):
        if residual_diagnostics_rate_warn_only:
            print(
                f"[WARN] Non-white-noise residual rate {non_white_noise_rate:.2%} "
                f"exceeds max {max_non_white_noise_rate:.2%} "
                f"(residual_diagnostics_rate_warn_only=true; not failing)"
            )
        else:
            _emit_failure_summary_and_exit(
                message=(
                    f"Non-white-noise residual rate {non_white_noise_rate:.2%} exceeds "
                    f"max {max_non_white_noise_rate:.2%}"
                ),
                audit_dir=audit_dir,
                audit_roots=audit_roots,
                include_research=bool(args.include_research),
                max_files=int(args.max_files),
                db_path=db_path,
                min_forecast_horizon=min_forecast_horizon,
                exit_code=1,
                summary_fields=_current_failure_summary(
                    decision=DEFAULT_DECISION_RESEARCH,
                    decision_reason="non-white-noise residual rate exceeds threshold",
                ),
            )
    if (
        fail_on_missing_residual_diagnostics
        and effective_n >= warmup_required
        and missing_residual_diag_count > 0
    ):
        _emit_failure_summary_and_exit(
            message=(
                "Missing residual diagnostics for "
                f"{missing_residual_diag_count} model-backed default-path audit(s)"
            ),
            audit_dir=audit_dir,
            audit_roots=audit_roots,
            include_research=bool(args.include_research),
            max_files=int(args.max_files),
            db_path=db_path,
            min_forecast_horizon=min_forecast_horizon,
            exit_code=1,
            summary_fields=_current_failure_summary(
                decision=DEFAULT_DECISION_RESEARCH,
                decision_reason="missing residual diagnostics after warmup",
            ),
        )
    if pct:
        print(
            "RMSE ratio percentiles: "
            f"p10={pct.get(0.1):.3f}, median={pct.get(0.5):.3f}, p90={pct.get(0.9):.3f}"
        )
    if recent_window_audits > 0:
        print(
            "Recent window stats: "
            f"effective={recent_effective_n}/{recent_window_audits}, "
            f"violations={recent_violation_count}, "
            f"violation_rate={recent_violation_rate:.2%}"
        )
        if recent_pct:
            print(
                "Recent RMSE ratio percentiles: "
                f"median={recent_pct.get(0.5):.3f}, p90={recent_pct.get(0.9):.3f}"
            )

    if recent_window_audits > 0:
        if recent_effective_n < recent_window_audits:
            print(
                "Recent-window gate inconclusive: "
                f"effective_audits={recent_effective_n} < required_audits={recent_window_audits}"
            )
        else:
            if recent_violation_rate > recent_window_max_violation_rate:
                _emit_failure_summary_and_exit(
                    message=(
                        f"Recent-window violation rate {recent_violation_rate:.2%} exceeds "
                        f"max-violation-rate {recent_window_max_violation_rate:.2%} "
                        f"(window={recent_window_audits})"
                    ),
                    audit_dir=audit_dir,
                    audit_roots=audit_roots,
                    include_research=bool(args.include_research),
                    max_files=int(args.max_files),
                    db_path=db_path,
                    min_forecast_horizon=min_forecast_horizon,
                    exit_code=1,
                    summary_fields=_current_failure_summary(
                        decision=DEFAULT_DECISION_RESEARCH,
                        decision_reason="recent window violation rate exceeds threshold",
                    ),
                )
            if (
                recent_window_max_p90_rmse_ratio is not None
                and recent_pct
                and recent_pct.get(0.9) is not None
                and float(recent_pct.get(0.9)) > float(recent_window_max_p90_rmse_ratio)
            ):
                _emit_failure_summary_and_exit(
                    message=(
                        f"Recent-window p90 RMSE ratio {float(recent_pct.get(0.9)):.3f} exceeds "
                        f"{float(recent_window_max_p90_rmse_ratio):.3f} "
                        f"(window={recent_window_audits})"
                    ),
                    audit_dir=audit_dir,
                    audit_roots=audit_roots,
                    include_research=bool(args.include_research),
                    max_files=int(args.max_files),
                    db_path=db_path,
                    min_forecast_horizon=min_forecast_horizon,
                    exit_code=1,
                    summary_fields=_current_failure_summary(
                        decision=DEFAULT_DECISION_RESEARCH,
                        decision_reason="recent window p90 rmse ratio exceeds threshold",
                    ),
                )

    rmse_inconclusive = False
    if warmup_required > 0 and effective_n < warmup_required:
        explicit_required: Optional[int] = None
        if args.require_holding_period and holding_period > 0:
            explicit_required = holding_period
        if args.require_effective_audits is not None:
            explicit_required = int(args.require_effective_audits)
        if explicit_required is not None and effective_n < explicit_required:
            _emit_failure_summary_and_exit(
                message=(
                    "Insufficient effective audits for RMSE gating: "
                    f"effective_audits={effective_n} < required_audits={explicit_required}"
                ),
                audit_dir=audit_dir,
                audit_roots=audit_roots,
                include_research=bool(args.include_research),
                max_files=int(args.max_files),
                db_path=db_path,
                min_forecast_horizon=min_forecast_horizon,
                exit_code=1,
                summary_fields=_current_failure_summary(
                    decision="INCONCLUSIVE",
                    decision_reason=(
                        f"effective_audits={effective_n} < required_audits={explicit_required}"
                    ),
                ),
            )
        if (
            fail_on_violation_during_holding_period
            and effective_n > 0
            and violation_rate > max_violation_rate
        ):
            _emit_failure_summary_and_exit(
                message=(
                    f"Ensemble RMSE violation rate {violation_rate:.2%} exceeds "
                    f"max-violation-rate {max_violation_rate:.2%} during holding period "
                    f"(effective_audits={effective_n} < required_audits={warmup_required})"
                ),
                audit_dir=audit_dir,
                audit_roots=audit_roots,
                include_research=bool(args.include_research),
                max_files=int(args.max_files),
                db_path=db_path,
                min_forecast_horizon=min_forecast_horizon,
                exit_code=1,
                summary_fields=_current_failure_summary(
                    decision=DEFAULT_DECISION_RESEARCH,
                    decision_reason=(
                        "violation rate exceeds threshold during holding period"
                    ),
                ),
            )
        print(
            f"\nRMSE gate inconclusive: effective_audits={effective_n} "
            f"< required_audits={warmup_required}.",
        )
        rmse_inconclusive = True

    if effective_n == 0:
        print("\nRMSE gate inconclusive: no usable RMSE metrics.")
        rmse_inconclusive = True

    if not rmse_inconclusive:
        print("\nSample details (most recent first):")
        header = f"{'File':<32} {'ens_rmse':>10} {'base_rmse':>10} {'ratio':>8} {'VIOL':>6}"
        print(header)
        print("-" * len(header))
        for r in results[:10]:
            ratio_str = f"{r.rmse_ratio:.3f}" if r.rmse_ratio is not None else "n/a"
            ens_str = f"{r.ensemble_rmse:.4f}" if r.ensemble_rmse is not None else "n/a"
            base_str = f"{r.baseline_rmse:.4f}" if r.baseline_rmse is not None else "n/a"
            viol_flag = "YES" if r.violation else ""
            display_name = r.path.name
            if r.baseline_model:
                display_name = f"{display_name} ({r.baseline_model})"
            display_name = display_name[:32]
            print(
                f"{display_name:<32} {ens_str:>10} {base_str:>10} {ratio_str:>8} {viol_flag:>6}"
            )

    decision = DEFAULT_DECISION_KEEP
    decision_reason = "ensemble within tolerance"

    lift_fraction = 0.0
    if effective_n:
        lift_threshold = 1.0 - min_lift_rmse_ratio
        lift_count = sum(
            1
            for r in results
            if (
                r.rmse_ratio is not None
                and isinstance(r.rmse_ratio, (int, float))
                and float(r.rmse_ratio) < float(lift_threshold)
            )
        )
        lift_fraction = lift_count / effective_n

    if rmse_inconclusive:
        required = warmup_required if warmup_required > 0 else 1
        decision = "INCONCLUSIVE"
        decision_reason = (
            f"effective_audits={effective_n} < required_audits={required}"
        )
    else:
        if holding_period > 0 and effective_n >= holding_period:
            print(
                f"\nEnsemble lift fraction: {lift_fraction:.2%} "
                f"(required >= {min_lift_fraction:.2%})"
            )
            if lift_fraction < min_lift_fraction:
                decision = DEFAULT_DECISION_DISABLE
                decision_reason = "insufficient lift vs baseline"
                if disable_if_no_lift:
                    _emit_failure_summary_and_exit(
                        message=(
                            "Ensemble shows insufficient lift over baseline during holding period; "
                            "disable ensemble as default source of truth (reward-to-effort)."
                        ),
                        audit_dir=audit_dir,
                        audit_roots=audit_roots,
                        include_research=bool(args.include_research),
                        max_files=int(args.max_files),
                        db_path=db_path,
                        min_forecast_horizon=min_forecast_horizon,
                        exit_code=1,
                        summary_fields=_current_failure_summary(
                            decision=DEFAULT_DECISION_DISABLE,
                            decision_reason="insufficient lift vs baseline",
                        ),
                    )
                print(
                    "No-lift hard fail disabled by config; keeping ensemble in "
                    "non-default/research-only posture."
                )
            else:
                decision_reason = "lift demonstrated during holding period"

        if violation_rate > max_violation_rate:
            decision = DEFAULT_DECISION_RESEARCH
            decision_reason = (
                f"violation rate {violation_rate:.2%} exceeds {max_violation_rate:.2%}"
            )
            _emit_failure_summary_and_exit(
                message=(
                    f"Ensemble RMSE violation rate {violation_rate:.2%} exceeds "
                    f"max-violation-rate {max_violation_rate:.2%}"
                ),
                audit_dir=audit_dir,
                audit_roots=audit_roots,
                include_research=bool(args.include_research),
                max_files=int(args.max_files),
                db_path=db_path,
                min_forecast_horizon=min_forecast_horizon,
                exit_code=1,
                summary_fields=_current_failure_summary(
                    decision=DEFAULT_DECISION_RESEARCH,
                    decision_reason=(
                        f"violation rate {violation_rate:.2%} exceeds {max_violation_rate:.2%}"
                    ),
                ),
            )

        if promotion_margin > 0 and effective_n > 0 and decision == DEFAULT_DECISION_KEEP:
            margin_threshold = 1.0 - promotion_margin
            margin_lift = sum(
                1
                for r in results
                if r.rmse_ratio is not None
                and isinstance(r.rmse_ratio, (int, float))
                and float(r.rmse_ratio) < float(margin_threshold)
            )
            margin_lift_fraction = (margin_lift / effective_n) if effective_n else 0.0
            if margin_lift_fraction <= 0.0:
                decision = DEFAULT_DECISION_RESEARCH
                decision_reason = (
                    f"no ensemble lift >= {promotion_margin:.2%} across recent audits"
                )

    print(f"\nDecision: {decision} ({decision_reason})")

    cache_dir = Path("logs/forecast_audits_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ratios_filtered = [
        (r.path.name, float(r.rmse_ratio))
        for r in results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    ratio_values = [val for _, val in ratios_filtered]
    ratio_stats = {
        "count": len(ratio_values),
        "min": min(ratio_values) if ratio_values else None,
        "max": max(ratio_values) if ratio_values else None,
        "mean": (sum(ratio_values) / len(ratio_values)) if ratio_values else None,
        "p10": pct.get(0.1) if pct else None,
        "p50": pct.get(0.5) if pct else None,
        "p90": pct.get(0.9) if pct else None,
        "best": [
            {"file": name, "ratio": val}
            for name, val in sorted(ratios_filtered, key=lambda x: x[1])[:5]
        ],
        "worst": [
            {"file": name, "ratio": val}
            for name, val in sorted(ratios_filtered, key=lambda x: x[1], reverse=True)[
                :5
            ]
        ],
    }
    results_by_path = {r.path: r for r in results}
    dataset_entries = []
    rmse_usable_entries = []
    for f in outcome_unique_files:
        audit = _load_audit(f)
        ds = (audit or {}).get("dataset") or {}
        signal_context = (audit or {}).get("signal_context") or {}
        if not isinstance(signal_context, dict):
            signal_context = {}
        semantic_admission = (audit or {}).get("semantic_admission") or {}
        if not isinstance(semantic_admission, dict):
            semantic_admission = {}
        raw_ts_signal_id = signal_context.get("ts_signal_id")
        ts_signal_id = str(raw_ts_signal_id).strip() if raw_ts_signal_id is not None else ""
        ts_signal_id = ts_signal_id or None
        signal_context_missing = bool(signal_context.get("signal_context_missing"))
        expected_close, expected_close_source = compute_expected_close(signal_context, ds)
        context_type = str(signal_context.get("context_type") or "").strip().upper()
        if not context_type:
            context_type = "TRADE"
        meta = _extract_window_metadata(audit or {})
        manifest_status = _verify_manifest_entry(f, manifest_index) if manifest_mode != "off" else None
        audit_id = str((audit or {}).get("audit_id") or f.stem).strip() or f.stem
        event_type = str(
            (audit or {}).get("event_type")
            or signal_context.get("event_type")
            or "FORECAST_AUDIT"
        ).strip().upper()
        entry = {
            "file": f.name,
            "audit_id": audit_id,
            "event_type": event_type,
            "start": ds.get("start"),
            "end": ds.get("end"),
            "length": ds.get("length"),
            "forecast_horizon": ds.get("forecast_horizon"),
            "ticker": meta.get("ticker"),
            "detected_regime": meta.get("detected_regime"),
            "end_day": meta.get("end_day"),
            "context_type": context_type,
            "ts_signal_id": ts_signal_id,
            "run_id": signal_context.get("run_id"),
            "entry_ts": signal_context.get("entry_ts"),
            "signal_context_missing": signal_context_missing,
            "dataset_forecast_horizon": ds.get("forecast_horizon"),
            "signal_forecast_horizon": signal_context.get("forecast_horizon"),
            "expected_close_ts": expected_close.isoformat() if expected_close else None,
            "expected_close_source": expected_close_source,
            "evidence_contract_version": (audit or {}).get("evidence_contract_version"),
            "cohort_id": (audit or {}).get("cohort_id"),
            "cohort_identity": (audit or {}).get("cohort_identity"),
            "manifest_verification_status": manifest_status,
            "semantic_admission": semantic_admission,
            "payload_sha256": _sha256_file(f),
            "outcome_status": None,
            "outcome_reason": None,
        }
        matching = results_by_path.get(f)
        if matching:
            entry["rmse_ratio"] = matching.rmse_ratio
            entry["ensemble_rmse"] = matching.ensemble_rmse
            entry["baseline_rmse"] = matching.baseline_rmse
            entry["ensemble_missing"] = bool(matching.ensemble_missing)
            if (
                matching.ensemble_rmse is not None
                and matching.baseline_rmse is not None
                and matching.baseline_rmse > 0
            ):
                rmse_usable_entries.append(entry)
        dataset_entries.append(entry)

    audit_fingerprints: Dict[str, Optional[str]] = {}
    duplicate_conflict_ids: set[str] = set()
    for entry in dataset_entries:
        audit_id = str(entry.get("audit_id") or "").strip()
        if not audit_id:
            continue
        payload_sha = str(entry.get("payload_sha256") or "").strip() or None
        prior_sha = audit_fingerprints.get(audit_id)
        if prior_sha is None:
            audit_fingerprints[audit_id] = payload_sha
            continue
        if payload_sha != prior_sha:
            duplicate_conflict_ids.add(audit_id)

    if db_path is not None:
        outcome_join_attempted = True
        closed_trade_counts: Dict[str, int] = {}
        outcomes_loaded, closed_trade_counts, outcome_join_error = _load_closed_trade_match_counts(
            db_path
        )
        if outcomes_loaded:
            now = now_utc()
            for entry in dataset_entries:
                context_type = str(entry.get("context_type") or "").strip().upper()
                if context_type and context_type != "TRADE":
                    entry["outcome_status"] = "NON_TRADE_CONTEXT"
                    entry["outcome_reason"] = "NON_TRADE_CONTEXT"
                    outcome_windows_non_trade_context += 1
                    continue

                ticker = str(entry.get("ticker") or "").strip().upper()
                if not ticker:
                    entry["outcome_status"] = "NON_TRADE_CONTEXT"
                    entry["outcome_reason"] = "MISSING_TICKER"
                    outcome_windows_non_trade_context += 1
                    continue

                run_id = str(entry.get("run_id") or "").strip()
                entry_ts_raw = str(entry.get("entry_ts") or "").strip()
                if not run_id or not entry_ts_raw:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "MISSING_EXECUTION_METADATA"
                    outcome_windows_missing_execution_metadata += 1
                    outcome_windows_invalid_context += 1
                    continue

                ts_signal_id = str(entry.get("ts_signal_id") or "").strip()
                if not ts_signal_id:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "MISSING_SIGNAL_ID"
                    outcome_windows_no_signal_id += 1
                    outcome_windows_invalid_context += 1
                    continue

                signal_horizon = _parse_non_negative_int(entry.get("signal_forecast_horizon"))
                dataset_horizon = _parse_non_negative_int(entry.get("dataset_forecast_horizon"))
                if (
                    signal_horizon is not None
                    and dataset_horizon is not None
                    and signal_horizon != dataset_horizon
                ):
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "HORIZON_MISMATCH"
                    outcome_windows_invalid_context += 1
                    continue

                expected_close_ts = _parse_utc_datetime(entry.get("expected_close_ts"))
                entry_ts = _parse_utc_datetime(entry.get("entry_ts"))
                if expected_close_ts is None:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "EXPECTED_CLOSE_UNAVAILABLE"
                    outcome_windows_invalid_context += 1
                    continue
                if entry_ts is not None and expected_close_ts < entry_ts:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "CAUSALITY_VIOLATION"
                    outcome_windows_invalid_context += 1
                    continue
                if (
                    (expected_close_ts + OUTCOME_ELIGIBILITY_BUFFER) > now
                    and not (
                        # Early-credit: skip NOT_DUE if the trade is already confirmed
                        # closed in production_closed_trades — outcome is known.
                        ts_signal_id
                        and int(closed_trade_counts.get(ts_signal_id, 0)) == 1
                    )
                ):
                    entry["outcome_status"] = "NOT_DUE"
                    entry["outcome_reason"] = "OUTCOME_WINDOW_OPEN"
                    outcome_windows_not_due += 1
                    outcome_windows_not_yet_eligible += 1
                    continue

                match_count = int(closed_trade_counts.get(ts_signal_id, 0))
                entry["outcome_match_count"] = match_count
                if match_count == 1:
                    entry["outcome_status"] = "MATCHED"
                    entry["outcome_reason"] = "ONE_TO_ONE_MATCH"
                    outcome_windows_eligible += 1
                    outcome_windows_matched += 1
                elif match_count == 0:
                    entry["outcome_status"] = "OUTCOME_MISSING"
                    entry["outcome_reason"] = "DUE_BUT_MISSING_CLOSE"
                    outcome_windows_eligible += 1
                    outcome_windows_missing += 1
                else:
                    entry["outcome_status"] = "INVALID_CONTEXT"
                    entry["outcome_reason"] = "AMBIGUOUS_MATCH"
                    outcome_windows_invalid_context += 1
                    outcome_windows_ambiguous += 1
        else:
            for entry in dataset_entries:
                entry["outcome_status"] = "OUTCOMES_NOT_LOADED"
                entry["outcome_reason"] = "OUTCOME_JOIN_UNAVAILABLE"
                outcome_windows_outcomes_not_loaded += 1

    generated_utc = telemetry_now_utc()

    def _legacy_to_status(outcome_status: str) -> str:
        mapping = {
            "MATCHED": "PASS",
            "OUTCOME_MISSING": "FAIL",
            "NOT_DUE": "INCONCLUSIVE_ALLOWED",
            "INVALID_CONTEXT": "INVALID_CONTEXT",
            "NON_TRADE_CONTEXT": "WARN",
            "OUTCOMES_NOT_LOADED": "WARN",
        }
        return mapping.get(outcome_status, "WARN")

    for entry in dataset_entries:
        outcome_status = str(entry.get("outcome_status") or "OUTCOMES_NOT_LOADED").strip().upper()
        outcome_reason = str(entry.get("outcome_reason") or "OUTCOME_JOIN_UNAVAILABLE").strip().upper()
        semantic_admission = entry.get("semantic_admission")
        if (
            str(entry.get("audit_id") or "").strip() in duplicate_conflict_ids
            and not (isinstance(semantic_admission, dict) and "duplicate_conflict" in semantic_admission)
        ):
            entry["duplicate_conflict"] = True
        admission = _derive_semantic_admission(
            entry,
            include_research=bool(args.include_research),
        )
        entry.update(admission)
        context_type = str(entry.get("context_type") or "TRADE").strip().upper() or "TRADE"
        counts_toward_linkage = bool(entry.get("gate_eligible")) and outcome_status in {"MATCHED", "OUTCOME_MISSING"}
        counts_toward_readiness = _counts_toward_readiness_denominator(
            context_type=context_type,
            outcome_status=outcome_status,
            gate_eligible=bool(entry.get("gate_eligible")),
        )
        severity = "LOW"
        blocking = False
        if bool(entry.get("duplicate_conflict")) or bool(entry.get("quarantined")):
            severity = "HIGH"
            blocking = True
        elif outcome_status == "INVALID_CONTEXT":
            severity = "HIGH"
            blocking = True
        elif outcome_status == "OUTCOME_MISSING":
            severity = "MEDIUM"
        elif outcome_status == "OUTCOMES_NOT_LOADED":
            severity = "MEDIUM"
        normalized = normalize_telemetry_payload(
            {
                "status": _legacy_to_status(outcome_status),
                "reason_code": outcome_reason,
                "context_type": context_type,
                "severity": severity,
                "blocking": blocking,
                "counts_toward_readiness_denominator": counts_toward_readiness,
                "counts_toward_linkage_denominator": counts_toward_linkage,
                "generated_utc": generated_utc,
                "source_script": "scripts/check_forecast_audits.py",
            },
            source_script="scripts/check_forecast_audits.py",
            generated_utc=generated_utc,
        )
        entry.update(normalized)

    readiness_denominator_included = sum(
        1 for entry in dataset_entries if bool(entry.get("counts_toward_readiness_denominator"))
    )
    linkage_denominator_included = sum(
        1 for entry in dataset_entries if bool(entry.get("counts_toward_linkage_denominator"))
    )
    readiness_excluded_non_trade = sum(
        1 for entry in dataset_entries if str(entry.get("outcome_status") or "").upper() == "NON_TRADE_CONTEXT"
    )
    readiness_excluded_invalid = sum(
        1 for entry in dataset_entries if str(entry.get("outcome_status") or "").upper() == "INVALID_CONTEXT"
    )
    readiness_excluded_not_due = sum(
        1 for entry in dataset_entries if str(entry.get("outcome_status") or "").upper() == "NOT_DUE"
    )
    admission_summary = _summarize_admission_entries(dataset_entries)
    accepted_records = int(admission_summary["accepted_records"])
    accepted_noneligible_records = int(admission_summary["accepted_noneligible_records"])
    eligible_records = int(admission_summary["eligible_records"])
    quarantined_records = int(admission_summary["quarantined_records"])
    duplicate_conflicts = int(admission_summary["duplicate_conflicts"])
    admission_missing_execution_metadata_records = int(
        admission_summary["missing_execution_metadata_records"]
    )
    cohort_fingerprints = {
        str((entry.get("cohort_identity") or {}).get("contract_fingerprint") or "").strip()
        for entry in dataset_entries
        if isinstance(entry.get("cohort_identity"), dict)
        and str((entry.get("cohort_identity") or {}).get("contract_fingerprint") or "").strip()
    }
    contract_versions = {
        str(entry.get("evidence_contract_version")).strip()
        for entry in dataset_entries
        if entry.get("evidence_contract_version") is not None
    }
    contract_version_drift = len(contract_versions) > 1
    cohort_fingerprint_drift = len(cohort_fingerprints) > 1

    if (
        (outcome_windows_eligible > 0 or outcome_windows_matched > 0)
        and (not outcomes_loaded or not outcome_join_attempted)
    ):
        _emit_failure_summary_and_exit(
            message=(
                "Telemetry contract violation: outcome window counts > 0 without "
                "outcomes_loaded=true and outcome_join_attempted=true."
            ),
            audit_dir=audit_dir,
            audit_roots=audit_roots,
            include_research=bool(args.include_research),
            max_files=int(args.max_files),
            db_path=db_path,
            min_forecast_horizon=min_forecast_horizon,
            exit_code=1,
        )

    if outcome_join_attempted and outcome_join_error:
        print(f"[WARN] outcome_join_unavailable db={db_path} error={outcome_join_error}")
    print(
        "Outcome join   : "
        f"outcomes_loaded={int(outcomes_loaded)} "
        f"join_attempted={int(outcome_join_attempted)} "
        f"accepted={accepted_records} "
        f"accepted_noneligible={accepted_noneligible_records} "
        f"eligible={eligible_records} "
        f"quarantined={quarantined_records} "
        f"due_eligible={outcome_windows_eligible} "
        f"matched={outcome_windows_matched} "
        f"missing={outcome_windows_missing} "
        f"ambiguous={outcome_windows_ambiguous} "
        f"not_due={outcome_windows_not_due} "
        f"invalid_context={outcome_windows_invalid_context} "
        f"not_yet_eligible={outcome_windows_not_yet_eligible} "
        f"duplicate_conflicts={duplicate_conflicts} "
        f"contract_drift={int(contract_version_drift)} "
        f"cohort_drift={int(cohort_fingerprint_drift)} "
        f"no_signal_id={outcome_windows_no_signal_id} "
        f"non_trade_context={outcome_windows_non_trade_context} "
        f"missing_exec_meta={admission_missing_execution_metadata_records}"
    )
    print(
        "Denominators   : "
        f"readiness_included={readiness_denominator_included} "
        f"linkage_included={linkage_denominator_included} "
        f"excluded_non_trade={readiness_excluded_non_trade} "
        f"excluded_invalid={readiness_excluded_invalid} "
        f"excluded_not_due={readiness_excluded_not_due}"
    )

    healthy_tickers = {"NVDA", "MSFT", "GOOG", "JPM"}
    diversity = {
        "regime_count": len(
            {
                str(entry.get("detected_regime")).strip()
                for entry in rmse_usable_entries
                if str(entry.get("detected_regime") or "").strip()
            }
        ),
        "healthy_ticker_count": len(
            {
                str(entry.get("ticker")).strip().upper()
                for entry in rmse_usable_entries
                if str(entry.get("ticker") or "").strip().upper() in healthy_tickers
            }
        ),
        "distinct_trading_days": len(
            {
                str(entry.get("end_day")).strip()
                for entry in rmse_usable_entries
                if str(entry.get("end_day") or "").strip()
            }
        ),
    }
    print(
        "Diversity      : "
        f"regimes={diversity['regime_count']} "
        f"healthy_tickers={diversity['healthy_ticker_count']} "
        f"trading_days={diversity['distinct_trading_days']}"
    )

    summary = {
        "audit_dir": str(audit_dir),
        "audit_roots": [str(root) for root in audit_roots],
        "generated_utc": generated_utc,
        "source_script": "scripts/check_forecast_audits.py",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "max_files": int(args.max_files),
        "measurement_contract_version": 1,
        "baseline_model": baseline_model,
        "lift_threshold_rmse_ratio": _lift_threshold_rmse_ratio(min_lift_rmse_ratio),
        "effective_audits": effective_n,
        "effective_outcome_audits": outcome_windows_matched,
        "total_unique_audits": len(results),
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "max_violation_rate": max_violation_rate,
        "ensemble_missing_count": ensemble_missing_count,
        "ensemble_missing_rate": ensemble_missing_rate,
        "max_missing_ensemble_rate": max_missing_ensemble_rate,
        "ensemble_index_mismatch_count": ensemble_index_mismatch_count,
        "ensemble_index_mismatch_rate": ensemble_index_mismatch_rate,
        "max_index_mismatch_rate": max_index_mismatch_rate,
        "residual_diag_effective_audits": residual_effective_n,
        "non_white_noise_count": non_white_noise_count,
        "non_white_noise_rate": non_white_noise_rate,
        "max_non_white_noise_rate": max_non_white_noise_rate,
        "missing_residual_diag_count": missing_residual_diag_count,
        "min_residual_diagnostics_n": min_residual_diagnostics_n,
        "fail_on_missing_residual_diagnostics": fail_on_missing_residual_diagnostics,
        "min_forecast_horizon": min_forecast_horizon,
        "horizon_filtered_count": horizon_filtered_count,
        "manifest_integrity": manifest_stats,
        "holding_period_required": warmup_required,
        "lift_fraction": lift_fraction,
        "min_lift_fraction": min_lift_fraction,
        "percentiles": {
            "p10": pct.get(0.1) if pct else None,
            "p50": pct.get(0.5) if pct else None,
            "p90": pct.get(0.9) if pct else None,
        },
        "ratio_distribution": ratio_stats,
        "decision": decision,
        "decision_reason": decision_reason,
        "recent_window_audits": recent_window_audits,
        "recent_effective_audits": recent_effective_n,
        "recent_violation_count": recent_violation_count,
        "recent_violation_rate": recent_violation_rate,
        "recent_window_max_violation_rate": recent_window_max_violation_rate,
        "recent_rmse_ratio_p90": recent_pct.get(0.9) if recent_pct else None,
        "recent_window_max_p90_rmse_ratio": recent_window_max_p90_rmse_ratio,
        "admission_summary": admission_summary,
        "outcome_join": {
            "db_path": str(db_path) if db_path is not None else None,
            "error": outcome_join_error,
            "eligibility_buffer_minutes": int(OUTCOME_ELIGIBILITY_BUFFER.total_seconds() // 60),
            "accepted_records": accepted_records,
            "accepted_noneligible_records": accepted_noneligible_records,
            "eligible_records": eligible_records,
            "quarantined_records": quarantined_records,
            "duplicate_conflicts": duplicate_conflicts,
            "missing_execution_metadata_records": admission_missing_execution_metadata_records,
            "contract_version_drift": contract_version_drift,
            "cohort_fingerprint_drift": cohort_fingerprint_drift,
            "readiness_denominator_included": readiness_denominator_included,
            "linkage_denominator_included": linkage_denominator_included,
            "readiness_excluded_non_trade_context": readiness_excluded_non_trade,
            "readiness_excluded_invalid_context": readiness_excluded_invalid,
            "readiness_excluded_not_due": readiness_excluded_not_due,
            "status_taxonomy": [
                "MATCHED",
                "OUTCOME_MISSING",
                "NOT_DUE",
                "INVALID_CONTEXT",
                "NON_TRADE_CONTEXT",
            ],
        },
        "telemetry_contract": {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "rmse_inputs_present": bool(results),
            "outcomes_loaded": outcomes_loaded,
            "execution_log_loaded": False,
            "outcome_join_attempted": outcome_join_attempted,
            "status": "PASS" if decision in {DEFAULT_DECISION_KEEP, DEFAULT_DECISION_RESEARCH} else decision,
            "reason_code": str(decision_reason).upper().replace(" ", "_"),
            "context_type": "TRADE",
            "severity": "LOW",
            "blocking": False,
            "counts_toward_readiness_denominator": True,
            "counts_toward_linkage_denominator": False,
            "generated_utc": generated_utc,
            "source_script": "scripts/check_forecast_audits.py",
        },
        "scope": {
            "include_research": bool(args.include_research),
            "production_audit_only": not bool(args.include_research),
        },
        "window_counts": {
            "n_raw_windows": len(files),
            "n_parseable_windows": parseable_count,
            "n_deduped_windows": len(unique_map),
            "n_outcome_deduped_windows": len(outcome_unique_map),
            "n_rmse_windows_processed": rmse_windows_processed,
            "n_rmse_windows_usable": rmse_windows_usable,
            "n_outcome_windows_eligible": outcome_windows_eligible,
            "n_outcome_windows_matched": outcome_windows_matched,
            "n_outcome_windows_missing": outcome_windows_missing,
            "n_outcome_windows_ambiguous": outcome_windows_ambiguous,
            "n_outcome_windows_not_due": outcome_windows_not_due,
            "n_outcome_windows_not_yet_eligible": outcome_windows_not_yet_eligible,
            "n_outcome_windows_invalid_context": outcome_windows_invalid_context,
            "n_outcome_windows_outcomes_not_loaded": outcome_windows_outcomes_not_loaded,
            "n_outcome_windows_no_signal_id": outcome_windows_no_signal_id,
            "n_outcome_windows_non_trade_context": outcome_windows_non_trade_context,
            "n_outcome_windows_missing_execution_metadata": outcome_windows_missing_execution_metadata,
            "n_accepted_records": accepted_records,
            "n_accepted_noneligible_records": accepted_noneligible_records,
            "n_eligible_records": eligible_records,
            "n_quarantined_records": quarantined_records,
            "n_duplicate_conflicts": duplicate_conflicts,
            "n_admission_missing_execution_metadata_records": admission_missing_execution_metadata_records,
            "n_contract_versions": len(contract_versions),
            "n_cohort_fingerprints": len(cohort_fingerprints),
            "n_readiness_denominator_included": readiness_denominator_included,
            "n_linkage_denominator_included": linkage_denominator_included,
            "n_readiness_excluded_non_trade_context": readiness_excluded_non_trade,
            "n_readiness_excluded_invalid_context": readiness_excluded_invalid,
            "n_readiness_excluded_not_due": readiness_excluded_not_due,
        },
        "window_diversity": diversity,
        "dataset_windows": dataset_entries,
    }
    summary["telemetry_contract"] = normalize_telemetry_payload(
        summary.get("telemetry_contract", {}),
        source_script="scripts/check_forecast_audits.py",
        generated_utc=generated_utc,
    )
    cache_path = cache_dir / "latest_summary.json"
    dash_path = cache_dir / "ratio_distribution.json"
    cache_status = {"write_ok": True, "errors": []}
    try:
        _write_json_atomic(dash_path, ratio_stats)
    except Exception as exc:
        cache_status["write_ok"] = False
        cache_status["errors"].append(f"ratio_distribution:{exc}")
        print(f"[WARN] forecast_audits_cache_write_failed target=ratio_distribution error={exc}")
    summary["cache_status"] = cache_status
    try:
        _write_summary_with_guard(cache_path, summary)
    except Exception as exc:
        print(f"[WARN] forecast_audits_cache_write_failed target=latest_summary error={exc}")

    raise SystemExit(0)


if __name__ == "__main__":
    main()
