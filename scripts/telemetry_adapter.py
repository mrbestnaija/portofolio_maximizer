#!/usr/bin/env python3
"""
Telemetry schema v3 adapter.

Adds backward-compatible normalization for legacy producer payloads without
renaming/removing existing keys.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


TELEMETRY_SCHEMA_VERSION_V3 = 3

VALID_STATUS = {
    "PASS",
    "FAIL",
    "BLOCKED",
    "INCONCLUSIVE_ALLOWED",
    "INCONCLUSIVE_BLOCKED",
    "WARN",
    "STALE",
    "INVALID_CONTEXT",
}

STATUS_SHIMS = {
    "INCONCLUSIVE": "INCONCLUSIVE_ALLOWED",
    "OK": "PASS",
    "SUCCESS": "PASS",
}

VALID_REASON_CODES = {
    "WARMUP_NOT_EXPIRED",
    "UNPROFITABLE_PROOF",
    "HORIZON_MISMATCH",
    "CAUSALITY_VIOLATION",
    "MISSING_TS_SIGNAL_ID",
    "NON_TRADE_CONTEXT",
    "DATE_FALLBACK_USED",
    "STALE_SIDECAR",
    "INSUFFICIENT_PRESELECTION_EVIDENCE",
    "MISSING_EXECUTION_METADATA",
}

VALID_SEVERITY = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def telemetry_now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def telemetry_age_minutes(
    payload: Dict[str, Any] | None,
    *,
    timestamp_keys: Iterable[str] = ("generated_utc", "timestamp_utc"),
    fallback_path: Path | None = None,
    now_utc: datetime | None = None,
) -> Optional[float]:
    parsed: Optional[datetime] = None
    if isinstance(payload, dict):
        for key in timestamp_keys:
            parsed = parse_utc_datetime(payload.get(key))
            if parsed is not None:
                break
    if parsed is None and fallback_path is not None:
        try:
            parsed = datetime.fromtimestamp(Path(fallback_path).stat().st_mtime, tz=timezone.utc)
        except Exception:
            parsed = None
    if parsed is None:
        return None
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (now - parsed).total_seconds() / 60.0
    return age if age >= 0 else 0.0


def sha256_file(
    path: Path,
    *,
    max_bytes: int = 0,
    chunk_bytes: int = 1024 * 1024,
) -> tuple[Optional[str], Optional[str]]:
    """
    Return (sha256_hex, skip_reason). Never raises.

    skip_reason values:
      - stat_failed
      - too_large><max_bytes>
      - read_failed
    """
    try:
        size = int(Path(path).stat().st_size)
    except Exception:
        return None, "stat_failed"

    if max_bytes > 0 and size > max_bytes:
        return None, f"too_large>{max_bytes}"

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest(), None
    except Exception:
        return None, "read_failed"


def normalize_status(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return "WARN"
    text = STATUS_SHIMS.get(text, text)
    if text in VALID_STATUS:
        return text
    if text in {"KEEP"}:
        return "PASS"
    if text in {"RESEARCH_ONLY", "DISABLE_DEFAULT"}:
        return "WARN"
    return "WARN"


def normalize_reason_code(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return "INSUFFICIENT_PRESELECTION_EVIDENCE"
    text = text.replace("-", "_").replace(" ", "_")
    if text in VALID_REASON_CODES:
        return text
    return text


def normalize_telemetry_payload(
    payload: Dict[str, Any],
    *,
    source_script: str,
    generated_utc: str | None = None,
) -> Dict[str, Any]:
    out = dict(payload or {})
    out["schema_version"] = TELEMETRY_SCHEMA_VERSION_V3
    out["status"] = normalize_status(out.get("status"))
    out["reason_code"] = normalize_reason_code(out.get("reason_code"))

    context_type = str(out.get("context_type") or "TRADE").strip().upper() or "TRADE"
    out["context_type"] = context_type

    severity = str(out.get("severity") or "LOW").strip().upper()
    if severity not in VALID_SEVERITY:
        severity = "LOW"
    out["severity"] = severity

    out["blocking"] = bool(out.get("blocking", False))
    out["counts_toward_readiness_denominator"] = bool(
        out.get("counts_toward_readiness_denominator", False)
    )
    out["counts_toward_linkage_denominator"] = bool(
        out.get("counts_toward_linkage_denominator", False)
    )
    out["generated_utc"] = str(generated_utc or out.get("generated_utc") or telemetry_now_utc())
    out["source_script"] = str(out.get("source_script") or source_script)
    return out
