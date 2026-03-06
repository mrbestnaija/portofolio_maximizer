#!/usr/bin/env python3
"""
Telemetry schema v3 adapter.

Adds backward-compatible normalization for legacy producer payloads without
renaming/removing existing keys.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


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

