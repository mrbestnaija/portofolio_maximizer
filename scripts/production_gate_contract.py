from __future__ import annotations

from typing import Any, Dict


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def readiness_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(payload.get("readiness"))


def production_gate_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(payload.get("production_profitability_gate"))


def legacy_phase3_ready(payload: Dict[str, Any]) -> bool:
    readiness = readiness_block(payload)
    if "phase3_ready" in readiness:
        return bool(readiness.get("phase3_ready"))
    return bool(payload.get("phase3_ready"))


def legacy_phase3_reason(payload: Dict[str, Any]) -> str:
    readiness = readiness_block(payload)
    reason = readiness.get("phase3_reason")
    if reason is None:
        reason = payload.get("phase3_reason")
    return str(reason or "").strip()


def gate_semantics_status(payload: Dict[str, Any]) -> str:
    gate = production_gate_block(payload)
    return str(gate.get("gate_semantics_status") or "").strip().upper()


def strict_gate_pass(payload: Dict[str, Any]) -> bool:
    gate = production_gate_block(payload)
    if "strict_pass" in gate:
        return bool(gate.get("strict_pass"))

    semantics = gate_semantics_status(payload)
    if semantics:
        return bool(gate.get("pass")) and semantics == "PASS"
    return bool(gate.get("pass"))


def phase3_strict_ready(payload: Dict[str, Any]) -> bool:
    readiness = readiness_block(payload)
    if "phase3_strict_ready" in readiness:
        return bool(readiness.get("phase3_strict_ready"))
    if "phase3_strict_ready" in payload:
        return bool(payload.get("phase3_strict_ready"))

    semantics = gate_semantics_status(payload)
    ready = legacy_phase3_ready(payload)
    if semantics:
        return ready and semantics == "PASS"
    return ready


def phase3_strict_reason(payload: Dict[str, Any]) -> str:
    readiness = readiness_block(payload)
    reason = readiness.get("phase3_strict_reason")
    if reason is None:
        reason = payload.get("phase3_strict_reason")
    if reason is not None and str(reason).strip():
        return str(reason).strip()

    legacy_reason = legacy_phase3_reason(payload)
    semantics = gate_semantics_status(payload)
    if legacy_phase3_ready(payload) and semantics and semantics != "PASS":
        strict_reason = f"GATE_SEMANTICS_{semantics}"
        return f"{legacy_reason},{strict_reason}" if legacy_reason else strict_reason
    return legacy_reason
