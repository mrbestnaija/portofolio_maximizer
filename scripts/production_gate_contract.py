from __future__ import annotations

from typing import Any, Dict


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def readiness_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(payload.get("readiness"))


def production_gate_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(payload.get("production_profitability_gate"))


def profitability_proof_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(payload.get("profitability_proof"))


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


def covered_state_reasons(payload: Dict[str, Any]) -> list[str]:
    gate = production_gate_block(payload)
    readiness = readiness_block(payload)
    raw = (
        gate.get("covered_state_reasons")
        if isinstance(gate.get("covered_state_reasons"), list)
        else readiness.get("covered_state_reasons")
    )
    if not isinstance(raw, list):
        raw = payload.get("covered_state_reasons")
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out

    reasons: list[str] = []
    if not legacy_phase3_ready(payload):
        return reasons

    semantics = gate_semantics_status(payload)
    if semantics and semantics != "PASS":
        reasons.append(f"gate_semantics_{semantics.lower()}")

    proof = profitability_proof_block(payload)
    evidence_progress = _as_dict(proof.get("evidence_progress"))
    if evidence_progress and evidence_progress.get("ready") is False:
        reasons.append("proof_evidence_incomplete")

    linkage_warmup_active = bool(readiness.get("linkage_warmup_active"))
    linkage_full_thresholds_pass = readiness.get("linkage_full_thresholds_pass")
    if linkage_warmup_active and linkage_full_thresholds_pass is False:
        reasons.append("linkage_warmup_exemption")

    return reasons


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


def phase3_posture(payload: Dict[str, Any]) -> str:
    gate = production_gate_block(payload)
    readiness = readiness_block(payload)
    explicit = (
        readiness.get("posture")
        if readiness.get("posture") is not None
        else gate.get("posture")
    )
    if explicit is None:
        explicit = payload.get("posture")
    explicit_text = str(explicit or "").strip().upper()
    if explicit_text in {"GENUINE_PASS", "WARMUP_COVERED_PASS", "FAIL"}:
        return explicit_text

    if not legacy_phase3_ready(payload):
        return "FAIL"
    if covered_state_reasons(payload):
        return "WARMUP_COVERED_PASS"
    return "GENUINE_PASS"


def phase3_genuine_ready(payload: Dict[str, Any]) -> bool:
    return phase3_posture(payload) == "GENUINE_PASS"


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
