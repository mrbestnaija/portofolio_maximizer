#!/usr/bin/env python3
"""
PowerShell-safe runtime status snapshot for Portfolio Maximizer.

Why this exists:
- OpenClaw `exec` runs in Windows PowerShell on some hosts.
- PowerShell 5 does not support `&&`, so chained shell commands can fail.
- This script runs key checks sequentially from Python without shell chaining.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import dashboard_db_bridge as dashboard_bridge

try:
    from scripts.production_gate_contract import (
        gate_semantics_status as _gate_semantics_status,
        phase3_strict_ready as _phase3_strict_ready,
    )
except Exception:  # pragma: no cover - script execution path fallback
    from production_gate_contract import (  # type: ignore
        gate_semantics_status as _gate_semantics_status,
        phase3_strict_ready as _phase3_strict_ready,
    )

VALID_EXEC_HOSTS = {"sandbox", "gateway", "node"}
VALID_SANDBOX_MODES_FOR_SANDBOX_HOST = {"non-main", "all"}
DEFAULT_DASHBOARD_PATH = PROJECT_ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_CANONICAL_SNAPSHOT_PATH = PROJECT_ROOT / "logs" / "canonical_snapshot_latest.json"
DEFAULT_PERSISTENCE_STATUS_PATH = PROJECT_ROOT / "logs" / "persistence_manager_status.json"
DEFAULT_RUNTIME_STATUS_PATH = PROJECT_ROOT / "logs" / "runtime_status_latest.json"
DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH = dashboard_bridge.DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH
try:
    DEFAULT_AUTOMATION_CYCLE_INTERVAL_SECONDS = max(
        int(os.getenv("PMX_AUTOMATION_CRON_INTERVAL_SECONDS", "1800")),
        1,
    )
except Exception:
    DEFAULT_AUTOMATION_CYCLE_INTERVAL_SECONDS = 1800
_PRODUCTION_GATE_SEMANTICS_RE = re.compile(r"semantics=([A-Z_]+)")


def _trim(text: str, max_chars: int = 1400) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "\n...[truncated]..."


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _agent_allows_exec(agent: dict[str, Any]) -> bool:
    tools = _as_dict(agent.get("tools"))
    deny = {str(item).strip().lower() for item in _as_list(tools.get("deny")) if str(item).strip()}
    if "exec" in deny or "group:runtime" in deny:
        return False

    allow = {str(item).strip().lower() for item in _as_list(tools.get("allow")) if str(item).strip()}
    if allow:
        return "exec" in allow or "group:runtime" in allow

    profile = str(tools.get("profile") or "").strip().lower()
    return profile != "messaging"


def _invalid_sandbox_override_agents(cfg: dict[str, Any]) -> list[str]:
    tools = _as_dict(cfg.get("tools"))
    exec_cfg = _as_dict(tools.get("exec"))
    host = str(exec_cfg.get("host") or "").strip().lower()
    if host != "sandbox":
        return []

    agents = _as_dict(cfg.get("agents"))
    invalid: list[str] = []
    for agent in _as_list(agents.get("list")):
        if not isinstance(agent, dict) or not _agent_allows_exec(agent):
            continue
        sandbox = _as_dict(agent.get("sandbox"))
        mode = str(sandbox.get("mode") or "").strip().lower()
        if mode and mode not in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST:
            aid = str(agent.get("id") or "?").strip() or "?"
            invalid.append(aid)
    return invalid


def _docker_sandbox_available(timeout_seconds: float = 5.0) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return int(proc.returncode) == 0


def _node_host_available(timeout_seconds: float = 6.0) -> bool:
    """Return True if at least one OpenClaw node is paired and reachable.

    Calls ``openclaw nodes list --json`` with a short timeout. Any failure
    (missing CLI, non-zero exit, empty list, JSON parse error) is treated as
    unavailable — matching the pattern of ``_docker_sandbox_available``.
    """
    try:
        proc = subprocess.run(
            ["openclaw", "nodes", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    if int(proc.returncode) != 0:
        return False
    try:
        payload = json.loads(proc.stdout or "")
    except Exception:
        return False
    # Accept a top-level list or a dict with a "nodes" key.
    if isinstance(payload, list):
        nodes = payload
    elif isinstance(payload, dict):
        nodes = payload.get("nodes") or []
    else:
        return False
    return bool(nodes)


def _read_openclaw_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _openclaw_config_candidates() -> list[Path]:
    candidates = [Path.home() / ".openclaw" / "openclaw.json"]
    try:
        project_home = PROJECT_ROOT.parents[2]
    except IndexError:
        project_home = None
    if project_home is not None:
        derived = project_home / ".openclaw" / "openclaw.json"
        if derived not in candidates:
            candidates.append(derived)
    return candidates


def _resolve_openclaw_json_path() -> tuple[Path, list[Path]]:
    candidates = _openclaw_config_candidates()
    for path in candidates:
        if path.exists():
            return path, candidates
    return candidates[0], candidates


def _read_json_file(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing:{path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {}, f"unreadable:{path.name}:{exc}"
    return (payload, None) if isinstance(payload, dict) else ({}, f"invalid:{path.name}:root_not_object")


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload_age_seconds(payload: dict[str, Any]) -> float | None:
    if not isinstance(payload, dict):
        return None
    candidates: list[Any] = [payload.get(key) for key in ("timestamp_utc", "generated_utc", "generated_at", "ts")]
    meta = payload.get("meta")
    if isinstance(meta, dict):
        candidates.extend(meta.get(key) for key in ("timestamp_utc", "generated_utc", "generated_at", "ts"))
    for value in candidates:
        parsed = _parse_iso_datetime(value)
        if parsed is not None:
            return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _make_check(
    name: str,
    *,
    ok: bool,
    returncode: int,
    command: str,
    stdout: str = "",
    stderr: str = "",
    duration_seconds: float = 0.0,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "ok": ok,
        "returncode": int(returncode),
        "duration_seconds": round(float(duration_seconds), 3),
        "command": command,
        "stdout": _trim(stdout),
        "stderr": _trim(stderr),
    }
    payload.update(extra)
    return payload


def _run_check(
    name: str,
    cmd: list[str],
    timeout_seconds: float,
    *,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.time()
    env = dict(os.environ)
    for key, value in (env_overrides or {}).items():
        text = str(value or "").strip()
        if text:
            env[key] = text
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            env=env,
        )
        duration = time.time() - start
        return {
            "name": name,
            "ok": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "duration_seconds": round(duration, 3),
            "command": " ".join(cmd),
            "stdout": _trim(proc.stdout or ""),
            "stderr": _trim(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        return {
            "name": name,
            "ok": False,
            "returncode": 124,
            "duration_seconds": round(duration, 3),
            "command": " ".join(cmd),
            "stdout": _trim(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr": _trim((exc.stderr if isinstance(exc.stderr, str) else "") or "timeout"),
        }
    except FileNotFoundError as exc:
        duration = time.time() - start
        return {
            "name": name,
            "ok": False,
            "returncode": 127,
            "duration_seconds": round(duration, 3),
            "command": " ".join(cmd),
            "stdout": "",
            "stderr": _trim(str(exc)),
        }


def _extract_production_gate_semantics(
    check: dict[str, Any],
    artifact: dict[str, Any],
) -> str:
    artifact_semantics = _gate_semantics_status(artifact)
    if artifact_semantics:
        return artifact_semantics
    for field in ("stdout", "stderr"):
        text = str(check.get(field) or "")
        match = _PRODUCTION_GATE_SEMANTICS_RE.search(text)
        if match:
            return match.group(1)
    if (
        bool(check.get("ok"))
        and bool(artifact.get("phase3_ready"))
        and bool(artifact.get("lift_inconclusive_allowed"))
        and not bool(artifact.get("warmup_expired", True))
    ):
        return "INCONCLUSIVE_ALLOWED"
    if bool(check.get("ok")) and bool(_phase3_strict_ready(artifact)):
        return "READY"
    return ""


def _strict_production_gate_check(base_check: dict[str, Any]) -> dict[str, Any]:
    artifact, err = _read_json_file(DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH)
    semantics = _extract_production_gate_semantics(base_check, artifact)
    errors: list[str] = []
    if not bool(base_check.get("ok")):
        errors.append("base_check_failed")
    if err:
        errors.append(err)
    if semantics == "INCONCLUSIVE_ALLOWED":
        errors.append("warmup_only_pass")
    return _make_check(
        "strict_production_gate",
        ok=not errors,
        returncode=0 if not errors else 1,
        command=f"validate {DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH.name}",
        stdout=f"semantics={semantics or 'UNKNOWN'}",
        stderr="; ".join(errors),
        semantics=semantics or "UNKNOWN",
        phase3_ready=bool(_phase3_strict_ready(artifact)),
        warmup_expired=bool(artifact.get("warmup_expired", True)),
    )


def _strict_dashboard_payload_check() -> dict[str, Any]:
    report = dashboard_bridge.validate_dashboard_payload_file(DEFAULT_DASHBOARD_PATH)
    return _make_check(
        "strict_dashboard_payload",
        ok=bool(report.get("ok")),
        returncode=0 if bool(report.get("ok")) else 1,
        command=f"validate {DEFAULT_DASHBOARD_PATH.name}",
        stdout=(
            f"generated_utc={report.get('generated_utc')} age_seconds={report.get('age_seconds')} "
            f"threshold_seconds={report.get('freshness_threshold_seconds')}"
        ),
        stderr="; ".join(str(item) for item in report.get("errors", [])),
        age_seconds=report.get("age_seconds"),
        freshness_threshold_seconds=report.get("freshness_threshold_seconds"),
        missing_keys=report.get("missing_keys", []),
    )


def _strict_canonical_snapshot_check() -> dict[str, Any]:
    payload, err = _read_json_file(DEFAULT_CANONICAL_SNAPSHOT_PATH)
    errors: list[str] = []
    warnings: list[str] = []
    if err:
        errors.append(err)

    schema_version = int(payload.get("schema_version") or 0) if payload else 0
    if schema_version == 0:
        errors.append("schema_version_0")
    elif schema_version < 4:
        errors.append("invalid_schema_version")
    emission_error = str(payload.get("emission_error") or "").strip() if payload else ""
    if emission_error:
        errors.append(f"emission_error:{emission_error}")

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    utilization = payload.get("utilization") if isinstance(payload.get("utilization"), dict) else {}
    alpha_objective = payload.get("alpha_objective") if isinstance(payload.get("alpha_objective"), dict) else {}
    source_contract = payload.get("source_contract") if isinstance(payload.get("source_contract"), dict) else {}
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    thin_linkage = payload.get("thin_linkage") if isinstance(payload.get("thin_linkage"), dict) else {}

    ann_roi = alpha_objective.get("roi_ann_pct")
    if ann_roi is None and utilization:
        ann_roi = utilization.get("roi_ann_pct")
    if ann_roi is None and summary:
        ann_roi = summary.get("roi_ann_pct") or summary.get("ann_roi_pct")
    if ann_roi is None:
        errors.append("missing_roi_ann_pct")
    elif summary.get("objective_valid") is False or alpha_objective.get("objective_valid") is False:
        warnings.append("objective_not_rankable")

    objective_valid = alpha_objective.get("objective_valid")
    objective_score = alpha_objective.get("objective_score")
    if objective_valid is True and objective_score is None:
        errors.append("missing_objective_score")
    if objective_valid is False and objective_score is not None:
        errors.append("objective_score_present_when_invalid")

    if summary.get("unattended_gate") is None:
        errors.append("missing_unattended_gate")
    if summary.get("unattended_ready") is None:
        errors.append("missing_unattended_ready")
    evidence_health = str(summary.get("evidence_health") or "").strip().lower()
    if not evidence_health:
        errors.append("missing_evidence_health")
    elif evidence_health == "degraded":
        errors.append("degraded_evidence_health:degraded")
    elif evidence_health in {"bridge_state", "warmup_expired_fail"}:
        warnings.append(f"non_green_evidence_health:{evidence_health}")
    elif evidence_health != "clean":
        errors.append(f"unexpected_evidence_health:{evidence_health}")

    freshness = gate.get("freshness_status") if isinstance(gate.get("freshness_status"), dict) else {}
    if not freshness:
        errors.append("missing_gate_freshness_status")
    else:
        freshness_status = str(freshness.get("status") or "").strip().lower()
        if freshness_status not in {"fresh", "stale", "missing"}:
            errors.append(f"invalid_gate_freshness_status:{freshness_status}")
        elif freshness_status != "fresh":
            errors.append(f"freshness_not_fresh:{freshness_status}")
    warmup_state = gate.get("warmup_state") if isinstance(gate.get("warmup_state"), dict) else {}
    if not warmup_state:
        errors.append("missing_gate_warmup_state")
    else:
        posture = str(warmup_state.get("posture") or "").strip().lower()
        if posture not in {"active", "expired"}:
            errors.append(f"invalid_warmup_posture:{posture}")
    trajectory_alarm = gate.get("trajectory_alarm") if isinstance(gate.get("trajectory_alarm"), dict) else {}
    if not trajectory_alarm:
        errors.append("missing_trajectory_alarm")
    elif trajectory_alarm.get("active") is None:
        errors.append("missing_trajectory_alarm.active")
    coverage_alarm = gate.get("coverage_ratio_alarm") if isinstance(gate.get("coverage_ratio_alarm"), dict) else {}
    if coverage_alarm and coverage_alarm.get("active"):
        warnings.append("coverage_ratio_alarm_active")
    post_deadline_estimate = gate.get("post_deadline_time_to_10_estimate") if isinstance(gate.get("post_deadline_time_to_10_estimate"), dict) else {}
    if not post_deadline_estimate:
        errors.append("missing_post_deadline_time_to_10_estimate")
    elif post_deadline_estimate.get("status") is None:
        errors.append("missing_post_deadline_time_to_10_estimate.status")

    if not source_contract:
        errors.append("missing_source_contract")
    else:
        contract_status = str(source_contract.get("status") or "").strip().lower()
        if contract_status != "clean":
            errors.append(f"source_contract_status:{contract_status or 'missing'}")
        canonical_sources = source_contract.get("canonical_sources")
        allowlisted_readers = source_contract.get("allowlisted_readers")
        violations_found = source_contract.get("violations_found")
        if not isinstance(canonical_sources, list):
            errors.append("missing_source_contract.canonical_sources")
        if not isinstance(allowlisted_readers, list):
            errors.append("missing_source_contract.allowlisted_readers")
        if not isinstance(violations_found, list):
            errors.append("missing_source_contract.violations_found")

    if thin_linkage.get("matched_current") is None:
        errors.append("missing_matched_current")

    if objective_valid is True and summary.get("roi_ann_pct") is None and ann_roi is not None:
        warnings.append("objective_valid_alias_missing")

    ann_roi_value = ann_roi

    return _make_check(
        "strict_canonical_snapshot",
        ok=not errors,
        returncode=0 if not errors else 1,
        command=f"validate {DEFAULT_CANONICAL_SNAPSHOT_PATH.name}",
        stdout=(
            f"schema_version={schema_version} "
            f"ann_roi_pct={ann_roi_value} "
            f"evidence_health={evidence_health or 'missing'} "
            f"freshness_status={freshness.get('status') if freshness else 'missing'} "
            f"objective_valid={objective_valid} "
            f"trajectory_alarm={trajectory_alarm.get('active') if trajectory_alarm else None} "
            f"coverage_ratio_alarm={coverage_alarm.get('active') if coverage_alarm else None} "
            f"unattended_gate={summary.get('unattended_gate') if summary else None} "
            f"posture={warmup_state.get('posture') if warmup_state else None}"
        ),
        stderr="; ".join(errors),
        schema_version=schema_version,
        ann_roi_pct=ann_roi_value,
        evidence_health=evidence_health or "missing",
        freshness_status=freshness.get("status") if freshness else "missing",
        freshness_age_minutes=freshness.get("age_minutes") if freshness else None,
        gate_artifact_age_minutes=gate.get("gate_artifact_age_minutes") if gate else None,
        objective_valid=objective_valid,
        objective_score=objective_score,
        trajectory_alarm_active=trajectory_alarm.get("active") if trajectory_alarm else None,
        coverage_ratio_alarm_active=coverage_alarm.get("active") if coverage_alarm else None,
        gap_to_hurdle_pp=summary.get("gap_to_hurdle_pp") if summary else None,
        unattended_gate=summary.get("unattended_gate") if summary else None,
        unattended_ready=summary.get("unattended_ready") if summary else None,
        gate_posture=warmup_state.get("posture") if warmup_state else None,
        source_contract_present=bool(source_contract),
        warnings=warnings,
    )


def _automation_ready_for_thin_linkage_check() -> dict[str, Any]:
    runtime_status_payload, runtime_status_err = _read_json_file(DEFAULT_RUNTIME_STATUS_PATH)
    runtime_status_source = DEFAULT_RUNTIME_STATUS_PATH
    runtime_status_age_seconds = _payload_age_seconds(runtime_status_payload)

    auto_trader_payload, auto_trader_err = _read_json_file(DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH)
    auto_trader_age_seconds = _payload_age_seconds(auto_trader_payload)

    snapshot_payload, snapshot_err = _read_json_file(DEFAULT_CANONICAL_SNAPSHOT_PATH)
    snapshot_age_minutes = None
    snapshot_freshness_status = "missing"
    snapshot_objective_valid = False
    snapshot_schema_version = 0
    snapshot_emission_error = ""
    source_contract_status = "missing"
    matched_current = None
    matched_needed = None
    reasons: list[str] = []

    if runtime_status_err is not None and runtime_status_age_seconds is None:
        runtime_status_source = DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH
        runtime_status_age_seconds = auto_trader_age_seconds
    if runtime_status_age_seconds is None:
        reasons.append("runtime_status_missing")
    elif runtime_status_age_seconds > 2 * DEFAULT_AUTOMATION_CYCLE_INTERVAL_SECONDS:
        reasons.append(
            f"runtime_status_stale:age_seconds={runtime_status_age_seconds:.1f}:"
            f"threshold_seconds={2 * DEFAULT_AUTOMATION_CYCLE_INTERVAL_SECONDS}"
        )

    auto_runtime = auto_trader_payload.get("runtime_status") if isinstance(auto_trader_payload, dict) else {}
    if not isinstance(auto_runtime, dict):
        auto_runtime = {}
    cycle_statuses = auto_runtime.get("eligibility_snapshot_statuses")
    latest_cycle_status = ""
    if isinstance(cycle_statuses, list) and cycle_statuses:
        latest_cycle_status = str(cycle_statuses[-1] or "").strip().upper()
    if not latest_cycle_status:
        latest_cycle_status = str(
            auto_runtime.get("eligibility_snapshot_status")
            or auto_trader_payload.get("cycle_status")
            or ""
        ).strip().upper()
    if not latest_cycle_status:
        reasons.append("missing_runtime_cycle_status")
    elif latest_cycle_status == "QUARANTINED":
        reasons.append("latest_cycle_quarantined")

    if snapshot_err is not None:
        reasons.append(snapshot_err)

    if isinstance(snapshot_payload, dict) and snapshot_payload:
        snapshot_schema_version = _safe_int(snapshot_payload.get("schema_version")) or 0
        snapshot_emission_error = str(snapshot_payload.get("emission_error") or "").strip()
        summary = snapshot_payload.get("summary") if isinstance(snapshot_payload.get("summary"), dict) else {}
        gate = snapshot_payload.get("gate") if isinstance(snapshot_payload.get("gate"), dict) else {}
        thin_linkage = snapshot_payload.get("thin_linkage") if isinstance(snapshot_payload.get("thin_linkage"), dict) else {}
        alpha_objective = snapshot_payload.get("alpha_objective") if isinstance(snapshot_payload.get("alpha_objective"), dict) else {}
        source_contract = snapshot_payload.get("source_contract") if isinstance(snapshot_payload.get("source_contract"), dict) else {}
        freshness = gate.get("freshness_status") if isinstance(gate.get("freshness_status"), dict) else {}
        snapshot_freshness_status = str(freshness.get("status") or "").strip().lower()
        snapshot_age_minutes = _safe_float(freshness.get("age_minutes"))
        source_contract_status = str(source_contract.get("status") or "").strip().lower() or "missing"
        snapshot_objective_valid = bool(alpha_objective.get("objective_valid"))
        matched_current = _safe_int(thin_linkage.get("matched_current"))
        matched_needed = _safe_int(thin_linkage.get("matched_needed"))

        if snapshot_schema_version != 4:
            reasons.append(f"schema_version_{snapshot_schema_version or 'missing'}")
        if snapshot_emission_error:
            reasons.append(f"emission_error:{snapshot_emission_error}")
        if snapshot_freshness_status != "fresh":
            reasons.append(f"snapshot_freshness:{snapshot_freshness_status or 'missing'}")
        if matched_current is None:
            reasons.append("missing_matched_current")
        if matched_needed is None:
            reasons.append("missing_matched_needed")
        elif matched_current is not None and matched_current < matched_needed:
            reasons.append(f"thin_linkage_shortfall:{matched_current}/{matched_needed}")
        if not snapshot_objective_valid:
            reasons.append("objective_invalid")
        if source_contract_status != "clean":
            reasons.append(f"source_contract:{source_contract_status}")
        if summary and str(summary.get("evidence_health") or "").strip().lower() == "degraded":
            reasons.append("evidence_health_degraded")
        if snapshot_age_minutes is not None:
            expected_max = _safe_float(freshness.get("expected_max_age_minutes")) or 0.0
            if expected_max > 0 and snapshot_age_minutes > expected_max:
                reasons.append(
                    f"snapshot_freshness_age_exceeded:{snapshot_age_minutes}:{expected_max}"
                )

    ready = not reasons and snapshot_schema_version == 4 and snapshot_objective_valid

    return {
        "ready": ready,
        "reasons": reasons,
        "runtime_status_path": str(DEFAULT_RUNTIME_STATUS_PATH),
        "runtime_status_source": str(runtime_status_source),
        "runtime_status_age_seconds": round(runtime_status_age_seconds, 3) if runtime_status_age_seconds is not None else None,
        "runtime_status_max_age_seconds": 2 * DEFAULT_AUTOMATION_CYCLE_INTERVAL_SECONDS,
        "run_auto_trader_artifact_path": str(DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH),
        "run_auto_trader_artifact_age_seconds": round(auto_trader_age_seconds, 3) if auto_trader_age_seconds is not None else None,
        "latest_cycle_status": latest_cycle_status or "UNKNOWN",
        "canonical_snapshot_path": str(DEFAULT_CANONICAL_SNAPSHOT_PATH),
        "canonical_snapshot_schema_version": snapshot_schema_version,
        "canonical_snapshot_freshness_status": snapshot_freshness_status or "missing",
        "canonical_snapshot_age_minutes": snapshot_age_minutes,
        "canonical_snapshot_objective_valid": snapshot_objective_valid,
        "canonical_snapshot_source_contract_status": source_contract_status,
        "thin_linkage_matched_current": matched_current,
        "thin_linkage_matched_needed": matched_needed,
    }


def _strict_persistence_manager_status_check() -> dict[str, Any]:
    payload, err = _read_json_file(DEFAULT_PERSISTENCE_STATUS_PATH)
    errors: list[str] = []
    if err:
        errors.append(err)
    contract = payload.get("status_contract") if isinstance(payload.get("status_contract"), dict) else {}
    if not contract:
        errors.append("missing_status_contract")
    max_age_seconds = 0
    try:
        max_age_seconds = max(int(contract.get("max_age_seconds") or 0), 0)
    except Exception:
        max_age_seconds = 0
    if max_age_seconds <= 0:
        errors.append("invalid_status_contract.max_age_seconds")

    generated = _parse_iso_datetime(payload.get("timestamp_utc"))
    if generated is None:
        errors.append("missing_timestamp_utc")
    age_seconds: float | None = None
    if generated is not None:
        age_seconds = max(0.0, (datetime.now(timezone.utc) - generated).total_seconds())
        if max_age_seconds > 0 and age_seconds > max_age_seconds:
            errors.append(
                f"stale_persistence_status:age_seconds={age_seconds:.1f}:threshold_seconds={max_age_seconds}"
            )

    dashboard = payload.get("dashboard") if isinstance(payload.get("dashboard"), dict) else {}
    bridge = dashboard.get("bridge") if isinstance(dashboard.get("bridge"), dict) else {}
    server = dashboard.get("http_server") if isinstance(dashboard.get("http_server"), dict) else {}
    watcher = dashboard.get("live_watcher") if isinstance(dashboard.get("live_watcher"), dict) else {}
    required_components = {
        "bridge": bool(bridge.get("running")),
        "http_server": bool(server.get("running")),
        "live_watcher": bool(watcher.get("running")),
    }
    for name, running in required_components.items():
        if not running:
            errors.append(f"{name}_not_running")

    warnings = dashboard.get("warnings") if isinstance(dashboard.get("warnings"), list) else []

    return _make_check(
        "strict_persistence_manager_status",
        ok=not errors,
        returncode=0 if not errors else 1,
        command=f"validate {DEFAULT_PERSISTENCE_STATUS_PATH.name}",
        stdout=(
            f"timestamp_utc={payload.get('timestamp_utc')} age_seconds={round(age_seconds, 3) if age_seconds is not None else None} "
            f"max_age_seconds={max_age_seconds}"
        ),
        stderr="; ".join(errors),
        age_seconds=round(age_seconds, 3) if age_seconds is not None else None,
        max_age_seconds=max_age_seconds,
        required_components=required_components,
        warnings=[str(item) for item in warnings if str(item).strip()],
    )


def _collect_observability_stack_check(*, timeout_seconds: float) -> dict[str, Any]:
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "scripts" / "status_observability_stack.ps1"),
        "-Json",
        "-RequireCurrent",
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
        duration = time.time() - start
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        return _make_check(
            "observability_stack",
            ok=False,
            returncode=124,
            command=" ".join(cmd),
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr if isinstance(exc.stderr, str) else "") or "timeout",
            duration_seconds=duration,
        )
    except FileNotFoundError as exc:
        duration = time.time() - start
        return _make_check(
            "observability_stack",
            ok=False,
            returncode=127,
            command=" ".join(cmd),
            stderr=str(exc),
            duration_seconds=duration,
        )

    payload: dict[str, Any] = {}
    parse_error = ""
    try:
        payload = json.loads(proc.stdout or "")
        if not isinstance(payload, dict):
            payload = {}
            parse_error = "stack_payload_not_object"
    except Exception as exc:
        parse_error = f"stack_payload_unreadable:{exc}"
    stack_status = str(payload.get("status") or "").strip().lower()
    errors: list[str] = []
    if parse_error:
        errors.append(parse_error)
    if stack_status != "ok":
        errors.append(f"stack_status={stack_status or 'unknown'}")
    if int(proc.returncode) != 0:
        errors.append(f"returncode={int(proc.returncode)}")
    if str(proc.stderr or "").strip():
        errors.append(_trim(proc.stderr or "", 400))
    return _make_check(
        "observability_stack",
        ok=not errors,
        returncode=int(proc.returncode),
        command=" ".join(cmd),
        stdout=(
            f"status={stack_status or 'unknown'} "
            f"legacy_sidecar_count={payload.get('legacy_sidecar_count')} "
            f"optional_missing_count={payload.get('optional_missing_count')}"
        ),
        stderr="; ".join(errors),
        duration_seconds=duration,
        stack_status=stack_status or "unknown",
        legacy_sidecar_count=int(payload.get("legacy_sidecar_count") or 0),
        optional_missing_count=int(payload.get("optional_missing_count") or 0),
    )


def _strict_observability_agreement_check(
    *,
    base_checks: list[dict[str, Any]],
    strict_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    base_ok = all(bool(check.get("ok")) for check in base_checks)
    stack_check = next((check for check in strict_checks if check.get("name") == "observability_stack"), None)
    stack_ok = bool(stack_check and stack_check.get("ok"))
    pmx_contract_failures = [
        str(check.get("name"))
        for check in strict_checks
        if check.get("name") != "observability_stack" and not bool(check.get("ok"))
    ]
    mismatches: list[str] = []
    if base_ok and not stack_ok:
        mismatches.append("runtime_ok_but_sidecar_stack_not_green")
    if stack_ok and pmx_contract_failures:
        mismatches.append("sidecar_stack_green_but_pmx_contract_not_green")
    if not base_ok and stack_ok:
        mismatches.append("sidecar_stack_green_but_base_runtime_degraded")
    if not mismatches and (not base_ok or not stack_ok or pmx_contract_failures):
        mismatches.append("strict_green_requirements_not_met")

    return _make_check(
        "strict_observability_agreement",
        ok=not mismatches,
        returncode=0 if not mismatches else 1,
        command="reconcile strict runtime vs sidecar stack",
        stdout=(
            f"base_runtime_ok={int(base_ok)} sidecar_stack_ok={int(stack_ok)} "
            f"pmx_contract_failures={','.join(pmx_contract_failures) if pmx_contract_failures else 'none'}"
        ),
        stderr="; ".join(mismatches),
    )


def _production_gate_artifact_check() -> dict[str, Any]:
    """Read the latest production gate artifact instead of re-running the gate.

    The live production audit gate is expensive and does not change THIN_LINKAGE; the
    runtime status probe only needs the latest artifact freshness/semantics for
    operator visibility.
    """
    artifact, err = _read_json_file(DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH)
    age_seconds = _payload_age_seconds(artifact)
    age_minutes = round(age_seconds / 60.0, 3) if age_seconds is not None else None
    errors: list[str] = []
    if err:
        errors.append(err)
    if not artifact:
        errors.append("missing_production_gate_artifact")
    if age_seconds is None:
        errors.append("missing_timestamp_utc")
    else:
        max_age_seconds = 2 * DEFAULT_AUTOMATION_CYCLE_INTERVAL_SECONDS
        if age_seconds > max_age_seconds:
            errors.append(
                f"production_gate_artifact_stale:age_seconds={age_seconds:.1f}:"
                f"threshold_seconds={max_age_seconds}"
            )

    semantics = _extract_production_gate_semantics({"ok": not errors}, artifact)
    stdout_bits = [f"semantics={semantics or 'UNKNOWN'}"]
    if age_minutes is not None:
        stdout_bits.append(f"artifact_age_minutes={age_minutes}")
    if artifact:
        stdout_bits.append(f"phase3_ready={int(bool(artifact.get('phase3_ready')))}")
        stdout_bits.append(f"warmup_expired={int(bool(artifact.get('warmup_expired', True)))}")

    return _make_check(
        "production_gate",
        ok=not errors,
        returncode=0 if not errors else 1,
        command=f"validate {DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH.name}",
        stdout=" ".join(stdout_bits),
        stderr="; ".join(errors),
        age_seconds=age_seconds,
        artifact_age_minutes=age_minutes,
        semantics=semantics or "UNKNOWN",
    )


def _openclaw_exec_environment_check() -> dict[str, Any]:
    cfg_path, cfg_candidates = _resolve_openclaw_json_path()
    check = {
        "name": "openclaw_exec_env",
        "ok": False,
        "returncode": 1,
        "duration_seconds": 0.0,
        "command": f"validate {cfg_path}",
        "stdout": "",
        "stderr": "",
        "signals": [],
        "checked_paths": [str(path) for path in cfg_candidates],
    }
    if not cfg_path.exists():
        check["signals"] = ["openclaw_config_missing"]
        check["stderr"] = f"openclaw config missing: {cfg_path}"
        return check
    try:
        cfg = _read_openclaw_json(cfg_path)
    except Exception as exc:
        check["signals"] = ["openclaw_config_unreadable"]
        check["stderr"] = f"openclaw config unreadable: {exc}"
        return check

    tools = cfg.get("tools", {}) if isinstance(cfg.get("tools"), dict) else {}
    exec_cfg = tools.get("exec", {}) if isinstance(tools.get("exec"), dict) else {}
    host = str(exec_cfg.get("host") or "").strip().lower()
    if host not in VALID_EXEC_HOSTS:
        check["signals"] = ["invalid_exec_host"]
        check["stderr"] = "tools.exec.host missing/invalid"
        return check

    agents = cfg.get("agents", {}) if isinstance(cfg.get("agents"), dict) else {}
    defaults = agents.get("defaults", {}) if isinstance(agents.get("defaults"), dict) else {}
    sandbox = defaults.get("sandbox", {}) if isinstance(defaults.get("sandbox"), dict) else {}
    sandbox_mode = str(sandbox.get("mode") or "").strip().lower()
    if host == "sandbox" and sandbox_mode not in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST:
        check["signals"] = ["invalid_sandbox_mode"]
        check["stderr"] = "agents.defaults.sandbox.mode invalid for sandbox host"
        return check

    if host == "sandbox" and not _docker_sandbox_available():
        check["signals"] = ["sandbox_runtime_unavailable"]
        check["stderr"] = "docker daemon unavailable for sandbox host"
        return check

    if host == "node" and not _node_host_available():
        check["signals"] = ["node_host_unavailable"]
        check["stderr"] = (
            "tools.exec.host=node but no paired node found via 'openclaw nodes list'; "
            "run: python scripts/enforce_openclaw_exec_environment.py --host gateway"
        )
        return check

    invalid_override_agents = _invalid_sandbox_override_agents(cfg)
    if invalid_override_agents:
        check["signals"] = ["invalid_sandbox_mode"]
        check["stderr"] = (
            "sandbox disabled by agent override(s): " + ",".join(sorted(invalid_override_agents))
        )
        return check

    acp = cfg.get("acp", {}) if isinstance(cfg.get("acp"), dict) else {}
    default_agent = str(acp.get("defaultAgent") or "").strip()
    if not default_agent:
        check["signals"] = ["missing_acp_default_agent"]
        check["stderr"] = "acp.defaultAgent missing"
        return check

    check["ok"] = True
    check["returncode"] = 0
    check["signals"] = ["exec_env_valid"]
    check["stdout"] = (
        f"host={host} sandbox_mode={sandbox_mode or '<unset>'} "
        f"acp.defaultAgent={default_agent} invalid_agent_overrides=0"
    )
    return check


def collect_runtime_status(*, timeout_seconds: float = 90.0, strict: bool = False) -> dict[str, Any]:
    py = sys.executable
    db_path = PROJECT_ROOT / "data" / "portfolio_maximizer.db"
    whitelist_ids = str(
        os.getenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", "66")
    ).strip() or "66"

    checks: list[dict[str, Any]] = []

    if db_path.exists():
        checks.append(
            _run_check(
                "pnl_integrity",
                [py, "-m", "integrity.pnl_integrity_enforcer", "--db", str(db_path)],
                timeout_seconds=timeout_seconds,
                env_overrides={
                    "INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS": whitelist_ids,
                },
            )
        )
    else:
        checks.append(
            {
                "name": "pnl_integrity",
                "ok": False,
                "returncode": 2,
                "duration_seconds": 0.0,
                "command": f"{py} -m integrity.pnl_integrity_enforcer --db {db_path}",
                "stdout": "",
                "stderr": "Database not found; skipped integrity enforcer.",
            }
        )

    checks.append(_production_gate_artifact_check())
    checks.append(
        _run_check(
            "error_monitor",
            [py, str(PROJECT_ROOT / "scripts" / "error_monitor.py"), "--check"],
            timeout_seconds=timeout_seconds,
        )
    )
    checks.append(_openclaw_exec_environment_check())

    if strict:
        production_gate_check = next(
            (check for check in checks if check.get("name") == "production_gate"),
            {},
        )
        strict_checks = [
            _strict_production_gate_check(production_gate_check),
            _strict_dashboard_payload_check(),
            _strict_canonical_snapshot_check(),
            _strict_persistence_manager_status_check(),
            _collect_observability_stack_check(timeout_seconds=timeout_seconds),
        ]
        strict_checks.append(
            _strict_observability_agreement_check(
                base_checks=checks,
                strict_checks=strict_checks,
            )
        )
        checks.extend(strict_checks)

    automation_ready_detail = _automation_ready_for_thin_linkage_check()
    failed = [c["name"] for c in checks if not bool(c.get("ok"))]
    return {
        "status": "ok" if not failed else "degraded",
        "failed_checks": failed,
        "check_count": len(checks),
        "strict_mode": bool(strict),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "automation_ready_for_thin_linkage": bool(automation_ready_detail.get("ready")),
        "automation_ready_for_thin_linkage_detail": automation_ready_detail,
        "checks": checks,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="PowerShell-safe PMX runtime status snapshot")
    p.add_argument("--timeout-seconds", type=float, default=90.0, help="Per-check timeout")
    p.add_argument("--strict", action="store_true", help="Enforce strict-green observability contract.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = p.parse_args(argv)

    payload = collect_runtime_status(timeout_seconds=float(args.timeout_seconds), strict=bool(args.strict))
    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, separators=(",", ":")))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
