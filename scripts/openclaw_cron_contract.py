from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from utils.repo_python import resolve_repo_python


PROJECT_ROOT = Path(__file__).resolve().parents[1]


AGENT_TURN_PAYLOAD_KIND = "agentTurn"
DEFAULT_SESSION_TARGET = "isolated"
LEGACY_WINDOWS_PYTHON_PATH_RE = re.compile(r"(?i)(?:\.\\)?simpleTrader_env[\\/]+Scripts[\\/]+python\.exe")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_session_target(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _rewrite_legacy_python_paths(value: Any, *, replacement: str) -> tuple[Any, bool]:
    if not replacement:
        return value, False
    if isinstance(value, str):
        new_value, count = LEGACY_WINDOWS_PYTHON_PATH_RE.subn(lambda _match: replacement, value)
        return new_value, count > 0
    if isinstance(value, list):
        changed = False
        items: list[Any] = []
        for item in value:
            new_item, item_changed = _rewrite_legacy_python_paths(item, replacement=replacement)
            changed = changed or item_changed
            items.append(new_item)
        return items, changed
    if isinstance(value, dict):
        changed = False
        out: dict[Any, Any] = {}
        for key, item in value.items():
            new_item, item_changed = _rewrite_legacy_python_paths(item, replacement=replacement)
            changed = changed or item_changed
            out[key] = new_item
        return out, changed
    return value, False


def _contains_legacy_python_path(value: Any) -> bool:
    if isinstance(value, str):
        return bool(LEGACY_WINDOWS_PYTHON_PATH_RE.search(value))
    if isinstance(value, list):
        return any(_contains_legacy_python_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_legacy_python_path(item) for item in value.values())
    return False


def load_cron_jobs_payload(path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    if not path.exists():
        return {}, f"cron jobs file missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {}, f"cron jobs file unreadable: {path} ({exc})"
    if not isinstance(payload, dict):
        return {}, f"cron jobs payload is not an object: {path}"
    return payload, None


def _job_row(job: Any, *, index: Optional[int] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "index": index,
        "id": None,
        "name": None,
        "agentId": None,
        "payload_kind": None,
        "sessionTarget": None,
        "is_agent_turn": False,
        "status": "OK",
        "issues": [],
        "detail": "ok",
    }

    if not isinstance(job, dict):
        row["status"] = "FAIL"
        row["issues"] = ["job_not_object"]
        row["detail"] = "cron job is not an object"
        return row

    row["id"] = _coerce_text(job.get("id")) or None
    row["name"] = _coerce_text(job.get("name")) or None
    row["agentId"] = _coerce_text(job.get("agentId")) or None

    payload = job.get("payload")
    payload_obj = payload if isinstance(payload, dict) else None
    payload_kind = _coerce_text(payload_obj.get("kind")) if payload_obj else ""
    if payload_obj is None:
        if payload is not None:
            row["issues"].append("payload_not_object")
            row["status"] = "WARN"
            row["detail"] = "payload is not an object"
        else:
            row["issues"].append("payload_missing")
            row["status"] = "WARN"
            row["detail"] = "payload missing"

    row["payload_kind"] = payload_kind or None
    is_agent_turn = payload_kind == AGENT_TURN_PAYLOAD_KIND
    row["is_agent_turn"] = is_agent_turn

    session_target_raw = job.get("sessionTarget")
    session_target = _normalize_session_target(session_target_raw)
    if session_target:
        row["sessionTarget"] = session_target

    if is_agent_turn:
        if session_target is None:
            if session_target_raw is None or (isinstance(session_target_raw, str) and not session_target):
                row["issues"].append("missing_sessionTarget")
                row["detail"] = "agentTurn job missing sessionTarget"
            else:
                row["issues"].append("sessionTarget_not_string")
                row["detail"] = "agentTurn job sessionTarget must be a non-empty string"
            row["status"] = "FAIL"

    if row["issues"] and row["status"] == "OK":
        row["status"] = "WARN"
        row["detail"] = ", ".join(row["issues"])
    elif row["issues"] and row["detail"] == "ok":
        row["detail"] = ", ".join(row["issues"])

    return row


def summarize_cron_jobs(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "FAIL",
            "detail": "cron jobs payload is not an object",
            "jobs_total": 0,
            "jobs_valid": 0,
            "jobs_invalid": 0,
            "agent_turn_jobs": 0,
            "agent_turn_invalid": 0,
            "invalid_session_target_count": 0,
            "malformed_job_count": 0,
            "stale_python_path_count": 0,
            "delivery_fallback_ready_count": 0,
            "job_rows": [],
            "invalid_jobs": [],
            "fallback_ready_jobs": [],
        }

    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return {
            "status": "FAIL",
            "detail": "cron jobs payload missing jobs list",
            "jobs_total": 0,
            "jobs_valid": 0,
            "jobs_invalid": 0,
            "agent_turn_jobs": 0,
            "agent_turn_invalid": 0,
            "invalid_session_target_count": 0,
            "malformed_job_count": 0,
            "stale_python_path_count": 0,
            "delivery_fallback_ready_count": 0,
            "job_rows": [],
            "invalid_jobs": [],
            "fallback_ready_jobs": [],
        }

    if not jobs:
        return {
            "status": "WARN",
            "detail": "No cron jobs found",
            "jobs_total": 0,
            "jobs_valid": 0,
            "jobs_invalid": 0,
            "agent_turn_jobs": 0,
            "agent_turn_invalid": 0,
            "invalid_session_target_count": 0,
            "malformed_job_count": 0,
            "stale_python_path_count": 0,
            "delivery_fallback_ready_count": 0,
            "job_rows": [],
            "invalid_jobs": [],
            "fallback_ready_jobs": [],
        }

    rows = [_job_row(job, index=index) for index, job in enumerate(jobs)]
    invalid_jobs = [row for row in rows if row["status"] == "FAIL"]
    malformed_jobs = [row for row in rows if row["issues"]]
    agent_turn_rows = [row for row in rows if row["is_agent_turn"]]
    invalid_session_target = [
        row
        for row in rows
        if row["is_agent_turn"]
        and (
            "missing_sessionTarget" in row["issues"]
            or "sessionTarget_not_string" in row["issues"]
        )
    ]
    stale_python_path_jobs = [
        row for row, job in zip(rows, jobs) if isinstance(job, dict) and _contains_legacy_python_path(job)
    ]
    fallback_ready_jobs: List[Dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            continue
        delivery = job.get("delivery")
        delivery_obj = delivery if isinstance(delivery, dict) else {}
        fallback = delivery_obj.get("fallback") if isinstance(delivery_obj, dict) else {}
        if not isinstance(fallback, dict):
            continue
        fallback_channel = _coerce_text(fallback.get("channel"))
        if not fallback_channel:
            continue
        fallback_ready_jobs.append(
            {
                "index": index,
                "name": _coerce_text(job.get("name")) or None,
                "id": _coerce_text(job.get("id")) or None,
                "delivery_channel": _coerce_text(delivery_obj.get("channel")) or None,
                "fallback_channel": fallback_channel,
            }
        )

    status = "FAIL" if invalid_jobs else ("WARN" if malformed_jobs else "OK")
    detail = (
        f"{len(rows)} jobs, {len(agent_turn_rows)} agentTurn, "
        f"{len(invalid_jobs)} malformed, {len(fallback_ready_jobs)} fallback-ready"
    )
    return {
        "status": status,
        "detail": detail,
        "jobs_total": len(rows),
        "jobs_valid": len(rows) - len(invalid_jobs),
        "jobs_invalid": len(invalid_jobs),
        "agent_turn_jobs": len(agent_turn_rows),
        "agent_turn_invalid": len(invalid_session_target),
        "invalid_session_target_count": len(invalid_session_target),
        "malformed_job_count": len(malformed_jobs),
        "stale_python_path_count": len(stale_python_path_jobs),
        "delivery_fallback_ready_count": len(fallback_ready_jobs),
        "job_rows": rows,
        "invalid_jobs": invalid_jobs,
        "fallback_ready_jobs": fallback_ready_jobs,
    }


def sanitize_cron_jobs_payload(
    payload: Any,
    *,
    default_session_target: str = DEFAULT_SESSION_TARGET,
    python_executable: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sanitized = copy.deepcopy(payload) if isinstance(payload, dict) else {}
    jobs = sanitized.get("jobs") if isinstance(sanitized.get("jobs"), list) else []
    sanitized_jobs: List[Dict[str, Any]] = []
    backfilled_jobs: List[Dict[str, Any]] = []
    quarantined_jobs: List[Dict[str, Any]] = []
    rewritten_jobs: List[Dict[str, Any]] = []

    resolved_python = str(python_executable or "").strip()
    if not resolved_python:
        try:
            resolved_python = resolve_repo_python(PROJECT_ROOT)
        except Exception:
            resolved_python = ""

    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            quarantined_jobs.append(
                {
                    "index": index,
                    "reason": "job_not_object",
                    "job_preview": _coerce_text(job)[:200],
                }
            )
            continue

        payload_obj = job.get("payload")
        if payload_obj is not None and not isinstance(payload_obj, dict):
            quarantined_jobs.append(
                {
                    "index": index,
                    "name": _coerce_text(job.get("name")) or None,
                    "id": _coerce_text(job.get("id")) or None,
                    "reason": "payload_not_object",
                }
            )
            continue

        payload_kind = _coerce_text(payload_obj.get("kind")) if isinstance(payload_obj, dict) else ""
        if payload_kind == AGENT_TURN_PAYLOAD_KIND:
            session_target = _normalize_session_target(job.get("sessionTarget"))
            if not session_target:
                job = copy.deepcopy(job)
                job["sessionTarget"] = default_session_target
                backfilled_jobs.append(
                    {
                        "index": index,
                        "name": _coerce_text(job.get("name")) or None,
                        "id": _coerce_text(job.get("id")) or None,
                        "sessionTarget": default_session_target,
                        "reason": "missing_or_invalid_sessionTarget_backfilled",
                    }
                )
        if resolved_python:
            rewritten_job, changed = _rewrite_legacy_python_paths(job, replacement=resolved_python)
            if changed:
                job = rewritten_job if isinstance(rewritten_job, dict) else job
                rewritten_jobs.append(
                    {
                        "index": index,
                        "name": _coerce_text(job.get("name")) or None,
                        "id": _coerce_text(job.get("id")) or None,
                        "replacement": resolved_python,
                    }
                )
        sanitized_jobs.append(job)

    sanitized["jobs"] = sanitized_jobs
    report = {
        "changed": bool(backfilled_jobs or quarantined_jobs or rewritten_jobs or len(sanitized_jobs) != len(jobs)),
        "default_session_target": default_session_target,
        "backfilled_count": len(backfilled_jobs),
        "quarantined_count": len(quarantined_jobs),
        "backfilled_jobs": backfilled_jobs,
        "quarantined_jobs": quarantined_jobs,
        "rewritten_count": len(rewritten_jobs),
        "rewritten_jobs": rewritten_jobs,
    }
    return sanitized, report
