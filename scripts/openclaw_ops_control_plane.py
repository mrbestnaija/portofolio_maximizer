#!/usr/bin/env python3
"""
Local-only OpenClaw ops control plane for safe production recovery.

This sidecar keeps service recovery narrow on purpose. It is allowed to:
- inspect OpenClaw, runtime, dashboard, and watcher health
- restart/recover the gateway, dashboard bridge/server, or live watcher
- persist one authoritative machine-readable ops verdict
- send concise anomaly/recovery notifications through OpenClaw

It is not allowed to:
- mutate strategy/risk/config settings
- perform reconciliation or data repair
- change repo files from chat-driven requests
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from etl.secret_loader import bootstrap_dotenv

    bootstrap_dotenv()
except Exception:
    pass

from scripts import openclaw_production_readiness as readiness_mod
from scripts import project_runtime_status as runtime_mod
from scripts import windows_dashboard_manager as dashboard_mod
from scripts import windows_persistence_manager as persistence_mod
from utils.openclaw_cli import resolve_openclaw_targets, send_message_multi


ALLOWED_RECOVERY_TARGETS = ("gateway", "dashboard", "watcher")
ISSUE_CLASSES = (
    "recoverable_service_failure",
    "human_security_action_required",
    "human_governance_action_required",
    "human_economic_action_required",
)
DEFAULT_GATE_ARTIFACT = PROJECT_ROOT / "logs" / "gate_status_latest.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_VERDICT_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_ops_control_plane_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_ops_control_plane_state.json"
DEFAULT_MAINTENANCE_REPORT_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_maintenance_latest.json"
DEFAULT_MAINTENANCE_STATE_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_maintenance_state.json"
DEFAULT_WATCHER_JSON = PROJECT_ROOT / "logs" / "overnight_denominator" / "live_denominator_latest.json"
DEFAULT_NOTIFY_COOLDOWN_SECONDS = 900
FALSEY_ENV_VALUES = {"0", "false", "no", "off"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw not in FALSEY_ENV_VALUES


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(minimum), int(value))


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fingerprint_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _run_command(cmd: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(5.0, float(timeout_seconds)),
            check=False,
        )
        return {
            "ok": int(proc.returncode) == 0,
            "returncode": int(proc.returncode),
            "command": list(cmd),
            "stdout_tail": [str(line) for line in (proc.stdout or "").splitlines()[-12:]],
            "stderr_tail": [str(line) for line in (proc.stderr or "").splitlines()[-12:]],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": 124,
            "command": list(cmd),
            "stdout_tail": [str(line) for line in ((exc.stdout or "") if isinstance(exc.stdout, str) else "").splitlines()[-12:]],
            "stderr_tail": [str(line) for line in ((((exc.stderr or "") if isinstance(exc.stderr, str) else "") or "timeout")).splitlines()[-12:]],
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": 127,
            "command": list(cmd),
            "stdout_tail": [],
            "stderr_tail": [str(exc)],
        }


def _tail_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [str(line) for line in str(text or "").splitlines() if str(line).strip()]
    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def _call_quietly(fn, /, *args, **kwargs) -> tuple[Any, list[str], list[str]]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        result = fn(*args, **kwargs)
    return result, _tail_lines(stdout_buf.getvalue()), _tail_lines(stderr_buf.getvalue())


def _read_pidfile_non_mutating(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _coalesce_bool(*values: Any) -> Optional[bool]:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _coalesce_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _issue(
    issue_class: str,
    *,
    code: str,
    detail: str,
    component: str,
    severity: str = "error",
    target: str = "",
) -> dict[str, Any]:
    return {
        "class": issue_class,
        "code": str(code or "").strip() or "unknown",
        "detail": str(detail or "").strip() or "unspecified",
        "component": str(component or "").strip() or "unknown",
        "severity": str(severity or "").strip() or "error",
        "target": str(target or "").strip(),
    }


def _dedupe_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("class") or "").strip(),
            str(row.get("code") or "").strip(),
            str(row.get("detail") or "").strip(),
            str(row.get("component") or "").strip(),
            str(row.get("target") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "class": key[0],
                "code": key[1],
                "detail": key[2],
                "component": key[3],
                "severity": str(row.get("severity") or "error"),
                "target": key[4],
            }
        )
    return out


def _group_issues(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {name: [] for name in ISSUE_CLASSES}
    for row in _dedupe_issues(rows):
        label = str(row.get("class") or "").strip()
        if label not in grouped:
            continue
        grouped[label].append(row)
    return grouped


def _flatten_issue_classes(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in ISSUE_CLASSES:
        rows = grouped.get(label) if isinstance(grouped.get(label), list) else []
        out.extend([row for row in rows if isinstance(row, dict)])
    return out


def _normalize_requested_targets(raw: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = [raw]
    for value in values:
        for token in str(value or "").replace(";", ",").split(","):
            text = token.strip().lower()
            if text:
                parts.append(text)
    deduped: list[str] = []
    for item in parts:
        if item not in deduped:
            deduped.append(item)
    invalid = [item for item in deduped if item not in ALLOWED_RECOVERY_TARGETS]
    if invalid:
        raise ValueError(f"unsupported recovery targets: {', '.join(invalid)}")
    return deduped


def extract_ops_intent_request(text: str) -> dict[str, Any]:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return {
            "recognized": False,
            "command": "status",
            "targets": [],
            "include_runtime": False,
            "detail": "empty_request",
        }

    if "heal gateway" in normalized or "restart gateway" in normalized:
        return {"recognized": True, "command": "recover", "targets": ["gateway"], "include_runtime": False, "detail": "heal_gateway"}
    if "heal dashboard" in normalized or "restart dashboard" in normalized:
        return {"recognized": True, "command": "recover", "targets": ["dashboard"], "include_runtime": False, "detail": "heal_dashboard"}
    if "heal watcher" in normalized or "restart watcher" in normalized or "heal denominator" in normalized:
        return {"recognized": True, "command": "recover", "targets": ["watcher"], "include_runtime": False, "detail": "heal_watcher"}
    if "show runtime" in normalized:
        return {"recognized": True, "command": "status", "targets": [], "include_runtime": True, "detail": "show_runtime"}
    if "show readiness" in normalized:
        return {"recognized": True, "command": "status", "targets": [], "include_runtime": False, "detail": "show_readiness"}
    if "what is broken" in normalized or "what broke" in normalized or "top blockers" in normalized:
        return {"recognized": True, "command": "status", "targets": [], "include_runtime": False, "detail": "what_is_broken"}

    return {
        "recognized": False,
        "command": "status",
        "targets": [],
        "include_runtime": False,
        "detail": "unsupported_ops_intent",
    }


def _find_runtime_check(payload: Optional[dict[str, Any]], name: str) -> dict[str, Any]:
    checks = payload.get("checks") if isinstance(payload, dict) and isinstance(payload.get("checks"), list) else []
    for row in checks:
        if isinstance(row, dict) and str(row.get("name") or "").strip() == name:
            return row
    return {}


def _collect_dashboard_components(
    *,
    root: Path,
    port: int,
    watcher_json_path: Path,
) -> dict[str, Any]:
    logs_dir = root / "logs"
    bridge_pid = _read_pidfile_non_mutating(logs_dir / "dashboard_bridge.pid")
    server_pid = _read_pidfile_non_mutating(logs_dir / f"dashboard_http_{port}.pid")
    watcher_pid = _read_pidfile_non_mutating(logs_dir / "live_denominator.pid")

    bridge_running = bool(bridge_pid and dashboard_mod._pid_alive(bridge_pid))
    server_running = bool(server_pid and dashboard_mod._pid_alive(server_pid))
    watcher_running = bool(watcher_pid and dashboard_mod._pid_alive(watcher_pid))
    port_open = bool(server_running and dashboard_mod._port_open(dashboard_mod.LOCALHOST_BIND, int(port)))

    watcher_payload = _safe_read_json(watcher_json_path)
    watcher_summary = (
        persistence_mod._extract_watcher_status(watcher_payload)
        if watcher_payload
        else {
            "run_id": "",
            "started_utc": None,
            "cycles_configured": 0,
            "cycles_completed": 0,
            "weekdays_only": True,
            "fresh_trade_rows": 0,
            "fresh_trade_context_rows_raw": 0,
            "fresh_trade_exclusions": {},
            "fresh_trade_diagnostics": {},
            "fresh_linkage_included": 0,
            "fresh_production_valid_matched": 0,
            "progress_triggered": False,
            "latest_cycle_completed_utc": None,
        }
    )
    startup_registration = persistence_mod._query_startup_registration()

    dashboard_component = {
        "healthy": bool(bridge_running and server_running and port_open),
        "bridge_pid": bridge_pid,
        "bridge_running": bridge_running,
        "server_pid": server_pid,
        "server_running": server_running,
        "port": int(port),
        "port_open": port_open,
        "url": f"http://{dashboard_mod.LOCALHOST_BIND}:{int(port)}/visualizations/live_dashboard.html",
    }
    watcher_component = {
        "healthy": watcher_running,
        "pid": watcher_pid,
        "running": watcher_running,
        **watcher_summary,
    }
    return {
        "dashboard": dashboard_component,
        "watcher": watcher_component,
        "startup_registration": startup_registration,
    }


def _collect_primary_channel_components(
    *,
    readiness_payload: dict[str, Any],
    maintenance_report_path: Path,
    maintenance_state_path: Path,
    primary_channel: str,
) -> dict[str, Any]:
    maintenance_report = _safe_read_json(maintenance_report_path)
    maintenance_state = _safe_read_json(maintenance_state_path)

    regression = readiness_payload.get("openclaw_regression") if isinstance(readiness_payload.get("openclaw_regression"), dict) else {}
    regression_status = str(regression.get("status") or "").upper()
    regression_checks = regression.get("checks") if isinstance(regression.get("checks"), dict) else {}
    primary_check = regression_checks.get("primary_channel") if isinstance(regression_checks.get("primary_channel"), dict) else {}
    primary_reason = str(primary_check.get("reason") or "")

    steps = maintenance_report.get("steps") if isinstance(maintenance_report.get("steps"), dict) else {}
    fast_supervisor = steps.get("fast_supervisor") if isinstance(steps.get("fast_supervisor"), dict) else {}
    fast_snapshot = fast_supervisor.get("snapshot") if isinstance(fast_supervisor.get("snapshot"), dict) else {}
    gateway_health = steps.get("gateway_health") if isinstance(steps.get("gateway_health"), dict) else {}

    configured = _coalesce_bool(fast_snapshot.get("configured"), regression_status == "PASS")
    linked = _coalesce_bool(
        fast_snapshot.get("linked"),
        False if primary_reason == "whatsapp_not_linked" else (True if regression_status == "PASS" and primary_channel == "whatsapp" else None),
    )
    running = _coalesce_bool(
        fast_snapshot.get("running"),
        False if primary_reason in {"channel_not_running", "enabled_account_not_running"} else (True if regression_status == "PASS" else None),
    )
    connected = _coalesce_bool(
        fast_snapshot.get("connected"),
        False if primary_reason == "channel_not_connected" else (True if regression_status == "PASS" else None),
    )
    relink_required = bool(linked is False or fast_snapshot.get("disconnect_logged_out"))
    primary_issue = _coalesce_text(
        gateway_health.get("primary_channel_issue_final"),
        gateway_health.get("primary_channel_issue_after_restart"),
        gateway_health.get("primary_channel_issue"),
        fast_supervisor.get("primary_issue"),
        primary_reason,
    )

    gateway_rpc_ok = gateway_health.get("rpc_ok")
    if not isinstance(gateway_rpc_ok, bool):
        gateway_rpc_ok = None

    gateway_component = {
        "healthy": bool(
            (gateway_rpc_ok is True or gateway_rpc_ok is None)
            and linked is not False
            and running is not False
            and connected is not False
            and regression_status in {"PASS", "SKIP"}
        ),
        "status": "healthy" if regression_status in {"PASS", "SKIP"} and gateway_rpc_ok is not False else "degraded",
        "rpc_ok": gateway_rpc_ok,
        "service_status": str(gateway_health.get("service_status") or ""),
        "service_state": str(gateway_health.get("service_state") or ""),
        "listener_pid": gateway_health.get("gateway_listener_pid"),
        "primary_issue": primary_issue,
        "actions": list(gateway_health.get("actions") or []),
        "manual_actions": list(gateway_health.get("manual_actions") or []),
        "last_gateway_restart_at": maintenance_state.get("last_gateway_restart_at"),
        "report_status": str(maintenance_report.get("status") or ""),
    }
    primary_channel_component = {
        "channel": primary_channel,
        "healthy": bool(linked is not False and running is not False and connected is not False and not relink_required),
        "configured": configured,
        "linked": linked,
        "running": running,
        "connected": connected,
        "relink_required": relink_required,
        "issue_reason": primary_issue,
    }
    maintenance_component = {
        "report_path": str(maintenance_report_path),
        "state_path": str(maintenance_state_path),
        "report_exists": maintenance_report_path.exists(),
        "state_exists": maintenance_state_path.exists(),
        "status": str(maintenance_report.get("status") or ""),
        "timestamp_utc": maintenance_report.get("timestamp_utc"),
        "watch_cycle": maintenance_report.get("watch_cycle"),
        "last_fast_supervisor_restart_at": maintenance_state.get("last_fast_supervisor_restart_at"),
        "last_gateway_restart_at": maintenance_state.get("last_gateway_restart_at"),
    }
    return {
        "gateway": gateway_component,
        "primary_channel": primary_channel_component,
        "maintenance": maintenance_component,
    }


def collect_ops_snapshot(
    *,
    gate_artifact_path: Path,
    db_path: Path,
    primary_channel: str,
    dashboard_port: int,
    include_runtime: bool,
    timeout_seconds: float,
    maintenance_report_path: Path,
    maintenance_state_path: Path,
    watcher_json_path: Path,
) -> dict[str, Any]:
    readiness_payload, readiness_stdout, readiness_stderr = _call_quietly(
        readiness_mod.collect_openclaw_production_readiness,
        gate_artifact_path=gate_artifact_path,
        fresh_runtime=False,
        timeout_seconds=float(timeout_seconds),
    )
    if include_runtime:
        runtime_payload, runtime_stdout, runtime_stderr = _call_quietly(
            runtime_mod.collect_runtime_status,
            timeout_seconds=max(5.0, float(timeout_seconds)),
        )
    else:
        runtime_payload, runtime_stdout, runtime_stderr = None, [], []
    dashboard_components = _collect_dashboard_components(
        root=PROJECT_ROOT,
        port=int(dashboard_port),
        watcher_json_path=watcher_json_path,
    )
    primary_components = _collect_primary_channel_components(
        readiness_payload=readiness_payload,
        maintenance_report_path=maintenance_report_path,
        maintenance_state_path=maintenance_state_path,
        primary_channel=primary_channel,
    )
    gate_artifact = readiness_payload.get("gate_artifact") if isinstance(readiness_payload.get("gate_artifact"), dict) else {}
    runtime_exec_env = _find_runtime_check(runtime_payload, "openclaw_exec_env") if runtime_payload else {}

    components = {
        **primary_components,
        **dashboard_components,
        "runtime": {
            "included": bool(runtime_payload),
            "status": str(runtime_payload.get("status") or "") if isinstance(runtime_payload, dict) else "",
            "failed_checks": list(runtime_payload.get("failed_checks") or []) if isinstance(runtime_payload, dict) else [],
            "check_count": int(runtime_payload.get("check_count") or 0) if isinstance(runtime_payload, dict) else 0,
            "openclaw_exec_env_ok": bool(runtime_exec_env.get("ok")) if runtime_exec_env else bool(
                (readiness_payload.get("openclaw_exec_env") or {}).get("ok")
            ),
        },
        "readiness": {
            "readiness_status": str(readiness_payload.get("readiness_status") or ""),
            "ready_now": bool(readiness_payload.get("ready_now")),
            "blocker_count": int(((readiness_payload.get("summary") or {}) if isinstance(readiness_payload.get("summary"), dict) else {}).get("blocker_count") or 0),
            "warning_count": int(((readiness_payload.get("summary") or {}) if isinstance(readiness_payload.get("summary"), dict) else {}).get("warning_count") or 0),
            "overall_passed": gate_artifact.get("overall_passed"),
            "phase3_ready": gate_artifact.get("phase3_ready"),
            "phase3_reason": str(gate_artifact.get("phase3_reason") or ""),
            "skipped_gate_labels": list(gate_artifact.get("skipped_gate_labels") or []),
        },
    }
    return {
        "timestamp_utc": _utc_now_iso(),
        "components": components,
        "readiness_payload": readiness_payload,
        "runtime_payload": runtime_payload,
        "captured_output": {
            "readiness_stdout_tail": readiness_stdout,
            "readiness_stderr_tail": readiness_stderr,
            "runtime_stdout_tail": runtime_stdout,
            "runtime_stderr_tail": runtime_stderr,
        },
    }


def _map_readiness_issue(row: dict[str, Any], *, severity: str) -> Optional[dict[str, Any]]:
    source = str(row.get("source") or "").strip()
    code = str(row.get("code") or "").strip()
    detail = str(row.get("detail") or "").strip() or "unspecified"

    if not source and not code and not detail:
        return None
    if source == "capital_readiness":
        return _issue(
            "human_economic_action_required",
            code=code or "capital_readiness_issue",
            detail=detail,
            component="capital_readiness",
            severity=severity,
        )
    if source in {"security", "openclaw_exec_env", "openclaw_models"}:
        return _issue(
            "human_security_action_required",
            code=code or "security_issue",
            detail=detail,
            component=source or "security",
            severity=severity,
        )
    if source == "openclaw_regression":
        if "whatsapp_not_linked" in detail or "logged_out" in detail or "relink" in detail:
            return _issue(
                "human_security_action_required",
                code=code or "primary_channel_relink_required",
                detail=detail,
                component="gateway",
                severity=severity,
            )
        return _issue(
            "recoverable_service_failure",
            code=code or "gateway_recovery_required",
            detail=detail,
            component="gateway",
            severity=severity,
            target="gateway",
        )
    if source == "runtime_status":
        if "openclaw_exec_env" in detail.lower():
            return _issue(
                "human_security_action_required",
                code=code or "runtime_exec_env_issue",
                detail=detail,
                component="runtime",
                severity=severity,
            )
        return _issue(
            "human_governance_action_required",
            code=code or "runtime_status_issue",
            detail=detail,
            component="runtime",
            severity=severity,
        )
    return _issue(
        "human_governance_action_required",
        code=code or "governance_issue",
        detail=detail,
        component=source or "readiness",
        severity=severity,
    )


def classify_ops_issues(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    components = snapshot.get("components") if isinstance(snapshot.get("components"), dict) else {}
    readiness = components.get("readiness") if isinstance(components.get("readiness"), dict) else {}
    gateway = components.get("gateway") if isinstance(components.get("gateway"), dict) else {}
    primary_channel = components.get("primary_channel") if isinstance(components.get("primary_channel"), dict) else {}
    dashboard = components.get("dashboard") if isinstance(components.get("dashboard"), dict) else {}
    watcher = components.get("watcher") if isinstance(components.get("watcher"), dict) else {}
    runtime_component = components.get("runtime") if isinstance(components.get("runtime"), dict) else {}
    readiness_payload = snapshot.get("readiness_payload") if isinstance(snapshot.get("readiness_payload"), dict) else {}

    rows: list[dict[str, Any]] = []

    if bool(primary_channel.get("relink_required")):
        rows.append(
            _issue(
                "human_security_action_required",
                code="primary_channel_relink_required",
                detail=_coalesce_text(primary_channel.get("issue_reason"), "Primary WhatsApp channel requires relink."),
                component="primary_channel",
            )
        )
    elif gateway.get("rpc_ok") is False:
        rows.append(
            _issue(
                "recoverable_service_failure",
                code="gateway_rpc_unhealthy",
                detail=_coalesce_text(gateway.get("primary_issue"), "Gateway RPC is unhealthy."),
                component="gateway",
                target="gateway",
            )
        )
    elif primary_channel.get("linked") is True and primary_channel.get("running") is False:
        rows.append(
            _issue(
                "recoverable_service_failure",
                code="primary_channel_not_running",
                detail=_coalesce_text(primary_channel.get("issue_reason"), "Primary channel is not running."),
                component="primary_channel",
                target="gateway",
            )
        )
    elif primary_channel.get("linked") is True and primary_channel.get("connected") is False:
        rows.append(
            _issue(
                "recoverable_service_failure",
                code="primary_channel_not_connected",
                detail=_coalesce_text(primary_channel.get("issue_reason"), "Primary channel is not connected."),
                component="primary_channel",
                target="gateway",
            )
        )

    if dashboard and not bool(dashboard.get("healthy")):
        rows.append(
            _issue(
                "recoverable_service_failure",
                code="dashboard_stack_not_running",
                detail="Dashboard bridge or local HTTP server is not healthy.",
                component="dashboard",
                target="dashboard",
            )
        )
    if watcher and not bool(watcher.get("healthy")):
        rows.append(
            _issue(
                "recoverable_service_failure",
                code="watcher_not_running",
                detail="Live denominator watcher is not running.",
                component="watcher",
                target="watcher",
            )
        )

    if readiness.get("overall_passed") is False:
        skipped = readiness.get("skipped_gate_labels") if isinstance(readiness.get("skipped_gate_labels"), list) else []
        detail = "overall_passed=false"
        if readiness.get("phase3_ready") is True:
            detail = "overall_passed=false even though phase3_ready=true"
        if skipped:
            detail = f"{detail}; skipped_gate_labels={', '.join(str(x) for x in skipped)}"
        rows.append(
            _issue(
                "human_governance_action_required",
                code="overall_passed_false",
                detail=detail,
                component="readiness",
            )
        )

    if readiness.get("ready_now") is False and readiness.get("readiness_status") == "WARN":
        rows.append(
            _issue(
                "human_governance_action_required",
                code="readiness_warn",
                detail="Production readiness is not PASS.",
                component="readiness",
                severity="warning",
            )
        )

    blockers = readiness_payload.get("blockers") if isinstance(readiness_payload.get("blockers"), list) else []
    warnings = readiness_payload.get("warnings") if isinstance(readiness_payload.get("warnings"), list) else []
    for row in blockers:
        if isinstance(row, dict):
            mapped = _map_readiness_issue(row, severity="error")
            if mapped:
                rows.append(mapped)
    for row in warnings:
        if isinstance(row, dict):
            mapped = _map_readiness_issue(row, severity="warning")
            if mapped:
                rows.append(mapped)

    if runtime_component.get("included") and not bool(runtime_component.get("openclaw_exec_env_ok")):
        rows.append(
            _issue(
                "human_security_action_required",
                code="runtime_exec_env_invalid",
                detail="Runtime snapshot reports an invalid OpenClaw exec environment.",
                component="runtime",
            )
        )

    return _group_issues(rows)


def _service_issue_targets(issue_classes: dict[str, list[dict[str, Any]]]) -> list[str]:
    rows = issue_classes.get("recoverable_service_failure") if isinstance(issue_classes.get("recoverable_service_failure"), list) else []
    targets: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        target = str(row.get("target") or "").strip().lower()
        if target and target in ALLOWED_RECOVERY_TARGETS and target not in targets:
            targets.append(target)
    return targets


def _recover_gateway(
    *,
    primary_channel: str,
    timeout_seconds: float,
    maintenance_report_path: Path,
) -> dict[str, Any]:
    started_at = _utc_now_iso()
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "openclaw_maintenance.py"),
        "--apply",
        "--strict",
        "--primary-channel",
        str(primary_channel),
        "--restart-gateway-on-rpc-failure",
        "--report-file",
        str(maintenance_report_path),
    ]
    result = _run_command(cmd, timeout_seconds=max(20.0, float(timeout_seconds)))
    return {
        "target": "gateway",
        "ok": bool(result.get("ok")),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "summary": "Attempted gateway maintenance recovery.",
        **result,
    }


def _recover_dashboard_like(
    *,
    ensure_live_watcher: bool,
    dashboard_port: int,
    db_path: Path,
    watcher_tickers: str,
    watcher_cycles: int,
    watcher_sleep_seconds: int,
) -> dict[str, Any]:
    label = "watcher" if ensure_live_watcher else "dashboard"
    started_at = _utc_now_iso()
    result = dashboard_mod._ensure_dashboard_stack(
        root=PROJECT_ROOT,
        python_bin=sys.executable,
        port=int(dashboard_port),
        prometheus_port=int(dashboard_mod.DEFAULT_PROMETHEUS_EXPORTER_PORT),
        db_path=db_path,
        persist_snapshot=True,
        require_bridge=True,
        ensure_prometheus_exporter=True,
        ensure_live_watcher=bool(ensure_live_watcher),
        watcher_tickers=str(watcher_tickers),
        watcher_cycles=int(watcher_cycles),
        watcher_sleep_seconds=int(watcher_sleep_seconds),
    )
    ok = bool(
        result.bridge_running
        and result.server_running
        and result.exporter_running
        and (result.watcher_running or not ensure_live_watcher)
    )
    return {
        "target": label,
        "ok": ok,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "summary": "Attempted watcher recovery." if ensure_live_watcher else "Attempted dashboard recovery.",
        "command": ["python", "scripts/windows_dashboard_manager.py", "ensure"],
        "returncode": 0 if ok else 1,
        "stdout_tail": [],
        "stderr_tail": list(result.warnings),
        "started_bridge": bool(result.started_bridge),
        "started_server": bool(result.started_server),
        "started_exporter": bool(result.started_exporter),
        "started_watcher": bool(result.started_watcher),
    }


def _run_recovery_targets(
    *,
    targets: list[str],
    primary_channel: str,
    timeout_seconds: float,
    maintenance_report_path: Path,
    dashboard_port: int,
    db_path: Path,
    watcher_tickers: str,
    watcher_cycles: int,
    watcher_sleep_seconds: int,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for target in targets:
        if target == "gateway":
            actions.append(
                _recover_gateway(
                    primary_channel=primary_channel,
                    timeout_seconds=timeout_seconds,
                    maintenance_report_path=maintenance_report_path,
                )
            )
        elif target == "dashboard":
            actions.append(
                _recover_dashboard_like(
                    ensure_live_watcher=False,
                    dashboard_port=dashboard_port,
                    db_path=db_path,
                    watcher_tickers=watcher_tickers,
                    watcher_cycles=watcher_cycles,
                    watcher_sleep_seconds=watcher_sleep_seconds,
                )
            )
        elif target == "watcher":
            actions.append(
                _recover_dashboard_like(
                    ensure_live_watcher=True,
                    dashboard_port=dashboard_port,
                    db_path=db_path,
                    watcher_tickers=watcher_tickers,
                    watcher_cycles=watcher_cycles,
                    watcher_sleep_seconds=watcher_sleep_seconds,
                )
            )
    return actions


def _completed_actions_from_targets(
    *,
    initial_issue_classes: dict[str, list[dict[str, Any]]],
    final_issue_classes: dict[str, list[dict[str, Any]]],
    attempted_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    initial_targets = set(_service_issue_targets(initial_issue_classes))
    final_targets = set(_service_issue_targets(final_issue_classes))
    out: list[dict[str, Any]] = []
    for row in attempted_actions:
        if not isinstance(row, dict):
            continue
        target = str(row.get("target") or "").strip().lower()
        if bool(row.get("ok")) and target in initial_targets and target not in final_targets:
            out.append(row)
    return out


def build_operator_summary(
    *,
    initial_issue_classes: dict[str, list[dict[str, Any]]],
    final_issue_classes: dict[str, list[dict[str, Any]]],
    attempted_actions: list[dict[str, Any]],
    completed_actions: list[dict[str, Any]],
    components: dict[str, Any],
    overall_status: dict[str, Any],
) -> dict[str, Any]:
    initial_rows = _flatten_issue_classes(initial_issue_classes)
    final_rows = _flatten_issue_classes(final_issue_classes)
    human_required = [
        row
        for row in final_rows
        if str(row.get("class") or "") != "recoverable_service_failure"
    ]
    service_healthy = str(overall_status.get("service_health") or "") == "healthy"
    production_ready = str(overall_status.get("production_status") or "") == "ready"

    if service_healthy and production_ready:
        headline = "Services healthy and production ready."
    elif service_healthy:
        headline = "Services healthy; production is still blocked."
    else:
        headline = "Service degradation detected; production is not ready."

    what_broke = [str(row.get("detail") or "") for row in initial_rows[:5] if str(row.get("detail") or "").strip()]
    auto_recovered = [str(row.get("summary") or "") for row in completed_actions[:5] if str(row.get("summary") or "").strip()]
    human_lines = [str(row.get("detail") or "") for row in human_required[:5] if str(row.get("detail") or "").strip()]

    message_lines = [
        f"OpenClaw ops: {headline}",
        f"What broke: {', '.join(what_broke) if what_broke else 'none'}",
        f"Auto-recovered: {', '.join(auto_recovered) if auto_recovered else 'none'}",
        f"Needs human action: {', '.join(human_lines) if human_lines else 'none'}",
        f"Service health={overall_status.get('service_health')} production_status={overall_status.get('production_status')}",
    ]
    if attempted_actions and not completed_actions:
        message_lines.append("Recovery attempted, but at least one service issue remains unresolved.")

    return {
        "headline": headline,
        "what_broke": what_broke,
        "auto_recovered": auto_recovered,
        "human_action_required": human_lines,
        "message": "\n".join(message_lines),
        "service_healthy": service_healthy,
        "production_ready": production_ready,
        "dashboard_url": ((components.get("dashboard") or {}) if isinstance(components.get("dashboard"), dict) else {}).get("url"),
    }


def _build_overall_status(
    *,
    issue_classes: dict[str, list[dict[str, Any]]],
    components: dict[str, Any],
) -> dict[str, Any]:
    service_issues = issue_classes.get("recoverable_service_failure") if isinstance(issue_classes.get("recoverable_service_failure"), list) else []
    human_rows = [
        row
        for label in ISSUE_CLASSES
        if label != "recoverable_service_failure"
        for row in (issue_classes.get(label) if isinstance(issue_classes.get(label), list) else [])
        if isinstance(row, dict)
    ]
    readiness = components.get("readiness") if isinstance(components.get("readiness"), dict) else {}

    service_health = "healthy" if not service_issues else "degraded"
    production_ready = bool(
        readiness.get("ready_now")
        and readiness.get("overall_passed") is not False
        and not human_rows
    )
    production_status = "ready" if production_ready else "blocked"
    status = "FAIL" if service_health == "degraded" else ("PASS" if production_ready else "WARN")
    return {
        "status": status,
        "service_health": service_health,
        "production_status": production_status,
        "service_issue_count": len(service_issues),
        "human_action_count": len(human_rows),
    }


def _read_notification_state(path: Path) -> dict[str, Any]:
    state = _safe_read_json(path)
    if not state:
        return {"version": 1}
    return state


def _build_cooldowns(
    *,
    state: dict[str, Any],
    cooldown_seconds: int,
    maintenance_component: dict[str, Any],
) -> dict[str, Any]:
    return {
        "notification": {
            "cooldown_seconds": int(cooldown_seconds),
            "last_notified_at_utc": state.get("last_notified_at_utc"),
            "next_notify_at_utc": state.get("next_notify_at_utc"),
            "last_notification_reason": state.get("last_notification_reason"),
        },
        "maintenance": {
            "last_fast_supervisor_restart_at": maintenance_component.get("last_fast_supervisor_restart_at"),
            "last_gateway_restart_at": maintenance_component.get("last_gateway_restart_at"),
        },
    }


def _decide_notification(
    *,
    final_issue_classes: dict[str, list[dict[str, Any]]],
    operator_summary: dict[str, Any],
    state: dict[str, Any],
    cooldown_seconds: int,
) -> dict[str, Any]:
    now = time.time()
    final_rows = _flatten_issue_classes(final_issue_classes)
    unresolved = [row for row in final_rows if isinstance(row, dict)]
    status_key = "issue" if unresolved else "clear"
    signature_source = [{"class": row.get("class"), "code": row.get("code"), "target": row.get("target")} for row in unresolved]
    signature = _fingerprint_payload(signature_source or ["clear"])

    previous_status = str(state.get("last_status") or "")
    previous_signature = str(state.get("last_issue_signature") or "")
    next_notify_at = _parse_ts(state.get("next_notify_at_utc"))
    due = next_notify_at is None or _utc_now() >= next_notify_at

    if status_key == "clear":
        should_notify = previous_status == "issue"
        reason = "recovered" if should_notify else "routine_success"
    elif previous_status != "issue" or previous_signature != signature:
        should_notify = True
        reason = "anomaly"
    elif due:
        should_notify = True
        reason = "repeat_unresolved"
    else:
        should_notify = False
        reason = "cooldown_active"

    if should_notify:
        next_allowed = datetime.fromtimestamp(now + int(cooldown_seconds), tz=timezone.utc)
    elif reason == "cooldown_active" and next_notify_at is not None:
        next_allowed = next_notify_at
    else:
        next_allowed = _utc_now()
    return {
        "should_notify": should_notify,
        "reason": reason,
        "status_key": status_key,
        "issue_signature": signature,
        "message": str(operator_summary.get("message") or "").strip(),
        "next_notify_at_utc": next_allowed.isoformat(),
    }


def _send_notification(
    *,
    decision: dict[str, Any],
    notify_targets: str,
    notify_to: str,
    primary_channel: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not bool(decision.get("should_notify")):
        return {
            "attempted": False,
            "sent": False,
            "reason": decision.get("reason"),
            "results": [],
        }

    targets = resolve_openclaw_targets(
        env_targets=notify_targets,
        env_to=notify_to,
        cfg_to=None,
        default_channel=primary_channel,
    )
    if not targets:
        return {
            "attempted": False,
            "sent": False,
            "reason": "no_notify_targets",
            "results": [],
        }

    results = send_message_multi(
        targets=targets,
        message=str(decision.get("message") or ""),
        command=str(os.getenv("OPENCLAW_COMMAND", "openclaw")),
        cwd=PROJECT_ROOT,
        timeout_seconds=max(10.0, float(timeout_seconds)),
        silent=True,
    )
    result_rows = [
        {
            "ok": bool(row.ok),
            "returncode": int(row.returncode),
            "command": list(row.command),
            "stdout": row.stdout[-300:] if isinstance(row.stdout, str) else "",
            "stderr": row.stderr[-300:] if isinstance(row.stderr, str) else "",
        }
        for row in results
    ]
    sent = bool(results) and all(bool(row.ok) for row in results)
    return {
        "attempted": True,
        "sent": sent,
        "reason": decision.get("reason"),
        "results": result_rows,
    }


def _update_notification_state(
    *,
    path: Path,
    previous_state: dict[str, Any],
    decision: dict[str, Any],
    send_result: dict[str, Any],
) -> dict[str, Any]:
    state = dict(previous_state)
    state["version"] = 1
    state["updated_at_utc"] = _utc_now_iso()
    state["last_status"] = decision.get("status_key")
    state["last_issue_signature"] = decision.get("issue_signature")
    state["next_notify_at_utc"] = decision.get("next_notify_at_utc")
    state["last_notification_reason"] = decision.get("reason")
    if bool(send_result.get("attempted")):
        state["last_notification_attempt_at_utc"] = _utc_now_iso()
    if bool(send_result.get("sent")):
        state["last_notified_at_utc"] = _utc_now_iso()
    _safe_write_json(path, state)
    return state


def build_ops_verdict(
    *,
    command_name: str,
    snapshot: dict[str, Any],
    initial_issue_classes: dict[str, list[dict[str, Any]]],
    final_issue_classes: dict[str, list[dict[str, Any]]],
    attempted_actions: list[dict[str, Any]],
    completed_actions: list[dict[str, Any]],
    cooldowns: dict[str, Any],
    operator_summary: dict[str, Any],
    notification: dict[str, Any],
) -> dict[str, Any]:
    components = snapshot.get("components") if isinstance(snapshot.get("components"), dict) else {}
    overall_status = _build_overall_status(issue_classes=final_issue_classes, components=components)
    human_required = [
        row
        for row in _flatten_issue_classes(final_issue_classes)
        if str(row.get("class") or "") != "recoverable_service_failure"
    ]
    return {
        "status": "PASS",
        "action": f"openclaw_ops_{command_name}",
        "timestamp_utc": _utc_now_iso(),
        "overall_status": overall_status,
        "issue_classes": final_issue_classes,
        "initial_issue_classes": initial_issue_classes,
        "components": components,
        "attempted_actions": attempted_actions,
        "completed_actions": completed_actions,
        "human_required": human_required,
        "cooldowns": cooldowns,
        "operator_summary": operator_summary,
        "notification": notification,
        "source_snapshots": {
            "readiness": snapshot.get("readiness_payload"),
            "runtime": snapshot.get("runtime_payload"),
            "captured_output": snapshot.get("captured_output"),
        },
    }


def _persist_verdict(path: Path, verdict: dict[str, Any]) -> None:
    _safe_write_json(path, verdict)


def run_ops_command(
    *,
    command_name: str,
    primary_channel: str,
    gate_artifact_path: Path,
    db_path: Path,
    dashboard_port: int,
    include_runtime: bool,
    timeout_seconds: float,
    apply_safe_recovery: bool,
    explicit_targets: list[str],
    maintenance_report_path: Path,
    maintenance_state_path: Path,
    watcher_json_path: Path,
    watcher_tickers: str,
    watcher_cycles: int,
    watcher_sleep_seconds: int,
    notify_on_change: bool,
    notify_targets: str,
    notify_to: str,
    cooldown_seconds: int,
    verdict_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    initial_snapshot = collect_ops_snapshot(
        gate_artifact_path=gate_artifact_path,
        db_path=db_path,
        primary_channel=primary_channel,
        dashboard_port=dashboard_port,
        include_runtime=include_runtime,
        timeout_seconds=timeout_seconds,
        maintenance_report_path=maintenance_report_path,
        maintenance_state_path=maintenance_state_path,
        watcher_json_path=watcher_json_path,
    )
    initial_issue_classes = classify_ops_issues(initial_snapshot)

    attempted_actions: list[dict[str, Any]] = []
    if explicit_targets:
        requested_targets = explicit_targets
    elif apply_safe_recovery:
        requested_targets = _service_issue_targets(initial_issue_classes)
    else:
        requested_targets = []

    if requested_targets:
        attempted_actions = _run_recovery_targets(
            targets=requested_targets,
            primary_channel=primary_channel,
            timeout_seconds=timeout_seconds,
            maintenance_report_path=maintenance_report_path,
            dashboard_port=dashboard_port,
            db_path=db_path,
            watcher_tickers=watcher_tickers,
            watcher_cycles=watcher_cycles,
            watcher_sleep_seconds=watcher_sleep_seconds,
        )

    final_snapshot = collect_ops_snapshot(
        gate_artifact_path=gate_artifact_path,
        db_path=db_path,
        primary_channel=primary_channel,
        dashboard_port=dashboard_port,
        include_runtime=include_runtime,
        timeout_seconds=timeout_seconds,
        maintenance_report_path=maintenance_report_path,
        maintenance_state_path=maintenance_state_path,
        watcher_json_path=watcher_json_path,
    )
    final_issue_classes = classify_ops_issues(final_snapshot)
    completed_actions = _completed_actions_from_targets(
        initial_issue_classes=initial_issue_classes,
        final_issue_classes=final_issue_classes,
        attempted_actions=attempted_actions,
    )
    overall_status = _build_overall_status(
        issue_classes=final_issue_classes,
        components=final_snapshot.get("components") if isinstance(final_snapshot.get("components"), dict) else {},
    )
    operator_summary = build_operator_summary(
        initial_issue_classes=initial_issue_classes,
        final_issue_classes=final_issue_classes,
        attempted_actions=attempted_actions,
        completed_actions=completed_actions,
        components=final_snapshot.get("components") if isinstance(final_snapshot.get("components"), dict) else {},
        overall_status=overall_status,
    )

    previous_state = _read_notification_state(state_path)
    maintenance_component = ((final_snapshot.get("components") or {}) if isinstance(final_snapshot.get("components"), dict) else {}).get("maintenance")
    maintenance_component = maintenance_component if isinstance(maintenance_component, dict) else {}
    cooldowns = _build_cooldowns(
        state=previous_state,
        cooldown_seconds=int(cooldown_seconds),
        maintenance_component=maintenance_component,
    )
    notification_decision = _decide_notification(
        final_issue_classes=final_issue_classes,
        operator_summary=operator_summary,
        state=previous_state,
        cooldown_seconds=int(cooldown_seconds),
    )
    send_result = (
        _send_notification(
            decision=notification_decision,
            notify_targets=notify_targets,
            notify_to=notify_to,
            primary_channel=primary_channel,
            timeout_seconds=timeout_seconds,
        )
        if notify_on_change
        else {"attempted": False, "sent": False, "reason": "notifications_disabled", "results": []}
    )
    updated_state = _update_notification_state(
        path=state_path,
        previous_state=previous_state,
        decision=notification_decision,
        send_result=send_result,
    )
    cooldowns = _build_cooldowns(
        state=updated_state,
        cooldown_seconds=int(cooldown_seconds),
        maintenance_component=maintenance_component,
    )

    verdict = build_ops_verdict(
        command_name=command_name,
        snapshot=final_snapshot,
        initial_issue_classes=initial_issue_classes,
        final_issue_classes=final_issue_classes,
        attempted_actions=attempted_actions,
        completed_actions=completed_actions,
        cooldowns=cooldowns,
        operator_summary=operator_summary,
        notification=send_result,
    )
    _persist_verdict(verdict_path, verdict)
    return verdict


def _print_human_summary(verdict: dict[str, Any]) -> None:
    overall = verdict.get("overall_status") if isinstance(verdict.get("overall_status"), dict) else {}
    summary = verdict.get("operator_summary") if isinstance(verdict.get("operator_summary"), dict) else {}
    print(
        f"[openclaw_ops_control_plane] status={overall.get('status')} "
        f"service_health={overall.get('service_health')} "
        f"production_status={overall.get('production_status')}"
    )
    text = str(summary.get("message") or "").strip()
    if text:
        print(text)


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--primary-channel", default=str(os.getenv("OPENCLAW_CHANNEL", "whatsapp")).strip().lower() or "whatsapp")
    common.add_argument("--gate-artifact", default=str(DEFAULT_GATE_ARTIFACT))
    common.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    common.add_argument("--dashboard-port", type=int, default=8000)
    common.add_argument("--timeout-seconds", type=float, default=30.0)
    common.add_argument("--maintenance-report", default=str(DEFAULT_MAINTENANCE_REPORT_PATH))
    common.add_argument("--maintenance-state", default=str(DEFAULT_MAINTENANCE_STATE_PATH))
    common.add_argument("--watcher-json", default=str(DEFAULT_WATCHER_JSON))
    common.add_argument("--watcher-tickers", default=",".join(dashboard_mod.DEFAULT_LIVE_WATCHER_TICKERS))
    common.add_argument("--watcher-cycles", type=int, default=30)
    common.add_argument("--watcher-sleep-seconds", type=int, default=86400)
    common.add_argument("--notify-targets", default=str(os.getenv("OPENCLAW_TARGETS", "")))
    common.add_argument("--notify-to", default=str(os.getenv("OPENCLAW_TO", "")))
    common.add_argument("--cooldown-seconds", type=int, default=_env_int("OPENCLAW_OPS_NOTIFY_COOLDOWN_SECONDS", DEFAULT_NOTIFY_COOLDOWN_SECONDS, minimum=60))
    common.add_argument("--verdict-path", default=str(DEFAULT_VERDICT_PATH))
    common.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    common.add_argument("--json", action="store_true")

    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    status = subparsers.add_parser("status", parents=[common], help="Collect current health without mutations.")
    status.add_argument("--include-runtime", action="store_true", help="Include a fresh runtime snapshot.")
    status.add_argument("--notify-on-change", action="store_true", help="Send anomaly/recovery notifications when state changed.")

    sweep = subparsers.add_parser("sweep", parents=[common], help="Collect health, classify issues, and optionally apply safe recovery.")
    sweep.add_argument("--include-runtime", action="store_true", default=True, help="Include a fresh runtime snapshot.")
    sweep.add_argument("--apply-safe-recovery", action="store_true", help="Auto-recover only approved service targets.")
    sweep.add_argument("--notify-on-change", action="store_true", help="Send anomaly/recovery notifications when state changed.")

    recover = subparsers.add_parser("recover", parents=[common], help="Recover approved service targets only.")
    recover.add_argument("--targets", required=True, help="Comma-separated recovery targets: gateway,dashboard,watcher")
    recover.add_argument("--include-runtime", action="store_true", help="Include a fresh runtime snapshot.")
    recover.add_argument("--notify-on-change", action="store_true", help="Send anomaly/recovery notifications when state changed.")

    intent = subparsers.add_parser("intent", parents=[common], help="Route a narrow WhatsApp-style ops request.")
    intent.add_argument("text", help="Ops-only request such as 'heal gateway' or 'show readiness'.")
    intent.add_argument("--notify-on-change", action="store_true", help="Send anomaly/recovery notifications when state changed.")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command_name = str(args.cmd)
    include_runtime = bool(getattr(args, "include_runtime", False))
    apply_safe_recovery = bool(getattr(args, "apply_safe_recovery", False))
    explicit_targets: list[str] = []

    if command_name == "recover":
        explicit_targets = _normalize_requested_targets(getattr(args, "targets", ""))
    elif command_name == "intent":
        intent_request = extract_ops_intent_request(getattr(args, "text", ""))
        command_name = str(intent_request.get("command") or "status")
        include_runtime = bool(intent_request.get("include_runtime"))
        explicit_targets = _normalize_requested_targets(intent_request.get("targets", []))
        apply_safe_recovery = command_name == "recover"

    try:
        verdict = run_ops_command(
            command_name=command_name,
            primary_channel=str(args.primary_channel),
            gate_artifact_path=Path(str(args.gate_artifact)).expanduser().resolve(),
            db_path=Path(str(args.db_path)).expanduser().resolve(),
            dashboard_port=int(args.dashboard_port),
            include_runtime=include_runtime,
            timeout_seconds=float(args.timeout_seconds),
            apply_safe_recovery=apply_safe_recovery,
            explicit_targets=explicit_targets,
            maintenance_report_path=Path(str(args.maintenance_report)).expanduser().resolve(),
            maintenance_state_path=Path(str(args.maintenance_state)).expanduser().resolve(),
            watcher_json_path=Path(str(args.watcher_json)).expanduser().resolve(),
            watcher_tickers=str(args.watcher_tickers),
            watcher_cycles=int(args.watcher_cycles),
            watcher_sleep_seconds=int(args.watcher_sleep_seconds),
            notify_on_change=bool(getattr(args, "notify_on_change", False)),
            notify_targets=str(args.notify_targets or ""),
            notify_to=str(args.notify_to or ""),
            cooldown_seconds=int(args.cooldown_seconds),
            verdict_path=Path(str(args.verdict_path)).expanduser().resolve(),
            state_path=Path(str(args.state_path)).expanduser().resolve(),
        )
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "action": "openclaw_ops_control_plane",
            "timestamp_utc": _utc_now_iso(),
            "error": str(exc),
        }
        if bool(args.json):
            print(json.dumps(failure, indent=2))
        else:
            print(f"[openclaw_ops_control_plane] FAIL - {exc}", file=sys.stderr)
        return 2

    if bool(args.json):
        print(json.dumps(verdict, indent=2))
    else:
        _print_human_summary(verdict)

    overall = verdict.get("overall_status") if isinstance(verdict.get("overall_status"), dict) else {}
    return 1 if str(overall.get("status") or "") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
