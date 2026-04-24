"""
openclaw_remote_workflow.py
---------------------------
OpenClaw remote workflow manager for Portfolio Maximizer.

This tool focuses on the channel-driven remote development path used by PMX:
the gateway stays local/loopback on Windows, while WhatsApp/Telegram/Discord
provide the actual remote interaction surface.

Commands:
  status          Full remote workflow status (default)
  health          Quick health check (exit 0=OK, 1=WARN, 2=FAIL)
  diagnose        Deep diagnostic: gateway, channels, delivery, agent bindings
  channel-test    Send a test notification on the primary channel
  bridge-test     Exercise the OpenClaw -> orchestrator -> WhatsApp reply path
  gateway-restart Restart the OpenClaw gateway and verify connectivity
  cron-health     Show cron job health + delivery failure breakdown
  failover-test   Verify Telegram fallback is configured for WhatsApp jobs

Usage:
  python scripts/openclaw_remote_workflow.py [command] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.openclaw_cron_contract import load_cron_jobs_payload, summarize_cron_jobs

OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
CRON_JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"
OPENCLAW_MAINTENANCE_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_maintenance_latest.json"
OPENCLAW_MAINTENANCE_STATE_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_maintenance_state.json"
PRIMARY_CHANNEL_DEFAULT = str(os.getenv("OPENCLAW_CHANNEL", "whatsapp")).strip().lower() or "whatsapp"
_GATEWAY_PORT = 18789


def _bootstrap_dotenv() -> None:
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        return


def _load_config() -> Dict[str, Any]:
    if not OPENCLAW_CONFIG.exists():
        return {}
    try:
        return json.loads(OPENCLAW_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_cron_jobs_payload() -> Tuple[Dict[str, Any], Optional[str]]:
    return load_cron_jobs_payload(CRON_JOBS_PATH)


def _load_cron_jobs() -> List[Dict[str, Any]]:
    payload, _ = _load_cron_jobs_payload()
    jobs = payload.get("jobs", [])
    return jobs if isinstance(jobs, list) else []


def _parse_json_best_effort(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_age_minutes(path: Path, payload: Dict[str, Any]) -> Optional[float]:
    candidates: List[str] = []
    for key in ("timestamp_utc", "generated_utc", "updated_utc"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    if path.exists():
        candidates.append(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat())
    for value in candidates:
        normalized = value.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(normalized)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 60.0)
    return None


def _load_recent_maintenance_payload() -> Tuple[Dict[str, Any], Optional[float]]:
    payload = _read_json_file(OPENCLAW_MAINTENANCE_PATH)
    age_minutes = _payload_age_minutes(OPENCLAW_MAINTENANCE_PATH, payload) if payload else None
    return payload, age_minutes


def _load_recent_maintenance_state() -> Tuple[Dict[str, Any], Optional[float]]:
    payload = _read_json_file(OPENCLAW_MAINTENANCE_STATE_PATH)
    age_minutes = _payload_age_minutes(OPENCLAW_MAINTENANCE_STATE_PATH, payload) if payload else None
    return payload, age_minutes


def _split_openclaw_command(command: str) -> List[str]:
    try:
        from utils.openclaw_cli import _split_command as _split  # type: ignore

        return [str(x) for x in _split(command)]
    except Exception:
        raw = (command or "").strip() or "openclaw"
        if os.name == "nt":
            return ["cmd", "/d", "/s", "/c", raw]
        return [raw]


def _openclaw_command() -> str:
    return str(os.getenv("OPENCLAW_COMMAND", "openclaw")).strip() or "openclaw"


def _openclaw_base() -> List[str]:
    return _split_openclaw_command(_openclaw_command())


def _run(cmd: List[str], timeout: float = 15.0, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    try:
        env = dict(os.environ)
        env.setdefault("NODE_NO_WARNINGS", "1")
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or PROJECT_ROOT,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout ({timeout}s) waiting for: {' '.join(cmd)}"
    except Exception as exc:
        return -1, "", str(exc)


def _run_openclaw(args: List[str], timeout: float = 15.0) -> Tuple[int, str, str]:
    return _run([*_openclaw_base(), *args], timeout=timeout, cwd=PROJECT_ROOT)


def _run_openclaw_json(args: List[str], timeout: float = 15.0) -> Tuple[int, Optional[Any], str, str]:
    final_args = ["--no-color", *args]
    if "--json" not in final_args:
        final_args.append("--json")
    rc, out, err = _run_openclaw(final_args, timeout=timeout)
    if rc != 0:
        return rc, None, out, err
    try:
        return rc, _parse_json_best_effort(out), out, err
    except Exception:
        return rc, None, out, err


def _openclaw_available() -> bool:
    base = _openclaw_base()
    if not base:
        return False
    prog = str(base[0]).strip().lower()
    if prog in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return shutil.which(base[0]) is not None
    if Path(base[0]).exists():
        return True
    return shutil.which(base[0]) is not None


def _gateway_local_ping() -> Tuple[bool, str]:
    try:
        import urllib.request

        url = f"http://127.0.0.1:{_GATEWAY_PORT}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, body[:200]
    except Exception as exc:
        return False, str(exc)


def _load_live_channels_payload(timeout: float = 20.0) -> Optional[Dict[str, Any]]:
    rc, payload, _, _ = _run_openclaw_json(["channels", "status"], timeout=timeout)
    return payload if rc == 0 and isinstance(payload, dict) else None


def _load_live_channels_status(timeout: float = 20.0) -> Dict[str, Any]:
    started = time.monotonic()
    rc, payload, out, err = _run_openclaw_json(["channels", "status"], timeout=timeout)
    elapsed_ms = int(round((time.monotonic() - started) * 1000.0))
    timeout_hit = bool(
        rc != 0
        and any(
            token in str(text or "")
            for text in (out, err)
            for token in (f"Timeout ({timeout}s)", f"Timeout ({int(timeout)}s)", "Timeout (")
        )
    )
    return {
        "rc": rc,
        "payload": payload if isinstance(payload, dict) else None,
        "parsed": isinstance(payload, dict),
        "elapsed_ms": elapsed_ms,
        "timeout": timeout_hit,
        "stdout": out[:500],
        "stderr": err[:200],
    }


def _load_channels_status_context(
    *,
    timeout: float = 20.0,
    prefer_recent_maintenance: bool = True,
    maintenance_max_age_minutes: float = 15.0,
) -> Dict[str, Any]:
    maintenance_payload, maintenance_age_minutes = _load_recent_maintenance_payload()
    if prefer_recent_maintenance and isinstance(maintenance_payload, dict):
        steps = maintenance_payload.get("steps") if isinstance(maintenance_payload.get("steps"), dict) else {}
        snapshot = (
            steps.get("channels_status_snapshot")
            if isinstance(steps.get("channels_status_snapshot"), dict)
            else {}
        )
        if maintenance_age_minutes is not None and maintenance_age_minutes <= maintenance_max_age_minutes:
            if snapshot:
                return {
                    "rc": 0,
                    "payload": snapshot,
                    "parsed": True,
                    "elapsed_ms": 0,
                    "timeout": False,
                    "stdout": "",
                    "stderr": "",
                    "source": "maintenance_snapshot",
                    "maintenance_age_minutes": round(maintenance_age_minutes, 2),
                }

    maintenance_state, maintenance_state_age_minutes = _load_recent_maintenance_state()
    if prefer_recent_maintenance and isinstance(maintenance_state, dict):
        snapshot = (
            maintenance_state.get("last_channels_status_snapshot")
            if isinstance(maintenance_state.get("last_channels_status_snapshot"), dict)
            else {}
        )
        if snapshot and maintenance_state_age_minutes is not None and maintenance_state_age_minutes <= maintenance_max_age_minutes:
            return {
                "rc": 0,
                "payload": snapshot,
                "parsed": True,
                "elapsed_ms": 0,
                "timeout": False,
                "stdout": "",
                "stderr": "",
                "source": "maintenance_state_snapshot",
                "maintenance_state_age_minutes": round(maintenance_state_age_minutes, 2),
            }

    if prefer_recent_maintenance and isinstance(maintenance_payload, dict):
        if maintenance_age_minutes is not None and maintenance_age_minutes <= maintenance_max_age_minutes:
            return {
                "rc": 0,
                "payload": None,
                "parsed": False,
                "elapsed_ms": 0,
                "timeout": False,
                "stdout": "",
                "stderr": "",
                "source": "maintenance_report",
                "maintenance_age_minutes": round(maintenance_age_minutes, 2),
            }

    channels_info = _load_live_channels_status(timeout=timeout)
    channels_info["source"] = "live"
    return channels_info


def _channel_row(payload: Dict[str, Any], channel: str) -> Dict[str, Any]:
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    row = channels.get(channel) if isinstance(channels, dict) else None
    return row if isinstance(row, dict) else {}


def _channel_account_row(payload: Dict[str, Any], channel: str, account_id: str) -> Dict[str, Any]:
    accounts = payload.get("channelAccounts") if isinstance(payload.get("channelAccounts"), dict) else {}
    rows = accounts.get(channel) if isinstance(accounts, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("accountId") or "").strip() == account_id:
                return row
    return {}


def _resolve_default_account_id(payload: Dict[str, Any], channel: str) -> str:
    defaults = payload.get("channelDefaultAccountId") if isinstance(payload.get("channelDefaultAccountId"), dict) else {}
    account_id = str(defaults.get(channel) or "").strip()
    return account_id or "default"


def _channel_ready_from_live(payload: Dict[str, Any], channel: str) -> Dict[str, Any]:
    row = _channel_row(payload, channel)
    account_id = _resolve_default_account_id(payload, channel)
    account = _channel_account_row(payload, channel, account_id)

    configured = bool(row.get("configured", account.get("configured", False)))
    enabled = bool(account.get("enabled", configured))
    running = bool(account.get("running", row.get("running", False)))
    linked = bool(account.get("linked", row.get("linked", False)))
    connected = bool(account.get("connected", row.get("connected", running)))
    last_error = str(account.get("lastError") or row.get("lastError") or "").strip()

    if channel == "whatsapp":
        ok = configured and enabled and linked and running and connected and not last_error
    else:
        ok = configured and enabled and running and not last_error

    status = "OK" if ok else ("WARN" if configured or enabled else "OFF")
    result = {
        "status": status,
        "configured": configured,
        "enabled": enabled,
        "running": running,
        "last_error": last_error or None,
        "account_id": account_id,
    }
    if channel == "whatsapp":
        result["linked"] = linked
        result["connected"] = connected
    return result


def _channel_ready_from_snapshot(snapshot: Dict[str, Any], channel: str) -> Dict[str, Any]:
    configured = bool(snapshot.get("configured", False))
    enabled = bool(snapshot.get("enabled", configured))
    running = bool(snapshot.get("running", False))
    linked = bool(snapshot.get("linked", False))
    connected = bool(snapshot.get("connected", running))
    last_error = str(snapshot.get("lastError") or snapshot.get("last_error") or "").strip()

    if channel == "whatsapp":
        ok = configured and enabled and linked and running and connected and not last_error
    else:
        ok = configured and enabled and running and not last_error

    result = {
        "status": "OK" if ok else ("WARN" if configured or enabled else "OFF"),
        "configured": configured,
        "enabled": enabled,
        "running": running,
        "last_error": last_error or None,
        "account_id": str(snapshot.get("accountId") or "default"),
        "source": "maintenance_snapshot",
    }
    if channel == "whatsapp":
        result["linked"] = linked
        result["connected"] = connected
    return result


def _channel_ready_from_config(cfg: Dict[str, Any], channel: str) -> Dict[str, Any]:
    channel_cfg = cfg.get("channels", {}).get(channel, {})
    accounts = channel_cfg.get("accounts", {}) if isinstance(channel_cfg.get("accounts"), dict) else {}
    account_cfg = accounts.get("default", {}) if isinstance(accounts, dict) else {}

    configured = bool(channel_cfg.get("enabled", False))
    enabled = bool(account_cfg.get("enabled", configured))
    result = {
        "status": "WARN" if configured else "OFF",
        "configured": configured,
        "enabled": enabled,
        "running": False,
        "last_error": None,
        "account_id": "default",
    }
    if channel == "whatsapp":
        result["linked"] = False
        result["connected"] = False
    return result


def _primary_channel_ready(payload: Optional[Dict[str, Any]], primary_channel: str) -> bool:
    if not isinstance(payload, dict):
        return False
    return _channel_ready_from_live(payload, primary_channel).get("status") == "OK"


def _recent_recovery_context(
    *,
    primary_channel: str,
    maintenance_payload: Optional[Dict[str, Any]],
    maintenance_age_minutes: Optional[float],
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "usable": False,
        "age_minutes": round(float(maintenance_age_minutes), 2) if maintenance_age_minutes is not None else None,
        "mode": "none",
        "detail": "",
        "events": [],
        "gateway_rpc_ok": None,
        "channel_snapshot": {},
    }
    if not isinstance(maintenance_payload, dict):
        return context

    freshness_limit_minutes = 15.0
    try:
        freshness_limit_minutes = max(
            1.0,
            float(os.getenv("PMX_OPENCLAW_RECOVERY_CONTEXT_MAX_AGE_MINUTES", "15")),
        )
    except ValueError:
        freshness_limit_minutes = 15.0

    if maintenance_age_minutes is None or maintenance_age_minutes > freshness_limit_minutes:
        return context

    steps = maintenance_payload.get("steps") if isinstance(maintenance_payload.get("steps"), dict) else {}
    fast_supervisor = steps.get("fast_supervisor") if isinstance(steps.get("fast_supervisor"), dict) else {}
    gateway_health = steps.get("gateway_health") if isinstance(steps.get("gateway_health"), dict) else {}
    channels_snapshot = (
        steps.get("channels_status_snapshot")
        if isinstance(steps.get("channels_status_snapshot"), dict)
        else {}
    )
    snapshot_channels = channels_snapshot.get("channels") if isinstance(channels_snapshot.get("channels"), dict) else {}
    primary_snapshot = snapshot_channels.get(primary_channel) if isinstance(snapshot_channels.get(primary_channel), dict) else {}

    gateway_warnings = [str(x) for x in gateway_health.get("warnings", [])] if isinstance(gateway_health, dict) else []
    fast_warnings = [str(x) for x in fast_supervisor.get("warnings", [])] if isinstance(fast_supervisor, dict) else []
    channel_row = _channel_ready_from_snapshot(primary_snapshot, primary_channel) if primary_snapshot else {}
    snapshot_ok = channel_row.get("status") == "OK"

    events: List[str] = []
    if fast_supervisor.get("action") == "soft_timeout_skip" or fast_supervisor.get("reason") == "channels_status_timeout_softened":
        events.append("channels_status_timeout_softened")
    if fast_supervisor.get("action") == "gateway_restart_triggered" and not gateway_health.get("primary_channel_issue_final"):
        events.append("gateway_restart_recovered")
    if (
        gateway_health.get("primary_channel_issue") == "whatsapp_handshake_timeout"
        and not gateway_health.get("primary_channel_issue_final")
    ):
        events.append("whatsapp_handshake_recovered")
    if "gateway_detached_listener_conflict" in gateway_warnings:
        events.append("gateway_detached_listener_conflict")

    detail_parts: List[str] = []
    if events:
        detail_parts.append(",".join(events))
    if snapshot_ok:
        detail_parts.append(f"{primary_channel}_snapshot_ok")
    if gateway_health.get("rpc_ok") is True:
        detail_parts.append("gateway_rpc_ok")

    context.update(
        {
            "usable": bool(events or snapshot_ok or gateway_health.get("rpc_ok") is True),
            "mode": events[0] if events else ("steady_state" if snapshot_ok else "none"),
            "detail": "; ".join(detail_parts),
            "events": events,
            "gateway_rpc_ok": gateway_health.get("rpc_ok"),
            "gateway_service_status": str(gateway_health.get("service_status") or ""),
            "channel_snapshot": channel_row,
            "fast_supervisor_action": str(fast_supervisor.get("action") or ""),
            "fast_supervisor_reason": str(fast_supervisor.get("reason") or ""),
            "gateway_warnings": gateway_warnings,
            "fast_supervisor_warnings": fast_warnings,
        }
    )
    return context


def _mask_target(target: str) -> str:
    raw = str(target or "").strip()
    if len(raw) <= 6:
        return raw
    return f"{raw[:4]}***{raw[-2:]}"


def _infer_whatsapp_target(cfg: Dict[str, Any]) -> str:
    try:
        from utils.openclaw_cli import infer_linked_whatsapp_target

        inferred = infer_linked_whatsapp_target(
            command=_openclaw_command(),
            cwd=PROJECT_ROOT,
            timeout_seconds=10.0,
        )
        if inferred:
            return str(inferred).strip()
    except Exception:
        pass

    allow_from = cfg.get("channels", {}).get("whatsapp", {}).get("allowFrom", [])
    if isinstance(allow_from, list):
        for item in allow_from:
            text = str(item or "").strip()
            if text:
                return text
    return ""


def _send_openclaw_message(*, channel: str, to: str, message: str) -> Tuple[bool, int, str]:
    try:
        from utils.openclaw_cli import send_message

        result = send_message(
            command=_openclaw_command(),
            to=to,
            message=message,
            channel=channel,
            cwd=PROJECT_ROOT,
        )
        detail = result.stderr[:200] if result.stderr else result.stdout[:200]
        return result.ok, int(result.returncode), detail
    except Exception as exc:
        return False, -1, str(exc)


def _run_bridge_test(*, channel: str, reply_to: str, message: str, timeout: float = 120.0) -> Tuple[int, str, str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "llm_multi_model_orchestrator.py"),
        "openclaw-bridge",
        "--channel",
        channel,
        "--reply-to",
        reply_to,
        "--message",
        message,
    ]
    return _run(cmd, timeout=timeout, cwd=PROJECT_ROOT)


def _check_version(cfg: Dict[str, Any]) -> Dict[str, Any]:
    installed = cfg.get("meta", {}).get("lastTouchedVersion", "unknown")
    return {
        "check": "version",
        "required": False,
        "status": "OK",
        "installed": installed,
        "detail": f"Config last touched version {installed}",
    }


def _check_gateway(
    cfg: Dict[str, Any],
    *,
    primary_channel: str,
    channels_payload: Optional[Dict[str, Any]],
    maintenance_payload: Optional[Dict[str, Any]] = None,
    maintenance_age_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    gw = cfg.get("gateway", {})
    mode = str(gw.get("mode", "unknown") or "unknown")
    bind = str(gw.get("bind", "unknown") or "unknown")
    remote_cfg = gw.get("remote", {}) if isinstance(gw.get("remote"), dict) else {}
    reachable, ping_detail = _gateway_local_ping()
    primary_ready = _primary_channel_ready(channels_payload, primary_channel)
    recovery = _recent_recovery_context(
        primary_channel=primary_channel,
        maintenance_payload=maintenance_payload,
        maintenance_age_minutes=maintenance_age_minutes,
    )

    issues: List[str] = []
    detail: str
    access_mode = "unknown"
    status = "OK"

    if not reachable:
        if recovery.get("usable"):
            issues.append("local gateway probe degraded")
            status = "WARN"
        else:
            issues.append("local gateway unreachable")
            status = "FAIL"

    if mode == "remote":
        access_mode = "direct-remote"
        if bind == "loopback":
            issues.append("bind=loopback (direct external gateway blocked)")
        if not remote_cfg.get("allowExternalAgentTrigger"):
            issues.append("allowExternalAgentTrigger not set")
        detail = "Gateway configured for direct external trigger mode"
    elif mode == "local" and bind == "loopback":
        access_mode = "channel-driven"
        if primary_ready:
            detail = f"Loopback gateway reachable; remote dev is channel-driven via {primary_channel}"
        elif recovery.get("usable") and recovery.get("channel_snapshot", {}).get("status") == "OK":
            status = "WARN"
            issues.append(f"{primary_channel} live probe unavailable; recent maintenance snapshot recovered")
            detail = (
                f"Loopback gateway is channel-driven via {primary_channel}; "
                f"live probe degraded but recent maintenance shows {recovery.get('detail') or 'recovered channel state'}"
            )
        else:
            issues.append(f"{primary_channel} channel not ready")
            detail = f"Loopback gateway reachable, but {primary_channel} is not ready for remote dev"
            status = "FAIL" if status != "WARN" else status
    else:
        access_mode = f"{mode}:{bind}"
        if not primary_ready:
            if recovery.get("usable") and recovery.get("channel_snapshot", {}).get("status") == "OK":
                issues.append(f"{primary_channel} live probe unavailable; recent maintenance snapshot recovered")
                status = "WARN"
            else:
                issues.append(f"{primary_channel} channel not ready")
                status = "FAIL" if status != "WARN" else status
        detail = f"Gateway mode={mode} bind={bind}"

    if status == "OK" and issues:
        status = "WARN"
    return {
        "check": "gateway",
        "required": True,
        "status": status,
        "mode": mode,
        "bind": bind,
        "reachable": reachable,
        "access_mode": access_mode,
        "remote_config": remote_cfg,
        "issues": issues,
        "primary_channel_ready": primary_ready,
        "ping_detail": ping_detail[:120] if ping_detail else "",
        "recovery_context": recovery,
        "detail": detail,
    }


def _check_channels(
    cfg: Dict[str, Any],
    *,
    primary_channel: str,
    channels_payload: Optional[Dict[str, Any]],
    maintenance_payload: Optional[Dict[str, Any]] = None,
    maintenance_age_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    channel_names = ("whatsapp", "telegram", "discord")
    recovery = _recent_recovery_context(
        primary_channel=primary_channel,
        maintenance_payload=maintenance_payload,
        maintenance_age_minutes=maintenance_age_minutes,
    )
    rows: Dict[str, Dict[str, Any]] = {}
    snapshot_channels = {}
    if isinstance(maintenance_payload, dict):
        steps = maintenance_payload.get("steps") if isinstance(maintenance_payload.get("steps"), dict) else {}
        channels_snapshot = (
            steps.get("channels_status_snapshot")
            if isinstance(steps.get("channels_status_snapshot"), dict)
            else {}
        )
        snapshot_channels = channels_snapshot.get("channels") if isinstance(channels_snapshot.get("channels"), dict) else {}
    for name in channel_names:
        if isinstance(channels_payload, dict):
            rows[name] = _channel_ready_from_live(channels_payload, name)
            rows[name]["source"] = "live"
        elif isinstance(snapshot_channels.get(name), dict):
            rows[name] = _channel_ready_from_snapshot(snapshot_channels.get(name), name)
        else:
            rows[name] = _channel_ready_from_config(cfg, name)
            rows[name]["source"] = "config"

    primary_status = rows.get(primary_channel, {}).get("status", "OFF")
    fallback_ready = [name for name in channel_names if name != primary_channel and rows.get(name, {}).get("status") == "OK"]

    using_live = isinstance(channels_payload, dict)
    if (
        not using_live
        and recovery.get("usable")
        and rows.get(primary_channel, {}).get("source") == "maintenance_snapshot"
        and primary_status == "OK"
    ):
        primary_status = "RECOVERING"
    if primary_status == "OK":
        overall = "OK" if (fallback_ready and using_live) else "WARN"
    elif primary_status == "WARN":
        overall = "WARN"
    elif primary_status == "RECOVERING":
        overall = "WARN"
    else:
        overall = "FAIL"

    detail = (
        f"Primary {primary_channel}={primary_status}; "
        f"fallback_ready={','.join(fallback_ready) if fallback_ready else 'none'}; "
        f"source={'live' if using_live else rows.get(primary_channel, {}).get('source', 'config')}"
    )
    return {
        "check": "channels",
        "required": True,
        "status": overall,
        "primary_channel": primary_channel,
        "primary_status": primary_status,
        "fallback_ready": fallback_ready,
        "channels": rows,
        "recovery_context": recovery,
        "detail": detail,
    }


def _check_bindings(cfg: Dict[str, Any], *, primary_channel: str) -> Dict[str, Any]:
    bindings = cfg.get("bindings") if isinstance(cfg.get("bindings"), list) else []
    matched = False
    for row in bindings:
        if not isinstance(row, dict):
            continue
        agent_id = str(row.get("agentId") or "").strip()
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        channel = str(match.get("channel") or "").strip().lower()
        account_id = str(match.get("accountId") or "").strip() or "default"
        if channel == primary_channel and account_id == "default" and agent_id == "ops":
            matched = True
            break

    return {
        "check": "bindings",
        "required": True,
        "status": "OK" if matched else "FAIL",
        "detail": f"{primary_channel}:default -> ops" if matched else f"Missing binding for {primary_channel}:default -> ops",
    }


def _check_agents(cfg: Dict[str, Any]) -> Dict[str, Any]:
    agents = cfg.get("agents", {}).get("list", [])
    agent_ids = [a["id"] for a in agents if isinstance(a, dict) and "id" in a]
    required = {"ops", "trading", "training", "notifier"}
    missing = sorted(required - set(agent_ids))

    return {
        "check": "agents",
        "required": False,
        "status": "OK" if not missing else "WARN",
        "agents": agent_ids,
        "missing_required": missing,
        "detail": "All required agents present" if not missing else f"Missing agents: {missing}",
    }


def _check_cron_jobs() -> Dict[str, Any]:
    payload, err = _load_cron_jobs_payload()
    summary = summarize_cron_jobs(payload)
    if err:
        summary = dict(summary)
        summary["status"] = "FAIL"
        summary["detail"] = err

    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
    if not jobs and summary.get("jobs_total", 0) == 0:
        return {
            "check": "cron_jobs",
            "required": False,
            "status": summary.get("status", "WARN"),
            "detail": summary.get("detail", "No cron jobs found"),
            "jobs_total": 0,
            "jobs_invalid": 0,
            "invalid_session_target_count": 0,
            "delivery_fallback_ready_count": 0,
            "stale_python_path_count": 0,
        }

    total = summary.get("jobs_total", len(jobs))
    enabled = [j for j in jobs if isinstance(j, dict) and j.get("enabled", False)]
    failing = [j for j in enabled if j.get("state", {}).get("consecutiveErrors", 0) > 0]
    delivery_failures = [
        j for j in failing if "delivery" in str(j.get("state", {}).get("lastError", "")).lower()
    ]
    with_fallback = [j for j in enabled if "fallback" in j.get("delivery", {})]

    malformed_count = int(summary.get("jobs_invalid", 0) or 0)
    summary_status = str(summary.get("status", "OK")).strip().upper() or "OK"
    invalid_session_target_count = int(summary.get("invalid_session_target_count", 0) or 0)
    status = "FAIL" if summary_status == "FAIL" else ("WARN" if summary_status == "WARN" or failing else "OK")
    return {
        "check": "cron_jobs",
        "required": False,
        "status": status,
        "total": total,
        "enabled": len(enabled),
        "failing": len(failing),
        "delivery_failures": len(delivery_failures),
        "with_telegram_fallback": len(with_fallback),
        "delivery_fallback_ready_count": int(summary.get("delivery_fallback_ready_count", 0) or 0),
        "jobs_invalid": malformed_count,
        "invalid_session_target_count": invalid_session_target_count,
        "stale_python_path_count": int(summary.get("stale_python_path_count", 0) or 0),
        "job_rows": summary.get("job_rows", []),
        "invalid_jobs": summary.get("invalid_jobs", []),
        "failing_names": [j["name"] for j in failing],
        "detail": summary.get(
            "detail",
            f"{len(enabled)} enabled, {len(failing)} failing "
            f"({len(delivery_failures)} delivery), {len(with_fallback)} have fallback",
        ),
    }


def _check_interactions_api() -> Dict[str, Any]:
    port = int(os.getenv("INTERACTIONS_PORT", "8000"))
    try:
        import urllib.request

        url = f"http://127.0.0.1:{port}/"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=4):
            return {
                "check": "interactions_api",
                "required": False,
                "status": "OK",
                "port": port,
                "detail": f"Responding on port {port}",
            }
    except Exception as exc:
        return {
            "check": "interactions_api",
            "required": False,
            "status": "OFF",
            "port": port,
            "detail": f"Optional API not running on port {port}: {exc}",
        }


def _evaluate_overall(checks: List[Dict[str, Any]]) -> str:
    required = [c for c in checks if c.get("required", True)]
    advisory = [c for c in checks if not c.get("required", True)]
    required_statuses = [c.get("status") for c in required]
    advisory_statuses = [c.get("status") for c in advisory]

    if "FAIL" in required_statuses:
        return "FAIL"
    if "WARN" in required_statuses:
        return "WARN"
    if any(status in {"WARN", "FAIL"} for status in advisory_statuses):
        return "WARN"
    return "OK"


def cmd_status(as_json: bool = False) -> int:
    cfg = _load_config()
    channels_info = _load_channels_status_context()
    channels_payload = channels_info.get("payload") if isinstance(channels_info.get("payload"), dict) else None
    maintenance_payload, maintenance_age_minutes = _load_recent_maintenance_payload()

    checks = [
        _check_version(cfg),
        _check_gateway(
            cfg,
            primary_channel=PRIMARY_CHANNEL_DEFAULT,
            channels_payload=channels_payload,
            maintenance_payload=maintenance_payload,
            maintenance_age_minutes=maintenance_age_minutes,
        ),
        _check_channels(
            cfg,
            primary_channel=PRIMARY_CHANNEL_DEFAULT,
            channels_payload=channels_payload,
            maintenance_payload=maintenance_payload,
            maintenance_age_minutes=maintenance_age_minutes,
        ),
        _check_bindings(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT),
        _check_agents(cfg),
        _check_cron_jobs(),
        _check_interactions_api(),
    ]
    overall = _evaluate_overall(checks)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "primary_channel": PRIMARY_CHANNEL_DEFAULT,
        "openclaw_available": _openclaw_available(),
        "channels_status": {
            "parsed": bool(channels_info.get("parsed")),
            "elapsed_ms": channels_info.get("elapsed_ms"),
            "timeout": bool(channels_info.get("timeout")),
        },
        "checks": checks,
    }

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n[remote-workflow] Overall: {overall}")
        print(f"  OpenClaw CLI:  {'available' if result['openclaw_available'] else 'NOT FOUND'}")
        for c in checks:
            status = c["status"]
            icon = "[OK]" if status == "OK" else ("[!]" if status == "WARN" else "[ ]" if status == "OFF" else "[X]")
            suffix = " (required)" if c.get("required", True) else " (advisory)"
            print(f"  {icon} {c['check']:<22} {c.get('detail', status)}{suffix}")
        print()
    return 0 if overall == "OK" else (1 if overall == "WARN" else 2)


def cmd_health(as_json: bool = False) -> int:
    cfg = _load_config()
    channels_info = _load_channels_status_context()
    channels_payload = channels_info.get("payload") if isinstance(channels_info.get("payload"), dict) else None
    maintenance_payload, maintenance_age_minutes = _load_recent_maintenance_payload()
    gateway = _check_gateway(
        cfg,
        primary_channel=PRIMARY_CHANNEL_DEFAULT,
        channels_payload=channels_payload,
        maintenance_payload=maintenance_payload,
        maintenance_age_minutes=maintenance_age_minutes,
    )
    channels = _check_channels(
        cfg,
        primary_channel=PRIMARY_CHANNEL_DEFAULT,
        channels_payload=channels_payload,
        maintenance_payload=maintenance_payload,
        maintenance_age_minutes=maintenance_age_minutes,
    )
    binding = _check_bindings(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT)
    checks = [gateway, channels, binding]
    overall = _evaluate_overall(checks)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "gateway_reachable": bool(gateway.get("reachable")),
        "primary_channel": PRIMARY_CHANNEL_DEFAULT,
        "primary_status": channels.get("primary_status"),
        "fallback_ready": channels.get("fallback_ready", []),
        "fallback_ready_count": len(channels.get("fallback_ready", [])) if isinstance(channels.get("fallback_ready", []), list) else 0,
        "binding_ok": binding.get("status") == "OK",
        "channels_status_timeout": bool(channels_info.get("timeout")),
        "channels_status_elapsed_ms": channels_info.get("elapsed_ms"),
        "recovery_mode": channels.get("recovery_context", {}).get("mode"),
    }
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(
            "[health] "
            f"{overall} gateway={result['gateway_reachable']} "
            f"primary={PRIMARY_CHANNEL_DEFAULT}:{result['primary_status']} "
            f"binding={result['binding_ok']}"
        )
    return 0 if overall == "OK" else (1 if overall == "WARN" else 2)


def cmd_diagnose(as_json: bool = False) -> int:
    cfg = _load_config()
    channels_info = _load_live_channels_status()
    channels_payload = channels_info.get("payload") if isinstance(channels_info.get("payload"), dict) else None
    maintenance_payload, maintenance_age_minutes = _load_recent_maintenance_payload()
    diag: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(OPENCLAW_CONFIG),
        "config_exists": OPENCLAW_CONFIG.exists(),
        "openclaw_command": _openclaw_base(),
        "openclaw_cli": _openclaw_available(),
        "maintenance_path": str(OPENCLAW_MAINTENANCE_PATH),
        "maintenance_exists": OPENCLAW_MAINTENANCE_PATH.exists(),
        "maintenance_age_minutes": round(maintenance_age_minutes, 2) if maintenance_age_minutes is not None else None,
    }

    gw_ok, gw_detail = _gateway_local_ping()
    diag["gateway_local_reachable"] = gw_ok
    diag["gateway_ping_detail"] = gw_detail[:200]

    rc, out, err = _run_openclaw(["status"], timeout=12.0)
    diag["openclaw_status_rc"] = rc
    diag["openclaw_status_stdout"] = out[:500]
    diag["openclaw_status_stderr"] = err[:200]

    diag["channels_status_rc"] = channels_info.get("rc")
    diag["channels_status_parsed"] = bool(channels_info.get("parsed"))
    diag["channels_status_elapsed_ms"] = channels_info.get("elapsed_ms")
    diag["channels_status_timeout"] = bool(channels_info.get("timeout"))
    diag["channels_status_stdout"] = channels_info.get("stdout")
    diag["channels_status_stderr"] = channels_info.get("stderr")

    if isinstance(channels_payload, dict):
        diag["primary_channel_snapshot"] = _channel_ready_from_live(channels_payload, PRIMARY_CHANNEL_DEFAULT)
    diag["recovery_context"] = _recent_recovery_context(
        primary_channel=PRIMARY_CHANNEL_DEFAULT,
        maintenance_payload=maintenance_payload,
        maintenance_age_minutes=maintenance_age_minutes,
    )

    cron_payload, cron_err = _load_cron_jobs_payload()
    cron_summary = summarize_cron_jobs(cron_payload)
    if cron_err:
        cron_summary = dict(cron_summary)
        cron_summary["status"] = "FAIL"
        cron_summary["detail"] = cron_err
    jobs = cron_payload.get("jobs") if isinstance(cron_payload.get("jobs"), list) else []
    failing = [j for j in jobs if isinstance(j, dict) and j.get("state", {}).get("consecutiveErrors", 0) > 0]
    diag["cron_total"] = cron_summary.get("jobs_total", len(jobs))
    diag["cron_failing"] = len(failing)
    diag["cron_failing_names"] = [j["name"] for j in failing]
    diag["cron_summary"] = cron_summary
    diag["interactions_api"] = _check_interactions_api()
    diag["gateway_check"] = _check_gateway(
        cfg,
        primary_channel=PRIMARY_CHANNEL_DEFAULT,
        channels_payload=channels_payload,
        maintenance_payload=maintenance_payload,
        maintenance_age_minutes=maintenance_age_minutes,
    )
    diag["channel_check"] = _check_channels(
        cfg,
        primary_channel=PRIMARY_CHANNEL_DEFAULT,
        channels_payload=channels_payload,
        maintenance_payload=maintenance_payload,
        maintenance_age_minutes=maintenance_age_minutes,
    )
    diag["binding_check"] = _check_bindings(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT)

    if as_json:
        print(json.dumps(diag, indent=2))
    else:
        for key, value in diag.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    return 0


def cmd_cron_health(as_json: bool = False) -> int:
    check = _check_cron_jobs()
    payload, _ = _load_cron_jobs_payload()
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
    validation_lookup = {
        row.get("index"): row
        for row in (check.get("job_rows", []) if isinstance(check.get("job_rows"), list) else [])
        if isinstance(row, dict)
    }
    rows = []
    for index, j in enumerate(jobs):
        validation_row = validation_lookup.get(index) or {}
        if not isinstance(j, dict):
            rows.append(
                {
                    "name": validation_row.get("name") or f"job-{index}",
                    "agent": validation_row.get("agentId", "?"),
                    "enabled": False,
                    "schedule": "?",
                    "consecutive_errors": 0,
                    "last_status": "invalid",
                    "last_error": "job_not_object",
                    "delivery_channel": "?",
                    "has_fallback": False,
                    "fallback_channel": "",
                    "payload_kind": validation_row.get("payload_kind") or "",
                    "session_target": validation_row.get("sessionTarget"),
                    "validation_status": validation_row.get("status", "FAIL"),
                    "validation_issues": validation_row.get("issues", ["job_not_object"]),
                }
            )
            continue
        state = j.get("state", {})
        delivery = j.get("delivery", {})
        rows.append(
            {
                "name": j.get("name", f"job-{index}"),
                "agent": j.get("agentId", "?"),
                "enabled": j.get("enabled", False),
                "schedule": j.get("schedule", {}).get("expr", "?"),
                "consecutive_errors": state.get("consecutiveErrors", 0),
                "last_status": state.get("lastStatus", "?"),
                "last_error": state.get("lastError", "")[:60],
                "delivery_channel": delivery.get("channel", "?"),
                "has_fallback": "fallback" in delivery,
                "fallback_channel": delivery.get("fallback", {}).get("channel", ""),
                "payload_kind": j.get("payload", {}).get("kind", "") if isinstance(j.get("payload"), dict) else "",
                "session_target": j.get("sessionTarget"),
                "validation_status": validation_row.get("status", "OK") if validation_row else "OK",
                "validation_issues": validation_row.get("issues", []) if validation_row else [],
            }
        )

    result = {"summary": check, "jobs": rows, "invalid_jobs": check.get("invalid_jobs", [])}
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n[cron-health] {check['detail']}")
    return 0 if check["status"] == "OK" else (1 if check["status"] == "WARN" else 2)


def cmd_channel_test(as_json: bool = False) -> int:
    cfg = _load_config()
    target = _infer_whatsapp_target(cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = f"[PMX WhatsApp Remote Dev Test] {timestamp}"

    result: Dict[str, Any] = {
        "primary_channel": PRIMARY_CHANNEL_DEFAULT,
        "target_masked": _mask_target(target),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if PRIMARY_CHANNEL_DEFAULT != "whatsapp":
        result["status"] = "SKIP"
        result["detail"] = f"Primary channel is {PRIMARY_CHANNEL_DEFAULT}, not whatsapp"
        if as_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[channel-test] {result['detail']}")
        return 1

    if not target:
        result["status"] = "FAIL"
        result["detail"] = "Unable to infer a WhatsApp target"
        if as_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[channel-test] {result['detail']}")
        return 2

    ok, returncode, detail = _send_openclaw_message(channel="whatsapp", to=target, message=message)
    result["status"] = "OK" if ok else "FAIL"
    result["returncode"] = returncode
    result["detail"] = detail[:200]

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[channel-test] {result['status']} target={result['target_masked']} rc={returncode}")
    return 0 if ok else 1


def cmd_bridge_test(as_json: bool = False) -> int:
    cfg = _load_config()
    target = _infer_whatsapp_target(cfg)
    message = "status"
    result: Dict[str, Any] = {
        "primary_channel": PRIMARY_CHANNEL_DEFAULT,
        "target_masked": _mask_target(target),
        "message": message,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if PRIMARY_CHANNEL_DEFAULT != "whatsapp":
        result["status"] = "SKIP"
        result["detail"] = f"Primary channel is {PRIMARY_CHANNEL_DEFAULT}, not whatsapp"
        if as_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[bridge-test] {result['detail']}")
        return 1

    if not target:
        result["status"] = "FAIL"
        result["detail"] = "Unable to infer a WhatsApp target"
        if as_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[bridge-test] {result['detail']}")
        return 2

    rc, out, err = _run_bridge_test(channel="whatsapp", reply_to=target, message=message)
    result["status"] = "OK" if rc == 0 else "FAIL"
    result["returncode"] = rc
    result["stdout_preview"] = out[:240]
    result["stderr_preview"] = err[:240]

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[bridge-test] {result['status']} target={result['target_masked']} rc={rc}")
    return 0 if rc == 0 else 1


def cmd_gateway_restart(as_json: bool = False) -> int:
    rc, out, err = _run_openclaw(["gateway", "restart"], timeout=45.0)
    if rc != 0:
        result = {"status": "FAIL", "returncode": rc, "stdout": out[:200], "stderr": err[:200]}
        if as_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[gateway-restart] FAIL rc={rc}")
        return 2

    for attempt in range(1, 7):
        time.sleep(3)
        ok, detail = _gateway_local_ping()
        if ok:
            result = {"status": "OK", "attempts": attempt, "detail": detail[:120]}
            if as_json:
                print(json.dumps(result, indent=2))
            else:
                print(f"[gateway-restart] Gateway OK after {attempt * 3}s")
            return 0

    result = {"status": "FAIL", "detail": "gateway did not respond after restart"}
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print("[gateway-restart] FAIL")
    return 2


def cmd_failover_test(as_json: bool = False) -> int:
    jobs = _load_cron_jobs()
    wa_jobs = [j for j in jobs if j.get("delivery", {}).get("channel") == "whatsapp" and j.get("enabled")]
    summary = summarize_cron_jobs({"jobs": jobs})
    valid_fallback_jobs = summary.get("fallback_ready_jobs") if isinstance(summary.get("fallback_ready_jobs"), list) else []
    valid_fallback_names = {
        str(job.get("name") or "").strip()
        for job in valid_fallback_jobs
        if isinstance(job, dict) and str(job.get("name") or "").strip()
    }
    with_fallback = [j for j in wa_jobs if str(j.get("name") or "").strip() in valid_fallback_names]
    without_fallback = [j for j in wa_jobs if str(j.get("name") or "").strip() not in valid_fallback_names]

    result = {
        "check": "failover_config",
        "whatsapp_primary_jobs": len(wa_jobs),
        "with_telegram_fallback": len(with_fallback),
        "without_fallback": len(without_fallback),
        "missing_fallback": [j["name"] for j in without_fallback],
        "delivery_fallback_invalid_count": int(summary.get("delivery_fallback_invalid_count", 0) or 0),
        "status": "OK" if not without_fallback and int(summary.get("delivery_fallback_invalid_count", 0) or 0) == 0 else "WARN",
    }
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[failover-test] {result['status']}")
    return 0 if result["status"] == "OK" else 1


def main(argv: Optional[List[str]] = None) -> int:
    _bootstrap_dotenv()
    parser = argparse.ArgumentParser(description="OpenClaw remote workflow manager")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=[
            "status",
            "health",
            "diagnose",
            "channel-test",
            "bridge-test",
            "gateway-restart",
            "cron-health",
            "failover-test",
        ],
        help="Command to run (default: status)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    dispatch = {
        "status": cmd_status,
        "health": cmd_health,
        "diagnose": cmd_diagnose,
        "channel-test": cmd_channel_test,
        "bridge-test": cmd_bridge_test,
        "gateway-restart": cmd_gateway_restart,
        "cron-health": cmd_cron_health,
        "failover-test": cmd_failover_test,
    }
    return dispatch[args.command](as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
