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

OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
CRON_JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"
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


def _load_cron_jobs() -> List[Dict[str, Any]]:
    if not CRON_JOBS_PATH.exists():
        return []
    try:
        return json.loads(CRON_JOBS_PATH.read_text(encoding="utf-8")).get("jobs", [])
    except Exception:
        return []


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


def _load_live_channels_payload(timeout: float = 12.0) -> Optional[Dict[str, Any]]:
    rc, payload, _, _ = _run_openclaw_json(["channels", "status"], timeout=timeout)
    return payload if rc == 0 and isinstance(payload, dict) else None


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


def _check_gateway(cfg: Dict[str, Any], *, primary_channel: str, channels_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    gw = cfg.get("gateway", {})
    mode = str(gw.get("mode", "unknown") or "unknown")
    bind = str(gw.get("bind", "unknown") or "unknown")
    remote_cfg = gw.get("remote", {}) if isinstance(gw.get("remote"), dict) else {}
    reachable, ping_detail = _gateway_local_ping()
    primary_ready = _primary_channel_ready(channels_payload, primary_channel)

    issues: List[str] = []
    detail: str
    access_mode = "unknown"

    if not reachable:
        issues.append("local gateway unreachable")

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
        else:
            issues.append(f"{primary_channel} channel not ready")
            detail = f"Loopback gateway reachable, but {primary_channel} is not ready for remote dev"
    else:
        access_mode = f"{mode}:{bind}"
        if not primary_ready:
            issues.append(f"{primary_channel} channel not ready")
        detail = f"Gateway mode={mode} bind={bind}"

    status = "OK" if not issues else "FAIL"
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
        "detail": detail,
    }


def _check_channels(cfg: Dict[str, Any], *, primary_channel: str, channels_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    channel_names = ("whatsapp", "telegram", "discord")
    rows: Dict[str, Dict[str, Any]] = {}
    for name in channel_names:
        if isinstance(channels_payload, dict):
            rows[name] = _channel_ready_from_live(channels_payload, name)
        else:
            rows[name] = _channel_ready_from_config(cfg, name)

    primary_status = rows.get(primary_channel, {}).get("status", "OFF")
    fallback_ready = [name for name in channel_names if name != primary_channel and rows.get(name, {}).get("status") == "OK"]

    if primary_status == "OK":
        overall = "OK" if fallback_ready else "WARN"
    elif primary_status == "WARN":
        overall = "WARN"
    else:
        overall = "FAIL"

    detail = (
        f"Primary {primary_channel}={primary_status}; "
        f"fallback_ready={','.join(fallback_ready) if fallback_ready else 'none'}"
    )
    return {
        "check": "channels",
        "required": True,
        "status": overall,
        "primary_channel": primary_channel,
        "primary_status": primary_status,
        "fallback_ready": fallback_ready,
        "channels": rows,
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
    jobs = _load_cron_jobs()
    if not jobs:
        return {
            "check": "cron_jobs",
            "required": False,
            "status": "WARN",
            "detail": "No cron jobs found",
        }

    total = len(jobs)
    enabled = [j for j in jobs if j.get("enabled", False)]
    failing = [j for j in enabled if j.get("state", {}).get("consecutiveErrors", 0) > 0]
    delivery_failures = [
        j for j in failing if "delivery" in str(j.get("state", {}).get("lastError", "")).lower()
    ]
    with_fallback = [j for j in enabled if "fallback" in j.get("delivery", {})]

    status = "WARN" if failing else "OK"
    return {
        "check": "cron_jobs",
        "required": False,
        "status": status,
        "total": total,
        "enabled": len(enabled),
        "failing": len(failing),
        "delivery_failures": len(delivery_failures),
        "with_telegram_fallback": len(with_fallback),
        "failing_names": [j["name"] for j in failing],
        "detail": (
            f"{len(enabled)} enabled, {len(failing)} failing "
            f"({len(delivery_failures)} delivery), {len(with_fallback)} have fallback"
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
    channels_payload = _load_live_channels_payload()

    checks = [
        _check_version(cfg),
        _check_gateway(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT, channels_payload=channels_payload),
        _check_channels(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT, channels_payload=channels_payload),
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
    channels_payload = _load_live_channels_payload()
    gateway = _check_gateway(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT, channels_payload=channels_payload)
    channels = _check_channels(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT, channels_payload=channels_payload)
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
        "binding_ok": binding.get("status") == "OK",
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
    channels_payload = _load_live_channels_payload()
    diag: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(OPENCLAW_CONFIG),
        "config_exists": OPENCLAW_CONFIG.exists(),
        "openclaw_command": _openclaw_base(),
        "openclaw_cli": _openclaw_available(),
    }

    gw_ok, gw_detail = _gateway_local_ping()
    diag["gateway_local_reachable"] = gw_ok
    diag["gateway_ping_detail"] = gw_detail[:200]

    rc, out, err = _run_openclaw(["status"], timeout=12.0)
    diag["openclaw_status_rc"] = rc
    diag["openclaw_status_stdout"] = out[:500]
    diag["openclaw_status_stderr"] = err[:200]

    rc, payload, out, err = _run_openclaw_json(["channels", "status"], timeout=12.0)
    diag["channels_status_rc"] = rc
    diag["channels_status_parsed"] = isinstance(payload, dict)
    diag["channels_status_stdout"] = out[:500]
    diag["channels_status_stderr"] = err[:200]

    if isinstance(channels_payload, dict):
        diag["primary_channel_snapshot"] = _channel_ready_from_live(channels_payload, PRIMARY_CHANNEL_DEFAULT)

    jobs = _load_cron_jobs()
    failing = [j for j in jobs if j.get("state", {}).get("consecutiveErrors", 0) > 0]
    diag["cron_total"] = len(jobs)
    diag["cron_failing"] = len(failing)
    diag["cron_failing_names"] = [j["name"] for j in failing]
    diag["interactions_api"] = _check_interactions_api()
    diag["gateway_check"] = _check_gateway(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT, channels_payload=channels_payload)
    diag["channel_check"] = _check_channels(cfg, primary_channel=PRIMARY_CHANNEL_DEFAULT, channels_payload=channels_payload)
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
    jobs = _load_cron_jobs()
    check = _check_cron_jobs()
    rows = []
    for j in jobs:
        state = j.get("state", {})
        delivery = j.get("delivery", {})
        rows.append(
            {
                "name": j["name"],
                "agent": j.get("agentId", "?"),
                "enabled": j.get("enabled", False),
                "schedule": j.get("schedule", {}).get("expr", "?"),
                "consecutive_errors": state.get("consecutiveErrors", 0),
                "last_status": state.get("lastStatus", "?"),
                "last_error": state.get("lastError", "")[:60],
                "delivery_channel": delivery.get("channel", "?"),
                "has_fallback": "fallback" in delivery,
                "fallback_channel": delivery.get("fallback", {}).get("channel", ""),
            }
        )

    result = {"summary": check, "jobs": rows}
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n[cron-health] {check['detail']}")
    return 0 if check["status"] == "OK" else 1


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
    with_fallback = [j for j in wa_jobs if "fallback" in j.get("delivery", {})]
    without_fallback = [j for j in wa_jobs if "fallback" not in j.get("delivery", {})]

    result = {
        "check": "failover_config",
        "whatsapp_primary_jobs": len(wa_jobs),
        "with_telegram_fallback": len(with_fallback),
        "without_fallback": len(without_fallback),
        "missing_fallback": [j["name"] for j in without_fallback],
        "status": "OK" if not without_fallback else "WARN",
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
