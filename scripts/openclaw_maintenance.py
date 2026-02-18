#!/usr/bin/env python3
"""
OpenClaw maintenance guard for resilient messaging operations.

Actions:
- Cleans stale OpenClaw session lock files (optionally archives stale session jsonl files).
- Checks gateway RPC health and restarts gateway when unhealthy.
- Optionally disables persistently broken non-primary channels to reduce noisy failures.

The script is safe by default (dry-run). Use --apply to perform mutations.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_dotenv() -> None:
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        return


def _split_command(command: str) -> list[str]:
    try:
        from utils.openclaw_cli import _split_command as _split  # type: ignore

        return _split(command)
    except Exception:
        raw = (command or "").strip()
        return [raw or "openclaw"]


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


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _derive_status(errors: list[str]) -> str:
    if not errors:
        return "PASS"
    if any(str(err).startswith("primary_channel_unresolved:") for err in errors):
        return "FAIL"
    return "WARN"


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


@dataclass(frozen=True)
class _CmdResult:
    ok: bool
    returncode: int
    command: list[str]
    stdout: str
    stderr: str


def _run_openclaw(
    *,
    oc_base: list[str],
    args: list[str],
    timeout_seconds: float = 20.0,
) -> _CmdResult:
    cmd = [*oc_base, *args]
    env = dict(os.environ)
    env.setdefault("NODE_NO_WARNINGS", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(3.0, float(timeout_seconds)),
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        return _CmdResult(False, 127, cmd, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return _CmdResult(False, 124, cmd, out, err or "timeout")
    return _CmdResult(int(proc.returncode) == 0, int(proc.returncode), cmd, proc.stdout or "", proc.stderr or "")


def _run_openclaw_json(
    *,
    oc_base: list[str],
    args: list[str],
    timeout_seconds: float = 20.0,
) -> tuple[_CmdResult, Optional[Any]]:
    final_args = ["--no-color", *args]
    if "--json" not in final_args:
        final_args.append("--json")
    res = _run_openclaw(oc_base=oc_base, args=final_args, timeout_seconds=timeout_seconds)
    if not res.ok:
        return res, None
    try:
        payload = _parse_json_best_effort(res.stdout)
    except Exception:
        return res, None
    return res, payload


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(minimum), int(value))


def _resolve_primary_account_id(channels_payload: dict[str, Any], primary_channel: str) -> str:
    channel_defaults = (
        channels_payload.get("channelDefaultAccountId")
        if isinstance(channels_payload.get("channelDefaultAccountId"), dict)
        else {}
    )
    default_account = str(channel_defaults.get(primary_channel) or "").strip()
    if default_account:
        return default_account

    channel_accounts = (
        channels_payload.get("channelAccounts")
        if isinstance(channels_payload.get("channelAccounts"), dict)
        else {}
    )
    rows = channel_accounts.get(primary_channel) if isinstance(channel_accounts, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            account_id = str(row.get("accountId") or "").strip()
            if account_id:
                return account_id
    return "default"


def _primary_channel_snapshot(channels_payload: dict[str, Any], primary_channel: str) -> dict[str, Any]:
    channels = channels_payload.get("channels") if isinstance(channels_payload.get("channels"), dict) else {}
    row = channels.get(primary_channel) if isinstance(channels, dict) else None
    row = row if isinstance(row, dict) else {}

    channel_accounts = (
        channels_payload.get("channelAccounts")
        if isinstance(channels_payload.get("channelAccounts"), dict)
        else {}
    )
    account_id = _resolve_primary_account_id(channels_payload, primary_channel)
    account_row: dict[str, Any] = {}
    account_rows = channel_accounts.get(primary_channel) if isinstance(channel_accounts, dict) else None
    if isinstance(account_rows, list):
        for item in account_rows:
            if not isinstance(item, dict):
                continue
            if str(item.get("accountId") or "").strip() == account_id:
                account_row = item
                break
        if not account_row and account_rows and isinstance(account_rows[0], dict):
            account_row = account_rows[0]
            account_id = str(account_row.get("accountId") or account_id).strip() or account_id

    account_last_disconnect = (
        account_row.get("lastDisconnect") if isinstance(account_row.get("lastDisconnect"), dict) else {}
    )
    channel_last_disconnect = row.get("lastDisconnect") if isinstance(row.get("lastDisconnect"), dict) else {}
    disconnect = account_last_disconnect if account_last_disconnect else channel_last_disconnect

    return {
        "configured": bool(row.get("configured")),
        "linked": bool(row.get("linked")) if "linked" in row else bool(account_row.get("linked")),
        "running": bool(row.get("running")),
        "connected": bool(row.get("connected", True)),
        "last_error": str(row.get("lastError") or ""),
        "account_id": account_id,
        "account_enabled": bool(account_row.get("enabled")) if account_row else None,
        "account_running": bool(account_row.get("running")) if account_row else None,
        "account_connected": bool(account_row.get("connected", True)) if account_row else None,
        "account_last_error": str(account_row.get("lastError") or "") if account_row else "",
        "disconnect_status": int(disconnect.get("status")) if str(disconnect.get("status") or "").isdigit() else None,
        "disconnect_error": str(disconnect.get("error") or ""),
        "disconnect_logged_out": bool(disconnect.get("loggedOut")) if disconnect else False,
    }


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        pass

    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            out = (proc.stdout or "").lower()
            return str(pid) in out and "no tasks are running" not in out
        except Exception:
            return False
    return False


def _archive_path(path: Path, suffix: str) -> Path:
    ts = _utc_now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}.stale.{ts}.{suffix}")


def _cleanup_stale_session_locks(
    *,
    apply: bool,
    session_stale_seconds: int,
) -> dict[str, Any]:
    agents_root = Path.home() / ".openclaw" / "agents"
    result: dict[str, Any] = {
        "agents_root": str(agents_root),
        "lock_files_scanned": 0,
        "locks_stale_found": 0,
        "locks_archived": 0,
        "sessions_archived": 0,
        "errors": [],
    }
    if not agents_root.exists():
        return result

    now = _utc_now()
    for sessions_dir in agents_root.glob("*/sessions"):
        if not sessions_dir.is_dir():
            continue
        for lock_path in sessions_dir.glob("*.jsonl.lock"):
            if ".stale." in lock_path.name:
                continue
            result["lock_files_scanned"] += 1
            lock_text = ""
            try:
                lock_text = lock_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                result["errors"].append(f"read_lock_failed:{lock_path.name}:{exc}")
                continue

            lock_payload: dict[str, Any] = {}
            try:
                parsed = json.loads(lock_text)
                if isinstance(parsed, dict):
                    lock_payload = parsed
            except Exception:
                lock_payload = {}

            pid = 0
            try:
                pid = int(lock_payload.get("pid") or 0)
            except Exception:
                pid = 0

            created_at = _parse_ts(lock_payload.get("createdAt"))
            age_seconds = None
            if created_at is not None:
                age_seconds = max(0, int((now - created_at).total_seconds()))

            stale = False
            if pid > 0 and not _pid_running(pid):
                stale = True
            elif pid <= 0 and age_seconds is not None and age_seconds >= max(60, int(session_stale_seconds)):
                stale = True

            if not stale:
                continue

            result["locks_stale_found"] += 1
            if not apply:
                continue

            try:
                lock_archive = _archive_path(lock_path, "lock")
                shutil.move(str(lock_path), str(lock_archive))
                result["locks_archived"] += 1
            except Exception as exc:
                result["errors"].append(f"archive_lock_failed:{lock_path.name}:{exc}")
                continue

            # Archive matching session file if present and old enough.
            session_path = lock_path.with_suffix("")
            if session_path.exists() and session_path.name.endswith(".jsonl"):
                should_archive_session = age_seconds is None or age_seconds >= max(300, int(session_stale_seconds))
                if should_archive_session:
                    try:
                        session_archive = _archive_path(session_path, "jsonl")
                        shutil.move(str(session_path), str(session_archive))
                        result["sessions_archived"] += 1
                    except Exception as exc:
                        result["errors"].append(f"archive_session_failed:{session_path.name}:{exc}")

    return result


def _channel_disable_candidates(
    channels_payload: dict[str, Any],
    *,
    primary_channel: str,
) -> list[tuple[str, str]]:
    channels = channels_payload.get("channels") if isinstance(channels_payload.get("channels"), dict) else {}
    out: list[tuple[str, str]] = []
    primary = str(primary_channel or "").strip().lower()
    for channel in ("telegram", "discord"):
        if channel == primary:
            continue
        row = channels.get(channel) if isinstance(channels, dict) else None
        if not isinstance(row, dict):
            continue
        configured = bool(row.get("configured"))
        running = bool(row.get("running"))
        if not configured or running:
            continue
        last_error = str(row.get("lastError") or "").strip()
        low = last_error.lower()
        error_match = False
        if channel == "telegram":
            error_match = any(tok in low for tok in ("404", "not found", "unauthorized", "token", "getme"))
        elif channel == "discord":
            error_match = any(tok in low for tok in ("not configured", "resolve discord application id", "401", "token"))
        if error_match:
            out.append((channel, last_error or "unknown_error"))
    return out


def _disable_broken_channels(
    *,
    oc_base: list[str],
    channels_payload: dict[str, Any],
    primary_channel: str,
    apply: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "primary_channel": primary_channel,
        "candidates": [],
        "disabled": [],
        "errors": [],
    }
    candidates = _channel_disable_candidates(channels_payload, primary_channel=primary_channel)
    result["candidates"] = [{"channel": c, "reason": r} for c, r in candidates]
    if not candidates or not apply:
        return result

    channel_accounts = (
        channels_payload.get("channelAccounts")
        if isinstance(channels_payload.get("channelAccounts"), dict)
        else {}
    )
    for channel, _reason in candidates:
        set_res = _run_openclaw(
            oc_base=oc_base,
            args=["--no-color", "config", "set", f"channels.{channel}.enabled", "false", "--json"],
            timeout_seconds=15.0,
        )
        if not set_res.ok:
            result["errors"].append(f"disable_channel_failed:{channel}:rc={set_res.returncode}")
            continue

        accounts = channel_accounts.get(channel) if isinstance(channel_accounts, dict) else None
        if isinstance(accounts, list):
            for row in accounts:
                if not isinstance(row, dict):
                    continue
                account_id = str(row.get("accountId") or "").strip()
                if not account_id:
                    continue
                _run_openclaw(
                    oc_base=oc_base,
                    args=[
                        "--no-color",
                        "config",
                        "set",
                        f"channels.{channel}.accounts.{account_id}.enabled",
                        "false",
                        "--json",
                    ],
                    timeout_seconds=12.0,
                )
        result["disabled"].append(channel)

    return result


def _detect_primary_channel_issue(channels_payload: dict[str, Any], primary_channel: str) -> Optional[str]:
    snapshot = _primary_channel_snapshot(channels_payload, primary_channel)
    if not snapshot.get("configured"):
        return None

    combined_last_error = " ".join(
        [
            str(snapshot.get("last_error") or ""),
            str(snapshot.get("account_last_error") or ""),
            str(snapshot.get("disconnect_error") or ""),
        ]
    ).strip()
    low = combined_last_error.lower()

    if primary_channel == "whatsapp":
        if snapshot.get("disconnect_logged_out") or "logged out" in low:
            return "whatsapp_session_logged_out"
        if any(tok in low for tok in ("enotfound", "getaddrinfo")) and "web.whatsapp.com" in low:
            return "whatsapp_dns_resolution_failed"
        if int(snapshot.get("disconnect_status") or 0) == 428:
            return "whatsapp_handshake_timeout"
        running = bool(snapshot.get("running"))
        connected = bool(snapshot.get("connected", True))
        if (not running) or (not connected):
            if any(tok in low for tok in ("handshake", "request time-out", "status=408", "connection was lost")):
                return "whatsapp_handshake_timeout"
            return "whatsapp_channel_down"
        return None

    running = bool(snapshot.get("running"))
    if not running:
        return f"{primary_channel}_channel_down"
    return None


def _set_channel_account_enabled(
    *,
    oc_base: list[str],
    channel: str,
    account_id: str,
    enabled: bool,
) -> _CmdResult:
    return _run_openclaw(
        oc_base=oc_base,
        args=[
            "--no-color",
            "config",
            "set",
            f"channels.{channel}.accounts.{account_id}.enabled",
            "true" if enabled else "false",
            "--json",
        ],
        timeout_seconds=15.0,
    )


def _set_channel_enabled(
    *,
    oc_base: list[str],
    channel: str,
    enabled: bool,
) -> _CmdResult:
    return _run_openclaw(
        oc_base=oc_base,
        args=[
            "--no-color",
            "config",
            "set",
            f"channels.{channel}.enabled",
            "true" if enabled else "false",
            "--json",
        ],
        timeout_seconds=15.0,
    )


def _resolve_dns(hostname: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": hostname,
        "ok": False,
        "addresses": [],
        "error": "",
    }
    try:
        infos = socket.getaddrinfo(str(hostname), None)
        addresses = sorted(
            {
                str(row[4][0])
                for row in infos
                if isinstance(row, tuple)
                and len(row) >= 5
                and isinstance(row[4], tuple)
                and len(row[4]) >= 1
                and row[4][0]
            }
        )
        result["addresses"] = addresses[:8]
        result["ok"] = bool(addresses)
        if not addresses:
            result["error"] = "no_addresses"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _fetch_channel_logs_tail(
    *,
    oc_base: list[str],
    channel: str,
    lines: int = 40,
) -> dict[str, Any]:
    line_count = max(10, int(lines))
    res = _run_openclaw(
        oc_base=oc_base,
        args=[
            "--no-color",
            "channels",
            "logs",
            "--channel",
            str(channel),
            "--lines",
            str(line_count),
        ],
        timeout_seconds=20.0,
    )
    if not res.ok:
        return {"ok": False, "error": f"channels_logs_failed:rc={res.returncode}"}
    raw = (res.stdout or "").splitlines()
    return {"ok": True, "lines": raw[-line_count:]}


def _recheck_channels_status(
    *,
    oc_base: list[str],
    wait_seconds: float,
) -> tuple[_CmdResult, Optional[dict[str, Any]]]:
    if wait_seconds > 0:
        time.sleep(max(0.0, float(wait_seconds)))
    res, payload = _run_openclaw_json(oc_base=oc_base, args=["channels", "status"], timeout_seconds=20.0)
    return res, payload if isinstance(payload, dict) else None


def _gateway_health_and_heal(
    *,
    oc_base: list[str],
    channels_payload: dict[str, Any],
    primary_channel: str,
    apply: bool,
    restart_on_rpc_failure: bool,
    recheck_delay_seconds: float,
    attempt_primary_reenable: bool,
    primary_restart_attempts: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rpc_ok": None,
        "service_status": None,
        "service_state": None,
        "dns": None,
        "primary_channel_issue": None,
        "primary_channel_issue_after_restart": None,
        "primary_channel_issue_final": None,
        "primary_channel_status_before": _primary_channel_snapshot(channels_payload, primary_channel),
        "primary_channel_status_after_restart": None,
        "primary_channel_status_final": None,
        "actions": [],
        "manual_actions": [],
        "warnings": [],
        "errors": [],
    }
    _, gateway_payload = _run_openclaw_json(oc_base=oc_base, args=["gateway", "status"], timeout_seconds=20.0)
    if not isinstance(gateway_payload, dict):
        out["errors"].append("gateway_status_unavailable")
        return out

    rpc = gateway_payload.get("rpc") if isinstance(gateway_payload.get("rpc"), dict) else {}
    service = gateway_payload.get("service") if isinstance(gateway_payload.get("service"), dict) else {}
    runtime = service.get("runtime") if isinstance(service.get("runtime"), dict) else {}
    out["rpc_ok"] = bool(rpc.get("ok"))
    out["service_status"] = str(runtime.get("status") or "")
    out["service_state"] = str(runtime.get("state") or "")

    primary_issue = _detect_primary_channel_issue(channels_payload, primary_channel)
    out["primary_channel_issue"] = primary_issue
    if primary_channel == "whatsapp":
        out["dns"] = _resolve_dns("web.whatsapp.com")

    if bool(out["rpc_ok"]) and str(out["service_status"]).strip().lower() in {"stopped", "ready"}:
        out["warnings"].append("gateway_service_runtime_mismatch")

    dns_diag = out.get("dns") if isinstance(out.get("dns"), dict) else {}
    skip_restart_for_dns = bool(
        primary_issue == "whatsapp_dns_resolution_failed"
        and isinstance(dns_diag, dict)
        and not bool(dns_diag.get("ok"))
    )
    should_restart = (
        bool(primary_issue) or ((out["rpc_ok"] is False) and bool(restart_on_rpc_failure))
    ) and not skip_restart_for_dns
    if skip_restart_for_dns:
        out["warnings"].append("skip_restart_due_to_dns_failure")

    restart_ok = False
    issue_after_restart = primary_issue
    channels_after_restart: dict[str, Any] | None = None
    restart_attempt_budget = 1
    if primary_issue:
        restart_attempt_budget = max(1, int(primary_restart_attempts))

    if should_restart and apply:
        for attempt in range(1, restart_attempt_budget + 1):
            restart = _run_openclaw(oc_base=oc_base, args=["gateway", "restart"], timeout_seconds=45.0)
            if not restart.ok:
                out["warnings"].append(f"gateway_restart_attempt_failed:{attempt}:rc={restart.returncode}")
                continue

            restart_ok = True
            out["actions"].append(
                "gateway_restart_primary_channel_recovery"
                if primary_issue
                else "gateway_restart"
            )
            if restart_attempt_budget > 1:
                out["actions"].append(f"gateway_restart_attempt:{attempt}")

            if not primary_issue:
                break

            _status_res, channels_after_restart = _recheck_channels_status(
                oc_base=oc_base,
                wait_seconds=recheck_delay_seconds,
            )
            if isinstance(channels_after_restart, dict):
                issue_after_restart = _detect_primary_channel_issue(channels_after_restart, primary_channel)
                out["primary_channel_issue_after_restart"] = issue_after_restart
                out["primary_channel_status_after_restart"] = _primary_channel_snapshot(channels_after_restart, primary_channel)
            else:
                out["warnings"].append(f"channels_status_unavailable_after_restart_attempt:{attempt}")

            if not issue_after_restart:
                break
            if issue_after_restart == "whatsapp_dns_resolution_failed":
                break
            if attempt < restart_attempt_budget:
                out["warnings"].append(
                    f"primary_issue_persist_after_restart_attempt:{attempt}:{issue_after_restart}"
                )
    elif should_restart and not apply:
        out["warnings"].append("dry_run_restart_required")

    if primary_issue and restart_ok and out["primary_channel_status_after_restart"] is None:
        _status_res, channels_after_restart = _recheck_channels_status(
            oc_base=oc_base,
            wait_seconds=max(1.0, recheck_delay_seconds),
        )
        if isinstance(channels_after_restart, dict):
            issue_after_restart = _detect_primary_channel_issue(channels_after_restart, primary_channel)
            out["primary_channel_issue_after_restart"] = issue_after_restart
            out["primary_channel_status_after_restart"] = _primary_channel_snapshot(channels_after_restart, primary_channel)
        else:
            out["warnings"].append("channels_status_unavailable_after_restart")

    final_channels_payload = channels_after_restart if isinstance(channels_after_restart, dict) else channels_payload
    final_issue = issue_after_restart if restart_ok else primary_issue

    if (
        bool(apply)
        and bool(attempt_primary_reenable)
        and bool(final_issue)
        and primary_channel == "whatsapp"
        and final_issue != "whatsapp_session_logged_out"
        and final_issue != "whatsapp_dns_resolution_failed"
    ):
        snapshot = _primary_channel_snapshot(final_channels_payload, primary_channel)
        account_id = str(snapshot.get("account_id") or "default").strip() or "default"
        if snapshot.get("configured") and snapshot.get("linked"):
            disable_res = _set_channel_account_enabled(
                oc_base=oc_base,
                channel=primary_channel,
                account_id=account_id,
                enabled=False,
            )
            if not disable_res.ok:
                out["warnings"].append(f"primary_account_disable_failed:{account_id}:rc={disable_res.returncode}")
            else:
                out["actions"].append(f"primary_account_disabled:{account_id}")

            enable_res = _set_channel_account_enabled(
                oc_base=oc_base,
                channel=primary_channel,
                account_id=account_id,
                enabled=True,
            )
            if not enable_res.ok:
                out["warnings"].append(f"primary_account_enable_failed:{account_id}:rc={enable_res.returncode}")
            else:
                out["actions"].append(f"primary_account_enabled:{account_id}")

            _set_channel_enabled(oc_base=oc_base, channel=primary_channel, enabled=True)

            if disable_res.ok and enable_res.ok:
                restart = _run_openclaw(oc_base=oc_base, args=["gateway", "restart"], timeout_seconds=45.0)
                if restart.ok:
                    out["actions"].append("gateway_restart_after_primary_reenable")
                else:
                    out["warnings"].append(f"gateway_restart_after_reenable_failed:rc={restart.returncode}")

                _status_res, channels_after_reenable = _recheck_channels_status(
                    oc_base=oc_base,
                    wait_seconds=recheck_delay_seconds,
                )
                if isinstance(channels_after_reenable, dict):
                    final_channels_payload = channels_after_reenable
                    final_issue = _detect_primary_channel_issue(channels_after_reenable, primary_channel)
                else:
                    out["warnings"].append("channels_status_unavailable_after_primary_reenable")

    out["primary_channel_issue_final"] = final_issue
    out["primary_channel_status_final"] = _primary_channel_snapshot(final_channels_payload, primary_channel)
    if final_issue:
        out["errors"].append(f"primary_channel_unresolved:{final_issue}")
        logs_tail = _fetch_channel_logs_tail(oc_base=oc_base, channel=primary_channel, lines=40)
        if bool(logs_tail.get("ok")):
            out["channel_logs_tail"] = logs_tail.get("lines", [])
        else:
            out["warnings"].append(str(logs_tail.get("error") or "channels_logs_unavailable"))
        if final_issue == "whatsapp_dns_resolution_failed":
            out["manual_actions"].append("verify_dns_resolution:web.whatsapp.com")
            out["manual_actions"].append("check_network_firewall_or_proxy_for_whatsapp_web")
        elif final_issue == "whatsapp_session_logged_out":
            out["manual_actions"].append(
                "run: openclaw channels login --channel whatsapp --account default --verbose"
            )
        elif final_issue in {"whatsapp_channel_down", "whatsapp_handshake_timeout"}:
            out["manual_actions"].append(
                "run: openclaw channels login --channel whatsapp --account default --verbose"
            )
            out["manual_actions"].append("run: openclaw channels logs --channel whatsapp --lines 200")
    return out


def main(argv: list[str]) -> int:
    _bootstrap_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run/report only.")
    parser.add_argument(
        "--disable-broken-channels",
        action="store_true",
        help="Disable broken non-primary channels (Telegram/Discord) when known auth/config errors are detected.",
    )
    parser.add_argument(
        "--primary-channel",
        default=os.getenv("OPENCLAW_CHANNEL", "whatsapp"),
        help="Primary channel to preserve while evaluating disable candidates.",
    )
    parser.add_argument(
        "--restart-gateway-on-rpc-failure",
        action="store_true",
        help="Restart gateway when rpc.ok is false.",
    )
    parser.add_argument(
        "--recheck-delay-seconds",
        type=float,
        default=8.0,
        help="Delay before post-heal channel status recheck.",
    )
    parser.add_argument(
        "--primary-restart-attempts",
        type=int,
        default=_env_int("OPENCLAW_PRIMARY_RESTART_ATTEMPTS", 2, minimum=1),
        help="How many gateway restart attempts to run for primary channel recovery.",
    )
    parser.add_argument(
        "--attempt-primary-reenable",
        dest="attempt_primary_reenable",
        action="store_true",
        default=_as_bool(os.getenv("OPENCLAW_ATTEMPT_PRIMARY_REENABLE", "1"), default=True),
        help="When primary channel stays down, toggle primary account enabled=false/true before final recheck.",
    )
    parser.add_argument(
        "--no-attempt-primary-reenable",
        dest="attempt_primary_reenable",
        action="store_false",
        help="Disable account toggle remediation when the primary channel stays down.",
    )
    parser.add_argument(
        "--session-stale-seconds",
        type=int,
        default=7200,
        help="Age threshold for stale lock/session archival decisions.",
    )
    parser.add_argument(
        "--report-file",
        default="logs/automation/openclaw_maintenance_latest.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--command",
        default=os.getenv("OPENCLAW_COMMAND", "openclaw"),
        help="OpenClaw command. Example: openclaw or wsl openclaw",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when unresolved health/channel errors remain.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously in watch mode (loop every --watch-interval seconds).",
    )
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=int(os.getenv("OPENCLAW_WATCH_INTERVAL_SECONDS", "900")),
        help="Seconds between watch cycles (default: 900 = 15 minutes).",
    )
    args = parser.parse_args(argv)

    report_file = Path(str(args.report_file)).expanduser()
    if not report_file.is_absolute():
        report_file = (PROJECT_ROOT / report_file).resolve()

    oc_base = _split_command(str(args.command))

    report: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "apply": bool(args.apply),
        "primary_channel": str(args.primary_channel or "whatsapp").strip().lower() or "whatsapp",
        "command": oc_base,
        "status": "PASS",
        "steps": {},
        "warnings": [],
        "errors": [],
    }

    lock_step = _cleanup_stale_session_locks(
        apply=bool(args.apply),
        session_stale_seconds=max(60, int(args.session_stale_seconds)),
    )
    report["steps"]["stale_session_cleanup"] = lock_step

    _status_res, channels_payload = _run_openclaw_json(oc_base=oc_base, args=["channels", "status"], timeout_seconds=20.0)
    if isinstance(channels_payload, dict):
        report["steps"]["channels_status_snapshot"] = {
            "timestamp_ms": channels_payload.get("ts"),
            "channels": channels_payload.get("channels"),
        }
    else:
        report["warnings"].append("channels_status_unavailable")

    gateway_step = _gateway_health_and_heal(
        oc_base=oc_base,
        channels_payload=channels_payload if isinstance(channels_payload, dict) else {},
        primary_channel=str(report["primary_channel"]),
        apply=bool(args.apply),
        restart_on_rpc_failure=bool(args.restart_gateway_on_rpc_failure),
        recheck_delay_seconds=max(0.0, float(args.recheck_delay_seconds)),
        attempt_primary_reenable=bool(args.attempt_primary_reenable),
        primary_restart_attempts=max(1, int(args.primary_restart_attempts)),
    )
    report["steps"]["gateway_health"] = gateway_step

    channel_step: dict[str, Any] = {
        "skipped": True,
        "reason": "disabled_by_flag",
        "primary_channel": report["primary_channel"],
    }
    if bool(args.disable_broken_channels):
        channel_step = _disable_broken_channels(
            oc_base=oc_base,
            channels_payload=channels_payload if isinstance(channels_payload, dict) else {},
            primary_channel=str(report["primary_channel"]),
            apply=bool(args.apply),
        )
    report["steps"]["broken_channel_disable"] = channel_step

    if gateway_step.get("errors"):
        report["errors"].extend([str(x) for x in gateway_step.get("errors", [])])
    if gateway_step.get("warnings"):
        report["warnings"].extend([str(x) for x in gateway_step.get("warnings", [])])
    if isinstance(channel_step, dict) and channel_step.get("errors"):
        report["errors"].extend([str(x) for x in channel_step.get("errors", [])])
    if lock_step.get("errors"):
        report["warnings"].extend([str(x) for x in lock_step.get("errors", [])])

    report["status"] = _derive_status([str(x) for x in report["errors"]])

    _safe_write_json(report_file, report)
    print(f"[openclaw_maintenance] status={report['status']} apply={args.apply}")
    print(f"[openclaw_maintenance] report={report_file}")
    if isinstance(channel_step, dict):
        disabled = channel_step.get("disabled") if isinstance(channel_step.get("disabled"), list) else []
        if disabled:
            print(f"[openclaw_maintenance] disabled_channels={','.join(str(x) for x in disabled)}")

    if args.strict and report["errors"]:
        if not args.watch:
            return 1

    if not args.watch:
        return 0

    # --- Watch mode: loop continuously ---
    interval = max(30, int(args.watch_interval))
    print(f"[openclaw_maintenance] Watch mode active (interval={interval}s). Press Ctrl+C to stop.")
    cycle = 1
    while True:
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n[openclaw_maintenance] Watch stopped after {cycle} cycles.")
            return 0

        cycle += 1
        report = {
            "timestamp_utc": _utc_now_iso(),
            "apply": bool(args.apply),
            "primary_channel": str(args.primary_channel or "whatsapp").strip().lower() or "whatsapp",
            "command": oc_base,
            "status": "PASS",
            "steps": {},
            "warnings": [],
            "errors": [],
            "watch_cycle": cycle,
        }

        lock_step = _cleanup_stale_session_locks(
            apply=bool(args.apply),
            session_stale_seconds=max(60, int(args.session_stale_seconds)),
        )
        report["steps"]["stale_session_cleanup"] = lock_step

        _status_res, channels_payload = _run_openclaw_json(
            oc_base=oc_base, args=["channels", "status"], timeout_seconds=20.0,
        )
        if isinstance(channels_payload, dict):
            report["steps"]["channels_status_snapshot"] = {
                "timestamp_ms": channels_payload.get("ts"),
                "channels": channels_payload.get("channels"),
            }
        else:
            report["warnings"].append("channels_status_unavailable")

        gateway_step = _gateway_health_and_heal(
            oc_base=oc_base,
            channels_payload=channels_payload if isinstance(channels_payload, dict) else {},
            primary_channel=str(report["primary_channel"]),
            apply=bool(args.apply),
            restart_on_rpc_failure=bool(args.restart_gateway_on_rpc_failure),
            recheck_delay_seconds=max(0.0, float(args.recheck_delay_seconds)),
            attempt_primary_reenable=bool(args.attempt_primary_reenable),
            primary_restart_attempts=max(1, int(args.primary_restart_attempts)),
        )
        report["steps"]["gateway_health"] = gateway_step

        if gateway_step.get("errors"):
            report["errors"].extend([str(x) for x in gateway_step.get("errors", [])])
        if gateway_step.get("warnings"):
            report["warnings"].extend([str(x) for x in gateway_step.get("warnings", [])])

        report["status"] = _derive_status([str(x) for x in report["errors"]])
        _safe_write_json(report_file, report)
        print(
            f"[openclaw_maintenance] watch cycle={cycle} status={report['status']} "
            f"errors={len(report['errors'])} warnings={len(report['warnings'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
