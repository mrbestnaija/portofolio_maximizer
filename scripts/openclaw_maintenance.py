#!/usr/bin/env python3
"""
OpenClaw maintenance guard for resilient messaging operations.

Actions:
- Cleans stale OpenClaw session lock files (optionally archives stale session jsonl files).
- Ensures only one maintenance runner is active at a time via a shared lock file.
- Checks gateway RPC health and restarts gateway when unhealthy (with cooldown/backoff).
- Optionally disables persistently broken non-primary channels to reduce noisy failures.
- Surfaces WhatsApp delivery/session degradation signals from recent channel logs.

The script is safe by default (dry-run). Use --apply to perform mutations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import shutil
import subprocess
import sys
import time
import traceback
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


def _normalize_openclaw_command(command_parts: list[str]) -> tuple[list[str], str]:
    parts = [str(x or "").strip() for x in (command_parts or [])]
    parts = [x for x in parts if x]
    if not parts:
        return _split_command("openclaw"), "openclaw_command_empty_fallback"

    script_name = Path(__file__).name.lower()
    contains_self = False
    for token in parts:
        low = token.lower()
        if script_name in low:
            contains_self = True
            break
        try:
            if Path(token).name.lower() == script_name:
                contains_self = True
                break
        except Exception:
            continue

    if contains_self:
        return _split_command("openclaw"), "openclaw_command_self_reference_reset"
    return parts, ""


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


def _read_json_dict_with_status(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return {}, "read_error"
    if not raw.strip():
        return {}, "empty"
    try:
        payload = json.loads(raw)
    except Exception:
        return {}, "invalid"
    if not isinstance(payload, dict):
        return {}, "non_dict"
    return payload, "ok"


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload, _status = _read_json_dict_with_status(path)
    return payload


def _append_unique(items: list[str], value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    if text not in items:
        items.append(text)


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


@dataclass(frozen=True)
class _RunLock:
    path: Path
    token: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class _RunLockAttempt:
    acquired: bool
    lock: Optional[_RunLock]
    holder: dict[str, Any]
    reason: str


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


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = float(default)
    return max(float(minimum), float(value))


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


def _process_command_line(pid: int) -> str:
    if pid <= 0:
        return ""

    if os.name == "nt":
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId = {int(pid)}\" "
                "-ErrorAction SilentlyContinue; "
                "if ($p) { $p.CommandLine }"
            ),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return (proc.stdout or "").strip()
        except Exception:
            return ""

    try:
        proc = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def _lock_holder_matches_process(holder: dict[str, Any]) -> bool:
    pid = 0
    try:
        pid = int(holder.get("pid") or 0)
    except Exception:
        pid = 0
    if pid <= 0:
        return False
    if not _pid_running(pid):
        return False

    expected_parts: list[str] = []
    script_name = Path(__file__).name.lower()
    raw_command = str(holder.get("command") or "").strip()
    if raw_command:
        low = raw_command.lower()
        if script_name in low:
            expected_parts.append(script_name)
        mode = str(holder.get("mode") or "").strip().lower()
        if mode == "watch" and "--watch" in low:
            expected_parts.append("--watch")

    if not expected_parts:
        return True

    process_command = _process_command_line(pid).lower()
    if not process_command:
        return False
    return all(part in process_command for part in expected_parts)


def _listener_matches_expected_gateway(listener: dict[str, Any], expected_port: int) -> bool:
    if not isinstance(listener, dict):
        return False
    command = str(listener.get("command") or "").strip().lower()
    command_line = str(listener.get("commandLine") or "").strip().lower()
    text = f"{command} {command_line}".strip()
    if not text:
        return False
    if "openclaw" not in text or "gateway" not in text:
        return False
    if int(expected_port) > 0:
        if f"--port {int(expected_port)}" not in text:
            return False
    return True


def _terminate_pid(pid: int) -> _CmdResult:
    if pid <= 0:
        return _CmdResult(False, 1, [], "", "invalid_pid")

    if os.name == "nt":
        cmd = ["taskkill", "/PID", str(int(pid)), "/F", "/T"]
    else:
        cmd = ["kill", "-TERM", str(int(pid))]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except FileNotFoundError as exc:
        return _CmdResult(False, 127, cmd, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return _CmdResult(False, 124, cmd, out, err or "timeout")
    return _CmdResult(int(proc.returncode) == 0, int(proc.returncode), cmd, proc.stdout or "", proc.stderr or "")


def _recover_detached_gateway_listener(
    *,
    oc_base: list[str],
    listener: dict[str, Any],
    expected_port: int,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "attempted": False,
        "matched_expected_gateway": False,
        "listener_pid": None,
        "actions": [],
        "warnings": [],
        "errors": [],
    }
    if not isinstance(listener, dict):
        return info

    listener_pid = int(listener.get("pid")) if str(listener.get("pid") or "").isdigit() else 0
    info["listener_pid"] = listener_pid or None
    if listener_pid <= 0:
        _append_unique(info["warnings"], "gateway_listener_pid_missing")
        return info

    if not _listener_matches_expected_gateway(listener, expected_port):
        _append_unique(info["warnings"], "gateway_listener_identity_unverified")
        return info

    info["matched_expected_gateway"] = True
    info["attempted"] = True

    stop_res = _run_openclaw(oc_base=oc_base, args=["gateway", "stop"], timeout_seconds=30.0)
    if stop_res.ok:
        _append_unique(info["actions"], "gateway_stop_detached_listener_recovery")
    else:
        _append_unique(info["warnings"], f"gateway_stop_detached_listener_failed:rc={stop_res.returncode}")

    if not _pid_running(listener_pid):
        _append_unique(info["actions"], f"gateway_listener_cleared:{listener_pid}")
        return info

    kill_res = _terminate_pid(listener_pid)
    if not kill_res.ok:
        _append_unique(info["errors"], f"gateway_listener_terminate_failed:{listener_pid}:rc={kill_res.returncode}")
        return info

    _append_unique(info["actions"], f"gateway_listener_terminated:{listener_pid}")
    return info


def _state_seconds_since(value: Any) -> Optional[float]:
    ts = _parse_ts(value)
    if ts is None:
        return None
    return max(0.0, (_utc_now() - ts).total_seconds())


def _path_age_seconds(path: Path) -> Optional[float]:
    try:
        return max(0.0, time.time() - float(path.stat().st_mtime))
    except Exception:
        return None


def _load_runtime_state(path: Path) -> dict[str, Any]:
    payload = _safe_read_json(path)
    state = payload if isinstance(payload, dict) else {}
    state.setdefault("version", 1)
    return state


def _save_runtime_state(path: Path, state: dict[str, Any]) -> None:
    payload = dict(state or {})
    payload["updated_at_utc"] = _utc_now_iso()
    _safe_write_json(path, payload)


def _acquire_run_lock(
    *,
    lock_path: Path,
    mode: str,
    wait_seconds: float,
    stale_seconds: int,
) -> _RunLockAttempt:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    poll_seconds = 0.5

    while True:
        token = f"{os.getpid()}-{int(time.time() * 1000)}"
        payload = {
            "pid": int(os.getpid()),
            "mode": str(mode or "run"),
            "created_at_utc": _utc_now_iso(),
            "token": token,
            "command": " ".join(sys.argv),
        }
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
            finally:
                os.close(fd)
            return _RunLockAttempt(
                acquired=True,
                lock=_RunLock(path=lock_path, token=token, payload=payload),
                holder={},
                reason="acquired",
            )
        except FileExistsError:
            holder, holder_status = _read_json_dict_with_status(lock_path)
            holder_pid = 0
            try:
                holder_pid = int(holder.get("pid") or 0)
            except Exception:
                holder_pid = 0
            holder_age = _state_seconds_since(holder.get("created_at_utc"))
            holder_file_age = _path_age_seconds(lock_path)
            stale = False
            if holder_pid > 0 and not _lock_holder_matches_process(holder):
                stale = True
            elif holder_pid <= 0 and holder_age is not None and holder_age >= max(60, int(stale_seconds)):
                stale = True
            elif holder_status in {"empty", "invalid", "non_dict"}:
                invalid_lock_grace_seconds = 10.0
                effective_age = holder_age if holder_age is not None else holder_file_age
                if effective_age is not None and effective_age >= invalid_lock_grace_seconds:
                    stale = True

            if stale:
                try:
                    lock_path.unlink(missing_ok=True)
                    continue
                except Exception:
                    pass

            if time.monotonic() >= deadline:
                return _RunLockAttempt(acquired=False, lock=None, holder=holder, reason="held")
            time.sleep(poll_seconds)
        except Exception as exc:
            return _RunLockAttempt(
                acquired=False,
                lock=None,
                holder={"error": str(exc)},
                reason="lock_error",
            )


def _release_run_lock(lock: _RunLock) -> None:
    try:
        holder = _safe_read_json(lock.path)
        if str(holder.get("token") or "") != str(lock.token):
            return
        lock.path.unlink(missing_ok=True)
    except Exception:
        return


def _archive_path(path: Path, suffix: str) -> Path:
    ts = _utc_now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}.stale.{ts}.{suffix}")


def _openclaw_home() -> Path:
    return Path.home() / ".openclaw"


def _openclaw_config_path() -> Path:
    return _openclaw_home() / "openclaw.json"


def _resolve_default_agent_id_from_config(cfg: dict[str, Any]) -> str:
    acp = cfg.get("acp") if isinstance(cfg.get("acp"), dict) else {}
    default_agent = str(acp.get("defaultAgent") or "").strip()
    if default_agent:
        return default_agent

    agents = cfg.get("agents") if isinstance(cfg.get("agents"), dict) else {}
    agent_list = agents.get("list") if isinstance(agents.get("list"), list) else []
    for row in agent_list:
        if not isinstance(row, dict):
            continue
        if bool(row.get("default")):
            agent_id = str(row.get("id") or "").strip()
            if agent_id:
                return agent_id
    return "ops"


def _resolve_bound_agent_id(
    cfg: dict[str, Any],
    *,
    channel: str,
    account_id: str = "default",
) -> str:
    bindings = cfg.get("bindings") if isinstance(cfg.get("bindings"), list) else []
    channel_text = str(channel or "").strip().lower()
    account_text = str(account_id or "").strip()

    for row in bindings:
        if not isinstance(row, dict):
            continue
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        match_channel = str(match.get("channel") or "").strip().lower()
        match_account = str(match.get("accountId") or "").strip()
        agent_id = str(row.get("agentId") or "").strip()
        if agent_id and match_channel == channel_text and match_account == account_text:
            return agent_id

    for row in bindings:
        if not isinstance(row, dict):
            continue
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        match_channel = str(match.get("channel") or "").strip().lower()
        agent_id = str(row.get("agentId") or "").strip()
        if agent_id and match_channel == channel_text:
            return agent_id

    return _resolve_default_agent_id_from_config(cfg)


def _session_index_path_for_agent(agent_id: str) -> Path:
    return _openclaw_home() / "agents" / str(agent_id or "").strip() / "sessions" / "sessions.json"


def _extract_direct_session_peer(session_key: str, *, agent_id: str, channel: str) -> Optional[str]:
    prefix = f"agent:{str(agent_id or '').strip()}:{str(channel or '').strip().lower()}:direct:"
    raw_key = str(session_key or "").strip()
    if not raw_key.startswith(prefix):
        return None
    peer = raw_key[len(prefix) :].strip()
    return peer or None


def _reconcile_bound_direct_sessions(
    *,
    primary_channel: str,
    apply: bool,
) -> dict[str, Any]:
    cfg = _safe_read_json(_openclaw_config_path())
    result: dict[str, Any] = {
        "config_path": str(_openclaw_config_path()),
        "primary_channel": str(primary_channel or "whatsapp").strip().lower() or "whatsapp",
        "bound_account_id": "default",
        "bound_agent_id": "",
        "expected_model": "",
        "peers_scanned": 0,
        "peers_with_conflicts": 0,
        "duplicate_wrong_agent_keys": 0,
        "refreshed_bound_keys": 0,
        "session_indexes_written": 0,
        "updated_agents": [],
        "warnings": [],
        "errors": [],
    }
    if not cfg:
        _append_unique(result["warnings"], "openclaw_config_unavailable")
        return result

    channel = str(result["primary_channel"])
    bound_agent_id = _resolve_bound_agent_id(cfg, channel=channel, account_id="default")
    result["bound_agent_id"] = bound_agent_id

    agents_cfg = cfg.get("agents") if isinstance(cfg.get("agents"), dict) else {}
    agent_rows = agents_cfg.get("list") if isinstance(agents_cfg.get("list"), list) else []
    agent_model_map: dict[str, str] = {}
    agent_ids: list[str] = []
    for row in agent_rows:
        if not isinstance(row, dict):
            continue
        agent_id = str(row.get("id") or "").strip()
        if not agent_id:
            continue
        agent_ids.append(agent_id)
        agent_model_map[agent_id] = str(row.get("model") or "").strip()

    if bound_agent_id and bound_agent_id not in agent_ids:
        agent_ids.append(bound_agent_id)
    expected_model = str(agent_model_map.get(bound_agent_id) or "").strip()
    result["expected_model"] = expected_model

    peer_entries: dict[str, list[dict[str, Any]]] = {}
    session_indexes: dict[str, dict[str, Any]] = {}
    for agent_id in agent_ids:
        index_path = _session_index_path_for_agent(agent_id)
        if not index_path.exists():
            continue
        payload = _safe_read_json(index_path)
        if not isinstance(payload, dict) or not payload:
            continue
        session_indexes[agent_id] = {"path": index_path, "payload": payload}
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            peer = _extract_direct_session_peer(str(key), agent_id=agent_id, channel=channel)
            if not peer:
                continue
            peer_entries.setdefault(peer, []).append(
                {
                    "agent_id": agent_id,
                    "key": str(key),
                    "model": str(value.get("model") or "").strip(),
                }
            )

    result["peers_scanned"] = len(peer_entries)
    changed_agents: set[str] = set()

    for peer, entries in peer_entries.items():
        bound_entries = [row for row in entries if row.get("agent_id") == bound_agent_id]
        wrong_entries = [row for row in entries if row.get("agent_id") != bound_agent_id]
        refresh_bound = False
        if bound_entries and expected_model:
            refresh_bound = any(str(row.get("model") or "").strip() != expected_model for row in bound_entries)
        if len(bound_entries) > 1:
            refresh_bound = True

        if not wrong_entries and not refresh_bound:
            continue

        result["peers_with_conflicts"] += 1
        if wrong_entries:
            result["duplicate_wrong_agent_keys"] += len(wrong_entries)
        if refresh_bound:
            result["refreshed_bound_keys"] += len(bound_entries)

        if not apply:
            continue

        for entry in [*wrong_entries, *(bound_entries if refresh_bound else [])]:
            agent_id = str(entry.get("agent_id") or "").strip()
            key = str(entry.get("key") or "").strip()
            session_idx = session_indexes.get(agent_id)
            if not agent_id or not key or not isinstance(session_idx, dict):
                continue
            payload = session_idx.get("payload")
            if not isinstance(payload, dict):
                continue
            if key in payload:
                del payload[key]
                changed_agents.add(agent_id)

    if apply:
        for agent_id in sorted(changed_agents):
            session_idx = session_indexes.get(agent_id)
            if not isinstance(session_idx, dict):
                continue
            path = session_idx.get("path")
            payload = session_idx.get("payload")
            if not isinstance(path, Path) or not isinstance(payload, dict):
                continue
            _safe_write_json(path, payload)
            result["session_indexes_written"] += 1
        result["updated_agents"] = sorted(changed_agents)

    return result


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
    try:
        session_dirs = list(agents_root.glob("*/sessions"))
    except Exception as exc:
        result["errors"].append(f"scan_sessions_root_failed:{exc}")
        return result

    for sessions_dir in session_dirs:
        try:
            if not sessions_dir.is_dir():
                continue
        except Exception as exc:
            result["errors"].append(f"scan_session_dir_failed:{sessions_dir}:{exc}")
            continue

        try:
            lock_files = list(sessions_dir.glob("*.jsonl.lock"))
        except Exception as exc:
            result["errors"].append(f"scan_lock_glob_failed:{sessions_dir}:{exc}")
            continue

        for lock_path in lock_files:
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


def _reprobe_dns(
    hostname: str,
    *,
    attempts: int,
    base_delay_seconds: float,
) -> dict[str, Any]:
    total_attempts = max(1, int(attempts))
    base_delay = max(0.0, float(base_delay_seconds))
    history: list[dict[str, Any]] = []

    for idx in range(1, total_attempts + 1):
        diag = _resolve_dns(hostname)
        row = {
            "attempt": idx,
            "ok": bool(diag.get("ok")),
            "addresses": list(diag.get("addresses") or []),
            "error": str(diag.get("error") or ""),
        }
        history.append(row)
        if row["ok"]:
            return {
                "hostname": hostname,
                "ok": True,
                "attempts": total_attempts,
                "base_delay_seconds": base_delay,
                "succeeded_on_attempt": idx,
                "history": history,
            }
        if idx < total_attempts and base_delay > 0:
            time.sleep(base_delay * float(2 ** (idx - 1)))

    return {
        "hostname": hostname,
        "ok": False,
        "attempts": total_attempts,
        "base_delay_seconds": base_delay,
        "succeeded_on_attempt": None,
        "history": history,
    }


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
    probe: bool = False,
) -> tuple[_CmdResult, Optional[dict[str, Any]]]:
    if wait_seconds > 0:
        time.sleep(max(0.0, float(wait_seconds)))
    args = ["channels", "status"]
    if probe:
        args.append("--probe")
    res, payload = _run_openclaw_json(oc_base=oc_base, args=args, timeout_seconds=20.0)
    return res, payload if isinstance(payload, dict) else None


def _classify_gateway_restart_failure(res: _CmdResult) -> str:
    text = " ".join([str(res.stdout or ""), str(res.stderr or "")]).lower()
    if any(token in text for token in ("gateway already running", "lock timeout", "port 18789 is already in use")):
        return "already_running_conflict"
    if "timeout" in text:
        return "timeout"
    return f"rc={int(res.returncode)}"


def _analyze_whatsapp_delivery_signals(lines: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {
        "closed_session_events": 0,
        "recovery_budget_exceeded_events": 0,
        "deferred_to_restart_events": 0,
        "retry_wait_events": 0,
    }
    retry_wait_max_ms = 0
    retry_wait_re = re.compile(r"waiting\s+(\d+)\s*ms\s+before retrying delivery", re.IGNORECASE)

    for raw in lines:
        line = str(raw or "")
        low = line.lower()
        if "decrypted message with closed session" in low:
            counts["closed_session_events"] += 1
        if "recovery time budget exceeded" in low:
            counts["recovery_budget_exceeded_events"] += 1
        if "deferred to next restart" in low:
            counts["deferred_to_restart_events"] += 1
        match = retry_wait_re.search(line)
        if match:
            counts["retry_wait_events"] += 1
            try:
                retry_wait_max_ms = max(retry_wait_max_ms, int(match.group(1)))
            except Exception:
                pass

    signal_count = int(sum(int(v) for v in counts.values()))
    return {
        "signal_count": signal_count,
        "counts": counts,
        "retry_wait_max_ms": int(retry_wait_max_ms),
    }


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
    state: Optional[dict[str, Any]] = None,
    gateway_restart_cooldown_seconds: float = 0.0,
    primary_reenable_cooldown_seconds: float = 0.0,
    restart_retry_base_delay_seconds: float = 2.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rpc_ok": None,
        "service_status": None,
        "service_state": None,
        "gateway_port_status": None,
        "gateway_listener_pid": None,
        "dns": None,
        "primary_channel_issue": None,
        "primary_channel_issue_after_restart": None,
        "primary_channel_issue_final": None,
        "primary_channel_status_before": _primary_channel_snapshot(channels_payload, primary_channel),
        "primary_channel_status_after_restart": None,
        "primary_channel_status_final": None,
        "delivery_recovery": None,
        "actions": [],
        "manual_actions": [],
        "warnings": [],
        "errors": [],
    }
    state_map = state if isinstance(state, dict) else {}
    dns_host = "web.whatsapp.com"
    _, gateway_payload = _run_openclaw_json(oc_base=oc_base, args=["gateway", "status"], timeout_seconds=20.0)
    if not isinstance(gateway_payload, dict):
        out["errors"].append("gateway_status_unavailable")
        return out

    rpc = gateway_payload.get("rpc") if isinstance(gateway_payload.get("rpc"), dict) else {}
    service = gateway_payload.get("service") if isinstance(gateway_payload.get("service"), dict) else {}
    runtime = service.get("runtime") if isinstance(service.get("runtime"), dict) else {}
    port = gateway_payload.get("port") if isinstance(gateway_payload.get("port"), dict) else {}
    listeners = port.get("listeners") if isinstance(port.get("listeners"), list) else []
    listener = listeners[0] if listeners and isinstance(listeners[0], dict) else {}
    gateway_port = int(gateway_payload.get("gateway", {}).get("port") or 0) if isinstance(gateway_payload.get("gateway"), dict) else 0
    out["rpc_ok"] = bool(rpc.get("ok"))
    out["service_status"] = str(runtime.get("status") or "")
    out["service_state"] = str(runtime.get("state") or "")
    out["gateway_port_status"] = str(port.get("status") or "")
    out["gateway_listener_pid"] = int(listener.get("pid")) if str(listener.get("pid") or "").isdigit() else None

    primary_issue = _detect_primary_channel_issue(channels_payload, primary_channel)
    out["primary_channel_issue"] = primary_issue
    if primary_channel == "whatsapp":
        out["dns"] = _resolve_dns(dns_host)
        before = out.get("primary_channel_status_before") if isinstance(out.get("primary_channel_status_before"), dict) else {}
        if bool(before.get("configured")) and bool(before.get("linked")) and (
            (not bool(before.get("running"))) or (not bool(before.get("connected", True)))
        ):
            _append_unique(out["warnings"], "whatsapp_provider_unavailable")

    if bool(out["rpc_ok"]) and str(out["service_status"]).strip().lower() in {"stopped", "ready"}:
        _append_unique(out["warnings"], "gateway_service_runtime_mismatch")

    listener_conflict = bool(
        bool(out["rpc_ok"])
        and str(out["gateway_port_status"]).strip().lower() == "busy"
        and str(out["service_status"]).strip().lower() in {"stopped", "ready"}
    )
    if listener_conflict:
        _append_unique(out["warnings"], "gateway_detached_listener_conflict")
        _append_unique(out["manual_actions"], "run: openclaw gateway status --json")
        if apply and primary_issue and _listener_matches_expected_gateway(listener, gateway_port):
            detached_recovery = _recover_detached_gateway_listener(
                oc_base=oc_base,
                listener=listener,
                expected_port=gateway_port,
            )
            out["detached_listener_recovery"] = detached_recovery
            for action in detached_recovery.get("actions") or []:
                _append_unique(out["actions"], str(action))
            for warning in detached_recovery.get("warnings") or []:
                _append_unique(out["warnings"], str(warning))
            for error in detached_recovery.get("errors") or []:
                _append_unique(out["errors"], str(error))
            if not detached_recovery.get("errors"):
                _, gateway_payload_after_detached = _run_openclaw_json(
                    oc_base=oc_base,
                    args=["gateway", "status"],
                    timeout_seconds=20.0,
                )
                if isinstance(gateway_payload_after_detached, dict):
                    gateway_payload = gateway_payload_after_detached
                    rpc = gateway_payload.get("rpc") if isinstance(gateway_payload.get("rpc"), dict) else {}
                    service = gateway_payload.get("service") if isinstance(gateway_payload.get("service"), dict) else {}
                    runtime = service.get("runtime") if isinstance(service.get("runtime"), dict) else {}
                    port = gateway_payload.get("port") if isinstance(gateway_payload.get("port"), dict) else {}
                    listeners = port.get("listeners") if isinstance(port.get("listeners"), list) else []
                    listener = listeners[0] if listeners and isinstance(listeners[0], dict) else {}
                    out["rpc_ok"] = bool(rpc.get("ok"))
                    out["service_status"] = str(runtime.get("status") or "")
                    out["service_state"] = str(runtime.get("state") or "")
                    out["gateway_port_status"] = str(port.get("status") or "")
                    out["gateway_listener_pid"] = int(listener.get("pid")) if str(listener.get("pid") or "").isdigit() else None
                listener_conflict = bool(
                    bool(out["rpc_ok"])
                    and str(out["gateway_port_status"]).strip().lower() == "busy"
                    and str(out["service_status"]).strip().lower() in {"stopped", "ready"}
                )

    dns_diag = out.get("dns") if isinstance(out.get("dns"), dict) else {}
    dns_reprobe_ok: Optional[bool] = None
    if primary_channel == "whatsapp" and primary_issue == "whatsapp_dns_resolution_failed":
        dns_reprobe = _reprobe_dns(
            dns_host,
            attempts=_env_int("OPENCLAW_DNS_REPROBE_ATTEMPTS", 3, minimum=1),
            base_delay_seconds=_env_float("OPENCLAW_DNS_REPROBE_BASE_DELAY_SECONDS", 1.0, minimum=0.0),
        )
        out["dns_reprobe"] = dns_reprobe
        dns_reprobe_ok = bool(dns_reprobe.get("ok"))
        if dns_reprobe_ok:
            _append_unique(out["actions"], "dns_reprobe_recovered")
            _probe_res, channels_after_probe = _recheck_channels_status(
                oc_base=oc_base,
                wait_seconds=0.0,
                probe=True,
            )
            if isinstance(channels_after_probe, dict):
                channels_payload = channels_after_probe
                out["primary_channel_status_before"] = _primary_channel_snapshot(channels_payload, primary_channel)
                primary_issue = _detect_primary_channel_issue(channels_after_probe, primary_channel)
                out["primary_channel_issue"] = primary_issue
            else:
                _append_unique(out["warnings"], "channels_status_unavailable_after_dns_reprobe")

    skip_restart_for_dns = bool(
        primary_issue == "whatsapp_dns_resolution_failed"
        and not (
            bool(dns_reprobe_ok)
            if dns_reprobe_ok is not None
            else bool(isinstance(dns_diag, dict) and dns_diag.get("ok"))
        )
    )
    last_restart_elapsed = _state_seconds_since(state_map.get("last_gateway_restart_at"))
    skip_restart_for_cooldown = bool(
        float(gateway_restart_cooldown_seconds) > 0
        and last_restart_elapsed is not None
        and last_restart_elapsed < float(gateway_restart_cooldown_seconds)
    )
    should_restart = (
        bool(primary_issue) or ((out["rpc_ok"] is False) and bool(restart_on_rpc_failure))
    ) and not skip_restart_for_dns and not listener_conflict and not skip_restart_for_cooldown
    if skip_restart_for_dns:
        _append_unique(out["warnings"], "skip_restart_due_to_dns_failure")
    if listener_conflict:
        _append_unique(out["warnings"], "skip_restart_due_to_listener_conflict")
    if skip_restart_for_cooldown:
        remaining = max(
            1,
            int(round(float(gateway_restart_cooldown_seconds) - float(last_restart_elapsed or 0.0))),
        )
        _append_unique(out["warnings"], f"gateway_restart_cooldown_active:{remaining}s")

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
                reason = _classify_gateway_restart_failure(restart)
                _append_unique(out["warnings"], f"gateway_restart_attempt_failed:{attempt}:{reason}")
                if reason == "already_running_conflict":
                    _append_unique(out["warnings"], "gateway_restart_conflict_detected")
                    break
                if attempt < restart_attempt_budget:
                    delay = max(0.0, float(restart_retry_base_delay_seconds)) * float(2 ** (attempt - 1))
                    if delay > 0:
                        _append_unique(out["warnings"], f"gateway_restart_backoff_wait:{int(round(delay))}s")
                        time.sleep(delay)
                continue

            restart_ok = True
            _append_unique(
                out["actions"],
                "gateway_restart_primary_channel_recovery"
                if primary_issue
                else "gateway_restart",
            )
            if restart_attempt_budget > 1:
                _append_unique(out["actions"], f"gateway_restart_attempt:{attempt}")
            state_map["last_gateway_restart_at"] = _utc_now_iso()
            state_map["last_gateway_restart_reason"] = str(primary_issue or "rpc_failure")

            if not primary_issue:
                break

            _status_res, channels_after_restart = _recheck_channels_status(
                oc_base=oc_base,
                wait_seconds=recheck_delay_seconds,
                probe=(primary_channel == "whatsapp"),
            )
            if isinstance(channels_after_restart, dict):
                issue_after_restart = _detect_primary_channel_issue(channels_after_restart, primary_channel)
                out["primary_channel_issue_after_restart"] = issue_after_restart
                out["primary_channel_status_after_restart"] = _primary_channel_snapshot(channels_after_restart, primary_channel)
            else:
                _append_unique(out["warnings"], f"channels_status_unavailable_after_restart_attempt:{attempt}")

            if not issue_after_restart:
                break
            if issue_after_restart == "whatsapp_dns_resolution_failed":
                break
            if attempt < restart_attempt_budget:
                _append_unique(out["warnings"], f"primary_issue_persist_after_restart_attempt:{attempt}:{issue_after_restart}")
                delay = max(0.0, float(restart_retry_base_delay_seconds)) * float(2 ** (attempt - 1))
                if delay > 0:
                    _append_unique(out["warnings"], f"gateway_restart_backoff_wait:{int(round(delay))}s")
                    time.sleep(delay)
    elif should_restart and not apply:
        _append_unique(out["warnings"], "dry_run_restart_required")

    if primary_issue and restart_ok and out["primary_channel_status_after_restart"] is None:
        _status_res, channels_after_restart = _recheck_channels_status(
            oc_base=oc_base,
            wait_seconds=max(1.0, recheck_delay_seconds),
            probe=(primary_channel == "whatsapp"),
        )
        if isinstance(channels_after_restart, dict):
            issue_after_restart = _detect_primary_channel_issue(channels_after_restart, primary_channel)
            out["primary_channel_issue_after_restart"] = issue_after_restart
            out["primary_channel_status_after_restart"] = _primary_channel_snapshot(channels_after_restart, primary_channel)
        else:
            _append_unique(out["warnings"], "channels_status_unavailable_after_restart")

    final_channels_payload = channels_after_restart if isinstance(channels_after_restart, dict) else channels_payload
    final_issue = issue_after_restart if restart_ok else primary_issue

    can_attempt_reenable = True
    if float(primary_reenable_cooldown_seconds) > 0:
        last_reenable_elapsed = _state_seconds_since(state_map.get("last_primary_reenable_at"))
        if last_reenable_elapsed is not None and last_reenable_elapsed < float(primary_reenable_cooldown_seconds):
            can_attempt_reenable = False
            remaining = max(
                1,
                int(round(float(primary_reenable_cooldown_seconds) - float(last_reenable_elapsed))),
            )
            _append_unique(out["warnings"], f"primary_reenable_cooldown_active:{remaining}s")

    if (
        bool(apply)
        and bool(attempt_primary_reenable)
        and can_attempt_reenable
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
                _append_unique(out["warnings"], f"primary_account_disable_failed:{account_id}:rc={disable_res.returncode}")
            else:
                _append_unique(out["actions"], f"primary_account_disabled:{account_id}")

            enable_res = _set_channel_account_enabled(
                oc_base=oc_base,
                channel=primary_channel,
                account_id=account_id,
                enabled=True,
            )
            if not enable_res.ok:
                _append_unique(out["warnings"], f"primary_account_enable_failed:{account_id}:rc={enable_res.returncode}")
            else:
                _append_unique(out["actions"], f"primary_account_enabled:{account_id}")
            state_map["last_primary_reenable_at"] = _utc_now_iso()
            # WhatsApp uses per-account enabled flags; writing channels.whatsapp.enabled
            # can trigger schema validation errors on newer OpenClaw builds.

            if disable_res.ok and enable_res.ok:
                restart = _run_openclaw(oc_base=oc_base, args=["gateway", "restart"], timeout_seconds=45.0)
                if restart.ok:
                    _append_unique(out["actions"], "gateway_restart_after_primary_reenable")
                    state_map["last_gateway_restart_at"] = _utc_now_iso()
                    state_map["last_gateway_restart_reason"] = "primary_reenable"
                else:
                    reason = _classify_gateway_restart_failure(restart)
                    _append_unique(out["warnings"], f"gateway_restart_after_reenable_failed:{reason}")

                _status_res, channels_after_reenable = _recheck_channels_status(
                    oc_base=oc_base,
                    wait_seconds=recheck_delay_seconds,
                    probe=(primary_channel == "whatsapp"),
                )
                if isinstance(channels_after_reenable, dict):
                    final_channels_payload = channels_after_reenable
                    final_issue = _detect_primary_channel_issue(channels_after_reenable, primary_channel)
                else:
                    _append_unique(out["warnings"], "channels_status_unavailable_after_primary_reenable")

    out["primary_channel_issue_final"] = final_issue
    out["primary_channel_status_final"] = _primary_channel_snapshot(final_channels_payload, primary_channel)
    logs_tail_lines: list[str] = []
    logs_tail_available = False
    if primary_channel == "whatsapp":
        logs_tail = _fetch_channel_logs_tail(
            oc_base=oc_base,
            channel=primary_channel,
            lines=_env_int("OPENCLAW_WHATSAPP_LOG_TAIL_LINES", 80, minimum=20),
        )
        if bool(logs_tail.get("ok")):
            lines = logs_tail.get("lines")
            logs_tail_lines = [str(x) for x in lines] if isinstance(lines, list) else []
            logs_tail_available = True
            signal_diag = _analyze_whatsapp_delivery_signals(logs_tail_lines)
            if int(signal_diag.get("signal_count") or 0) > 0:
                out["delivery_recovery"] = signal_diag
                counts = signal_diag.get("counts") if isinstance(signal_diag.get("counts"), dict) else {}
                if int(counts.get("closed_session_events") or 0) > 0:
                    _append_unique(out["warnings"], "whatsapp_closed_session_signal_detected")
                    _append_unique(
                        out["manual_actions"],
                        "run: openclaw channels login --channel whatsapp --account default --verbose",
                    )
                if int(counts.get("recovery_budget_exceeded_events") or 0) > 0:
                    _append_unique(out["warnings"], "delivery_recovery_budget_exceeded")
                if int(counts.get("deferred_to_restart_events") or 0) > 0:
                    _append_unique(out["warnings"], "delivery_recovery_deferred_to_restart")
                if (
                    int(counts.get("recovery_budget_exceeded_events") or 0) > 0
                    or int(counts.get("deferred_to_restart_events") or 0) > 0
                ):
                    _append_unique(out["manual_actions"], "run: openclaw channels logs --channel whatsapp --lines 300")
                    _append_unique(out["manual_actions"], "check_openclaw_delivery_recovery_budget_and_retry_settings")
                    _append_unique(out["manual_actions"], "avoid_concurrent_gateway_restarts_during_delivery_recovery")
                retry_wait_max_ms = int(signal_diag.get("retry_wait_max_ms") or 0)
                retry_warn_ms = _env_int("OPENCLAW_DELIVERY_RETRY_WARN_MS", 20000, minimum=1000)
                if retry_wait_max_ms >= retry_warn_ms:
                    _append_unique(out["warnings"], f"delivery_retry_wait_high:{retry_wait_max_ms}ms")
        elif final_issue:
            _append_unique(out["warnings"], str(logs_tail.get("error") or "channels_logs_unavailable"))

    if final_issue:
        _append_unique(out["errors"], f"primary_channel_unresolved:{final_issue}")
        if logs_tail_available:
            out["channel_logs_tail"] = logs_tail_lines
        if final_issue == "whatsapp_dns_resolution_failed":
            _append_unique(out["manual_actions"], "verify_dns_resolution:web.whatsapp.com")
            _append_unique(out["manual_actions"], "check_network_firewall_or_proxy_for_whatsapp_web")
            _append_unique(out["manual_actions"], "run: python scripts/openclaw_tls_dns_diagnostics.py --json")
            _append_unique(out["manual_actions"], "run: openclaw channels status --probe --json")
        elif final_issue == "whatsapp_session_logged_out":
            _append_unique(
                out["manual_actions"],
                "run: openclaw channels login --channel whatsapp --account default --verbose"
            )
        elif final_issue == "whatsapp_handshake_timeout":
            # 428/408 handshake failures require a Baileys version hotfix + fresh auth,
            # not just a re-login. Direct to the dedicated relink helper.
            _append_unique(
                out["manual_actions"],
                "run: python scripts/openclaw_whatsapp_relink.py --fresh-auth --force-wa-version-hotfix",
            )
            _append_unique(out["manual_actions"], "run: openclaw channels logs --channel whatsapp --lines 200")
        elif final_issue == "whatsapp_channel_down":
            _append_unique(
                out["manual_actions"],
                "run: openclaw channels login --channel whatsapp --account default --verbose"
            )
            _append_unique(out["manual_actions"], "run: openclaw channels logs --channel whatsapp --lines 200")
    return out


def _build_cycle_exception_report(
    *,
    apply: bool,
    primary_channel: str,
    oc_base: list[str],
    lock_file: Path,
    lock_token: str,
    watch_cycle: Optional[int],
    phase: str,
    exc: Exception,
) -> dict[str, Any]:
    warning = f"{phase}_exception:{type(exc).__name__}:{str(exc)}"
    report: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "apply": bool(apply),
        "primary_channel": str(primary_channel or "whatsapp").strip().lower() or "whatsapp",
        "command": list(oc_base),
        "status": "WARN",
        "steps": {
            "lock": {
                "acquired": True,
                "lock_file": str(lock_file),
                "token": str(lock_token),
                "pid": int(os.getpid()),
            },
            "cycle_exception": {
                "phase": str(phase),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        },
        "warnings": [warning],
        "errors": [],
    }
    if watch_cycle is not None:
        report["watch_cycle"] = int(watch_cycle)
    return report


def _fast_supervisor_tick(
    *,
    oc_base: list[str],
    primary_channel: str,
    apply: bool,
    state_map: dict[str, Any],
    fast_state: dict[str, Any],
    failure_threshold: int,
    restart_cooldown_seconds: float,
    probe_timeout_seconds: float,
    post_restart_recheck_seconds: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "ok": True,
        "action": "none",
        "reason": "",
        "consecutive_failures": int(fast_state.get("consecutive_failures") or 0),
        "primary_channel": str(primary_channel or "whatsapp"),
        "warnings": [],
        "errors": [],
    }
    threshold = max(1, int(failure_threshold))
    cooldown = max(0.0, float(restart_cooldown_seconds))
    timeout_seconds = max(3.0, float(probe_timeout_seconds))
    recheck_delay = max(0.0, float(post_restart_recheck_seconds))

    status_res, channels_payload = _run_openclaw_json(
        oc_base=oc_base,
        args=["channels", "status", "--probe"],
        timeout_seconds=timeout_seconds,
    )
    if isinstance(channels_payload, dict):
        out["snapshot"] = _primary_channel_snapshot(channels_payload, primary_channel)
        issue = _detect_primary_channel_issue(channels_payload, primary_channel)
        out["primary_issue"] = issue
        if issue:
            out["ok"] = False
            out["reason"] = f"primary_issue:{issue}"
    else:
        gateway_soft_ok = False
        if status_res.returncode == 124 or status_res.ok:
            gateway_res, gateway_payload = _run_openclaw_json(
                oc_base=oc_base,
                args=["gateway", "status"],
                timeout_seconds=max(5.0, min(10.0, timeout_seconds)),
            )
            if isinstance(gateway_payload, dict):
                rpc = gateway_payload.get("rpc") if isinstance(gateway_payload.get("rpc"), dict) else {}
                service = gateway_payload.get("service") if isinstance(gateway_payload.get("service"), dict) else {}
                runtime = service.get("runtime") if isinstance(service.get("runtime"), dict) else {}
                rpc_ok = bool(rpc.get("ok"))
                runtime_status = str(runtime.get("status") or "").strip().lower()
                out["gateway_fallback"] = {
                    "rpc_ok": rpc_ok,
                    "runtime_status": runtime_status,
                    "gateway_status_ok": bool(gateway_res.ok),
                }
                if rpc_ok or runtime_status == "running":
                    gateway_soft_ok = True

        if gateway_soft_ok:
            fast_state["consecutive_failures"] = 0
            out["consecutive_failures"] = 0
            out["ok"] = True
            out["action"] = "soft_timeout_skip"
            out["reason"] = "channels_status_timeout_softened"
            _append_unique(out["warnings"], "channels_status_timeout_softened_by_gateway_health")
            return out

        out["ok"] = False
        if status_res.ok:
            out["reason"] = "channels_status_unavailable"
        else:
            out["reason"] = f"channels_status_call_failed:rc={status_res.returncode}"
            _append_unique(out["warnings"], out["reason"])

    if out["ok"]:
        fast_state["consecutive_failures"] = 0
        out["consecutive_failures"] = 0
        return out

    next_failures = int(fast_state.get("consecutive_failures") or 0) + 1
    fast_state["consecutive_failures"] = next_failures
    out["consecutive_failures"] = next_failures
    if next_failures < threshold:
        out["action"] = "defer_until_threshold"
        return out

    last_restart_elapsed = _state_seconds_since(state_map.get("last_fast_supervisor_restart_at"))
    if cooldown > 0 and last_restart_elapsed is not None and last_restart_elapsed < cooldown:
        remaining = max(1, int(round(cooldown - last_restart_elapsed)))
        _append_unique(out["warnings"], f"fast_restart_cooldown_active:{remaining}s")
        out["action"] = "cooldown_skip"
        return out

    if not apply:
        _append_unique(out["warnings"], "fast_restart_required_dry_run")
        out["action"] = "dry_run_restart_required"
        return out

    restart = _run_openclaw(oc_base=oc_base, args=["gateway", "restart"], timeout_seconds=45.0)
    if not restart.ok:
        reason = _classify_gateway_restart_failure(restart)
        _append_unique(out["errors"], f"fast_gateway_restart_failed:{reason}")
        out["action"] = "restart_failed"
        return out

    state_map["last_fast_supervisor_restart_at"] = _utc_now_iso()
    state_map["last_fast_supervisor_restart_reason"] = str(out.get("reason") or "unknown")
    fast_state["consecutive_failures"] = 0
    out["consecutive_failures"] = 0
    out["action"] = "gateway_restart_triggered"
    _append_unique(out["warnings"], "fast_gateway_restart_triggered")

    if recheck_delay > 0:
        time.sleep(recheck_delay)
    _, post_payload = _run_openclaw_json(
        oc_base=oc_base,
        args=["channels", "status", "--probe"],
        timeout_seconds=timeout_seconds,
    )
    if isinstance(post_payload, dict):
        post_issue = _detect_primary_channel_issue(post_payload, primary_channel)
        out["post_restart_primary_issue"] = post_issue
        if post_issue:
            _append_unique(out["warnings"], f"fast_post_restart_primary_issue:{post_issue}")
        else:
            _append_unique(out["warnings"], "fast_post_restart_primary_recovered")
    else:
        _append_unique(out["warnings"], "fast_post_restart_channels_unavailable")

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
        "--state-file",
        default="logs/automation/openclaw_maintenance_state.json",
        help="Path to shared runtime state used for cooldown/backoff controls.",
    )
    parser.add_argument(
        "--lock-file",
        default="logs/automation/openclaw_maintenance.lock.json",
        help="Path to process lock file that prevents concurrent maintenance loops.",
    )
    parser.add_argument(
        "--lock-wait-seconds",
        type=float,
        default=_env_float("OPENCLAW_MAINTENANCE_LOCK_WAIT_SECONDS", 0.0, minimum=0.0),
        help="How long to wait for an existing maintenance lock before skipping this run.",
    )
    parser.add_argument(
        "--lock-stale-seconds",
        type=int,
        default=_env_int("OPENCLAW_MAINTENANCE_LOCK_STALE_SECONDS", 1800, minimum=60),
        help="Treat lock files older than this as stale when owner PID is not valid.",
    )
    parser.add_argument(
        "--gateway-restart-cooldown-seconds",
        type=float,
        default=_env_float("OPENCLAW_GATEWAY_RESTART_COOLDOWN_SECONDS", 120.0, minimum=0.0),
        help="Minimum time between gateway restart actions across maintenance cycles.",
    )
    parser.add_argument(
        "--primary-reenable-cooldown-seconds",
        type=float,
        default=_env_float("OPENCLAW_PRIMARY_REENABLE_COOLDOWN_SECONDS", 300.0, minimum=0.0),
        help="Minimum time between primary-account disable/enable remediation attempts.",
    )
    parser.add_argument(
        "--restart-retry-base-delay-seconds",
        type=float,
        default=_env_float("OPENCLAW_RESTART_RETRY_BASE_DELAY_SECONDS", 2.0, minimum=0.0),
        help="Base delay for exponential backoff between restart attempts.",
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
    parser.add_argument(
        "--fast-supervisor",
        dest="fast_supervisor",
        action="store_true",
        default=_as_bool(os.getenv("OPENCLAW_FAST_SUPERVISOR_ENABLED", "1"), default=True),
        help="Enable fast health/restart checks between full watch cycles.",
    )
    parser.add_argument(
        "--no-fast-supervisor",
        dest="fast_supervisor",
        action="store_false",
        help="Disable fast health/restart checks between full watch cycles.",
    )
    parser.add_argument(
        "--fast-supervisor-interval-seconds",
        type=float,
        default=_env_float("OPENCLAW_FAST_SUPERVISOR_INTERVAL_SECONDS", 5.0, minimum=1.0),
        help="Fast supervisor probe interval while in watch mode.",
    )
    parser.add_argument(
        "--fast-supervisor-failure-threshold",
        type=int,
        default=_env_int("OPENCLAW_FAST_SUPERVISOR_FAILURE_THRESHOLD", 2, minimum=1),
        help="Consecutive failed fast probes required before restart.",
    )
    parser.add_argument(
        "--fast-supervisor-restart-cooldown-seconds",
        type=float,
        default=_env_float("OPENCLAW_FAST_SUPERVISOR_RESTART_COOLDOWN_SECONDS", 20.0, minimum=0.0),
        help="Minimum seconds between fast supervisor restart actions.",
    )
    parser.add_argument(
        "--fast-supervisor-probe-timeout-seconds",
        type=float,
        default=_env_float("OPENCLAW_FAST_SUPERVISOR_PROBE_TIMEOUT_SECONDS", 8.0, minimum=3.0),
        help="Timeout per fast supervisor probe.",
    )
    parser.add_argument(
        "--fast-supervisor-post-restart-recheck-seconds",
        type=float,
        default=_env_float("OPENCLAW_FAST_SUPERVISOR_POST_RESTART_RECHECK_SECONDS", 4.0, minimum=0.0),
        help="Delay before fast supervisor post-restart channel recheck.",
    )
    args = parser.parse_args(argv)

    report_file = Path(str(args.report_file)).expanduser()
    if not report_file.is_absolute():
        report_file = (PROJECT_ROOT / report_file).resolve()
    state_file = Path(str(args.state_file)).expanduser()
    if not state_file.is_absolute():
        state_file = (PROJECT_ROOT / state_file).resolve()
    lock_file = Path(str(args.lock_file)).expanduser()
    if not lock_file.is_absolute():
        lock_file = (PROJECT_ROOT / lock_file).resolve()

    oc_base_raw = _split_command(str(args.command))
    oc_base, command_warning = _normalize_openclaw_command(oc_base_raw)
    lock_attempt = _acquire_run_lock(
        lock_path=lock_file,
        mode="watch" if bool(args.watch) else "run_once",
        wait_seconds=max(0.0, float(args.lock_wait_seconds)),
        stale_seconds=max(60, int(args.lock_stale_seconds)),
    )
    if not lock_attempt.acquired or lock_attempt.lock is None:
        holder = lock_attempt.holder if isinstance(lock_attempt.holder, dict) else {}
        holder_pid = 0
        try:
            holder_pid = int(holder.get("pid") or 0)
        except Exception:
            holder_pid = 0
        warning = "maintenance_lock_unavailable"
        if lock_attempt.reason == "held":
            mode = str(holder.get("mode") or "unknown")
            warning = f"maintenance_lock_held:pid={holder_pid}:mode={mode}"
        elif lock_attempt.reason == "lock_error":
            warning = f"maintenance_lock_error:{str(holder.get('error') or 'unknown')}"

        report: dict[str, Any] = {
            "timestamp_utc": _utc_now_iso(),
            "apply": bool(args.apply),
            "primary_channel": str(args.primary_channel or "whatsapp").strip().lower() or "whatsapp",
            "command": oc_base,
            "status": "WARN",
            "steps": {
                "lock": {
                    "acquired": False,
                    "reason": lock_attempt.reason,
                    "lock_file": str(lock_file),
                    "holder": holder,
                }
            },
            "warnings": [warning],
            "errors": [],
        }
        if command_warning:
            report["warnings"].append(command_warning)
        _safe_write_json(report_file, report)
        print(f"[openclaw_maintenance] status={report['status']} apply={args.apply}")
        print(f"[openclaw_maintenance] report={report_file}")
        print(f"[openclaw_maintenance] skipped_reason={warning}")
        return 0

    state = _load_runtime_state(state_file)

    def _run_cycle(
        *,
        watch_cycle: Optional[int],
        include_disable: bool,
        fast_tick: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "timestamp_utc": _utc_now_iso(),
            "apply": bool(args.apply),
            "primary_channel": str(args.primary_channel or "whatsapp").strip().lower() or "whatsapp",
            "command": oc_base,
            "status": "PASS",
            "steps": {
                "lock": {
                    "acquired": True,
                    "lock_file": str(lock_file),
                    "token": lock_attempt.lock.token,
                    "pid": int(os.getpid()),
                }
            },
            "warnings": [],
            "errors": [],
        }
        if watch_cycle is not None:
            report["watch_cycle"] = int(watch_cycle)
        if command_warning:
            _append_unique(report["warnings"], command_warning)
        if isinstance(fast_tick, dict) and fast_tick:
            report["steps"]["fast_supervisor"] = dict(fast_tick)

        lock_step = _cleanup_stale_session_locks(
            apply=bool(args.apply),
            session_stale_seconds=max(60, int(args.session_stale_seconds)),
        )
        report["steps"]["stale_session_cleanup"] = lock_step

        session_route_step = _reconcile_bound_direct_sessions(
            primary_channel=str(report["primary_channel"]),
            apply=bool(args.apply),
        )
        report["steps"]["session_route_reconcile"] = session_route_step

        _status_res, channels_payload = _run_openclaw_json(
            oc_base=oc_base,
            args=["channels", "status", "--probe"],
            timeout_seconds=20.0,
        )
        if isinstance(channels_payload, dict):
            report["steps"]["channels_status_snapshot"] = {
                "timestamp_ms": channels_payload.get("ts"),
                "channels": channels_payload.get("channels"),
            }
        else:
            _append_unique(report["warnings"], "channels_status_unavailable")

        gateway_step = _gateway_health_and_heal(
            oc_base=oc_base,
            channels_payload=channels_payload if isinstance(channels_payload, dict) else {},
            primary_channel=str(report["primary_channel"]),
            apply=bool(args.apply),
            restart_on_rpc_failure=bool(args.restart_gateway_on_rpc_failure),
            recheck_delay_seconds=max(0.0, float(args.recheck_delay_seconds)),
            attempt_primary_reenable=bool(args.attempt_primary_reenable),
            primary_restart_attempts=max(1, int(args.primary_restart_attempts)),
            state=state,
            gateway_restart_cooldown_seconds=max(0.0, float(args.gateway_restart_cooldown_seconds)),
            primary_reenable_cooldown_seconds=max(0.0, float(args.primary_reenable_cooldown_seconds)),
            restart_retry_base_delay_seconds=max(0.0, float(args.restart_retry_base_delay_seconds)),
        )
        report["steps"]["gateway_health"] = gateway_step

        channel_step: dict[str, Any] = {
            "skipped": True,
            "reason": "disabled_by_flag_or_watch_mode",
            "primary_channel": report["primary_channel"],
        }
        if include_disable and bool(args.disable_broken_channels):
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
        if isinstance(channel_step, dict) and channel_step.get("warnings"):
            report["warnings"].extend([str(x) for x in channel_step.get("warnings", [])])
        if lock_step.get("errors"):
            report["warnings"].extend([str(x) for x in lock_step.get("errors", [])])
        if isinstance(session_route_step, dict) and session_route_step.get("errors"):
            report["warnings"].extend([str(x) for x in session_route_step.get("errors", [])])
        if isinstance(session_route_step, dict) and session_route_step.get("warnings"):
            report["warnings"].extend([str(x) for x in session_route_step.get("warnings", [])])

        report["status"] = _derive_status([str(x) for x in report["errors"]])
        return report

    try:
        try:
            report = _run_cycle(watch_cycle=None, include_disable=True)
        except Exception as exc:
            report = _build_cycle_exception_report(
                apply=bool(args.apply),
                primary_channel=str(args.primary_channel or "whatsapp"),
                oc_base=oc_base,
                lock_file=lock_file,
                lock_token=lock_attempt.lock.token,
                watch_cycle=None,
                phase="initial_cycle",
                exc=exc,
            )
            _safe_write_json(report_file, report)
            print(f"[openclaw_maintenance] status={report['status']} apply={args.apply}")
            print(f"[openclaw_maintenance] report={report_file}")
            print(f"[openclaw_maintenance] warning={report['warnings'][0]}")
            if not args.watch:
                return 1
        else:
            _save_runtime_state(state_file, state)
            _safe_write_json(report_file, report)
            print(f"[openclaw_maintenance] status={report['status']} apply={args.apply}")
            print(f"[openclaw_maintenance] report={report_file}")
            channel_step = report["steps"].get("broken_channel_disable")
            if isinstance(channel_step, dict):
                disabled = channel_step.get("disabled") if isinstance(channel_step.get("disabled"), list) else []
                if disabled:
                    print(f"[openclaw_maintenance] disabled_channels={','.join(str(x) for x in disabled)}")

            if args.strict and report["errors"] and not args.watch:
                return 1

            if not args.watch:
                return 0

        interval = max(30, int(args.watch_interval))
        fast_enabled = bool(args.fast_supervisor)
        fast_interval = max(1.0, float(args.fast_supervisor_interval_seconds))
        fast_state: dict[str, Any] = {"consecutive_failures": 0, "last_tick": {}}
        next_full_cycle_at = time.monotonic() + float(interval)
        print(
            f"[openclaw_maintenance] Watch mode active (interval={interval}s, "
            f"fast_supervisor={fast_enabled}, fast_interval={fast_interval:.1f}s). Press Ctrl+C to stop."
        )
        cycle = 1
        while True:
            sleep_for = max(0.5, float(interval))
            if fast_enabled:
                remaining = max(0.5, float(next_full_cycle_at - time.monotonic()))
                sleep_for = max(0.5, min(fast_interval, remaining))
            try:
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                print(f"\n[openclaw_maintenance] Watch stopped after {cycle} cycles.")
                return 0

            fast_tick: Optional[dict[str, Any]] = None
            if fast_enabled:
                try:
                    fast_tick = _fast_supervisor_tick(
                        oc_base=oc_base,
                        primary_channel=str(args.primary_channel or "whatsapp").strip().lower() or "whatsapp",
                        apply=bool(args.apply),
                        state_map=state,
                        fast_state=fast_state,
                        failure_threshold=max(1, int(args.fast_supervisor_failure_threshold)),
                        restart_cooldown_seconds=max(0.0, float(args.fast_supervisor_restart_cooldown_seconds)),
                        probe_timeout_seconds=max(3.0, float(args.fast_supervisor_probe_timeout_seconds)),
                        post_restart_recheck_seconds=max(
                            0.0, float(args.fast_supervisor_post_restart_recheck_seconds)
                        ),
                    )
                    fast_state["last_tick"] = dict(fast_tick)
                    if str(fast_tick.get("action") or "") not in {"none", "defer_until_threshold"}:
                        print(
                            f"[openclaw_maintenance] fast action={fast_tick.get('action')} "
                            f"reason={fast_tick.get('reason')} warnings={len(fast_tick.get('warnings') or [])} "
                            f"errors={len(fast_tick.get('errors') or [])}"
                        )
                        _save_runtime_state(state_file, state)
                except Exception as exc:
                    print(f"[openclaw_maintenance] fast supervisor exception: {type(exc).__name__}: {exc}")

            if time.monotonic() < next_full_cycle_at:
                continue
            cycle += 1
            try:
                report = _run_cycle(
                    watch_cycle=cycle,
                    include_disable=False,
                    fast_tick=fast_state.get("last_tick") if fast_enabled else None,
                )
            except Exception as exc:
                report = _build_cycle_exception_report(
                    apply=bool(args.apply),
                    primary_channel=str(args.primary_channel or "whatsapp"),
                    oc_base=oc_base,
                    lock_file=lock_file,
                    lock_token=lock_attempt.lock.token,
                    watch_cycle=cycle,
                    phase="watch_cycle",
                    exc=exc,
                )
                _safe_write_json(report_file, report)
                print(
                    f"[openclaw_maintenance] watch cycle={cycle} status={report['status']} "
                    f"warning={report['warnings'][0]}"
                )
                continue

            _save_runtime_state(state_file, state)
            _safe_write_json(report_file, report)
            print(
                f"[openclaw_maintenance] watch cycle={cycle} status={report['status']} "
                f"errors={len(report['errors'])} warnings={len(report['warnings'])}"
            )
            next_full_cycle_at = time.monotonic() + float(interval)
    finally:
        _release_run_lock(lock_attempt.lock)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
