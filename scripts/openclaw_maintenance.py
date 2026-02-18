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
import shutil
import subprocess
import sys
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
    channels = channels_payload.get("channels") if isinstance(channels_payload.get("channels"), dict) else {}
    row = channels.get(primary_channel) if isinstance(channels, dict) else None
    if not isinstance(row, dict):
        return None
    configured = bool(row.get("configured"))
    if not configured:
        return None

    running = bool(row.get("running"))
    connected = bool(row.get("connected", True))
    last_error = str(row.get("lastError") or "")
    low = last_error.lower()

    if primary_channel == "whatsapp":
        if (not running) or (not connected):
            if any(tok in low for tok in ("handshake", "request time-out", "status=408", "connection was lost")):
                return "whatsapp_handshake_timeout"
            return "whatsapp_channel_down"
        return None

    if not running:
        return f"{primary_channel}_channel_down"
    return None


def _gateway_health_and_heal(
    *,
    oc_base: list[str],
    channels_payload: dict[str, Any],
    primary_channel: str,
    apply: bool,
    restart_on_rpc_failure: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rpc_ok": None,
        "service_status": None,
        "service_state": None,
        "primary_channel_issue": None,
        "actions": [],
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

    should_restart = bool(primary_issue) or ((out["rpc_ok"] is False) and bool(restart_on_rpc_failure))
    if should_restart and apply:
        restart = _run_openclaw(oc_base=oc_base, args=["gateway", "restart"], timeout_seconds=45.0)
        if restart.ok:
            out["actions"].append(
                "gateway_restart_primary_channel_recovery"
                if primary_issue
                else "gateway_restart"
            )
        else:
            out["errors"].append(f"gateway_restart_failed:rc={restart.returncode}")
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
    if isinstance(channel_step, dict) and channel_step.get("errors"):
        report["errors"].extend([str(x) for x in channel_step.get("errors", [])])
    if lock_step.get("errors"):
        report["warnings"].extend([str(x) for x in lock_step.get("errors", [])])

    if report["errors"]:
        report["status"] = "WARN"

    _safe_write_json(report_file, report)
    print(f"[openclaw_maintenance] status={report['status']} apply={args.apply}")
    print(f"[openclaw_maintenance] report={report_file}")
    if isinstance(channel_step, dict):
        disabled = channel_step.get("disabled") if isinstance(channel_step.get("disabled"), list) else []
        if disabled:
            print(f"[openclaw_maintenance] disabled_channels={','.join(str(x) for x in disabled)}")

    if args.strict and report["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
