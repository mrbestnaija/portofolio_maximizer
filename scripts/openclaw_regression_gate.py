#!/usr/bin/env python3
"""
OpenClaw connectivity regression gate.

Use this before merge/deploy on OpenClaw-enabled environments to ensure:
1) The primary channel is healthy via `openclaw channels status --probe --json`.
2) PMX maintenance strict gate passes (`scripts/openclaw_maintenance.py --strict`).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _CmdResult:
    ok: bool
    returncode: int
    command: list[str]
    stdout: str
    stderr: str


def _split_command(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        return ["openclaw"]
    try:
        parts = shlex.split(raw, posix=(os.name != "nt"))
    except Exception:
        parts = raw.split()
    if os.name == "nt" and len(parts) == 1:
        return ["cmd", "/d", "/s", "/c", parts[0]]
    return parts


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
    raise ValueError("invalid json output")


def _run(cmd: list[str], *, timeout_seconds: float) -> _CmdResult:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(3.0, float(timeout_seconds)),
            check=False,
        )
    except FileNotFoundError as exc:
        return _CmdResult(False, 127, cmd, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return _CmdResult(False, 124, cmd, out, err or "timeout")

    return _CmdResult(
        ok=int(proc.returncode) == 0,
        returncode=int(proc.returncode),
        command=cmd,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _channel_ready(payload: dict[str, Any], channel: str) -> tuple[bool, str]:
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    row = channels.get(channel) if isinstance(channels, dict) else None
    if not isinstance(row, dict):
        return False, "channel_missing"
    if not bool(row.get("configured")):
        return False, "channel_not_configured"
    if channel == "whatsapp" and not bool(row.get("linked")):
        return False, "whatsapp_not_linked"
    if not bool(row.get("running")):
        return False, "channel_not_running"
    if not bool(row.get("connected", True)):
        return False, "channel_not_connected"

    channel_accounts = payload.get("channelAccounts") if isinstance(payload.get("channelAccounts"), dict) else {}
    rows = channel_accounts.get(channel) if isinstance(channel_accounts, dict) else None
    if isinstance(rows, list):
        enabled_rows = [r for r in rows if isinstance(r, dict) and bool(r.get("enabled", True))]
        if enabled_rows and not any(bool(r.get("running")) and bool(r.get("connected", True)) for r in enabled_rows):
            return False, "enabled_account_not_running"

    return True, "ok"


def run_regression_gate(
    *,
    openclaw_command: str,
    python_bin: str,
    primary_channel: str,
    timeout_seconds: float,
    allow_missing_openclaw: bool,
) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "status": "FAIL",
        "primary_channel": primary_channel,
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    status_cmd = [
        *_split_command(openclaw_command),
        "--no-color",
        "channels",
        "status",
        "--probe",
        "--json",
    ]
    status_res = _run(status_cmd, timeout_seconds=timeout_seconds)
    report["checks"]["channels_probe"] = {
        "ok": status_res.ok,
        "returncode": status_res.returncode,
        "command": status_res.command,
    }

    if status_res.returncode == 127 and allow_missing_openclaw:
        report["status"] = "SKIP"
        report["warnings"].append("openclaw_cli_missing")
        return True, report

    if not status_res.ok:
        report["errors"].append("channels_probe_failed")
        report["checks"]["channels_probe"]["stderr"] = "\n".join((status_res.stderr or "").splitlines()[-10:])
        return False, report

    try:
        payload = _parse_json_best_effort(status_res.stdout)
    except Exception:
        report["errors"].append("channels_probe_invalid_json")
        report["checks"]["channels_probe"]["stdout"] = "\n".join((status_res.stdout or "").splitlines()[-10:])
        return False, report

    if not isinstance(payload, dict):
        report["errors"].append("channels_probe_invalid_payload")
        return False, report

    ready, reason = _channel_ready(payload, primary_channel)
    report["checks"]["primary_channel"] = {"ok": ready, "reason": reason}
    if not ready:
        report["errors"].append(f"primary_channel_not_ready:{reason}")
        return False, report

    maintenance_cmd = [
        str(python_bin),
        str(PROJECT_ROOT / "scripts" / "openclaw_maintenance.py"),
        "--strict",
        "--primary-channel",
        str(primary_channel),
    ]
    maintenance_res = _run(maintenance_cmd, timeout_seconds=max(20.0, timeout_seconds * 2.0))
    report["checks"]["maintenance_strict"] = {
        "ok": maintenance_res.ok,
        "returncode": maintenance_res.returncode,
        "command": maintenance_res.command,
    }
    if not maintenance_res.ok:
        report["errors"].append("maintenance_strict_failed")
        report["checks"]["maintenance_strict"]["stderr"] = "\n".join((maintenance_res.stderr or "").splitlines()[-10:])
        report["checks"]["maintenance_strict"]["stdout"] = "\n".join((maintenance_res.stdout or "").splitlines()[-10:])
        return False, report

    report["status"] = "PASS"
    return True, report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        default=os.getenv("OPENCLAW_COMMAND", "openclaw"),
        help="OpenClaw command (default: OPENCLAW_COMMAND or 'openclaw').",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used for openclaw_maintenance.py.",
    )
    parser.add_argument(
        "--primary-channel",
        default=os.getenv("OPENCLAW_CHANNEL", "whatsapp"),
        help="Primary channel to validate (default: OPENCLAW_CHANNEL or whatsapp).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="Timeout budget per command.",
    )
    parser.add_argument(
        "--allow-missing-openclaw",
        action="store_true",
        help="Return success with SKIP status when OpenClaw CLI is unavailable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report.",
    )
    args = parser.parse_args(argv)

    ok, report = run_regression_gate(
        openclaw_command=str(args.command),
        python_bin=str(args.python_bin),
        primary_channel=str(args.primary_channel).strip().lower() or "whatsapp",
        timeout_seconds=float(args.timeout_seconds),
        allow_missing_openclaw=bool(args.allow_missing_openclaw),
    )

    if bool(args.json):
        print(json.dumps(report, indent=2))
    else:
        print(
            f"[openclaw_regression_gate] status={report.get('status')} "
            f"errors={len(report.get('errors', []))} warnings={len(report.get('warnings', []))}"
        )
        for err in report.get("errors", []):
            print(f"[openclaw_regression_gate] error: {err}")
        for warn in report.get("warnings", []):
            print(f"[openclaw_regression_gate] warning: {warn}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
