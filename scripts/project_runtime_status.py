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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_EXEC_HOSTS = {"sandbox", "gateway", "node"}
VALID_SANDBOX_MODES_FOR_SANDBOX_HOST = {"non-main", "all"}


def _trim(text: str, max_chars: int = 1400) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "\n...[truncated]..."


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


def _openclaw_exec_environment_check() -> dict[str, Any]:
    cfg_path = Path.home() / ".openclaw" / "openclaw.json"
    check = {
        "name": "openclaw_exec_env",
        "ok": False,
        "returncode": 1,
        "duration_seconds": 0.0,
        "command": f"validate {cfg_path}",
        "stdout": "",
        "stderr": "",
        "signals": [],
    }
    if not cfg_path.exists():
        check["stderr"] = f"openclaw config missing: {cfg_path}"
        return check
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        check["stderr"] = f"openclaw config unreadable: {exc}"
        return check
    if not isinstance(cfg, dict):
        check["stderr"] = "openclaw config is not a JSON object"
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

    acp = cfg.get("acp", {}) if isinstance(cfg.get("acp"), dict) else {}
    default_agent = str(acp.get("defaultAgent") or "").strip()
    if not default_agent:
        check["signals"] = ["missing_acp_default_agent"]
        check["stderr"] = "acp.defaultAgent missing"
        return check

    check["ok"] = True
    check["returncode"] = 0
    check["signals"] = ["exec_env_valid"]
    check["stdout"] = f"host={host} sandbox_mode={sandbox_mode or '<unset>'} acp.defaultAgent={default_agent}"
    return check


def collect_runtime_status(*, timeout_seconds: float = 90.0) -> dict[str, Any]:
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

    checks.append(
        _run_check(
            "production_gate",
            [
                py,
                str(PROJECT_ROOT / "scripts" / "production_audit_gate.py"),
                "--unattended-profile",
            ],
            timeout_seconds=timeout_seconds,
        )
    )
    checks.append(
        _run_check(
            "error_monitor",
            [py, str(PROJECT_ROOT / "scripts" / "error_monitor.py"), "--check"],
            timeout_seconds=timeout_seconds,
        )
    )
    checks.append(_openclaw_exec_environment_check())

    failed = [c["name"] for c in checks if not bool(c.get("ok"))]
    return {
        "status": "ok" if not failed else "degraded",
        "failed_checks": failed,
        "check_count": len(checks),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="PowerShell-safe PMX runtime status snapshot")
    p.add_argument("--timeout-seconds", type=float, default=90.0, help="Per-check timeout")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = p.parse_args(argv)

    payload = collect_runtime_status(timeout_seconds=float(args.timeout_seconds))
    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, separators=(",", ":")))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
