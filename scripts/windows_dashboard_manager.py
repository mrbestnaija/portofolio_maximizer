#!/usr/bin/env python3
"""
windows_dashboard_manager.py
----------------------------

Windows-native helper that ensures the live dashboard stack is running:
1) DB -> JSON bridge (`scripts/dashboard_db_bridge.py`)
2) Local static HTTP server (`python -m http.server`)

Security defaults:
- bind host is hardcoded to 127.0.0.1 (never 0.0.0.0)
- dashboard URL is localhost-only
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
LOCALHOST_BIND = "127.0.0.1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _creation_flags() -> int:
    return (
        getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        out = (proc.stdout or "").strip()
        if "No tasks are running" in out:
            return False
        return f'"{pid}"' in out or f",{pid}," in out or str(pid) in out
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def _read_pidfile(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        pid = int(raw)
    except Exception:
        return None
    if _pid_alive(pid):
        return pid
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
    return None


def _write_pidfile(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(pid)), encoding="utf-8")


def _port_open(host: str, port: int, timeout: float = 0.8) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, int(port))) == 0
        except Exception:
            return False


@dataclass
class EnsureResult:
    url: str
    bridge_pid: Optional[int]
    server_pid: Optional[int]
    started_bridge: bool
    started_server: bool
    bridge_running: bool
    server_running: bool
    warnings: list[str]


def _start_detached(cmd: list[str], cwd: Path) -> Optional[int]:
    try:
        with open(os.devnull, "rb") as stdin, open(os.devnull, "ab") as stdout, open(
            os.devnull, "ab"
        ) as stderr:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                creationflags=_creation_flags(),
                close_fds=True,
            )
        time.sleep(0.35)
        if _pid_alive(proc.pid):
            return int(proc.pid)
    except Exception:
        return None
    return None


def _ensure_dashboard_stack(
    *,
    root: Path,
    python_bin: str,
    port: int,
    db_path: Path,
    persist_snapshot: bool,
    require_bridge: bool,
) -> EnsureResult:
    logs_dir = root / "logs"
    bridge_pidfile = logs_dir / "dashboard_bridge.pid"
    server_pidfile = logs_dir / f"dashboard_http_{port}.pid"

    bridge_script = root / "scripts" / "dashboard_db_bridge.py"
    dashboard_html = root / "visualizations" / "live_dashboard.html"
    url = f"http://{LOCALHOST_BIND}:{port}/visualizations/live_dashboard.html"

    warnings: list[str] = []
    started_bridge = False
    started_server = False

    if not dashboard_html.exists():
        warnings.append(f"dashboard HTML missing: {dashboard_html}")
    if not bridge_script.exists():
        warnings.append(f"dashboard bridge missing: {bridge_script}")

    bridge_pid = _read_pidfile(bridge_pidfile)
    if bridge_pid is None and bridge_script.exists():
        if not db_path.exists():
            warnings.append(f"bridge skipped; DB missing: {db_path}")
        else:
            bridge_cmd = [
                python_bin,
                str(bridge_script),
                "--interval-seconds",
                "5",
                "--db-path",
                str(db_path),
            ]
            if persist_snapshot:
                bridge_cmd.append("--persist-snapshot")
            new_pid = _start_detached(bridge_cmd, cwd=root)
            if new_pid is not None:
                _write_pidfile(bridge_pidfile, new_pid)
                bridge_pid = new_pid
                started_bridge = True
            else:
                warnings.append("failed to start dashboard DB bridge")

    server_pid = _read_pidfile(server_pidfile)
    if server_pid is None:
        server_cmd = [
            python_bin,
            "-m",
            "http.server",
            str(port),
            "--bind",
            LOCALHOST_BIND,
            "--directory",
            str(root),
        ]
        new_pid = _start_detached(server_cmd, cwd=root)
        if new_pid is not None:
            _write_pidfile(server_pidfile, new_pid)
            server_pid = new_pid
            started_server = True
        else:
            warnings.append("failed to start local dashboard HTTP server")

    bridge_running = bool(bridge_pid and _pid_alive(bridge_pid))
    server_running = bool(server_pid and _pid_alive(server_pid))

    if server_running and not _port_open(LOCALHOST_BIND, port):
        warnings.append(f"HTTP server process exists but localhost:{port} is not reachable yet")

    if require_bridge and not bridge_running:
        warnings.append("bridge is required but not running")

    return EnsureResult(
        url=url,
        bridge_pid=bridge_pid,
        server_pid=server_pid,
        started_bridge=started_bridge,
        started_server=started_server,
        bridge_running=bridge_running,
        server_running=server_running,
        warnings=warnings,
    )


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cmd_ensure(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    python_bin = str(Path(args.python_bin).expanduser())
    port = int(args.port)

    if port < 1024 or port > 65535:
        print(f"[ERROR] Invalid dashboard port: {port}")
        return 1
    if not root.exists():
        print(f"[ERROR] Root path missing: {root}")
        return 1

    result = _ensure_dashboard_stack(
        root=root,
        python_bin=python_bin,
        port=port,
        db_path=db_path,
        persist_snapshot=bool(args.persist_snapshot),
        require_bridge=bool(args.require_bridge),
    )

    if bool(args.open_browser):
        try:
            webbrowser.open(result.url, new=2, autoraise=True)
        except Exception:
            result.warnings.append("failed to auto-open default browser")

    status: Dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "caller": str(args.caller or ""),
        "run_id": str(args.run_id or ""),
        "url": result.url,
        "root": str(root),
        "db_path": str(db_path),
        "python_bin": python_bin,
        "security": {
            "bind_host": LOCALHOST_BIND,
            "localhost_only": True,
        },
        "bridge": {
            "pid": result.bridge_pid,
            "running": result.bridge_running,
            "started_now": result.started_bridge,
        },
        "http_server": {
            "pid": result.server_pid,
            "running": result.server_running,
            "started_now": result.started_server,
            "port": port,
        },
        "warnings": result.warnings,
    }

    if args.status_json:
        _write_status(Path(args.status_json).expanduser().resolve(), status)

    print(f"[DASHBOARD] URL: {result.url}")
    print(
        f"[DASHBOARD] bridge_pid={result.bridge_pid} running={result.bridge_running} | "
        f"server_pid={result.server_pid} running={result.server_running}"
    )
    for warning in result.warnings:
        print(f"[DASHBOARD][WARN] {warning}")

    hard_fail = False
    if bool(args.strict):
        if not result.server_running:
            hard_fail = True
        if bool(args.require_bridge) and not result.bridge_running:
            hard_fail = True
    return 1 if hard_fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows dashboard process manager.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ensure = sub.add_parser("ensure", help="Ensure dashboard bridge + HTTP server are running.")
    p_ensure.add_argument("--root", default=str(ROOT), help="Repository root path.")
    p_ensure.add_argument("--python-bin", default=sys.executable, help="Python interpreter to spawn child services.")
    p_ensure.add_argument("--port", type=int, default=8000, help="Dashboard HTTP port (localhost-only).")
    p_ensure.add_argument("--db-path", default=str(ROOT / "data" / "portfolio_maximizer.db"), help="Trading DB path for bridge.")
    p_ensure.add_argument("--persist-snapshot", dest="persist_snapshot", action="store_true", default=True, help="Enable dashboard snapshot persistence.")
    p_ensure.add_argument("--no-persist-snapshot", dest="persist_snapshot", action="store_false", help="Disable dashboard snapshot persistence.")
    p_ensure.add_argument("--require-bridge", action="store_true", help="Fail in strict mode when bridge is not running.")
    p_ensure.add_argument("--open-browser", action="store_true", help="Auto-open dashboard URL in default browser.")
    p_ensure.add_argument("--status-json", default="", help="Optional status JSON output path.")
    p_ensure.add_argument("--caller", default="", help="Calling entrypoint label.")
    p_ensure.add_argument("--run-id", default="", help="Current run ID for audit traceability.")
    p_ensure.add_argument("--strict", dest="strict", action="store_true", default=True, help="Fail if required services are not running.")
    p_ensure.add_argument("--no-strict", dest="strict", action="store_false", help="Best-effort mode; never fail on service warnings.")

    args = parser.parse_args()
    if args.cmd == "ensure":
        return _cmd_ensure(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

