#!/usr/bin/env python3
"""
windows_dashboard_manager.py
----------------------------

Windows-native helper that ensures the live dashboard stack is running:
1) DB -> JSON bridge (`scripts/dashboard_db_bridge.py`)
2) Local static HTTP server (`python -m http.server`)
3) Local Prometheus alert exporter (`scripts/prometheus_alert_exporter.py`)

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
DEFAULT_PROMETHEUS_EXPORTER_PORT = 9108
DEFAULT_LIVE_WATCHER_TICKERS = ["AAPL", "AMZN", "GOOG", "GS", "JPM", "META", "MSFT", "NVDA", "TSLA", "V"]


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
    watcher_pid: Optional[int]
    started_bridge: bool
    started_server: bool
    started_watcher: bool
    bridge_running: bool
    server_running: bool
    watcher_running: bool
    warnings: list[str]
    exporter_pid: Optional[int] = None
    started_exporter: bool = False
    exporter_running: bool = False
    exporter_url: str = ""


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
    prometheus_port: int,
    db_path: Path,
    persist_snapshot: bool,
    require_bridge: bool,
    ensure_prometheus_exporter: bool,
    ensure_live_watcher: bool,
    watcher_tickers: str,
    watcher_cycles: int,
    watcher_sleep_seconds: int,
) -> EnsureResult:
    logs_dir = root / "logs"
    bridge_pidfile = logs_dir / "dashboard_bridge.pid"
    server_pidfile = logs_dir / f"dashboard_http_{port}.pid"
    exporter_pidfile = logs_dir / f"prometheus_alert_exporter_{prometheus_port}.pid"
    watcher_pidfile = logs_dir / "live_denominator.pid"

    bridge_script = root / "scripts" / "dashboard_db_bridge.py"
    exporter_script = root / "scripts" / "prometheus_alert_exporter.py"
    watcher_script = root / "scripts" / "run_live_denominator_overnight.py"
    dashboard_html = root / "visualizations" / "live_dashboard.html"
    url = f"http://{LOCALHOST_BIND}:{port}/visualizations/live_dashboard.html"
    exporter_url = f"http://{LOCALHOST_BIND}:{prometheus_port}/metrics"

    warnings: list[str] = []
    started_bridge = False
    started_server = False
    started_exporter = False
    started_watcher = False

    if not dashboard_html.exists():
        warnings.append(f"dashboard HTML missing: {dashboard_html}")
    if not bridge_script.exists():
        warnings.append(f"dashboard bridge missing: {bridge_script}")
    if ensure_prometheus_exporter and not exporter_script.exists():
        warnings.append(f"Prometheus exporter missing: {exporter_script}")
    if ensure_live_watcher and not watcher_script.exists():
        warnings.append(f"live watcher missing: {watcher_script}")

    bridge_pid = _read_pidfile(bridge_pidfile)
    if bridge_pid is None and bridge_script.exists():
        if not db_path.exists():
            warnings.append(f"bridge skipped; DB missing: {db_path}")
        else:
            bridge_cmd_base = [
                python_bin,
                "-m",
                "scripts.dashboard_db_bridge",
                "--interval-seconds",
                "5",
                "--db-path",
                str(db_path),
            ]
            bridge_cmd = list(bridge_cmd_base)
            if persist_snapshot:
                bridge_cmd.append("--persist-snapshot")
            new_pid = _start_detached(bridge_cmd, cwd=root)
            if new_pid is None and persist_snapshot:
                warnings.append("dashboard bridge persist snapshot unavailable; retrying without audit snapshot persistence")
                new_pid = _start_detached(bridge_cmd_base, cwd=root)
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

    exporter_pid = _read_pidfile(exporter_pidfile)
    if ensure_prometheus_exporter and exporter_pid is None and exporter_script.exists():
        exporter_cmd = [
            python_bin,
            "-m",
            "scripts.prometheus_alert_exporter",
            "--listen-host",
            LOCALHOST_BIND,
            "--port",
            str(int(prometheus_port)),
            "--dashboard-json",
            str(root / "visualizations" / "dashboard_data.json"),
            "--production-gate-json",
            str(root / "logs" / "audit_gate" / "production_gate_latest.json"),
            "--audit-db",
            str(root / "data" / "dashboard_audit.db"),
        ]
        new_pid = _start_detached(exporter_cmd, cwd=root)
        if new_pid is not None:
            _write_pidfile(exporter_pidfile, new_pid)
            exporter_pid = new_pid
            started_exporter = True
        else:
            warnings.append("failed to start Prometheus alert exporter")

    exporter_running = bool(exporter_pid and _pid_alive(exporter_pid))

    watcher_pid = _read_pidfile(watcher_pidfile)
    if ensure_live_watcher and watcher_pid is None and watcher_script.exists():
        watcher_cmd = [
            python_bin,
            str(watcher_script),
            "--tickers",
            watcher_tickers,
            "--cycles",
            str(int(watcher_cycles)),
            "--sleep-seconds",
            str(int(watcher_sleep_seconds)),
            "--resume",
            "--stop-on-progress",
            "--db",
            str(db_path),
        ]
        new_pid = _start_detached(watcher_cmd, cwd=root)
        if new_pid is not None:
            _write_pidfile(watcher_pidfile, new_pid)
            watcher_pid = new_pid
            started_watcher = True
        else:
            warnings.append("failed to start live denominator watcher")

    watcher_running = bool(watcher_pid and _pid_alive(watcher_pid))

    if server_running and not _port_open(LOCALHOST_BIND, port):
        warnings.append(f"HTTP server process exists but localhost:{port} is not reachable yet")
    if exporter_running and not _port_open(LOCALHOST_BIND, prometheus_port):
        warnings.append(f"Prometheus exporter process exists but localhost:{prometheus_port} is not reachable yet")

    if require_bridge and not bridge_running:
        warnings.append("bridge is required but not running")
    if ensure_prometheus_exporter and not exporter_running:
        warnings.append("Prometheus exporter is required but not running")
    if ensure_live_watcher and not watcher_running:
        warnings.append("live denominator watcher is required but not running")

    return EnsureResult(
        url=url,
        bridge_pid=bridge_pid,
        server_pid=server_pid,
        watcher_pid=watcher_pid,
        started_bridge=started_bridge,
        started_server=started_server,
        started_watcher=started_watcher,
        bridge_running=bridge_running,
        server_running=server_running,
        watcher_running=watcher_running,
        warnings=warnings,
        exporter_pid=exporter_pid,
        started_exporter=started_exporter,
        exporter_running=exporter_running,
        exporter_url=exporter_url,
    )


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validate_args(
    *,
    root: Path,
    port: int,
    prometheus_port: int,
    ensure_prometheus_exporter: bool,
) -> int:
    if port < 1024 or port > 65535:
        print(f"[ERROR] Invalid dashboard port: {port}")
        return 1
    if prometheus_port < 1024 or prometheus_port > 65535:
        print(f"[ERROR] Invalid Prometheus exporter port: {prometheus_port}")
        return 1
    if bool(ensure_prometheus_exporter) and port == prometheus_port:
        print("[ERROR] Dashboard HTTP port and Prometheus exporter port must differ")
        return 1
    if not root.exists():
        print(f"[ERROR] Root path missing: {root}")
        return 1
    return 0


def _refresh_dashboard_payload(
    *,
    root: Path,
    python_bin: str,
    db_path: Path,
    persist_snapshot: bool,
) -> Dict[str, Any]:
    bridge_script = root / "scripts" / "dashboard_db_bridge.py"
    refresh_status: Dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "persist_snapshot": bool(persist_snapshot),
        "retried_without_persist_snapshot": False,
        "returncode": None,
        "command": [],
        "warnings": [],
    }
    if not bridge_script.exists():
        refresh_status["warnings"].append(f"dashboard bridge missing: {bridge_script}")
        return refresh_status
    if not db_path.exists():
        refresh_status["warnings"].append(f"dashboard refresh skipped; DB missing: {db_path}")
        return refresh_status

    cmd_base = [
        python_bin,
        "-m",
        "scripts.dashboard_db_bridge",
        "--once",
        "--db-path",
        str(db_path),
    ]
    cmd = list(cmd_base)
    if persist_snapshot:
        cmd.append("--persist-snapshot")
    refresh_status["command"] = list(cmd)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except Exception as exc:
        refresh_status["returncode"] = 127
        refresh_status["warnings"].append(f"dashboard refresh failed to execute: {exc}")
        return refresh_status

    refresh_status["returncode"] = int(proc.returncode)
    if int(proc.returncode) == 0:
        refresh_status["ok"] = True
        return refresh_status

    stderr_tail = [line.strip() for line in (proc.stderr or "").splitlines()[-4:] if line.strip()]
    if persist_snapshot:
        refresh_status["warnings"].append(
            "dashboard refresh with audit snapshot persistence failed; retrying without persistence"
        )
        try:
            retry = subprocess.run(
                cmd_base,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
        except Exception as exc:
            refresh_status["returncode"] = 127
            refresh_status["warnings"].append(f"dashboard refresh retry failed to execute: {exc}")
            return refresh_status
        refresh_status["retried_without_persist_snapshot"] = True
        refresh_status["returncode"] = int(retry.returncode)
        refresh_status["command"] = list(cmd_base)
        if int(retry.returncode) == 0:
            refresh_status["ok"] = True
            return refresh_status
        stderr_tail = [line.strip() for line in (retry.stderr or "").splitlines()[-4:] if line.strip()]

    if stderr_tail:
        refresh_status["warnings"].append(f"dashboard refresh stderr: {' | '.join(stderr_tail)}")
    return refresh_status


def _build_status_payload(
    *,
    args: argparse.Namespace,
    root: Path,
    db_path: Path,
    python_bin: str,
    port: int,
    prometheus_port: int,
    result: EnsureResult,
    refresh_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        "prometheus_exporter": {
            "pid": result.exporter_pid,
            "running": result.exporter_running,
            "started_now": result.started_exporter,
            "port": prometheus_port,
            "url": result.exporter_url,
        },
        "live_watcher": {
            "pid": result.watcher_pid,
            "running": result.watcher_running,
            "started_now": result.started_watcher,
            "tickers": [t.strip().upper() for t in str(args.watcher_tickers).split(",") if t.strip()],
            "sleep_seconds": int(args.watcher_sleep_seconds),
        },
        "warnings": list(result.warnings),
    }
    if refresh_status is not None:
        status["refresh"] = dict(refresh_status)
        status["warnings"].extend(list(refresh_status.get("warnings") or []))
    return status


def _print_status(*, result: EnsureResult, status: Dict[str, Any], refresh_status: Optional[Dict[str, Any]] = None) -> None:
    print(f"[DASHBOARD] URL: {result.url}")
    print(f"[PROMETHEUS] URL: {result.exporter_url}")
    if refresh_status is not None:
        print(
            f"[DASHBOARD] refresh_ok={bool(refresh_status.get('ok'))} "
            f"rc={refresh_status.get('returncode')}"
        )
    print(
        f"[DASHBOARD] bridge_pid={result.bridge_pid} running={result.bridge_running} | "
        f"server_pid={result.server_pid} running={result.server_running} | "
        f"exporter_pid={result.exporter_pid} running={result.exporter_running} | "
        f"watcher_pid={result.watcher_pid} running={result.watcher_running}"
    )
    for warning in status["warnings"]:
        print(f"[DASHBOARD][WARN] {warning}")


def _cmd_ensure(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    python_bin = str(Path(args.python_bin).expanduser())
    port = int(args.port)
    prometheus_port = int(args.prometheus_port)

    rc = _validate_args(
        root=root,
        port=port,
        prometheus_port=prometheus_port,
        ensure_prometheus_exporter=bool(args.ensure_prometheus_exporter),
    )
    if rc != 0:
        return rc

    result = _ensure_dashboard_stack(
        root=root,
        python_bin=python_bin,
        port=port,
        prometheus_port=prometheus_port,
        db_path=db_path,
        persist_snapshot=bool(args.persist_snapshot),
        require_bridge=bool(args.require_bridge),
        ensure_prometheus_exporter=bool(args.ensure_prometheus_exporter),
        ensure_live_watcher=bool(args.ensure_live_watcher),
        watcher_tickers=str(args.watcher_tickers),
        watcher_cycles=int(args.watcher_cycles),
        watcher_sleep_seconds=int(args.watcher_sleep_seconds),
    )

    if bool(args.open_browser):
        try:
            webbrowser.open(result.url, new=2, autoraise=True)
        except Exception:
            result.warnings.append("failed to auto-open default browser")

    status = _build_status_payload(
        args=args,
        root=root,
        db_path=db_path,
        python_bin=python_bin,
        port=port,
        prometheus_port=prometheus_port,
        result=result,
    )

    if args.status_json:
        _write_status(Path(args.status_json).expanduser().resolve(), status)

    _print_status(result=result, status=status)

    hard_fail = False
    if bool(args.strict):
        if not result.server_running:
            hard_fail = True
        if bool(args.require_bridge) and not result.bridge_running:
            hard_fail = True
        if bool(args.ensure_prometheus_exporter) and not result.exporter_running:
            hard_fail = True
        if bool(args.ensure_live_watcher) and not result.watcher_running:
            hard_fail = True
    return 1 if hard_fail else 0


def _cmd_launch(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    python_bin = str(Path(args.python_bin).expanduser())
    port = int(args.port)
    prometheus_port = int(args.prometheus_port)

    rc = _validate_args(
        root=root,
        port=port,
        prometheus_port=prometheus_port,
        ensure_prometheus_exporter=bool(args.ensure_prometheus_exporter),
    )
    if rc != 0:
        return rc

    refresh_status: Optional[Dict[str, Any]] = None
    if bool(args.refresh_now):
        refresh_status = _refresh_dashboard_payload(
            root=root,
            python_bin=python_bin,
            db_path=db_path,
            persist_snapshot=bool(args.persist_snapshot),
        )

    result = _ensure_dashboard_stack(
        root=root,
        python_bin=python_bin,
        port=port,
        prometheus_port=prometheus_port,
        db_path=db_path,
        persist_snapshot=bool(args.persist_snapshot),
        require_bridge=bool(args.require_bridge),
        ensure_prometheus_exporter=bool(args.ensure_prometheus_exporter),
        ensure_live_watcher=bool(args.ensure_live_watcher),
        watcher_tickers=str(args.watcher_tickers),
        watcher_cycles=int(args.watcher_cycles),
        watcher_sleep_seconds=int(args.watcher_sleep_seconds),
    )

    if bool(args.open_browser):
        try:
            webbrowser.open(result.url, new=2, autoraise=True)
        except Exception:
            result.warnings.append("failed to auto-open default browser")

    status = _build_status_payload(
        args=args,
        root=root,
        db_path=db_path,
        python_bin=python_bin,
        port=port,
        prometheus_port=prometheus_port,
        result=result,
        refresh_status=refresh_status,
    )

    if args.status_json:
        _write_status(Path(args.status_json).expanduser().resolve(), status)

    _print_status(result=result, status=status, refresh_status=refresh_status)

    hard_fail = False
    if bool(args.strict):
        if refresh_status is not None and not bool(refresh_status.get("ok")):
            hard_fail = True
        if not result.server_running:
            hard_fail = True
        if bool(args.require_bridge) and not result.bridge_running:
            hard_fail = True
        if bool(args.ensure_prometheus_exporter) and not result.exporter_running:
            hard_fail = True
        if bool(args.ensure_live_watcher) and not result.watcher_running:
            hard_fail = True
    return 1 if hard_fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows dashboard process manager.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ensure = sub.add_parser("ensure", help="Ensure dashboard bridge + HTTP server are running.")
    p_ensure.add_argument("--root", default=str(ROOT), help="Repository root path.")
    p_ensure.add_argument("--python-bin", default=sys.executable, help="Python interpreter to spawn child services.")
    p_ensure.add_argument("--port", type=int, default=8000, help="Dashboard HTTP port (localhost-only).")
    p_ensure.add_argument("--prometheus-port", type=int, default=DEFAULT_PROMETHEUS_EXPORTER_PORT, help="Prometheus exporter port (localhost-only).")
    p_ensure.add_argument("--db-path", default=str(ROOT / "data" / "portfolio_maximizer.db"), help="Trading DB path for bridge.")
    p_ensure.add_argument("--persist-snapshot", dest="persist_snapshot", action="store_true", default=True, help="Enable dashboard snapshot persistence.")
    p_ensure.add_argument("--no-persist-snapshot", dest="persist_snapshot", action="store_false", help="Disable dashboard snapshot persistence.")
    p_ensure.add_argument("--require-bridge", action="store_true", help="Fail in strict mode when bridge is not running.")
    p_ensure.add_argument("--ensure-prometheus-exporter", dest="ensure_prometheus_exporter", action="store_true", default=True, help="Ensure the localhost Prometheus alert exporter is running (default: on).")
    p_ensure.add_argument("--no-prometheus-exporter", dest="ensure_prometheus_exporter", action="store_false", help="Do not manage the Prometheus alert exporter.")
    p_ensure.add_argument("--ensure-live-watcher", dest="ensure_live_watcher", action="store_true", default=True, help="Ensure the live denominator watcher is running (default: on).")
    p_ensure.add_argument("--no-live-watcher", dest="ensure_live_watcher", action="store_false", help="Do not manage the live denominator watcher.")
    p_ensure.add_argument("--watcher-tickers", default=",".join(DEFAULT_LIVE_WATCHER_TICKERS), help="Comma-separated live watcher ticker universe.")
    p_ensure.add_argument("--watcher-cycles", type=int, default=30, help="Watcher cycles to schedule before exit.")
    p_ensure.add_argument("--watcher-sleep-seconds", type=int, default=86400, help="Watcher cadence in seconds (default: 86400).")
    p_ensure.add_argument("--open-browser", dest="open_browser", action="store_true", default=False, help="Auto-open dashboard URL in default browser.")
    p_ensure.add_argument("--no-open-browser", dest="open_browser", action="store_false", help="Do not auto-open dashboard URL in default browser.")
    p_ensure.add_argument("--status-json", default="", help="Optional status JSON output path.")
    p_ensure.add_argument("--caller", default="", help="Calling entrypoint label.")
    p_ensure.add_argument("--run-id", default="", help="Current run ID for audit traceability.")
    p_ensure.add_argument("--strict", dest="strict", action="store_true", default=True, help="Fail if required services are not running.")
    p_ensure.add_argument("--no-strict", dest="strict", action="store_false", help="Best-effort mode; never fail on service warnings.")

    p_launch = sub.add_parser("launch", help="Refresh payload, ensure the full local dashboard stack, and open the browser.")
    p_launch.add_argument("--root", default=str(ROOT), help="Repository root path.")
    p_launch.add_argument("--python-bin", default=sys.executable, help="Python interpreter to spawn child services.")
    p_launch.add_argument("--port", type=int, default=8000, help="Dashboard HTTP port (localhost-only).")
    p_launch.add_argument("--prometheus-port", type=int, default=DEFAULT_PROMETHEUS_EXPORTER_PORT, help="Prometheus exporter port (localhost-only).")
    p_launch.add_argument("--db-path", default=str(ROOT / "data" / "portfolio_maximizer.db"), help="Trading DB path for bridge.")
    p_launch.add_argument("--persist-snapshot", dest="persist_snapshot", action="store_true", default=True, help="Enable dashboard snapshot persistence.")
    p_launch.add_argument("--no-persist-snapshot", dest="persist_snapshot", action="store_false", help="Disable dashboard snapshot persistence.")
    p_launch.add_argument("--require-bridge", action="store_true", default=True, help="Require the dashboard bridge to be running (default: on).")
    p_launch.add_argument("--ensure-prometheus-exporter", dest="ensure_prometheus_exporter", action="store_true", default=True, help="Ensure the localhost Prometheus alert exporter is running (default: on).")
    p_launch.add_argument("--no-prometheus-exporter", dest="ensure_prometheus_exporter", action="store_false", help="Do not manage the Prometheus alert exporter.")
    p_launch.add_argument("--ensure-live-watcher", dest="ensure_live_watcher", action="store_true", default=False, help="Also ensure the live denominator watcher is running (default: off for human launch).")
    p_launch.add_argument("--no-live-watcher", dest="ensure_live_watcher", action="store_false", help="Do not manage the live denominator watcher.")
    p_launch.add_argument("--watcher-tickers", default=",".join(DEFAULT_LIVE_WATCHER_TICKERS), help="Comma-separated live watcher ticker universe.")
    p_launch.add_argument("--watcher-cycles", type=int, default=30, help="Watcher cycles to schedule before exit.")
    p_launch.add_argument("--watcher-sleep-seconds", type=int, default=86400, help="Watcher cadence in seconds (default: 86400).")
    p_launch.add_argument("--refresh-now", dest="refresh_now", action="store_true", default=True, help="Refresh dashboard_data.json immediately before launch (default: on).")
    p_launch.add_argument("--no-refresh-now", dest="refresh_now", action="store_false", help="Skip the immediate payload refresh step.")
    p_launch.add_argument("--open-browser", dest="open_browser", action="store_true", default=True, help="Open dashboard URL in default browser (default: on).")
    p_launch.add_argument("--no-open-browser", dest="open_browser", action="store_false", help="Do not auto-open dashboard URL in default browser.")
    p_launch.add_argument("--status-json", default="", help="Optional status JSON output path.")
    p_launch.add_argument("--caller", default="launch_live_dashboard", help="Calling entrypoint label.")
    p_launch.add_argument("--run-id", default="", help="Current run ID for audit traceability.")
    p_launch.add_argument("--strict", dest="strict", action="store_true", default=True, help="Fail if refresh or required services are not running.")
    p_launch.add_argument("--no-strict", dest="strict", action="store_false", help="Best-effort mode; never fail on refresh/service warnings.")

    args = parser.parse_args()
    if args.cmd == "ensure":
        return _cmd_ensure(args)
    if args.cmd == "launch":
        return _cmd_launch(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
