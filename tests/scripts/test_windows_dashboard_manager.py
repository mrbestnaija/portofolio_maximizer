from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts import windows_dashboard_manager as mod


def test_ensure_dashboard_stack_starts_live_watcher(tmp_path, monkeypatch) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "visualizations").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "visualizations" / "live_dashboard.html").write_text("<html></html>", encoding="utf-8")
    (root / "scripts" / "dashboard_db_bridge.py").write_text("# stub", encoding="utf-8")
    (root / "scripts" / "prometheus_alert_exporter.py").write_text("# stub", encoding="utf-8")
    (root / "scripts" / "run_live_denominator_overnight.py").write_text("# stub", encoding="utf-8")
    (root / "data" / "portfolio_maximizer.db").write_text("db", encoding="utf-8")

    pids = iter([101, 202, 303, 404])

    started_cmds: list[list[str]] = []

    monkeypatch.setattr(mod, "_read_pidfile", lambda path: None)

    def _fake_start(cmd, cwd):
        started_cmds.append(list(cmd))
        return next(pids)

    monkeypatch.setattr(mod, "_start_detached", _fake_start)
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(mod, "_port_open", lambda host, port: True)

    result = mod._ensure_dashboard_stack(
        root=root,
        python_bin="python",
        port=8000,
        prometheus_port=9108,
        db_path=root / "data" / "portfolio_maximizer.db",
        persist_snapshot=True,
        require_bridge=True,
        ensure_prometheus_exporter=True,
        ensure_live_watcher=True,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
    )

    assert result.bridge_running is True
    assert result.server_running is True
    assert result.exporter_running is True
    assert result.watcher_running is True
    assert result.started_exporter is True
    assert result.started_watcher is True
    assert started_cmds[0][:3] == ["python", "-m", "scripts.dashboard_db_bridge"]
    assert started_cmds[2][:3] == ["python", "-m", "scripts.prometheus_alert_exporter"]
    assert result.exporter_url == "http://127.0.0.1:9108/metrics"


def test_ensure_dashboard_stack_retries_bridge_without_persist_snapshot(tmp_path, monkeypatch) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "visualizations").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "visualizations" / "live_dashboard.html").write_text("<html></html>", encoding="utf-8")
    (root / "scripts" / "dashboard_db_bridge.py").write_text("# stub", encoding="utf-8")
    (root / "scripts" / "prometheus_alert_exporter.py").write_text("# stub", encoding="utf-8")
    (root / "scripts" / "run_live_denominator_overnight.py").write_text("# stub", encoding="utf-8")
    (root / "data" / "portfolio_maximizer.db").write_text("db", encoding="utf-8")

    started_cmds: list[list[str]] = []
    start_results = iter([None, 202, 303, 404, 505])

    monkeypatch.setattr(mod, "_read_pidfile", lambda path: None)

    def _fake_start(cmd, cwd):
        started_cmds.append(list(cmd))
        return next(start_results)

    monkeypatch.setattr(mod, "_start_detached", _fake_start)
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(mod, "_port_open", lambda host, port: True)

    result = mod._ensure_dashboard_stack(
        root=root,
        python_bin="python",
        port=8000,
        prometheus_port=9108,
        db_path=root / "data" / "portfolio_maximizer.db",
        persist_snapshot=True,
        require_bridge=True,
        ensure_prometheus_exporter=True,
        ensure_live_watcher=True,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
    )

    assert result.bridge_running is True
    assert result.exporter_running is True
    assert "--persist-snapshot" in started_cmds[0]
    assert "--persist-snapshot" not in started_cmds[1]
    assert started_cmds[3][:3] == ["python", "-m", "scripts.prometheus_alert_exporter"]
    assert any("retrying without audit snapshot persistence" in warning for warning in result.warnings)


def test_cmd_launch_refreshes_payload_and_opens_browser(tmp_path, monkeypatch) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "portfolio_maximizer.db").write_text("db", encoding="utf-8")
    status_json = root / "logs" / "dashboard_status.json"

    refresh_calls: list[dict[str, object]] = []
    ensure_calls: list[dict[str, object]] = []
    browser_calls: list[str] = []

    monkeypatch.setattr(
        mod,
        "_refresh_dashboard_payload",
        lambda **kwargs: refresh_calls.append(dict(kwargs)) or {
            "attempted": True,
            "ok": True,
            "persist_snapshot": True,
            "retried_without_persist_snapshot": False,
            "returncode": 0,
            "command": ["python", "-m", "scripts.dashboard_db_bridge", "--once"],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        mod,
        "_ensure_dashboard_stack",
        lambda **kwargs: ensure_calls.append(dict(kwargs)) or mod.EnsureResult(
            url="http://127.0.0.1:8000/visualizations/live_dashboard.html",
            bridge_pid=101,
            server_pid=202,
            watcher_pid=303,
            started_bridge=False,
            started_server=False,
            started_watcher=False,
            bridge_running=True,
            server_running=True,
            watcher_running=True,
            warnings=[],
            exporter_pid=404,
            started_exporter=False,
            exporter_running=True,
            exporter_url="http://127.0.0.1:9108/metrics",
        ),
    )
    monkeypatch.setattr(mod.webbrowser, "open", lambda url, new=2, autoraise=True: browser_calls.append(url) or True)

    args = Namespace(
        root=str(root),
        python_bin="python",
        port=8000,
        prometheus_port=9108,
        db_path=str(root / "data" / "portfolio_maximizer.db"),
        persist_snapshot=True,
        require_bridge=True,
        ensure_prometheus_exporter=True,
        ensure_live_watcher=False,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
        refresh_now=True,
        open_browser=True,
        status_json=str(status_json),
        caller="test_launch",
        run_id="RID-1",
        strict=True,
    )

    rc = mod._cmd_launch(args)

    assert rc == 0
    assert refresh_calls
    assert refresh_calls[0]["persist_snapshot"] is True
    assert ensure_calls
    assert ensure_calls[0]["ensure_prometheus_exporter"] is True
    assert ensure_calls[0]["ensure_live_watcher"] is False
    assert browser_calls == ["http://127.0.0.1:8000/visualizations/live_dashboard.html"]
    payload = json.loads(status_json.read_text(encoding="utf-8"))
    assert payload["refresh"]["ok"] is True
    assert payload["prometheus_exporter"]["running"] is True


def test_cmd_launch_fails_strict_when_refresh_fails(tmp_path, monkeypatch) -> None:
    root = tmp_path
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "portfolio_maximizer.db").write_text("db", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "_refresh_dashboard_payload",
        lambda **kwargs: {
            "attempted": True,
            "ok": False,
            "persist_snapshot": True,
            "retried_without_persist_snapshot": True,
            "returncode": 1,
            "command": ["python", "-m", "scripts.dashboard_db_bridge", "--once"],
            "warnings": ["refresh failed"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_ensure_dashboard_stack",
        lambda **kwargs: mod.EnsureResult(
            url="http://127.0.0.1:8000/visualizations/live_dashboard.html",
            bridge_pid=101,
            server_pid=202,
            watcher_pid=303,
            started_bridge=False,
            started_server=False,
            started_watcher=False,
            bridge_running=True,
            server_running=True,
            watcher_running=True,
            warnings=[],
            exporter_pid=404,
            started_exporter=False,
            exporter_running=True,
            exporter_url="http://127.0.0.1:9108/metrics",
        ),
    )
    monkeypatch.setattr(mod.webbrowser, "open", lambda url, new=2, autoraise=True: True)

    args = Namespace(
        root=str(root),
        python_bin="python",
        port=8000,
        prometheus_port=9108,
        db_path=str(root / "data" / "portfolio_maximizer.db"),
        persist_snapshot=True,
        require_bridge=True,
        ensure_prometheus_exporter=True,
        ensure_live_watcher=False,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
        refresh_now=True,
        open_browser=False,
        status_json="",
        caller="test_launch",
        run_id="RID-2",
        strict=True,
    )

    rc = mod._cmd_launch(args)

    assert rc == 1
