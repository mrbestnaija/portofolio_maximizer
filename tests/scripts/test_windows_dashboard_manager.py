from __future__ import annotations

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
    (root / "scripts" / "run_live_denominator_overnight.py").write_text("# stub", encoding="utf-8")
    (root / "data" / "portfolio_maximizer.db").write_text("db", encoding="utf-8")

    pids = iter([101, 202, 303])

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
        db_path=root / "data" / "portfolio_maximizer.db",
        persist_snapshot=True,
        require_bridge=True,
        ensure_live_watcher=True,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
    )

    assert result.bridge_running is True
    assert result.server_running is True
    assert result.watcher_running is True
    assert result.started_watcher is True
    assert started_cmds[0][:3] == ["python", "-m", "scripts.dashboard_db_bridge"]


def test_ensure_dashboard_stack_retries_bridge_without_persist_snapshot(tmp_path, monkeypatch) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "visualizations").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "visualizations" / "live_dashboard.html").write_text("<html></html>", encoding="utf-8")
    (root / "scripts" / "dashboard_db_bridge.py").write_text("# stub", encoding="utf-8")
    (root / "scripts" / "run_live_denominator_overnight.py").write_text("# stub", encoding="utf-8")
    (root / "data" / "portfolio_maximizer.db").write_text("db", encoding="utf-8")

    started_cmds: list[list[str]] = []
    start_results = iter([None, 202, 303, 404])

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
        db_path=root / "data" / "portfolio_maximizer.db",
        persist_snapshot=True,
        require_bridge=True,
        ensure_live_watcher=True,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
    )

    assert result.bridge_running is True
    assert "--persist-snapshot" in started_cmds[0]
    assert "--persist-snapshot" not in started_cmds[1]
    assert any("retrying without audit snapshot persistence" in warning for warning in result.warnings)
