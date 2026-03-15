from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts import windows_dashboard_manager as dash_mod
from scripts import windows_persistence_manager as mod


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "logs" / "forecast_audits_cache").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "overnight_denominator").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "portfolio_maximizer.db").write_text("db", encoding="utf-8")
    (root / "logs" / "forecast_audits_cache" / "latest_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": "2026-03-08T08:00:00Z",
                "decision": "INCONCLUSIVE",
                "decision_reason": "effective_audits=16 < required_audits=20",
                "effective_audits": 16,
                "holding_period_required": 20,
                "lift_fraction": 0.125,
                "min_lift_fraction": 0.25,
                "outcome_join": {
                    "readiness_denominator_included": 3,
                    "linkage_denominator_included": 1,
                },
                "telemetry_contract": {
                    "outcomes_loaded": True,
                    "outcome_join_attempted": True,
                },
                "window_counts": {
                    "n_rmse_windows_usable": 16,
                    "n_outcome_windows_eligible": 1,
                    "n_outcome_windows_matched": 0,
                    "n_linkage_denominator_included": 1,
                    "n_outcome_windows_non_trade_context": 45,
                    "n_outcome_windows_invalid_context": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "logs" / "overnight_denominator" / "live_denominator_latest.json").write_text(
        json.dumps(
            {
                "run_meta": {
                    "run_id": "20260308_074125",
                    "started_utc": "2026-03-08T07:41:25Z",
                    "cycles": 30,
                    "weekdays_only": True,
                },
                "cycles": [
                    {
                        "fresh_trade_rows": 1,
                        "fresh_trade_context_rows_raw": 4,
                        "fresh_trade_exclusions": {
                            "non_trade_context": 0,
                            "invalid_context": 0,
                            "missing_execution_metadata": 0,
                        },
                        "fresh_trade_diagnostics": {"non_trade_context_rows": 3},
                        "fresh_linkage_included": 1,
                        "fresh_production_valid_matched": 0,
                        "progress_triggered": False,
                        "completed_utc": "2026-03-08T07:41:26Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_cmd_ensure_writes_status_and_succeeds(tmp_path, monkeypatch) -> None:
    root = _make_root(tmp_path)
    status_json = root / "logs" / "persistence_manager_status.json"

    monkeypatch.setattr(
        dash_mod,
        "_ensure_dashboard_stack",
        lambda **kwargs: dash_mod.EnsureResult(
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
        ),
    )

    counts = iter(
        [
            {"count": 2, "sample_ids": [11, 12], "error": None},
            {"count": 0, "sample_ids": [], "error": None},
        ]
    )
    monkeypatch.setattr(mod, "_count_unlinked_closes", lambda db_path: next(counts))
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda cmd, cwd, timeout_seconds=180.0: {
            "ok": True,
            "returncode": 0,
            "command": cmd,
            "stdout_tail": ["ok"],
            "stderr_tail": [],
        },
    )
    monkeypatch.setattr(
        mod,
        "_query_startup_registration",
        lambda: {
            "ok": True,
            "method": "registry_run_key",
            "returncode": 0,
            "details_tail": ["PortfolioMaximizer_PersistenceManager    REG_SZ    C:\\run_persistence_manager.bat"],
        },
    )

    args = Namespace(
        root=str(root),
        python_bin="python",
        db_path=str(root / "data" / "portfolio_maximizer.db"),
        audit_dir=str(root / "logs" / "forecast_audits"),
        audit_max_files=500,
        dashboard_port=8000,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
        reconcile_apply=True,
        status_json=str(status_json),
        strict=True,
        cmd="ensure",
    )

    rc = mod._cmd_ensure(args)
    assert rc == 0

    payload = json.loads(status_json.read_text(encoding="utf-8"))
    assert payload["dashboard"]["live_watcher"]["running"] is True
    assert payload["reconciliation"]["before"]["count"] == 2
    assert payload["reconciliation"]["after"]["count"] == 0
    assert payload["audit_refresh"]["summary"]["effective_audits"] == 16
    assert payload["watcher"]["fresh_linkage_included"] == 1
    assert payload["startup_registration"]["method"] == "registry_run_key"
    assert payload["startup_registration"]["ok"] is True


def test_cmd_ensure_fails_strict_when_unlinked_remain(tmp_path, monkeypatch) -> None:
    root = _make_root(tmp_path)
    status_json = root / "logs" / "persistence_manager_status.json"

    monkeypatch.setattr(
        dash_mod,
        "_ensure_dashboard_stack",
        lambda **kwargs: dash_mod.EnsureResult(
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
        ),
    )
    monkeypatch.setattr(
        mod,
        "_count_unlinked_closes",
        lambda db_path: {"count": 3, "sample_ids": [11, 12, 13], "error": None},
    )
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda cmd, cwd, timeout_seconds=180.0: {
            "ok": True,
            "returncode": 0,
            "command": cmd,
            "stdout_tail": ["ok"],
            "stderr_tail": [],
        },
    )
    monkeypatch.setattr(
        mod,
        "_query_startup_registration",
        lambda: {
            "ok": False,
            "method": "none",
            "returncode": 1,
            "details_tail": ["not registered"],
        },
    )

    args = Namespace(
        root=str(root),
        python_bin="python",
        db_path=str(root / "data" / "portfolio_maximizer.db"),
        audit_dir=str(root / "logs" / "forecast_audits"),
        audit_max_files=500,
        dashboard_port=8000,
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
        reconcile_apply=True,
        status_json=str(status_json),
        strict=True,
        cmd="ensure",
    )

    rc = mod._cmd_ensure(args)
    assert rc == 1


def test_build_task_command_targets_startup_supervisor(tmp_path) -> None:
    original_wrapper = mod.DEFAULT_TASK_WRAPPER
    mod.DEFAULT_TASK_WRAPPER = tmp_path / "scripts" / "run_persistence_manager.bat"
    cmd = mod._build_task_command(
        python_bin="C:\\Python314\\python.exe",
        script_path=tmp_path / "scripts" / "windows_persistence_manager.py",
        status_json=tmp_path / "logs" / "persistence_manager_status.json",
        db_path=tmp_path / "data" / "portfolio_maximizer.db",
        audit_dir=tmp_path / "logs" / "forecast_audits",
        watcher_tickers="AAPL,MSFT",
    )
    mod.DEFAULT_TASK_WRAPPER = original_wrapper

    assert "schtasks /Create" in cmd
    assert "/SC ONSTART" in cmd
    assert "run_persistence_manager.bat" in cmd


def test_register_startup_task_uses_schtasks(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    original_wrapper = mod.DEFAULT_TASK_WRAPPER
    mod.DEFAULT_TASK_WRAPPER = tmp_path / "scripts" / "run_persistence_manager.bat"

    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda cmd, cwd, timeout_seconds=180.0: calls.append(list(cmd)) or {
            "ok": True,
            "returncode": 0,
            "command": cmd,
            "stdout_tail": ["SUCCESS"],
            "stderr_tail": [],
        },
    )

    result = mod._register_startup_task(
        python_bin="C:\\Python314\\python.exe",
        script_path=tmp_path / "scripts" / "windows_persistence_manager.py",
        status_json=tmp_path / "logs" / "persistence_manager_status.json",
        db_path=tmp_path / "data" / "portfolio_maximizer.db",
        audit_dir=tmp_path / "logs" / "forecast_audits",
        watcher_tickers="AAPL,MSFT",
    )

    assert result["ok"] is True
    assert calls
    assert calls[0][0] == "schtasks"
    assert "/SC" in calls[0]
    assert "ONSTART" in calls[0]
    assert "run_persistence_manager.bat" in calls[0][calls[0].index("/TR") + 1]
    mod.DEFAULT_TASK_WRAPPER = original_wrapper


def test_register_startup_task_falls_back_to_registry(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    original_wrapper = mod.DEFAULT_TASK_WRAPPER
    mod.DEFAULT_TASK_WRAPPER = tmp_path / "scripts" / "run_persistence_manager.bat"

    def _fake_run(cmd, cwd, timeout_seconds=180.0):
        calls.append(list(cmd))
        if cmd[0] == "schtasks":
            return {
                "ok": False,
                "returncode": 1,
                "command": cmd,
                "stdout_tail": [],
                "stderr_tail": ["Access is denied."],
            }
        return {
            "ok": True,
            "returncode": 0,
            "command": cmd,
            "stdout_tail": ["The operation completed successfully."],
            "stderr_tail": [],
        }

    monkeypatch.setattr(mod, "_run_command", _fake_run)

    result = mod._register_startup_task(
        python_bin="C:\\Python314\\python.exe",
        script_path=tmp_path / "scripts" / "windows_persistence_manager.py",
        status_json=tmp_path / "logs" / "persistence_manager_status.json",
        db_path=tmp_path / "data" / "portfolio_maximizer.db",
        audit_dir=tmp_path / "logs" / "forecast_audits",
        watcher_tickers="AAPL,MSFT",
    )

    assert result["ok"] is True
    assert result["method"] == "registry_run_key"
    assert calls[-1][0] == "reg"
    mod.DEFAULT_TASK_WRAPPER = original_wrapper


def test_query_startup_registration_prefers_registry_fallback(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, cwd, timeout_seconds=180.0):
        calls.append(list(cmd))
        if cmd[0] == "schtasks":
            return {
                "ok": False,
                "returncode": 1,
                "command": cmd,
                "stdout_tail": [],
                "stderr_tail": ["ERROR: The system cannot find the file specified."],
            }
        return {
            "ok": True,
            "returncode": 0,
            "command": cmd,
            "stdout_tail": [
                "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                "    PortfolioMaximizer_PersistenceManager    REG_SZ    C:\\repo\\scripts\\run_persistence_manager.bat",
            ],
            "stderr_tail": [],
        }

    monkeypatch.setattr(mod, "_run_command", _fake_run)

    result = mod._query_startup_registration()

    assert result["ok"] is True
    assert result["method"] == "registry_run_key"
    assert calls[0][0] == "schtasks"
    assert calls[1][0] == "reg"
