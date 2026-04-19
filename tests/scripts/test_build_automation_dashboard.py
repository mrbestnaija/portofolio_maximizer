from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts import build_automation_dashboard as mod


class _FakeDatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_best_strategy_config(self):
        return {"name": "shadow-best"}

    def close(self) -> None:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_dashboard_snapshot_includes_nav_rebalance_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "ROOT_PATH", tmp_path)
    monkeypatch.setattr(mod, "DatabaseManager", _FakeDatabaseManager)

    nav_plan_path = tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest.json"
    _write_json(
        nav_plan_path,
        {
            "meta": {"generated_utc": "2026-04-18T12:00:00Z"},
            "rollout": {"mode": "shadow", "live_apply_allowed": False},
            "bucket_allocations": [],
            "targets": [],
            "summary": {"healthy": ["NVDA"], "weak": ["AAPL"], "lab_only": [], "promotions": ["NVDA"], "demotions": ["AAPL"], "total_targets": 2},
            "evidence": {},
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        mod.main,
        [
            "--db-path",
            str(tmp_path / "data" / "portfolio_maximizer.db"),
            "--nav-rebalance-path",
            "logs/automation/nav_rebalance_plan_latest.json",
            "--output",
            "visualizations/dashboard_automation.json",
        ],
    )

    assert result.exit_code == 0, result.output

    dashboard_path = tmp_path / "visualizations" / "dashboard_automation.json"
    assert dashboard_path.exists()
    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert payload["models"]["best_strategy_config"] == {"name": "shadow-best"}
    assert payload["inputs"]["nav_rebalance_plan"]["rollout"]["mode"] == "shadow"
    assert payload["inputs"]["nav_rebalance_plan"]["summary"]["healthy"] == ["NVDA"]


def test_dashboard_snapshot_includes_canonical_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "ROOT_PATH", tmp_path)
    monkeypatch.setattr(mod, "DatabaseManager", _FakeDatabaseManager)

    canonical_path = tmp_path / "logs" / "canonical_snapshot_latest.json"
    _write_json(
        canonical_path,
        {
            "schema_version": 2,
            "summary": {
                "ann_roi_pct": 9.86,
                "ngn_hurdle_pct": 28.0,
                "gap_to_hurdle_pp": 18.14,
                "unattended_gate": "FAIL",
                "unattended_ready": False,
            },
            "source_contract": {
                "canonical": {"closed_pnl": "production_closed_trades"},
                "ui_only": {"metrics_summary": "visualizations/performance/metrics_summary.json"},
            },
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        mod.main,
        [
            "--db-path",
            str(tmp_path / "data" / "portfolio_maximizer.db"),
            "--canonical-snapshot-path",
            "logs/canonical_snapshot_latest.json",
            "--output",
            "visualizations/dashboard_automation.json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "visualizations" / "dashboard_automation.json").read_text(encoding="utf-8"))
    assert payload["inputs"]["canonical_snapshot"]["schema_version"] == 2
    assert payload["inputs"]["canonical_snapshot"]["summary"]["ann_roi_pct"] == pytest.approx(9.86)
