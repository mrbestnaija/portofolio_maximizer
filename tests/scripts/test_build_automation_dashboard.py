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
            "schema_version": 4,
            "gate": {
                "freshness_status": {"status": "fresh", "age_minutes": 15.0, "expected_max_age_minutes": 1440.0},
                "warmup_state": {"posture": "expired", "deadline_utc": "2026-04-24T20:00:00Z", "matched_needed": 0},
                "trajectory_alarm": {"active": False},
            },
            "summary": {
                "ann_roi_pct": 9.86,
                "roi_ann_pct": 9.86,
                "deployment_pct": 1.83,
                "objective_score": 18.05,
                "objective_valid": True,
                "ngn_hurdle_pct": 28.0,
                "gap_to_hurdle_pp": 18.14,
                "unattended_gate": "FAIL",
                "unattended_ready": False,
                "evidence_health": "clean",
            },
            "alpha_objective": {
                "roi_ann_pct": 9.86,
                "deployment_pct": 1.83,
                "objective_score": 18.05,
                "objective_valid": True,
            },
            "alpha_model_quality": {
                "status": "available",
                "target_amplitude_hit_rate": 0.75,
                "target_amplitude_hit_rate_rolling_20": 0.80,
                "target_amplitude_hit_count": 8,
                "target_amplitude_support": 8,
                "domain_objective_version": "v1.0.0",
            },
            "thin_linkage": {"matched_current": 10, "matched_needed": 0},
            "source_contract": {
                "status": "clean",
                "canonical_sources": [
                    {"metric": "closed_pnl", "source_file": "production_closed_trades", "query_or_key": "production_closed_trades"}
                ],
                "allowlisted_readers": ["scripts/build_automation_dashboard.py"],
                "violations_found": [],
                "scan_timestamp_utc": "2026-04-18T12:00:00Z",
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
    assert payload["inputs"]["canonical_snapshot"]["schema_version"] == 4
    assert payload["inputs"]["canonical_snapshot"]["summary"]["ann_roi_pct"] == pytest.approx(9.86)
    assert payload["inputs"]["canonical_snapshot"]["alpha_model_quality"]["status"] == "available"
    assert payload["inputs"]["canonical_snapshot_contract"]["ok"] is True
    assert payload["inputs"]["canonical_snapshot_contract"]["alpha_model_quality_status"] == "available"
