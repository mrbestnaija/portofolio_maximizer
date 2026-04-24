from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_weekly_sleeve_maintenance_runs_nav_sidecar() -> None:
    text = (_repo_root() / "bash" / "weekly_sleeve_maintenance.sh").read_text(encoding="utf-8")

    assert "scripts/evaluate_sleeve_promotions.py" in text
    assert "scripts.build_nav_rebalance_plan" in text
    assert " -m scripts.build_nav_rebalance_plan" in text
    assert "scripts/run_nav_rebalance_handoff.py" in text
    assert "NAV_REBALANCE_PATH" in text
    assert "NAV_HANDOFF_STATUS_PATH" in text
    assert "--sleeve-summary-path" in text
    assert "--lookback-days" in text
    assert "metrics_summary.json" not in text
    assert "shadow-first NAV rebalance plan" in text
