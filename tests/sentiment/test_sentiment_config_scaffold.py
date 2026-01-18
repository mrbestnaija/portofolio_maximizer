from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sentiment.yml"


def test_sentiment_config_defaults_are_safe_and_gated():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    # Hard gating: off by default and profit thresholds remain strict
    assert cfg["enabled"] is False
    gating = cfg["gating"]
    assert gating["min_sharpe"] >= 1.1
    assert gating["max_drawdown"] <= 0.22
    assert all(day in gating["require_positive_pnl_days"] for day in (90, 180))
    assert gating["min_win_rate"] >= 0.52

    # Ops and adjustments stay conservative in dormant mode
    assert cfg["ops"]["dry_run"] is True
    assert cfg["adjustments"]["max_position_nudge_pct"] <= 0.15
