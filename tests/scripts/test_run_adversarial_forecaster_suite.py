from __future__ import annotations

from pathlib import Path

import scripts.run_adversarial_forecaster_suite as mod


def test_evaluate_thresholds_flags_breaches() -> None:
    summary = {
        "prod_like_conf_off": {
            "errors": 0,
            "ensemble_under_best_rate": 1.0,
            "avg_ensemble_ratio_vs_best": 1.30,
            "ensemble_worse_than_rw_rate": 0.66,
        }
    }
    thresholds = {
        "max_ensemble_under_best_rate": 1.0,
        "max_avg_ensemble_ratio_vs_best": 1.2,
        "max_ensemble_worse_than_rw_rate": 0.3,
        "require_zero_errors": True,
    }
    breaches = mod.evaluate_thresholds(summary, thresholds)
    assert any("avg_ensemble_ratio_vs_best" in item for item in breaches)
    assert any("ensemble_worse_than_rw_rate" in item for item in breaches)


def test_load_thresholds_from_monitor_config(tmp_path: Path) -> None:
    cfg = tmp_path / "forecaster_monitoring_ci.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    adversarial_suite:",
                "      max_ensemble_under_best_rate: 0.95",
                "      max_avg_ensemble_ratio_vs_best: 1.10",
                "      max_ensemble_worse_than_rw_rate: 0.20",
                "      require_zero_errors: true",
            ]
        ),
        encoding="utf-8",
    )
    thresholds = mod._load_thresholds(cfg)
    assert thresholds["max_ensemble_under_best_rate"] == 0.95
    assert thresholds["max_avg_ensemble_ratio_vs_best"] == 1.10
    assert thresholds["max_ensemble_worse_than_rw_rate"] == 0.20
    assert thresholds["require_zero_errors"] is True

