from pathlib import Path

import pytest
import yaml

from scripts import validate_forecasting_configs as vfc


def test_validate_configs_passes_for_repo_defaults() -> None:
    report = vfc.validate_configs(
        forecasting_config_path=Path("config/forecasting_config.yml"),
        pipeline_config_path=Path("config/pipeline_config.yml"),
    )
    assert report["ok"] is True, report["errors"]
    assert not report["errors"]


def test_validate_configs_fails_when_required_sections_missing(tmp_path: Path) -> None:
    broken_forecasting = tmp_path / "forecasting.yml"
    broken_pipeline = tmp_path / "pipeline.yml"

    broken_forecasting.write_text(
        yaml.safe_dump({"forecasting": {"ensemble": {"candidate_weights": []}}}),
        encoding="utf-8",
    )
    broken_pipeline.write_text(
        yaml.safe_dump({"pipeline": {"forecasting": {"ensemble": {"candidate_weights": []}}}}),
        encoding="utf-8",
    )

    report = vfc.validate_configs(
        forecasting_config_path=broken_forecasting,
        pipeline_config_path=broken_pipeline,
    )
    assert report["ok"] is False
    assert any("missing required section" in msg for msg in report["errors"])


def test_validate_configs_flags_sarimax_in_regime_candidates(tmp_path: Path) -> None:
    forecasting = tmp_path / "forecasting.yml"
    pipeline = tmp_path / "pipeline.yml"

    base_forecasting = {
        "forecasting": {
            "sarimax": {"enabled": False},
            "garch": {},
            "samossa": {},
            "mssa_rl": {},
            "ensemble": {
                "minimum_component_weight": 0.05,
                "candidate_weights": [{"garch": 0.5, "samossa": 0.5}],
            },
            "regime_detection": {
                "regime_candidate_weights": {
                    "CRISIS": [{"sarimax": 0.3, "garch": 0.7}],
                }
            },
        }
    }
    base_pipeline = {
        "pipeline": {
            "forecasting": {
                "sarimax": {"enabled": False},
                "garch": {},
                "samossa": {},
                "mssa_rl": {},
                "ensemble": {
                    "minimum_component_weight": 0.05,
                    "candidate_weights": [{"garch": 0.5, "samossa": 0.5}],
                },
                "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}}
            }
        }
    }

    forecasting.write_text(yaml.safe_dump(base_forecasting), encoding="utf-8")
    pipeline.write_text(yaml.safe_dump(base_pipeline), encoding="utf-8")

    report = vfc.validate_configs(forecasting, pipeline)
    assert report["ok"] is False
    assert any("includes disabled 'sarimax'" in msg for msg in report["errors"])


def test_validate_configs_flags_sarimax_in_ensemble_candidates(tmp_path: Path) -> None:
    forecasting = tmp_path / "forecasting.yml"
    pipeline = tmp_path / "pipeline.yml"

    base_forecasting = {
        "forecasting": {
            "sarimax": {"enabled": False},
            "garch": {"enabled": True},
            "samossa": {"enabled": True},
            "mssa_rl": {"enabled": True},
            "ensemble": {
                "minimum_component_weight": 0.05,
                "candidate_weights": [{"sarimax": 0.2, "garch": 0.8}],
            },
            "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}},
        }
    }
    base_pipeline = {
        "pipeline": {
            "forecasting": {
                "sarimax": {"enabled": False},
                "garch": {"enabled": True},
                "samossa": {"enabled": True},
                "mssa_rl": {"enabled": True},
                "ensemble": {
                    "minimum_component_weight": 0.05,
                    "candidate_weights": [{"garch": 1.0}],
                },
                "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}}
            }
        }
    }

    forecasting.write_text(yaml.safe_dump(base_forecasting), encoding="utf-8")
    pipeline.write_text(yaml.safe_dump(base_pipeline), encoding="utf-8")

    report = vfc.validate_configs(forecasting, pipeline)
    assert report["ok"] is False
    assert any("candidate_weights[0] includes disabled 'sarimax'" in msg for msg in report["errors"])
