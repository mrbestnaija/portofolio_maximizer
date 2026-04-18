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
    """When sarimax is disabled, regime candidates with sarimax should be flagged."""
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


def test_validate_configs_allows_sarimax_in_regime_candidates_when_enabled(tmp_path: Path) -> None:
    """When sarimax is enabled, sarimax in regime candidates should pass validation."""
    forecasting = tmp_path / "forecasting.yml"
    pipeline = tmp_path / "pipeline.yml"

    base_forecasting = {
        "forecasting": {
            "sarimax": {"enabled": True},
            "garch": {},
            "samossa": {},
            "mssa_rl": {},
            "ensemble": {
                "minimum_component_weight": 0.05,
                "candidate_weights": [{"sarimax": 0.5, "garch": 0.5}],
            },
            "regime_detection": {
                "regime_candidate_weights": {
                    "CRISIS": [{"sarimax": 0.45, "garch": 0.40, "mssa_rl": 0.15}],
                }
            },
        }
    }
    base_pipeline = {
        "pipeline": {
            "forecasting": {
                "sarimax": {"enabled": True},
                "garch": {},
                "samossa": {},
                "mssa_rl": {},
                "ensemble": {
                    "minimum_component_weight": 0.05,
                    "candidate_weights": [{"sarimax": 0.5, "garch": 0.5}],
                },
                "regime_detection": {
                    "regime_candidate_weights": {
                        "CRISIS": [{"sarimax": 0.45, "garch": 0.40, "mssa_rl": 0.15}],
                    }
                }
            }
        }
    }

    forecasting.write_text(yaml.safe_dump(base_forecasting), encoding="utf-8")
    pipeline.write_text(yaml.safe_dump(base_pipeline), encoding="utf-8")

    report = vfc.validate_configs(forecasting, pipeline)
    assert report["ok"] is True, f"Enabled SARIMAX in regime candidates should pass: {report['errors']}"


def test_validate_configs_accepts_sarimax_enabled_true(tmp_path: Path) -> None:
    """Validator accepts sarimax.enabled=True (was previously erroring)."""
    forecasting = tmp_path / "forecasting.yml"
    pipeline = tmp_path / "pipeline.yml"

    base = {
        "sarimax": {"enabled": True},
        "garch": {"enabled": True},
        "samossa": {"enabled": True},
        "mssa_rl": {"enabled": True},
        "ensemble": {
            "minimum_component_weight": 0.05,
            "candidate_weights": [{"garch": 1.0}],
        },
        "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}},
    }
    forecasting.write_text(yaml.safe_dump({"forecasting": base}), encoding="utf-8")
    pipeline.write_text(yaml.safe_dump({"pipeline": {"forecasting": base}}), encoding="utf-8")

    report = vfc.validate_configs(forecasting, pipeline)
    assert report["ok"] is True, f"sarimax.enabled=True should pass validator: {report['errors']}"


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


def test_validate_configs_flags_monte_carlo_sync_mismatch(tmp_path: Path) -> None:
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
                "candidate_weights": [{"garch": 1.0}],
            },
            "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}},
            "monte_carlo": {"enabled": True, "paths": 500, "seed": 7},
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
                "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}},
                "monte_carlo": {"enabled": False, "paths": 1000, "seed": None},
            }
        }
    }

    forecasting.write_text(yaml.safe_dump(base_forecasting), encoding="utf-8")
    pipeline.write_text(yaml.safe_dump(base_pipeline), encoding="utf-8")

    report = vfc.validate_configs(forecasting, pipeline)
    assert report["ok"] is False
    assert any("sync: monte_carlo mismatch" in msg for msg in report["errors"])


def test_validate_configs_flags_ensemble_confidence_sync_mismatch(tmp_path: Path) -> None:
    forecasting = tmp_path / "forecasting.yml"
    pipeline = tmp_path / "pipeline.yml"

    base_forecasting = {
        "forecasting": {
            "sarimax": {"enabled": True},
            "garch": {"enabled": True},
            "samossa": {"enabled": True},
            "mssa_rl": {"enabled": True},
            "ensemble": {
                "enabled": True,
                "confidence_scaling": True,
                "track_directional_accuracy": True,
                "prefer_diversified_candidate": True,
                "diversity_tolerance": 0.05,
                "minimum_component_weight": 0.05,
                "da_floor": 0.1,
                "da_weight_cap": 0.1,
                "candidate_weights": [{"garch": 1.0}],
            },
            "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}},
            "monte_carlo": {"enabled": False, "paths": 1000, "seed": None},
        }
    }
    base_pipeline = {
        "pipeline": {
            "forecasting": {
                "sarimax": {"enabled": True},
                "garch": {"enabled": True},
                "samossa": {"enabled": True},
                "mssa_rl": {"enabled": True},
                "ensemble": {
                    "enabled": True,
                    "confidence_scaling": False,
                    "track_directional_accuracy": True,
                    "prefer_diversified_candidate": True,
                    "diversity_tolerance": 0.15,
                    "minimum_component_weight": 0.05,
                    "da_floor": 0.2,
                    "da_weight_cap": 0.05,
                    "candidate_weights": [{"garch": 1.0}],
                },
                "regime_detection": {"regime_candidate_weights": {"CRISIS": [{"garch": 1.0}]}},
                "monte_carlo": {"enabled": False, "paths": 1000, "seed": None},
            }
        }
    }

    forecasting.write_text(yaml.safe_dump(base_forecasting), encoding="utf-8")
    pipeline.write_text(yaml.safe_dump(base_pipeline), encoding="utf-8")

    report = vfc.validate_configs(forecasting, pipeline)
    assert report["ok"] is False
    assert any("sync: ensemble.confidence_scaling mismatch" in msg for msg in report["errors"])
    assert any("sync: ensemble.diversity_tolerance mismatch" in msg for msg in report["errors"])
    assert any("sync: ensemble.da_floor mismatch" in msg for msg in report["errors"])
    assert any("sync: ensemble.da_weight_cap mismatch" in msg for msg in report["errors"])
