from __future__ import annotations

import pytest

from scripts.run_auto_trader import (
    ConfigurationError,
    _resolve_best_single_regression_metrics,
    validate_production_ensemble_config,
)


def test_validate_production_ensemble_config_blocks_live_false() -> None:
    with pytest.raises(ConfigurationError):
        validate_production_ensemble_config(
            ensemble_kwargs={"confidence_scaling": False},
            execution_mode="live",
        )


def test_validate_production_ensemble_config_allows_non_live_false() -> None:
    validate_production_ensemble_config(
        ensemble_kwargs={"confidence_scaling": False},
        execution_mode="synthetic",
    )


def test_resolve_best_single_regression_metrics_prefers_lowest_rmse() -> None:
    baseline_name, baseline_metrics = _resolve_best_single_regression_metrics(
        {
            "ensemble": {"rmse": 0.9},
            "samossa": {"rmse": 1.1},
            "garch": {"rmse": 0.8},
            "mssa_rl": {"rmse": 0.95},
        }
    )

    assert baseline_name == "garch"
    assert baseline_metrics["rmse"] == 0.8
