from __future__ import annotations

import pytest

from scripts.run_auto_trader import ConfigurationError, validate_production_ensemble_config


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

