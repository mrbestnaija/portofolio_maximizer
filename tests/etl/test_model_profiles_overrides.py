from pathlib import Path
from typing import Sequence

import numpy as np

from etl.model_profiles import (
    TSModelOverride,
    load_ts_model_overrides,
    lookup_ts_model_override,
    select_profile_with_overrides,
)
from etl.regime_detector import RegimeState, RegimeDetector


class DummyRegimeDetector(RegimeDetector):
    """Detector that always returns a fixed volatility regime."""

    def __init__(self, state: RegimeState) -> None:
        super().__init__(window_size=10, significance_level=0.05)
        self._state = state

    def detect_volatility_regime(self, returns: np.ndarray) -> RegimeState:  # type: ignore[override]
        return self._state


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_and_lookup_ts_model_overrides(tmp_path: Path) -> None:
    """Overrides are parsed correctly and lookup prefers exact regime over default."""
    overrides_yaml = tmp_path / "ts_model_overrides.yml"
    _write_yaml(
        overrides_yaml,
        """
overrides:
  - ticker: "AAPL"
    regime: "high_vol"
    candidate_name: "sarimax_only"
    profile_hint: "default"
    notes: "test entry"
  - ticker: "MSFT"
    candidate_name: "samossa_only"
""",
    )

    overrides_map = load_ts_model_overrides(path=overrides_yaml)

    # Canonical keys: (TICKER_UPPER, regime_lower)
    assert ("AAPL", "high_vol") in overrides_map
    assert ("MSFT", "default") in overrides_map

    # Exact regime match wins.
    override_aapl = lookup_ts_model_override(
        ticker="aapl",
        regime="high_vol",
        overrides=overrides_map,
    )
    assert isinstance(override_aapl, TSModelOverride)
    assert override_aapl.candidate_name == "sarimax_only"
    assert override_aapl.profile_hint == "default"

    # Fallback to default regime when no exact match.
    override_msft = lookup_ts_model_override(
        ticker="MSFT",
        regime="normal_vol",
        overrides=overrides_map,
    )
    assert isinstance(override_msft, TSModelOverride)
    assert override_msft.candidate_name == "samossa_only"


def test_select_profile_with_overrides_applies_profile_hint(tmp_path: Path) -> None:
    """
    select_profile_with_overrides:
    - Uses sleeve/regime selection for baseline profile.
    - Applies override.profile_hint to swap to a different profile when present.
    """
    profiles_yaml = tmp_path / "model_profiles.yml"
    overrides_yaml = tmp_path / "ts_model_overrides.yml"

    # Baseline model profiles: "default" has a regime tag; "high_vol_risk" has no
    # regimes so it will not be chosen by the baseline selector but can be used
    # as an override via profile_hint.
    _write_yaml(
        profiles_yaml,
        """
model_profiles:
  default:
    sleeves: ["core"]
    regimes: ["high_vol"]
    sarimax:
      enabled: true
  high_vol_risk:
    sleeves: ["core"]
    sarimax:
      enabled: true
      max_p: 1
""",
    )

    _write_yaml(
        overrides_yaml,
        """
overrides:
  - ticker: "AAPL"
    regime: "high_vol"
    candidate_name: "samossa_only"
    profile_hint: "high_vol_risk"
""",
    )

    regime_state = RegimeState(
        regime_type="high_vol",
        confidence=0.9,
        duration=10,
        transition_probability=0.1,
    )
    detector = DummyRegimeDetector(regime_state)

    returns: Sequence[float] = [0.01, -0.02, 0.015, -0.005, 0.02]

    profile, detected_state, override = select_profile_with_overrides(
        ticker="AAPL",
        sleeve="core",
        returns=returns,
        detector=detector,
        profiles_path=profiles_yaml,
        overrides_path=overrides_yaml,
    )

    assert detected_state.regime_type == "high_vol"
    assert isinstance(override, TSModelOverride)
    assert override.candidate_name == "samossa_only"
    # Override should swap the profile to the hinted one.
    assert profile is not None
    assert profile.name == "high_vol_risk"


def test_select_profile_with_overrides_falls_back_when_no_override(tmp_path: Path) -> None:
    """When no override exists, selection should match the baseline profile."""
    profiles_yaml = tmp_path / "model_profiles.yml"
    overrides_yaml = tmp_path / "ts_model_overrides.yml"

    _write_yaml(
        profiles_yaml,
        """
model_profiles:
  default:
    sleeves: ["core"]
    regimes: ["high_vol"]
    sarimax:
      enabled: true
""",
    )

    # Empty overrides block.
    _write_yaml(
        overrides_yaml,
        """
overrides: []
""",
    )

    regime_state = RegimeState(
        regime_type="high_vol",
        confidence=0.8,
        duration=5,
        transition_probability=0.2,
    )
    detector = DummyRegimeDetector(regime_state)

    returns: Sequence[float] = [0.01, -0.01, 0.02, -0.005, 0.003]

    profile, detected_state, override = select_profile_with_overrides(
        ticker="MSFT",
        sleeve="core",
        returns=returns,
        detector=detector,
        profiles_path=profiles_yaml,
        overrides_path=overrides_yaml,
    )

    assert detected_state.regime_type == "high_vol"
    assert override is None
    assert profile is not None
    assert profile.name == "default"

