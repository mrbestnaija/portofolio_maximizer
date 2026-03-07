"""Phase 7.40 — Regime detector stability tests for degenerate inputs.

Contracts:
  1. Flat price series produces no NaN/inf in any output field.
  2. Flat price series trend_strength == 0.0 (no trend by definition).
  3. Flat price series Hurst exponent is finite and in [0, 1].
  4. detect_regime() on a flat series returns a valid regime string.
  5. _regime_confidence is always finite and in [0.3, 0.95].
  6. Single-element series does not crash.
  7. Near-flat series (numeric noise only) still produces finite outputs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from forcester_ts.regime_detector import RegimeDetector, RegimeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_series(n: int = 60, value: float = 150.0) -> pd.Series:
    return pd.Series([value] * n, dtype=float)


def _near_flat_series(n: int = 60, value: float = 150.0, noise: float = 1e-9) -> pd.Series:
    rng = np.random.default_rng(seed=0)
    return pd.Series(value + rng.normal(0, noise, n), dtype=float)


# ---------------------------------------------------------------------------
# Trend strength
# ---------------------------------------------------------------------------

class TestTrendStrengthDegenerate:
    def test_flat_series_trend_strength_is_zero(self):
        """Flat price → correlation undefined → should return 0.0, not NaN."""
        rd = RegimeDetector()
        result = rd._calculate_trend_strength(_flat_series())
        assert result == pytest.approx(0.0)
        assert math.isfinite(result)

    def test_near_flat_series_trend_strength_is_finite(self):
        """Near-constant series (noise ~1e-9) must not produce NaN."""
        rd = RegimeDetector()
        result = rd._calculate_trend_strength(_near_flat_series())
        assert math.isfinite(result), f"trend_strength={result} is not finite"
        assert 0.0 <= result <= 1.0

    def test_short_series_returns_zero(self):
        """Series shorter than 14 points returns 0.0."""
        rd = RegimeDetector()
        assert rd._calculate_trend_strength(pd.Series([1.0] * 10)) == 0.0


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------

class TestHurstDegenerate:
    def test_flat_series_hurst_is_finite(self):
        """Flat series → tau = [0, ...] → log(0) clamped → no NaN in polyfit output."""
        rd = RegimeDetector()
        result = rd._calculate_hurst_exponent(_flat_series())
        assert math.isfinite(result), f"Hurst={result} is not finite on flat series"
        assert 0.0 <= result <= 1.0

    def test_near_flat_series_hurst_finite(self):
        rd = RegimeDetector()
        result = rd._calculate_hurst_exponent(_near_flat_series(noise=1e-8))
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_short_series_returns_half(self):
        """Too-short series returns 0.5 (random walk assumption)."""
        rd = RegimeDetector()
        assert rd._calculate_hurst_exponent(pd.Series([1.0] * 5)) == 0.5


# ---------------------------------------------------------------------------
# Full detect_regime pipeline
# ---------------------------------------------------------------------------

class TestDetectRegimeDegenerate:
    def test_flat_series_no_nan_in_features(self):
        """detect_regime on a flat series must produce all-finite features."""
        rd = RegimeDetector()
        result = rd.detect_regime(_flat_series())
        features = result["features"]
        for key, val in features.items():
            assert math.isfinite(val), f"features[{key!r}]={val} is not finite"

    def test_flat_series_returns_valid_regime_string(self):
        rd = RegimeDetector()
        result = rd.detect_regime(_flat_series())
        assert isinstance(result["regime"], str)
        assert len(result["regime"]) > 0

    def test_flat_series_confidence_in_valid_range(self):
        rd = RegimeDetector()
        result = rd.detect_regime(_flat_series())
        conf = result["confidence"]
        assert math.isfinite(conf), f"confidence={conf} is not finite"
        assert 0.3 <= conf <= 0.95

    def test_single_point_does_not_crash(self):
        """Single-element series falls through all guards without exception."""
        rd = RegimeDetector()
        try:
            result = rd.detect_regime(pd.Series([100.0]))
            assert "regime" in result
        except Exception as exc:
            pytest.fail(f"detect_regime raised on single-point series: {exc}")

    def test_trend_strength_zero_for_flat_series_in_features(self):
        rd = RegimeDetector()
        result = rd.detect_regime(_flat_series())
        assert result["features"]["trend_strength"] == pytest.approx(0.0, abs=1e-9)

    def test_hurst_finite_for_flat_series_in_features(self):
        rd = RegimeDetector()
        result = rd.detect_regime(_flat_series())
        hurst = result["features"]["hurst_exponent"]
        assert math.isfinite(hurst)
        assert 0.0 <= hurst <= 1.0
