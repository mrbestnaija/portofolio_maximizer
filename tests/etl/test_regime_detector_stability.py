from __future__ import annotations

import math

import numpy as np

from etl.regime_detector import RegimeDetector


def test_detect_volatility_regime_flat_series_returns_finite_metrics() -> None:
    detector = RegimeDetector(window_size=10)
    state = detector.detect_volatility_regime(np.zeros(30, dtype=float))
    assert math.isfinite(state.confidence)
    assert math.isfinite(state.transition_probability)
    assert 0.0 <= state.confidence <= 1.0
    assert 0.0 <= state.transition_probability <= 1.0


def test_detect_trend_regime_flat_series_returns_finite_metrics() -> None:
    detector = RegimeDetector(window_size=10)
    state = detector.detect_trend_regime(np.zeros(30, dtype=float))
    assert math.isfinite(state.confidence)
    assert math.isfinite(state.transition_probability)
    assert 0.0 <= state.confidence <= 1.0
    assert 0.0 <= state.transition_probability <= 1.0
