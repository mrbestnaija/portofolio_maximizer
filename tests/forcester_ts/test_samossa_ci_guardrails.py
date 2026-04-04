from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from forcester_ts.samossa import SAMOSSAForecaster


def _expected_ci_band(forecaster: SAMOSSAForecaster, steps: int, index: pd.Index) -> pd.Series:
    noise_level = float(forecaster._residuals.std()) if forecaster._residuals is not None else 0.0
    max_scale = np.sqrt(max(steps / 2, 1.0))
    horizon_scale = np.minimum(
        np.sqrt(np.arange(1, steps + 1, dtype=float)),
        max_scale,
    )
    return pd.Series(noise_level * horizon_scale, index=index)


def test_samossa_ci_uses_bounded_horizon_scale_on_ordinary_positive_series() -> None:
    rng = np.random.default_rng(7)
    series = pd.Series(
        100.0 + np.cumsum(rng.normal(0.0, 0.8, 240)),
        index=pd.date_range("2024-01-01", periods=240, freq="D"),
        name="Close",
    )
    forecaster = SAMOSSAForecaster(window_length=40, n_components=4)
    forecaster.fit(series)

    steps = 12
    result = forecaster.forecast(steps=steps)
    expected_band = _expected_ci_band(forecaster, steps, result["forecast"].index)
    expected_lower = result["forecast"] - expected_band
    expected_upper = result["forecast"] + expected_band

    pdt.assert_series_equal(result["lower_ci"], expected_lower)
    pdt.assert_series_equal(result["upper_ci"], expected_upper)
    assert expected_band.iloc[-1] < float(forecaster._residuals.std()) * np.sqrt(steps)


def test_samossa_clamps_negative_lower_ci_for_strictly_positive_low_price_series() -> None:
    rng = np.random.default_rng(11)
    series = pd.Series(
        np.clip(0.30 + rng.normal(0.0, 0.25, 240), 0.05, None),
        index=pd.date_range("2024-01-01", periods=240, freq="D"),
        name="Close",
    )
    forecaster = SAMOSSAForecaster(window_length=40, n_components=1)
    forecaster.fit(series)

    steps = 12
    result = forecaster.forecast(steps=steps)
    expected_band = _expected_ci_band(forecaster, steps, result["forecast"].index)
    unclamped_lower = result["forecast"] - expected_band

    assert unclamped_lower.min() < 0.0, "fixture must exercise the negative-CI path"
    assert (result["lower_ci"] >= 0.0).all()
    assert (result["lower_ci"] == 0.0).any()
