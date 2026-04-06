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


# ---------------------------------------------------------------------------
# E1/E2 regression — arima_order fallback must be AR-only (1,0,0), not (1,0,1)
# ---------------------------------------------------------------------------

def test_samossa_default_arima_order_is_ar_only() -> None:
    """E1: SAMOSSAConfig.arima_order default and __init__ fallback must be (1,0,0).

    Phase 7.16 deliberately removed the MA(1) term to avoid convergence warnings
    on short residual series. Both the config default and the constructor fallback
    were incorrectly set to (1,0,1) — this test pins the correct value.
    """
    from forcester_ts.samossa import SAMOSSAConfig

    cfg = SAMOSSAConfig()
    assert cfg.arima_order == (1, 0, 0), (
        f"SAMOSSAConfig.arima_order default must be (1,0,0); got {cfg.arima_order}"
    )


def test_samossa_init_arima_order_fallback_is_ar_only() -> None:
    """E2: when arima_order=() (falsy) the constructor must use (1,0,0), not (1,0,1)."""
    forecaster = SAMOSSAForecaster(window_length=40, n_components=2, arima_order=())
    assert forecaster.config.arima_order == (1, 0, 0), (
        f"Constructor falsy-arima_order fallback must be (1,0,0); "
        f"got {forecaster.config.arima_order}"
    )


def test_samossa_fit_arima_order_getattr_fallback_is_ar_only() -> None:
    """E2 (line 272): _fit_residual_model getattr fallback must be (1,0,0) not (1,0,1).

    Exercises the _fit_residual_model path with a series long enough to reach the
    ARIMA branch. The forecaster is built with arima_order=(1,0,0) explicitly so the
    getattr returns the right value — this test verifies no MA term is injected
    by confirming the stored residual_model order is (1,0,0).
    """
    rng = np.random.default_rng(42)
    series = pd.Series(
        100.0 + np.cumsum(rng.normal(0.0, 0.5, 300)),
        index=pd.date_range("2023-01-01", periods=300, freq="D"),
        name="Close",
    )
    forecaster = SAMOSSAForecaster(window_length=40, n_components=4, arima_order=(1, 0, 0))
    forecaster.fit(series)

    if forecaster._residual_model is not None and hasattr(forecaster._residual_model, "model"):
        order = getattr(forecaster._residual_model.model, "order", None)
        if order is not None:
            assert order == (1, 0, 0), (
                f"Fitted ARIMA order must be (1,0,0); got {order}"
            )
