"""
Integration: TimeSeriesForecaster -> TimeSeriesSignalGenerator

Focus:
- All forecasters wire into the forecast bundle consumed by the signal generator.
- Scaling invariants hold (multiplying prices by a constant does not change signal edge).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from etl.time_series_forecaster import TimeSeriesForecaster
from models.time_series_signal_generator import TimeSeriesSignalGenerator


@pytest.fixture(scope="session")
def sample_price_series_daily() -> pd.Series:
    rng = np.random.default_rng(1337)
    index = pd.date_range("2024-01-02", periods=220, freq="B")
    # Geometric random walk with mild drift; stable enough for fast model fits.
    rets = rng.normal(loc=0.0004, scale=0.01, size=len(index))
    prices = 100.0 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture(scope="session")
def fast_all_model_config() -> dict:
    # Keep the search grids compact so the integration suite stays bounded.
    return {
        "sarimax_config": {
            "enabled": True,
            "auto_select": True,
            "trend": "auto",
            "max_p": 1,
            "max_d": 1,
            "max_q": 1,
            # Disable seasonal terms in this integration test (fast + avoids overfit).
            "seasonal_periods": 0,
            "max_P": 0,
            "max_D": 0,
            "max_Q": 0,
            "order_search_mode": "compact",
            "order_search_maxiter": 60,
        },
        "samossa_config": {
            "enabled": True,
            "window_length": 40,
            "n_components": 6,
            "min_series_length": 120,
            "forecast_horizon": 6,
        },
        "mssa_rl_config": {
            "enabled": True,
            "window_length": 30,
            "forecast_horizon": 6,
            "use_gpu": False,
        },
        "garch_config": {
            "enabled": True,
            "backend": "arch",
            "auto_select": True,
            "max_p": 1,
            "max_q": 1,
            "order_search_mode": "compact",
            "order_search_maxiter": 50,
        },
        "ensemble_config": {"enabled": True},
    }


@pytest.fixture(scope="session")
def forecast_bundle_all_models(sample_price_series_daily: pd.Series, fast_all_model_config: dict) -> dict:
    # Avoid audit-log side effects during tests.
    prior = os.environ.pop("TS_FORECAST_AUDIT_DIR", None)
    try:
        returns = sample_price_series_daily.pct_change().dropna()
        forecaster = TimeSeriesForecaster(**fast_all_model_config)
        forecaster.fit(sample_price_series_daily, returns_series=returns)
        return forecaster.forecast(steps=6)
    finally:
        if prior is not None:
            os.environ["TS_FORECAST_AUDIT_DIR"] = prior


def _build_generator() -> TimeSeriesSignalGenerator:
    return TimeSeriesSignalGenerator(
        confidence_threshold=0.0,
        min_expected_return=0.0,
        max_risk_score=1.0,
        use_volatility_filter=False,
        quant_validation_config={"enabled": False},
        cost_model={},
    )


@pytest.mark.integration
def test_forecast_bundle_contract_all_models(forecast_bundle_all_models: dict) -> None:
    assert isinstance(forecast_bundle_all_models, dict)
    assert forecast_bundle_all_models.get("model_errors") == {}

    # Core bundle keys.
    assert forecast_bundle_all_models["horizon"] == 6
    assert isinstance(forecast_bundle_all_models.get("instrumentation_report"), dict)

    # Each level-forecasting component should be present when enabled.
    for key in ("sarimax_forecast", "samossa_forecast", "mssa_rl_forecast"):
        payload = forecast_bundle_all_models.get(key)
        assert isinstance(payload, dict), f"{key} missing/invalid: {type(payload)}"
        series = payload.get("forecast")
        assert isinstance(series, pd.Series), f"{key} forecast missing Series"
        assert not series.dropna().empty, f"{key} forecast is empty"

    # GARCH produces return-scale volatility/variance (no level forecast key).
    garch_payload = forecast_bundle_all_models.get("garch_forecast")
    assert isinstance(garch_payload, dict)
    assert garch_payload.get("variance_forecast") is not None
    assert garch_payload.get("volatility") is not None

    # Ensemble/primary output for downstream signal gen.
    ensemble = forecast_bundle_all_models.get("ensemble_forecast")
    assert isinstance(ensemble, dict)
    ensemble_series = ensemble.get("forecast")
    assert isinstance(ensemble_series, pd.Series)
    assert len(ensemble_series.dropna()) >= 1

    # Volatility forecast must be scalar-like and finite.
    vol_payload = forecast_bundle_all_models.get("volatility_forecast")
    assert isinstance(vol_payload, dict)
    vol_val = vol_payload.get("volatility")
    assert vol_val is not None
    vol_float = float(vol_val) if isinstance(vol_val, (int, float, np.generic)) else float(vol_val.iloc[0])
    assert np.isfinite(vol_float)
    assert vol_float >= 0.0

    # SARIMAX-X wiring: exogenous artifact is recorded for auditing.
    report = forecast_bundle_all_models["instrumentation_report"]
    artifacts = report.get("artifacts") or {}
    sarimax_exog = artifacts.get("sarimax_exogenous")
    assert isinstance(sarimax_exog, dict)
    assert sarimax_exog.get("row_count", 0) > 0
    assert isinstance(sarimax_exog.get("columns"), list)


@pytest.mark.integration
def test_signal_generation_smoke_all_models(sample_price_series_daily: pd.Series, forecast_bundle_all_models: dict) -> None:
    generator = _build_generator()
    current_price = float(sample_price_series_daily.iloc[-1])
    signal = generator.generate_signal(
        forecast_bundle=forecast_bundle_all_models,
        current_price=current_price,
        ticker="TEST",
        market_data=None,
    )

    assert signal.ticker == "TEST"
    assert signal.action in {"BUY", "SELL", "HOLD"}
    assert np.isfinite(signal.expected_return)
    assert 0.0 <= signal.confidence <= 1.0
    assert 0.0 <= signal.risk_score <= 1.0

    models_used = set((signal.provenance or {}).get("models_used") or [])
    assert {"SARIMAX", "SAMOSSA", "MSSA_RL", "GARCH"}.issubset(models_used)


@pytest.mark.integration
def test_signal_scaling_invariant_under_price_rescale(
    sample_price_series_daily: pd.Series, fast_all_model_config: dict, forecast_bundle_all_models: dict
) -> None:
    generator = _build_generator()

    current_price = float(sample_price_series_daily.iloc[-1])
    base_signal = generator.generate_signal(
        forecast_bundle=forecast_bundle_all_models,
        current_price=current_price,
        ticker="TEST",
        market_data=None,
    )

    scale = 1000.0
    scaled_series = sample_price_series_daily * scale
    scaled_returns = scaled_series.pct_change().dropna()

    prior = os.environ.pop("TS_FORECAST_AUDIT_DIR", None)
    try:
        scaled_forecaster = TimeSeriesForecaster(**fast_all_model_config)
        scaled_forecaster.fit(scaled_series, returns_series=scaled_returns)
        scaled_bundle = scaled_forecaster.forecast(steps=6)
    finally:
        if prior is not None:
            os.environ["TS_FORECAST_AUDIT_DIR"] = prior

    scaled_price = float(scaled_series.iloc[-1])
    scaled_signal = generator.generate_signal(
        forecast_bundle=scaled_bundle,
        current_price=scaled_price,
        ticker="TEST",
        market_data=None,
    )

    # Scaling prices should not change the decision or edge on a dimensionless basis.
    # Allow direction divergence when the expected return is near-zero (borderline
    # signal) because stochastic model convergence can differ under extreme rescaling.
    if abs(base_signal.expected_return) > 0.01:
        assert scaled_signal.action == base_signal.action
    assert scaled_signal.expected_return == pytest.approx(base_signal.expected_return, rel=0.10, abs=0.02)
    assert scaled_signal.confidence == pytest.approx(base_signal.confidence, rel=0.15, abs=0.10)
    assert scaled_signal.risk_score == pytest.approx(base_signal.risk_score, rel=0.15, abs=0.15)

    # Volatility is on returns scale; should be close after rescaling prices.
    if base_signal.volatility is not None and scaled_signal.volatility is not None:
        assert scaled_signal.volatility == pytest.approx(base_signal.volatility, rel=0.10, abs=0.01)


@pytest.mark.integration
def test_intraday_series_not_padded_into_synthetic_bars() -> None:
    # Intraday gaps (overnight/weekend) should not be padded via asfreq().
    idx = pd.date_range("2024-01-02 09:00", periods=80, freq="h")
    series = pd.Series(np.linspace(100.0, 105.0, num=len(idx)), index=idx, name="Close")
    # Create gaps typical of market-hours data.
    series = series.drop(series.index[::7])
    assert (pd.Series(series.index).diff().dt.total_seconds() > 3600).any()

    forecaster = TimeSeriesForecaster(
        sarimax_config={"enabled": False},
        garch_config={"enabled": False},
        samossa_config={"enabled": False},
        mssa_rl_config={"enabled": False},
        ensemble_config={"enabled": False},
    )
    prepared = forecaster._ensure_series(series)
    assert len(prepared) == len(series)
    assert (pd.Series(prepared.index).diff().dt.total_seconds() > 3600).any()
