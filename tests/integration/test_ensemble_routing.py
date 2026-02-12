from __future__ import annotations

import pandas as pd
import pytest

from etl.time_series_forecaster import TimeSeriesForecaster
from models.time_series_signal_generator import TimeSeriesSignalGenerator


class _DummySamossaModel:
    def forecast(self, steps: int):
        idx = pd.date_range("2025-01-01", periods=steps, freq="B")
        forecast = pd.Series([101.0 + float(i) for i in range(steps)], index=idx)
        return {"forecast": forecast}

    def get_model_summary(self):
        return {"trend_strength": 0.5, "seasonal_strength": 0.2}


def _build_generator() -> TimeSeriesSignalGenerator:
    return TimeSeriesSignalGenerator(
        confidence_threshold=0.0,
        min_expected_return=0.0,
        max_risk_score=1.0,
        use_volatility_filter=False,
        quant_validation_config={"enabled": False},
        cost_model={},
    )


def _build_forecaster(monkeypatch: pytest.MonkeyPatch, *, allow_as_default: bool) -> TimeSeriesForecaster:
    forecaster = TimeSeriesForecaster(forecast_horizon=3)
    forecaster._audit_dir = None
    forecaster._regime_result = None
    forecaster._samossa = _DummySamossaModel()

    def _fake_build_ensemble(_results):
        idx = pd.date_range("2025-01-01", periods=3, freq="B")
        ensemble_forecast = pd.Series([96.0, 95.0, 94.0], index=idx)
        metadata = {
            "allow_as_default": allow_as_default,
            "ensemble_status": "KEEP" if allow_as_default else "DISABLE_DEFAULT",
            "primary_model": "SARIMAX",
        }
        if not allow_as_default:
            metadata["default_model"] = "SAMOSSA"
        return {
            "forecast_bundle": {
                "forecast": ensemble_forecast,
                "lower_ci": ensemble_forecast - 1.0,
                "upper_ci": ensemble_forecast + 1.0,
                "weights": {"sarimax": 0.6, "samossa": 0.4},
                "confidence": 0.8,
                "selection_score": 0.9,
                "primary_model": "SARIMAX",
            },
            "metadata": metadata,
        }

    monkeypatch.setattr(forecaster, "_build_ensemble", _fake_build_ensemble)
    return forecaster


@pytest.mark.integration
def test_ensemble_blocked_routes_signal_to_single_model(monkeypatch: pytest.MonkeyPatch) -> None:
    forecaster = _build_forecaster(monkeypatch, allow_as_default=False)
    bundle = forecaster.forecast(steps=3)

    assert bundle.get("default_model") == "SAMOSSA"
    assert bundle.get("mean_forecast") == bundle.get("samossa_forecast")
    assert bundle.get("ensemble_metadata", {}).get("allow_as_default") is False

    signal = _build_generator().generate_signal(
        forecast_bundle=bundle,
        current_price=100.0,
        ticker="TEST",
        market_data=None,
    )

    assert signal.model_type == "SAMOSSA"
    assert signal.provenance.get("selected_forecast_source") == "mean_forecast"
    assert signal.expected_return > 0.0


@pytest.mark.integration
def test_ensemble_allowed_routes_signal_to_ensemble(monkeypatch: pytest.MonkeyPatch) -> None:
    forecaster = _build_forecaster(monkeypatch, allow_as_default=True)
    bundle = forecaster.forecast(steps=3)

    assert bundle.get("default_model") == "ENSEMBLE"
    assert bundle.get("ensemble_metadata", {}).get("allow_as_default") is True

    signal = _build_generator().generate_signal(
        forecast_bundle=bundle,
        current_price=100.0,
        ticker="TEST",
        market_data=None,
    )

    assert signal.model_type == "ENSEMBLE"
    assert signal.provenance.get("selected_forecast_source") == "ensemble_forecast"
    assert signal.expected_return < 0.0
