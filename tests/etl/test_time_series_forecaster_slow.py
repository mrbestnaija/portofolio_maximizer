"""
Slow integration tests for the time-series forecasting stack.
"""

from __future__ import annotations

from datetime import datetime
import math

import numpy as np
import pandas as pd
import pytest

try:
    from etl.time_series_forecaster import (
        TimeSeriesForecaster,
        TimeSeriesForecasterConfig,
    )
    from forcester_ts.ensemble import EnsembleConfig

    FORECASTING_AVAILABLE = True
except ImportError:  # pragma: no cover - protective
    FORECASTING_AVAILABLE = False
    pytestmark = pytest.mark.skip("Forecasting modules not available")


pytestmark = pytest.mark.slow


@pytest.fixture(scope="function")
def price_series() -> pd.Series:
    dates = pd.date_range(datetime(2020, 1, 1), periods=240, freq="D")
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="Close")


@pytest.fixture(scope="function")
def returns_series(price_series: pd.Series) -> pd.Series:
    returns = price_series.pct_change().dropna()
    returns.name = "Returns"
    return returns


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestUnifiedForecaster:
    def test_parallel_forecasts(self, price_series: pd.Series, returns_series: pd.Series) -> None:
        config = TimeSeriesForecasterConfig(
            forecast_horizon=7,
            sarimax_enabled=True,
            sarimax_kwargs={"auto_select": True},
        )
        forecaster = TimeSeriesForecaster(config=config)
        forecaster.fit(price_series=price_series, returns_series=returns_series)

        results = forecaster.forecast()
        assert results["sarimax_forecast"] is not None
        assert results["garch_forecast"] is not None
        assert results["samossa_forecast"] is not None
        assert results["mssa_rl_forecast"] is not None
        assert results["ensemble_forecast"] is not None
        metadata = results.get("ensemble_metadata", {})
        assert "weights" in metadata and metadata["weights"]
        assert metadata["selection_score"] >= 0

        summaries = forecaster.get_component_summaries()
        assert "sarimax" in summaries
        assert "mssa_rl" in summaries

    def test_regression_metric_evaluation(self, price_series: pd.Series, returns_series: pd.Series) -> None:
        train = price_series.iloc[:-10]
        holdout = price_series.iloc[-10:]
        train_returns = returns_series.iloc[:-10]

        config = TimeSeriesForecasterConfig(forecast_horizon=10, sarimax_enabled=True, sarimax_kwargs={"auto_select": True})
        forecaster = TimeSeriesForecaster(config=config)
        forecaster.fit(price_series=train, returns_series=train_returns)
        forecaster.forecast()
        metrics = forecaster.evaluate(holdout)

        assert "sarimax" in metrics
        sarimax_metrics = metrics["sarimax"]
        assert sarimax_metrics["rmse"] >= 0
        assert 0 <= sarimax_metrics["smape"] <= 2
        assert sarimax_metrics["tracking_error"] >= 0

    def test_ensemble_tracks_regime_shift(self) -> None:
        np.random.seed(3)
        periods = 260
        dates = pd.date_range(datetime(2019, 1, 1), periods=periods, freq="D")
        trend = 0.05 * np.arange(periods)
        seasonal = 1.8 * np.sin(2 * np.pi * np.arange(periods) / 30)
        regime = np.where(np.arange(periods) > 130, 3.0, 0.0)
        noise = np.random.normal(0, 0.2, size=periods)
        prices = 80 + trend + seasonal + regime + noise
        returns = np.diff(prices, prepend=prices[0]) / prices[0]
        price_series_full = pd.Series(prices, index=dates, name="Close")
        returns_series_full = pd.Series(returns, index=dates, name="Returns")
        train_series = price_series_full.iloc[:-14]
        holdout_series = price_series_full.iloc[-14:]
        train_returns = returns_series_full.iloc[:-14]

        config = TimeSeriesForecasterConfig(
            forecast_horizon=14,
            sarimax_enabled=True,
            sarimax_kwargs={"auto_select": True},
            samossa_kwargs={"window_length": 40, "n_components": 5},
            mssa_rl_kwargs={"window_length": 30},
            ensemble_kwargs={
                "enabled": True,
                "candidate_weights": [
                    {"sarimax": 0.6, "samossa": 0.4},
                    {"sarimax": 0.4, "samossa": 0.4, "mssa_rl": 0.2},
                ],
            },
        )
        forecaster = TimeSeriesForecaster(config=config)
        forecaster.fit(price_series=train_series, returns_series=train_returns)
        forecaster.forecast()
        metrics = forecaster.evaluate(holdout_series)

        ensemble_metrics = metrics["ensemble"]
        sarimax_metrics = metrics["sarimax"]
        assert ensemble_metrics["rmse"] <= sarimax_metrics["rmse"] + 0.2
        assert ensemble_metrics["smape"] < 0.3
        assert ensemble_metrics["n_observations"] == len(holdout_series)


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestRegimeCandidateOverrides:
    def test_regime_candidate_weights_override_applied(self) -> None:
        np.random.seed(7)
        periods = 260
        dates = pd.date_range(datetime(2020, 1, 1), periods=periods, freq="D")
        oscillation = ((-1) ** np.arange(periods)) * 0.5
        noise = np.random.normal(0, 0.05, size=periods)
        prices = 100 + oscillation + noise

        series = pd.Series(prices, index=dates, name="Close")
        train = series.iloc[:-12]
        holdout = series.iloc[-12:]
        train_returns = train.pct_change().dropna()

        config = TimeSeriesForecasterConfig(
            forecast_horizon=len(holdout),
            sarimax_enabled=True,
            garch_enabled=False,
            samossa_enabled=True,
            mssa_rl_enabled=False,
            ensemble_enabled=True,
            sarimax_kwargs={"auto_select": True},
            samossa_kwargs={"window_length": 40, "n_components": 5},
            ensemble_kwargs={
                "enabled": True,
                "confidence_scaling": False,
                "candidate_weights": [{"sarimax": 1.0}],
            },
            regime_detection_enabled=True,
            regime_detection_kwargs={
                # Make LIQUID_RANGEBOUND easy to satisfy (hurst/adf still apply).
                "lookback_window": 60,
                "vol_threshold_low": 1.0,
                "vol_threshold_high": 2.0,
                "trend_threshold_weak": 0.99,
                "trend_threshold_strong": 1.0,
                "regime_candidate_weights": {
                    "LIQUID_RANGEBOUND": [{"samossa": 1.0}],
                },
            },
        )

        forecaster = TimeSeriesForecaster(config=config)
        forecaster.fit(price_series=train, returns_series=train_returns)
        result = forecaster.forecast(steps=len(holdout))

        assert result.get("regime") == "LIQUID_RANGEBOUND"
        ensemble = result.get("ensemble_forecast") or {}
        weights = ensemble.get("weights") or {}
        assert weights.get("samossa") == pytest.approx(1.0)
