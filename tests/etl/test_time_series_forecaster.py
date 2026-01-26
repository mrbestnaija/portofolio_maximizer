"""
Tests for the time-series forecasting stack (SARIMAX, GARCH, SAMOSSA, MSSA-RL).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import pytest

try:
    from etl.time_series_forecaster import (
        GARCHForecaster,
        MSSARLForecaster,
        SAMOSSAForecaster,
        SARIMAXForecaster,
        TimeSeriesForecaster,
        TimeSeriesForecasterConfig,
    )
    from forcester_ts.ensemble import EnsembleCoordinator, EnsembleConfig, derive_model_confidence

    FORECASTING_AVAILABLE = True
except ImportError:  # pragma: no cover - protective
    FORECASTING_AVAILABLE = False
    pytestmark = pytest.mark.skip("Forecasting modules not available")


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
class TestSAMOSSA:
    def test_fit_and_forecast(self, price_series: pd.Series) -> None:
        forecaster = SAMOSSAForecaster(window_length=30, n_components=5)
        forecaster.fit(price_series)

        summary = forecaster.get_model_summary()
        assert summary["window_length_used"] >= 5
        assert summary["n_components"] >= 1

        result = forecaster.forecast(steps=8)
        assert len(result["forecast"]) == 8
        assert "explained_variance_ratio" in result

    def test_scaling_normalization(self, price_series: pd.Series) -> None:
        warped_series = price_series * 1000 + 5000
        forecaster = SAMOSSAForecaster(window_length=25, n_components=4, normalize=True)
        forecaster.fit(warped_series)
        summary = forecaster.get_model_summary()
        assert summary["scale_mean"] > 0
        assert summary["scale_std"] > 0
        assert abs(summary["normalized_mean"]) < 1e-9
        assert math.isclose(summary["normalized_std"], 1.0, rel_tol=1e-6)


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestSARIMAX:
    def test_auto_select_fit(self, price_series: pd.Series) -> None:
        forecaster = SARIMAXForecaster(auto_select=True)
        forecaster.fit(price_series)

        summary = forecaster.get_model_summary()
        assert "order" in summary
        assert summary["aic"] is not None

    def test_forecast_outputs(self, price_series: pd.Series) -> None:
        forecaster = SARIMAXForecaster(auto_select=True)
        forecaster.fit(price_series)

        result = forecaster.forecast(steps=5)
        assert len(result["forecast"]) == 5
        assert len(result["lower_ci"]) == 5
        assert "z_score" in result

    def test_manual_orders_disallowed(self, price_series: pd.Series) -> None:
        forecaster = SARIMAXForecaster(auto_select=False)
        with pytest.raises(ValueError):
            forecaster.fit(price_series)

    def test_sarimax_forecast_accuracy(self) -> None:
        np.random.seed(42)
        periods = 320
        dates = pd.date_range(datetime(2020, 1, 1), periods=periods, freq="D")
        trend = 0.08 * np.arange(periods)
        seasonal = 2.5 * np.sin(2 * np.pi * np.arange(periods) / 12)
        noise = np.random.normal(0, 0.6, size=periods)
        series = pd.Series(100 + trend + seasonal + noise, index=dates, name="Close")

        train = series.iloc[:-20]
        test = series.iloc[-20:]

        forecaster = SARIMAXForecaster(auto_select=True)
        forecaster.fit(train)
        forecast = forecaster.forecast(steps=len(test))["forecast"].reindex(test.index)

        mape = np.mean(np.abs((test.values - forecast.values) / np.maximum(np.abs(test.values), 1e-6)))
        assert mape < 0.12


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestGARCH:
    def test_garch_pipeline(self, returns_series: pd.Series) -> None:
        forecaster = GARCHForecaster(max_p=1, max_q=1)
        forecaster.fit(returns_series)

        result = forecaster.forecast(steps=5)
        assert "variance_forecast" in result
        assert len(result["variance_forecast"]) >= 1
        assert np.all(result["variance_forecast"].values > 0)

    def test_manual_orders_disallowed(self, returns_series: pd.Series) -> None:
        forecaster = GARCHForecaster(auto_select=False, max_p=1, max_q=1)
        with pytest.raises(ValueError):
            forecaster.fit(returns_series)


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestSARIMAXXInstrumentation:
    def test_exogenous_artifact_recorded(
        self,
        price_series: pd.Series,
        returns_series: pd.Series,
    ) -> None:
        config = TimeSeriesForecasterConfig(
            sarimax_enabled=True,
            garch_enabled=False,
            samossa_enabled=False,
            mssa_rl_enabled=False,
            ensemble_enabled=False,
            forecast_horizon=3,
            sarimax_kwargs={
                "auto_select": True,
                "trend": "auto",
                "max_p": 1,
                "max_d": 1,
                "max_q": 1,
                "max_P": 1,
                "max_D": 1,
                "max_Q": 1,
                "order_search_mode": "compact",
                "order_search_maxiter": 60,
            },
        )
        forecaster = TimeSeriesForecaster(config=config)
        forecaster.fit(price_series=price_series, returns_series=returns_series)
        report = forecaster.get_instrumentation_report()
        exog = (report.get("artifacts") or {}).get("sarimax_exogenous") or {}
        assert exog.get("columns") == ["ret_1", "vol_10", "mom_5", "ema_gap_10", "zscore_20"]
        sarimax_summary = (forecaster.get_component_summaries() or {}).get("sarimax") or {}
        assert sarimax_summary.get("aic") is not None
        assert sarimax_summary.get("order") is not None

    def test_garch_detects_volatility_regime(self) -> None:
        np.random.seed(0)
        low_vol = np.random.normal(0, 0.01, size=250)
        high_vol = np.random.normal(0, 0.04, size=250)
        returns = np.concatenate([low_vol, high_vol])
        dates = pd.date_range(datetime(2021, 1, 1), periods=len(returns), freq="D")
        series = pd.Series(returns, index=dates)

        forecaster = GARCHForecaster(max_p=1, max_q=1)
        forecaster.fit(series)
        forecast = forecaster.forecast(steps=1)
        baseline_var = np.var(low_vol)
        assert forecast["variance_forecast"].iloc[0] > baseline_var


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestMSSARL:
    def test_mssa_rl_forecast(self, price_series: pd.Series) -> None:
        forecaster = MSSARLForecaster()
        forecaster.fit(price_series)

        result = forecaster.forecast(steps=6)
        assert len(result["forecast"]) == 6
        assert "change_points" in result
        diagnostics = forecaster.get_diagnostics()
        assert "q_table" in diagnostics

    def test_mssa_rl_change_point_detection(self) -> None:
        np.random.seed(7)
        block_1 = np.random.normal(50, 1, size=120)
        block_2 = np.random.normal(60, 1, size=120)
        block_3 = np.random.normal(45, 1.5, size=120)
        data = np.concatenate([block_1, block_2, block_3])
        dates = pd.date_range(datetime(2020, 1, 1), periods=len(data), freq="D")
        series = pd.Series(data, index=dates)

        forecaster = MSSARLForecaster()
        forecaster.fit(series)
        diagnostics = forecaster.get_diagnostics()
        change_points = diagnostics.get("change_points", [])
        change_points = [pd.to_datetime(cp) for cp in change_points]
        assert any(abs((cp - dates[120]).days) <= 5 for cp in change_points)


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestUnifiedForecaster:
    def test_parallel_forecasts(self, price_series: pd.Series, returns_series: pd.Series) -> None:
        config = TimeSeriesForecasterConfig(
            forecast_horizon=7,
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

        config = TimeSeriesForecasterConfig(forecast_horizon=10, sarimax_kwargs={"auto_select": True})
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


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestForecasterIntegration:
    """Integration tests for forecasting."""

    def test_insufficient_data_handling(self):
        nan_data = pd.Series([np.nan] * 20, index=pd.date_range('2020-01-01', periods=20))
        forecaster = TimeSeriesForecaster()
        with pytest.raises(ValueError):
            forecaster.fit(price_series=nan_data)

    def test_forecast_with_missing_data(self, price_series: pd.Series):
        data_with_missing = price_series.copy()
        data_with_missing.iloc[10:15] = np.nan

        forecaster = SARIMAXForecaster(auto_select=True)
        forecaster.fit(data_with_missing.dropna())

        assert forecaster.fitted_model is not None


@pytest.mark.skipif(not FORECASTING_AVAILABLE, reason="Forecasting modules not available")
class TestEnsembleCoordinator:
    def test_weight_selection_prefers_high_confidence(self) -> None:
        config = EnsembleConfig(
            candidate_weights=[
                {"sarimax": 0.7, "samossa": 0.3},
                {"sarimax": 0.4, "samossa": 0.4, "mssa_rl": 0.2},
                {"samossa": 0.9, "mssa_rl": 0.1},
            ]
        )
        coordinator = EnsembleCoordinator(config)
        weights, score = coordinator.select_weights({"sarimax": 0.8, "samossa": 0.6, "mssa_rl": 0.2})
        assert weights
        assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-6)
        assert "sarimax" in weights and weights["sarimax"] >= 0.6
        assert score > 0

    def test_confidence_uses_regression_metrics_and_change_points(self) -> None:
        summaries = {
            "sarimax": {
                "aic": 120.0,
                "bic": 130.0,
                "regression_metrics": {"tracking_error": 0.8, "n_observations": 20},
            },
            "samossa": {
                "explained_variance_ratio": 0.6,
                "regression_metrics": {"tracking_error": 0.4, "n_observations": 20},
            },
            "mssa_rl": {
                "baseline_variance": 0.4,
                "regression_metrics": {"tracking_error": 0.3, "n_observations": 20},
                "change_point_density": 0.2,
                "recent_change_point_days": 2,
            },
        }
        confidence = derive_model_confidence(summaries)
        assert confidence["samossa"] > confidence["sarimax"]
        assert confidence["mssa_rl"] > confidence["samossa"]
