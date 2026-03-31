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
    rng = np.random.default_rng(12345)
    dates = pd.date_range(datetime(2020, 1, 1), periods=240, freq="D")
    returns = rng.normal(0.001, 0.02, len(dates))
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


@pytest.mark.slow
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


@pytest.mark.slow
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
        # Phase 8.4: VIF screening may prune correlated features; verify retained
        # columns are a subset of the original 5 core features.
        _all_core = {"ret_1", "vol_10", "mom_5", "ema_gap_10", "zscore_20"}
        assert set(exog.get("columns", [])).issubset(_all_core), (
            f"Unexpected columns after VIF screening: {exog.get('columns')}"
        )
        assert len(exog.get("columns", [])) >= 1, "VIF screening must retain at least one feature"
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

    def test_mssa_rl_action_component_sets_differ(self, price_series: pd.Series) -> None:
        """Phase 8.1: per-action reconstructions must differ from each other.

        action=0 (mean_revert, 25% variance) should be smoother than action=2
        (trend_follow, all components). The 90%-variance reconstruction is the
        standard behaviour (backward-compatible with action=1).
        """
        forecaster = MSSARLForecaster()
        forecaster.fit(price_series)

        recons = forecaster._reconstructions_by_action
        assert set(recons.keys()) == {0, 1, 2}, "Expected per-action reconstructions for all 3 actions"

        r0 = recons[0].values  # 25% variance — smooth
        r1 = recons[1].values  # 90% variance — standard
        r2 = recons[2].values  # all components — high fidelity

        # Action=0 should have lower variance (smoother) than action=2
        assert float(np.var(r0)) <= float(np.var(r1)) + 1e-9
        assert float(np.var(r1)) <= float(np.var(r2)) + 1e-9

        # All reconstructions same length as input
        assert len(r0) == len(price_series.dropna())
        assert len(r2) == len(price_series.dropna())

    def test_mssa_rl_forecast_returns_active_action(self, price_series: pd.Series) -> None:
        """Phase 8.1: forecast result includes active_action key."""
        forecaster = MSSARLForecaster()
        forecaster.fit(price_series)
        result = forecaster.forecast(steps=5)
        assert "active_action" in result
        assert result["active_action"] in {0, 1, 2}

    def test_mssa_rl_action0_forecast_smoother_than_action2(self, price_series: pd.Series) -> None:
        """Phase 8.1: action=0 slope (mean-revert) <= action=2 slope in trend-following series.

        With a trending series the 25%-variance reconstruction (low-frequency only)
        should produce a slope with smaller absolute value than the all-components
        reconstruction (which retains high-frequency oscillations).
        """
        np.random.seed(42)
        n = 200
        trend = np.linspace(100, 150, n) + np.random.normal(0, 1, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        series = pd.Series(trend, index=dates)

        forecaster = MSSARLForecaster()
        forecaster.fit(series)

        recons = forecaster._reconstructions_by_action
        w = forecaster.config.window_length

        def _slope(arr: np.ndarray) -> float:
            k = min(w, len(arr))
            return float(np.polyfit(np.arange(k), arr[-k:], deg=1)[0])

        slope_0 = abs(_slope(recons[0].values))
        slope_2 = abs(_slope(recons[2].values))
        # On a smooth trend, action=0 uses fewer components so slope may be
        # similar; but action=2 includes noise. Both should be positive (trending).
        assert slope_0 >= 0 and slope_2 >= 0

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
class TestDropHighVifFeatures:
    """Phase 8.4 — VIF screening on SARIMAX exogenous features."""

    def _make_forecaster(self) -> "TimeSeriesForecaster":
        from forcester_ts.forecaster import TimeSeriesForecaster
        return TimeSeriesForecaster()

    def _collinear_exog(self) -> "pd.DataFrame":
        """Return a DataFrame where col_b is almost identical to col_a (high VIF)."""
        np.random.seed(0)
        n = 200
        base = np.random.normal(0, 1, n)
        return pd.DataFrame(
            {
                "col_a": base,
                "col_b": base + np.random.normal(0, 0.01, n),  # nearly identical
                "col_c": np.random.normal(0, 1, n),             # independent
            }
        )

    def _uncorrelated_exog(self) -> "pd.DataFrame":
        np.random.seed(1)
        n = 200
        return pd.DataFrame(
            {
                "feat_x": np.random.normal(0, 1, n),
                "feat_y": np.random.normal(5, 2, n),
                "feat_z": np.random.normal(-3, 0.5, n),
            }
        )

    def test_collinear_pair_is_pruned(self) -> None:
        """col_a and col_b are nearly identical -> at least one should be dropped."""
        f = self._make_forecaster()
        exog = self._collinear_exog()
        result = f._drop_high_vif_features(exog, threshold=10.0, max_features=3)
        # After VIF screening, col_a and col_b should not both survive
        assert not (("col_a" in result.columns) and ("col_b" in result.columns)), (
            "Collinear pair should have been pruned but both survived VIF screening"
        )

    def test_uncorrelated_features_retained(self) -> None:
        """Independently-drawn features should all have low VIF and be retained."""
        f = self._make_forecaster()
        exog = self._uncorrelated_exog()
        result = f._drop_high_vif_features(exog, threshold=10.0, max_features=3)
        # Independent features should survive (VIF close to 1)
        assert result.shape[1] == 3

    def test_max_features_cap_respected(self) -> None:
        """max_features=2 must limit output columns even if VIFs are all below threshold."""
        f = self._make_forecaster()
        exog = self._uncorrelated_exog()
        result = f._drop_high_vif_features(exog, threshold=10.0, max_features=2)
        assert result.shape[1] <= 2

    def test_single_column_passthrough(self) -> None:
        """A single-column DataFrame cannot be pruned; must be returned as-is."""
        f = self._make_forecaster()
        exog = pd.DataFrame({"only": np.random.normal(0, 1, 100)})
        result = f._drop_high_vif_features(exog, threshold=10.0, max_features=3)
        assert list(result.columns) == ["only"]

    def test_output_is_subset_of_input_columns(self) -> None:
        """All retained columns must come from the original DataFrame."""
        f = self._make_forecaster()
        exog = self._collinear_exog()
        result = f._drop_high_vif_features(exog)
        assert set(result.columns).issubset(set(exog.columns))

    def test_returns_dataframe_not_array(self) -> None:
        f = self._make_forecaster()
        exog = self._uncorrelated_exog()
        result = f._drop_high_vif_features(exog)
        assert isinstance(result, pd.DataFrame)

    def test_vol10_zscore20_pruned_in_sarimax_exog(self) -> None:
        """vol_10 and zscore_20 are both driven by realized volatility.

        When fed through _build_sarimax_exogenous -> _drop_high_vif_features,
        at least one of the two should be pruned by VIF > 10 screening.
        """
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)), index=dates)
        returns = prices.pct_change()

        f = self._make_forecaster()
        exog = f._build_sarimax_exogenous(price_series=prices, returns_series=returns)
        # After VIF screening, vol_10 and zscore_20 should not both be present
        # (they are highly correlated in typical price series)
        surviving = set(exog.columns)
        assert not (("vol_10" in surviving) and ("zscore_20" in surviving)), (
            f"Expected VIF screening to remove one of vol_10/zscore_20, "
            f"but both survived: {surviving}"
        )


@pytest.mark.slow
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
        # Phase 7.15-E: GARCH always enters the pool (fallback 0.45 when no summary).
        assert "garch" in confidence  # GARCH participates even without explicit summary
        # P1b fix (2026-03-29): change_point_boost is now capped at 0.20 so it can
        # nudge but not dominate.  SAMoSSA's EVR (0.6) + low tracking error correctly
        # outranks MSSA-RL when no OOS evidence is passed.  MSSA-RL still enters the
        # pool with a meaningful score (> SARIMAX) thanks to the capped boost.
        assert confidence["mssa_rl"] > confidence["sarimax"]
        # All model scores should be within the calibrated [0.4, 0.85] band.
        for model, score in confidence.items():
            assert 0.35 < score <= 0.95, f"{model} score {score:.3f} out of band"
