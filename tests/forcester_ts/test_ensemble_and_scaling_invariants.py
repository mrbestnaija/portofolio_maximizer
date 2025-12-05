import math
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from forcester_ts.ensemble import EnsembleConfig, EnsembleCoordinator
from forcester_ts.forecaster import TimeSeriesForecaster
from forcester_ts.samossa import SAMOSSAForecaster
from forcester_ts.garch import GARCHForecaster
from forcester_ts.sarimax import SARIMAXForecaster


class DummyForecastResult:
    """Minimal stub mimicking arch_model().fit().forecast()."""

    def __init__(self, variance: float, mean: float):
        # In the real arch API these are DataFrame-like; a Series is enough here.
        self.variance = pd.Series([variance])
        self.mean = pd.Series([mean])


class DummyFittedGarchModel:
    """Stub for GARCHForecaster.fitted_model to test rescaling logic in isolation."""

    def __init__(self, variance: float, mean: float, aic: float = 1.0, bic: float = 2.0):
        self._variance = variance
        self._mean = mean
        self.aic = aic
        self.bic = bic

    def forecast(self, horizon: int) -> DummyForecastResult:  # pragma: no cover - trivial wiring
        return DummyForecastResult(self._variance, self._mean)


class DummySARIMAXForecastResult:
    """Stub matching the subset of SARIMAXResults used in SARIMAXForecaster.forecast."""

    def __init__(self, mean: pd.Series, lower: pd.Series, upper: pd.Series):
        self._mean = mean
        self._lower = lower
        self._upper = upper

    @property
    def predicted_mean(self) -> pd.Series:  # pragma: no cover - simple accessor
        return self._mean

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:  # pragma: no cover - simple accessor
        return pd.DataFrame({"lower": self._lower, "upper": self._upper})


class DummySARIMAXFittedModel:
    """Stub for SARIMAXForecaster.fitted_model focusing on scaling/residual logic."""

    def __init__(self, mean: pd.Series, lower: pd.Series, upper: pd.Series, resid: pd.Series):
        self._forecast = DummySARIMAXForecastResult(mean, lower, upper)
        self.resid = resid
        self.aic = 1.0
        self.bic = 2.0

    def get_forecast(self, steps: int, exog=None) -> DummySARIMAXForecastResult:  # pragma: no cover - trivial wiring
        return self._forecast


def _sum_weights(weights: Dict[str, float]) -> float:
    return float(sum(weights.values()))


def test_ensemble_normalize_filters_non_positive_and_sums_to_one():
    cfg = EnsembleConfig(enabled=True, candidate_weights=[{"a": 1.0, "b": 2.0, "c": -1.0, "d": 0.0}])
    coord = EnsembleCoordinator(cfg)

    # Use neutral confidence so selection reduces to candidate weights.
    model_confidence = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    weights, score = coord.select_weights(model_confidence)

    # Negative/zero entries should be removed.
    assert "c" not in weights
    assert "d" not in weights

    # Remaining weights should be strictly positive and convex (sum to one).
    assert set(weights.keys()) == {"a", "b"}
    assert all(w > 0.0 for w in weights.values())
    assert math.isclose(_sum_weights(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9)

    # Selection score should be finite when at least one candidate survives.
    assert np.isfinite(score)


def test_ensemble_weights_stable_under_re_normalization():
    cfg = EnsembleConfig(
        enabled=True,
        candidate_weights=[{"sarimax": 0.3333333, "samossa": 0.3333333, "mssa_rl": 0.3333334}],
    )
    coord = EnsembleCoordinator(cfg)
    model_confidence = {"sarimax": 0.8, "samossa": 0.9, "mssa_rl": 0.85}

    weights, _ = coord.select_weights(model_confidence)
    # Simulate repeated use/round-trip of weights and ensure convexity preserved.
    for _ in range(5):
        weights = TimeSeriesForecaster._enforce_convexity(weights)
        assert all(w >= 0.0 for w in weights.values())
        assert math.isclose(_sum_weights(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_enforce_convexity_handles_degenerate_and_negative_weights():
    # All non-positive -> empty dict.
    empty = TimeSeriesForecaster._enforce_convexity({"a": -1.0, "b": 0.0})
    assert empty == {}

    # Mixed positive/negative should clamp negatives to zero and renormalize.
    raw = {"a": -0.5, "b": 0.25, "c": 0.75}
    normalized = TimeSeriesForecaster._enforce_convexity(raw)
    assert set(normalized.keys()) == {"a", "b", "c"}
    assert normalized["a"] == pytest.approx(0.0)
    assert normalized["b"] > 0.0
    assert normalized["c"] > 0.0
    assert all(w >= 0.0 for w in normalized.values())
    assert math.isclose(_sum_weights(normalized), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_samossa_normalization_produces_zero_mean_unit_std():
    # Construct a simple increasing series with mild noise so std is non-zero.
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    values = np.linspace(100.0, 120.0, num=len(idx)) + np.random.default_rng(0).normal(
        scale=0.5, size=len(idx)
    )
    series = pd.Series(values, index=idx)

    forecaster = SAMOSSAForecaster(
        window_length=40,
        n_components=4,
        min_series_length=150,
        forecast_horizon=10,
        normalize=True,
    )
    forecaster.fit(series)
    summary = forecaster.get_model_summary()

    # Normalized stats should be near 0 mean / 1 std, within reasonable tolerance.
    assert summary["scale_std"] > 0.0
    assert abs(summary["normalized_mean"]) < 1e-2
    assert math.isclose(summary["normalized_std"], 1.0, rel_tol=1e-2, abs_tol=1e-2)


def test_samossa_without_normalization_reports_raw_stats():
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    series = pd.Series(np.linspace(10.0, 20.0, num=len(idx)), index=idx)

    forecaster = SAMOSSAForecaster(
        window_length=40,
        n_components=4,
        min_series_length=150,
        forecast_horizon=10,
        normalize=False,
    )
    forecaster.fit(series)
    summary = forecaster.get_model_summary()

    # When normalize=False, scale_mean/std should be identity and normalized stats
    # should match the cleaned series statistics (up to floating-point noise).
    assert summary["scale_mean"] == pytest.approx(0.0)
    assert summary["scale_std"] == pytest.approx(1.0)
    raw_mean = float(series.sort_index().dropna().mean())
    raw_std = float(series.sort_index().dropna().std())
    assert summary["normalized_mean"] == pytest.approx(raw_mean, rel=1e-6, abs=1e-6)
    assert summary["normalized_std"] == pytest.approx(raw_std, rel=1e-6, abs=1e-6)


def test_garch_rescaling_inverse_of_input_scaling():
    # Create a GARCHForecaster instance without invoking __init__ so tests
    # do not depend on the arch library being available.
    forecaster = object.__new__(GARCHForecaster)  # type: ignore[misc]

    # Attach the minimal attributes used inside forecast().
    forecaster.p = 1
    forecaster.q = 1
    forecaster.vol = "GARCH"
    forecaster.dist = "normal"

    # Simulate a model that was fitted on returns scaled by 100x.
    forecaster._scale_factor = 100.0
    forecaster.fitted_model = DummyFittedGarchModel(variance=0.04, mean=0.2)

    results = forecaster.forecast(steps=5)

    # Internally, variance and mean are on the original (unscaled) return scale.
    variance_forecast = results["variance_forecast"]
    mean_forecast = results["mean_forecast"]
    volatility = results["volatility"]

    # After rescaling back, variance should be divided by scale_factor^2,
    # mean by scale_factor, and volatility is sqrt(variance).
    expected_var = 0.04 / (100.0**2)
    expected_mean = 0.2 / 100.0
    assert variance_forecast == pytest.approx(expected_var, rel=1e-9, abs=1e-9)
    assert mean_forecast == pytest.approx(expected_mean, rel=1e-9, abs=1e-9)
    assert volatility == pytest.approx(math.sqrt(expected_var), rel=1e-9, abs=1e-9)


def test_garch_no_rescaling_when_scale_factor_one():
    forecaster = object.__new__(GARCHForecaster)  # type: ignore[misc]
    forecaster.p = 1
    forecaster.q = 1
    forecaster.vol = "GARCH"
    forecaster.dist = "normal"
    forecaster._scale_factor = 1.0
    forecaster.fitted_model = DummyFittedGarchModel(variance=0.04, mean=0.2)

    results = forecaster.forecast(steps=3)

    assert results["variance_forecast"] == pytest.approx(0.04, rel=1e-9, abs=1e-9)
    assert results["mean_forecast"] == pytest.approx(0.2, rel=1e-9, abs=1e-9)
    assert results["volatility"] == pytest.approx(math.sqrt(0.04), rel=1e-9, abs=1e-9)


def test_sarimax_scale_series_maps_into_stable_range():
    # Very small magnitudes should be scaled up with factor >= 1.
    small_vals = pd.Series([1e-4, -2e-4, 5e-4])
    scaled_small, factor_small = SARIMAXForecaster._scale_series(small_vals)
    assert factor_small >= 1.0
    max_abs_small = float(scaled_small.dropna().abs().max())
    assert 0.0 < max_abs_small <= 1000.0

    # Very large magnitudes should be scaled down into [1, 1000].
    large_vals = pd.Series([1e5, -2e5, 5e4])
    scaled_large, factor_large = SARIMAXForecaster._scale_series(large_vals)
    assert factor_large <= 1.0
    max_abs_large = float(scaled_large.dropna().abs().max())
    assert 1.0 <= max_abs_large <= 1000.0


def test_sarimax_forecast_rescaling_inverse_of_scale_factor():
    # Build a SARIMAXForecaster without invoking __init__ so we can bypass statsmodels.
    forecaster = object.__new__(SARIMAXForecaster)  # type: ignore[misc]

    # Minimal attributes required by forecast().
    forecaster.best_order = (1, 0, 1)
    forecaster.best_seasonal_order = (0, 0, 0, 0)
    forecaster._scale_factor = 10.0
    forecaster.log_transform = False
    forecaster._series_transform = None

    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    mean_scaled = pd.Series([10.0, 20.0, 30.0], index=idx)
    lower_scaled = pd.Series([5.0, 15.0, 25.0], index=idx)
    upper_scaled = pd.Series([15.0, 25.0, 35.0], index=idx)
    resid_scaled = pd.Series([1.0, -2.0, 3.0], index=idx)

    forecaster.fitted_model = DummySARIMAXFittedModel(
        mean=mean_scaled,
        lower=lower_scaled,
        upper=upper_scaled,
        resid=resid_scaled,
    )

    result = forecaster.forecast(steps=3)

    forecast = result["forecast"]
    lower_ci = result["lower_ci"]
    upper_ci = result["upper_ci"]
    diagnostics = result["diagnostics"]

    # Values should be divided by the scale factor.
    assert forecast.equals(mean_scaled / forecaster._scale_factor)
    assert lower_ci.equals(lower_scaled / forecaster._scale_factor)
    assert upper_ci.equals(upper_scaled / forecaster._scale_factor)

    # Residual statistics should reflect the rescaled residuals.
    expected_resid = resid_scaled / forecaster._scale_factor
    assert diagnostics["residual_mean"] == pytest.approx(expected_resid.mean(), rel=1e-9, abs=1e-9)
    assert diagnostics["residual_std"] == pytest.approx(expected_resid.std(), rel=1e-9, abs=1e-9)

