"""
Regression test: Forecasters vs random-walk baseline.

Goal:
Confirm core forecasters (SARIMAX/SAMOSSA/Ensemble) achieve lower RMSE and sMAPE
than a random-walk (last-value) baseline on a structured price series.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.metrics import compute_regression_metrics


def _random_walk_baseline(train: pd.Series, holdout_index: pd.DatetimeIndex) -> pd.Series:
    cleaned = train.dropna()
    last = float(cleaned.iloc[-1])
    return pd.Series(last, index=holdout_index, name="rw_baseline")


def _build_forecaster_config(horizon: int) -> TimeSeriesForecasterConfig:
    return TimeSeriesForecasterConfig(
        forecast_horizon=horizon,
        garch_enabled=False,
        sarimax_kwargs={
            "auto_select": True,
            "trend": "auto",
            "max_p": 1,
            "max_d": 1,
            "max_q": 1,
            # Keep runtime bounded; daily tests don't need seasonal search here.
            "seasonal_periods": 0,
            "max_P": 0,
            "max_D": 0,
            "max_Q": 0,
            "order_search_mode": "compact",
            "order_search_maxiter": 60,
        },
        samossa_kwargs={
            "window_length": 40,
            "n_components": 6,
            "min_series_length": 120,
            "forecast_horizon": horizon,
        },
        mssa_rl_kwargs={
            "window_length": 30,
            # Reduce false-positive change-points so MSSA-RL behaves as a stable baseline.
            "change_point_threshold": 10.0,
            "forecast_horizon": horizon,
            "use_gpu": False,
        },
        ensemble_kwargs={"enabled": True},
    )


def _evaluate_against_random_walk(price_series: pd.Series, *, horizon: int) -> dict:
    train = price_series.iloc[:-horizon]
    holdout = price_series.iloc[-horizon:]

    baseline_forecast = _random_walk_baseline(train, holdout.index)
    baseline_metrics = compute_regression_metrics(holdout, baseline_forecast) or {}

    cfg = _build_forecaster_config(horizon)

    prior_audit = os.environ.pop("TS_FORECAST_AUDIT_DIR", None)
    try:
        forecaster = TimeSeriesForecaster(config=cfg)
        returns = train.pct_change().dropna()
        forecaster.fit(price_series=train, returns_series=returns)
        forecaster.forecast(steps=horizon)
        model_metrics = forecaster.evaluate(holdout)
    finally:
        if prior_audit is not None:
            os.environ["TS_FORECAST_AUDIT_DIR"] = prior_audit

    return {
        "baseline": baseline_metrics,
        "models": model_metrics,
        "horizon": horizon,
    }


@pytest.fixture(scope="session")
def structured_price_series() -> pd.Series:
    """
    Deterministic synthetic series with strong structure.

    - Strong linear trend (so RW baseline is predictably wrong in holdout).
    - Mild seasonality + noise (so models still need to generalize).
    """
    rng = np.random.default_rng(20260113)
    periods = 260
    index = pd.date_range("2023-01-01", periods=periods, freq="D")
    t = np.arange(periods, dtype=float)
    trend = 0.6 * t
    seasonal = 3.0 * np.sin(2.0 * np.pi * t / 14.0)
    noise = rng.normal(0.0, 0.35, size=periods)
    prices = 100.0 + trend + seasonal + noise
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture(scope="session")
def random_walk_price_series() -> pd.Series:
    """Deterministic random walk (hard case: no exploitable structure)."""
    rng = np.random.default_rng(20260113 + 7)
    periods = 260
    index = pd.date_range("2023-01-01", periods=periods, freq="D")
    returns = rng.normal(0.0, 0.01, size=periods)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture(scope="session")
def structured_forecast_comparison(structured_price_series: pd.Series) -> dict:
    return _evaluate_against_random_walk(structured_price_series, horizon=20)


@pytest.fixture(scope="session")
def random_walk_forecast_comparison(random_walk_price_series: pd.Series) -> dict:
    return _evaluate_against_random_walk(random_walk_price_series, horizon=20)


@pytest.mark.integration
def test_forecasters_beat_random_walk_baseline_on_structured_series(
    structured_forecast_comparison: dict,
) -> None:
    baseline = structured_forecast_comparison["baseline"]
    models = structured_forecast_comparison["models"]
    horizon = int(structured_forecast_comparison["horizon"])

    assert "rmse" in baseline and "smape" in baseline
    baseline_rmse = float(baseline["rmse"])
    baseline_smape = float(baseline["smape"])
    assert np.isfinite(baseline_rmse) and baseline_rmse > 0.0
    assert np.isfinite(baseline_smape) and 0.0 <= baseline_smape <= 2.0

    # Core forecasters should materially outperform a naive last-value baseline.
    required = ("sarimax", "samossa", "ensemble")
    for name in required:
        assert name in models, f"Missing metrics for {name}: keys={sorted(models.keys())}"
        metrics = models[name]
        rmse_val = float(metrics["rmse"])
        smape_val = float(metrics["smape"])
        assert int(metrics.get("n_observations", 0)) == horizon
        assert np.isfinite(rmse_val) and rmse_val >= 0.0
        assert np.isfinite(smape_val) and 0.0 <= smape_val <= 2.0

        # Require a material improvement on this strongly-structured series.
        # Note: sMAPE can be small on high-price series; keep the threshold
        # conservative to avoid flakiness while still catching regressions.
        improvement_ratio = 0.85  # >= 15% better than RW baseline
        assert rmse_val <= baseline_rmse * improvement_ratio, (
            f"{name} RMSE did not beat RW baseline: model={rmse_val:.4f} "
            f"baseline={baseline_rmse:.4f} metrics={metrics}"
        )
        assert smape_val <= baseline_smape * improvement_ratio, (
            f"{name} sMAPE did not beat RW baseline: model={smape_val:.4f} "
            f"baseline={baseline_smape:.4f} metrics={metrics}"
        )

    # MSSA-RL is primarily a diagnostics/regime model; require it is not catastrophically worse.
    if "mssa_rl" in models:
        mssa = models["mssa_rl"]
        rmse_val = float(mssa["rmse"])
        smape_val = float(mssa["smape"])
        assert int(mssa.get("n_observations", 0)) == horizon
        assert rmse_val <= baseline_rmse * 2.0
        assert smape_val <= baseline_smape * 2.0


@pytest.mark.integration
def test_forecasters_do_not_collapse_on_random_walk_series(
    random_walk_forecast_comparison: dict,
) -> None:
    baseline = random_walk_forecast_comparison["baseline"]
    models = random_walk_forecast_comparison["models"]
    horizon = int(random_walk_forecast_comparison["horizon"])

    assert "rmse" in baseline and "smape" in baseline
    baseline_rmse = float(baseline["rmse"])
    baseline_smape = float(baseline["smape"])
    assert np.isfinite(baseline_rmse) and baseline_rmse > 0.0
    assert np.isfinite(baseline_smape) and 0.0 <= baseline_smape <= 2.0

    # On a true random-walk, models should not catastrophically underperform a naive baseline.
    # Keep tolerance tight enough to catch regressions but wide enough to avoid flakiness.
    max_degradation = 1.25  # <= 25% worse than RW baseline
    checked = ("sarimax", "samossa", "mssa_rl", "ensemble")
    for name in checked:
        if name not in models:
            continue
        metrics = models[name]
        rmse_val = float(metrics["rmse"])
        smape_val = float(metrics["smape"])
        assert int(metrics.get("n_observations", 0)) == horizon
        assert rmse_val <= baseline_rmse * max_degradation, (
            f"{name} RMSE collapsed vs RW baseline: model={rmse_val:.4f} "
            f"baseline={baseline_rmse:.4f} metrics={metrics}"
        )
        assert smape_val <= baseline_smape * max_degradation, (
            f"{name} sMAPE collapsed vs RW baseline: model={smape_val:.4f} "
            f"baseline={baseline_smape:.4f} metrics={metrics}"
        )
