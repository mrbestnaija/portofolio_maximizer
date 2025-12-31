"""
Tests for rolling-window cross-validation utilities.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from etl.time_series_forecaster import (
    RollingWindowCVConfig,
    RollingWindowValidator,
    TimeSeriesForecasterConfig,
)


def _build_price_series(start: str = "2020-01-01", periods: int = 240) -> pd.Series:
    dates = pd.date_range(datetime.fromisoformat(start), periods=periods, freq="D")
    trend = 0.05 * np.arange(periods)
    seasonal = 2.0 * np.sin(2 * np.pi * np.arange(periods) / 20)
    noise = np.random.normal(0, 0.3, size=periods)
    values = 100 + trend + seasonal + noise
    return pd.Series(values, index=dates, name="Close")


class TestRollingWindowValidator:
    def test_generates_fold_metrics(self) -> None:
        price_series = _build_price_series()
        returns = price_series.pct_change().dropna()
        forecaster_config = TimeSeriesForecasterConfig(
            forecast_horizon=5,
            sarimax_enabled=True,
            garch_enabled=False,
            samossa_enabled=False,
            mssa_rl_enabled=False,
            ensemble_enabled=False,
            sarimax_kwargs={"auto_select": True},
        )
        cv_config = RollingWindowCVConfig(min_train_size=120, horizon=5, step_size=20, max_folds=3)
        validator = RollingWindowValidator(forecaster_config=forecaster_config, cv_config=cv_config)
        results = validator.run(price_series=price_series, returns_series=returns)

        assert results["fold_count"] == len(results["folds"])
        assert results["fold_count"] > 0

        first_fold = results["folds"][0]
        assert first_fold["train_range"]["end"] < first_fold["test_range"]["start"]
        assert "sarimax" in first_fold["metrics"]

        aggregate = results["aggregate_metrics"]
        assert "sarimax" in aggregate
        assert aggregate["sarimax"]["rmse"] >= 0
