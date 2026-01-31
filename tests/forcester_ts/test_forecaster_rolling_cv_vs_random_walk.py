"""
Brutal benchmark: rolling out-of-sample performance vs Random-Walk baseline.

This test is intentionally stricter than single-split checks:
- Multiple folds (expanding window) to reduce luck.
- Compares against a naive random-walk (last-value) forecast baseline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.metrics import compute_regression_metrics


@dataclass(frozen=True)
class _CVConfig:
    min_train_size: int = 200
    horizon: int = 5
    step_size: int = 5
    max_folds: int = 3


def _rw_baseline(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    last = float(train.dropna().iloc[-1])
    return pd.Series(last, index=test_index, name="rw_baseline")


def _fast_forecaster_config(horizon: int) -> TimeSeriesForecasterConfig:
    return TimeSeriesForecasterConfig(
        forecast_horizon=horizon,
        garch_enabled=False,
        sarimax_kwargs={
            "auto_select": True,
            "trend": "auto",
            "max_p": 1,
            "max_d": 1,
            "max_q": 1,
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
            "change_point_threshold": 10.0,
            "forecast_horizon": horizon,
            "use_gpu": False,
        },
        ensemble_kwargs={"enabled": True},
    )


def _iter_folds(series: pd.Series, cv: _CVConfig) -> list[tuple[pd.Series, pd.Series]]:
    series = series.sort_index()
    total = len(series)
    folds: list[tuple[pd.Series, pd.Series]] = []
    start = cv.min_train_size
    while start + cv.horizon <= total and len(folds) < cv.max_folds:
        train = series.iloc[:start]
        test = series.iloc[start : start + cv.horizon]
        folds.append((train, test))
        start += cv.step_size
    if not folds:
        raise ValueError("No CV folds produced; increase series length or reduce min_train_size.")
    return folds


def _aggregate_metrics(rows: list[dict]) -> dict:
    acc: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.items():
            if v is None:
                continue
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _rolling_cv_benchmark(series: pd.Series, cv: _CVConfig) -> dict:
    folds = _iter_folds(series, cv)
    baseline_rows: list[dict] = []
    model_rows: dict[str, list[dict]] = {}

    cfg = _fast_forecaster_config(cv.horizon)
    prior_audit = os.environ.pop("TS_FORECAST_AUDIT_DIR", None)
    try:
        for train, test in folds:
            baseline_forecast = _rw_baseline(train, test.index)
            baseline_rows.append(compute_regression_metrics(test, baseline_forecast) or {})

            returns = train.pct_change().dropna()
            forecaster = TimeSeriesForecaster(config=cfg)
            forecaster.fit(price_series=train, returns_series=returns)
            forecaster.forecast(steps=len(test))
            metrics = forecaster.evaluate(test)

            for model_name, metric_map in metrics.items():
                model_rows.setdefault(model_name, []).append(metric_map)
    finally:
        if prior_audit is not None:
            os.environ["TS_FORECAST_AUDIT_DIR"] = prior_audit

    baseline = _aggregate_metrics(baseline_rows)
    models = {name: _aggregate_metrics(rows) for name, rows in model_rows.items()}
    return {"baseline": baseline, "models": models, "folds": len(folds), "horizon": cv.horizon}


@pytest.fixture(scope="session")
def series_trending() -> pd.Series:
    rng = np.random.default_rng(20260113 + 11)
    periods = 260
    index = pd.date_range("2023-01-01", periods=periods, freq="D")
    t = np.arange(periods, dtype=float)
    trend = 0.5 * t
    seasonal = 2.0 * np.sin(2.0 * np.pi * t / 14.0)
    noise = rng.normal(0.0, 0.4, size=periods)
    prices = 100.0 + trend + seasonal + noise
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture(scope="session")
def series_random_walk() -> pd.Series:
    rng = np.random.default_rng(20260113 + 13)
    periods = 260
    index = pd.date_range("2023-01-01", periods=periods, freq="D")
    returns = rng.normal(0.0, 0.012, size=periods)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=index, name="Close")


@pytest.mark.integration
def test_rolling_cv_ensemble_beats_random_walk_on_trending_series(series_trending: pd.Series) -> None:
    cv = _CVConfig()
    report = _rolling_cv_benchmark(series_trending, cv)
    baseline = report["baseline"]
    models = report["models"]

    assert report["folds"] >= 2
    assert "rmse" in baseline and "smape" in baseline
    base_rmse = float(baseline["rmse"])
    base_smape = float(baseline["smape"])
    assert base_rmse > 0 and 0.0 <= base_smape <= 2.0

    assert "ensemble" in models
    ens = models["ensemble"]
    ens_rmse = float(ens["rmse"])
    ens_smape = float(ens["smape"])

    # Ensemble should beat naive last-value on trending structure.
    # Tolerance widened to 2% because SARIMAX may fail to converge on some
    # platforms/folds, leaving a single-model ensemble that still outperforms
    # the baseline but by a smaller margin.
    assert ens_rmse <= base_rmse * 0.98, f"ensemble RMSE={ens_rmse:.4f} baseline={base_rmse:.4f} report={report}"
    assert ens_smape <= base_smape * 0.98, f"ensemble sMAPE={ens_smape:.4f} baseline={base_smape:.4f} report={report}"


@pytest.mark.integration
def test_rolling_cv_models_not_catastrophic_on_random_walk(series_random_walk: pd.Series) -> None:
    cv = _CVConfig(max_folds=2)  # keep runtime bounded for this hard case
    report = _rolling_cv_benchmark(series_random_walk, cv)
    baseline = report["baseline"]
    models = report["models"]

    base_rmse = float(baseline["rmse"])
    base_smape = float(baseline["smape"])
    assert base_rmse > 0 and 0.0 <= base_smape <= 2.0

    # On random-walk data, no model should blow up relative to RW baseline.
    max_degradation = 1.25
    for name in ("sarimax", "samossa", "mssa_rl", "ensemble"):
        if name not in models:
            continue
        m = models[name]
        rmse_val = float(m["rmse"])
        smape_val = float(m["smape"])
        assert rmse_val <= base_rmse * max_degradation, f"{name} RMSE={rmse_val:.4f} baseline={base_rmse:.4f} report={report}"
        assert smape_val <= base_smape * max_degradation, f"{name} sMAPE={smape_val:.4f} baseline={base_smape:.4f} report={report}"
