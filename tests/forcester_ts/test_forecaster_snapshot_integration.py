from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig


def test_samossa_snapshot_restore_skips_refit(monkeypatch) -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    series = pd.Series(np.linspace(100.0, 120.0, 80), index=index)

    class _FakeStore:
        def __init__(self) -> None:
            self.load_calls: list[dict] = []
            self.save_calls = 0

        def load(self, **kwargs):
            self.load_calls.append(dict(kwargs))
            return {
                "reconstructed": series.copy(),
                "residuals": pd.Series(np.zeros(len(series)), index=index),
                "residual_model": None,
                "scale_mean": 0.0,
                "scale_std": 1.0,
                "evr": 1.0,
                "trend_slope": 0.0,
                "trend_intercept": 0.0,
                "trend_strength": 0.0,
                "last_index": index[-1],
                "target_freq": "B",
                "last_observed": float(series.iloc[-1]),
                "normalized_stats": {"mean": 0.0, "std": 1.0},
            }

        def save(self, **kwargs):
            self.save_calls += 1

    def _unexpected_fit(self, *args, **kwargs):
        raise AssertionError("SAMOSSA fit should be skipped when exact snapshot restore succeeds")

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=True,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
        order_learning_config={"enabled": True},
    )
    forecaster = TimeSeriesForecaster(config=config)
    fake_store = _FakeStore()
    forecaster._snapshot_store = fake_store

    monkeypatch.setattr("forcester_ts.samossa.SAMOSSAForecaster.fit", _unexpected_fit)
    forecaster.fit(series, ticker="AAPL")

    assert fake_store.load_calls
    assert fake_store.load_calls[0]["strict_hash"] is True
    assert fake_store.load_calls[0]["max_obs_delta"] == 0
    assert fake_store.save_calls == 0
    assert forecaster._samossa is not None
    assert getattr(forecaster._samossa, "_fitted", False) is True


def test_samossa_snapshot_restore_uses_series_name_when_ticker_missing(monkeypatch) -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    series = pd.Series(np.linspace(100.0, 120.0, 80), index=index, name="MSFT")

    class _FakeStore:
        def __init__(self) -> None:
            self.load_calls: list[dict] = []

        def load(self, **kwargs):
            self.load_calls.append(dict(kwargs))
            return {
                "reconstructed": series.copy(),
                "residuals": pd.Series(np.zeros(len(series)), index=index),
                "residual_model": None,
                "scale_mean": 0.0,
                "scale_std": 1.0,
                "evr": 1.0,
                "trend_slope": 0.0,
                "trend_intercept": 0.0,
                "trend_strength": 0.0,
                "last_index": index[-1],
                "target_freq": "B",
                "last_observed": float(series.iloc[-1]),
                "normalized_stats": {"mean": 0.0, "std": 1.0},
            }

        def save(self, **kwargs):
            raise AssertionError("save should not run on a snapshot restore hit")

    def _unexpected_fit(self, *args, **kwargs):
        raise AssertionError("SAMOSSA fit should be skipped when exact snapshot restore succeeds")

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=True,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
        order_learning_config={"enabled": True},
    )
    forecaster = TimeSeriesForecaster(config=config)
    fake_store = _FakeStore()
    forecaster._snapshot_store = fake_store

    monkeypatch.setattr("forcester_ts.samossa.SAMOSSAForecaster.fit", _unexpected_fit)
    forecaster.fit(series)

    assert fake_store.load_calls
    assert fake_store.load_calls[0]["ticker"] == "MSFT"


def test_samossa_snapshot_restore_restores_metadata_and_only_marks_usage(monkeypatch) -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    series = pd.Series(np.linspace(100.0, 120.0, 80), index=index)

    class _FakeStore:
        def load(self, **kwargs):  # noqa: ARG002
            return {
                "reconstructed": series.copy(),
                "residuals": pd.Series(np.zeros(len(series)), index=index),
                "residual_model": None,
                "scale_mean": 0.0,
                "scale_std": 1.0,
                "evr": 1.0,
                "trend_slope": 0.0,
                "trend_intercept": 0.0,
                "trend_strength": 0.0,
                "last_index": index[-1],
                "target_freq": "B",
                "last_observed": float(series.iloc[-1]),
                "normalized_stats": {"mean": 0.0, "std": 1.0},
                "config_window_length": 87,
                "config_n_components": 3,
                "learned_ar_lag": 4,
                "learned_ar_aic": 12.5,
                "learned_ar_bic": 13.5,
            }

        def save(self, **kwargs):  # noqa: ARG002
            raise AssertionError("save should not run on a snapshot restore hit")

    class _FakeLearner:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def record_usage(self, **kwargs):
            self.calls.append(dict(kwargs))
            return True

    def _unexpected_fit(self, *args, **kwargs):
        raise AssertionError("SAMOSSA fit should be skipped when exact snapshot restore succeeds")

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=False,
        garch_enabled=False,
        samossa_enabled=True,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
        order_learning_config={"enabled": True},
    )
    forecaster = TimeSeriesForecaster(config=config)
    forecaster._snapshot_store = _FakeStore()
    fake_learner = _FakeLearner()
    forecaster._order_learner = fake_learner

    monkeypatch.setattr("forcester_ts.samossa.SAMOSSAForecaster.fit", _unexpected_fit)
    forecaster.fit(series, ticker="AAPL")

    assert forecaster._samossa is not None
    summary = forecaster._samossa.get_model_summary()
    assert summary["window_length_used"] == 87
    assert summary["n_components"] == 3
    assert fake_learner.calls == [
        {
            "ticker": "AAPL",
            "model_type": "SAMOSSA_ARIMA",
            "regime": None,
            "order_params": {"ar_lag": 4},
        }
    ]
