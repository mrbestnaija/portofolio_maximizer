from __future__ import annotations

import math

import numpy as np
import pandas as pd

from forcester_ts.sarimax import SARIMAXForecaster


class _DummyResult:
    def __init__(self, aic, bic=0.0, llf=0.0, nobs=10):
        self.aic = aic
        self.bic = bic
        self.llf = llf
        self.nobs = nobs


def test_select_preferred_fit_chooses_lowest_finite_aic():
    primary_model = object()
    fallback_model = object()
    candidates = [
        ("primary_strict", primary_model, _DummyResult(float("inf"))),
        ("strict_powell_retry", primary_model, _DummyResult(220.0)),
        ("relaxed_powell_fallback", fallback_model, _DummyResult(180.0)),
    ]

    label, model, result = SARIMAXForecaster._select_preferred_fit(candidates)
    assert label == "relaxed_powell_fallback"
    assert model is fallback_model
    assert result.aic == 180.0


def test_select_preferred_fit_defaults_to_first_when_all_invalid():
    primary_model = object()
    candidates = [
        ("primary_strict", primary_model, _DummyResult(float("nan"))),
        ("strict_powell_retry", primary_model, _DummyResult(None)),
    ]

    label, model, _ = SARIMAXForecaster._select_preferred_fit(candidates)
    assert label == "primary_strict"
    assert model is primary_model


def test_model_summary_includes_convergence_metadata():
    forecaster = object.__new__(SARIMAXForecaster)  # bypass __init__
    forecaster.fitted_model = _DummyResult(aic=123.4, bic=130.1, llf=-55.5, nobs=240)
    forecaster.best_order = (1, 1, 1)
    forecaster.best_seasonal_order = (0, 0, 0, 0)
    forecaster._log_shift = None
    forecaster._fit_metadata = {
        "fit_strategy": "strict_powell_retry",
        "primary_converged": False,
        "powell_retry_attempted": True,
        "powell_retry_converged": True,
        "fallback_attempted": False,
        "fallback_converged": False,
        "selected_constraints": "strict",
        "stationarity_source": "forecaster_hint",
        "stationarity_verdict": "conflicted",
        "force_difference": True,
    }

    summary = forecaster.get_model_summary()
    assert summary["fit_strategy"] == "strict_powell_retry"
    assert summary["convergence"]["powell_retry_converged"] is True
    assert summary["stationarity_source"] == "forecaster_hint"
    assert summary["stationarity_verdict"] == "conflicted"
    assert summary["force_difference"] is True
    assert math.isclose(summary["aic"], 123.4)


def test_stationarity_hint_forces_differencing():
    forecaster = SARIMAXForecaster()
    series = pd.Series(np.linspace(100.0, 120.0, 80), index=pd.date_range("2024-01-01", periods=80, freq="D"))
    forecaster._stationarity_hint = {
        "stationarity_verdict": "conflicted",
        "force_difference": True,
    }

    stationary, recommend_d, meta = forecaster._resolve_stationarity_choice(series)

    assert stationary is False
    assert recommend_d == 1
    assert meta["stationarity_source"] == "forecaster_hint"
    assert meta["stationarity_verdict"] == "conflicted"
    assert meta["force_difference"] is True


def test_stationarity_hint_stationary_caps_d_zero():
    forecaster = SARIMAXForecaster()
    series = pd.Series(np.linspace(100.0, 120.0, 80), index=pd.date_range("2024-01-01", periods=80, freq="D"))
    forecaster._stationarity_hint = {
        "stationarity_verdict": "stationary",
        "force_difference": False,
    }

    stationary, recommend_d, meta = forecaster._resolve_stationarity_choice(series)

    assert stationary is True
    assert recommend_d == 0
    assert meta["stationarity_source"] == "forecaster_hint"
    assert meta["stationarity_verdict"] == "stationary"
    assert meta["force_difference"] is False


def test_stationarity_choice_falls_back_to_local_test_when_hint_missing(monkeypatch):
    forecaster = SARIMAXForecaster()
    series = pd.Series(np.linspace(100.0, 120.0, 80), index=pd.date_range("2024-01-01", periods=80, freq="D"))
    monkeypatch.setattr(
        SARIMAXForecaster,
        "_test_stationarity",
        staticmethod(lambda data: (False, 1)),
    )

    stationary, recommend_d, meta = forecaster._resolve_stationarity_choice(series)

    assert stationary is False
    assert recommend_d == 1
    assert meta["stationarity_source"] == "sarimax_local_test"
