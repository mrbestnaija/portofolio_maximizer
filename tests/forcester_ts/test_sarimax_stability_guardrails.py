from __future__ import annotations

import math

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
    }

    summary = forecaster.get_model_summary()
    assert summary["fit_strategy"] == "strict_powell_retry"
    assert summary["convergence"]["powell_retry_converged"] is True
    assert math.isclose(summary["aic"], 123.4)
