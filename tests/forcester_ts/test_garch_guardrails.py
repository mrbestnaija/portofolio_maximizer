from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from forcester_ts.garch import GARCHForecaster


def _returns(n: int = 150, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, size=n), name="returns")


class _FakeArchFit:
    def __init__(
        self,
        *,
        alpha: float = 0.05,
        beta: float = 0.80,
        conditional_volatility: float = 0.02,
        n_obs: int = 150,
    ) -> None:
        self.aic = 10.0
        self.bic = 11.0
        self.loglikelihood = -5.0
        self.convergence_flag = 0
        self.params = {"alpha[1]": alpha, "beta[1]": beta}
        self.resid = pd.Series(np.random.default_rng(0).normal(0.0, 1.0, n_obs))
        self.conditional_volatility = pd.Series([conditional_volatility] * n_obs)


def test_fit_falls_back_to_ewma_when_sample_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_arch = MagicMock()
    monkeypatch.setattr("forcester_ts.garch.ARCH_AVAILABLE", True)
    monkeypatch.setattr("forcester_ts.garch.arch_model", mock_arch)

    forecaster = GARCHForecaster(
        backend="arch",
        auto_select=True,
        max_p=1,
        max_q=1,
        min_arch_sample_size=120,
    )

    forecaster.fit(_returns(n=60))

    mock_arch.assert_not_called()
    assert forecaster.backend == "ewma"
    assert forecaster.get_model_summary()["fallback_reason"] == "insufficient_sample_size"


def test_fit_falls_back_to_ewma_when_volatility_ratio_explodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("forcester_ts.garch.ARCH_AVAILABLE", True)

    def _fake_arch_model(*args, **kwargs):  # noqa: ARG001
        model = MagicMock()
        model.fit = lambda disp="off": _FakeArchFit(conditional_volatility=1.0)
        return model

    monkeypatch.setattr("forcester_ts.garch.arch_model", _fake_arch_model)

    forecaster = GARCHForecaster(
        backend="arch",
        auto_select=True,
        max_p=1,
        max_q=1,
        min_arch_sample_size=120,
        max_volatility_ratio_to_realized=4.0,
    )

    forecaster.fit(_returns(n=150))

    assert forecaster.backend == "ewma"
    summary = forecaster.get_model_summary()
    assert summary["fallback_reason"] == "exploding_variance_ratio"
    assert summary["volatility_ratio_to_realized"] > 4.0


def test_ewma_forecast_reports_guardrail_metadata() -> None:
    forecaster = object.__new__(GARCHForecaster)
    forecaster.backend = "ewma"
    forecaster.fitted_model = True
    forecaster.p = 1
    forecaster.q = 1
    forecaster.vol = "GARCH"
    forecaster.dist = "normal"
    forecaster._residual_diagnostics = {}
    forecaster._fallback_state = {
        "last_variance": 0.0004,
        "mean": 0.001,
        "fallback_reason": "near_igarch",
        "persistence": 0.995,
        "volatility_ratio_to_realized": 4.5,
    }

    result = forecaster.forecast(steps=3)

    assert result["fallback_reason"] == "near_igarch"
    assert result["persistence"] == pytest.approx(0.995)
    assert result["volatility_ratio_to_realized"] == pytest.approx(4.5)
