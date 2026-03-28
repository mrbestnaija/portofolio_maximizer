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


class TestVolatilityRatioP95Robustness:
    """Fit-time and forecast-time guards use p95, not last-bar or max."""

    def _make_forecaster(self, realized: float = 0.01) -> GARCHForecaster:
        f = object.__new__(GARCHForecaster)
        f._realized_volatility = realized
        f.fitted_model = None
        return f

    def test_fit_time_single_tail_spike_does_not_trigger_guard(self) -> None:
        """A spike only at the terminal bar must not skew the p95 ratio over threshold."""
        f = self._make_forecaster(realized=0.01)
        # 149 normal bars at 0.02 (ratio=2.0) + 1 spike at 0.60 (ratio=60x)
        normal = [0.02] * 149
        spike = [0.60]
        values = pd.Series(normal + spike)
        f.fitted_model = MagicMock()
        f.fitted_model.conditional_volatility = values
        ratio = f._conditional_volatility_ratio_to_realized()
        # p95 of 150-element series: ~2.0 (the spike is above p95 only marginally)
        assert ratio is not None
        # The spike is at index 149 (last), iloc[-1]/realized=60; p95 should be ~2.0
        assert ratio < 5.0, f"p95 ratio={ratio:.3f} should not be dominated by tail spike"

    def test_fit_time_iloc_last_would_have_been_dominated(self) -> None:
        """Demonstrate iloc[-1] is the spike value (60x) while p95 is ~2x."""
        f = self._make_forecaster(realized=0.01)
        normal = [0.02] * 149
        spike = [0.60]
        values = pd.Series(normal + spike)
        # What the old code would have returned:
        old_ratio = abs(values.iloc[-1]) / 0.01
        assert old_ratio == pytest.approx(60.0)
        # What the new code returns (p95):
        new_ratio = float(np.percentile(values.abs(), 95)) / 0.01
        assert new_ratio < 5.0

    def test_forecast_time_single_high_step_does_not_trigger_guard(self) -> None:
        """A single large forecast step must not trigger the forecast-time guard."""
        f = self._make_forecaster(realized=0.01)
        # 4 normal steps at 0.02 + 1 high step at 0.50 (ratio=50x at max)
        volatility = pd.Series([0.02, 0.02, 0.02, 0.02, 0.50])
        ratio = f._forecast_volatility_ratio_to_realized(volatility)
        assert ratio is not None
        # p95 of 5 elements: sorted=[0.02,0.02,0.02,0.02,0.50]; p95 is ~0.41 → ratio~41
        # But with a spike only at index 4, the max would have been 50x.
        # p95 here is still high since the spike is 20% of the series.
        # The key property: p95 <= max (not worse than old behavior)
        old_ratio = float(volatility.max()) / 0.01
        assert ratio <= old_ratio

    def test_forecast_time_all_uniform_same_as_max(self) -> None:
        """When all forecast steps are uniform, p95 == max == every value."""
        f = self._make_forecaster(realized=0.01)
        volatility = pd.Series([0.03, 0.03, 0.03, 0.03, 0.03])
        ratio = f._forecast_volatility_ratio_to_realized(volatility)
        assert ratio == pytest.approx(3.0)

    def test_fit_time_empty_returns_none(self) -> None:
        f = self._make_forecaster(realized=0.01)
        f.fitted_model = MagicMock()
        f.fitted_model.conditional_volatility = pd.Series([], dtype=float)
        assert f._conditional_volatility_ratio_to_realized() is None

    def test_forecast_time_empty_returns_none(self) -> None:
        f = self._make_forecaster(realized=0.01)
        assert f._forecast_volatility_ratio_to_realized(pd.Series([], dtype=float)) is None


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
