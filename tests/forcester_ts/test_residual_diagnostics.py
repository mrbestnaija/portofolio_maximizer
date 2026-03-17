"""Tests for Phase 8.2 — shared residual diagnostics utility and model integration.

Covers:
  - run_residual_diagnostics() utility (white_noise True/False, missing data, short series)
  - GARCH forecast includes residual_diagnostics key
  - SAMOSSA forecast includes residual_diagnostics key
  - MSSA-RL forecast includes residual_diagnostics key
  - CI inflation triggered when white_noise=False in _enrich_garch_forecast
"""
from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from forcester_ts.residual_diagnostics import run_residual_diagnostics


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestRunResidualDiagnostics:
    def test_white_noise_series_passes(self):
        np.random.seed(0)
        residuals = np.random.normal(0, 1, 200)
        result = run_residual_diagnostics(residuals)
        assert "lb_pvalue" in result
        assert "jb_pvalue" in result
        assert "white_noise" in result
        assert "n" in result
        assert result["n"] == 200

    def test_autocorrelated_series_fails_white_noise(self):
        """Strongly AR(1) residuals should be flagged as not white noise."""
        np.random.seed(42)
        n = 200
        resid = np.zeros(n)
        resid[0] = np.random.normal()
        for i in range(1, n):
            resid[i] = 0.95 * resid[i - 1] + np.random.normal(0, 0.1)
        result = run_residual_diagnostics(resid)
        assert result["white_noise"] is False
        assert result["lb_pvalue"] is not None
        assert result["lb_pvalue"] < 0.05

    def test_too_short_series_returns_no_test_results(self):
        result = run_residual_diagnostics(np.array([1.0, 2.0, 3.0]))
        assert result["lb_pvalue"] is None
        assert result["jb_pvalue"] is None
        assert result["white_noise"] is False
        assert result["n"] == 3

    def test_accepts_pandas_series(self):
        np.random.seed(1)
        s = pd.Series(np.random.normal(0, 1, 150))
        result = run_residual_diagnostics(s)
        assert result["n"] == 150
        assert isinstance(result["white_noise"], bool)

    def test_handles_nan_in_residuals(self):
        np.random.seed(2)
        arr = np.random.normal(0, 1, 100).tolist()
        arr[10] = float("nan")
        arr[50] = float("nan")
        result = run_residual_diagnostics(arr)
        assert result["n"] == 98  # two NaNs dropped

    def test_returns_dict_on_empty_input(self):
        result = run_residual_diagnostics(np.array([]))
        assert result["n"] == 0
        assert result["white_noise"] is False


# ---------------------------------------------------------------------------
# Model integration — residual_diagnostics key present in forecast output
# ---------------------------------------------------------------------------

class TestMSSARLResidualDiagnostics:
    def test_forecast_includes_residual_diagnostics(self):
        from forcester_ts.mssa_rl import MSSARLForecaster
        np.random.seed(7)
        n = 150
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        series = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)), index=dates)
        f = MSSARLForecaster()
        f.fit(series)
        result = f.forecast(steps=5)
        assert "residual_diagnostics" in result
        rd = result["residual_diagnostics"]
        assert isinstance(rd, dict)
        assert "white_noise" in rd
        assert "lb_pvalue" in rd
        assert "n" in rd

    def test_fit_populates_residual_diagnostics_attr(self):
        from forcester_ts.mssa_rl import MSSARLForecaster
        np.random.seed(8)
        n = 150
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        series = pd.Series(100 + np.cumsum(np.random.normal(0, 0.5, n)), index=dates)
        f = MSSARLForecaster()
        f.fit(series)
        assert hasattr(f, "_residual_diagnostics")
        assert isinstance(f._residual_diagnostics, dict)


class TestSAMOSSAResidualDiagnostics:
    def test_forecast_includes_residual_diagnostics(self):
        from forcester_ts.samossa import SAMOSSAForecaster as SAMoSSAForecaster
        np.random.seed(3)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        series = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)), index=dates)
        f = SAMoSSAForecaster()
        f.fit(series)
        result = f.forecast(steps=10)
        assert "residual_diagnostics" in result
        rd = result["residual_diagnostics"]
        assert isinstance(rd, dict)
        assert "white_noise" in rd
        assert rd["n"] > 0


class TestGARCHResidualDiagnostics:
    def test_forecast_includes_residual_diagnostics_key_via_object_new(self):
        """forecast() must tolerate instances created via object.__new__ (no __init__)."""
        from forcester_ts.garch import GARCHForecaster

        class _DummyFit:
            aic = 100.0
            bic = 110.0
            params = {}

            def forecast(self, horizon):
                idx = pd.Index(range(horizon), name="horizon")
                return MagicMock(
                    variance=pd.DataFrame([[1e-4] * horizon], index=[0], columns=range(horizon)),
                    mean=pd.DataFrame([[0.001] * horizon], index=[0], columns=range(horizon)),
                )

        forecaster = object.__new__(GARCHForecaster)
        forecaster.p = 1
        forecaster.q = 1
        forecaster.vol = "GARCH"
        forecaster.dist = "normal"
        forecaster._scale_factor = 1.0
        forecaster.fitted_model = _DummyFit()
        forecaster._convergence_ok = True

        result = forecaster.forecast(steps=3)
        # Must not raise; residual_diagnostics should be present (empty dict is fine)
        assert "residual_diagnostics" in result
        assert isinstance(result["residual_diagnostics"], dict)


# ---------------------------------------------------------------------------
# CI inflation in _enrich_garch_forecast (Phase 8.2)
# ---------------------------------------------------------------------------

class TestGARCHCIInflationOnNonWhiteNoise:
    def _make_forecaster(self):
        from forcester_ts.forecaster import TimeSeriesForecaster
        f = TimeSeriesForecaster()
        f._last_price = 100.0
        return f

    def _base_payload(self, white_noise: bool) -> dict:
        idx = pd.Index(range(1, 4), name="horizon")
        variance = pd.Series([0.0004, 0.0004, 0.0004], index=idx)
        mean = pd.Series([0.001, 0.001, 0.001], index=idx)
        return {
            "variance_forecast": variance,
            "mean_forecast": mean,
            "volatility": pd.Series([0.02, 0.02, 0.02], index=idx),
            "steps": 3,
            "p": 1, "q": 1,
            "vol": "GARCH", "dist": "normal",
            "aic": 100.0, "bic": 110.0,
            "convergence_ok": True,
            "residual_diagnostics": {
                "lb_pvalue": 0.001 if not white_noise else 0.50,
                "jb_pvalue": 0.001 if not white_noise else 0.50,
                "white_noise": white_noise,
                "n": 150,
            },
        }

    def test_ci_wider_when_residuals_not_white_noise(self):
        f = self._make_forecaster()
        clean_result = f._enrich_garch_forecast(self._base_payload(white_noise=True))
        noisy_result = f._enrich_garch_forecast(self._base_payload(white_noise=False))

        # CI half-width should be wider for noisy residuals
        clean_hw = (clean_result["upper_ci"] - clean_result["lower_ci"]).mean()
        noisy_hw = (noisy_result["upper_ci"] - noisy_result["lower_ci"]).mean()
        assert noisy_hw > clean_hw
        assert noisy_result.get("residual_diagnostics_ci_inflated") is True

    def test_ci_unchanged_when_white_noise_true(self):
        f = self._make_forecaster()
        result = f._enrich_garch_forecast(self._base_payload(white_noise=True))
        assert result.get("residual_diagnostics_ci_inflated") is not True

    def test_inflation_factor_is_approximately_1_2x(self):
        f = self._make_forecaster()
        clean = f._enrich_garch_forecast(self._base_payload(white_noise=True))
        noisy = f._enrich_garch_forecast(self._base_payload(white_noise=False))
        clean_hw = float((clean["upper_ci"] - clean["lower_ci"]).iloc[0])
        noisy_hw = float((noisy["upper_ci"] - noisy["lower_ci"]).iloc[0])
        # 1.2x inflation applied to half-width on each side -> 2 * 1.2 * hw_half
        assert pytest.approx(noisy_hw / clean_hw, rel=0.02) == 1.2
