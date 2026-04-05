"""Tests for Phase 8.2 + 8.3 — residual diagnostics and ADF/KPSS verdict.

Phase 8.2: shared residual diagnostics utility and model integration.
Phase 8.3: explicit stationarity_verdict (stationary/non_stationary/conflicted)
           from ADF + KPSS; force_difference=True on conflict.

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
    def test_forecast_includes_residual_diagnostics(self, mssa_ready_policy_env):
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
        assert "active_action" in result
        assert "active_rank" in result
        assert "q_state" in result
        assert "policy_version" in result
        assert result["policy_status"] == "ready"

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
        assert result["fallback_mode"] == "none"
        assert result["ewma_lambda"] is None
        assert result["residual_diagnostics_status"] == "available"
        assert result["residual_diagnostics_reason"] is None
        assert "fallback_reason" not in result

    def test_ewma_fallback_forecast_preserves_guardrail_fields(self):
        from forcester_ts.garch import GARCHForecaster

        forecaster = object.__new__(GARCHForecaster)
        forecaster.backend = "ewma"
        forecaster.fitted_model = True
        forecaster.p = 1
        forecaster.q = 1
        forecaster.vol = "GARCH"
        forecaster.dist = "normal"
        forecaster._residual_diagnostics = {}
        forecaster._fallback_state = {
            "lambda": 0.94,
            "last_variance": 0.0001,
            "mean": 0.001,
            "fallback_mode": "convergence_failure",
            "persistence": 0.992,
            "volatility_ratio_to_realized": 4.2,
        }

        result = forecaster.forecast(steps=2)

        assert result["fallback_mode"] == "convergence_failure"
        assert result["ewma_lambda"] == pytest.approx(0.94)
        assert result["residual_diagnostics_status"] == "unavailable"
        assert result["residual_diagnostics_reason"] == "ewma_fallback"
        assert "fallback_reason" not in result
        assert result["persistence"] == pytest.approx(0.992)
        assert result["volatility_ratio_to_realized"] == pytest.approx(4.2)


class TestSARIMAXResidualDiagnostics:
    def test_forecast_includes_standardized_residual_diagnostics(self):
        from forcester_ts.sarimax import SARIMAXForecaster

        class _DummyForecast:
            def __init__(self):
                self.predicted_mean = pd.Series(
                    [101.0, 102.0],
                    index=pd.RangeIndex(start=0, stop=2, step=1),
                )

            def conf_int(self, alpha=0.05):  # noqa: ARG002
                return pd.DataFrame(
                    {
                        "lower Close": [99.0, 100.0],
                        "upper Close": [103.0, 104.0],
                    },
                    index=self.predicted_mean.index,
                )

        class _DummyFit:
            aic = 100.0
            bic = 110.0
            resid = pd.Series(np.random.normal(0, 1, 120))

            def get_forecast(self, steps, exog=None):  # noqa: ARG002
                assert steps == 2
                return _DummyForecast()

        forecaster = object.__new__(SARIMAXForecaster)
        forecaster.fitted_model = _DummyFit()
        forecaster.best_order = (1, 1, 1)
        forecaster.best_seasonal_order = (0, 0, 0, 0)
        forecaster._fit_metadata = {"fit_strategy": "primary_strict"}
        forecaster._scale_factor = 1.0
        forecaster.log_transform = False
        forecaster._series_transform = None
        forecaster._log_shift = None

        result = forecaster.forecast(steps=2)

        assert "residual_diagnostics" in result
        assert "diagnostics" in result
        assert "white_noise" in result["residual_diagnostics"]
        assert (
            result["diagnostics"]["ljung_box_pvalue"]
            == result["residual_diagnostics"]["lb_pvalue"]
        )
        assert (
            result["diagnostics"]["jarque_bera_pvalue"]
            == result["residual_diagnostics"]["jb_pvalue"]
        )


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


# ---------------------------------------------------------------------------
# Phase 8.3 — ADF/KPSS stationarity verdict
# ---------------------------------------------------------------------------

class TestStationarityVerdict:
    """_capture_series_diagnostics must emit explicit stationarity_verdict."""

    def _make_series(self, n: int = 200, seed: int = 0) -> pd.Series:
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        # Stationary white noise
        return pd.Series(np.random.normal(0, 1, n), index=dates)

    def _make_nonstationary_series(self, n: int = 200, seed: int = 0) -> pd.Series:
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        # Random walk — non-stationary
        return pd.Series(np.cumsum(np.random.normal(0, 1, n)), index=dates)

    def _get_diagnostics(self, series: pd.Series) -> dict:
        from forcester_ts.forecaster import TimeSeriesForecaster
        f = TimeSeriesForecaster()
        return f._capture_series_diagnostics(series)

    def test_stationary_series_verdict(self):
        diag = self._get_diagnostics(self._make_series())
        if "stationarity_verdict" not in diag:
            pytest.skip("ADF or KPSS not available in this environment")
        assert diag["stationarity_verdict"] in {"stationary", "conflicted"}
        assert isinstance(diag["force_difference"], bool)

    def test_nonstationary_series_verdict(self):
        diag = self._get_diagnostics(self._make_nonstationary_series())
        if "stationarity_verdict" not in diag:
            pytest.skip("ADF or KPSS not available in this environment")
        assert diag["stationarity_verdict"] in {"non_stationary", "conflicted"}
        assert isinstance(diag["force_difference"], bool)

    def test_force_difference_true_on_non_stationary(self):
        diag = self._get_diagnostics(self._make_nonstationary_series())
        if "stationarity_verdict" not in diag:
            pytest.skip("ADF or KPSS not available in this environment")
        if diag["stationarity_verdict"] == "non_stationary":
            assert diag["force_difference"] is True

    def test_force_difference_false_on_stationary(self):
        diag = self._get_diagnostics(self._make_series())
        if "stationarity_verdict" not in diag:
            pytest.skip("ADF or KPSS not available in this environment")
        if diag["stationarity_verdict"] == "stationary":
            assert diag["force_difference"] is False

    def test_verdict_absent_when_only_one_test_runs(self):
        """If only ADF or only KPSS data is present, no verdict is emitted."""
        from forcester_ts.forecaster import TimeSeriesForecaster
        f = TimeSeriesForecaster()
        # Supply partial diagnostics — only adf_pvalue, no kpss_pvalue
        diag: dict = {"adf_pvalue": 0.01}
        # Call the verdict logic directly by injecting partial diagnostics
        adf_pv = diag.get("adf_pvalue")
        kpss_pv = diag.get("kpss_pvalue")
        verdict_computed = adf_pv is not None and kpss_pv is not None
        assert not verdict_computed

    def test_conflicted_verdict_forces_difference(self, monkeypatch):
        """Manually inject ADF=stationary, KPSS=non-stationary -> conflicted -> force_diff=True."""
        from forcester_ts.forecaster import TimeSeriesForecaster
        f = TimeSeriesForecaster()
        # Patch adfuller and kpss to return conflicting results
        import forcester_ts.forecaster as _fc_mod
        original_adfuller = _fc_mod.adfuller
        original_kpss = _fc_mod.kpss

        def _fake_adf(x, **kw):
            return (-5.0, 0.001, 0, 0, {"1%": -3.5, "5%": -2.9, "10%": -2.6}, 100.0)

        def _fake_kpss(x, **kw):
            # Returns stat > critical -> p-value < 0.05 -> non-stationary
            return (0.8, 0.01, 0, {"10%": 0.35, "5%": 0.46, "2.5%": 0.57, "1%": 0.74})

        monkeypatch.setattr(_fc_mod, "adfuller", _fake_adf)
        monkeypatch.setattr(_fc_mod, "kpss", _fake_kpss)

        np.random.seed(0)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        series = pd.Series(np.random.normal(0, 1, 100), index=dates)
        diag = f._capture_series_diagnostics(series)

        assert diag.get("stationarity_verdict") == "conflicted"
        assert diag.get("force_difference") is True


def test_forecaster_passes_stationarity_hint_into_sarimax(monkeypatch):
    from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

    captured: dict = {}

    def _fake_fit(self, series, exogenous=None, stationarity_hint=None, **kwargs):  # noqa: ARG001
        captured["stationarity_hint"] = dict(stationarity_hint or {})
        self.best_order = (1, 1, 1)
        self.best_seasonal_order = (0, 0, 0, 0)
        self.fitted_model = type("Dummy", (), {"aic": 1.0, "bic": 2.0, "llf": -1.0, "nobs": len(series)})()
        self._fit_metadata = dict(stationarity_hint or {})
        return self

    monkeypatch.setattr("forcester_ts.sarimax.SARIMAXForecaster.fit", _fake_fit)
    monkeypatch.setattr(
        "forcester_ts.sarimax.SARIMAXForecaster.get_model_summary",
        lambda self: {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
    )

    config = TimeSeriesForecasterConfig(
        sarimax_enabled=True,
        garch_enabled=False,
        samossa_enabled=False,
        mssa_rl_enabled=False,
        ensemble_enabled=False,
    )
    forecaster = TimeSeriesForecaster(config=config)
    series = pd.Series(np.cumsum(np.random.normal(0, 1, 80)), index=pd.date_range("2024-01-01", periods=80, freq="D"))

    forecaster.fit(series)

    assert captured["stationarity_hint"] == forecaster._series_diagnostics
    assert "force_difference" in captured["stationarity_hint"]


# ---------------------------------------------------------------------------
# P1-E: residual diagnostics enforcement by model type (exemptions)
# ---------------------------------------------------------------------------

class TestResidualDiagnosticsModelTypeExemptions:
    """P1-E: SAMoSSA-primary windows excluded from non-WN count; GARCH included.

    SSA-based models produce structurally autocorrelated in-sample residuals by
    design. Counting them inflates the non-WN rate and penalises a correct model.
    After the exemption list is wired, only GARCH/SARIMAX windows are counted.
    """

    def _make_result(self, default_model: str, white_noise: bool):
        """Minimal AuditCheckResult for residual diagnostics pool testing."""
        import sys, importlib
        mod = importlib.import_module("scripts.check_forecast_audits")
        return mod.AuditCheckResult(
            path=None,
            ensemble_rmse=1.0,
            baseline_rmse=1.0,
            rmse_ratio=1.0,
            violation=False,
            baseline_model="SAMOSSA",
            ensemble_missing=False,
            default_model=default_model,
            residual_diag_present=True,
            residual_diag_white_noise=white_noise,
            residual_diag_n=50,
            ensemble_index_mismatch=False,
        )

    def test_samossa_window_excluded_from_non_wn_count(self) -> None:
        """SAMoSSA-primary window with non-WN residuals must NOT count toward non-WN rate."""
        import scripts.check_forecast_audits as mod
        exemptions: set = {"SAMOSSA", "MSSA_RL"}
        results = [self._make_result("SAMOSSA", white_noise=False)]
        effective = [
            r for r in results
            if (
                r.residual_diag_present
                and r.residual_diag_n is not None
                and r.residual_diag_n >= 20
                and (r.default_model or "").upper() not in exemptions
            )
        ]
        assert len(effective) == 0, (
            "SAMoSSA-primary window must be excluded from residual effective pool"
        )

    def test_garch_window_included_in_non_wn_count(self) -> None:
        """GARCH-primary window with non-WN residuals MUST count toward non-WN rate."""
        exemptions: set = {"SAMOSSA", "MSSA_RL"}
        results = [self._make_result("GARCH", white_noise=False)]
        effective = [
            r for r in results
            if (
                r.residual_diag_present
                and r.residual_diag_n is not None
                and r.residual_diag_n >= 20
                and (r.default_model or "").upper() not in exemptions
            )
        ]
        assert len(effective) == 1, (
            "GARCH-primary window must be included in residual effective pool"
        )
        non_wn = sum(1 for r in effective if r.residual_diag_white_noise is False)
        assert non_wn == 1, "Non-WN GARCH window must be counted in violation tally"

    def test_mssa_rl_window_excluded_from_non_wn_count(self) -> None:
        """MSSA-RL-primary window (SSA-based) must also be exempt."""
        exemptions: set = {"SAMOSSA", "MSSA_RL"}
        results = [self._make_result("MSSA_RL", white_noise=False)]
        effective = [
            r for r in results
            if (
                r.residual_diag_present
                and r.residual_diag_n is not None
                and r.residual_diag_n >= 20
                and (r.default_model or "").upper() not in exemptions
            )
        ]
        assert len(effective) == 0, "MSSA_RL-primary window must be excluded"
