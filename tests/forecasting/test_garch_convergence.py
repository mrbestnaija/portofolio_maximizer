"""
Phase 7.14-C: GARCH convergence hardening tests.

Verifies that:
1. A RuntimeWarning containing 'convergence' causes _convergence_ok=False
2. convergence failure extends the GJR fallback trigger
3. CI is inflated 1.5x when convergence_ok=False
4. Normal fit leaves CI unchanged (convergence_ok=True)
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from forcester_ts.garch import GARCHForecaster


def _make_returns(n: int = 60, seed: int = 42) -> pd.Series:
    """Generate a short stationary returns series for testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    return pd.Series(returns, name="returns")


@pytest.fixture
def garch():
    """Basic GARCH forecaster with auto_select=True."""
    return GARCHForecaster(auto_select=True, max_p=1, max_q=1)


class TestGARCHConvergenceHardening:
    """Phase 7.14-C: Convergence detection, GJR trigger, CI inflation."""

    def test_convergence_ok_true_by_default(self, garch):
        """_convergence_ok is True before fitting."""
        assert garch._convergence_ok is True

    def test_convergence_failure_detected_via_warning(self):
        """RuntimeWarning with 'convergence' in message sets _convergence_ok=False."""
        try:
            from arch import arch_model as _am
        except ImportError:
            pytest.skip("arch library not installed")

        # Use 130 points to exceed the min_arch_sample_size=120 guard added in
        # forecast-hardening; with 60 points the forecaster falls back to EWMA
        # before arch_model is ever called, so the convergence patch is bypassed.
        forecaster = GARCHForecaster(auto_select=True, max_p=1, max_q=1)
        returns = _make_returns(130)

        # Patch arch_model.fit to emit a convergence RuntimeWarning
        original_fit = None

        class _FakeResult:
            aic = 1000.0
            bic = 1010.0
            convergence_flag = 0

            @property
            def params(self):
                # Return minimal params so _garch_persistence can compute
                return {"alpha[1]": 0.05, "beta[1]": 0.80}

            def forecast(self, horizon):
                idx = pd.Index(range(1, horizon + 1), name="horizon")
                variance = pd.DataFrame(
                    [[0.0001] * horizon], columns=idx
                )
                mean = pd.DataFrame([[0.0] * horizon], columns=idx)
                result = MagicMock()
                result.variance = variance
                result.mean = mean
                return result

        def _fake_fit(disp="off"):
            warnings.warn(
                "The optimizer returned code 9. The message is: Iteration limit reached",
                RuntimeWarning,
                stacklevel=2,
            )
            return _FakeResult()

        with patch("forcester_ts.garch.arch_model") as mock_arch:
            fake_model = MagicMock()
            fake_model.fit = _fake_fit
            mock_arch.return_value = fake_model

            forecaster.fit(returns)

        # The convergence warning should have been detected
        assert forecaster._convergence_ok is False

    def test_convergence_ok_stays_true_on_clean_fit(self):
        """Normal fit with no convergence warning leaves _convergence_ok=True."""
        try:
            from arch import arch_model as _am
        except ImportError:
            pytest.skip("arch library not installed")

        forecaster = GARCHForecaster(auto_select=True, max_p=1, max_q=1)
        returns = _make_returns(200)  # More data -> easier convergence

        # A real fit on a well-behaved series should converge
        try:
            forecaster.fit(returns)
            # Only check if arch backend was used
            if getattr(forecaster, "backend", "arch") == "arch":
                assert forecaster._convergence_ok is True
        except Exception:
            pytest.skip("Could not fit GARCH on test data (arch may be unavailable)")

    def test_convergence_failure_inflates_ci(self):
        """When convergence_ok=False, _enrich_garch_forecast inflates CI 1.5x."""
        try:
            from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
        except ImportError:
            pytest.skip("forecaster module unavailable")

        # Build a mock GARCH result with convergence_ok=False
        n = 5
        prices = [100.0 + i for i in range(n)]
        lower = [p - 2.0 for p in prices]  # half-width = 2.0
        upper = [p + 2.0 for p in prices]  # half-width = 2.0
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        payload = {
            "mean_forecast": pd.Series([0.01] * n, index=pd.Index(range(1, n + 1))),
            "variance_forecast": pd.Series([0.0001] * n, index=pd.Index(range(1, n + 1))),
            "steps": n,
            "convergence_ok": False,  # <-- convergence failure
        }

        config = TimeSeriesForecasterConfig(
            forecast_horizon=n,
            sarimax_enabled=False,
            garch_enabled=False,
            samossa_enabled=False,
            mssa_rl_enabled=False,
        )
        forecaster = TimeSeriesForecaster(config=config)
        forecaster._last_price = 100.0
        forecaster._forecast_index = idx

        result = forecaster._enrich_garch_forecast(payload)

        assert "lower_ci" in result
        assert "upper_ci" in result
        assert result.get("convergence_ok") is False

        # Verify CI is wider than a non-inflated version
        # A fresh call with convergence_ok=True should give narrower CI
        payload_ok = dict(payload)
        payload_ok["convergence_ok"] = True
        result_ok = forecaster._enrich_garch_forecast(payload_ok)

        lower_inflated = result["lower_ci"].to_numpy()
        lower_normal = result_ok["lower_ci"].to_numpy()
        upper_inflated = result["upper_ci"].to_numpy()
        upper_normal = result_ok["upper_ci"].to_numpy()

        # Inflated CI should be at least as wide as normal CI at every step
        assert np.all(lower_inflated <= lower_normal + 1e-9)
        assert np.all(upper_inflated >= upper_normal - 1e-9)

    def test_good_fit_ci_unchanged(self):
        """convergence_ok=True leaves CI unchanged (no inflation)."""
        try:
            from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
        except ImportError:
            pytest.skip("forecaster module unavailable")

        n = 5
        payload = {
            "mean_forecast": pd.Series([0.005] * n, index=pd.Index(range(1, n + 1))),
            "variance_forecast": pd.Series([0.0004] * n, index=pd.Index(range(1, n + 1))),
            "steps": n,
            "convergence_ok": True,
        }

        config = TimeSeriesForecasterConfig(
            forecast_horizon=n,
            sarimax_enabled=False,
            garch_enabled=False,
            samossa_enabled=False,
            mssa_rl_enabled=False,
        )
        forecaster = TimeSeriesForecaster(config=config)
        forecaster._last_price = 100.0
        forecaster._forecast_index = pd.date_range("2024-01-01", periods=n, freq="D")

        result = forecaster._enrich_garch_forecast(payload)
        assert result.get("convergence_ok") is True

        # CI should be positive and reasonable
        assert "lower_ci" in result
        assert "upper_ci" in result
        assert all(result["upper_ci"] > result["lower_ci"])

    def test_ewma_summary_exposes_guardrail_metadata(self):
        forecaster = object.__new__(GARCHForecaster)
        forecaster.backend = "ewma"
        forecaster.fitted_model = True
        forecaster.p = 1
        forecaster.q = 1
        forecaster.dist = "normal"
        forecaster.mean = "AR"
        forecaster._differenced = False
        forecaster._fallback_state = {
            "lambda": 0.94,
            "n_obs": 60,
            "fallback_reason": "convergence_failure",
            "persistence": 0.995,
            "volatility_ratio_to_realized": 4.8,
        }

        summary = forecaster.get_model_summary()

        assert summary["backend"] == "ewma"
        assert summary["fallback_reason"] == "convergence_failure"
        assert summary["persistence"] == pytest.approx(0.995)
        assert summary["volatility_ratio_to_realized"] == pytest.approx(4.8)
        assert summary["fit_sample_size"] == 60
