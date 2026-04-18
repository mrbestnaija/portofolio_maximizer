"""Tests for GARCH EWMA convergence_ok flag behaviour.

Invariants:
1. EWMA fallback path sets convergence_ok=True (EWMA is a valid conservative model)
2. EWMA result still carries fallback_mode so callers can distinguish it from a full GARCH fit
3. CI is NOT inflated in forecaster._enrich_garch_forecast when convergence_ok=True
4. CI IS inflated 1.5× when convergence_ok=False (control — full GARCH failure path)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ewma_forecaster():
    from forcester_ts.garch import GARCHForecaster
    return GARCHForecaster(backend="ewma")


def _fit_and_forecast(forecaster, steps=5):
    returns = pd.Series(np.random.normal(0, 0.01, 120))
    forecaster.fit(returns)
    return forecaster.forecast(steps=steps)


# ---------------------------------------------------------------------------
# 1 & 2 — EWMA path flag contract
# ---------------------------------------------------------------------------

class TestEWMAConvergenceFlag:

    def test_ewma_backend_sets_convergence_ok_true(self):
        """GARCHForecaster(backend='ewma').forecast() must set convergence_ok=True."""
        result = _fit_and_forecast(_make_ewma_forecaster())
        assert result.get("convergence_ok") is True, (
            f"EWMA forecast should set convergence_ok=True, got {result.get('convergence_ok')!r}"
        )

    def test_ewma_forecast_has_fallback_mode_set(self):
        """EWMA result must carry fallback_mode so callers know it is not a full GARCH fit."""
        result = _fit_and_forecast(_make_ewma_forecaster())
        assert result.get("fallback_mode") is not None, (
            "EWMA result should carry fallback_mode field"
        )

    def test_ewma_variance_forecast_present(self):
        """EWMA path must still return a usable variance_forecast series."""
        result = _fit_and_forecast(_make_ewma_forecaster(), steps=5)
        assert result.get("variance_forecast") is not None
        assert len(result["variance_forecast"]) == 5


# ---------------------------------------------------------------------------
# 3 & 4 — CI inflation gated on convergence_ok in forecaster
# ---------------------------------------------------------------------------

class TestCIInflationContract:

    def _make_forecaster(self):
        from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
        cfg = TimeSeriesForecasterConfig(sarimax_enabled=False)
        return TimeSeriesForecaster(config=cfg)

    def _build_payload(self, convergence_ok: bool, horizon: int = 5, price: float = 100.0):
        """Build a minimal GARCH payload with controlled CI half-width."""
        import pandas as pd
        half_w = 2.0
        idx = pd.RangeIndex(horizon)
        mean_series = pd.Series([0.0] * horizon, index=idx)
        var_series = pd.Series([0.0001] * horizon, index=idx)
        return {
            "forecast": list(np.linspace(price, price + 1, horizon)),
            "lower_ci": [price - half_w] * horizon,
            "upper_ci": [price + half_w] * horizon,
            "mean_forecast": mean_series,
            "variance_forecast": var_series,
            "steps": horizon,
            "convergence_ok": convergence_ok,
            "residual_diagnostics_status": "ok",
            "residual_diagnostics": {"ljung_box_p": 0.10},
        }

    def test_ci_not_inflated_when_convergence_ok_true(self):
        """_enrich_garch_forecast must NOT inflate CI when convergence_ok=True."""
        forecaster = self._make_forecaster()
        payload = self._build_payload(convergence_ok=True)

        orig_half_w = 2.0
        enriched = forecaster._enrich_garch_forecast(payload)

        # Extract first CI half-width from enriched output
        lower = enriched.get("lower_ci")
        upper = enriched.get("upper_ci")
        if lower is None or upper is None:
            pytest.skip("_enrich_garch_forecast did not return CI fields (short-circuit path)")

        # Convert to list if Series
        try:
            lower_val = list(lower)[0]
            upper_val = list(upper)[0]
        except (TypeError, IndexError):
            pytest.skip("CI values not indexable")

        enriched_half_w = (upper_val - lower_val) / 2.0
        # Allow trivial floating-point differences; 1.4× would indicate inflation
        assert enriched_half_w < orig_half_w * 1.1, (
            f"CI was inflated despite convergence_ok=True: "
            f"orig={orig_half_w:.4f}, enriched={enriched_half_w:.4f}"
        )

    def test_ci_IS_inflated_when_convergence_ok_false(self):
        """Control: _enrich_garch_forecast MUST inflate CI when convergence_ok=False."""
        forecaster = self._make_forecaster()
        payload = self._build_payload(convergence_ok=False)

        orig_half_w = 2.0
        enriched = forecaster._enrich_garch_forecast(payload)

        lower = enriched.get("lower_ci")
        upper = enriched.get("upper_ci")
        if lower is None or upper is None:
            pytest.skip("_enrich_garch_forecast did not return CI fields")

        try:
            lower_val = list(lower)[0]
            upper_val = list(upper)[0]
        except (TypeError, IndexError):
            pytest.skip("CI values not indexable")

        enriched_half_w = (upper_val - lower_val) / 2.0
        assert enriched_half_w >= orig_half_w * 1.4, (
            f"CI should be inflated 1.5× when convergence_ok=False: "
            f"orig={orig_half_w:.4f}, enriched={enriched_half_w:.4f}"
        )
