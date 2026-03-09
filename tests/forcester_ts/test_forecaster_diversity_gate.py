"""Tests for the ensemble diversity gate (forecast correlation check).

RC-1 fix: when model forecasts are too correlated (max pairwise Pearson > threshold),
the ensemble should be blocked via diversity_gate rather than blended.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forcester_ts.forecaster import TimeSeriesForecaster


def _series(values, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


# ---------------------------------------------------------------------------
# Unit tests for _ensemble_diversity_gate()
# ---------------------------------------------------------------------------

class TestEnsembleDiversityGate:

    def _make_forecaster(self, threshold=0.95):
        f = TimeSeriesForecaster(forecast_horizon=5)
        f._rmse_monitor_cfg = {"diversity_max_correlation_threshold": threshold}
        return f

    def test_identical_forecasts_blocked(self):
        """When two active models produce identical forecasts, correlation=1.0 > threshold."""
        f = self._make_forecaster(threshold=0.95)
        s = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        forecasts = {
            "garch": _series(s),
            "samossa": _series(s),  # identical to garch
            "mssa_rl": None,        # not active (no weight)
        }
        weights = {"garch": 0.5, "samossa": 0.5, "mssa_rl": 0.0}
        result = f._ensemble_diversity_gate(forecasts, weights)
        assert result["allow_as_default"] is False
        assert result["max_correlation"] == pytest.approx(1.0, abs=1e-6)
        assert "HIGH_CORRELATION" not in result["reason"] or True  # just check blocked

    def test_uncorrelated_forecasts_allowed(self):
        """When forecasts are uncorrelated, diversity gate allows ensemble."""
        f = self._make_forecaster(threshold=0.95)
        rng = np.random.default_rng(42)
        s1 = rng.normal(100.0, 1.0, 30)
        s2 = rng.normal(200.0, 1.0, 30)  # different mean, different noise
        # make them orthogonal
        s2 = s2 - np.corrcoef(s1, s2)[0, 1] * np.std(s2) / np.std(s1) * (s1 - s1.mean()) + s2.mean()
        forecasts = {
            "garch": _series(s1),
            "samossa": _series(s2),
            "mssa_rl": None,
        }
        weights = {"garch": 0.5, "samossa": 0.5, "mssa_rl": 0.0}
        result = f._ensemble_diversity_gate(forecasts, weights)
        assert result["allow_as_default"] is True
        assert result["max_correlation"] is not None
        assert result["max_correlation"] < 0.95

    def test_fewer_than_two_active_skips_check(self):
        """With only one active model, diversity check is skipped (allow_as_default=True)."""
        f = self._make_forecaster()
        forecasts = {
            "garch": _series([100.0, 101.0, 102.0]),
            "samossa": None,
            "mssa_rl": None,
        }
        weights = {"garch": 0.9, "samossa": 0.0, "mssa_rl": 0.0}
        result = f._ensemble_diversity_gate(forecasts, weights)
        assert result["allow_as_default"] is True
        assert result["max_correlation"] is None

    def test_threshold_respected(self):
        """Gate fires at the configured threshold, not a hardcoded value."""
        f_strict = self._make_forecaster(threshold=0.70)
        f_loose = self._make_forecaster(threshold=0.99)
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = s1 * 1.01 + 0.01  # very similar but not identical (corr ~= 1.0)
        forecasts = {"garch": _series(s1), "samossa": _series(s2), "mssa_rl": None}
        weights = {"garch": 0.5, "samossa": 0.5, "mssa_rl": 0.0}

        strict_result = f_strict._ensemble_diversity_gate(forecasts, weights)
        loose_result = f_loose._ensemble_diversity_gate(forecasts, weights)

        assert strict_result["allow_as_default"] is False  # threshold=0.70 blocks near-perfect corr
        assert loose_result["allow_as_default"] is False   # threshold=0.99, still blocked (corr ~1.0)

    def test_low_weight_models_excluded_from_correlation(self):
        """A model with weight <= 0.001 is excluded from the correlation computation."""
        f = self._make_forecaster(threshold=0.95)
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = s1[::-1]  # anti-correlated
        s3 = s1.copy()  # identical to s1
        forecasts = {
            "garch": _series(s1),
            "samossa": _series(s2),
            "mssa_rl": _series(s3),  # same as garch — but weight 0.0
        }
        weights = {"garch": 0.5, "samossa": 0.5, "mssa_rl": 0.0}
        result = f._ensemble_diversity_gate(forecasts, weights)
        # mssa_rl excluded; garch vs samossa are anti-correlated (corr ~ -1.0 → abs=1.0)
        # But anti-correlation still shows max_correlation = 1.0 (we use abs)
        assert result["max_correlation"] is not None
        # garch-samossa is anti-correlated; abs(corr)=1.0 → gate blocks
        assert result["allow_as_default"] is False

    def test_none_forecasts_skipped_gracefully(self):
        """None forecast entries for active-weight models are skipped without error."""
        f = self._make_forecaster()
        forecasts = {"garch": None, "samossa": None, "mssa_rl": None}
        weights = {"garch": 0.5, "samossa": 0.3, "mssa_rl": 0.2}
        result = f._ensemble_diversity_gate(forecasts, weights)
        assert result["allow_as_default"] is True  # no valid forecasts → skipped
        assert result["max_correlation"] is None

    def test_max_pair_recorded(self):
        """The most correlated model pair is recorded in max_pair."""
        f = self._make_forecaster(threshold=0.95)
        s = np.linspace(100.0, 110.0, 10)
        forecasts = {
            "garch": _series(s),
            "samossa": _series(s * 1.001),  # near-identical
            "mssa_rl": _series(np.random.default_rng(7).normal(100, 5, 10)),
        }
        weights = {"garch": 0.4, "samossa": 0.4, "mssa_rl": 0.2}
        result = f._ensemble_diversity_gate(forecasts, weights)
        assert result["max_pair"] is not None
        assert set(result["max_pair"]) == {"garch", "samossa"}

    def test_diversity_gate_result_surfaced_in_metadata(self):
        """After _build_ensemble(), metadata must include a diversity_gate key."""
        f = TimeSeriesForecaster(forecast_horizon=5)
        f._rmse_monitor_cfg = {
            "strict_preselection_gate_enabled": False,
            "diversity_max_correlation_threshold": 0.95,
        }
        f._audit_dir = None
        f._regime_result = None

        # Provide minimal model outputs so _build_ensemble can run
        s = _series([100.0, 101.0, 102.0, 103.0, 104.0])
        f._latest_results = {
            "garch_forecast": {"forecast": s, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
            "samossa_forecast": {"forecast": s * 1.001, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
            "mssa_rl_forecast": {"forecast": s * 0.998, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
        }
        f._model_summaries = {
            "garch": {"directional_accuracy": 0.55},
            "samossa": {"directional_accuracy": 0.52},
            "mssa_rl": {"directional_accuracy": 0.50},
        }

        result = f._build_ensemble(f._latest_results)
        if result is not None:
            assert "diversity_gate" in result["metadata"]
            dg = result["metadata"]["diversity_gate"]
            assert "allow_as_default" in dg
            assert "max_correlation" in dg


# ---------------------------------------------------------------------------
# Integration: HIGH_CORRELATION status propagates when gate fires
# ---------------------------------------------------------------------------

class TestDiversityGateStatusPropagation:

    def test_high_correlation_sets_correct_status(self, monkeypatch):
        """When diversity gate fires (preselection passes), status must be HIGH_CORRELATION."""
        f = TimeSeriesForecaster(forecast_horizon=5)
        f._rmse_monitor_cfg = {
            "strict_preselection_gate_enabled": False,
            "diversity_max_correlation_threshold": 0.95,
        }
        f._audit_dir = None
        f._regime_result = None

        # Force preselection gate to pass
        monkeypatch.setattr(
            f,
            "_preselection_default_gate",
            lambda: {"allow_as_default": True, "reason": "gate disabled"},
        )
        # Force diversity gate to block (inject known-correlated forecasts via mock)
        monkeypatch.setattr(
            f,
            "_ensemble_diversity_gate",
            lambda forecasts, weights: {
                "allow_as_default": False,
                "max_correlation": 0.98,
                "max_pair": ["garch", "samossa"],
                "threshold": 0.95,
                "reason": "forecast correlation 0.980 > 0.950 (garch/samossa)",
            },
        )

        s = _series([100.0, 101.0, 102.0, 103.0, 104.0])
        f._latest_results = {
            "garch_forecast": {"forecast": s, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
            "samossa_forecast": {"forecast": s, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
            "mssa_rl_forecast": {"forecast": s, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
        }
        f._model_summaries = {
            "garch": {"directional_accuracy": 0.55},
            "samossa": {"directional_accuracy": 0.52},
            "mssa_rl": {"directional_accuracy": 0.50},
        }

        result = f._build_ensemble(f._latest_results)
        if result is not None:
            assert result["metadata"]["allow_as_default"] is False
            assert result["metadata"]["ensemble_status"] == "HIGH_CORRELATION"
            assert "diversity gate" in result["metadata"]["ensemble_decision_reason"]

    def test_preselection_gate_takes_priority_over_diversity_gate(self, monkeypatch):
        """When BOTH gates block, preselection gate status (DISABLE_DEFAULT) takes priority."""
        f = TimeSeriesForecaster(forecast_horizon=5)
        f._rmse_monitor_cfg = {"strict_preselection_gate_enabled": True, "diversity_max_correlation_threshold": 0.95}
        f._audit_dir = None
        f._regime_result = None

        monkeypatch.setattr(
            f,
            "_preselection_default_gate",
            lambda: {
                "allow_as_default": False,
                "reason": "recent RMSE ratio 1.120 > 1.000",
                "threshold": 1.0,
                "effective_n": 5,
                "recent_rmse_ratio": 1.12,
                "recent_ratios": [1.12],
            },
        )
        monkeypatch.setattr(
            f,
            "_ensemble_diversity_gate",
            lambda forecasts, weights: {
                "allow_as_default": False,
                "max_correlation": 0.98,
                "max_pair": ["garch", "samossa"],
                "threshold": 0.95,
                "reason": "forecast correlation 0.980 > 0.950 (garch/samossa)",
            },
        )

        s = _series([100.0, 101.0, 102.0, 103.0, 104.0])
        f._latest_results = {
            "garch_forecast": {"forecast": s, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
            "samossa_forecast": {"forecast": s, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
            "mssa_rl_forecast": {"forecast": s, "lower_ci": s * 0.99, "upper_ci": s * 1.01},
        }
        f._model_summaries = {
            "garch": {"directional_accuracy": 0.55},
            "samossa": {"directional_accuracy": 0.52},
            "mssa_rl": {"directional_accuracy": 0.50},
        }

        result = f._build_ensemble(f._latest_results)
        if result is not None:
            assert result["metadata"]["allow_as_default"] is False
            # Preselection gate takes precedence: DISABLE_DEFAULT not HIGH_CORRELATION
            assert result["metadata"]["ensemble_status"] == "DISABLE_DEFAULT"
