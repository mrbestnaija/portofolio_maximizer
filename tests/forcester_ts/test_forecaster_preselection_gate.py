from __future__ import annotations

import pandas as pd

from forcester_ts.forecaster import TimeSeriesForecaster


class _DummySamossaModel:
    def forecast(self, steps: int):
        idx = pd.date_range("2025-01-01", periods=steps, freq="B")
        return {"forecast": pd.Series([100.0 + float(i) for i in range(steps)], index=idx)}

    def get_model_summary(self):
        return {"trend_strength": 0.5, "seasonal_strength": 0.2}


def test_preselection_gate_blocks_ensemble_as_default(monkeypatch):
    forecaster = TimeSeriesForecaster(forecast_horizon=3)
    forecaster._audit_dir = None
    forecaster._regime_result = None
    forecaster._samossa = _DummySamossaModel()

    def _blocked_gate():
        return {
            "enabled": True,
            "allow_as_default": False,
            "reason": "recent RMSE ratio 1.120 > 1.000",
            "threshold": 1.0,
            "effective_n": 8,
            "recent_window": 5,
            "recent_rmse_ratio": 1.12,
            "recent_ratios": [1.12, 1.08],
        }

    monkeypatch.setattr(forecaster, "_preselection_default_gate", _blocked_gate)

    out = forecaster.forecast(steps=3)
    assert out.get("default_model") == "SAMOSSA"
    assert out.get("effective_default_model") == "SAMOSSA"
    assert out.get("mean_forecast") == out.get("samossa_forecast")


def test_preselection_gate_decision_from_recent_ratios(monkeypatch):
    forecaster = TimeSeriesForecaster(forecast_horizon=3)
    forecaster._rmse_monitor_cfg = {
        "strict_preselection_gate_enabled": True,
        "strict_preselection_max_rmse_ratio": 1.0,
        "strict_preselection_recent_window": 3,
        "strict_preselection_min_effective_audits": 2,
    }

    monkeypatch.setattr(
        forecaster,
        "_audit_history_stats",
        lambda **kwargs: {"effective_n": 4, "ratios": [1.05, 1.01, 0.99]},
    )
    blocked = forecaster._preselection_default_gate()
    assert blocked["allow_as_default"] is False

    monkeypatch.setattr(
        forecaster,
        "_audit_history_stats",
        lambda **kwargs: {"effective_n": 4, "ratios": [0.99, 0.98, 0.97]},
    )
    allowed = forecaster._preselection_default_gate()
    assert allowed["allow_as_default"] is True


import pandas as pd


def _make_dummy_forecast_series(start=100.0, steps=3):
    idx = pd.date_range("2025-01-01", periods=steps, freq="B")
    return pd.Series([start + float(i) for i in range(steps)], index=idx)


def _make_results(available_models=("samossa", "mssa_rl", "garch")):
    """Build a minimal results dict with dummy forecast payloads."""
    results = {}
    for m in available_models:
        results[f"{m}_forecast"] = {"forecast": _make_dummy_forecast_series()}
    return results


class TestDisableDefaultFallbackSelection:
    """Phase 7.15-F: _select_disable_default_fallback uses OOS holdout + flat-trend guard."""

    def test_oos_holdout_tier1_selects_best_rmse_model(self):
        """Tier 1: prior evaluate() OOS metrics → _best_single_from_metrics wins."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        # Simulate OOS metrics from previous evaluate() call:
        # GARCH has best (lowest) RMSE
        forecaster._latest_metrics = {
            "samossa": {"rmse": 20.0},
            "mssa_rl": {"rmse": 12.0},
            "garch": {"rmse": 8.0},  # best
        }
        forecaster._model_summaries = {}
        results = _make_results()

        preferred = forecaster._select_disable_default_fallback(
            results, ensemble_meta={"primary_model": "SAMOSSA"}
        )
        assert preferred.upper() == "GARCH"

    def test_flat_trend_guard_tier2_skips_samossa(self):
        """Tier 2: SAMoSSA trend_strength below threshold → prefer GARCH."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        forecaster._latest_metrics = {}  # no prior OOS data
        forecaster._model_summaries = {
            "samossa": {"trend_strength": 0.002}  # flat: well below 0.10 threshold
        }
        results = _make_results()

        preferred = forecaster._select_disable_default_fallback(
            results, ensemble_meta={"primary_model": "SAMOSSA"}
        )
        assert preferred.upper() == "GARCH"

    def test_healthy_samossa_uses_tier3_original_behaviour(self):
        """Tier 3: SAMoSSA has non-flat trend → use ensemble_meta primary_model."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        forecaster._latest_metrics = {}
        forecaster._model_summaries = {
            "samossa": {"trend_strength": 0.30}  # healthy trend
        }
        results = _make_results()

        preferred = forecaster._select_disable_default_fallback(
            results, ensemble_meta={"primary_model": "SAMOSSA"}
        )
        assert preferred.upper() == "SAMOSSA"

    def test_tier1_overrides_flat_trend(self):
        """OOS holdout (Tier 1) supersedes flat-trend guard even if SAMoSSA is flat."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        forecaster._latest_metrics = {
            "garch": {"rmse": 8.0},  # best OOS
            "samossa": {"rmse": 25.0},
        }
        forecaster._model_summaries = {
            "samossa": {"trend_strength": 0.001}  # flat
        }
        results = _make_results(available_models=("samossa", "garch"))

        preferred = forecaster._select_disable_default_fallback(
            results, ensemble_meta={"primary_model": "SAMOSSA"}
        )
        # Tier 1 wins: GARCH has best OOS RMSE
        assert preferred.upper() == "GARCH"

    def test_near_flat_trend_msft_range_skips_samossa(self):
        """Tier 2 fires for trend_strength in the 0.069-0.092 MSFT range.

        Live observation (2026-03-28): MSFT SAMoSSA trend_strength=0.069-0.092
        produced expected_return=0.0 despite being above the old 0.05 threshold.
        Threshold raised to 0.10 to catch this near-flat regime.
        """
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        forecaster._latest_metrics = {}
        for ts in (0.069, 0.080, 0.092, 0.099):
            forecaster._model_summaries = {"samossa": {"trend_strength": ts}}
            preferred = forecaster._select_disable_default_fallback(
                _make_results(), ensemble_meta={"primary_model": "SAMOSSA"}
            )
            assert preferred.upper() == "GARCH", (
                f"trend_strength={ts} should route to GARCH (below 0.10 threshold)"
            )

    def test_above_new_threshold_stays_with_samossa(self):
        """trend_strength >= 0.10 keeps SAMoSSA as Tier 3 default (not flat)."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        forecaster._latest_metrics = {}
        for ts in (0.10, 0.15, 0.30):
            forecaster._model_summaries = {"samossa": {"trend_strength": ts}}
            preferred = forecaster._select_disable_default_fallback(
                _make_results(), ensemble_meta={"primary_model": "SAMOSSA"}
            )
            assert preferred.upper() == "SAMOSSA", (
                f"trend_strength={ts} should keep SAMOSSA (at or above 0.10 threshold)"
            )

    def test_flat_trend_falls_through_to_garch_when_mssa_rl_absent(self):
        """Tier 2: if MSSA_RL absent, next preference is GARCH."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        forecaster._latest_metrics = {}
        forecaster._model_summaries = {
            "samossa": {"trend_strength": 0.002}
        }
        results = _make_results(available_models=("samossa", "garch"))  # no mssa_rl

        preferred = forecaster._select_disable_default_fallback(
            results, ensemble_meta={"primary_model": "SAMOSSA"}
        )
        assert preferred.upper() == "GARCH"

    def test_flat_trend_prefers_sarimax_before_mssa_rl_when_garch_absent(self):
        """Tier 2 keeps MSSA_RL containment-only when SARIMAX is available."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        forecaster._latest_metrics = {}
        forecaster._model_summaries = {
            "samossa": {"trend_strength": 0.002}
        }
        results = _make_results(available_models=("samossa", "sarimax", "mssa_rl"))

        preferred = forecaster._select_disable_default_fallback(
            results, ensemble_meta={"primary_model": "SAMOSSA"}
        )
        assert preferred.upper() == "SARIMAX"

    def test_default_single_selection_deprioritises_mssa_rl_preference(self):
        """Containment: MSSA_RL may remain available, but not as the preferred fallback."""
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        results = _make_results(available_models=("garch", "mssa_rl"))

        default_model, payload = forecaster._select_default_single_forecast(
            results,
            preferred_model="MSSA_RL",
        )

        assert default_model == "GARCH"
        assert payload == results["garch_forecast"]

    def test_default_single_selection_uses_mssa_rl_only_as_last_resort(self):
        forecaster = TimeSeriesForecaster(forecast_horizon=3)
        results = _make_results(available_models=("mssa_rl",))

        default_model, payload = forecaster._select_default_single_forecast(
            results,
            preferred_model="MSSA_RL",
        )

        assert default_model == "MSSA_RL"
        assert payload == results["mssa_rl_forecast"]
