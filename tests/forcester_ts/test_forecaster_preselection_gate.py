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
    assert out.get("ensemble_forecast") is not None
    assert out.get("default_model") == "SAMOSSA"
    assert out.get("mean_forecast") == out.get("samossa_forecast")
    assert out.get("ensemble_metadata", {}).get("allow_as_default") is False
    assert out.get("ensemble_metadata", {}).get("ensemble_status") == "DISABLE_DEFAULT"


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
