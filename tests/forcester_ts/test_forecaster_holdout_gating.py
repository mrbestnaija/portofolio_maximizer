from __future__ import annotations

from pathlib import Path

import pandas as pd

from forcester_ts.forecaster import TimeSeriesForecaster


def test_forecaster_holdout_reweight_never_worse_than_best_single() -> None:
    # Construct a forecaster and inject synthetic forecast payloads so we can
    # validate the "best model is weightier" invariant without fitting models.
    forecaster = TimeSeriesForecaster(forecast_horizon=3)
    forecaster._rmse_monitor_cfg = {}
    forecaster._audit_dir = None  # Avoid writing audit files during tests.

    index = pd.date_range("2024-01-01", periods=3, freq="B")
    actual = pd.Series([10.0, 11.0, 12.0], index=index)

    # SARIMAX is the best predictor here.
    sarimax_pred = pd.Series([10.0, 11.0, 12.0], index=index)
    samossa_pred = pd.Series([10.0, 10.0, 10.0], index=index)
    mssa_pred = pd.Series([0.0, 0.0, 0.0], index=index)

    # Start with a deliberately bad ensemble (SAMOSSA-only).
    forecaster._latest_results = {
        "sarimax_forecast": {"forecast": sarimax_pred},
        "samossa_forecast": {"forecast": samossa_pred},
        "mssa_rl_forecast": {"forecast": mssa_pred},
        "ensemble_forecast": {"forecast": samossa_pred},
        "ensemble_metadata": {"weights": {"samossa": 1.0}, "confidence": {"samossa": 1.0}},
    }

    metrics = forecaster.evaluate(actual)
    assert "sarimax" in metrics
    assert "ensemble" in metrics

    best_rmse = min(
        metrics["sarimax"]["rmse"],
        metrics["samossa"]["rmse"],
        metrics["mssa_rl"]["rmse"],
    )
    assert metrics["ensemble"]["rmse"] <= best_rmse


def test_forecaster_waits_for_min_effective_audits(monkeypatch, tmp_path: Path) -> None:
    # Configure a high min_effective_audits so reweighting is skipped until enough
    # evidence exists, keeping the initial (worse) ensemble unchanged.
    tmp_cfg = tmp_path / "forecaster_monitoring.yml"
    tmp_cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    min_effective_audits: 5",
                "    max_rmse_ratio_vs_baseline: 5.0",
                "    promotion_margin: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TS_FORECAST_MONITOR_CONFIG", str(tmp_cfg))

    forecaster = TimeSeriesForecaster(forecast_horizon=3)
    forecaster._audit_dir = None

    index = pd.date_range("2024-01-01", periods=3, freq="B")
    actual = pd.Series([10.0, 11.0, 12.0], index=index)

    sarimax_pred = pd.Series([10.0, 11.0, 12.0], index=index)
    samossa_pred = pd.Series([10.0, 10.0, 10.0], index=index)

    # Start with an ensemble that mirrors the weaker SAMOSSA model.
    forecaster._latest_results = {
        "sarimax_forecast": {"forecast": sarimax_pred},
        "samossa_forecast": {"forecast": samossa_pred},
        "ensemble_forecast": {"forecast": samossa_pred},
        "ensemble_metadata": {"weights": {"samossa": 1.0}},
    }

    metrics = forecaster.evaluate(actual)
    assert metrics["ensemble"]["rmse"] > metrics["sarimax"]["rmse"]
