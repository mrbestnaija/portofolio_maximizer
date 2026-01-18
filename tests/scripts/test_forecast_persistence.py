from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from etl.database_manager import DatabaseManager
from scripts import run_auto_trader


def test_extract_forecast_scalar_uses_horizon_end() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    payload = {
        "forecast": pd.Series([1.0, np.nan, 2.0], index=idx),
        "lower_ci": pd.Series([0.5, 0.75, np.nan], index=idx),
        "upper_ci": pd.Series([1.5, np.nan, 2.5], index=idx),
    }

    value, lower, upper = run_auto_trader._extract_forecast_scalar(payload)
    assert value == 2.0
    assert lower == 0.75
    assert upper == 2.5


def test_persist_forecast_snapshots_writes_rows() -> None:
    calls: list[tuple[str, str, dict]] = []

    class DummyDB:
        def save_forecast(self, ticker: str, forecast_date: str, forecast_data: dict) -> int:
            calls.append((ticker, forecast_date, forecast_data))
            return 1

    bar_ts = pd.Timestamp("2024-01-10")
    horizon_idx = pd.date_range("2024-01-11", periods=2, freq="D")
    forecast_bundle = {
        "horizon": 2,
        "ensemble_forecast": {
            "forecast": pd.Series([100.0, 101.5], index=horizon_idx),
            "lower_ci": pd.Series([99.0, 100.0], index=horizon_idx),
            "upper_ci": pd.Series([101.0, 103.0], index=horizon_idx),
        },
        "samossa_forecast": {"forecast": 102.0},
        "ensemble_metadata": {"weights": {"sarimax": 0.5}},
        "model_errors": {"sarimax": 0.1},
    }

    run_auto_trader._persist_forecast_snapshots(
        db_manager=DummyDB(),  # type: ignore[arg-type]
        ticker="AAPL",
        bar_ts=bar_ts,
        forecast_bundle=forecast_bundle,
    )

    assert len(calls) == 2
    combined = next(call for call in calls if call[2]["model_type"] == "COMBINED")
    assert combined[0] == "AAPL"
    assert combined[1] == "2024-01-10"
    assert combined[2]["forecast_horizon"] == 2
    assert combined[2]["forecast_value"] == 101.5
    assert combined[2]["lower_ci"] == 100.0
    assert combined[2]["upper_ci"] == 103.0

    samossa = next(call for call in calls if call[2]["model_type"] == "SAMOSSA")
    assert samossa[2]["forecast_value"] == 102.0


def test_backfill_forecast_regression_metrics_updates_row(tmp_path: Path) -> None:
    db_path = tmp_path / "forecast_backfill.db"
    manager = DatabaseManager(str(db_path))
    try:
        row_id = manager.save_forecast(
            "AAPL",
            "2024-01-10",
            {
                "model_type": "COMBINED",
                "forecast_horizon": 2,
                "forecast_value": 111.0,
                "model_order": {},
            },
        )
        assert row_id != -1

        idx = pd.date_range("2024-01-10", periods=5, freq="D")
        close_series = pd.Series([100.0, 101.0, 110.0, 111.0, 112.0], index=idx)

        updated = run_auto_trader._backfill_forecast_regression_metrics(
            db_manager=manager,
            ticker="AAPL",
            close_series=close_series,
            model_types=["COMBINED"],
            max_updates=10,
        )
        assert updated == 1

        cursor = manager.conn.cursor()
        cursor.execute("SELECT regression_metrics FROM time_series_forecasts WHERE id = ?", (row_id,))
        stored = cursor.fetchone()
        assert stored is not None
        metrics = json.loads(stored[0])
        assert metrics["rmse"] == 1.0
        assert metrics["directional_accuracy"] == 1.0
        assert metrics["n_observations"] == 1
        assert "evaluated_at" in metrics
    finally:
        manager.close()
