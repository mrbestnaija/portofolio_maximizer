from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import scripts.run_ensemble_diagnostics as mod


@dataclass
class _DummyConn:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_extract_forecast_data_uses_parameterized_queries(monkeypatch) -> None:
    ticker = "AAPL'; DROP TABLE time_series_forecasts;--"
    conn = _DummyConn()
    calls: list[tuple[str, object]] = []

    forecast_df = pd.DataFrame(
        {
            "model_name": ["sarimax", "ensemble"],
            "forecast_date": ["2026-01-01", "2026-01-01"],
            "forecast_value": [100.0, 101.0],
            "lower_ci": [95.0, 96.0],
            "upper_ci": [105.0, 106.0],
            "volatility": [0.2, 0.2],
            "created_at": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        }
    )
    actuals_df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "close": [100.0, 101.5]})

    def _fake_read_sql_query(query, _conn, params=None, *args, **kwargs):  # noqa: ANN001
        calls.append((str(query), params))
        text = str(query)
        if "FROM time_series_forecasts" in text and "model_type as model_name" in text:
            return forecast_df
        if "FROM ohlcv_data" in text:
            return actuals_df
        raise AssertionError(f"Unexpected query: {text}")

    monkeypatch.setattr(mod, "guarded_sqlite_connect", lambda _path: conn)
    monkeypatch.setattr(mod.pd, "read_sql_query", _fake_read_sql_query)

    model_forecasts, actual_values = mod.extract_forecast_data_from_db(
        ticker=ticker,
        days=30,
        pipeline_id="pipeline_20260120_021448",
    )

    assert conn.closed is True
    assert len(calls) == 2

    query_1, params_1 = calls[0]
    assert "ticker = ?" in query_1
    assert "DROP TABLE" not in query_1
    assert isinstance(params_1, list)
    assert params_1[0] == ticker
    assert any(str(p).endswith("%") for p in params_1[1:])  # created_at LIKE param

    query_2, params_2 = calls[1]
    assert "WHERE ticker = ?" in query_2
    assert "ticker = '" not in query_2
    assert params_2 == [ticker, ticker]

    assert "sarimax" in model_forecasts
    assert isinstance(actual_values, np.ndarray)
    assert actual_values.size == 2

