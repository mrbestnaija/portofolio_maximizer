from __future__ import annotations

import pandas as pd

from forcester_ts.forecaster import TimeSeriesForecaster


def test_macro_context_leading_gaps_are_zero_filled_without_bfill() -> None:
    forecaster = TimeSeriesForecaster()
    price_index = pd.date_range("2024-01-01", periods=5, freq="D")
    price_series = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=price_index)
    macro_context = pd.DataFrame(
        {"vix_level": [18.0, 19.0, 20.0]},
        index=pd.date_range("2024-01-03", periods=3, freq="D"),
    )

    exog = forecaster._build_sarimax_exogenous(
        price_series=price_series,
        returns_series=None,
        macro_context=macro_context,
    )

    assert exog.loc[pd.Timestamp("2024-01-01"), "vix_level"] == 0.0
    assert exog.loc[pd.Timestamp("2024-01-02"), "vix_level"] == 0.0
    assert exog.loc[pd.Timestamp("2024-01-03"), "vix_level"] == 18.0
    assert forecaster._sarimax_exog_policy["fit_alignment"] == "ffill_only_zero_leading"
    assert forecaster._sarimax_exog_policy["leading_zero_fill_count"]["vix_level"] == 2


def test_macro_context_is_clipped_to_price_window() -> None:
    forecaster = TimeSeriesForecaster()
    price_index = pd.date_range("2024-01-02", periods=3, freq="D")
    price_series = pd.Series([100.0, 101.0, 102.0], index=price_index)
    macro_context = pd.DataFrame(
        {"vix_level": [9.0, 10.0, 11.0, 12.0, 13.0]},
        index=pd.date_range("2023-12-31", periods=5, freq="D"),
    )

    exog = forecaster._build_sarimax_exogenous(
        price_series=price_series,
        returns_series=None,
        macro_context=macro_context,
    )

    assert list(exog["vix_level"]) == [11.0, 12.0, 13.0]
    assert forecaster._sarimax_exog_policy["macro_context_clipped_to_price_window"] is True
    assert forecaster._sarimax_exog_policy["macro_context_rows_before_clip"] == 5
    assert forecaster._sarimax_exog_policy["macro_context_rows_after_clip"] == 3


def test_forecast_exog_uses_last_observation_hold_policy() -> None:
    forecaster = TimeSeriesForecaster()
    forecaster._sarimax_exog_columns = ["ret_1", "vix_level"]
    forecaster._sarimax_exog_last_row = pd.Series({"ret_1": 0.1, "vix_level": 18.5})

    exog = forecaster._build_sarimax_forecast_exogenous(3)

    assert exog is not None
    assert exog.shape == (3, 2)
    assert exog["ret_1"].tolist() == [0.1, 0.1, 0.1]
    assert exog["vix_level"].tolist() == [18.5, 18.5, 18.5]
    assert forecaster._sarimax_forecast_exog_policy == {
        "mode": "last_observation_hold",
        "horizon": 3,
        "columns": ["ret_1", "vix_level"],
    }
