from __future__ import annotations

import pandas as pd

import forcester_ts.cross_validation as cv_mod
from forcester_ts.forecaster import TimeSeriesForecasterConfig


def test_rolling_window_validator_passes_explicit_ticker_to_forecaster(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class StubForecaster:
        def __init__(self, config=None) -> None:
            self.config = config

        def fit(self, price_series, returns_series=None, ticker="", macro_context=None):  # noqa: ARG002
            captured["ticker"] = ticker
            return self

        def forecast(self, steps=None):  # noqa: ARG002
            return {}

        def evaluate(self, actual_series):  # noqa: ARG002
            return {"ensemble": {"rmse": 1.0, "directional_accuracy": 0.5}}

    monkeypatch.setattr(cv_mod, "TimeSeriesForecaster", StubForecaster)

    series = pd.Series(
        [100.0 + float(i) for i in range(20)],
        index=pd.date_range("2025-01-01", periods=20, freq="D"),
        name="Close",
    )

    validator = cv_mod.RollingWindowValidator(
        forecaster_config=TimeSeriesForecasterConfig(),
        cv_config=cv_mod.RollingWindowCVConfig(
            min_train_size=10,
            horizon=5,
            step_size=5,
            max_folds=1,
        ),
    )
    report = validator.run(price_series=series, ticker="AAPL")

    assert report["fold_count"] == 1
    assert captured["ticker"] == "AAPL"
