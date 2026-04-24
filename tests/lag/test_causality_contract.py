from __future__ import annotations

import pandas as pd

from forcester_ts import cross_validation as cv_mod
from forcester_ts.cross_validation import RollingWindowCVConfig, RollingWindowValidator


class _SyntheticCausalPipeline:
    def run(self) -> dict[str, pd.Timestamp]:
        feature_ts = pd.Timestamp("2026-04-18 13:00:00", tz="UTC")
        signal_ts = feature_ts + pd.Timedelta(minutes=2)
        execution_ts = signal_ts + pd.Timedelta(minutes=1)
        close_ts = execution_ts + pd.Timedelta(minutes=57)
        return {
            "feature_ts": feature_ts,
            "signal_ts": signal_ts,
            "execution_ts": execution_ts,
            "close_ts": close_ts,
        }


def _validate_event_order(event: dict[str, pd.Timestamp]) -> None:
    assert event["feature_ts"] <= event["signal_ts"] <= event["execution_ts"] <= event["close_ts"]


def test_causal_timestamp_ordering_contract() -> None:
    pipeline = _SyntheticCausalPipeline()
    event = pipeline.run()

    _validate_event_order(event)

    broken = dict(event)
    broken["execution_ts"] = broken["signal_ts"] - pd.Timedelta(seconds=1)

    try:
        _validate_event_order(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected causal ordering assertion to fail for broken timeline")


def test_rolling_window_cv_keeps_training_and_evaluation_disjoint(monkeypatch) -> None:
    price_index = pd.date_range("2026-01-01", periods=28, freq="D")
    price_series = pd.Series(
        [100.0 + i for i in range(len(price_index))],
        index=price_index,
        name="Close",
    )
    returns_series = price_series.pct_change().dropna()

    captured_windows: list[tuple[pd.Index, pd.Index]] = []

    class _DummyForecaster:
        def __init__(self, config=None) -> None:
            self.config = config
            self._cv_fold_metrics = {}
            self._train_index = pd.Index([])

        def fit(self, price_series, returns_series=None, ticker=""):
            del returns_series, ticker
            self._train_index = price_series.index.copy()
            return self

        def forecast(self, steps):
            del steps
            return {}

        def evaluate(self, actual_series):
            test_index = actual_series.index.copy()
            captured_windows.append((self._train_index.copy(), test_index))
            return {
                "samossa": {
                    "rmse": 1.0,
                    "directional_accuracy": 0.5,
                    "n_observations": float(len(actual_series)),
                }
            }

    monkeypatch.setattr(cv_mod, "TimeSeriesForecaster", _DummyForecaster)

    validator = RollingWindowValidator(
        forecaster_config=cv_mod.TimeSeriesForecasterConfig(
            forecast_horizon=4,
            sarimax_enabled=False,
            garch_enabled=False,
            samossa_enabled=False,
            mssa_rl_enabled=False,
            ensemble_enabled=False,
        ),
        cv_config=RollingWindowCVConfig(min_train_size=12, horizon=4, step_size=4, max_folds=3),
    )
    results = validator.run(price_series=price_series, returns_series=returns_series, ticker="AAPL")

    assert results["fold_count"] == len(captured_windows) == len(results["folds"])
    assert results["fold_count"] > 0
    for fold, (train_index, test_index) in zip(results["folds"], captured_windows):
        assert fold["train_range"]["end"] < fold["test_range"]["start"]
        assert train_index.intersection(test_index).empty
        assert train_index.max() < test_index.min()
