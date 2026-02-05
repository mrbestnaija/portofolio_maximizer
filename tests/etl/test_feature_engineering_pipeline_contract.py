"""
Institutional-grade contract tests for the Feature Engineering Pipeline.

These tests focus on data integrity, scaling consistency, drift diagnostics,
and baseline feature construction. They also include xfail contract checks
for TODO feature blocks that are not yet implemented.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import pytest

from etl.preprocessor import Preprocessor
from etl.split_diagnostics import drift_metrics
from etl.time_series_analyzer import TimeSeriesDatasetAnalyzer
from etl.time_series_feature_builder import TimeSeriesFeatureBuilder


def _make_price_frame(
    ticker: str,
    *,
    start: str = "2024-01-02",
    periods: int = 120,
    freq: str = "B",
    seed: int = 7,
    drift: float = 0.0005,
    vol: float = 0.01,
    missing_idx: Iterable[int] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq=freq)
    returns = rng.normal(drift, vol, size=periods)
    prices = 100 * np.exp(np.cumsum(returns))
    volume = rng.integers(900_000, 1_100_000, size=periods)
    frame = pd.DataFrame({"Close": prices, "Volume": volume}, index=dates)
    frame["ticker"] = ticker

    if missing_idx:
        idx = [dates[i] for i in missing_idx if 0 <= i < len(dates)]
        frame.loc[idx, "Close"] = np.nan

    return frame


def _make_multi_ticker_frame() -> pd.DataFrame:
    aapl = _make_price_frame("AAPL", seed=1, missing_idx=[10, 25, 40])
    msft = _make_price_frame("MSFT", seed=2, drift=0.0002)
    nvda = _make_price_frame("NVDA", seed=3, drift=0.001, vol=0.02)
    return pd.concat([aapl, msft, nvda]).sort_index()


def test_data_integrity_gap_detection() -> None:
    frame = _make_price_frame("AAPL", periods=50)
    # Create an intentional temporal gap.
    gap_block = frame.index[15:25]
    frame = frame.drop(index=gap_block)

    analyzer = TimeSeriesDatasetAnalyzer("gap_detection_test")
    analyzer.data = frame
    structure = analyzer.identify_temporal_structure()

    assert structure["is_time_series"] is True
    assert structure["temporal_gaps_detected"] >= 1


def test_missingness_forward_fill_respects_ticker_boundaries() -> None:
    frame = _make_multi_ticker_frame()
    preprocessor = Preprocessor()

    missing_mask = (frame["ticker"] == "AAPL") & frame["Close"].isna()
    assert missing_mask.sum() > 0

    prev_close = frame.groupby("ticker")["Close"].shift(1)
    filled = preprocessor.handle_missing(frame, method="forward")

    assert filled.loc[missing_mask, "Close"].notna().all()
    assert np.allclose(
        filled.loc[missing_mask, "Close"].to_numpy(),
        prev_close.loc[missing_mask].to_numpy(),
        equal_nan=False,
    )


def test_scaling_consistency_and_stats_roundtrip() -> None:
    frame = _make_multi_ticker_frame()
    preprocessor = Preprocessor()

    columns = ["Close", "Volume"]
    normalized, stats = preprocessor.normalize(frame, method="zscore", columns=columns)
    reapplied = preprocessor.apply_normalization(frame, stats, method="zscore", columns=columns)

    for col in columns:
        assert col in stats
        assert "mean" in stats[col]
        assert "std" in stats[col]

    assert np.allclose(
        normalized[columns].to_numpy(),
        reapplied[columns].to_numpy(),
        atol=1e-8,
        equal_nan=True,
    )


def test_feature_builder_output_is_consistent() -> None:
    builder = TimeSeriesFeatureBuilder()
    aapl = _make_price_frame("AAPL", periods=220)
    msft = _make_price_frame("MSFT", periods=220, seed=11)

    features_aapl = builder.build_features(aapl[["Close"]])
    features_msft = builder.build_features(msft[["Close"]])

    assert not features_aapl.empty
    assert features_aapl.index.is_monotonic_increasing
    assert features_aapl.index.has_duplicates is False

    expected_columns = {
        "price_lag_1",
        "return_lag_1",
        "rolling_mean_5",
        "rolling_std_20",
        "rolling_skew_20",
        "diff_1",
        "is_month_end",
    }
    assert expected_columns.issubset(set(features_aapl.columns))
    assert set(features_aapl.columns) == set(features_msft.columns)
    assert features_aapl.isna().sum().sum() == 0


def test_drift_metrics_detect_shift() -> None:
    train = _make_price_frame("AAPL", periods=160, drift=0.0003, vol=0.01)
    shifted = _make_price_frame("AAPL", periods=160, drift=0.002, vol=0.03, seed=42)

    metrics = drift_metrics(train, shifted)

    assert metrics["psi"] >= 0.0
    assert metrics["vol_psi"] >= 0.0
    assert metrics["psi"] > 0.05 or metrics["vol_psi"] > 0.05


def test_feature_builder_handles_short_series_without_crashing() -> None:
    builder = TimeSeriesFeatureBuilder()
    short_series = _make_price_frame("AAPL", periods=10)
    features = builder.build_features(short_series[["Close"]])
    assert isinstance(features, pd.DataFrame)


def test_feature_builder_requires_close_column() -> None:
    builder = TimeSeriesFeatureBuilder()
    with pytest.raises(ValueError):
        builder.build_features(pd.DataFrame({"Open": [1.0, 2.0]}))


def test_preprocessor_rejects_unknown_missing_method() -> None:
    preprocessor = Preprocessor()
    frame = _make_price_frame("AAPL", periods=20)
    with pytest.raises(ValueError):
        preprocessor.handle_missing(frame, method="unknown")


def test_contract_requires_missingness_and_drift_features() -> None:
    builder = TimeSeriesFeatureBuilder()
    frame = _make_price_frame("AAPL", periods=220)
    features = builder.build_features(frame[["Close"]])

    required_columns = {
        "missing_gap_flag",
        "missing_gap_count",
        "drift_intensity",
        "vol_regime_flag",
        "downside_vol_20",
        "drawdown_depth_60",
        "cvar_proxy_95",
        "cross_sectional_rank_5d",
        "cross_sectional_zscore_20d",
        "microstructure_atr",
        "feature_registry_version",
    }

    missing = required_columns.difference(set(features.columns))
    assert not missing
