"""Tests for TimeSeriesFeatureBuilder and FeatureHealth telemetry (Signal Quality B)."""

from __future__ import annotations

import json
import logging
from typing import List

import numpy as np
import pandas as pd
import pytest

from etl.time_series_feature_builder import (
    CROSS_SECTIONAL_FALLBACK_WARN_THRESHOLD,
    FeatureHealth,
    TimeSeriesFeatureBuilder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_history(
    n: int = 300,
    ticker: str = "AAPL",
    with_ohlcv: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic single-ticker OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 150.0 + rng.standard_normal(n).cumsum()
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame({"Close": close}, index=dates)
    if with_ohlcv:
        df["High"] = high
        df["Low"] = low
        df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    return df


def _make_multi_ticker_history(
    n: int = 300,
    tickers: List[str] = ("AAPL", "MSFT", "NVDA"),
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic multi-ticker OHLCV DataFrame with a 'ticker' column."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    frames = []
    for t in tickers:
        close = 100.0 + rng.standard_normal(n).cumsum()
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)
        f = pd.DataFrame(
            {"Close": close, "High": high, "Low": low, "ticker": t},
            index=dates,
        )
        frames.append(f)
    return pd.concat(frames).sort_index()


# ---------------------------------------------------------------------------
# Construction and interface
# ---------------------------------------------------------------------------

class TestTimeSeriesFeatureBuilderInterface:
    def test_builder_has_last_health_report_attribute(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        assert builder.last_health_report is None

    def test_build_features_returns_dataframe(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        df = _make_price_history()
        result = builder.build_features(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_build_features_requires_close_column(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        df = pd.DataFrame({"Open": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Close"):
            builder.build_features(df)

    def test_last_health_report_populated_after_build(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        df = _make_price_history()
        builder.build_features(df)
        assert builder.last_health_report is not None
        assert isinstance(builder.last_health_report, FeatureHealth)

    def test_last_health_report_replaced_on_second_call(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        df = _make_price_history(n=300, ticker="AAPL")
        builder.build_features(df, ticker="AAPL")
        first = builder.last_health_report
        builder.build_features(df, ticker="MSFT")
        second = builder.last_health_report
        assert first is not second
        assert second.ticker == "MSFT"

    def test_health_as_dict_is_json_serialisable(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history())
        d = builder.last_health_report.as_dict()
        # Must round-trip through JSON without error
        json.dumps(d)


# ---------------------------------------------------------------------------
# Cross-sectional health
# ---------------------------------------------------------------------------

class TestCrossSectionalHealth:
    def test_single_ticker_cross_sectional_inactive(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history())
        h = builder.last_health_report
        assert h.cross_sectional_active is False

    def test_single_ticker_all_rows_in_fallback(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        df = _make_price_history(n=300)
        result = builder.build_features(df)
        h = builder.last_health_report
        assert h.cross_sectional_fallback_rows == h.output_rows
        assert h.cross_sectional_fallback_rate == 1.0

    def test_single_ticker_fallback_rate_above_threshold(self) -> None:
        assert CROSS_SECTIONAL_FALLBACK_WARN_THRESHOLD < 1.0
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(n=300))
        h = builder.last_health_report
        assert h.cross_sectional_fallback_rate > CROSS_SECTIONAL_FALLBACK_WARN_THRESHOLD

    def test_multi_ticker_cross_sectional_active(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_multi_ticker_history())
        h = builder.last_health_report
        assert h.cross_sectional_active is True

    def test_multi_ticker_zero_fallback_rows(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_multi_ticker_history())
        h = builder.last_health_report
        assert h.cross_sectional_fallback_rows == 0
        assert h.cross_sectional_fallback_rate == 0.0

    def test_single_ticker_neutral_feature_values(self) -> None:
        """Neutral values are rank=0.5 and zscore=0.0 per feature-builder contract."""
        builder = TimeSeriesFeatureBuilder()
        result = builder.build_features(_make_price_history(n=300))
        assert (result["cross_sectional_rank_5d"] == 0.5).all()
        assert (result["cross_sectional_zscore_20d"] == 0.0).all()


# ---------------------------------------------------------------------------
# Seasonal decomposition health
# ---------------------------------------------------------------------------

class TestSeasonalDecompHealth:
    def test_sufficient_history_decomp_available(self) -> None:
        # 300 business days > 252 (period) -- decomp should succeed
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(n=300))
        h = builder.last_health_report
        assert h.seasonal_decomp_available is True

    def test_insufficient_history_decomp_unavailable(self) -> None:
        # 50 rows < 252 (period) -- decomp must fail gracefully
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(n=50))
        h = builder.last_health_report
        assert h.seasonal_decomp_available is False

    def test_short_series_does_not_raise(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        result = builder.build_features(_make_price_history(n=50))
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Microstructure OHLCV health
# ---------------------------------------------------------------------------

class TestMicrostructureHealth:
    def test_ohlcv_microstructure_available(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(with_ohlcv=True))
        assert builder.last_health_report.microstructure_ohlcv_available is True

    def test_close_only_microstructure_fallback(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(with_ohlcv=False))
        assert builder.last_health_report.microstructure_ohlcv_available is False

    def test_close_only_still_produces_atr_column(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        result = builder.build_features(_make_price_history(with_ohlcv=False))
        assert "microstructure_atr" in result.columns


# ---------------------------------------------------------------------------
# Warning emission
# ---------------------------------------------------------------------------

class TestHealthWarnings:
    def test_single_ticker_emits_cross_sectional_warning(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(n=300))
        h = builder.last_health_report
        assert any("cross_sectional_neutralized" in w for w in h.warnings)

    def test_multi_ticker_no_cross_sectional_warning(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_multi_ticker_history())
        h = builder.last_health_report
        assert not any("cross_sectional_neutralized" in w for w in h.warnings)

    def test_short_series_emits_seasonal_warning(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(n=50))
        h = builder.last_health_report
        assert any("seasonal_decomp_unavailable" in w for w in h.warnings)

    def test_close_only_emits_microstructure_warning(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(with_ohlcv=False))
        h = builder.last_health_report
        assert any("microstructure_close_only" in w for w in h.warnings)

    def test_full_ohlcv_long_multi_ticker_no_warnings(self) -> None:
        """Multi-ticker + OHLCV + sufficient history = no health warnings."""
        builder = TimeSeriesFeatureBuilder()
        df = _make_multi_ticker_history(n=300)
        builder.build_features(df)
        h = builder.last_health_report
        # cross-sectional active, seasonal available, ohlcv available -> no warnings
        assert h.warnings == []

    def test_warnings_logged_as_warning_level(self, caplog: pytest.LogCaptureFixture) -> None:
        builder = TimeSeriesFeatureBuilder()
        with caplog.at_level(logging.WARNING, logger="etl.time_series_feature_builder"):
            builder.build_features(_make_price_history(n=300))
        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("FEATURE_HEALTH" in t for t in warning_texts)

    def test_health_summary_logged_as_info(self, caplog: pytest.LogCaptureFixture) -> None:
        builder = TimeSeriesFeatureBuilder()
        with caplog.at_level(logging.INFO, logger="etl.time_series_feature_builder"):
            builder.build_features(_make_price_history(n=300))
        info_texts = [r.message for r in caplog.records if r.levelno == logging.INFO]
        json_lines = [t for t in info_texts if "feature_health" in t]
        assert len(json_lines) >= 1
        # Must be valid JSON after stripping the leading prefix
        payload = json_lines[0].replace("feature_health ", "", 1)
        parsed = json.loads(payload)
        assert "cross_sectional_fallback_rate" in parsed
        assert "seasonal_decomp_available" in parsed


# ---------------------------------------------------------------------------
# Row count consistency
# ---------------------------------------------------------------------------

class TestRowCounts:
    def test_total_rows_matches_input(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        n = 250
        df = _make_price_history(n=n)
        builder.build_features(df)
        h = builder.last_health_report
        assert h.total_rows == n

    def test_output_rows_le_total_rows(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        df = _make_price_history(n=300)
        result = builder.build_features(df)
        h = builder.last_health_report
        assert h.output_rows == len(result)
        assert h.output_rows <= h.total_rows

    def test_ticker_propagated_to_health(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(), ticker="NVDA")
        assert builder.last_health_report.ticker == "NVDA"

    def test_ticker_none_when_not_provided(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history())
        assert builder.last_health_report.ticker is None


# ---------------------------------------------------------------------------
# Signal Quality A: macro context enrichment
# ---------------------------------------------------------------------------

def _make_macro_context(price_df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Synthetic macro DataFrame aligned to price_df's index."""
    rng = np.random.default_rng(seed)
    n = len(price_df)
    return pd.DataFrame(
        {
            "vix_level": 15.0 + rng.standard_normal(n) * 3.0,
            "yield_spread_10y_2y": 0.5 + rng.standard_normal(n) * 0.2,
            "sector_momentum_5d": rng.standard_normal(n) * 0.01,
        },
        index=price_df.index,
    )


class TestMacroContextEnrichment:
    """Signal Quality A: verify macro columns appear in features and FeatureHealth."""

    def test_macro_columns_added_to_features(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        price_df = _make_price_history(n=300)
        macro = _make_macro_context(price_df)
        result = builder.build_features(price_df, macro_context=macro)
        for col in TimeSeriesFeatureBuilder.MACRO_COLUMNS:
            assert col in result.columns, f"Expected macro column {col!r} in output"

    def test_macro_context_available_true_in_health(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        price_df = _make_price_history(n=300)
        macro = _make_macro_context(price_df)
        builder.build_features(price_df, macro_context=macro)
        assert builder.last_health_report.macro_context_available is True

    def test_macro_context_available_false_when_not_provided(self) -> None:
        builder = TimeSeriesFeatureBuilder()
        builder.build_features(_make_price_history(n=300))
        assert builder.last_health_report.macro_context_available is False

    def test_macro_partial_columns_only_present_added(self) -> None:
        """Only the columns present in macro_context should appear."""
        builder = TimeSeriesFeatureBuilder()
        price_df = _make_price_history(n=300)
        # Provide only vix_level
        macro = pd.DataFrame({"vix_level": 15.0}, index=price_df.index)
        result = builder.build_features(price_df, macro_context=macro)
        assert "vix_level" in result.columns
        assert "yield_spread_10y_2y" not in result.columns
        assert "sector_momentum_5d" not in result.columns
        assert builder.last_health_report.macro_context_available is True

    def test_macro_context_none_no_change(self) -> None:
        """When macro_context=None output is identical to default call."""
        builder1 = TimeSeriesFeatureBuilder()
        builder2 = TimeSeriesFeatureBuilder()
        price_df = _make_price_history(n=300)
        r1 = builder1.build_features(price_df, macro_context=None)
        r2 = builder2.build_features(price_df)
        assert list(r1.columns) == list(r2.columns)

    def test_macro_values_forward_filled_on_gap(self) -> None:
        """Values must be ffilled/bfilled to handle business-day alignment gaps."""
        builder = TimeSeriesFeatureBuilder()
        price_df = _make_price_history(n=50)
        macro = _make_macro_context(price_df)
        # Introduce NaNs at some positions
        macro.loc[macro.index[5:10], "vix_level"] = float("nan")
        result = builder.build_features(price_df, macro_context=macro)
        assert result["vix_level"].isna().sum() == 0, "vix_level must have no NaNs after fill"
