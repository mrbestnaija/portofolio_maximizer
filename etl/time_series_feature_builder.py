"""Feature builder for time-series forecasting stacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


class TimeSeriesFeatureBuilder:
    """Create lag, seasonal, and volatility features for downstream models."""

    def build_features(
        self,
        price_history: pd.DataFrame,
        ticker: Optional[str] = None,
        persist_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        if "Close" not in price_history.columns:
            raise ValueError("price_history must contain a Close column")
        df = price_history.copy()
        df = df.sort_index()

        features: dict[str, pd.Series] = {}

        # Lag + returns
        for lag in (1, 5, 10, 20):
            features[f"price_lag_{lag}"] = df["Close"].shift(lag)
            features[f"return_lag_{lag}"] = df["Close"].pct_change(lag)

        # Rolling stats
        for window in (5, 10, 20, 60):
            roll = df["Close"].rolling(window)
            features[f"rolling_mean_{window}"] = roll.mean()
            features[f"rolling_std_{window}"] = roll.std()
            features[f"rolling_skew_{window}"] = roll.skew()

        # Differencing
        features["diff_1"] = df["Close"].diff(1)
        features["diff_5"] = df["Close"].diff(5)

        # Seasonal decomposition (safe fallback)
        try:
            decomposition = seasonal_decompose(df["Close"], model="additive", period=252, two_sided=False, extrapolate_trend="freq")
            features["trend"] = decomposition.trend
            features["seasonal"] = decomposition.seasonal
            features["residual"] = decomposition.resid
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Seasonal decomposition unavailable: %s", exc)

        # Calendar flags
        idx = df.index
        if hasattr(idx, "is_month_end"):
            features["is_month_end"] = idx.is_month_end.astype(int)
            features["is_quarter_end"] = idx.is_quarter_end.astype(int)

        feature_df = pd.DataFrame(features, index=df.index).dropna()

        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_parquet(persist_path)
            logger.info("Saved time-series features to %s (%s rows)", persist_path, len(feature_df))

        return feature_df
