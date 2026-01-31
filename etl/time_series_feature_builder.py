"""Feature builder for time-series forecasting stacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
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
        close = df["Close"]
        returns = close.pct_change()

        features: dict[str, pd.Series] = {}

        # Lag + returns
        for lag in (1, 5, 10, 20):
            features[f"price_lag_{lag}"] = close.shift(lag)
            features[f"return_lag_{lag}"] = close.pct_change(lag)

        # Rolling stats
        for window in (5, 10, 20, 60):
            roll = close.rolling(window)
            features[f"rolling_mean_{window}"] = roll.mean()
            features[f"rolling_std_{window}"] = roll.std()
            features[f"rolling_skew_{window}"] = roll.skew()

        # Differencing
        features["diff_1"] = close.diff(1)
        features["diff_5"] = close.diff(5)

        # Missingness indicators
        missing_mask = close.isna()
        features["missing_gap_flag"] = missing_mask.astype(int)
        if missing_mask.any():
            gap_groups = missing_mask.ne(missing_mask.shift()).cumsum()
            gap_sizes = missing_mask.groupby(gap_groups).transform("sum")
            features["missing_gap_count"] = gap_sizes.where(missing_mask, 0).astype(float)
        else:
            features["missing_gap_count"] = pd.Series(0.0, index=df.index)

        # Drift and regime indicators
        short_mean = returns.rolling(20).mean()
        long_mean = returns.rolling(60).mean()
        long_std = returns.rolling(60).std()
        features["drift_intensity"] = ((short_mean - long_mean).abs() / (long_std + 1e-9)).fillna(0.0)

        vol = returns.rolling(20).std()
        if vol.notna().any():
            low_q = float(vol.quantile(0.33))
            high_q = float(vol.quantile(0.66))
            regime = pd.cut(
                vol,
                bins=[-np.inf, low_q, high_q, np.inf],
                labels=[0, 1, 2],
            ).astype(float)
            features["vol_regime_flag"] = regime.fillna(0.0)
        else:
            features["vol_regime_flag"] = pd.Series(0.0, index=df.index)

        # Volatility & tail-risk features
        downside = returns.where(returns < 0, 0.0)
        features["downside_vol_20"] = downside.rolling(20).std().fillna(0.0)
        rolling_max = close.rolling(60).max()
        drawdown = (close / rolling_max) - 1.0
        features["drawdown_depth_60"] = drawdown.rolling(60).min().abs().fillna(0.0)
        features["cvar_proxy_95"] = returns.rolling(60).quantile(0.05).abs().fillna(0.0)

        # Cross-sectional features (fallback to neutral when ticker context is missing)
        if "ticker" in df.columns and df["ticker"].nunique() > 1:
            ret_5d = df.groupby("ticker")["Close"].pct_change(5)
            ret_20d = df.groupby("ticker")["Close"].pct_change(20)
            tmp = pd.DataFrame({"ret_5d": ret_5d, "ret_20d": ret_20d}, index=df.index)
            features["cross_sectional_rank_5d"] = (
                tmp.groupby(tmp.index)["ret_5d"].rank(pct=True).fillna(0.5)
            )

            def _zscore(series: pd.Series) -> pd.Series:
                mean = series.mean()
                std = series.std()
                if std == 0 or np.isnan(std):
                    return series * 0.0
                return (series - mean) / std

            features["cross_sectional_zscore_20d"] = (
                tmp.groupby(tmp.index)["ret_20d"].transform(_zscore).fillna(0.0)
            )
        else:
            features["cross_sectional_rank_5d"] = pd.Series(0.5, index=df.index)
            features["cross_sectional_zscore_20d"] = pd.Series(0.0, index=df.index)

        # Microstructure proxy (ATR fallback)
        if {"High", "Low"}.issubset(df.columns):
            prev_close = close.shift(1)
            high = df["High"]
            low = df["Low"]
            tr = pd.concat(
                [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
        else:
            tr = close.diff().abs()
        features["microstructure_atr"] = tr.rolling(14).mean().fillna(0.0)

        # Seasonal decomposition (safe fallback)
        try:
            decomposition = seasonal_decompose(close, model="additive", period=252, two_sided=False, extrapolate_trend="freq")
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

        features["feature_registry_version"] = pd.Series("v1", index=df.index)

        feature_df = pd.DataFrame(features, index=df.index).dropna()

        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_parquet(persist_path)
            logger.info("Saved time-series features to %s (%s rows)", persist_path, len(feature_df))

        return feature_df
