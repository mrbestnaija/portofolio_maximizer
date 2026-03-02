"""Feature builder for time-series forecasting stacks."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)

# Signal Quality B: warn when cross-sectional neutralization exceeds this fraction
# of scored rows.  >20% means most rows lack peer-context discrimination.
CROSS_SECTIONAL_FALLBACK_WARN_THRESHOLD = 0.20


@dataclass
class FeatureHealth:
    """Per-build telemetry for feature degradation paths.

    Emitted as a structured JSON log line after every call to
    ``TimeSeriesFeatureBuilder.build_features()``.  Stored on the builder
    instance as ``last_health_report`` for programmatic inspection.

    Signal Quality B: surfaces cross-sectional neutralization, seasonal
    decomposition availability, and OHLCV coverage so overnight pipelines
    can detect silent feature degradation.
    """

    ticker: Optional[str]
    # Row counts
    total_rows: int           # rows in input price_history
    output_rows: int          # rows after dropna() -- what models see
    # Cross-sectional health
    cross_sectional_active: bool      # True = multi-ticker context present
    cross_sectional_fallback_rows: int  # rows using neutral rank=0.5/zscore=0.0
    cross_sectional_fallback_rate: float  # fallback_rows / output_rows
    # Other feature availability
    seasonal_decomp_available: bool   # True = decomposition succeeded
    microstructure_ohlcv_available: bool  # True = High/Low columns present
    # Signal Quality A: macro enrichment coverage
    macro_context_available: bool = False  # True = vix/yield/sector columns present
    # Degradation alerts (non-empty = actionable)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cross_sectional_fallback_rate"] = round(self.cross_sectional_fallback_rate, 4)
        return d


class TimeSeriesFeatureBuilder:
    """Create lag, seasonal, and volatility features for downstream models."""

    def __init__(self) -> None:
        self._last_health_report: Optional[FeatureHealth] = None

    @property
    def last_health_report(self) -> Optional[FeatureHealth]:
        """FeatureHealth from the most recent ``build_features()`` call."""
        return self._last_health_report

    # Signal Quality A: macro column names expected in macro_context
    MACRO_COLUMNS = ("vix_level", "yield_spread_10y_2y", "sector_momentum_5d")

    def build_features(
        self,
        price_history: pd.DataFrame,
        ticker: Optional[str] = None,
        persist_path: Optional[Path] = None,
        macro_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build lag, seasonal, volatility, cross-sectional, and optional macro features.

        Args:
            price_history: OHLCV DataFrame with a ``Close`` column.
            ticker: Optional ticker symbol for telemetry.
            persist_path: When provided, saves the feature DataFrame to this path.
            macro_context: Optional DataFrame indexed like ``price_history`` containing
                any subset of ``vix_level``, ``yield_spread_10y_2y``,
                ``sector_momentum_5d`` columns.  Present columns are merged in;
                missing columns are silently omitted.  All values are forward-filled
                then back-filled to handle business-day alignment gaps.
        """
        if "Close" not in price_history.columns:
            raise ValueError("price_history must contain a Close column")
        df = price_history.copy()
        df = df.sort_index()
        close = df["Close"]
        returns = close.pct_change()

        features: dict[str, pd.Series] = {}

        # Signal Quality B: track degradation paths as we build
        _cross_sectional_active = False
        _seasonal_decomp_available = False
        _microstructure_ohlcv_available = {"High", "Low"}.issubset(df.columns)
        _macro_context_available = False

        # Lag + returns
        for lag in (1, 5, 10, 20):
            features[f"price_lag_{lag}"] = close.shift(lag)
            features[f"return_lag_{lag}"] = close.pct_change(lag)

        # Rolling stats -- min_periods=1 so short series don't yield all-NaN rows
        for window in (5, 10, 20, 60):
            roll = close.rolling(window, min_periods=1)
            features[f"rolling_mean_{window}"] = roll.mean()
            features[f"rolling_std_{window}"] = roll.std().fillna(0.0)
            features[f"rolling_skew_{window}"] = roll.skew().fillna(0.0)

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
            _cross_sectional_active = True
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
            # Signal Quality B: neutral fallback -- cross-sectional rank/zscore
            # lose discriminative power without peer-context.  Logged below.
            features["cross_sectional_rank_5d"] = pd.Series(0.5, index=df.index)
            features["cross_sectional_zscore_20d"] = pd.Series(0.0, index=df.index)

        # Microstructure proxy (ATR fallback)
        if _microstructure_ohlcv_available:
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

        # Seasonal decomposition (safe fallback).
        # period=126 (~6 months of trading days): statsmodels requires 2 complete
        # cycles (2×126=252), so 300-row inputs succeed and <252-row inputs fail
        # gracefully.  period=252 (annual) would need 504+ rows.
        try:
            decomposition = seasonal_decompose(close, model="additive", period=126, two_sided=False, extrapolate_trend="freq")
            features["trend"] = decomposition.trend
            features["seasonal"] = decomposition.seasonal
            features["residual"] = decomposition.resid
            _seasonal_decomp_available = True
        except Exception as exc:
            logger.debug("Seasonal decomposition unavailable: %s", exc)

        # Calendar flags
        idx = df.index
        if hasattr(idx, "is_month_end"):
            features["is_month_end"] = idx.is_month_end.astype(int)
            features["is_quarter_end"] = idx.is_quarter_end.astype(int)

        features["feature_registry_version"] = pd.Series("v1", index=df.index)

        # Signal Quality A: merge optional macro context columns.
        # Only columns in MACRO_COLUMNS that are present in macro_context are merged.
        # LEAK-02 fix: clip macro_context to price_history date range before alignment
        # to prevent bfill() from filling past feature rows with future macro values.
        if macro_context is not None and not macro_context.empty:
            price_end = df.index.max()
            clipped_macro = macro_context[macro_context.index <= price_end]
            for col in self.MACRO_COLUMNS:
                if col in clipped_macro.columns:
                    aligned = (
                        clipped_macro[col]
                        .reindex(df.index)
                        .ffill()
                        .fillna(0.0)
                    )
                    features[col] = aligned.astype(float)
                    _macro_context_available = True

        feature_df = pd.DataFrame(features, index=df.index).dropna()

        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_parquet(persist_path)
            logger.info("Saved time-series features to %s (%s rows)", persist_path, len(feature_df))

        # ------------------------------------------------------------------
        # Signal Quality B: build and emit feature-health telemetry
        # ------------------------------------------------------------------
        output_rows = len(feature_df)
        cross_sectional_fallback_rows = 0 if _cross_sectional_active else output_rows
        cross_sectional_fallback_rate = (
            cross_sectional_fallback_rows / output_rows if output_rows > 0 else 0.0
        )

        health_warnings: List[str] = []
        if cross_sectional_fallback_rate > CROSS_SECTIONAL_FALLBACK_WARN_THRESHOLD:
            health_warnings.append(
                f"cross_sectional_neutralized: {cross_sectional_fallback_rows}/{output_rows} rows "
                f"({cross_sectional_fallback_rate:.1%}) use neutral rank=0.5/zscore=0.0 -- "
                "run in multi-ticker mode to restore discriminative power"
            )
        if not _seasonal_decomp_available:
            health_warnings.append(
                "seasonal_decomp_unavailable: trend/seasonal/residual features absent -- "
                "insufficient history or decomposition error"
            )
        if not _microstructure_ohlcv_available:
            health_warnings.append(
                "microstructure_close_only: ATR computed from close diff (no High/Low) -- "
                "OHLCV data preferred for accurate microstructure proxy"
            )

        health = FeatureHealth(
            ticker=ticker,
            total_rows=len(df),
            output_rows=output_rows,
            cross_sectional_active=_cross_sectional_active,
            cross_sectional_fallback_rows=cross_sectional_fallback_rows,
            cross_sectional_fallback_rate=round(cross_sectional_fallback_rate, 4),
            seasonal_decomp_available=_seasonal_decomp_available,
            microstructure_ohlcv_available=_microstructure_ohlcv_available,
            macro_context_available=_macro_context_available,
            warnings=health_warnings,
        )
        self._last_health_report = health

        logger.info("feature_health %s", json.dumps(health.as_dict()))

        if health_warnings:
            for w in health_warnings:
                logger.warning("[FEATURE_HEALTH] %s", w)

        return feature_df
