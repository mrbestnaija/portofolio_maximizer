"""
Regime detection for adaptive model selection.
Identifies market conditions and recommends appropriate forecasting models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    enabled: bool = True
    lookback_window: int = 60  # Days to analyze
    vol_threshold_low: float = 0.15  # Daily vol < 15% = low vol
    vol_threshold_high: float = 0.30  # Daily vol > 30% = high vol
    trend_threshold_weak: float = 0.30  # ADX < 30 = weak trend
    trend_threshold_strong: float = 0.60  # ADX > 60 = strong trend


class RegimeDetector:
    """
    Detect market regimes and recommend appropriate models.

    Regimes:
    - LIQUID_RANGEBOUND: Low vol, weak trend → GARCH optimal
    - MODERATE_TRENDING: Medium vol, medium trend → Mixed ensemble
    - HIGH_VOL_TRENDING: High vol, strong trend → SAMoSSA/Neural
    - CRISIS: Extreme vol, structural breaks → Defensive
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

    def detect_regime(
        self,
        price_series: pd.Series,
        returns_series: Optional[pd.Series] = None,
    ) -> Dict[str, any]:
        """
        Detect current market regime based on price/returns history.

        Returns:
            Dict with regime classification and model recommendations
        """
        if not self.config.enabled:
            return {'regime': 'UNKNOWN', 'recommendations': []}

        if returns_series is None:
            returns_series = price_series.pct_change().dropna()

        # Extract features
        features = self._extract_regime_features(price_series, returns_series)

        # Classify regime
        regime = self._classify_regime(features)

        # Generate model recommendations
        recommendations = self._recommend_models(regime, features)

        return {
            'regime': regime,
            'features': features,
            'recommendations': recommendations,
            'confidence': self._regime_confidence(features),
        }

    def _extract_regime_features(
        self,
        price_series: pd.Series,
        returns_series: pd.Series,
    ) -> Dict[str, float]:
        """Extract quantitative regime features."""
        window = min(self.config.lookback_window, len(returns_series))
        recent_returns = returns_series.iloc[-window:]

        # Volatility metrics
        realized_vol = recent_returns.std() * np.sqrt(252)  # Annualized
        vol_of_vol = recent_returns.rolling(5).std().std()  # Vol clustering

        # Trend strength (using ADX-like calculation)
        trend_strength = self._calculate_trend_strength(price_series.iloc[-window:])

        # Mean reversion test (Hurst exponent)
        hurst = self._calculate_hurst_exponent(price_series.iloc[-window:])

        # Stationarity (ADF test)
        adf_pvalue = self._adf_test(returns_series.iloc[-window:])

        # Jump detection (tail risk)
        # Flat returns series produces NaN for skew/kurtosis (undefined moments).
        skewness = recent_returns.skew()
        kurtosis = recent_returns.kurtosis()

        return {
            'realized_volatility': float(realized_vol) if np.isfinite(realized_vol) else 0.0,
            'vol_of_vol': float(vol_of_vol) if np.isfinite(vol_of_vol) else 0.0,
            'trend_strength': float(trend_strength) if np.isfinite(trend_strength) else 0.0,
            'hurst_exponent': float(hurst) if np.isfinite(hurst) else 0.5,
            'adf_pvalue': float(adf_pvalue) if np.isfinite(adf_pvalue) else 1.0,
            'skewness': float(skewness) if np.isfinite(skewness) else 0.0,
            'kurtosis': float(kurtosis) if np.isfinite(kurtosis) else 0.0,
            'mean_return': float(recent_returns.mean()) if np.isfinite(recent_returns.mean()) else 0.0,
        }

    def _calculate_trend_strength(self, price_series: pd.Series) -> float:
        """
        Calculate trend strength using directional movement (ADX-like).
        Returns 0-1, where 0 = no trend, 1 = very strong trend.
        """
        if len(price_series) < 14:
            return 0.0

        # Simple trend strength: linear regression R²
        x = np.arange(len(price_series))
        y = price_series.values

        # Flat series has zero variance → correlation undefined (NaN from linregress).
        # A constant price is definitionally trend-free.
        if np.std(y) < 1e-10:
            return 0.0

        # Fit linear regression
        slope, intercept, r_value, _, _ = scipy_stats.linregress(x, y)

        # Guard against NaN r_value (e.g., near-constant series with numerical noise)
        if not np.isfinite(r_value):
            return 0.0

        # R² as trend strength
        r_squared = r_value ** 2

        return float(np.clip(r_squared, 0.0, 1.0))

    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent for mean reversion detection.
        H < 0.5: Mean reverting (GARCH good)
        H = 0.5: Random walk
        H > 0.5: Trending (SAMoSSA/Neural better)
        """
        if len(series) < max_lag + 1:
            return 0.5  # Assume random walk if insufficient data

        lags = list(range(2, min(max_lag, len(series) // 2)))
        # Need at least 2 lag points for polyfit(degree=1) to be defined.
        if len(lags) < 2:
            return 0.5  # Insufficient lags — assume random walk

        # Constant series: all diffs are zero -> all tau == 0.
        # Clamping to 1e-12 gives a horizontal log-log line (slope=0 → H=0),
        # but a flat/constant series is mean-reverting in spirit.  Detect this
        # early and return 0.0 explicitly so regime classification sees it as
        # strongly mean-reverting (GARCH-appropriate) rather than random walk.
        prices = np.asarray(series.values, dtype=float)
        if np.std(prices) < 1e-10:
            return 0.0  # Constant / near-constant series: maximally mean-reverting

        tau = []
        for lag in lags:
            diffs = np.subtract(prices[lag:], prices[:-lag])
            tau.append(float(np.std(diffs)))

        # Fit power law: tau ~ lag^H
        # Clamp tau to avoid log(0) = -inf which causes polyfit to return NaN
        # silently (it does not raise on all-infinite inputs).
        try:
            tau_clamped = [max(t, 1e-12) for t in tau]
            poly = np.polyfit(np.log(lags), np.log(tau_clamped), 1)
            hurst = float(poly[0])
            if not np.isfinite(hurst):
                # All tau values were clamped to 1e-12 (near-constant series):
                # treat as maximally mean-reverting, not random walk.
                return 0.0 if np.std(prices) < 1e-5 else 0.5
            return float(np.clip(hurst, 0.0, 1.0))
        except (ValueError, np.linalg.LinAlgError):
            return 0.0 if np.std(prices) < 1e-5 else 0.5

    def _adf_test(self, series: pd.Series) -> float:
        """
        Augmented Dickey-Fuller test for stationarity.
        Returns p-value: < 0.05 = stationary (GARCH good)
        """
        from statsmodels.tsa.stattools import adfuller

        try:
            result = adfuller(series.dropna(), maxlag=int(len(series) ** 0.25))
            return float(result[1])  # p-value
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return 1.0  # Assume non-stationary on error

    def _classify_regime(self, features: Dict[str, float]) -> str:
        """Classify regime based on extracted features.

        Phase 7.10: Relaxed thresholds to reduce 90% MODERATE_MIXED/TRENDING
        concentration.  Previous logic required ALL conditions for most regimes,
        causing 90.3% of observations to fall into low-confidence catch-all
        buckets with HIGH_VOL_TRENDING at 1.0% and CRISIS at 0.4%.
        """
        vol = features['realized_volatility']
        trend = features['trend_strength']
        hurst = features['hurst_exponent']
        adf_p = features['adf_pvalue']

        # Extreme vol → CRISIS (lowered from 0.50 to 0.40 to catch more
        # volatile periods; 40% annualized vol is already severe)
        if vol > 0.40:
            return 'CRISIS'

        # High vol OR strong trend → HIGH_VOL_TRENDING
        # Relaxed from AND to OR: previous required both vol > 0.40 AND
        # trend > 0.60 which only triggered for 1% of data.
        if (vol > self.config.vol_threshold_high and
                trend > self.config.trend_threshold_weak):
            return 'HIGH_VOL_TRENDING'
        if (vol > self.config.vol_threshold_low and
                trend > self.config.trend_threshold_strong):
            return 'HIGH_VOL_TRENDING'

        # Low vol, weak trend → LIQUID_RANGEBOUND
        # Relaxed: removed hurst < 0.5 AND adf_p < 0.05 requirements;
        # low vol + weak trend is sufficient for rangebound classification.
        if (vol < self.config.vol_threshold_low and
                trend < self.config.trend_threshold_weak):
            if hurst < 0.5 and adf_p < 0.05:
                return 'LIQUID_RANGEBOUND'
            return 'MODERATE_RANGEBOUND'

        # Clear trend, medium vol → MODERATE_TRENDING
        if trend > self.config.trend_threshold_weak:
            return 'MODERATE_TRENDING'

        # Default: MODERATE_MIXED
        return 'MODERATE_MIXED'

    def _recommend_models(self, regime: str, features: Dict[str, float]) -> List[str]:
        """Recommend models based on regime."""
        recommendations = {
            'LIQUID_RANGEBOUND': ['garch', 'sarimax'],
            'MODERATE_RANGEBOUND': ['garch', 'sarimax', 'samossa'],
            'MODERATE_TRENDING': ['samossa', 'garch', 'patchtst'],
            'HIGH_VOL_TRENDING': ['samossa', 'patchtst', 'mssa_rl'],
            'CRISIS': ['garch', 'sarimax'],  # Defensive, stick to vol forecasting
            'MODERATE_MIXED': ['garch', 'samossa', 'sarimax'],
        }

        return recommendations.get(regime, ['garch', 'samossa'])

    def _regime_confidence(self, features: Dict[str, float]) -> float:
        """
        Calculate confidence in regime classification (0-1).
        Higher confidence when features are clear/extreme.
        """
        vol = features['realized_volatility']
        trend = features['trend_strength']

        # Guard against NaN/inf propagated from degenerate feature extraction
        if not np.isfinite(vol):
            vol = 0.0
        if not np.isfinite(trend):
            trend = 0.0

        # Strong signals = high confidence
        vol_signal = min(vol / 0.5, 1.0)  # 50% vol = max signal
        trend_signal = trend  # Already 0-1

        # Combine signals
        confidence = (vol_signal + trend_signal) / 2

        return float(np.clip(confidence, 0.3, 0.95))

    def get_preferred_candidates(
        self,
        regime_result: Dict,
        all_candidates: List[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        """
        Reorder candidates based on regime recommendations.
        Puts recommended models first.
        """
        recommended_models = set(regime_result['recommendations'])

        # Score candidates by alignment with recommendations
        scored_candidates = []
        for candidate in all_candidates:
            # Calculate overlap with recommendations
            candidate_models = set(candidate.keys())
            overlap = len(candidate_models & recommended_models)
            total_weight = sum(
                weight for model, weight in candidate.items()
                if model in recommended_models
            )

            score = overlap + total_weight
            scored_candidates.append((score, candidate))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        return [candidate for _, candidate in scored_candidates]
