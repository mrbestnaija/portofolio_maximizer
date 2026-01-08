"""
Market Regime Detection and Adaptation
--------------------------------------

Statistical regime detection helpers for dynamic strategy adjustment.

This module is a concrete implementation of the design sketched in
Documentation/OPTIMIZATION_IMPLEMENTATION_PLAN.md (Phase 2 – Regime Detection
and Adaptation) and is intended to be reused by:

- Time-series hyper-parameter search / model-profile routing,
- Risk and sizing logic that needs regime-aware multipliers,
- Research notebooks analysing regime transitions.

It does not mutate any global state and makes no assumptions about callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RegimeState:
    """Market regime state."""

    regime_type: str  # 'bull', 'bear', 'sideways', 'high_vol', 'low_vol', ...
    confidence: float  # 0–1
    duration: int  # Points in current regime
    transition_probability: float  # Approximate probability of regime change


class RegimeDetector:
    """Statistical market regime detection."""

    def __init__(self, window_size: int = 60, significance_level: float = 0.05) -> None:
        self.window_size = int(window_size)
        self.significance_level = float(significance_level)
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []

    def detect_volatility_regime(self, returns: np.ndarray) -> RegimeState:
        """Detect volatility regime using rolling standard deviation and a t-test."""
        arr = np.asarray(returns, dtype=float)
        if arr.size < self.window_size * 2:
            return RegimeState("insufficient_data", 0.0, int(arr.size), 0.0)

        series = pd.Series(arr)
        rolling_vol = series.rolling(self.window_size).std()
        current_vol = float(rolling_vol.iloc[-1])
        historical_vol = rolling_vol.iloc[-self.window_size - 1 : -1].dropna()
        if historical_vol.empty:
            return RegimeState("insufficient_data", 0.0, int(arr.size), 0.0)

        # Statistical test for regime change
        _, p_value = stats.ttest_1samp(historical_vol, current_vol)

        if p_value < self.significance_level:
            if current_vol > float(historical_vol.mean()):
                regime_type = "high_vol"
                confidence = 1.0 - float(p_value)
            else:
                regime_type = "low_vol"
                confidence = 1.0 - float(p_value)
        else:
            regime_type = "normal_vol"
            confidence = float(p_value)

        state = RegimeState(
            regime_type=regime_type,
            confidence=confidence,
            duration=self._calculate_regime_duration(regime_type),
            transition_probability=float(p_value),
        )
        self._update_history(state)
        return state

    def detect_trend_regime(self, returns: np.ndarray) -> RegimeState:
        """Detect trend regime using rolling mean returns and a t-test."""
        arr = np.asarray(returns, dtype=float)
        if arr.size < self.window_size * 2:
            return RegimeState("insufficient_data", 0.0, int(arr.size), 0.0)

        series = pd.Series(arr)
        rolling_mean = series.rolling(self.window_size).mean()
        current_mean = float(rolling_mean.iloc[-1])
        historical_mean = rolling_mean.iloc[-self.window_size - 1 : -1].dropna()
        if historical_mean.empty:
            return RegimeState("insufficient_data", 0.0, int(arr.size), 0.0)

        _, p_value = stats.ttest_1samp(historical_mean, current_mean)

        if p_value < self.significance_level:
            if current_mean > float(historical_mean.mean()):
                regime_type = "bull"
                confidence = 1.0 - float(p_value)
            else:
                regime_type = "bear"
                confidence = 1.0 - float(p_value)
        else:
            regime_type = "sideways"
            confidence = float(p_value)

        state = RegimeState(
            regime_type=regime_type,
            confidence=confidence,
            duration=self._calculate_regime_duration(regime_type),
            transition_probability=float(p_value),
        )
        self._update_history(state)
        return state

    def adapt_strategy_parameters(self, signal: Dict, regime_state: RegimeState) -> Dict:
        """
        Adapt strategy parameters based on current regime.

        This is a generic helper; callers remain responsible for ensuring that
        `signal` contains `confidence` and `position_size` fields and for
        enforcing any additional risk limits.
        """
        regime_multipliers = {
            "bull": {"confidence_multiplier": 1.2, "position_multiplier": 1.1},
            "bear": {"confidence_multiplier": 0.8, "position_multiplier": 0.9},
            "sideways": {"confidence_multiplier": 1.0, "position_multiplier": 1.0},
            "high_vol": {"confidence_multiplier": 0.7, "position_multiplier": 0.8},
            "low_vol": {"confidence_multiplier": 1.3, "position_multiplier": 1.2},
            "normal_vol": {"confidence_multiplier": 1.0, "position_multiplier": 1.0},
        }

        default_multipliers = {"confidence_multiplier": 1.0, "position_multiplier": 1.0}
        multiplier = regime_multipliers.get(regime_state.regime_type, default_multipliers)

        adapted = dict(signal)
        try:
            adapted_conf = float(adapted.get("confidence", 0.0))
            adapted_size = float(adapted.get("position_size", 0.0))
        except (TypeError, ValueError):
            adapted_conf = float(adapted.get("confidence") or 0.0)
            adapted_size = float(adapted.get("position_size") or 0.0)

        adapted["confidence"] = adapted_conf * float(multiplier["confidence_multiplier"])
        adapted["position_size"] = adapted_size * float(multiplier["position_multiplier"])
        adapted["regime"] = regime_state.regime_type
        adapted["regime_confidence"] = regime_state.confidence

        return adapted

    def _calculate_regime_duration(self, regime_type: str) -> int:
        """Calculate duration of current regime based on history."""
        if not self.regime_history:
            return 1

        duration = 1
        for state in reversed(self.regime_history):
            if state.regime_type == regime_type:
                duration += 1
            else:
                break
        return duration

    def _update_history(self, state: RegimeState) -> None:
        """Append the new regime state to history and track as current."""
        self.current_regime = state
        self.regime_history.append(state)


__all__ = ["RegimeState", "RegimeDetector"]

