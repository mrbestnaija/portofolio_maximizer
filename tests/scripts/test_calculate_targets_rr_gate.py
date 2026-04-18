"""Tests for R:R ≥ 2:1 gate in TimeSeriesSignalGenerator._calculate_targets()

Pins three invariants:
1. When forecast target is too close (R:R < 2:1), target is extended to entry + 2×stop_dist
2. When forecast target already satisfies R:R ≥ 2:1, it is preserved unchanged
3. HOLD action returns (None, None) — no target or stop
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


def _make_generator():
    """Minimal TimeSeriesSignalGenerator with only _calculate_targets wired."""
    from models.time_series_signal_generator import TimeSeriesSignalGenerator
    cfg = MagicMock()
    cfg.get = MagicMock(return_value=None)
    with patch.object(TimeSeriesSignalGenerator, "__init__", lambda self, *a, **kw: None):
        gen = TimeSeriesSignalGenerator.__new__(TimeSeriesSignalGenerator)
    # Patch _compute_atr to return a known ATR value
    gen._compute_atr = MagicMock(return_value=None)
    return gen


class TestRRGate:

    def test_low_rr_target_extended_to_2x_stop_dist_buy(self):
        """BUY: forecast target only 0.5% away, stop 3% away — must extend to 6% away."""
        gen = _make_generator()
        current = 100.0
        # stop = 100 * (1 - 0.03) = 97.0 → stop_dist = 3.0
        # forecast target = 100.5 → target_dist = 0.5 < 2×3.0=6.0 → must extend to 106.0
        gen._compute_atr = MagicMock(return_value=None)

        with patch.object(type(gen), "_compute_atr", return_value=None):
            # Use volatility fallback: vol=0.06, stop_loss_pct = max(0.015, min(0.05, 0.06*0.5))=0.03
            target, stop = gen._calculate_targets(
                current_price=current,
                forecast_price=100.5,
                volatility=0.06,
                action="BUY",
                market_data=None,
            )

        stop_dist = abs(current - stop)
        target_dist = abs(target - current)
        assert target_dist >= 2.0 * stop_dist - 1e-9, (
            f"R:R gate failed: target_dist={target_dist:.4f}, 2×stop_dist={2*stop_dist:.4f}"
        )
        assert target > current, "BUY target must be above entry"

    def test_good_rr_target_preserved(self):
        """BUY: forecast target 10% away, stop 3% away — R:R = 3.33:1, must not be changed."""
        gen = _make_generator()
        current = 100.0
        forecast = 110.0  # 10% above entry

        with patch.object(type(gen), "_compute_atr", return_value=None):
            target, stop = gen._calculate_targets(
                current_price=current,
                forecast_price=forecast,
                volatility=0.06,  # stop_loss_pct=0.03 → stop=97
                action="BUY",
                market_data=None,
            )

        assert target == pytest.approx(forecast, rel=1e-6), (
            "Target with R:R ≥ 2:1 must not be modified"
        )

    def test_hold_returns_none_none(self):
        """HOLD action must return (None, None) — no target, no stop."""
        gen = _make_generator()
        target, stop = gen._calculate_targets(
            current_price=100.0,
            forecast_price=105.0,
            volatility=0.03,
            action="HOLD",
            market_data=None,
        )
        assert target is None
        assert stop is None

    def test_sell_target_extended_below_entry(self):
        """SELL: forecast target only slightly below entry — must extend downward to 2× stop_dist."""
        gen = _make_generator()
        current = 100.0
        # volatility=0.06 → stop_loss_pct=0.03 for SELL: stop = 103.0, stop_dist=3.0
        # forecast target = 99.5 → target_dist=0.5 < 6.0 → extend to 94.0
        with patch.object(type(gen), "_compute_atr", return_value=None):
            target, stop = gen._calculate_targets(
                current_price=current,
                forecast_price=99.5,
                volatility=0.06,
                action="SELL",
                market_data=None,
            )

        stop_dist = abs(stop - current)
        target_dist = abs(current - target)
        assert target_dist >= 2.0 * stop_dist - 1e-9
        assert target < current, "SELL target must be below entry"

    def test_rr_gate_with_atr_stop(self):
        """ATR-based stop: stop_dist = ATR×2. R:R gate uses same 2:1 minimum."""
        gen = _make_generator()
        current = 200.0
        atr = 5.0  # stop_loss_pct = max(5*2/200, 0.015) = max(0.05, 0.015) = 0.05 → stop=190
        # Forecast target = 205 → target_dist=5 < 2×10=20 → extend to 220.0

        # Override instance-level mock to return known ATR
        gen._compute_atr = MagicMock(return_value=atr)

        target, stop = gen._calculate_targets(
            current_price=current,
            forecast_price=205.0,
            volatility=None,
            action="BUY",
            market_data=None,
        )

        stop_dist = abs(current - stop)
        target_dist = abs(target - current)
        assert target_dist >= 2.0 * stop_dist - 1e-9
        assert stop == pytest.approx(current - atr * 2.0, rel=1e-6)
