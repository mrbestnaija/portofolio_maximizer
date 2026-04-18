"""Tests for trailing stop ratchet in PaperTradingEngine._evaluate_exit_reason()

Invariants:
1. Break-even ratchet: when profit ≥ 1×ATR, stop moves to entry price (never below)
2. Profit-lock ratchet: when profit ≥ 1.5×ATR, stop moves to entry + 0.5×ATR
3. Stop never moves backward (ratchet is one-directional)
4. Short positions ratchet symmetrically (stop moves DOWN toward entry)
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


def _make_engine_with_position(
    *,
    ticker="NVDA",
    shares=10,
    entry_price=200.0,
    initial_stop=190.0,   # stop_dist=10, effective_atr=5
    target_price=230.0,
    max_holding_days=15,
):
    """Build a minimal PTE with one long position loaded into portfolio."""
    from execution.paper_trading_engine import PaperTradingEngine, Portfolio
    from unittest.mock import MagicMock

    engine = MagicMock(spec=PaperTradingEngine)
    portfolio = Portfolio(cash=25000.0)
    portfolio.positions[ticker] = shares
    portfolio.entry_prices[ticker] = entry_price
    portfolio.stop_losses[ticker] = initial_stop
    portfolio.target_prices[ticker] = target_price
    portfolio.max_holding_days[ticker] = max_holding_days
    portfolio.holding_bars[ticker] = 1
    portfolio.entry_timestamps[ticker] = datetime(2026, 1, 1, tzinfo=timezone.utc)
    portfolio.entry_bar_timestamps[ticker] = datetime(2026, 1, 1, tzinfo=timezone.utc)
    portfolio.last_bar_timestamps[ticker] = datetime(2026, 1, 1, tzinfo=timezone.utc)

    engine.portfolio = portfolio
    # Wire the real method
    engine._evaluate_exit_reason = PaperTradingEngine._evaluate_exit_reason.__get__(engine)
    return engine, portfolio


class TestTrailingStopRatchet:

    def test_breakeven_ratchet_fires_at_1x_atr(self):
        """At profit = 1×ATR (effective_atr=5, profit=5), stop should ratchet to entry_price=200."""
        engine, portfolio = _make_engine_with_position(
            entry_price=200.0, initial_stop=190.0  # stop_dist=10, effective_atr=5
        )
        # current_price = entry + 1×ATR = 200 + 5 = 205
        result = engine._evaluate_exit_reason(
            ticker="NVDA",
            shares=10,
            current_price=205.0,
            as_of=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
        # Stop should have moved to entry_price (200.0), no exit triggered yet
        assert portfolio.stop_losses["NVDA"] == pytest.approx(200.0, abs=1e-6), (
            f"Break-even ratchet failed: stop={portfolio.stop_losses['NVDA']}"
        )
        assert result is None  # price 205 > new_stop 200, no exit

    def test_profit_lock_ratchet_fires_at_1p5x_atr(self):
        """At profit = 1.5×ATR (profit=7.5), stop locks at entry + 0.5×ATR = 202.5."""
        engine, portfolio = _make_engine_with_position(
            entry_price=200.0, initial_stop=190.0
        )
        # current_price = entry + 1.5×ATR = 200 + 7.5 = 207.5
        engine._evaluate_exit_reason(
            ticker="NVDA",
            shares=10,
            current_price=207.5,
            as_of=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
        # Stop should lock at 200 + 0.5×5 = 202.5
        assert portfolio.stop_losses["NVDA"] == pytest.approx(202.5, abs=1e-6), (
            f"Profit-lock ratchet failed: stop={portfolio.stop_losses['NVDA']}"
        )

    def test_stop_never_moves_backward(self):
        """Once stop is ratcheted to break-even, it must not revert if price pulls back."""
        engine, portfolio = _make_engine_with_position(
            entry_price=200.0, initial_stop=190.0
        )
        # First call: ratchet to break-even at 205
        engine._evaluate_exit_reason(
            ticker="NVDA", shares=10, current_price=205.0,
            as_of=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
        stop_after_first = portfolio.stop_losses["NVDA"]

        # Second call: price pulls back to 201 (still above new stop but below initial ratchet level)
        engine._evaluate_exit_reason(
            ticker="NVDA", shares=10, current_price=201.0,
            as_of=datetime(2026, 1, 4, tzinfo=timezone.utc),
        )
        stop_after_pullback = portfolio.stop_losses["NVDA"]

        assert stop_after_pullback >= stop_after_first, (
            f"Stop moved backward: {stop_after_first} -> {stop_after_pullback}"
        )

    def test_short_position_ratchets_downward(self):
        """SHORT: when price falls (profit), stop should move DOWN toward entry (break-even)."""
        engine, portfolio = _make_engine_with_position(
            ticker="AAPL",
            shares=-10,
            entry_price=200.0,
            initial_stop=210.0,  # short stop above entry; stop_dist=10, effective_atr=5
            target_price=170.0,
            max_holding_days=15,
        )
        # current_price = entry - 1×ATR = 200 - 5 = 195 → profit = 5
        engine._evaluate_exit_reason(
            ticker="AAPL",
            shares=-10,
            current_price=195.0,
            as_of=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
        # Stop should move to entry_price = 200 (break-even for short)
        assert portfolio.stop_losses["AAPL"] == pytest.approx(200.0, abs=1e-6), (
            f"Short break-even ratchet failed: stop={portfolio.stop_losses['AAPL']}"
        )

    def test_no_ratchet_below_1x_atr_profit(self):
        """Below 1×ATR profit, stop must remain at original level."""
        engine, portfolio = _make_engine_with_position(
            entry_price=200.0, initial_stop=190.0
        )
        original_stop = portfolio.stop_losses["NVDA"]
        # current_price = 203 → profit=3, ATR=5 → profit < 1×ATR → no ratchet
        engine._evaluate_exit_reason(
            ticker="NVDA", shares=10, current_price=203.0,
            as_of=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
        assert portfolio.stop_losses["NVDA"] == pytest.approx(original_stop, abs=1e-6), (
            "Stop should not ratchet when profit < 1×ATR"
        )
