"""
Ticker discovery subsystem for Portfolio Maximizer.

This package exposes a lightweight pipeline for sourcing, validating,
and maintaining a universe of tradable tickers.  The implementation is
designed to be backward compatible – importing the package has no side
effects and the higher level orchestration only interacts with it when
explicitly enabled.
"""
from .base_ticker_loader import BaseTickerLoader
from .alpha_vantage_loader import AlphaVantageTickerLoader
from .ticker_validator import TickerValidator
from .ticker_universe import TickerUniverseManager

__all__ = [
    "BaseTickerLoader",
    "AlphaVantageTickerLoader",
    "TickerValidator",
    "TickerUniverseManager",
]
