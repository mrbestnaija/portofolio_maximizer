"""
Models package - Signal generation and routing

This package contains:
- TimeSeriesSignalGenerator: Converts time series forecasts to trading signals
- SignalRouter: Routes signals with Time Series as default, LLM as fallback
- SignalAdapter: Unified signal interface for backward compatibility
"""

from models.time_series_signal_generator import (
    TimeSeriesSignal,
    TimeSeriesSignalGenerator
)

from models.signal_router import (
    SignalBundle,
    SignalRouter
)

from models.signal_adapter import (
    UnifiedSignal,
    SignalAdapter
)

__all__ = [
    'TimeSeriesSignal',
    'TimeSeriesSignalGenerator',
    'SignalBundle',
    'SignalRouter',
    'UnifiedSignal',
    'SignalAdapter'
]

