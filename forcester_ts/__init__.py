"""
Forcester TS package

Centralises time-series forecasting models (SARIMAX, GARCH, SAMOSSA, mSSA-RL)
so they can be shared across ETL and analytics layers.
"""

from .sarimax import SARIMAXForecaster  # noqa: F401
from .garch import GARCHForecaster  # noqa: F401
from .samossa import SAMOSSAForecaster  # noqa: F401
from .mssa_rl import MSSARLForecaster  # noqa: F401
from .forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig  # noqa: F401
from .ensemble import EnsembleConfig  # noqa: F401
from .metrics import compute_regression_metrics  # noqa: F401
from .cross_validation import RollingWindowValidator, RollingWindowCVConfig  # noqa: F401

__all__ = [
    "SARIMAXForecaster",
    "GARCHForecaster",
    "SAMOSSAForecaster",
    "MSSARLForecaster",
    "TimeSeriesForecaster",
    "TimeSeriesForecasterConfig",
    "EnsembleConfig",
    "compute_regression_metrics",
    "RollingWindowValidator",
    "RollingWindowCVConfig",
]
