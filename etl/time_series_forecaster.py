"""
Backward-compatible shim for the forecasting stack.

Historical imports reference `etl.time_series_forecaster`. The actual
implementations now live inside the `forcester_ts` package so they can
be reused outside the ETL layer (dashboards, notebooks, services).
"""

from forcester_ts import (
    GARCHForecaster,
    MSSARLForecaster,
    SAMOSSAForecaster,
    SARIMAXForecaster,
    TimeSeriesForecaster,
    TimeSeriesForecasterConfig,
    RollingWindowValidator,
    RollingWindowCVConfig,
)

__all__ = [
    "GARCHForecaster",
    "MSSARLForecaster",
    "SAMOSSAForecaster",
    "SARIMAXForecaster",
    "TimeSeriesForecaster",
    "TimeSeriesForecasterConfig",
    "RollingWindowValidator",
    "RollingWindowCVConfig",
]
