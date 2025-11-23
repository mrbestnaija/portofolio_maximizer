"""
Rolling-window cross-validation helpers for the unified time-series forecaster.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

logger = logging.getLogger(__name__)


@dataclass
class RollingWindowCVConfig:
    """Configuration for the rolling-window validator."""

    min_train_size: int = 180
    horizon: int = 5
    step_size: int = 5
    max_folds: Optional[int] = None


class RollingWindowValidator:
    """
    Execute rolling-window cross-validation using the TimeSeriesForecaster.

    For each fold an expanding training window is fitted and evaluated on the
    subsequent out-of-sample horizon. Metrics are aggregated across folds to
    provide a robust view of generalisation performance.
    """

    def __init__(
        self,
        forecaster_config: Optional[TimeSeriesForecasterConfig] = None,
        cv_config: Optional[RollingWindowCVConfig] = None,
    ) -> None:
        self.forecaster_config = forecaster_config or TimeSeriesForecasterConfig()
        self.cv_config = cv_config or RollingWindowCVConfig()

    def _iter_folds(self, series: pd.Series) -> List[slice]:
        total_points = len(series)
        min_train = self.cv_config.min_train_size
        horizon = self.cv_config.horizon
        step = max(1, self.cv_config.step_size)
        if total_points < min_train + horizon:
            raise ValueError(
                f"Insufficient data for rolling CV (need >= {min_train + horizon},"
                f" received {total_points})"
            )
        folds: List[slice] = []
        fold_index = min_train
        while fold_index + horizon <= total_points:
            folds.append(slice(fold_index, fold_index + horizon))
            if self.cv_config.max_folds and len(folds) >= self.cv_config.max_folds:
                break
            fold_index += step
        return folds

    def run(
        self,
        price_series: pd.Series,
        returns_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        price_series = price_series.sort_index()
        folds = self._iter_folds(price_series)
        fold_results: List[Dict[str, Any]] = []

        for fold_number, fold_slice in enumerate(folds, start=1):
            train = price_series.iloc[: fold_slice.start]
            test = price_series.iloc[fold_slice]
            horizon = len(test)
            config = copy.deepcopy(self.forecaster_config)
            config.forecast_horizon = horizon
            rolling_forecaster = TimeSeriesForecaster(config=config)

            returns_subset = None
            if returns_series is not None:
                aligned_returns = returns_series.sort_index()
                returns_subset = aligned_returns.reindex(train.index).dropna()

            rolling_forecaster.fit(
                price_series=train,
                returns_series=returns_subset,
            )
            rolling_forecaster.forecast(steps=horizon)
            metrics = rolling_forecaster.evaluate(test)
            fold_results.append(
                {
                    "fold": fold_number,
                    "train_range": {
                        "start": str(train.index.min()),
                        "end": str(train.index.max()),
                    },
                    "test_range": {
                        "start": str(test.index.min()),
                        "end": str(test.index.max()),
                    },
                    "metrics": metrics,
                }
            )

        aggregate_metrics = self._aggregate_metrics(fold_results)
        payload = {
            "folds": fold_results,
            "aggregate_metrics": aggregate_metrics,
            "fold_count": len(fold_results),
            "horizon": self.cv_config.horizon,
        }
        logger.info(
            "Rolling-window CV completed (%s folds, horizon=%s)",
            payload["fold_count"],
            payload["horizon"],
        )
        return payload

    @staticmethod
    def _aggregate_metrics(
        folds: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        accumulator: Dict[str, Dict[str, List[float]]] = {}
        for fold in folds:
            for model, metrics in fold.get("metrics", {}).items():
                model_store = accumulator.setdefault(model, {})
                for key, value in metrics.items():
                    model_store.setdefault(key, []).append(float(value))
        aggregated: Dict[str, Dict[str, float]] = {}
        for model, metrics_dict in accumulator.items():
            aggregated[model] = {
                key: float(np.mean(values)) for key, values in metrics_dict.items()
            }
        return aggregated


__all__ = ["RollingWindowCVConfig", "RollingWindowValidator"]
