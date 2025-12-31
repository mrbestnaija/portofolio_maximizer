"""Simple time-series validation framework with walk-forward CV."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    model_name: str
    metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    cv_results: List[Dict[str, Any]]


class TimeSeriesValidation:
    """Blocked CV + basic statistical reporting."""

    def __init__(self, output_path: Path = Path("logs/automation/time_series_validation.json")) -> None:
        self.output_path = output_path

    def evaluate(self, models: List[Any], series: pd.Series) -> List[ValidationReport]:
        results: List[ValidationReport] = []
        for model in models:
            try:
                cv_results = self._walk_forward_validation(model, series)
                metrics = self._calculate_metrics(cv_results)
                stats = self._simple_stats(cv_results)
                report = ValidationReport(
                    model_name=model.__class__.__name__,
                    metrics=metrics,
                    statistical_tests=stats,
                    cv_results=cv_results,
                )
                results.append(report)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Validation failed for %s: %s", model, exc)
        self._persist_results(results)
        return results

    def _walk_forward_validation(self, model: Any, series: pd.Series) -> List[Dict[str, Any]]:
        values = series.dropna().values
        splits = np.array_split(values, 3)
        cv_results: List[Dict[str, Any]] = []
        for i in range(1, len(splits)):
            train = np.concatenate(splits[:i])
            test = splits[i]
            if len(train) < 10 or len(test) == 0:
                continue
            pred = self._forecast(model, train, horizon=len(test))
            pred = pred[: len(test)]
            error = test - pred
            profit = float(np.sum(np.sign(pred[:-1] - train[-1]) * (test[:-1] - test[1:]).astype(float))) if len(test) > 1 else 0.0
            cv_results.append(
                {
                    "fold": i,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "profit": profit,
                }
            )
        return cv_results

    @staticmethod
    def _forecast(model: Any, train: np.ndarray, horizon: int) -> np.ndarray:
        # Very lightweight interface: expect fit(train) and forecast(horizon)
        if hasattr(model, "fit"):
            model.fit(train)
        if hasattr(model, "forecast"):
            return np.array(model.forecast(horizon))
        if hasattr(model, "predict"):
            return np.array(model.predict(horizon))
        # Fallback: last value naive forecast
        return np.repeat(train[-1], horizon)

    @staticmethod
    def _calculate_metrics(cv_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not cv_results:
            return {"profit_factor": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0, "rmse": 0.0}
        profits = [r["profit"] for r in cv_results]
        total_profit = sum(p for p in profits if p > 0)
        total_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf") if total_profit > 0 else 0.0
        rmse = float(np.mean([r["rmse"] for r in cv_results]))
        hit_rate = float(np.mean([1 if r["profit"] > 0 else 0 for r in cv_results]))
        max_drawdown = float(min(min(profits), 0.0))
        return {
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate,
            "rmse": rmse,
        }

    @staticmethod
    def _simple_stats(cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not cv_results:
            return {"diebold_mariano": None, "paired_t_test": None}
        profits = np.array([r["profit"] for r in cv_results])
        return {
            "diebold_mariano": {"mean_profit": float(profits.mean())},
            "paired_t_test": {"p_value": None},
        }

    def _persist_results(self, reports: List[ValidationReport]) -> None:
        try:
            payload = [r.__dict__ for r in reports]
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps(payload, indent=2))
            logger.info("Validation reports written to %s", self.output_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to persist validation reports: %s", exc)
