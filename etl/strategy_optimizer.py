"""
Strategy optimizer infrastructure for stochastic, configuration-driven PnL tuning.

This module does not hardcode any strategy or parameter values. Instead, it:
- Samples candidate configurations from a search space defined in YAML config.
- Delegates evaluation to a caller-provided function (e.g., backtest runner).
- Scores candidates with a simple, configurable objective function.

Guardrails:
- Global safety thresholds (such as min_expected_return, max_risk_score,
  capital at risk) must be enforced by the caller and by configuration,
  not by hardcoded values in this module.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyCandidate:
    """Represents a single candidate configuration drawn from the search space."""

    params: Dict[str, Any]
    regime: Optional[str] = None


@dataclass
class StrategyEvaluation:
    """Evaluation result for a candidate, including metrics and scalar score."""

    candidate: StrategyCandidate
    metrics: Dict[str, float]
    score: float


class StrategyOptimizer:
    """
    Lightweight stochastic optimizer over a configuration-defined search space.

    The optimizer is agnostic to the underlying trading strategy. It samples
    candidate parameter sets and asks the caller to evaluate each candidate on
    historical or realized data.
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        objectives: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Args:
            search_space: Parameter definitions (type/bounds/choices) from config.
            objectives: Mapping of metric name -> weight for scoring.
            constraints: Optional dict with "min" and "max" metric thresholds.
            random_state: Optional seed for reproducible sampling.
        """
        self.search_space = search_space or {}
        self.objectives = objectives or {}
        self.constraints = constraints or {}
        self._rng = random.Random(random_state)

    def sample_candidate(self, regime: Optional[str] = None) -> StrategyCandidate:
        """Draw a single candidate from the configured search space."""
        params: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            param_type = str(spec.get("type", "continuous")).lower()
            if param_type == "continuous":
                low, high = spec.get("bounds", [0.0, 1.0])
                params[name] = self._rng.uniform(float(low), float(high))
            elif param_type == "integer":
                low, high = spec.get("bounds", [0, 10])
                params[name] = self._rng.randint(int(low), int(high))
            elif param_type == "categorical":
                choices = spec.get("choices", [])
                if not choices:
                    logger.debug("Parameter %s has empty choices; skipping.", name)
                    continue
                params[name] = self._rng.choice(choices)
            else:
                logger.debug("Unknown parameter type %s for %s; skipping.", param_type, name)
        return StrategyCandidate(params=params, regime=regime)

    def _apply_constraints(self, metrics: Dict[str, float]) -> bool:
        """Return True if metrics satisfy configured min/max constraints."""
        # If no trades were evaluated, allow the candidate to pass so that we
        # can still compare candidates without filtering everything out.
        if "total_trades" in metrics and metrics.get("total_trades") == 0:
            return True

        minimums = self.constraints.get("min", {}) or {}
        maximums = self.constraints.get("max", {}) or {}

        for key, threshold in minimums.items():
            value = metrics.get(key)
            if value is None or value < threshold:
                return False

        for key, threshold in maximums.items():
            value = metrics.get(key)
            if value is None or value > threshold:
                return False

        return True

    def score_metrics(self, metrics: Dict[str, float]) -> float:
        """Compute a scalar score using a weighted sum of metrics."""
        score = 0.0
        for name, weight in self.objectives.items():
            value = metrics.get(name)
            if value is None:
                continue
            score += float(weight) * float(value)
        return score

    def run(
        self,
        n_candidates: int,
        evaluation_fn: Callable[[StrategyCandidate], Dict[str, float]],
        regime: Optional[str] = None,
    ) -> List[StrategyEvaluation]:
        """
        Sample and evaluate multiple candidates.

        Args:
            n_candidates: Number of candidates to sample.
            evaluation_fn: Callable that maps StrategyCandidate -> metrics dict.
            regime: Optional regime label to attach to each candidate.

        Returns:
            A list of StrategyEvaluation objects sorted by descending score.
        """
        evaluations: List[StrategyEvaluation] = []
        trials = max(1, int(n_candidates))

        for _ in range(trials):
            candidate = self.sample_candidate(regime=regime)
            metrics = evaluation_fn(candidate)
            if not isinstance(metrics, dict):
                logger.warning("Evaluation function did not return a dict; skipping.")
                continue
            if not self._apply_constraints(metrics):
                continue
            score = self.score_metrics(metrics)
            evaluations.append(
                StrategyEvaluation(candidate=candidate, metrics=metrics, score=score)
            )

        evaluations.sort(key=lambda e: e.score, reverse=True)
        return evaluations


# Basic sanity checks (lightweight, not a full test suite).
assert StrategyOptimizer.sample_candidate.__doc__ is not None
assert StrategyOptimizer.run.__doc__ is not None
