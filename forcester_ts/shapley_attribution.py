"""
shapley_attribution.py
----------------------
Predictor-Based Shapley Values (PBSV) for ensemble forecast error attribution.

Decomposes out-of-sample prediction error (MAE, MSE, or pinball loss) across
ensemble components (GARCH, SAMOSSA, MSSA-RL) using exact Shapley enumeration.

With N=3 models, 2^3=8 subsets are evaluated — tractable without approximation.

Usage:
    from forcester_ts.shapley_attribution import ShapleyAttributor
    sa = ShapleyAttributor()
    svs = sa.compute(
        component_forecasts={"garch": arr1, "samossa": arr2, "mssa_rl": arr3},
        weights={"garch": 0.5, "samossa": 0.3, "mssa_rl": 0.2},
        actual=actual_returns,
        loss_fn="mae",
    )
    # svs = {"garch": 0.08, "samossa": -0.02, "mssa_rl": 0.14}
"""
from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _compute_loss(
    actual: np.ndarray,
    forecast: np.ndarray,
    loss_fn: str,
    tau: float,
) -> float:
    """Compute scalar loss between actual and forecast."""
    n = min(len(actual), len(forecast))
    if n == 0:
        return float("nan")
    y = actual[:n]
    f = forecast[:n]
    if loss_fn == "mse":
        return float(np.mean((y - f) ** 2))
    elif loss_fn == "pinball":
        diff = y - f
        loss = diff * (tau - (diff < 0).astype(float))
        return float(np.mean(loss))
    else:  # default: mae
        return float(np.mean(np.abs(y - f)))


def _subset_forecast(
    component_forecasts: dict[str, np.ndarray],
    weights: dict[str, float],
    subset: tuple[str, ...],
    n: int,
) -> np.ndarray:
    """
    Compute the weighted-average forecast using only the models in `subset`.
    Renormalizes weights within the subset to sum to 1.
    """
    if not subset:
        # Empty subset — return grand mean of all components as baseline
        all_fc = np.array([component_forecasts[k][:n] for k in component_forecasts])
        return np.mean(all_fc, axis=0)

    raw_weights = {k: weights.get(k, 1.0 / len(subset)) for k in subset}
    total_w = sum(raw_weights.values())
    if total_w <= 0:
        total_w = 1.0

    fc = np.zeros(n, dtype=float)
    for k in subset:
        w = raw_weights[k] / total_w
        arr = component_forecasts[k][:n]
        fc += w * arr

    return fc


class ShapleyAttributor:
    """
    Exact Shapley Value decomposition of OOS prediction error for ensemble models.

    For N models, enumerates all 2^N power-set subsets.
    Shapley value for model i = weighted average of marginal contributions
    across all subsets S not containing i vs S ∪ {i}.

    Efficiency axiom: sum(shapley_values) ≈ loss(ensemble) - loss(grand_mean)
    """

    def compute(
        self,
        component_forecasts: dict[str, np.ndarray],
        weights: dict[str, float],
        actual: np.ndarray,
        loss_fn: str = "mae",
        tau: float = 0.5,
    ) -> dict[str, float]:
        """
        Compute Shapley value per model.

        Args:
            component_forecasts: {model_name: forecast_array} — all must cover actual
            weights: ensemble weights per model (will be renormalized within each subset)
            actual: realized values (returns, prices, etc.)
            loss_fn: "mae" | "mse" | "pinball"
            tau: quantile level used when loss_fn="pinball"

        Returns:
            dict: {model_name: shapley_value}
            Positive value = model contributes to prediction error.
            Negative value = model reduces prediction error vs grand mean.
        """
        players = list(component_forecasts.keys())
        N = len(players)

        if N == 0:
            return {}

        y = np.asarray(actual, dtype=float)
        # Align all forecasts to shortest length
        n = len(y)
        for arr in component_forecasts.values():
            n = min(n, len(arr))

        if n == 0:
            logger.warning("ShapleyAttributor.compute: zero-length arrays")
            return {p: float("nan") for p in players}

        # Pre-compute loss for all 2^N subsets (including empty set)
        subset_losses: dict[frozenset, float] = {}

        for r in range(N + 1):
            for combo in combinations(players, r):
                key = frozenset(combo)
                fc = _subset_forecast(component_forecasts, weights, combo, n)
                subset_losses[key] = _compute_loss(y, fc, loss_fn, tau)

        # Shapley formula (exact):
        # phi_i = sum over S not containing i of
        #         [|S|! * (N-|S|-1)! / N!] * [v(S ∪ {i}) - v(S)]
        shapley: dict[str, float] = {}
        n_fact = math.factorial(N)

        for player in players:
            phi = 0.0
            others = [p for p in players if p != player]

            for r in range(N):  # |S| = 0 .. N-1
                for combo in combinations(others, r):
                    S = frozenset(combo)
                    S_with = S | {player}
                    s_size = len(S)
                    weight = math.factorial(s_size) * math.factorial(N - s_size - 1) / n_fact
                    marginal = subset_losses.get(S_with, 0.0) - subset_losses.get(S, 0.0)
                    phi += weight * marginal

            shapley[player] = float(phi)

        logger.debug(
            "ShapleyAttributor.compute loss_fn=%s N=%d players=%s values=%s",
            loss_fn, N, players, {k: f"{v:.4f}" for k, v in shapley.items()},
        )
        return shapley

    def aggregate_by_regime(
        self,
        fold_results: list[dict],
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate Shapley values by detected regime across walk-forward folds.

        Args:
            fold_results: list of dicts, each with:
                {"regime": str, "shapley": {model_name: float}}

        Returns:
            {regime: {model_name: mean_shapley_value}}
        """
        regime_buckets: dict[str, dict[str, list[float]]] = {}

        for fold in fold_results:
            regime = fold.get("regime") or "UNKNOWN"
            shapley = fold.get("shapley", {})
            if regime not in regime_buckets:
                regime_buckets[regime] = {}
            for model, val in shapley.items():
                if val == val:  # NaN guard
                    regime_buckets[regime].setdefault(model, []).append(val)

        result: dict[str, dict[str, float]] = {}
        for regime, model_vals in regime_buckets.items():
            result[regime] = {
                model: float(np.mean(vals))
                for model, vals in model_vals.items()
                if vals
            }

        return result

    def dominant_driver(
        self,
        shapley_values: dict[str, float],
        ensemble_loss: float,
        threshold: float = 0.05,
    ) -> str | None:
        """
        Return the name of the dominant error driver if any model's |shapley|
        exceeds threshold * ensemble_loss.

        Returns None if no dominant driver is found.
        """
        if not shapley_values or ensemble_loss != ensemble_loss:
            return None

        abs_threshold = abs(threshold * ensemble_loss)
        candidates = {
            k: v for k, v in shapley_values.items()
            if abs(v) > abs_threshold
        }
        if not candidates:
            return None

        return max(candidates, key=lambda k: abs(candidates[k]))
