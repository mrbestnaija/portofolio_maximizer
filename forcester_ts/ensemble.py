"""
Ensemble coordination utilities for blending multiple time-series forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)
EPSILON = 1e-9


@dataclass
class EnsembleConfig:
    enabled: bool = True
    confidence_scaling: bool = True
    candidate_weights: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"sarimax": 0.6, "samossa": 0.4},
            {"sarimax": 0.5, "samossa": 0.3, "mssa_rl": 0.2},
            {"sarimax": 0.5, "mssa_rl": 0.5},
            # Allow non-SARIMAX-dominated blends when diagnostics support it.
            {"samossa": 1.0},
            {"mssa_rl": 1.0},
            {"samossa": 0.7, "mssa_rl": 0.3},
        ]
    )
    minimum_component_weight: float = 0.05


class EnsembleCoordinator:
    """Selects weightings for combining individual model forecasts."""

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self.selected_weights: Dict[str, float] = {}
        self.selection_score: float = 0.0

    def select_weights(
        self,
        model_confidence: Dict[str, float],
    ) -> Tuple[Dict[str, float], float]:
        if not self.config.enabled or not self.config.candidate_weights:
            self.selected_weights = {}
            self.selection_score = 0.0
            return self.selected_weights, self.selection_score

        best_score = float("-inf")
        best_weights: Dict[str, float] = {}

        for candidate in self.config.candidate_weights:
            normalized = self._normalize(candidate)
            if not normalized:
                continue

            if self.config.confidence_scaling:
                scaled = {
                    model: weight * model_confidence.get(model, 0.5)
                    for model, weight in normalized.items()
                }
                normalized = self._normalize(scaled)
                if not normalized:
                    continue

            score = sum(
                normalized.get(model, 0.0) * model_confidence.get(model, 0.0)
                for model in normalized.keys()
            )
            if score > best_score:
                best_score = score
                best_weights = normalized

        self.selected_weights = best_weights
        self.selection_score = best_score if best_score != float("-inf") else 0.0
        return self.selected_weights, self.selection_score

    @staticmethod
    def _normalize(candidate: Dict[str, float]) -> Dict[str, float]:
        filtered = {k: max(float(v), 0.0) for k, v in candidate.items() if float(v) > 0.0}
        total = sum(filtered.values())
        if total == 0.0:
            return {}
        return {k: v / total for k, v in filtered.items()}

    def blend_forecasts(
        self,
        model_series: Dict[str, pd.Series],
        lower_bounds: Dict[str, Optional[pd.Series]],
        upper_bounds: Dict[str, Optional[pd.Series]],
    ) -> Optional[Dict[str, pd.Series]]:
        if not self.selected_weights:
            return None

        contributing = {m: s for m, s in model_series.items() if m in self.selected_weights and isinstance(s, pd.Series)}
        if not contributing:
            return None

        combined_df = pd.DataFrame(contributing)
        weights = pd.Series(self.selected_weights)
        aligned_weights = weights[combined_df.columns]

        forecast = combined_df.mul(aligned_weights, axis=1).sum(axis=1)

        def _blend_ci(bounds: Dict[str, Optional[pd.Series]]) -> Optional[pd.Series]:
            series_map = {
                model: bound for model, bound in bounds.items()
                if model in aligned_weights.index and isinstance(bound, pd.Series)
            }
            if not series_map:
                return None
            df = pd.DataFrame(series_map)
            return df.mul(aligned_weights[df.columns], axis=1).sum(axis=1)

        lower = _blend_ci(lower_bounds)
        upper = _blend_ci(upper_bounds)

        return {
            "forecast": forecast,
            "lower_ci": lower,
            "upper_ci": upper,
        }


def derive_model_confidence(
    summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Convert per-model diagnostics into comparable confidence scores.
    Scores blend information criteria, realised regression metrics,
    and one-sided F-tests (variance ratio) similar to Diebold-Mariano style
    screening used in production quant stacks.
    """

    confidence: Dict[str, float] = {}

    sarimax_summary = summaries.get("sarimax", {})
    samossa_summary = summaries.get("samossa", {})
    mssa_summary = summaries.get("mssa_rl", {})

    # Treat SAMOSSA as the primary TS baseline for variance tests when
    # available; fall back to SARIMAX metrics otherwise.
    samossa_metrics_baseline = samossa_summary.get("regression_metrics", {}) or {}
    sarimax_metrics_baseline = sarimax_summary.get("regression_metrics", {}) or {}
    baseline_metrics = samossa_metrics_baseline or sarimax_metrics_baseline

    def _combine_scores(*scores: Optional[float]) -> Optional[float]:
        valid = [float(np.clip(s, 0.0, 1.0)) for s in scores if s is not None]
        if not valid:
            return None
        return float(np.clip(np.mean(valid), 0.0, 1.0))

    def _score_from_metrics(metrics: Dict[str, float]) -> Optional[float]:
        if not metrics:
            return None
        components = []
        rmse_val = metrics.get("rmse")
        if rmse_val is not None:
            components.append(1.0 / (1.0 + float(rmse_val)))
        smape_val = metrics.get("smape")
        if smape_val is not None:
            components.append(max(0.0, 1.0 - min(float(smape_val), 2.0) / 2.0))
        te_val = metrics.get("tracking_error")
        if te_val is not None:
            components.append(1.0 / (1.0 + float(te_val)))
        da_val = metrics.get("directional_accuracy")
        if da_val is not None:
            da = float(da_val)
            # Treat 0.5 as random baseline; reward edges above that.
            da_score = max(0.0, (da - 0.5) / 0.5)
            components.append(da_score)
        if not components:
            return None
        return float(np.clip(np.mean(components), 0.0, 1.0))

    def _variance_test_score(metrics: Dict[str, float], baseline: Dict[str, float]) -> Optional[float]:
        if not metrics or not baseline:
            return None
        te = metrics.get("tracking_error")
        base_te = baseline.get("tracking_error")
        n = metrics.get("n_observations")
        base_n = baseline.get("n_observations")
        if not all([te, base_te, n, base_n]):
            return None
        te = float(te)
        base_te = float(base_te)
        f_stat = (te**2 + EPSILON) / (base_te**2 + EPSILON)
        dfn = max(int(n) - 1, 1)
        dfd = max(int(base_n) - 1, 1)
        if f_stat <= 1:
            p_value = scipy_stats.f.cdf(f_stat, dfn, dfd)
        else:
            p_value = 1 - scipy_stats.f.cdf(f_stat, dfn, dfd)
        return float(np.clip(1.0 - p_value, 0.0, 1.0))

    def _change_point_boost(summary: Dict[str, Any]) -> Optional[float]:
        density = summary.get("change_point_density", 0.0) or 0.0
        recent_days = summary.get("recent_change_point_days")
        if recent_days is None:
            return None
        if recent_days <= 7:
            recency = max(0.0, 1.0 - (recent_days / 7.0))
            boost = 0.2 + 0.6 * recency + 0.2 * min(density * 10.0, 1.0)
            return float(np.clip(boost, 0.0, 1.0))
        if density > 0.05:
            return float(np.clip(0.2 * density * 10.0, 0.0, 0.6))
        return None

    aic = sarimax_summary.get("aic")
    bic = sarimax_summary.get("bic")
    sarimax_score = None
    if aic is not None and bic is not None:
        sarimax_score = np.exp(-0.5 * (aic + bic) / max(abs(aic) + abs(bic), 1e-6))
    sarimax_metrics = sarimax_summary.get("regression_metrics", {}) or {}
    sarimax_score = _combine_scores(
        sarimax_score,
        _score_from_metrics(sarimax_metrics),
        _variance_test_score(sarimax_metrics, baseline_metrics)
        if baseline_metrics
        else None,
    )
    if sarimax_score is not None:
        confidence["sarimax"] = sarimax_score

    evr = samossa_summary.get("explained_variance_ratio")
    samossa_score = None
    if evr is not None:
        samossa_score = float(np.clip(evr, 0.0, 1.0))
    samossa_metrics = samossa_summary.get("regression_metrics", {}) or {}
    samossa_score = _combine_scores(
        samossa_score,
        _score_from_metrics(samossa_metrics),
        # SAMOSSA is treated as the primary baseline; skip variance
        # test against itself to avoid degenerate scores.
        None,
    )
    if samossa_score is not None:
        confidence["samossa"] = samossa_score

    baseline_var = mssa_summary.get("baseline_variance")
    mssa_score = None
    if baseline_var is not None:
        mssa_score = 1.0 / (1.0 + baseline_var)
    mssa_metrics = mssa_summary.get("regression_metrics", {}) or {}
    mssa_score = _combine_scores(
        mssa_score,
        _score_from_metrics(mssa_metrics),
        _variance_test_score(mssa_metrics, baseline_metrics)
        if baseline_metrics
        else None,
        _change_point_boost(mssa_summary),
    )
    if mssa_score is not None:
        confidence["mssa_rl"] = mssa_score

    # If SAMOSSA has strictly lower residual variance than SARIMAX but
    # ended up with a lower raw score (e.g. due to information-criteria
    # heuristics), gently bump it above SARIMAX so variance improvements
    # are reflected in ordering before normalisation.
    samossa_te = samossa_metrics.get("tracking_error")
    sarimax_te = sarimax_metrics.get("tracking_error")
    if (
        samossa_score is not None
        and sarimax_score is not None
        and isinstance(samossa_te, (int, float))
        and isinstance(sarimax_te, (int, float))
        and samossa_te < sarimax_te
        and samossa_score <= sarimax_score
    ):
        samossa_score = min(1.0, sarimax_score + 0.05)
        confidence["samossa"] = samossa_score

    # Normalise to 0..1 range and avoid zero weights
    if confidence:
        values = np.array(list(confidence.values()), dtype=float)
        max_val = values.max()
        min_val = values.min()
        if max_val > min_val:
            normalized = (values - min_val) / (max_val - min_val + EPSILON)
        else:
            normalized = np.ones_like(values)
        confidence = {
            model: float(np.clip(val, 0.0, 1.0))
            for model, val in zip(confidence.keys(), normalized)
        }
    return confidence
