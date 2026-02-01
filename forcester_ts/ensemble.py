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
CANONICAL_MODEL_MAP = {
    "SARIMAX": "sarimax",
    "GARCH": "garch",
    "SAMOSSA": "samossa",
    "MSSA_RL": "mssa_rl",
}
TRACKED_MODELS = {"sarimax", "garch", "samossa", "mssa_rl"}


@dataclass
class EnsembleConfig:
    enabled: bool = True
    confidence_scaling: bool = True
    regime_detection_enabled: bool = True  # Phase 7.4: Enable regime-aware selection
    candidate_weights: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"garch": 0.85, "samossa": 0.10, "mssa_rl": 0.05},
            {"garch": 0.70, "samossa": 0.20, "mssa_rl": 0.10},
            {"garch": 0.60, "samossa": 0.25, "mssa_rl": 0.15},
            {"samossa": 0.60, "mssa_rl": 0.40},
            {"samossa": 0.45, "garch": 0.35, "mssa_rl": 0.20},
            {"garch": 1.0},
            {"samossa": 1.0},
            {"mssa_rl": 1.0},
        ]
    )
    minimum_component_weight: float = 0.05


class EnsembleCoordinator:
    """Selects weightings for combining individual model forecasts."""

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self.selected_weights: Dict[str, float] = {}
        self.selection_score: float = 0.0

    def _apply_minimum_component_weight(self, weights: Dict[str, float]) -> Dict[str, float]:
        minimum = max(float(self.config.minimum_component_weight), 0.0)
        if minimum <= 0.0 or not weights:
            return weights

        pruned = {model: weight for model, weight in weights.items() if weight >= minimum}
        if not pruned:
            # If everything gets pruned, keep the strongest component so callers
            # get a deterministic fallback rather than an empty selection.
            top_model = max(weights.items(), key=lambda item: item[1])[0]
            return {top_model: 1.0}
        return self._normalize(pruned)

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
            normalized = self._apply_minimum_component_weight(normalized)

            # Phase 7.3 FIX: When confidence_scaling is disabled, score candidates purely
            # on config weights (first viable candidate wins) rather than confidence-adjusted scores
            if self.config.confidence_scaling:
                score = sum(
                    normalized.get(model, 0.0) * model_confidence.get(model, 0.0)
                    for model in normalized.keys()
                )
            else:
                # Score = sum of weights (should be ~1.0 after normalization)
                # This makes all candidates equal, so first in config wins
                score = sum(normalized.values())

            # Phase 7.3 DEBUG: Log candidate evaluation
            logger.info(
                "Candidate evaluation: raw=%s normalized=%s score=%.4f",
                candidate,
                normalized,
                score,
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
        weights = pd.Series(self.selected_weights, dtype=float)
        aligned_weights = weights.reindex(combined_df.columns).fillna(0.0)

        def _rowwise_blend(df: pd.DataFrame) -> pd.Series:
            available = df.notna()
            effective_weights = available.mul(aligned_weights, axis=1)
            weight_sum = effective_weights.sum(axis=1)
            normalized_weights = effective_weights.div(weight_sum.replace(0.0, np.nan), axis=0)
            blended = df.mul(normalized_weights, axis=1).sum(axis=1)
            return blended.dropna()

        forecast = _rowwise_blend(combined_df)

        def _blend_ci(bounds: Dict[str, Optional[pd.Series]]) -> Optional[pd.Series]:
            series_map = {
                model: bound for model, bound in bounds.items()
                if model in aligned_weights.index and isinstance(bound, pd.Series)
            }
            if not series_map:
                return None
            df = pd.DataFrame(series_map).reindex(forecast.index)
            return _rowwise_blend(df)

        lower = _blend_ci(lower_bounds)
        upper = _blend_ci(upper_bounds)

        return {
            "forecast": forecast,
            "lower_ci": lower,
            "upper_ci": upper,
        }


def canonical_model_key(key: str) -> str:
    if not isinstance(key, str):
        return ""
    canonical = CANONICAL_MODEL_MAP.get(key, None)
    return canonical if canonical else key.lower()


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

    normalized_summaries: Dict[str, Dict[str, Any]] = {}
    for raw_key, summary in (summaries or {}).items():
        canon = canonical_model_key(raw_key)
        normalized_summaries[canon] = summary or {}
    summaries = normalized_summaries

    # Instrumentation: show which models are present and whether metrics exist.
    model_keys = sorted(k for k in summaries.keys() if k not in {"errors", "events", "series_diagnostics"})
    metrics_presence = {
        k: bool((summaries.get(k) or {}).get("regression_metrics"))
        for k in model_keys
        if k in TRACKED_MODELS
    }
    logger.info("Ensemble summaries keys=%s regression_metrics_present=%s", model_keys, metrics_presence)

    sarimax_summary = summaries.get("sarimax", {})
    garch_summary = summaries.get("garch", {})
    samossa_summary = summaries.get("samossa", {})
    mssa_summary = summaries.get("mssa_rl", {})

    # Treat SAMOSSA as the primary TS baseline for variance tests when
    # available; fall back to SARIMAX metrics otherwise.
    samossa_metrics_baseline = samossa_summary.get("regression_metrics", {}) or {}
    sarimax_metrics_baseline = sarimax_summary.get("regression_metrics", {}) or {}
    baseline_metrics = samossa_metrics_baseline or sarimax_metrics_baseline

    metrics_map = {
        model: (summaries.get(model) or {}).get("regression_metrics", {}) or {}
        for model in TRACKED_MODELS
    }
    rmse_candidates = [m.get("rmse") for m in metrics_map.values() if m.get("rmse") is not None]
    baseline_rmse = float(min(rmse_candidates)) if rmse_candidates else None
    baseline_te = baseline_metrics.get("tracking_error")

    def _combine_scores(*scores: Optional[float]) -> Optional[float]:
        valid = [float(np.clip(s, 0.0, 1.0)) for s in scores if s is not None]
        if not valid:
            return None
        return float(np.clip(np.mean(valid), 0.05, 0.95))

    def _relative_rmse_score(rmse: float, baseline: Optional[float]) -> Optional[float]:
        if baseline is None or baseline <= 0.0:
            return None
        ratio = max(float(rmse) / max(float(baseline), EPSILON), EPSILON)
        # ratio=1.0 -> ~0.7, ratio=1.1 -> ~0.55, ratio=1.5 -> ~0.25
        score = 1.0 / (1.0 + 1.5 * (ratio - 1.0))
        return float(np.clip(score, 0.05, 0.95))

    def _relative_te_score(te: float, baseline: Optional[float]) -> Optional[float]:
        if baseline is None or baseline <= 0.0:
            return None
        ratio = max(float(te) / max(float(baseline), EPSILON), EPSILON)
        score = 1.0 / (1.0 + 1.2 * (ratio - 1.0))
        return float(np.clip(score, 0.05, 0.95))

    def _score_from_metrics(metrics: Dict[str, float]) -> Optional[float]:
        if not metrics:
            return None
        components = []
        rmse_val = metrics.get("rmse")
        if rmse_val is not None:
            rmse_score = _relative_rmse_score(float(rmse_val), baseline_rmse)
            if rmse_score is not None:
                components.append(rmse_score)
        smape_val = metrics.get("smape")
        if smape_val is not None:
            smape = max(float(smape_val), 0.0)
            smape_score = 1.0 / (1.0 + 0.5 * smape)
            components.append(float(np.clip(smape_score, 0.05, 0.95)))
        te_val = metrics.get("tracking_error")
        if te_val is not None:
            te_score = _relative_te_score(float(te_val), baseline_te)
            if te_score is not None:
                components.append(te_score)
        da_val = metrics.get("directional_accuracy")
        if da_val is not None:
            da = float(da_val)
            # Treat 0.5 as random baseline; reward edges above that.
            da_score = max(0.0, (da - 0.5) / 0.5)
            components.append(float(np.clip(da_score, 0.05, 0.95)))
        if not components:
            return None
        return float(np.clip(np.mean(components), 0.05, 0.95))

    def _variance_test_score(metrics: Dict[str, float], baseline: Dict[str, float]) -> Optional[float]:
        if not metrics or not baseline:
            return None
        te = metrics.get("tracking_error")
        base_te = baseline.get("tracking_error")
        n = metrics.get("n_observations")
        base_n = baseline.get("n_observations")
        if te is None or base_te is None or n is None or base_n is None:
            return None
        te = float(te)
        base_te = float(base_te)
        if te > base_te:
            # One-sided screening: do not reward models with higher residual variance
            # than the baseline.
            return 0.0
        f_stat = (te**2 + EPSILON) / (base_te**2 + EPSILON)
        dfn = max(int(n) - 1, 1)
        dfd = max(int(base_n) - 1, 1)
        # One-sided p-value for variance reduction (H1: sigma_model^2 < sigma_baseline^2).
        p_value = float(scipy_stats.f.cdf(f_stat, dfn, dfd))
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

    # GARCH confidence scoring - Phase 7.3 addition for ensemble integration
    # Use AIC/BIC (like SARIMAX) as primary confidence indicator
    aic = garch_summary.get("aic")
    bic = garch_summary.get("bic")
    garch_score = None
    if aic is not None and bic is not None:
        garch_score = np.exp(-0.5 * (aic + bic) / max(abs(aic) + abs(bic), 1e-6))

    # If regression_metrics available, blend with AIC/BIC score
    garch_metrics = garch_summary.get("regression_metrics", {}) or {}
    garch_score = _combine_scores(
        garch_score,  # AIC/BIC score (if available)
        _score_from_metrics(garch_metrics),
        _variance_test_score(garch_metrics, baseline_metrics)
        if baseline_metrics
        else None,
    )
    if garch_score is None and garch_summary:
        garch_score = 0.35  # fallback if AIC/BIC also unavailable
    if garch_score is not None:
        confidence["garch"] = garch_score

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

    # Keep scores within a bounded, non-saturating range to avoid winner-takes-all
    clipped_confidence = {model: float(np.clip(score, 0.05, 0.95)) for model, score in confidence.items()}

    # Phase 7.4 FIX: Quantile-based calibration instead of min-max normalization
    # This prevents SAMoSSA from always getting 1.0 and makes scores truly comparable
    if len(clipped_confidence) > 1:
        values = np.array(list(clipped_confidence.values()))

        # Use rank-based normalization (more robust than min-max)
        # Ranks models from 0 to 1 based on relative performance
        ranks = scipy_stats.rankdata(values, method='average')

        # Normalize ranks to 0.3-0.9 range to avoid extremes
        # This gives even the worst model some weight for diversity
        min_rank, max_rank = ranks.min(), ranks.max()
        if max_rank > min_rank:
            normalized_ranks = 0.3 + 0.6 * (ranks - min_rank) / (max_rank - min_rank)
            calibrated_confidence = {
                model: float(normalized_ranks[i])
                for i, model in enumerate(clipped_confidence.keys())
            }

            logger.info(
                "Calibrated confidence (Phase 7.4 quantile-based): raw=%s calibrated=%s",
                clipped_confidence,
                calibrated_confidence,
            )
            return calibrated_confidence
        else:
            # All models have same confidence - use uniform distribution
            uniform_score = 0.6  # Middle of 0.3-0.9 range
            calibrated_confidence = {model: uniform_score for model in clipped_confidence.keys()}
            logger.info(
                "Calibrated confidence (uniform - all equal): raw=%s calibrated=%s",
                clipped_confidence,
                calibrated_confidence,
            )
            return calibrated_confidence

    return clipped_confidence
