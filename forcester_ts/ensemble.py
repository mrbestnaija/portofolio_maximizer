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


def _apply_da_cap(
    weights: Dict[str, float],
    da_scores: Dict[str, float],
    da_floor: float,
    da_weight_cap: float,
) -> Dict[str, float]:
    """Phase 7.17: Cap and proportionally redistribute weights for models whose
    directional accuracy is below da_floor.

    CONTRACT / INVARIANTS (machine-checked at runtime):
    - If ALL models in ``weights`` are DA-penalized, returns ``{}``.
      Callers must treat ``{}`` as "skip this candidate" (never score it).
    - Otherwise returns a non-empty dict where:
        * every value is in [0, 1].
        * sum of values ≈ 1.0 (within 1e-6 after redistribution).
        * every model m with DA < da_floor has weight ≤ da_weight_cap.
    - ``weights`` is assumed to be normalized (sum ≈ 1.0) on entry.
    - da_floor and da_weight_cap must be positive floats.

    Violation of the sum-to-1 post-condition is treated as a bug:
    the function logs an ERROR and self-corrects via renormalization.
    """
    # Identify ALL DA-penalized models (DA < da_floor), regardless of current weight.
    penalized = {m for m in weights if da_scores.get(m, 1.0) < da_floor}
    # Only those currently ABOVE the cap need to be reduced.
    capped_set = {m for m in penalized if weights[m] > da_weight_cap}

    if not capped_set:
        # No model is above the cap → nothing to change.
        return weights

    result = dict(weights)
    for m in capped_set:
        result[m] = da_weight_cap

    # Redistribute only to NON-penalized models.
    # Sending budget to other penalized models (even below-cap ones) would let them
    # grow above da_weight_cap after redistribution, violating the contract.
    non_penalized = {m: result[m] for m in result if m not in penalized}
    if not non_penalized:
        # All models in this candidate are DA-penalized.
        # Return {} so the caller's `if not normalized: continue` skips it cleanly.
        return {}

    # Budget freed from capping: 1.0 minus the fixed portions.
    sum_fixed = sum(result[m] for m in penalized)
    remaining = max(0.0, 1.0 - sum_fixed)

    np_total = sum(non_penalized.values())
    if np_total > EPSILON:
        for m in non_penalized:
            result[m] = non_penalized[m] / np_total * remaining
    else:
        share = remaining / len(non_penalized)
        for m in non_penalized:
            result[m] = share

    # Runtime invariant guard: result must sum to 1.0.
    # This catches floating-point drift and any logic errors in the redistribution.
    _total = sum(result.values())
    if abs(_total - 1.0) > 1e-6:
        logger.error(
            "_apply_da_cap: weight sum invariant violated (sum=%.9f). "
            "Bug in redistribution logic — renormalizing. input_weights=%s",
            _total, weights,
        )
        if _total > EPSILON:
            result = {m: v / _total for m, v in result.items()}

    return result


@dataclass
class EnsembleConfig:
    enabled: bool = True
    confidence_scaling: bool = True
    regime_detection_enabled: bool = True  # Phase 7.4: Enable regime-aware selection
    # Phase 7.10b: Track per-model directional accuracy and generate a
    # directional-accuracy-weighted candidate during selection.
    track_directional_accuracy: bool = True
    # Prefer diversified candidates when they are close in quality to a
    # concentrated winner. This prevents brittle winner-takes-all selections
    # unless the single model is materially better.
    prefer_diversified_candidate: bool = True
    diversity_tolerance: float = 0.35
    candidate_weights: List[Dict[str, float]] = field(
        default_factory=lambda: [
            # Phase 7.10: Defaults exclude SARIMAX (disabled by default).
            # YAML config adds SARIMAX candidates that activate when re-enabled.
            {"garch": 0.85, "samossa": 0.10, "mssa_rl": 0.05},
            {"garch": 0.70, "samossa": 0.20, "mssa_rl": 0.10},
            {"garch": 0.60, "samossa": 0.25, "mssa_rl": 0.15},
            {"samossa": 0.60, "mssa_rl": 0.40},
            {"samossa": 0.45, "garch": 0.35, "mssa_rl": 0.20},
            {"mssa_rl": 0.70, "garch": 0.30},
            {"mssa_rl": 0.70, "samossa": 0.30},
            {"garch": 1.0},
            {"samossa": 1.0},
            {"mssa_rl": 1.0},
        ]
    )
    # Phase 7.17: Auto-computed adaptive candidates prepended before static ones.
    # Written by scripts/ensemble_health_audit.py --write-config.
    # Empty list = use only static candidate_weights (default / safe fallback).
    adaptive_candidate_weights: List[Dict[str, float]] = field(default_factory=list)
    # Phase 7.17: Configurable DA penalty — cap weight of models with chronic near-zero DA.
    da_floor: float = 0.10       # DA below this triggers penalty
    da_weight_cap: float = 0.10  # Max weight for DA-penalized models
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
        model_directional_accuracy: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], float]:
        if not self.config.enabled or not self.config.candidate_weights:
            self.selected_weights = {}
            self.selection_score = 0.0
            return self.selected_weights, self.selection_score

        # Phase 7.10b: Build directional-accuracy-weighted candidate when tracking is enabled.
        # This adds a data-driven candidate whose weights are proportional to each model's
        # CV directional accuracy (hit rate above 0.5 baseline).
        # Phase 7.17: Prepend adaptive candidates (from ensemble_health_audit) before static ones.
        _adaptive = getattr(self.config, "adaptive_candidate_weights", []) or []
        candidate_list = (
            list(_adaptive) + list(self.config.candidate_weights)
            if _adaptive
            else list(self.config.candidate_weights)
        )
        if _adaptive:
            logger.info("Ensemble: prepending %d adaptive candidate(s) from Phase 7.17 audit", len(_adaptive))
        if getattr(self.config, "track_directional_accuracy", True) and model_directional_accuracy:
            da_weights: Dict[str, float] = {}
            for model in TRACKED_MODELS:
                da = float(model_directional_accuracy.get(model, 0.5))
                # Map [0.4, 0.6] to [0.01, 1.0] — weight = max(0, da - 0.4) / 0.2
                da_weight = max(0.0, (da - 0.40) / 0.20)
                if da_weight > 0:
                    da_weights[model] = da_weight
            if da_weights:
                candidate_list = [{"auto_directional": 0.0, **da_weights}] + candidate_list
                logger.info(
                    "Ensemble: auto_directional candidate=%s (from DA=%s)",
                    da_weights,
                    {m: f"{v:.3f}" for m, v in model_directional_accuracy.items()},
                )

        scored_candidates: List[Tuple[Dict[str, float], float]] = []

        for candidate in candidate_list:
            # Strip meta-keys like 'auto_directional' before normalizing
            candidate = {k: v for k, v in candidate.items() if k != "auto_directional"}
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

            # Phase 7.17: Apply DA penalty — cap and redistribute weight for
            # chronic near-zero DA models (SAMOSSA DA=0 anomaly fix).
            if model_directional_accuracy:
                _da_floor = float(getattr(self.config, "da_floor", 0.10))
                _da_cap = float(getattr(self.config, "da_weight_cap", 0.10))
                normalized = _apply_da_cap(normalized, model_directional_accuracy, _da_floor, _da_cap)
                if not normalized:
                    continue
                logger.debug(
                    "Ensemble: DA-adjusted weights=%s (da_floor=%.2f, da_cap=%.2f)",
                    {m: f"{v:.3f}" for m, v in normalized.items()},
                    _da_floor,
                    _da_cap,
                )

            # Always rank candidates by confidence-weighted expected quality.
            # `confidence_scaling` controls whether candidate *weights* are
            # scaled, not whether selection quality should ignore confidence.
            score = sum(
                normalized.get(model, 0.0) * model_confidence.get(model, 0.5)
                for model in normalized.keys()
            )

            # Phase 7.3 DEBUG: Log candidate evaluation
            logger.info(
                "Candidate evaluation: raw=%s normalized=%s score=%.4f",
                candidate,
                normalized,
                score,
            )
            scored_candidates.append((normalized, score))

        if not scored_candidates:
            self.selected_weights = {}
            self.selection_score = 0.0
            return self.selected_weights, self.selection_score

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        best_weights, best_score = scored_candidates[0]

        if self.config.prefer_diversified_candidate and len(best_weights) <= 1:
            tolerance = float(np.clip(self.config.diversity_tolerance, 0.0, 0.95))
            min_allowed = best_score * (1.0 - tolerance)
            diversified = [
                (weights, score)
                for weights, score in scored_candidates
                if len(weights) >= 2 and score >= min_allowed
            ]
            if diversified:
                best_weights, best_score = diversified[0]

        # Runtime invariant: selected weights must be normalized, non-negative, finite.
        _wtotal = sum(best_weights.values())
        if best_weights and abs(_wtotal - 1.0) > 1e-6:
            logger.error(
                "select_weights: selected weights not normalized (sum=%.9f). "
                "Renormalizing to prevent score corruption.",
                _wtotal,
            )
            if _wtotal > EPSILON:
                best_weights = {m: v / _wtotal for m, v in best_weights.items()}

        self.selected_weights = best_weights
        self.selection_score = best_score
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

        # Phase 8.6: detect critically-misaligned forecast indices.
        # Partial date-range overlap is expected and handled by NaN-fill in _rowwise_blend.
        # Only flag when the index intersection is < 50% of the longest contributing series,
        # which indicates models returned entirely different horizon windows (a real bug).
        index_mismatch = False
        if len(contributing) > 1:
            ref_name, ref_series = next(iter(contributing.items()))
            all_indices = [s.index for s in contributing.values()]
            max_len = max(len(s) for s in contributing.values())
            common_idx = all_indices[0]
            for idx in all_indices[1:]:
                common_idx = common_idx.intersection(idx)
            if len(common_idx) < max_len * 0.5:
                mismatched_models = [
                    m for m, s in contributing.items()
                    if m != ref_name and (len(s) != len(ref_series) or not s.index.equals(ref_series.index))
                ]
                index_mismatch = True
                logger.error(
                    "Phase 8.6 ensemble index mismatch: models %s have critically different "
                    "index from reference '%s' (intersection=%d < 50%% of max=%d). "
                    "Using intersection to recover.",
                    mismatched_models, ref_name, len(common_idx), max_len,
                )
                contributing = {m: s.reindex(common_idx) for m, s in contributing.items()}

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

        result: Dict[str, Any] = {
            "forecast": forecast,
            "lower_ci": lower,
            "upper_ci": upper,
        }
        if index_mismatch:
            result["ensemble_index_mismatch"] = True
        return result


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

    def _garch_domain_normalize(
        raw_aic_bic_score: Optional[float],
        igarch_fallback: bool,
    ) -> float:
        """
        Map GARCH's volatility-domain AIC/BIC score to the price-domain
        confidence band used by SAMOSSA and MSSA-RL.

        GARCH models variance dynamics; its AIC/BIC measures volatility model
        fit quality -- not directly comparable to SAMOSSA's EVR (always ~1.0 by
        SSA construction) or MSSA-RL's baseline_variance metric.  Mixing raw
        AIC/BIC magnitudes with EVR distorts weight selection because the scales
        are incommensurable (different signal domains).

        Strategy: compress GARCH's contribution to a neutral [0.42, 0.58] band
        that keeps it in the 3-model confidence pool without claiming top-model
        status via volatility-fit quality alone.  With 3 models present, the
        quantile normalizer maps to [0.40, 0.625, 0.65], enabling GARCH-heavy
        blend candidates to clear the diversity_tolerance threshold.

        Phase 7.15-E: replaces the `if garch_score is None and garch_summary`
        dead-code block where `{}` (falsy) prevented GARCH from ever getting a
        fallback score, causing permanent 2-model collapse.
        """
        if igarch_fallback:
            # IGARCH/EWMA degrades to exponential smoothing -- lower information
            # content than converged GARCH; keep below the neutral midpoint.
            return 0.28
        if raw_aic_bic_score is None:
            # GARCH fit converged but produced no information criteria; assign a
            # neutral participation score just above the MSSA-RL floor (0.40).
            return 0.45
        # Compress AIC/BIC-derived raw score [0, 1] to [0.42, 0.58].
        # Prevents a high AIC/BIC ratio from claiming SAMOSSA's top rank
        # (earned via price-level EVR -- a separate measurement domain).
        return float(np.clip(0.42 + 0.16 * float(np.clip(raw_aic_bic_score, 0.0, 1.0)), 0.28, 0.58))

    # GARCH confidence scoring
    # Phase 7.3+: Use AIC/BIC as primary indicator, then blend with regression metrics.
    # Phase 7.15-E: Apply domain normalization.  GARCH models variance dynamics (not
    # price levels); raw AIC/BIC is incommensurable with SAMOSSA's EVR.  The helper
    # _garch_domain_normalize() maps GARCH's volatility-domain score to the same
    # effective confidence band as the other models, ensuring GARCH always
    # participates in the 3-model pool and blend candidates can clear diversity gate.
    aic = garch_summary.get("aic")
    bic = garch_summary.get("bic")
    _raw_garch_aic_bic: Optional[float] = None
    if aic is not None and bic is not None:
        _raw_garch_aic_bic = float(np.exp(-0.5 * (aic + bic) / max(abs(aic) + abs(bic), 1e-6)))

    garch_normalized = _garch_domain_normalize(
        _raw_garch_aic_bic, bool(garch_summary.get("igarch_fallback"))
    )
    logger.info(
        "GARCH domain normalization: raw_aic_bic_score=%.4f normalized=%.4f igarch=%s",
        _raw_garch_aic_bic if _raw_garch_aic_bic is not None else float("nan"),
        garch_normalized,
        garch_summary.get("igarch_fallback", False),
    )

    # Blend domain-normalized score with regression metrics when available.
    garch_metrics = garch_summary.get("regression_metrics", {}) or {}
    garch_score = _combine_scores(
        garch_normalized,
        _score_from_metrics(garch_metrics),
        _variance_test_score(garch_metrics, baseline_metrics)
        if baseline_metrics
        else None,
    )
    # garch_normalized is always non-None, so _combine_scores always returns
    # non-None.  GARCH is always included in the confidence pool.
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
        # Phase 7.10: Use log-scaled baseline_variance to avoid crushing score
        # when variance is large.  Previous formula 1/(1+var) gave ~0.09 for
        # var=10; log-scale gives ~0.30 which is fairer relative to other models.
        mssa_score = 1.0 / (1.0 + max(0.0, float(np.log1p(baseline_var))))
    mssa_metrics = mssa_summary.get("regression_metrics", {}) or {}
    mssa_score = _combine_scores(
        mssa_score,
        _score_from_metrics(mssa_metrics),
        _variance_test_score(mssa_metrics, baseline_metrics)
        if baseline_metrics
        else None,
        _change_point_boost(mssa_summary),
    )
    # Phase 7.10: MSSA-RL floor -- adversarial audit shows MSSA-RL is best single
    # model 60% of the time but receives only 8.7% weight.  Ensure minimum score.
    # Raised from 0.30 to 0.40 to match narrowed calibration range (0.4-0.85).
    if mssa_score is not None:
        mssa_score = max(mssa_score, 0.40)
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

    # Phase 7.10: Floor confidence at 0.05 to avoid division-by-zero downstream.
    floored_confidence = {
        model: float(max(score, 0.05))
        for model, score in confidence.items()
    }

    # Phase 7.4 FIX: Quantile-based calibration instead of min-max normalization
    # This prevents SAMoSSA from always getting 1.0 and makes scores truly comparable
    if len(floored_confidence) > 1:
        values = np.array(list(floored_confidence.values()))

        # Use rank-based normalization (more robust than min-max)
        # Ranks models from 0 to 1 based on relative performance
        ranks = scipy_stats.rankdata(values, method='average')

        # Phase 7.10: Narrowed from 0.3-0.9 to 0.4-0.85 to prevent
        # winner-takes-all when confidence_scaling amplifies rank gaps.
        # Previous 0.3-0.9 gave 3:1 ratio (MSSA-RL 8.7% weight);
        # 0.4-0.85 gives ~2:1 ratio which allows MSSA-RL meaningful weight.
        min_rank, max_rank = ranks.min(), ranks.max()
        if max_rank > min_rank:
            normalized_ranks = 0.4 + 0.45 * (ranks - min_rank) / (max_rank - min_rank)
            calibrated_confidence = {
                model: float(normalized_ranks[i])
                for i, model in enumerate(floored_confidence.keys())
            }
        else:
            # All models have same confidence - use uniform distribution
            uniform_score = 0.625  # Middle of 0.4-0.85 range
            calibrated_confidence = {model: uniform_score for model in floored_confidence.keys()}

        # Phase 7.10: Cap confidence at empirical accuracy ceiling AFTER ranking.
        # Adversarial audit: 0.9+ confidence with 41% win rate.  Downstream
        # position sizing treats confidence as probability-of-profit, so cap at
        # 0.65 (generous vs 41% actual) until backtested accuracy improves.
        # Applied post-ranking to preserve relative ordering between models.
        CONFIDENCE_ACCURACY_CAP = 0.65
        calibrated_confidence = {
            model: float(min(score, CONFIDENCE_ACCURACY_CAP))
            for model, score in calibrated_confidence.items()
        }

        logger.info(
            "Calibrated confidence (Phase 7.4 quantile-based): raw=%s calibrated=%s",
            floored_confidence,
            calibrated_confidence,
        )
        return calibrated_confidence

    return floored_confidence
