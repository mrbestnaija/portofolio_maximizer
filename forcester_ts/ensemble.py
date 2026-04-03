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
            # Phase 10: SARIMAX-anchored (model-class diversity vs spectral/RL/GARCH)
            {"sarimax": 0.50, "garch": 0.30, "mssa_rl": 0.20},
            {"sarimax": 0.40, "mssa_rl": 0.35, "garch": 0.25},
            # Phase 10: MSSA-RL elevated to positions 3-4 (was buried at 6-7)
            {"mssa_rl": 0.55, "garch": 0.30, "samossa": 0.15},
            {"mssa_rl": 0.50, "sarimax": 0.30, "garch": 0.20},
            # GARCH-heavy blends (retained from Phase 7.9)
            {"garch": 0.85, "samossa": 0.10, "mssa_rl": 0.05},
            {"garch": 0.70, "samossa": 0.20, "mssa_rl": 0.10},
            {"garch": 0.60, "samossa": 0.25, "mssa_rl": 0.15},
            # SAMoSSA-anchored
            {"samossa": 0.60, "mssa_rl": 0.40},
            {"samossa": 0.45, "garch": 0.35, "mssa_rl": 0.20},
            # MSSA-RL dominant (Phase 7.9)
            {"mssa_rl": 0.70, "garch": 0.30},
            {"mssa_rl": 0.70, "samossa": 0.30},
            # Single-model anchors
            {"garch": 1.0},
            {"samossa": 1.0},
            {"mssa_rl": 1.0},
            {"sarimax": 1.0},
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
            result = blended.dropna()
            dropped = len(blended) - len(result)
            if dropped > 0:
                logger.warning(
                    "[ENSEMBLE] _rowwise_blend dropped %d/%d timestamps with all-NaN "
                    "model forecasts — forecast horizon shortened unexpectedly.",
                    dropped, len(blended),
                )
            return result

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
    oos_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Convert per-model diagnostics into comparable confidence scores.
    Scores blend information criteria, realised regression metrics,
    and one-sided F-tests (variance ratio) similar to Diebold-Mariano style
    screening used in production quant stacks.

    Args:
        summaries: Per-model in-sample diagnostic summaries (component_summaries).
        oos_metrics: Optional trailing out-of-sample evaluation metrics from the
            previous forecast window (self._latest_metrics).  When provided,
            the RMSE-rank hybrid (Phase 10) uses these OOS values instead of the
            always-empty in-sample regression_metrics, making the ranking live.
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
    # Wiring fix (2026-03-29): metrics_map is always {} in production because
    # get_component_summaries() rebuilds _model_summaries from in-sample fit
    # diagnostics on every forecast() call, overwriting the regression_metrics
    # that evaluate() wrote in the prior window.  _score_from_metrics,
    # _relative_rmse_score, _relative_te_score, and _variance_test_score are
    # therefore all dead code at selection time unless we supply OOS values.
    # When _oos_component_metrics is available use it to override per-model
    # metrics and the baseline values so those scoring functions actually fire.

    # Phase 10: Hybrid RMSE-rank component — rank-normalized RMSE across available models.
    # EVR (SAMoSSA) is always ~1.0 by SSA construction; this counterbalances it with realized
    # forecast accuracy, preventing confidence_scaling from over-weighting near-flat forecasts.
    #
    # P0 fix (2026-03-29): Use trailing OOS metrics when provided.  metrics_map reads
    # component_summaries["regression_metrics"] which is always {} in production because
    # evaluate() writes those values AFTER this function returns (forecaster.py:2391).
    # oos_metrics (self._latest_metrics from the prior evaluate() call) contains real
    # OOS RMSE values so the RMSE-rank scores are non-trivial on every production run.
    #
    # Wiring fix (2026-03-29): _latest_metrics also contains an "ensemble" key alongside
    # the four component models.  Filter oos_metrics to TRACKED_MODELS only so the
    # ensemble's RMSE does not shift _min_rmse/_max_rmse and distort component rankings.
    _oos_component_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    if oos_metrics:
        _oos_component_metrics = {
            model: m
            for model, m in oos_metrics.items()
            if model in TRACKED_MODELS
        }
    _rmse_source = _oos_component_metrics if (_oos_component_metrics and any(
        (m or {}).get("rmse") is not None for m in _oos_component_metrics.values()
    )) else metrics_map
    _rmse_values = {
        model: float(m.get("rmse"))
        for model, m in _rmse_source.items()
        if (m or {}).get("rmse") is not None and np.isfinite(float(m.get("rmse")))
    }
    if _rmse_source is _oos_component_metrics and _rmse_values:
        logger.info("Phase 10 RMSE-rank: using trailing OOS metrics (%d models)", len(_rmse_values))
    if len(_rmse_values) >= 2:
        _min_rmse = min(_rmse_values.values())
        _max_rmse = max(_rmse_values.values())
        _rmse_rank_scores: Dict[str, float] = {
            model: float(np.clip(
                1.0 - (rmse - _min_rmse) / (_max_rmse - _min_rmse + 1e-10),
                0.05, 0.95,
            ))
            for model, rmse in _rmse_values.items()
        }
        logger.info("Phase 10 RMSE-rank scores: %s", {m: f"{v:.3f}" for m, v in _rmse_rank_scores.items()})
    else:
        _rmse_rank_scores: Dict[str, float] = {}

    # Wiring fix cont. (2026-03-29): if OOS component metrics are available,
    # override baseline_rmse, baseline_te, baseline_metrics, and the per-model
    # *_metrics variables so _score_from_metrics and _variance_test_score fire.
    # These are currently dead because metrics_map / *_metrics come from in-sample
    # summaries (always {} at selection time — see comment above).
    if _oos_component_metrics:
        # Baseline consistency fix (2026-03-29): use a SINGLE reference model for
        # all three baseline values so _relative_rmse_score, _relative_te_score, and
        # _variance_test_score all compare against the same denominator.
        # Previous code set baseline_rmse = min(all_models) but baseline_te from
        # SAMoSSA/SARIMAX — a scoring-contract mismatch.
        # Priority: SAMoSSA OOS (primary TS baseline), then SARIMAX OOS fallback.
        _oos_samossa = _oos_component_metrics.get("samossa") or {}
        _oos_sarimax = _oos_component_metrics.get("sarimax") or {}
        _oos_baseline = _oos_samossa if _oos_samossa.get("rmse") is not None else _oos_sarimax
        if _oos_baseline:
            baseline_metrics = _oos_baseline
            _bl_rmse = _oos_baseline.get("rmse")
            if _bl_rmse is not None and np.isfinite(float(_bl_rmse)):
                baseline_rmse = float(_bl_rmse)
            _bl_te = _oos_baseline.get("tracking_error")
            if _bl_te is not None and np.isfinite(float(_bl_te)):
                baseline_te = float(_bl_te)
        logger.info(
            "OOS baseline override: model=%s baseline_rmse=%.4f baseline_te=%s",
            "samossa" if _oos_samossa.get("rmse") is not None else "sarimax",
            baseline_rmse if baseline_rmse is not None else float("nan"),
            f"{baseline_te:.4f}" if baseline_te is not None else "None",
        )

    def _combine_scores(*scores: Optional[float]) -> Optional[float]:
        valid = [
            float(np.clip(s, 0.0, 1.0))
            for s in scores
            if s is not None and np.isfinite(float(s))
        ]
        if not valid:
            return None
        return float(np.clip(np.mean(valid), 0.05, 0.95))

    def _relative_rmse_score(rmse: float, baseline: Optional[float]) -> Optional[float]:
        if baseline is None or baseline <= 0.0:
            return None
        if not np.isfinite(float(rmse)):
            return None
        ratio = max(float(rmse) / max(float(baseline), EPSILON), EPSILON)
        # ratio=1.0 -> ~0.7, ratio=1.1 -> ~0.55, ratio=1.5 -> ~0.25
        score = 1.0 / (1.0 + 1.5 * (ratio - 1.0))
        return float(np.clip(score, 0.05, 0.95))

    def _relative_te_score(te: float, baseline: Optional[float]) -> Optional[float]:
        if baseline is None or baseline <= 0.0:
            return None
        if not np.isfinite(float(te)):
            return None
        ratio = max(float(te) / max(float(baseline), EPSILON), EPSILON)
        score = 1.0 / (1.0 + 1.2 * (ratio - 1.0))
        return float(np.clip(score, 0.05, 0.95))

    def _score_from_metrics(metrics: Dict[str, float]) -> Optional[float]:
        """Compute model confidence from metrics, separating fit quality from prediction quality.

        Architecture:
          - Fit quality  (60% weight): RMSE-rank, SMAPE, tracking error.
            These measure how well the model fits the training data.
          - Prediction quality (40% weight): 1-step DA, terminal DA, CI coverage.
            These measure whether the model's outputs are directionally correct and
            probabilistically calibrated on out-of-sample data.

        Separating these prevents fit quality from masking poor prediction quality
        (e.g. SAMoSSA EVR=1.0 by SSA construction does not imply directional accuracy).
        Both components require at least one score; the blend uses whatever is available.
        """
        if not metrics:
            return None

        _n_obs = metrics.get("n_observations", 0)
        _n_obs_int = int(_n_obs) if _n_obs is not None and np.isfinite(float(_n_obs)) else 0

        # --- Fit quality components ---
        fit_components = []
        rmse_val = metrics.get("rmse")
        if rmse_val is not None:
            rmse_score = _relative_rmse_score(float(rmse_val), baseline_rmse)
            if rmse_score is not None:
                fit_components.append(rmse_score)
        smape_val = metrics.get("smape")
        if smape_val is not None and np.isfinite(float(smape_val)):
            smape_s = max(float(smape_val), 0.0)
            fit_components.append(float(np.clip(1.0 / (1.0 + 0.5 * smape_s), 0.05, 0.95)))
        te_val = metrics.get("tracking_error")
        if te_val is not None and np.isfinite(float(te_val)):
            te_score = _relative_te_score(float(te_val), baseline_te)
            if te_score is not None:
                fit_components.append(te_score)

        # --- Prediction quality components (require n_obs >= 30) ---
        pred_components = []
        # 1-step DA: fraction of periods where forecast direction matched actual direction.
        da_val = metrics.get("directional_accuracy")
        if da_val is not None and np.isfinite(float(da_val)) and _n_obs_int >= 30:
            da = float(np.clip(da_val, 0.0, 1.0))
            pred_components.append(float(np.clip(max(0.0, (da - 0.5) / 0.5), 0.05, 0.95)))

        # Terminal DA: did forecast[-1] correctly predict direction vs forecast[0]?
        # This maps directly to multi-step trade P&L and is harder to fake than 1-step DA.
        tda_val = metrics.get("terminal_directional_accuracy")
        if tda_val is not None and np.isfinite(float(tda_val)) and _n_obs_int >= 30:
            tda = float(np.clip(tda_val, 0.0, 1.0))
            pred_components.append(float(np.clip(max(0.0, (tda - 0.5) / 0.5), 0.05, 0.95)))

        # CI coverage: did the actual terminal price fall within the predicted CI?
        # Low coverage → CI too narrow → SNR inflated → block miscalibrated models.
        cov_val = metrics.get("terminal_ci_coverage")
        if cov_val is not None and np.isfinite(float(cov_val)) and _n_obs_int >= 30:
            pred_components.append(float(np.clip(cov_val, 0.05, 0.95)))

        # Blend: 60% fit quality, 40% prediction quality when both are available.
        # Fall back to whichever component class has data.
        if fit_components and pred_components:
            fit_score = float(np.mean(fit_components))
            pred_score = float(np.mean(pred_components))
            blended = 0.60 * fit_score + 0.40 * pred_score
        elif fit_components:
            blended = float(np.mean(fit_components))
        elif pred_components:
            blended = float(np.mean(pred_components))
        else:
            return None

        return float(np.clip(blended, 0.05, 0.95))

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
        # Guard: non-finite tracking errors cannot be compared meaningfully.
        if not np.isfinite(te) or not np.isfinite(base_te):
            return None
        if te > base_te:
            # One-sided screening: do not reward models with higher residual variance
            # than the baseline.
            return 0.0
        f_stat = (te**2 + EPSILON) / (base_te**2 + EPSILON)
        if not np.isfinite(f_stat):
            return None
        n_int = int(n) if np.isfinite(float(n)) else 2
        base_n_int = int(base_n) if np.isfinite(float(base_n)) else 2
        dfn = max(n_int - 1, 1)
        dfd = max(base_n_int - 1, 1)
        # One-sided p-value for variance reduction (H1: sigma_model^2 < sigma_baseline^2).
        p_value = float(scipy_stats.f.cdf(f_stat, dfn, dfd))
        if not np.isfinite(p_value):
            return None
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

    # Per-model regression_metrics override: prefer OOS values when available.
    # In-sample summaries always have empty regression_metrics at selection time
    # (get_component_summaries() rebuilds them without OOS data each forecast call).
    _per_model_oos: Dict[str, Dict[str, Any]] = _oos_component_metrics or {}

    aic = sarimax_summary.get("aic")
    bic = sarimax_summary.get("bic")
    sarimax_score = None
    if aic is not None and bic is not None and np.isfinite(float(aic)) and np.isfinite(float(bic)):
        sarimax_score = np.exp(-0.5 * (aic + bic) / max(abs(aic) + abs(bic), 1e-6))
        if not np.isfinite(sarimax_score):
            sarimax_score = None
    sarimax_metrics = (
        _per_model_oos.get("sarimax")
        or sarimax_summary.get("regression_metrics", {})
        or {}
    )
    sarimax_score = _combine_scores(
        sarimax_score,
        _score_from_metrics(sarimax_metrics),
        _variance_test_score(sarimax_metrics, baseline_metrics)
        if baseline_metrics
        else None,
        _rmse_rank_scores.get("sarimax"),  # Phase 10: RMSE-rank hybrid
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
    garch_metrics = (
        _per_model_oos.get("garch")
        or garch_summary.get("regression_metrics", {})
        or {}
    )
    garch_score = _combine_scores(
        garch_normalized,
        _score_from_metrics(garch_metrics),
        _variance_test_score(garch_metrics, baseline_metrics)
        if baseline_metrics
        else None,
        _rmse_rank_scores.get("garch"),  # Phase 10: RMSE-rank hybrid
    )
    # garch_normalized is always non-None, so _combine_scores always returns
    # non-None.  GARCH is always included in the confidence pool.
    if garch_score is not None:
        confidence["garch"] = garch_score

    evr = samossa_summary.get("explained_variance_ratio")
    samossa_score = None
    if evr is not None:
        samossa_score = float(np.clip(evr, 0.0, 1.0))
    samossa_metrics = (
        _per_model_oos.get("samossa")
        or samossa_summary.get("regression_metrics", {})
        or {}
    )
    samossa_score = _combine_scores(
        samossa_score,
        _score_from_metrics(samossa_metrics),
        # SAMOSSA is treated as the primary baseline; skip variance
        # test against itself to avoid degenerate scores.
        None,
        _rmse_rank_scores.get("samossa"),  # Phase 10: RMSE-rank hybrid
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
    mssa_metrics = (
        _per_model_oos.get("mssa_rl")
        or mssa_summary.get("regression_metrics", {})
        or {}
    )
    # P1b fix (2026-03-29): cap change_point_boost before passing to _combine_scores.
    # When recent_change_point_days=0, the unguarded formula returns 1.0 (maximum),
    # which entered _combine_scores with equal weight to EVR and RMSE-rank, flipping
    # candidate selection even when OOS RMSE strongly favours another model.
    # Cap at 0.20 so it can provide a small nudge without overriding quality signals.
    _cpb = _change_point_boost(mssa_summary)
    _cpb_capped = float(np.clip(_cpb, 0.0, 0.20)) if _cpb is not None else None
    mssa_score = _combine_scores(
        mssa_score,
        _score_from_metrics(mssa_metrics),
        _variance_test_score(mssa_metrics, baseline_metrics)
        if baseline_metrics
        else None,
        _cpb_capped,
        _rmse_rank_scores.get("mssa_rl"),  # Phase 10: RMSE-rank hybrid
    )
    # P1c fix (2026-03-29): MSSA-RL hard floor removed from selection path.
    # The floor (0.40) was added when the selector had no OOS signal and MSSA-RL
    # appeared under-weighted.  With P0 wiring OOS RMSE-rank into selection,
    # the floor artificially props MSSA-RL above models with better OOS RMSE,
    # re-introducing the same distortion in the opposite direction.
    # If OOS evidence shows MSSA-RL is best, the RMSE-rank score will reflect
    # that naturally; no floor needed.
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

        # P1a fix (2026-03-29): CONFIDENCE_ACCURACY_CAP removed from candidate
        # scoring.  Applying a flat 0.65 cap here collapsed discrimination between
        # SAMoSSA (RMSE=9.77) and MSSA-RL (RMSE=16.53) to the same value, making
        # select_weights() insertion-order dependent rather than quality-dependent.
        # The cap now lives at the call site (forecaster.py) and is applied only to
        # the *ensemble* output confidence surfaced for position sizing — not to the
        # per-model scores used to pick which candidate to run.

        logger.info(
            "Calibrated confidence (Phase 7.4 quantile-based): raw=%s calibrated=%s",
            floored_confidence,
            calibrated_confidence,
        )
        return calibrated_confidence

    return floored_confidence
