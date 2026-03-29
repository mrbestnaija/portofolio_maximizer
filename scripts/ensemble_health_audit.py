"""
Ensemble health audit: per-model OOS performance decomposition,
proxy Shapley attribution, and adaptive weight computation.

Reads all forecast_audit_*.json files and produces:
  - Per-model summary (RMSE, DA, DA=0 anomaly, times best single)
  - Proxy Shapley attribution (CAUTION: use with caution — RMSE proxy, not realized values)
  - Adaptive candidate weights (exp-decay formula, DA penalty, diversity guard)
  - Markdown report in logs/ensemble_health/
  - Updated adaptive_candidate_weights section in forecasting_config.yml

Usage:
    python scripts/ensemble_health_audit.py
    python scripts/ensemble_health_audit.py --write-config --write-report
    python scripts/ensemble_health_audit.py --audit-dir logs/forecast_audits --recent-n 30
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE_HEALTH_DIR = REPO_ROOT / "logs" / "ensemble_health"
FORECASTING_CONFIG = REPO_ROOT / "config" / "forecasting_config.yml"
MODELS = ("garch", "samossa", "mssa_rl")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window fingerprint (identical to dedupe_audit_windows.py)
# ---------------------------------------------------------------------------

def _window_fingerprint(audit: dict) -> str:
    ds = audit.get("dataset", {})
    summary = audit.get("summary", {})
    return hashlib.sha1(
        json.dumps(
            {
                "ticker": ds.get("ticker"),
                "start": ds.get("start"),
                "end": ds.get("end"),
                "length": ds.get("length"),
                "horizon": summary.get("forecast_horizon") or ds.get("forecast_horizon"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_audit_windows(audit_dir: Path, dedupe: bool = True) -> list[dict]:
    """Load all forecast_audit_*.json files, optionally deduplicating by window fingerprint.

    Uses rglob so that subdirectory splits (research/, production/) are included
    alongside any root-level audit files from older pipeline runs.
    """
    audit_dir = Path(audit_dir)
    files = sorted(audit_dir.rglob("forecast_audit_*.json"))
    raw: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_path"] = f.name
            data["_mtime"] = f.stat().st_mtime
            raw.append(data)
        except Exception as exc:
            log.warning("Skipping malformed JSON %s: %s", f.name, exc)

    if not dedupe:
        return raw

    # Deduplicate: keep newest per fingerprint (later mtime overwrites earlier)
    seen: dict[str, dict] = {}
    for audit in sorted(raw, key=lambda a: (a.get("_mtime", 0), a.get("_path", ""))):
        fp = _window_fingerprint(audit)
        seen[fp] = audit
    return list(seen.values())


# ---------------------------------------------------------------------------
# Extract per-window metrics
# ---------------------------------------------------------------------------

def _load_baseline_mode() -> str:
    """Read baseline_model from config/forecaster_monitoring.yml; default BEST_SINGLE."""
    try:
        cfg_path = REPO_ROOT / "config" / "forecaster_monitoring.yml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return (
            (cfg or {})
            .get("forecaster_monitoring", {})
            .get("regression_metrics", {})
            .get("baseline_model", "BEST_SINGLE")
        ).strip().upper()
    except Exception:
        return "BEST_SINGLE"


def _get_regime(audit: dict) -> str | None:
    """Extract detected regime from runs list (first 'regime' model run)."""
    for run in audit.get("runs", []):
        if run.get("model") == "regime":
            return run.get("metadata", {}).get("regime")
    return None


def _parse_plausible_float(
    value: Any,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float | None:
    """Return a finite float inside the configured range, else None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if lower is not None and parsed < lower:
        return None
    if upper is not None and parsed > upper:
        return None
    return parsed


def extract_window_metrics(audit: dict, baseline_mode: str = "BEST_SINGLE") -> dict | None:
    """
    Extract per-window metrics from a forecast audit JSON.

    Returns None (with a WARNING log) if core fields are missing.
    Deterministic tie-breaking for best_single_model: min RMSE, then min sMAPE, then name.

    baseline_mode:
      "BEST_SINGLE"      — oracle: ex-post minimum RMSE across all single models (default)
      "EFFECTIVE_DEFAULT" — causal: the primary model chosen by the ensemble's own confidence
                            scores (artifacts.ensemble_selection.primary_model). Falls back to
                            BEST_SINGLE when the field is absent.
    """
    artifacts = audit.get("artifacts", {})
    eval_metrics = artifacts.get("evaluation_metrics", {})
    ensemble_weights = artifacts.get("ensemble_weights", {})
    ds = audit.get("dataset", {})
    summary = audit.get("summary", {})
    window_id = audit.get("_path", "?")

    if not eval_metrics:
        log.warning("Window %s: missing evaluation_metrics — skipped", window_id)
        return None

    model_metrics: dict[str, dict] = {}
    for m in MODELS:
        m_data = eval_metrics.get(m, {})
        rmse = _parse_plausible_float(m_data.get("rmse"), lower=1e-12, upper=100.0)
        if rmse is None:
            log.warning(
                "Window %s: implausible or missing rmse for model %s — skipped",
                window_id,
                m,
            )
            return None
        da_raw = m_data.get("directional_accuracy")
        da = (
            float("nan")
            if da_raw is None
            else _parse_plausible_float(da_raw, lower=0.0, upper=1.0)
        )
        if da_raw is not None and da is None:
            log.warning(
                "Window %s: implausible directional_accuracy for model %s — skipped",
                window_id,
                m,
            )
            return None
        smape = _parse_plausible_float(m_data.get("smape"), lower=0.0)
        model_metrics[m] = {
            "rmse": rmse,
            "da": da if da_raw is not None else float("nan"),
            "smape": smape if smape is not None else float("nan"),
        }

    ens_data = eval_metrics.get("ensemble", {})
    ensemble_rmse = _parse_plausible_float(
        ens_data.get("rmse"), lower=1e-12, upper=100.0
    )
    ensemble_da_raw = ens_data.get("directional_accuracy")
    ensemble_da = (
        float("nan")
        if ensemble_da_raw is None
        else _parse_plausible_float(ensemble_da_raw, lower=0.0, upper=1.0)
    )
    if ensemble_rmse is None:
        log.warning("Window %s: implausible or missing ensemble rmse — skipped", window_id)
        return None
    if ensemble_da_raw is not None and ensemble_da is None:
        log.warning("Window %s: implausible ensemble directional_accuracy — skipped", window_id)
        return None

    # Oracle best single: always the ex-post minimum-RMSE model from the MODELS tuple.
    # This field is independent of baseline_mode and is used by compute_per_model_summary
    # to count times_best_single / pct_best_single for per-model health reporting.
    oracle_best_model = min(
        MODELS,
        key=lambda m: (
            model_metrics[m]["rmse"],
            model_metrics[m]["smape"] if not math.isnan(model_metrics[m]["smape"]) else 1e9,
            m,
        ),
    )
    oracle_best_rmse = model_metrics[oracle_best_model]["rmse"]
    if ensemble_rmse > 3.0 * oracle_best_rmse:
        log.warning(
            "Window %s: ensemble rmse %.4f exceeds 3x best-single rmse %.4f — skipped",
            window_id,
            ensemble_rmse,
            oracle_best_rmse,
        )
        return None

    # Configured baseline: the model the ensemble is compared against for lift metrics.
    # EFFECTIVE_DEFAULT = primary_model chosen by the ensemble's own confidence scores
    #   (causal — measures "does blending add value over what we'd have picked anyway?").
    # BEST_SINGLE = oracle minimum-RMSE across MODELS (hindsight, unachievable for blend).
    # Falls back to oracle when EFFECTIVE_DEFAULT cannot be resolved from the audit payload.
    resolved_baseline_mode = "BEST_SINGLE"
    baseline_model_name: str = oracle_best_model
    baseline_rmse: float = oracle_best_rmse

    if baseline_mode == "EFFECTIVE_DEFAULT":
        ens_sel = artifacts.get("ensemble_selection") or {}
        raw_primary = str(ens_sel.get("primary_model", "") or "").lower().strip()
        if raw_primary:
            primary_data = eval_metrics.get(raw_primary)
            primary_rmse_val = primary_data.get("rmse") if isinstance(primary_data, dict) else None
            if primary_rmse_val is not None:
                baseline_model_name = raw_primary
                baseline_rmse = float(primary_rmse_val)
                resolved_baseline_mode = "EFFECTIVE_DEFAULT"

    baseline_rmse_ratio = ensemble_rmse / baseline_rmse if baseline_rmse > 0 else float("nan")

    return {
        "window_id": window_id,
        "ticker": ds.get("ticker"),
        "regime": _get_regime(audit),
        "window_start": ds.get("start"),
        "window_end": ds.get("end"),
        "n_obs": int(ds.get("length", 0)),
        "horizon": int(
            summary.get("forecast_horizon") or ds.get("forecast_horizon") or 0
        ),
        "model_metrics": model_metrics,
        "ensemble_rmse": ensemble_rmse,
        "ensemble_da": ensemble_da if ensemble_da_raw is not None else float("nan"),
        "ensemble_weights": {k: float(v) for k, v in (ensemble_weights or {}).items()},
        # Oracle fields — always reflect ex-post min-RMSE winner across MODELS.
        "best_single_model": oracle_best_model,
        "oracle_best_rmse": oracle_best_rmse,
        # Configured-baseline fields — reflect the baseline_mode in effect.
        "baseline_model_name": baseline_model_name,
        "baseline_rmse": baseline_rmse,
        "baseline_rmse_ratio": baseline_rmse_ratio,
        # Backward-compat alias: downstream that reads rmse_ratio gets baseline-relative value.
        "rmse_ratio": baseline_rmse_ratio,
        "baseline_mode": resolved_baseline_mode,
    }


# ---------------------------------------------------------------------------
# Per-model summary
# ---------------------------------------------------------------------------

def compute_per_model_summary(windows: list[dict]) -> dict[str, dict]:
    """Aggregate per-model metrics across all windows."""
    summary: dict[str, dict] = {}
    n_windows = len(windows)
    for m in MODELS:
        rmses = [w["model_metrics"][m]["rmse"] for w in windows]
        das = [
            w["model_metrics"][m]["da"]
            for w in windows
            if not math.isnan(w["model_metrics"][m]["da"])
        ]
        smapes = [
            w["model_metrics"][m]["smape"]
            for w in windows
            if not math.isnan(w["model_metrics"][m]["smape"])
        ]
        times_best = sum(1 for w in windows if w["best_single_model"] == m)
        da_zero = sum(1 for w in windows if w["model_metrics"][m]["da"] < 0.01)
        weights_used = [
            w["ensemble_weights"].get(m, 0.0)
            for w in windows
            if m in w.get("ensemble_weights", {})
        ]
        summary[m] = {
            "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "median_rmse": float(np.median(rmses)) if rmses else float("nan"),
            "mean_da": float(np.mean(das)) if das else float("nan"),
            "times_best_single": times_best,
            "pct_best_single": times_best / n_windows if n_windows > 0 else 0.0,
            "mean_weight_when_selected": float(np.mean(weights_used)) if weights_used else 0.0,
            "da_zero_windows": da_zero,
        }
    return summary


# ---------------------------------------------------------------------------
# Proxy Shapley attribution
# ---------------------------------------------------------------------------

def compute_shapley_attribution(windows: list[dict]) -> dict[str, float]:
    """
    Compute proxy Shapley values using RMSE as a scalar forecast proxy.

    CAUTION: realized values are not stored in audit JSONs.
    Proxy: actual=zeros(horizon), component_forecasts={model: full(horizon, rmse)}.
    Results are directional indicators of relative contribution — not causal attribution.
    Report marks them as 'proxy (use with caution)'.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from forcester_ts.shapley_attribution import ShapleyAttributor
    except ImportError as exc:
        log.warning(
            "ShapleyAttributor not available (%s) — returning zero Shapley values", exc
        )
        return {m: 0.0 for m in MODELS}

    attributor = ShapleyAttributor()
    per_model: dict[str, list[float]] = {m: [] for m in MODELS}

    for w in windows:
        horizon = max(w.get("horizon", 10), 1)
        mm = w["model_metrics"]
        component_forecasts = {m: np.full(horizon, mm[m]["rmse"]) for m in MODELS}
        weights = w.get("ensemble_weights") or {m: 1.0 / len(MODELS) for m in MODELS}
        actual = np.zeros(horizon)
        try:
            sv = attributor.compute(
                component_forecasts, weights, actual, loss_fn="mae"
            )
        except Exception as exc:
            log.debug("Shapley failed for window %s: %s", w["window_id"], exc)
            continue
        for m in MODELS:
            if m in sv:
                per_model[m].append(sv[m])

    return {
        m: float(np.mean(vals)) if vals else 0.0
        for m, vals in per_model.items()
    }


# ---------------------------------------------------------------------------
# Adaptive weights
# ---------------------------------------------------------------------------

def compute_adaptive_weights(
    windows: list[dict],
    recent_n: int = 20,
    lambda_decay: float = 1.0,
    da_floor: float = 0.10,
    da_cap_weight: float = 0.10,
) -> tuple[list[dict], dict]:
    """Compute adaptive candidate weights from the most recent `recent_n` audit windows.

    CONTRACT / INVARIANTS (machine-checked at runtime before return):
    - Every candidate in the returned list sums to 1.0 (within 1e-6).
    - No negative weights.
    - When ``degraded_da_fallback=True``, DA penalty was skipped (RMSE-only).
    - The primary candidate (index 0) respects the diversity guard:
        top model weight ≤ 0.90.
    - If all models have DA < da_floor, params["degraded_da_fallback"] = True.
    - da_floor > 0 and da_cap_weight > 0 are assumed (not validated here).
    Violation of any invariant is logged as ERROR and self-corrected.

    Algorithm:
      1. Take last recent_n windows (sorted by window_id timestamp order)
      2. For each model: mean_rmse over recent windows
      3. rmse_median = median across models
      4. raw_weight = exp(-lambda * mean_rmse / rmse_median)
      5. Hard zero if mean_rmse > 1.2 * rmse_median
      6. DA penalty: if mean_da < da_floor, cap at da_cap_weight
         All-DA-zero fallback: if ALL models have DA < da_floor, skip penalty
         and use RMSE-only weighting, recording degraded_da_fallback=True
      7. Normalize to sum=1.0
      8. Diversity guard: clamp top weight to 0.90, redistribute excess proportionally
      9. Return 3 candidates: primary adaptive, top-2 hedge, pure winner

    Returns:
        (candidates_list, params_dict)
    """
    sorted_windows = sorted(windows, key=lambda w: (w.get("window_id", ""), w.get("window_end", "")))
    recent = sorted_windows[-recent_n:] if len(sorted_windows) > recent_n else sorted_windows

    if not recent:
        return [], {}

    model_rmses = {m: [w["model_metrics"][m]["rmse"] for w in recent] for m in MODELS}
    model_das = {
        m: [
            w["model_metrics"][m]["da"]
            for w in recent
            if not math.isnan(w["model_metrics"][m]["da"])
        ]
        for m in MODELS
    }

    mean_rmse = {
        m: float(np.mean(vals)) if vals else float("inf")
        for m, vals in model_rmses.items()
    }
    mean_da = {
        m: float(np.mean(vals)) if vals else 0.0
        for m, vals in model_das.items()
    }

    finite_rmses = [v for v in mean_rmse.values() if math.isfinite(v)]
    rmse_median = float(np.median(finite_rmses)) if finite_rmses else 1.0

    # Exp-decay weights with hard zero for outlier RMSE
    raw_weights: dict[str, float] = {}
    for m in MODELS:
        if not math.isfinite(mean_rmse[m]) or mean_rmse[m] > 1.2 * rmse_median:
            raw_weights[m] = 0.0
        else:
            ratio = mean_rmse[m] / rmse_median if rmse_median > 0 else 1.0
            raw_weights[m] = math.exp(-lambda_decay * ratio)

    # All-DA-zero fallback
    all_da_zero = all(mean_da.get(m, 0.0) < da_floor for m in MODELS)
    degraded_da_fallback = False
    if all_da_zero:
        log.warning(
            "[WARN] All models have mean DA < %.2f — skipping DA penalty (RMSE-only weighting)",
            da_floor,
        )
        degraded_da_fallback = True
    else:
        # Pre-normalization DA cap
        for m in MODELS:
            if mean_da.get(m, 1.0) < da_floor and raw_weights[m] > da_cap_weight:
                log.warning(
                    "[WARN] DA floor clamp: %s mean_da=%.4f < da_floor=%.2f — "
                    "capping raw_weight %.4f -> %.4f (da_cap_weight)",
                    m, mean_da.get(m, 0.0), da_floor, raw_weights[m], da_cap_weight,
                )
                raw_weights[m] = da_cap_weight

    # Normalize
    total = sum(raw_weights.values())
    if total <= 0:
        adaptive: dict[str, float] = {m: 1.0 / len(MODELS) for m in MODELS}
    else:
        adaptive = {m: raw_weights[m] / total for m in MODELS}

    # Post-normalization DA cap: hard-set capped models and proportionally redistribute
    # the remaining budget ONLY to non-penalized models (DA >= da_floor).
    # Redistributing to other penalized-but-below-cap models would allow them to grow
    # above da_cap_weight, violating the cap contract.
    if not all_da_zero:
        penalized_set = {m for m in MODELS if mean_da.get(m, 1.0) < da_floor}
        capped_set = {m for m in penalized_set if adaptive[m] > da_cap_weight}
        if capped_set:
            for m in capped_set:
                adaptive[m] = da_cap_weight
            # Remaining budget = total budget for non-penalized models.
            # Must use ALL penalized weights (not just capped) to compute the pool.
            remaining = max(0.0, 1.0 - sum(adaptive[m] for m in penalized_set))
            non_penalized = {m: adaptive[m] for m in MODELS if m not in penalized_set}
            # `all_da_zero=False` guarantees non_penalized is non-empty, but be defensive.
            if not non_penalized:
                total_penalized = sum(adaptive[m] for m in penalized_set)
                if total_penalized > 0:
                    for m in penalized_set:
                        adaptive[m] = adaptive[m] / total_penalized
            else:
                np_total = sum(non_penalized.values())
                if np_total > 0:
                    for m in non_penalized:
                        adaptive[m] = non_penalized[m] / np_total * remaining
                else:
                    share = remaining / len(non_penalized)
                    for m in non_penalized:
                        adaptive[m] = share

    # Diversity guard: clamp top weight to 0.90
    diversity_clamped = False
    top_model = max(adaptive, key=lambda m: adaptive[m])
    if adaptive[top_model] > 0.90:
        excess = adaptive[top_model] - 0.90
        original_top = adaptive[top_model]
        adaptive[top_model] = 0.90
        others = {m: w for m, w in adaptive.items() if m != top_model}
        others_total = sum(others.values())
        if others_total > 0:
            for m in others:
                adaptive[m] += excess * (others[m] / others_total)
        else:
            for m in others:
                adaptive[m] += excess / len(others)
        diversity_clamped = True
        log.warning(
            "[WARN] Diversity guard: clamped %s weight %.3f -> 0.90",
            top_model,
            original_top,
        )

    # Build 3 candidates
    sorted_models = sorted(MODELS, key=lambda m: adaptive[m], reverse=True)
    top1, top2 = sorted_models[0], sorted_models[1]

    candidates = [
        # Use 6 dp so exact binary fractions (e.g. 0.109375 = 7/64) are not rounded
        # up past da_cap_weight.  round(..., 4) would turn 0.109375 into 0.1094 which
        # violates the cap contract.
        {m: round(adaptive[m], 6) for m in MODELS if adaptive[m] > 0.001},
        {top1: 0.70, top2: 0.30},
        {top1: 1.0},
    ]

    params: dict[str, Any] = {
        "recent_n": recent_n,
        "lambda_decay": lambda_decay,
        "da_floor": da_floor,
        "da_cap_weight": da_cap_weight,
        "diversity_clamped": diversity_clamped,
        "degraded_da_fallback": degraded_da_fallback,
        "mean_rmse": {m: round(mean_rmse[m], 4) for m in MODELS},
        "mean_da": {m: round(mean_da.get(m, 0.0), 4) for m in MODELS},
    }

    # Runtime invariant guard: every candidate must sum to 1.0 and be non-negative.
    for i, cand in enumerate(candidates):
        neg = {m: v for m, v in cand.items() if v < 0}
        if neg:
            log.error(
                "compute_adaptive_weights: candidate[%d] has negative weights %s — "
                "setting to 0 (bug in diversity/DA logic).",
                i, neg,
            )
            candidates[i] = {m: max(v, 0.0) for m, v in cand.items()}
            cand = candidates[i]
        total = sum(cand.values())
        if abs(total - 1.0) > 1e-4:
            log.error(
                "compute_adaptive_weights: candidate[%d] sums to %.6f (expected 1.0) — "
                "renormalizing (bug in distribution logic).",
                i, total,
            )
            if total > 0:
                candidates[i] = {m: v / total for m, v in cand.items()}

    return candidates, params


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _lift_fraction(windows: list[dict]) -> float:
    if not windows:
        return 0.0
    lifting = sum(
        1
        for w in windows
        if not math.isnan(w.get("rmse_ratio", float("nan"))) and w["rmse_ratio"] < 1.0
    )
    return lifting / len(windows)


def compute_lift_significance(
    windows: list[dict],
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    min_windows: int = 5,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for ensemble lift over the configured baseline RMSE.

    Per-window lift delta_i = baseline_rmse_i - ensemble_rmse_i.
    Positive delta means ensemble wins that window.

    Preferred window fields:
        baseline_rmse, ensemble_rmse

    Backward-compatible legacy fields:
        best_single_rmse, ensemble_rmse
        rmse_ratio (proxy only when direct RMSE values are absent)

    Returns a dict with keys:
        mean_lift           -- mean(delta) over valid windows
        ci_low              -- lower bootstrap percentile
        ci_high             -- upper bootstrap percentile
        lift_win_fraction   -- fraction of windows with delta > 0
        n_windows           -- number of valid windows used
        insufficient_data   -- True when n_valid < min_windows

    All float fields are NaN when insufficient_data=True.
    Invariants (when sufficient data):
        ci_low <= mean_lift <= ci_high
        0.0 <= lift_win_fraction <= 1.0
    """
    # Extract per-window lift deltas; skip NaN/inf entries.
    deltas: list[float] = []
    for w in windows:
        best = w.get("baseline_rmse")
        if best is None:
            best = w.get("best_single_rmse")
        ens = w.get("ensemble_rmse")
        if best is None or ens is None:
            # Fall back to rmse_ratio if direct RMSE values missing.
            ratio = w.get("rmse_ratio")
            if ratio is None or not math.isfinite(ratio):
                continue
            # delta sign: ratio < 1 means ensemble wins, so delta = 1 - ratio as proxy.
            # We use actual RMSE when available; ratio-derived delta for legacy windows.
            deltas.append(1.0 - ratio)
        else:
            d = float(best) - float(ens)
            if math.isfinite(d):
                deltas.append(d)

    n_valid = len(deltas)
    nan = float("nan")
    if n_valid < min_windows:
        return {
            "mean_lift": nan,
            "ci_low": nan,
            "ci_high": nan,
            "lift_win_fraction": 0.0 if n_valid == 0 else sum(1 for d in deltas if d > 0) / n_valid,
            "n_windows": n_valid,
            "insufficient_data": True,
        }

    arr = np.array(deltas, dtype=float)
    mean_lift = float(arr.mean())
    lift_win_fraction = float((arr > 0).mean())

    # Bootstrap CI: resample with replacement, compute mean each time.
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(arr, size=n_valid, replace=True)
        boot_means[b] = sample.mean()

    alpha = (1.0 - confidence_level) / 2.0
    ci_low = float(np.percentile(boot_means, 100.0 * alpha))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    return {
        "mean_lift": mean_lift,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "lift_win_fraction": lift_win_fraction,
        "n_windows": n_valid,
        "insufficient_data": False,
    }


def generate_markdown_report(
    windows: list[dict],
    summary: dict[str, dict],
    shapley: dict[str, float],
    adaptive_weights: list[dict],
    params: dict,
    duplicate_count: int,
) -> str:
    lift = _lift_fraction(windows)
    _ens_rmse_vals = [
        float(w["ensemble_rmse"])
        for w in windows
        if math.isfinite(float(w.get("ensemble_rmse", float("nan"))))
    ]
    ensemble_loss = float(np.mean(_ens_rmse_vals)) if _ens_rmse_vals else 1.0
    lines = [
        "# Ensemble Health Audit",
        f"**Generated**: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"**Audit windows**: {len(windows)} (after deduplication; {duplicate_count} duplicates found)",
        f"**Lift fraction** (ensemble RMSE < best single): {lift:.1%} (required: 25%)",
        "",
        "## Per-Model Summary",
        "",
        "| Model | Avg RMSE | Median RMSE | Avg DA | DA=0 Windows | Times Best Single | Avg Weight |",
        "|-------|----------|-------------|--------|--------------|-------------------|------------|",
    ]
    for m in MODELS:
        s = summary[m]
        lines.append(
            f"| {m} | {s['mean_rmse']:.4f} | {s['median_rmse']:.4f} | "
            f"{s['mean_da']:.3f} | {s['da_zero_windows']} | "
            f"{s['times_best_single']} ({s['pct_best_single']:.0%}) | "
            f"{s['mean_weight_when_selected']:.3f} |"
        )
    lines.append("")

    samossa_da_zero = summary.get("samossa", {}).get("da_zero_windows", 0)
    if samossa_da_zero > 5:
        lines += [
            f"> **[WARNING] SAMOSSA DA=0 anomaly**: DA < 0.01 in {samossa_da_zero} windows.",
            "> Near-flat forecasts win on RMSE but contribute zero directional signal.",
            f"> DA penalty caps SAMOSSA weight at {params.get('da_cap_weight', 0.10):.0%} "
            f"when mean DA < {params.get('da_floor', 0.10):.0%}.",
            "",
        ]

    lines += [
        "## Proxy Shapley Attribution",
        "",
        "> **Caution**: Proxy Shapley uses RMSE as scalar forecast proxy (no realized values).",
        "> Results are directional indicators of relative contribution — not causal attribution.",
        "",
        "| Model | Mean Shapley | Interpretation |",
        "|-------|-------------|----------------|",
    ]
    for m in MODELS:
        sv = shapley.get(m, 0.0)
        if abs(sv) > 0.05 * ensemble_loss:
            interp = f"dominant {'driver' if sv > 0 else 'reducer'}"
        else:
            interp = "neutral"
        lines.append(f"| {m} | {sv:.4f} | {interp} |")
    lines.append("")

    lines += [
        "## Adaptive Candidate Weights",
        "",
        f"Computed from last {params.get('recent_n', 20)} windows "
        f"(lambda_decay={params.get('lambda_decay', 1.0)}, "
        f"da_floor={params.get('da_floor', 0.10)}).",
    ]
    if params.get("diversity_clamped"):
        lines.append("> [INFO] Diversity guard applied: top model clamped to weight=0.90.")
    if params.get("degraded_da_fallback"):
        lines.append(
            "> [WARN] Degraded DA fallback: all models DA < floor — RMSE-only weighting used."
        )
    lines.append("")
    for i, cand in enumerate(adaptive_weights):
        label = ["primary adaptive", "top-2 hedge", "pure winner"][i] if i < 3 else f"candidate {i+1}"
        lines.append(f"- **{label}**: `{json.dumps(cand)}`")
    lines.append("")

    lines += [
        "## Model Diagnostics (mean RMSE & DA over recent windows)",
        "",
    ]
    for m in MODELS:
        p = params.get("mean_rmse", {}).get(m, float("nan"))
        d = params.get("mean_da", {}).get(m, float("nan"))
        p_str = f"{p:.4f}" if math.isfinite(p) else "N/A"
        d_str = f"{d:.3f}" if math.isfinite(d) else "N/A"
        lines.append(f"- **{m}**: RMSE={p_str}, DA={d_str}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def update_config_adaptive_weights(
    config_path: Path,
    adaptive_weights: list[dict],
    params: dict,
) -> None:
    """Write adaptive_candidate_weights section to forecasting_config.yml.

    Does NOT overwrite the static candidate_weights (rollback safety).
    Uses plain yaml.dump — comment preservation is best-effort only.
    """
    try:
        config_text = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(config_text) or {}
    except Exception as exc:
        log.error("Failed to read config %s: %s", config_path, exc)
        return

    config["adaptive_candidate_weights"] = {
        "computed_at": datetime.date.today().isoformat(),
        "recent_n": params.get("recent_n", 20),
        "lambda_decay": params.get("lambda_decay", 1.0),
        "da_floor": params.get("da_floor", 0.10),
        "da_cap_weight": params.get("da_cap_weight", 0.10),
        "weights": adaptive_weights,
    }

    try:
        config_path.write_text(
            yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        log.info("Updated adaptive_candidate_weights in %s", config_path)
    except Exception as exc:
        log.error("Failed to write config %s: %s", config_path, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ensemble health audit with adaptive weight computation."
    )
    parser.add_argument(
        "--audit-dir",
        default=str(REPO_ROOT / "logs" / "forecast_audits"),
        help="Directory containing forecast_audit_*.json files",
    )
    parser.add_argument(
        "--recent-n",
        type=int,
        default=20,
        help="Number of recent windows for adaptive weight computation",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write adaptive_candidate_weights to forecasting_config.yml",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write markdown report to logs/ensemble_health/",
    )
    parser.add_argument(
        "--config-path",
        default=str(FORECASTING_CONFIG),
        help="Path to forecasting_config.yml",
    )
    args = parser.parse_args(argv)

    audit_dir = Path(args.audit_dir)
    if not audit_dir.exists():
        log.error("Audit directory not found: %s", audit_dir)
        return 1

    # Count duplicates before dedup for report
    all_raw = load_audit_windows(audit_dir, dedupe=False)
    all_deduped = load_audit_windows(audit_dir, dedupe=True)
    duplicate_count = len(all_raw) - len(all_deduped)
    log.info(
        "Loaded %d audit files (%d duplicates removed by fingerprint)",
        len(all_raw),
        duplicate_count,
    )

    baseline_mode = _load_baseline_mode()
    log.info("Layer 1 baseline mode: %s", baseline_mode)
    windows: list[dict] = []
    for audit in all_deduped:
        metrics = extract_window_metrics(audit, baseline_mode=baseline_mode)
        if metrics is not None:
            windows.append(metrics)
    log.info("Extracted valid metrics from %d/%d windows", len(windows), len(all_deduped))

    if not windows:
        log.error("No valid windows to process. Check audit directory: %s", audit_dir)
        return 1

    summary = compute_per_model_summary(windows)
    shapley = compute_shapley_attribution(windows)
    candidates, params = compute_adaptive_weights(
        windows,
        recent_n=args.recent_n,
        lambda_decay=1.0,
        da_floor=0.10,
        da_cap_weight=0.10,
    )

    # Terminal summary
    log.info("=== Per-model summary (%d windows) ===", len(windows))
    for m in MODELS:
        s = summary[m]
        log.info(
            "  %s: mean_rmse=%.4f  mean_da=%.3f  da_zero=%d  times_best=%d (%.0f%%)",
            m,
            s["mean_rmse"],
            s["mean_da"],
            s["da_zero_windows"],
            s["times_best_single"],
            s["pct_best_single"] * 100,
        )
    log.info("=== Adaptive candidate weights ===")
    for i, cand in enumerate(candidates):
        labels = ["primary", "top-2 hedge", "pure winner"]
        log.info("  %s: %s", labels[i] if i < 3 else str(i), cand)
    lift = _lift_fraction(windows)
    log.info("=== Lift fraction: %.1f%% (required: 25%%) ===", lift * 100)

    if args.write_report:
        report_text = generate_markdown_report(
            windows, summary, shapley, candidates, params, duplicate_count
        )
        ENSEMBLE_HEALTH_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = ENSEMBLE_HEALTH_DIR / f"ensemble_health_{ts}.md"
        report_path.write_text(report_text, encoding="utf-8")
        log.info("Report written: %s", report_path)

    if args.write_config and candidates:
        update_config_adaptive_weights(Path(args.config_path), candidates, params)

    # Golden-metric structured log — parseable by CI, monitoring, and OpenClaw.
    # Alert externally if lift_fraction or samossa_da degraded vs expected band.
    _samossa = summary.get("samossa", {})
    _primary = candidates[0] if candidates else {}
    golden = {
        "lift_fraction": round(lift, 3),
        "n_windows": len(windows),
        "samossa_mean_da": round(_samossa.get("mean_da", 0.0), 3),
        "samossa_da_zero_pct": round(
            _samossa.get("da_zero_windows", 0) / max(len(windows), 1), 3
        ),
        "samossa_weight_primary": round(_primary.get("samossa", 0.0), 3),
        "degraded_da_fallback": params.get("degraded_da_fallback", False),
        "diversity_clamped": params.get("diversity_clamped", False),
    }
    log.info("GOLDEN_METRICS %s", json.dumps(golden, sort_keys=True))

    # Alert thresholds (informational — not CI-blocking)
    if golden["lift_fraction"] < 0.05 and golden["n_windows"] >= 10:
        log.warning(
            "[ALERT] lift_fraction=%.1f%% is below 5%% with %d windows. "
            "Ensemble provides no measurable benefit — investigate model composition.",
            golden["lift_fraction"] * 100, golden["n_windows"],
        )
    if golden["samossa_da_zero_pct"] > 0.40:
        log.warning(
            "[ALERT] SAMOSSA DA=0 in %.0f%% of windows — DA penalty is active. "
            "Primary candidate caps SAMOSSA at %.0f%%.",
            golden["samossa_da_zero_pct"] * 100,
            golden["samossa_weight_primary"] * 100,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
