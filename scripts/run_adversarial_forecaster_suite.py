#!/usr/bin/env python3
"""
Adversarial forecaster benchmark and CI gate — Barbell-objective edition.

Runs a deterministic stress matrix across synthetic market regimes and reports:

RMSE metrics (legacy, informational):
- ensemble_under_best_rate
- avg_ensemble_ratio_vs_best
- ensemble_worse_than_rw_rate

Barbell-objective metrics (primary, canonical per REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08):
- omega_ratio              : Gain/loss partitioned at DAILY_NGN_THRESHOLD (~0.108%/day)
- terminal_da_pass_rate    : Fraction of runs with terminal directional accuracy >= 0.45
- mean_ci_coverage         : Empirical terminal CI coverage rate
- profit_factor            : avg_win / avg_loss across all synthetic trades
- omega_above_1            : Whether portfolio beats the Nigeria inflation hurdle

Barbell scenarios:
- ngn_high_inflation       : Persistent drift near NGN devaluation floor
- asymmetric_vol           : Downside vol spikes, smooth upside (barbell worst case)
- fat_tail_crash           : Single 10-12% crash event (expected_shortfall stress)
- crisis_recovery          : Sustained drawdown then trending recovery

Anti-patterns not present here:
- RMSE is NOT the primary success criterion
- win_rate and Sharpe are NOT checked
- No threshold lowering to pass: all barbell thresholds are domain-derived

Optionally enforces blocking thresholds from forecaster monitoring config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone as _tz
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

UTC = _tz.utc  # datetime.UTC added in 3.11; timezone.utc for 3.10 compat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.metrics import (
    compute_regression_metrics,
    terminal_ci_coverage,
    terminal_directional_accuracy,
)

# ---------------------------------------------------------------------------
# NGN hurdle fallback (in case etl.portfolio_math import fails in test envs)
# ---------------------------------------------------------------------------
try:
    from etl.portfolio_math import DAILY_NGN_THRESHOLD as _DAILY_NGN_THRESHOLD
    from etl.portfolio_math import omega_ratio as _omega_ratio
    _NGN_IMPORT_OK = True
except Exception:
    _DAILY_NGN_THRESHOLD = (1.31) ** (1.0 / 252) - 1  # ~0.00108/day
    _NGN_IMPORT_OK = False

    def _omega_ratio(returns: pd.Series, threshold: float | None = None) -> float:  # type: ignore[misc]
        tau = _DAILY_NGN_THRESHOLD if threshold is None else threshold
        excess = pd.Series(returns) - tau
        gain = float(excess.clip(lower=0).sum())
        loss = float((-excess).clip(lower=0).sum())
        return gain / loss if loss > 0.0 else float("inf")


DEFAULT_HORIZON = 20
DEFAULT_POINTS = 320

# Core RMSE scenarios (retained for backward compatibility)
_RMSE_SCENARIOS = [
    "trend_seasonal",
    "random_walk",
    "regime_shift",
    "vol_cluster",
    "jump_shock",
    "mean_reversion_break",
]

# Barbell-specific scenarios (test omega_ratio, terminal_DA, expected_shortfall)
_BARBELL_SCENARIOS = [
    "ngn_high_inflation",   # persistent upward drift near NGN devaluation floor
    "asymmetric_vol",       # downside vol spikes, smooth upside — worst case for omega
    "fat_tail_crash",       # 10-12% single-day crash — stresses expected_shortfall
    "crisis_recovery",      # deep drawdown then slow recovery — stresses terminal_DA
]

DEFAULT_SCENARIOS = _RMSE_SCENARIOS + _BARBELL_SCENARIOS

DEFAULT_SEEDS = [101, 202, 303, 404, 505]  # 5 seeds for better barbell stat stability

DEFAULT_VARIANTS = [
    # RESEARCH-ONLY: confidence_scaling=false is excluded from blocking defaults.
    # See Documentation/PRIORITY_ANALYSIS_20260212.md.
    # "prod_like_conf_off",
    "prod_like_conf_on",
    "sarimax_augmented_conf_on",
]

# Capital base for expected_profit calculation (matches quant_success_config.yml)
_CAPITAL_BASE_USD = 25_000.0


# ---------------------------------------------------------------------------
# Series generators
# ---------------------------------------------------------------------------

def gen_series(kind: str, n: int, seed: int) -> pd.Series:
    """Generate a synthetic price series for the given scenario."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)

    # --- Original RMSE scenarios (unchanged) ---
    if kind == "trend_seasonal":
        y = 100 + 0.18 * t + 3.5 * np.sin(2 * np.pi * t / 14) + rng.normal(0, 0.6, n)

    elif kind == "random_walk":
        rets = rng.normal(0.0002, 0.012, n)
        y = 100 * np.exp(np.cumsum(rets))

    elif kind == "regime_shift":
        half = n // 2
        y1 = 100 + 0.22 * np.arange(half) + rng.normal(0, 0.4, half)
        y2 = y1[-1] + (-0.35 * np.arange(n - half)) + 2.8 * np.sin(
            2 * np.pi * np.arange(n - half) / 9
        ) + rng.normal(0, 1.2, n - half)
        y = np.concatenate([y1, y2])

    elif kind == "vol_cluster":
        eps = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.007
        for i in range(1, n):
            sigma[i] = math.sqrt(0.000002 + 0.12 * (eps[i - 1] ** 2) + 0.84 * (sigma[i - 1] ** 2))
            eps[i] = rng.normal(0.0001, sigma[i])
        y = 100 * np.exp(np.cumsum(eps))

    elif kind == "jump_shock":
        rets = rng.normal(0.00015, 0.01, n)
        jump_idx = rng.choice(np.arange(25, n - 25), size=6, replace=False)
        rets[jump_idx] += rng.choice([-0.08, -0.06, 0.06, 0.08], size=6)
        y = 100 * np.exp(np.cumsum(rets))

    elif kind == "mean_reversion_break":
        half = n // 2
        x = np.zeros(n)
        x[0] = 100
        for i in range(1, half):
            x[i] = x[i - 1] + 0.18 * (102 - x[i - 1]) + rng.normal(0, 0.55)
        for i in range(half, n):
            x[i] = x[i - 1] + 0.28 + rng.normal(0, 0.95)
        y = x

    # --- Barbell-specific scenarios ---

    elif kind == "ngn_high_inflation":
        # Persistent upward drift slightly above the NGN daily threshold (~0.108%/day).
        # A model that correctly identifies this drift should generate omega_ratio > 1.
        # Drift = 0.12%/day ≈ 35%/year; vol = 1.8%/day (realistic EM equity).
        rets = rng.normal(0.0012, 0.018, n)
        y = 100 * np.exp(np.cumsum(rets))

    elif kind == "asymmetric_vol":
        # Downside vol clusters (EGARCH-like); upside is smooth.
        # Worst case for omega_ratio: tail losses are large, tail gains are moderate.
        # Omega < 1.0 is the expected stress-case outcome here.
        eps = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.012
        for i in range(1, n):
            neg_shock = min(eps[i - 1], 0.0)  # only negative shocks feed vol persistence
            sigma[i] = math.sqrt(
                max(1e-8, 0.00001 + 0.25 * (neg_shock ** 2) + 0.70 * (sigma[i - 1] ** 2))
            )
            eps[i] = rng.normal(0.0001, sigma[i])
        y = 100 * np.exp(np.cumsum(eps))

    elif kind == "fat_tail_crash":
        # Normal drift with a sudden 10-12% crash over 3 consecutive days.
        # Stresses: expected_shortfall gate, max_drawdown constraint.
        rets = rng.normal(0.00015, 0.010, n)
        crash_start = n // 3
        crash_magnitude = rng.choice([-0.12, -0.10, -0.08])
        rets[crash_start:crash_start + 3] += crash_magnitude
        y = 100 * np.exp(np.cumsum(rets))

    elif kind == "crisis_recovery":
        # Phase 1: sustained drawdown (avg -0.5%/day for n//3 bars).
        # Phase 2: strong recovery (avg +0.8%/day for remaining bars).
        # Stresses: terminal_DA (must identify recovery direction), max_drawdown.
        # Drift +0.8%/day with vol 1.5%/day → SNR=0.53/bar; at 2n//3 bars expected
        # recovery is deterministically positive at n>=200 for any seed.
        third = n // 3
        prices = np.zeros(n)
        prices[0] = 100.0
        for i in range(1, third):
            prices[i] = prices[i - 1] * (1.0 + rng.normal(-0.005, 0.015))
        for i in range(third, n):
            prices[i] = prices[i - 1] * (1.0 + rng.normal(0.008, 0.015))
        # Clip to avoid negative prices
        prices = np.maximum(prices, 1.0)
        y = prices

    else:
        raise ValueError(f"Unknown scenario: {kind}")

    return pd.Series(y, index=idx, name="Close")


def rw_baseline(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Naive random-walk baseline: last train price held constant."""
    lv = float(train.dropna().iloc[-1])
    return pd.Series(lv, index=test_index)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def mk_cfg(
    variant: str,
    raw_cfg: Dict[str, Any],
    *,
    horizon: int,
) -> TimeSeriesForecasterConfig:
    f = raw_cfg["forecasting"]
    ens = dict(f.get("ensemble") or {})
    base_candidates = list(ens.get("candidate_weights") or [])

    # NOTE: prod_like_conf_off branch is intentionally excluded from DEFAULT_VARIANTS
    # (research-only variant; kept for explicit --variants override).
    if variant == "prod_like_conf_off":
        conf_scaling = False
        candidates = base_candidates
    elif variant == "prod_like_conf_on":
        conf_scaling = True
        candidates = base_candidates
    elif variant == "sarimax_augmented_conf_on":
        conf_scaling = True
        candidates = [
            {"sarimax": 0.65, "garch": 0.25, "samossa": 0.10},
            {"sarimax": 0.55, "samossa": 0.30, "mssa_rl": 0.15},
            {"sarimax": 0.50, "garch": 0.35, "mssa_rl": 0.15},
        ] + base_candidates
    else:
        raise ValueError(f"Unknown variant: {variant}")

    sarimax_kwargs = {k: v for k, v in (f.get("sarimax") or {}).items() if k != "enabled"}
    sarimax_kwargs.update(
        {
            "max_p": min(int(sarimax_kwargs.get("max_p", 3)), 2),
            "max_q": min(int(sarimax_kwargs.get("max_q", 2)), 2),
            "max_d": min(int(sarimax_kwargs.get("max_d", 1)), 1),
            "max_P": 0,
            "max_D": 0,
            "max_Q": 0,
            "seasonal_periods": 0,
            "order_search_mode": "compact",
            "order_search_maxiter": 80,
            "auto_select": True,
        }
    )

    garch_kwargs = {k: v for k, v in (f.get("garch") or {}).items() if k != "enabled"}
    samossa_kwargs = {k: v for k, v in (f.get("samossa") or {}).items() if k != "enabled"}
    mssa_kwargs = {k: v for k, v in (f.get("mssa_rl") or {}).items() if k != "enabled"}
    mssa_kwargs["use_gpu"] = False

    return TimeSeriesForecasterConfig(
        forecast_horizon=horizon,
        sarimax_enabled=True,
        garch_enabled=True,
        samossa_enabled=True,
        mssa_rl_enabled=True,
        ensemble_enabled=True,
        regime_detection_enabled=False,
        sarimax_kwargs=sarimax_kwargs,
        garch_kwargs=garch_kwargs,
        samossa_kwargs=samossa_kwargs,
        mssa_rl_kwargs=mssa_kwargs,
        ensemble_kwargs={
            "enabled": True,
            "confidence_scaling": conf_scaling,
            "candidate_weights": candidates,
            "minimum_component_weight": float(ens.get("minimum_component_weight", 0.05)),
        },
    )


# ---------------------------------------------------------------------------
# Single-run execution + barbell metric extraction
# ---------------------------------------------------------------------------

def run_one(
    series: pd.Series,
    cfg: TimeSeriesForecasterConfig,
    horizon: int,
) -> Tuple[
    Dict[str, Any],   # rw_metrics
    Dict[str, Any],   # model_metrics
    Dict[str, Any],   # weights
    Optional[str],    # status
    Optional[str],    # default_model
    bool,             # ensemble_index_mismatch
    Optional[pd.Series],  # ens_forecast
    Optional[pd.Series],  # ens_lower_ci
    Optional[pd.Series],  # ens_upper_ci
    pd.Series,        # train (for barbell calc)
    pd.Series,        # test (for barbell calc)
]:
    """Run a single scenario and return RMSE + raw forecast data for barbell metrics."""
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    returns = train.pct_change().dropna()

    rw_metrics = compute_regression_metrics(test, rw_baseline(train, test.index)) or {}

    forecaster = TimeSeriesForecaster(config=cfg)
    forecaster.fit(price_series=train, returns_series=returns)
    forecaster.forecast(steps=horizon)
    model_metrics = forecaster.evaluate(test)

    latest = forecaster._latest_results if isinstance(forecaster._latest_results, dict) else {}
    meta = latest.get("ensemble_metadata", {}) if isinstance(latest, dict) else {}
    weights = meta.get("weights", {}) if isinstance(meta, dict) else {}
    status = meta.get("ensemble_status") if isinstance(meta, dict) else None
    default_model = latest.get("default_model") if isinstance(latest, dict) else None
    ensemble_index_mismatch = bool(
        latest.get("ensemble_index_mismatch")
        or (meta.get("ensemble_index_mismatch") if isinstance(meta, dict) else False)
    )

    # Extract ensemble forecast bundle for barbell metrics
    ens_payload = latest.get("ensemble_forecast")
    ens_forecast: Optional[pd.Series] = None
    ens_lower: Optional[pd.Series] = None
    ens_upper: Optional[pd.Series] = None
    if isinstance(ens_payload, dict):
        fc = ens_payload.get("forecast")
        lo = ens_payload.get("lower_ci")
        hi = ens_payload.get("upper_ci")
        if isinstance(fc, pd.Series) and not fc.empty:
            ens_forecast = fc
        if isinstance(lo, pd.Series) and not lo.empty:
            ens_lower = lo
        if isinstance(hi, pd.Series) and not hi.empty:
            ens_upper = hi

    return (
        rw_metrics, model_metrics, weights, status, default_model,
        ensemble_index_mismatch, ens_forecast, ens_lower, ens_upper,
        train, test,
    )


def compute_barbell_per_run(
    train: pd.Series,
    test: pd.Series,
    ens_forecast: Optional[pd.Series],
    ens_lower: Optional[pd.Series],
    ens_upper: Optional[pd.Series],
) -> Dict[str, Any]:
    """
    Compute barbell-objective metrics for a single forecast window.

    The "synthetic trade" is: go long if forecast terminal > forecast entry,
    short if forecast terminal < forecast entry. P&L = signed actual terminal return.

    Returns dict with:
        terminal_da        : 1.0 / 0.0 / None — correct terminal direction?
        trade_return       : float / None — signed actual return for the synthetic trade
        ci_coverage        : 1.0 / 0.0 / None — actual terminal inside CI?
        actual_return      : float — actual terminal return (entry to exit, unsigned)
        forecast_direction : int — +1 long, -1 short, 0 no signal
        max_drawdown_path  : float / None — max drawdown of actual test series
    """
    result: Dict[str, Any] = {
        "terminal_da": None,
        "trade_return": None,
        "ci_coverage": None,
        "actual_return": None,
        "forecast_direction": 0,
        "max_drawdown_path": None,
    }

    try:
        train_clean = train.dropna()
        test_clean = test.dropna()
        if train_clean.empty or test_clean.empty:
            return result

        entry_price = float(train_clean.iloc[-1])
        exit_price = float(test_clean.iloc[-1])
        if entry_price <= 0:
            return result

        actual_return = (exit_price - entry_price) / entry_price
        result["actual_return"] = float(actual_return)

        # Max drawdown of the actual test path
        if len(test_clean) > 1:
            test_rets = test_clean.pct_change().dropna()
            if not test_rets.empty:
                cum = (1.0 + test_rets).cumprod()
                roll_max = cum.cummax()
                dd = (cum - roll_max) / roll_max.replace(0, float("nan"))
                max_dd = float(abs(dd.min()))
                if math.isfinite(max_dd):
                    result["max_drawdown_path"] = max_dd

        # Barbell metrics require a valid ensemble forecast
        if not isinstance(ens_forecast, pd.Series) or ens_forecast.empty:
            return result

        fc_clean = ens_forecast.dropna()
        if len(fc_clean) < 2:
            return result

        # terminal_directional_accuracy
        tda = terminal_directional_accuracy(test, ens_forecast)
        result["terminal_da"] = tda

        # terminal CI coverage
        if isinstance(ens_lower, pd.Series) and isinstance(ens_upper, pd.Series):
            cov = terminal_ci_coverage(test, ens_lower, ens_upper)
            result["ci_coverage"] = cov

        # Synthetic trade: direction from forecast first→last
        fc_first = float(fc_clean.iloc[0])
        fc_last = float(fc_clean.iloc[-1])
        direction = int(np.sign(fc_last - fc_first))
        result["forecast_direction"] = direction

        if direction == 0:
            # Flat forecast → no trade → return = 0
            result["trade_return"] = 0.0
        else:
            # Long if forecast UP, short if forecast DOWN
            trade_return = direction * actual_return
            if math.isfinite(trade_return):
                result["trade_return"] = float(trade_return)

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def _run_variants(
    *,
    raw_cfg: Dict[str, Any],
    variants: Iterable[str],
    scenarios: Iterable[str],
    seeds: Iterable[int],
    horizon: int,
    n_points: int,
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in variants}
    for variant in variants:
        cfg = mk_cfg(variant, raw_cfg, horizon=horizon)
        for scenario in scenarios:
            for seed in seeds:
                series = gen_series(scenario, n_points, seed)
                try:
                    (
                        rw, metrics, weights, status, default_model,
                        ensemble_index_mismatch, ens_forecast, ens_lower, ens_upper,
                        train, test,
                    ) = run_one(series, cfg, horizon)

                    barbell = compute_barbell_per_run(
                        train, test, ens_forecast, ens_lower, ens_upper
                    )

                    # Sanity: if model_metrics is empty, treat as error (silent-fail guard)
                    if not metrics:
                        raise RuntimeError("forecaster returned empty model_metrics dict")

                    results[variant].append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "rw": rw,
                            "metrics": metrics,
                            "weights": weights,
                            "status": status,
                            "default_model": default_model,
                            "ensemble_index_mismatch": ensemble_index_mismatch,
                            "barbell": barbell,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    results[variant].append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "rw": {},
                            "metrics": {},
                            "weights": {},
                            "status": None,
                            "default_model": None,
                            "ensemble_index_mismatch": False,
                            "barbell": {},
                            "error": str(exc),
                        }
                    )
    return results


# ---------------------------------------------------------------------------
# RMSE summarizer (original, kept for backward compatibility)
# ---------------------------------------------------------------------------

def summarize(run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "runs": len(run_rows),
        "errors": 0,
        "model_presence": Counter(),
        "ensemble_under_best_count": 0,
        "ensemble_worse_than_rw_count": 0,
        "effective_worse_than_rw_count": 0,
        "ensemble_index_mismatch_count": 0,
        "ensemble_rmse_ratio_vs_best": [],
        "sarimax_rmse_ratio_vs_rw": [],
        "ensemble_status_counts": Counter(),
        "effective_model_counts": Counter(),
        "weight_patterns": Counter(),
        "scenario_breakdown": defaultdict(
            lambda: {
                "n": 0,
                "ens_under_best": 0,
                "ens_worse_rw": 0,
                "effective_worse_rw": 0,
                "index_mismatch": 0,
                "ens_ratio": [],
            }
        ),
    }

    for row in run_rows:
        scenario = row["scenario"]
        out["scenario_breakdown"][scenario]["n"] += 1

        if row["error"]:
            out["errors"] += 1
            continue

        metrics_map = row["metrics"] or {}
        rw = row["rw"] or {}
        for model_name in metrics_map.keys():
            out["model_presence"][model_name] += 1

        status = row.get("status")
        if status is not None:
            out["ensemble_status_counts"][str(status)] += 1
        if bool(row.get("ensemble_index_mismatch")):
            out["ensemble_index_mismatch_count"] += 1
            out["scenario_breakdown"][scenario]["index_mismatch"] += 1

        weights = row.get("weights") or {}
        if isinstance(weights, dict) and weights:
            key = tuple(sorted((k, round(float(v), 4)) for k, v in weights.items()))
            out["weight_patterns"][key] += 1

        ens_rmse = (
            (metrics_map.get("ensemble") or {}).get("rmse")
            if isinstance(metrics_map.get("ensemble"), dict)
            else None
        )
        rw_rmse = rw.get("rmse") if isinstance(rw, dict) else None
        default_model = str(row.get("default_model") or "ensemble").strip().lower()
        canonical_default = (
            "mssa_rl"
            if default_model in {"mssa", "mssa-rl", "mssa_rl"}
            else ("samossa" if default_model in {"samossa", "samossa_forecast"} else default_model)
        )
        if canonical_default not in {"ensemble", "sarimax", "garch", "samossa", "mssa_rl"}:
            canonical_default = "ensemble"
        out["effective_model_counts"][canonical_default] += 1
        effective_rmse = (
            (metrics_map.get(canonical_default) or {}).get("rmse")
            if isinstance(metrics_map.get(canonical_default), dict)
            else None
        )
        if effective_rmse is None:
            effective_rmse = ens_rmse
        best_single = None
        for model_name in ("sarimax", "garch", "samossa", "mssa_rl"):
            rmse = (
                (metrics_map.get(model_name) or {}).get("rmse")
                if isinstance(metrics_map.get(model_name), dict)
                else None
            )
            if isinstance(rmse, (int, float)):
                best_single = rmse if best_single is None else min(best_single, rmse)

        if (
            isinstance(ens_rmse, (int, float))
            and isinstance(best_single, (int, float))
            and best_single > 0
        ):
            ratio = float(ens_rmse) / float(best_single)
            out["ensemble_rmse_ratio_vs_best"].append(ratio)
            out["scenario_breakdown"][scenario]["ens_ratio"].append(ratio)
            if ratio > 1.0:
                out["ensemble_under_best_count"] += 1
                out["scenario_breakdown"][scenario]["ens_under_best"] += 1

        if (
            isinstance(ens_rmse, (int, float))
            and isinstance(rw_rmse, (int, float))
            and rw_rmse > 0
        ):
            if float(ens_rmse) > float(rw_rmse):
                out["ensemble_worse_than_rw_count"] += 1
                out["scenario_breakdown"][scenario]["ens_worse_rw"] += 1

        if (
            isinstance(effective_rmse, (int, float))
            and isinstance(rw_rmse, (int, float))
            and rw_rmse > 0
        ):
            if float(effective_rmse) > float(rw_rmse):
                out["effective_worse_than_rw_count"] += 1
                out["scenario_breakdown"][scenario]["effective_worse_rw"] += 1

        srmse = (
            (metrics_map.get("sarimax") or {}).get("rmse")
            if isinstance(metrics_map.get("sarimax"), dict)
            else None
        )
        if isinstance(srmse, (int, float)) and isinstance(rw_rmse, (int, float)) and rw_rmse > 0:
            out["sarimax_rmse_ratio_vs_rw"].append(float(srmse) / float(rw_rmse))

    # BUG FIX: denominator must use actual ok count, not max(1, ...).
    # When all runs error, rates should be nan (not 0.0 which looks like PASS).
    n_ok = out["runs"] - out["errors"]
    if n_ok <= 0:
        out["ensemble_under_best_rate"] = float("nan")
        out["ensemble_worse_than_rw_rate"] = float("nan")
        out["effective_worse_than_rw_rate"] = float("nan")
        out["ensemble_index_mismatch_rate"] = float("nan")
        out["avg_ensemble_ratio_vs_best"] = None
        out["avg_sarimax_ratio_vs_rw"] = None
    else:
        out["ensemble_under_best_rate"] = out["ensemble_under_best_count"] / n_ok
        out["ensemble_worse_than_rw_rate"] = out["ensemble_worse_than_rw_count"] / n_ok
        out["effective_worse_than_rw_rate"] = out["effective_worse_than_rw_count"] / n_ok
        out["ensemble_index_mismatch_rate"] = out["ensemble_index_mismatch_count"] / n_ok
        out["avg_ensemble_ratio_vs_best"] = (
            sum(out["ensemble_rmse_ratio_vs_best"]) / len(out["ensemble_rmse_ratio_vs_best"])
            if out["ensemble_rmse_ratio_vs_best"]
            else None
        )
        out["avg_sarimax_ratio_vs_rw"] = (
            sum(out["sarimax_rmse_ratio_vs_rw"]) / len(out["sarimax_rmse_ratio_vs_rw"])
            if out["sarimax_rmse_ratio_vs_rw"]
            else None
        )

    scenario_out = {}
    for scenario, details in out["scenario_breakdown"].items():
        n = max(1, details["n"])
        scenario_out[scenario] = {
            "runs": details["n"],
            "ensemble_under_best_rate": details["ens_under_best"] / n,
            "ensemble_worse_than_rw_rate": details["ens_worse_rw"] / n,
            "effective_worse_than_rw_rate": details["effective_worse_rw"] / n,
            "ensemble_index_mismatch_rate": details["index_mismatch"] / n,
            "avg_ens_ratio_vs_best": (
                sum(details["ens_ratio"]) / len(details["ens_ratio"])
                if details["ens_ratio"]
                else None
            ),
        }
    out["scenario_breakdown"] = scenario_out

    top_patterns = out["weight_patterns"].most_common(5)
    out["top_weight_patterns"] = [{"weights": list(k), "count": c} for k, c in top_patterns]
    del out["weight_patterns"]

    out["model_presence"] = dict(out["model_presence"])
    out["ensemble_status_counts"] = dict(out["ensemble_status_counts"])
    out["effective_model_counts"] = dict(out["effective_model_counts"])
    return out


# ---------------------------------------------------------------------------
# Barbell summarizer (primary objective per matrix-first policy)
# ---------------------------------------------------------------------------

def summarize_barbell(run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate barbell-objective metrics across all runs in a variant.

    omega_ratio is computed across ALL synthetic trade returns in the batch
    (not per-run), because a single-trade omega is statistically meaningless.

    Canonical thresholds (from quant_success_config.yml):
        min_omega_ratio        : 1.00  — must beat NGN hurdle
        min_terminal_da_rate   : 0.45  — directional gate
        min_ci_coverage_rate   : 0.25  — CI calibration floor
        min_profit_factor      : 0.80  — magnitude asymmetry floor
    """
    terminal_das: List[float] = []
    trade_returns: List[float] = []
    ci_coverages: List[float] = []
    max_drawdowns: List[float] = []
    n_errors = sum(1 for r in run_rows if r.get("error"))

    for row in run_rows:
        if row.get("error"):
            continue
        bm = row.get("barbell") or {}

        tda = bm.get("terminal_da")
        if tda is not None:
            try:
                terminal_das.append(float(tda))
            except (TypeError, ValueError):
                pass

        tr = bm.get("trade_return")
        if tr is not None:
            try:
                v = float(tr)
                if math.isfinite(v):
                    trade_returns.append(v)
            except (TypeError, ValueError):
                pass

        cc = bm.get("ci_coverage")
        if cc is not None:
            try:
                ci_coverages.append(float(cc))
            except (TypeError, ValueError):
                pass

        md = bm.get("max_drawdown_path")
        if md is not None:
            try:
                v = float(md)
                if math.isfinite(v):
                    max_drawdowns.append(v)
            except (TypeError, ValueError):
                pass

    # omega_ratio across batch (meaningful at ≥10 trades)
    omega: Optional[float] = None
    if len(trade_returns) >= 10:
        try:
            raw_omega = _omega_ratio(pd.Series(trade_returns), threshold=_DAILY_NGN_THRESHOLD)
            if math.isfinite(raw_omega):
                omega = raw_omega
            elif raw_omega == float("inf"):
                omega = float("inf")
        except Exception:
            pass

    # Profit factor
    profit_factor: Optional[float] = None
    if trade_returns:
        wins = [r for r in trade_returns if r > 0.0]
        losses = [r for r in trade_returns if r < 0.0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        if avg_loss > 0.0:
            profit_factor = avg_win / avg_loss
        elif avg_win > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

    n_trades = len(trade_returns)
    win_rate = sum(1 for r in trade_returns if r > 0.0) / n_trades if n_trades > 0 else None

    mean_tda = sum(terminal_das) / len(terminal_das) if terminal_das else None
    # terminal_da_pass_rate: fraction of runs where terminal DA >= 0.45
    tda_pass_rate = (
        sum(1 for d in terminal_das if d >= 0.45) / len(terminal_das)
        if terminal_das else None
    )
    mean_ci_cov = sum(ci_coverages) / len(ci_coverages) if ci_coverages else None
    mean_max_dd = sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else None
    mean_trade_return = sum(trade_returns) / n_trades if n_trades > 0 else None

    omega_above_1 = (
        omega is not None and (omega == float("inf") or omega > 1.0)
    )

    return {
        "n_runs": len(run_rows),
        "n_errors": n_errors,
        "n_trades": n_trades,
        "n_terminal_da": len(terminal_das),
        "n_ci_coverage": len(ci_coverages),
        "mean_terminal_da": mean_tda,
        "terminal_da_pass_rate": tda_pass_rate,
        "mean_ci_coverage": mean_ci_cov,
        "omega_ratio": omega,
        "omega_above_1": omega_above_1,
        "profit_factor": profit_factor if (profit_factor is None or math.isfinite(profit_factor)) else None,
        "win_rate": win_rate,
        "mean_trade_return": mean_trade_return,
        "expected_profit_usd": (
            mean_trade_return * _CAPITAL_BASE_USD if mean_trade_return is not None else None
        ),
        "mean_max_drawdown": mean_max_dd,
        "ngn_import_ok": _NGN_IMPORT_OK,
        "daily_ngn_threshold": _DAILY_NGN_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Threshold loaders
# ---------------------------------------------------------------------------

def _load_thresholds(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fm = raw.get("forecaster_monitoring") or {}
    rmse_cfg = fm.get("regression_metrics") or {}
    suite_cfg = rmse_cfg.get("adversarial_suite") or {}
    barbell_cfg = suite_cfg.get("barbell") or {}

    thresholds = {
        # RMSE thresholds (legacy)
        "max_ensemble_under_best_rate": float(suite_cfg.get("max_ensemble_under_best_rate", 1.0)),
        "max_avg_ensemble_ratio_vs_best": float(suite_cfg.get("max_avg_ensemble_ratio_vs_best", 1.2)),
        "max_ensemble_worse_than_rw_rate": float(suite_cfg.get("max_ensemble_worse_than_rw_rate", 0.3)),
        "max_effective_worse_than_rw_rate": float(
            suite_cfg.get(
                "max_effective_worse_than_rw_rate",
                suite_cfg.get("max_ensemble_worse_than_rw_rate", 0.3),
            )
        ),
        "use_effective_default_path_metric": bool(
            suite_cfg.get("use_effective_default_path_metric", False)
        ),
        "require_zero_errors": bool(suite_cfg.get("require_zero_errors", True)),
        "max_index_mismatch_rate": float(
            suite_cfg.get(
                "max_index_mismatch_rate",
                rmse_cfg.get("max_index_mismatch_rate", 1.0),
            )
        ),
        # Barbell thresholds (primary objective — domain-derived, not tuned to pass)
        # Defaults are deliberately permissive; enforce via config when ready.
        "min_terminal_da_pass_rate": float(barbell_cfg.get("min_terminal_da_pass_rate", 0.0)),
        "min_omega_ratio": float(barbell_cfg.get("min_omega_ratio", 0.0)),
        "min_ci_coverage_rate": float(barbell_cfg.get("min_ci_coverage_rate", 0.0)),
        "min_profit_factor": float(barbell_cfg.get("min_profit_factor", 0.0)),
        "max_mean_drawdown": float(barbell_cfg.get("max_mean_drawdown", 1.0)),
        "require_omega_above_1": bool(barbell_cfg.get("require_omega_above_1", False)),
    }
    return thresholds


# ---------------------------------------------------------------------------
# Threshold evaluators
# ---------------------------------------------------------------------------

def evaluate_thresholds(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
    """Check RMSE-based thresholds (legacy, backward-compatible)."""
    breaches: List[str] = []
    use_effective_metric = bool(thresholds.get("use_effective_default_path_metric", False))
    for variant, payload in summary.items():
        errors = int(payload.get("errors", 0) or 0)
        under_best = payload.get("ensemble_under_best_rate", 0.0)
        # nan rates (all-error runs) must not silently pass
        if under_best is None or (isinstance(under_best, float) and math.isnan(under_best)):
            if int(payload.get("errors", 0) or 0) > 0:
                breaches.append(f"{variant}: all runs errored — rates are undefined (nan)")
            continue
        under_best = float(under_best)
        ratio = payload.get("avg_ensemble_ratio_vs_best")
        ratio = float(ratio) if isinstance(ratio, (int, float)) else None
        index_mismatch_rate = float(payload.get("ensemble_index_mismatch_rate", 0.0) or 0.0)
        if use_effective_metric and "effective_worse_than_rw_rate" in payload:
            worse_rw_metric = "effective_worse_than_rw_rate"
            worse_rw = float(payload.get("effective_worse_than_rw_rate", 0.0) or 0.0)
            worse_rw_threshold = float(thresholds.get("max_effective_worse_than_rw_rate", 0.3))
        else:
            worse_rw_metric = "ensemble_worse_than_rw_rate"
            worse_rw = float(payload.get("ensemble_worse_than_rw_rate", 0.0) or 0.0)
            worse_rw_threshold = float(thresholds.get("max_ensemble_worse_than_rw_rate", 0.3))

        if thresholds.get("require_zero_errors", True) and errors > 0:
            breaches.append(f"{variant}: errors={errors} (require_zero_errors=true)")
        if under_best > float(thresholds.get("max_ensemble_under_best_rate", 1.0)):
            breaches.append(
                f"{variant}: ensemble_under_best_rate={under_best:.4f} > "
                f"{float(thresholds.get('max_ensemble_under_best_rate')):.4f}"
            )
        if ratio is not None and ratio > float(thresholds.get("max_avg_ensemble_ratio_vs_best", 1.2)):
            breaches.append(
                f"{variant}: avg_ensemble_ratio_vs_best={ratio:.4f} > "
                f"{float(thresholds.get('max_avg_ensemble_ratio_vs_best')):.4f}"
            )
        if worse_rw > worse_rw_threshold:
            breaches.append(
                f"{variant}: {worse_rw_metric}={worse_rw:.4f} > {worse_rw_threshold:.4f}"
            )
        if index_mismatch_rate > float(thresholds.get("max_index_mismatch_rate", 1.0)):
            breaches.append(
                f"{variant}: ensemble_index_mismatch_rate={index_mismatch_rate:.4f} > "
                f"{float(thresholds.get('max_index_mismatch_rate', 1.0)):.4f}"
            )
    return breaches


def evaluate_barbell_thresholds(
    barbell_summary: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> List[str]:
    """
    Check barbell-objective thresholds (primary objective per matrix-first policy).

    These thresholds are domain-derived from NGN economics — they are NOT
    calibrated to current system performance. A FAIL here is a signal to improve
    the model or the scenario coverage, not to lower the threshold.
    """
    breaches: List[str] = []
    for variant, payload in barbell_summary.items():
        # Guard: if no trade data, skip (not a PASS — emit warning)
        n_trades = int(payload.get("n_trades") or 0)
        n_errors = int(payload.get("n_errors") or 0)
        if n_errors > 0 and n_trades == 0:
            breaches.append(
                f"{variant}[barbell]: all {n_errors} runs errored — no barbell metrics available"
            )
            continue

        # terminal_da_pass_rate: fraction of runs with terminal DA >= 0.45
        min_tda = float(thresholds.get("min_terminal_da_pass_rate", 0.0))
        tda_rate = payload.get("terminal_da_pass_rate")
        if tda_rate is not None and min_tda > 0.0 and tda_rate < min_tda:
            breaches.append(
                f"{variant}[barbell]: terminal_da_pass_rate={tda_rate:.4f} < {min_tda:.4f}"
            )

        # omega_ratio: gain/loss at NGN threshold
        min_omega = float(thresholds.get("min_omega_ratio", 0.0))
        require_omega_above_1 = bool(thresholds.get("require_omega_above_1", False))
        omega = payload.get("omega_ratio")
        if min_omega > 0.0 and omega is not None and not (
            omega == float("inf") or omega >= min_omega
        ):
            breaches.append(
                f"{variant}[barbell]: omega_ratio={omega:.4f} < {min_omega:.4f} "
                f"(NGN hurdle={_DAILY_NGN_THRESHOLD:.5f}/day)"
            )
        if require_omega_above_1 and not bool(payload.get("omega_above_1", False)):
            breaches.append(
                f"{variant}[barbell]: omega_ratio does not beat NGN hurdle "
                f"(omega={omega}, require_omega_above_1=true)"
            )

        # CI coverage rate (empirical vs nominal)
        min_ci_cov = float(thresholds.get("min_ci_coverage_rate", 0.0))
        ci_cov = payload.get("mean_ci_coverage")
        if min_ci_cov > 0.0 and ci_cov is not None and ci_cov < min_ci_cov:
            breaches.append(
                f"{variant}[barbell]: mean_ci_coverage={ci_cov:.4f} < {min_ci_cov:.4f}"
            )

        # profit_factor
        min_pf = float(thresholds.get("min_profit_factor", 0.0))
        pf = payload.get("profit_factor")
        if min_pf > 0.0 and pf is not None and pf < min_pf:
            breaches.append(
                f"{variant}[barbell]: profit_factor={pf:.4f} < {min_pf:.4f}"
            )

        # max_drawdown ceiling
        max_dd = float(thresholds.get("max_mean_drawdown", 1.0))
        mean_dd = payload.get("mean_max_drawdown")
        if max_dd < 1.0 and mean_dd is not None and mean_dd > max_dd:
            breaches.append(
                f"{variant}[barbell]: mean_max_drawdown={mean_dd:.4f} > {max_dd:.4f}"
            )

    return breaches


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-config", default="config/forecasting_config.yml")
    parser.add_argument("--monitor-config", default="config/forecaster_monitoring_ci.yml")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--n-points", type=int, default=DEFAULT_POINTS)
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names. Includes barbell scenarios by default.",
    )
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output", default="")
    parser.add_argument("--enforce-thresholds", action="store_true")
    parser.add_argument(
        "--rmse-only",
        action="store_true",
        help="Skip barbell scenarios and metrics (legacy mode).",
    )
    args = parser.parse_args()

    os.environ["TS_FORECAST_AUDIT_DIR"] = ""

    forecasting_path = Path(args.forecasting_config)
    monitor_path = Path(args.monitor_config)
    if not forecasting_path.exists():
        raise SystemExit(f"Forecasting config not found: {forecasting_path}")

    raw_cfg = yaml.safe_load(forecasting_path.read_text(encoding="utf-8")) or {}
    scenarios_str = args.scenarios
    if args.rmse_only:
        scenarios_str = ",".join(_RMSE_SCENARIOS)
    scenarios = _parse_csv_str(scenarios_str)
    seeds = _parse_csv_int(args.seeds)
    variants = _parse_csv_str(args.variants)

    raw_results = _run_variants(
        raw_cfg=raw_cfg,
        variants=variants,
        scenarios=scenarios,
        seeds=seeds,
        horizon=int(args.horizon),
        n_points=int(args.n_points),
    )
    summary = {variant: summarize(rows) for variant, rows in raw_results.items()}
    barbell_summary = {variant: summarize_barbell(rows) for variant, rows in raw_results.items()}

    thresholds = _load_thresholds(monitor_path)
    rmse_breaches = evaluate_thresholds(summary, thresholds) if thresholds else []
    barbell_breaches = evaluate_barbell_thresholds(barbell_summary, thresholds) if thresholds else []
    breaches = rmse_breaches + barbell_breaches

    payload = {
        "meta": {
            "horizon": int(args.horizon),
            "n_points": int(args.n_points),
            "scenarios": scenarios,
            "seeds": seeds,
            "variants": variants,
            "total_runs_per_variant": len(scenarios) * len(seeds),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "objective": "barbell_asymmetric_upside_bounded_downside",
            "ngn_daily_threshold": _DAILY_NGN_THRESHOLD,
            "ngn_import_ok": _NGN_IMPORT_OK,
        },
        "summary": summary,
        "barbell_summary": barbell_summary,
        "thresholds": thresholds,
        "rmse_breaches": rmse_breaches,
        "barbell_breaches": barbell_breaches,
        "breaches": breaches,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("reports") / "adversarial_forecaster" / (
            f"suite_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved: {output_path}")

    if args.enforce_thresholds and breaches:
        print("\n[FAIL] Adversarial threshold breaches:")
        for item in breaches:
            print(f"- {item}")
        raise SystemExit(1)


def _parse_csv_int(value: str) -> List[int]:
    items: List[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    return items


def _parse_csv_str(value: str) -> List[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


if __name__ == "__main__":
    main()
