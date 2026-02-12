#!/usr/bin/env python3
"""
Adversarial forecaster benchmark and CI gate.

Runs a deterministic stress matrix across synthetic market regimes and reports:
- ensemble_under_best_rate
- avg_ensemble_ratio_vs_best
- ensemble_worse_than_rw_rate

Optionally enforces blocking thresholds from forecaster monitoring config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.metrics import compute_regression_metrics


DEFAULT_HORIZON = 20
DEFAULT_POINTS = 320
DEFAULT_SCENARIOS = [
    "trend_seasonal",
    "random_walk",
    "regime_shift",
    "vol_cluster",
    "jump_shock",
    "mean_reversion_break",
]
DEFAULT_SEEDS = [101, 202, 303]
DEFAULT_VARIANTS = [
    "prod_like_conf_off",
    "prod_like_conf_on",
    "sarimax_augmented_conf_on",
]


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


def gen_series(kind: str, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)

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
    else:
        raise ValueError(f"Unknown scenario: {kind}")

    return pd.Series(y, index=idx, name="Close")


def rw_baseline(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    lv = float(train.dropna().iloc[-1])
    return pd.Series(lv, index=test_index)


def mk_cfg(
    variant: str,
    raw_cfg: Dict[str, Any],
    *,
    horizon: int,
) -> TimeSeriesForecasterConfig:
    f = raw_cfg["forecasting"]
    ens = dict(f.get("ensemble") or {})
    base_candidates = list(ens.get("candidate_weights") or [])

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


def run_one(series: pd.Series, cfg: TimeSeriesForecasterConfig, horizon: int):
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    returns = train.pct_change().dropna()

    rw_metrics = compute_regression_metrics(test, rw_baseline(train, test.index)) or {}

    forecaster = TimeSeriesForecaster(config=cfg)
    forecaster.fit(price_series=train, returns_series=returns)
    forecaster.forecast(steps=horizon)
    model_metrics = forecaster.evaluate(test)

    meta = forecaster._latest_results.get("ensemble_metadata", {}) if isinstance(forecaster._latest_results, dict) else {}
    weights = meta.get("weights", {}) if isinstance(meta, dict) else {}
    status = meta.get("ensemble_status") if isinstance(meta, dict) else None

    return rw_metrics, model_metrics, weights, status


def summarize(run_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "runs": len(run_rows),
        "errors": 0,
        "model_presence": Counter(),
        "ensemble_under_best_count": 0,
        "ensemble_worse_than_rw_count": 0,
        "ensemble_rmse_ratio_vs_best": [],
        "sarimax_rmse_ratio_vs_rw": [],
        "ensemble_status_counts": Counter(),
        "weight_patterns": Counter(),
        "scenario_breakdown": defaultdict(lambda: {"n": 0, "ens_under_best": 0, "ens_worse_rw": 0, "ens_ratio": []}),
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

        weights = row.get("weights") or {}
        if isinstance(weights, dict) and weights:
            key = tuple(sorted((k, round(float(v), 4)) for k, v in weights.items()))
            out["weight_patterns"][key] += 1

        ens_rmse = (metrics_map.get("ensemble") or {}).get("rmse") if isinstance(metrics_map.get("ensemble"), dict) else None
        rw_rmse = rw.get("rmse") if isinstance(rw, dict) else None
        best_single = None
        for model_name in ("sarimax", "garch", "samossa", "mssa_rl"):
            rmse = (metrics_map.get(model_name) or {}).get("rmse") if isinstance(metrics_map.get(model_name), dict) else None
            if isinstance(rmse, (int, float)):
                best_single = rmse if best_single is None else min(best_single, rmse)

        if isinstance(ens_rmse, (int, float)) and isinstance(best_single, (int, float)) and best_single > 0:
            ratio = float(ens_rmse) / float(best_single)
            out["ensemble_rmse_ratio_vs_best"].append(ratio)
            out["scenario_breakdown"][scenario]["ens_ratio"].append(ratio)
            if ratio > 1.0:
                out["ensemble_under_best_count"] += 1
                out["scenario_breakdown"][scenario]["ens_under_best"] += 1

        if isinstance(ens_rmse, (int, float)) and isinstance(rw_rmse, (int, float)) and rw_rmse > 0:
            if float(ens_rmse) > float(rw_rmse):
                out["ensemble_worse_than_rw_count"] += 1
                out["scenario_breakdown"][scenario]["ens_worse_rw"] += 1

        srmse = (metrics_map.get("sarimax") or {}).get("rmse") if isinstance(metrics_map.get("sarimax"), dict) else None
        if isinstance(srmse, (int, float)) and isinstance(rw_rmse, (int, float)) and rw_rmse > 0:
            out["sarimax_rmse_ratio_vs_rw"].append(float(srmse) / float(rw_rmse))

    n_ok = max(1, out["runs"] - out["errors"])
    out["ensemble_under_best_rate"] = out["ensemble_under_best_count"] / n_ok
    out["ensemble_worse_than_rw_rate"] = out["ensemble_worse_than_rw_count"] / n_ok
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
    return out


def _load_thresholds(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fm = raw.get("forecaster_monitoring") or {}
    rmse_cfg = fm.get("regression_metrics") or {}
    suite_cfg = rmse_cfg.get("adversarial_suite") or {}
    return {
        "max_ensemble_under_best_rate": float(suite_cfg.get("max_ensemble_under_best_rate", 1.0)),
        "max_avg_ensemble_ratio_vs_best": float(suite_cfg.get("max_avg_ensemble_ratio_vs_best", 1.2)),
        "max_ensemble_worse_than_rw_rate": float(suite_cfg.get("max_ensemble_worse_than_rw_rate", 0.3)),
        "require_zero_errors": bool(suite_cfg.get("require_zero_errors", True)),
    }


def evaluate_thresholds(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
    breaches: List[str] = []
    for variant, payload in summary.items():
        errors = int(payload.get("errors", 0) or 0)
        under_best = float(payload.get("ensemble_under_best_rate", 0.0) or 0.0)
        ratio = payload.get("avg_ensemble_ratio_vs_best")
        ratio = float(ratio) if isinstance(ratio, (int, float)) else None
        worse_rw = float(payload.get("ensemble_worse_than_rw_rate", 0.0) or 0.0)

        if thresholds.get("require_zero_errors", True) and errors > 0:
            breaches.append(f"{variant}: errors={errors} (require_zero_errors=true)")
        if under_best > float(thresholds.get("max_ensemble_under_best_rate", 1.0)):
            breaches.append(
                f"{variant}: ensemble_under_best_rate={under_best:.4f} > {float(thresholds.get('max_ensemble_under_best_rate')):.4f}"
            )
        if ratio is not None and ratio > float(thresholds.get("max_avg_ensemble_ratio_vs_best", 1.2)):
            breaches.append(
                f"{variant}: avg_ensemble_ratio_vs_best={ratio:.4f} > {float(thresholds.get('max_avg_ensemble_ratio_vs_best')):.4f}"
            )
        if worse_rw > float(thresholds.get("max_ensemble_worse_than_rw_rate", 0.3)):
            breaches.append(
                f"{variant}: ensemble_worse_than_rw_rate={worse_rw:.4f} > {float(thresholds.get('max_ensemble_worse_than_rw_rate')):.4f}"
            )
    return breaches


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
                    rw, metrics, weights, status = run_one(series, cfg, horizon)
                    results[variant].append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "rw": rw,
                            "metrics": metrics,
                            "weights": weights,
                            "status": status,
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
                            "error": str(exc),
                        }
                    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-config", default="config/forecasting_config.yml")
    parser.add_argument("--monitor-config", default="config/forecaster_monitoring_ci.yml")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--n-points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output", default="")
    parser.add_argument("--enforce-thresholds", action="store_true")
    args = parser.parse_args()

    os.environ["TS_FORECAST_AUDIT_DIR"] = ""

    forecasting_path = Path(args.forecasting_config)
    monitor_path = Path(args.monitor_config)
    if not forecasting_path.exists():
        raise SystemExit(f"Forecasting config not found: {forecasting_path}")

    raw_cfg = yaml.safe_load(forecasting_path.read_text(encoding="utf-8")) or {}
    scenarios = _parse_csv_str(args.scenarios)
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

    thresholds = _load_thresholds(monitor_path)
    breaches = evaluate_thresholds(summary, thresholds) if thresholds else []

    payload = {
        "meta": {
            "horizon": int(args.horizon),
            "n_points": int(args.n_points),
            "scenarios": scenarios,
            "seeds": seeds,
            "variants": variants,
            "total_runs_per_variant": len(scenarios) * len(seeds),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
        "summary": summary,
        "thresholds": thresholds,
        "breaches": breaches,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("reports") / "adversarial_forecaster" / (
            f"suite_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {output_path}")

    if args.enforce_thresholds and breaches:
        print("\n[FAIL] Adversarial threshold breaches:")
        for item in breaches:
            print(f"- {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
