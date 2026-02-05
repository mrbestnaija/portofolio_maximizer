#!/usr/bin/env python3
"""
check_forecast_audits.py
------------------------

Brutal-style sanity check for Time Series forecaster performance.

Reads the most recent forecast audit JSON files emitted by
forcester_ts/forecaster.py (via ModelInstrumentation) from
logs/forecast_audits/, compares ensemble regression metrics to a
baseline model, and exits non-zero if the ensemble underperforms
systematically.

This script is read-only and safe to call from brutal/dry-run
or CI workflows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_AUDIT_DIR = Path("logs/forecast_audits")
DEFAULT_MONITORING_CONFIG = Path("config/forecaster_monitoring.yml")
DEFAULT_BASELINE_MODEL = "BEST_SINGLE"
DEFAULT_DECISION_KEEP = "KEEP"
DEFAULT_DECISION_RESEARCH = "RESEARCH_ONLY"
DEFAULT_DECISION_DISABLE = "DISABLE_DEFAULT"


@dataclass
class AuditCheckResult:
    path: Path
    ensemble_rmse: Optional[float]
    baseline_rmse: Optional[float]
    rmse_ratio: Optional[float]
    violation: bool
    baseline_model: Optional[str] = None


def _load_audit(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _extract_metrics(
    audit: Dict[str, Any], *, baseline_model: str
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[str]]:
    """
    Return (ensemble_metrics, baseline_metrics, resolved_baseline_model) from an audit payload.

    Ensemble metrics are taken from artifacts['evaluation_metrics']['ensemble']
    when available.

    Baseline selection:
    - BEST_SINGLE: choose the available single-model entry with the smallest RMSE.
    - SAMOSSA: use samossa when present; else fall back to BEST_SINGLE.
    - SARIMAX: use sarimax when present; else fall back to BEST_SINGLE.
    """
    artifacts = audit.get("artifacts") or {}
    eval_metrics = artifacts.get("evaluation_metrics") or {}
    if not isinstance(eval_metrics, dict):
        return None, None, None

    ensemble = eval_metrics.get("ensemble")
    sarimax = eval_metrics.get("sarimax")
    samossa = eval_metrics.get("samossa")

    if ensemble is None and sarimax is None and samossa is None:
        return None, None, None

    ensemble_metrics = ensemble or sarimax or samossa

    baseline_model = (baseline_model or DEFAULT_BASELINE_MODEL).strip().upper()

    def _best_single() -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for name in ("sarimax", "samossa", "mssa_rl"):
            payload = eval_metrics.get(name)
            if isinstance(payload, dict):
                candidates.append((name, payload))
        best_payload: Optional[Dict[str, Any]] = None
        best_rmse: Optional[float] = None
        best_name: Optional[str] = None
        for name, payload in candidates:
            rmse = _rmse_from(payload)
            if rmse is None:
                continue
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_payload = payload
                best_name = name.upper()
        if best_payload is not None:
            return best_payload, best_name
        if isinstance(sarimax, dict):
            return sarimax, "SARIMAX"
        if isinstance(samossa, dict):
            return samossa, "SAMOSSA"
        return ensemble_metrics if isinstance(ensemble_metrics, dict) else None, "ENSEMBLE"

    if baseline_model == "SAMOSSA":
        if isinstance(samossa, dict):
            baseline_metrics = samossa
            resolved_baseline = "SAMOSSA"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    elif baseline_model == "SARIMAX":
        if isinstance(sarimax, dict):
            baseline_metrics = sarimax
            resolved_baseline = "SARIMAX"
        else:
            baseline_metrics, resolved_baseline = _best_single()
    else:
        baseline_metrics, resolved_baseline = _best_single()

    return (
        ensemble_metrics if isinstance(ensemble_metrics, dict) else None,
        baseline_metrics if isinstance(baseline_metrics, dict) else None,
        resolved_baseline,
    )


def _rmse_from(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("rmse")
    return float(val) if isinstance(val, (int, float)) else None


def check_audit_file(
    path: Path,
    tolerance: float,
    *,
    baseline_model: str,
) -> Optional[AuditCheckResult]:
    audit = _load_audit(path)
    if not audit:
        return None

    ensemble_metrics, baseline_metrics, resolved_baseline = _extract_metrics(
        audit, baseline_model=baseline_model
    )
    ensemble_rmse = _rmse_from(ensemble_metrics)
    baseline_rmse = _rmse_from(baseline_metrics)

    if ensemble_rmse is None or baseline_rmse is None or baseline_rmse <= 0:
        return AuditCheckResult(
            path=path,
            ensemble_rmse=ensemble_rmse,
            baseline_rmse=baseline_rmse,
            rmse_ratio=None,
            violation=False,
            baseline_model=resolved_baseline,
        )

    rmse_ratio = ensemble_rmse / baseline_rmse
    violation = rmse_ratio > (1.0 + tolerance)

    return AuditCheckResult(
        path=path,
        ensemble_rmse=ensemble_rmse,
        baseline_rmse=baseline_rmse,
        rmse_ratio=rmse_ratio,
        violation=violation,
        baseline_model=resolved_baseline,
    )


def _load_monitoring_thresholds(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path or not config_path.exists():
        return {}
    try:
        import yaml  # Local import to keep dependency optional
    except ImportError:
        return {}

    raw = yaml.safe_load(config_path.read_text()) or {}
    fm = raw.get("forecaster_monitoring") or {}
    return fm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check TS forecast audit files for ensemble underperformance."
    )
    parser.add_argument(
        "--audit-dir",
        default=str(DEFAULT_AUDIT_DIR),
        help="Directory containing forecast_audit_*.json files "
        "(default: logs/forecast_audits)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum number of most recent audit files to inspect (default: 50)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Allowed RMSE degradation vs baseline before flagging a violation. "
        "If omitted, will fall back to config/forecaster_monitoring.yml or 0.10.",
    )
    parser.add_argument(
        "--max-violation-rate",
        type=float,
        default=None,
        help="Maximum fraction of checked audits allowed to violate the RMSE tolerance "
        "before exiting non-zero. If omitted, will fall back to config/forecaster_monitoring.yml or 0.25.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_MONITORING_CONFIG),
        help="Optional path to forecaster_monitoring.yml "
        "(default: config/forecaster_monitoring.yml if present)",
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Baseline model for the RMSE gate: BEST_SINGLE, SAMOSSA, or SARIMAX. "
        "If omitted, uses forecaster_monitoring.regression_metrics.baseline_model "
        f"or {DEFAULT_BASELINE_MODEL}.",
    )
    parser.add_argument(
        "--require-effective-audits",
        type=int,
        default=None,
        help="If set, exit non-zero when effective audits with RMSE metrics are below this count.",
    )
    parser.add_argument(
        "--require-holding-period",
        action="store_true",
        help="If set, require effective audits to meet holding_period_audits from the monitoring config.",
    )
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    if not audit_dir.exists():
        raise SystemExit(f"Audit directory not found: {audit_dir}")

    files = sorted(
        audit_dir.glob("forecast_audit_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    files = files[: args.max_files]
    if not files:
        raise SystemExit("No forecast_audit_*.json files found.")

    monitoring_cfg = _load_monitoring_thresholds(
        Path(args.config_path) if args.config_path else None
    )
    rmse_cfg = monitoring_cfg.get("regression_metrics") if monitoring_cfg else {}

    tolerance = (
        float(args.tolerance)
        if args.tolerance is not None
        else float(rmse_cfg.get("max_rmse_ratio_vs_baseline", 1.10)) - 1.0
    )
    max_violation_rate = (
        float(args.max_violation_rate)
        if args.max_violation_rate is not None
        else float(rmse_cfg.get("max_violation_rate", 0.25))
    )
    min_effective_audits = int(rmse_cfg.get("min_effective_audits", 0) or 0)
    baseline_model = (
        str(args.baseline_model)
        if args.baseline_model
        else str(rmse_cfg.get("baseline_model", DEFAULT_BASELINE_MODEL))
    )
    holding_period = int(rmse_cfg.get("holding_period_audits", 0) or 0)
    fail_on_violation_during_holding_period = bool(
        rmse_cfg.get("fail_on_violation_during_holding_period", False)
    )
    disable_if_no_lift = bool(rmse_cfg.get("disable_ensemble_if_no_lift", False))
    min_lift_rmse_ratio = float(rmse_cfg.get("min_lift_rmse_ratio", 0.0) or 0.0)
    min_lift_fraction = float(rmse_cfg.get("min_lift_fraction", 0.0) or 0.0)
    promotion_margin = float(rmse_cfg.get("promotion_margin", 0.0) or 0.0)

    def _dedupe_key(path: Path) -> Optional[Tuple[Any, ...]]:
        audit = _load_audit(path)
        if not audit:
            return None
        dataset = audit.get("dataset") or {}
        ds_key = (
            dataset.get("start"),
            dataset.get("end"),
            dataset.get("length"),
            dataset.get("forecast_horizon"),
        )
        # Deduplicate by data window (start/end/length/horizon) only and keep the
        # most recent file as the authoritative result. This prevents stale
        # earlier runs (often with different ensemble weights) from inflating
        # the violation rate and aligns with "latest evidence wins" monitoring.
        return ds_key

    unique_map: dict[Tuple[Any, ...], Path] = {}

    def _has_effective_metrics(path: Path) -> bool:
        audit = _load_audit(path)
        if not audit:
            return False
        ensemble_metrics, baseline_metrics, _ = _extract_metrics(
            audit, baseline_model=baseline_model
        )
        ensemble_rmse = _rmse_from(ensemble_metrics)
        baseline_rmse = _rmse_from(baseline_metrics)
        return (
            ensemble_rmse is not None
            and baseline_rmse is not None
            and baseline_rmse > 0
        )

    for f in files:
        key = _dedupe_key(f)
        if key is None:
            continue
        # Skip entries without usable metrics; they do not inform RMSE gating.
        if not _has_effective_metrics(f):
            continue
        # Files are sorted newest-first, so keep the first (newest) entry we see
        # for each dataset window and ignore older duplicates.
        if key in unique_map:
            continue
        unique_map[key] = f

    unique_files: List[Path] = list(unique_map.values())

    results: List[AuditCheckResult] = []
    for f in unique_files:
        res = check_audit_file(f, tolerance, baseline_model=baseline_model)
        if res is not None:
            results.append(res)

    print("=== Forecast Audit Regression Check ===")
    print(f"Audit directory : {audit_dir}")
    print(
        f"Files inspected : {len(results)} unique (raw={len(files)}, max_files={args.max_files})"
    )
    print(f"Baseline model  : {baseline_model}")
    print(f"RMSE tolerance  : ensemble_rmse <= (1 + {tolerance:.2f}) * baseline_rmse")
    if min_effective_audits > 0:
        print(f"Min effective   : {min_effective_audits} audit(s) before hard gating")
    if holding_period > 0:
        print(f"Holding period  : {holding_period} effective audit(s)")
        if fail_on_violation_during_holding_period:
            print("Warmup behavior : fail on violations during holding period")
    if disable_if_no_lift:
        print(
            "No-lift gate    : enabled "
            f"(min_lift_rmse_ratio={min_lift_rmse_ratio:.2%}, "
            f"min_lift_fraction={min_lift_fraction:.2%})"
        )
    if promotion_margin > 0:
        print(f"Promotion margin: requires >= {promotion_margin:.2%} lift to keep ensemble as default")

    violation_count = sum(1 for r in results if r.violation)
    effective_n = sum(
        1
        for r in results
        if (
            r.ensemble_rmse is not None
            and r.baseline_rmse is not None
            and r.baseline_rmse > 0
        )
    )
    violation_rate = (violation_count / effective_n) if effective_n else 0.0

    def _percentiles(values: list[float], percents: list[float]) -> Dict[float, float]:
        if not values:
            return {}
        vals = sorted(values)
        out: Dict[float, float] = {}
        for p in percents:
            if p <= 0:
                out[p] = vals[0]
                continue
            if p >= 1:
                out[p] = vals[-1]
                continue
            idx = (len(vals) - 1) * p
            lower = int(idx)
            upper = min(lower + 1, len(vals) - 1)
            weight = idx - lower
            out[p] = vals[lower] * (1 - weight) + vals[upper] * weight
        return out

    ratios = [
        r.rmse_ratio
        for r in results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    pct = _percentiles(ratios, [0.1, 0.5, 0.9]) if ratios else {}

    print(f"\nEffective audits with RMSE: {effective_n}")
    print(f"Violations (ensemble worse than baseline beyond tolerance): {violation_count}")
    print(f"Violation rate: {violation_rate:.2%} (max allowed {max_violation_rate:.2%})")
    if pct:
        print(
            "RMSE ratio percentiles: "
            f"p10={pct.get(0.1):.3f}, median={pct.get(0.5):.3f}, p90={pct.get(0.9):.3f}"
        )

    warmup_required = max(min_effective_audits, holding_period, 0)
    if warmup_required > 0 and effective_n < warmup_required:
        explicit_required: Optional[int] = None
        if args.require_holding_period and holding_period > 0:
            explicit_required = holding_period
        if args.require_effective_audits is not None:
            explicit_required = int(args.require_effective_audits)
        if explicit_required is not None and effective_n < explicit_required:
            raise SystemExit(
                f"Insufficient effective audits for RMSE gating: effective_audits={effective_n} "
                f"< required_audits={explicit_required}"
            )
        if (
            fail_on_violation_during_holding_period
            and effective_n > 0
            and violation_rate > max_violation_rate
        ):
            raise SystemExit(
                f"Ensemble RMSE violation rate {violation_rate:.2%} exceeds "
                f"max-violation-rate {max_violation_rate:.2%} during holding period "
                f"(effective_audits={effective_n} < required_audits={warmup_required})"
            )
        print(
            f"\nRMSE gate inconclusive: effective_audits={effective_n} "
            f"< required_audits={warmup_required}.",
        )
        raise SystemExit(0)

    print("\nSample details (most recent first):")
    header = f"{'File':<32} {'ens_rmse':>10} {'base_rmse':>10} {'ratio':>8} {'VIOL':>6}"
    print(header)
    print("-" * len(header))
    for r in results[:10]:
        ratio_str = f"{r.rmse_ratio:.3f}" if r.rmse_ratio is not None else "n/a"
        ens_str = f"{r.ensemble_rmse:.4f}" if r.ensemble_rmse is not None else "n/a"
        base_str = f"{r.baseline_rmse:.4f}" if r.baseline_rmse is not None else "n/a"
        viol_flag = "YES" if r.violation else ""
        display_name = r.path.name
        if r.baseline_model:
            display_name = f"{display_name} ({r.baseline_model})"
        display_name = display_name[:32]
        print(
            f"{display_name:<32} {ens_str:>10} {base_str:>10} {ratio_str:>8} {viol_flag:>6}"
        )

    if effective_n == 0:
        # No usable metrics; do not fail hard, but signal that checks were inconclusive.
        raise SystemExit(0)

    decision = DEFAULT_DECISION_KEEP
    decision_reason = "ensemble within tolerance"

    lift_fraction = 0.0
    if effective_n:
        lift_threshold = 1.0 - min_lift_rmse_ratio
        lift_count = sum(
            1
            for r in results
            if (
                r.rmse_ratio is not None
                and isinstance(r.rmse_ratio, (int, float))
                and float(r.rmse_ratio) < float(lift_threshold)
            )
        )
        lift_fraction = lift_count / effective_n

    if disable_if_no_lift and holding_period > 0 and effective_n >= holding_period:
        print(
            f"\nEnsemble lift fraction: {lift_fraction:.2%} "
            f"(required >= {min_lift_fraction:.2%})"
        )
        if lift_fraction < min_lift_fraction:
            decision = DEFAULT_DECISION_DISABLE
            decision_reason = "insufficient lift vs baseline"
            raise SystemExit(
                "Ensemble shows insufficient lift over baseline during holding period; "
                "disable ensemble as default source of truth (reward-to-effort)."
            )
        decision_reason = "lift demonstrated during holding period"

    if violation_rate > max_violation_rate:
        decision = DEFAULT_DECISION_RESEARCH
        decision_reason = (
            f"violation rate {violation_rate:.2%} exceeds {max_violation_rate:.2%}"
        )
        raise SystemExit(
            f"Ensemble RMSE violation rate {violation_rate:.2%} exceeds "
            f"max-violation-rate {max_violation_rate:.2%}"
        )

    if promotion_margin > 0 and effective_n > 0:
        margin_threshold = 1.0 - promotion_margin
        margin_lift = sum(
            1
            for r in results
            if r.rmse_ratio is not None
            and isinstance(r.rmse_ratio, (int, float))
            and float(r.rmse_ratio) < float(margin_threshold)
        )
        margin_lift_fraction = (margin_lift / effective_n) if effective_n else 0.0
        if margin_lift_fraction <= 0.0:
            decision = DEFAULT_DECISION_RESEARCH
            decision_reason = (
                f"no ensemble lift >= {promotion_margin:.2%} across recent audits"
            )

    print(f"\nDecision: {decision} ({decision_reason})")

    cache_dir = Path("logs/forecast_audits_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ratios_filtered = [
        (r.path.name, float(r.rmse_ratio))
        for r in results
        if r.rmse_ratio is not None and isinstance(r.rmse_ratio, (int, float))
    ]
    ratio_values = [val for _, val in ratios_filtered]
    ratio_stats = {
        "count": len(ratio_values),
        "min": min(ratio_values) if ratio_values else None,
        "max": max(ratio_values) if ratio_values else None,
        "mean": (sum(ratio_values) / len(ratio_values)) if ratio_values else None,
        "p10": pct.get(0.1) if pct else None,
        "p50": pct.get(0.5) if pct else None,
        "p90": pct.get(0.9) if pct else None,
        "best": [
            {"file": name, "ratio": val}
            for name, val in sorted(ratios_filtered, key=lambda x: x[1])[:5]
        ],
        "worst": [
            {"file": name, "ratio": val}
            for name, val in sorted(ratios_filtered, key=lambda x: x[1], reverse=True)[
                :5
            ]
        ],
    }
    dataset_entries = []
    for f in unique_files:
        audit = _load_audit(f)
        ds = (audit or {}).get("dataset") or {}
        entry = {
            "file": f.name,
            "start": ds.get("start"),
            "end": ds.get("end"),
            "length": ds.get("length"),
            "forecast_horizon": ds.get("forecast_horizon"),
        }
        matching = next((r for r in results if r.path == f), None)
        if matching:
            entry["rmse_ratio"] = matching.rmse_ratio
            entry["ensemble_rmse"] = matching.ensemble_rmse
            entry["baseline_rmse"] = matching.baseline_rmse
        dataset_entries.append(entry)

    summary = {
        "audit_dir": str(audit_dir),
        "effective_audits": effective_n,
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "max_violation_rate": max_violation_rate,
        "holding_period_required": warmup_required,
        "lift_fraction": lift_fraction,
        "min_lift_fraction": min_lift_fraction,
        "percentiles": {
            "p10": pct.get(0.1) if pct else None,
            "p50": pct.get(0.5) if pct else None,
            "p90": pct.get(0.9) if pct else None,
        },
        "ratio_distribution": ratio_stats,
        "decision": decision,
        "decision_reason": decision_reason,
        "dataset_windows": dataset_entries,
    }
    cache_path = cache_dir / "latest_summary.json"
    dash_path = cache_dir / "ratio_distribution.json"
    try:
        cache_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        dash_path.write_text(json.dumps(ratio_stats, indent=2), encoding="utf-8")
    except Exception:
        pass

    raise SystemExit(0)


if __name__ == "__main__":
    main()
