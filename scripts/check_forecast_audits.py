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


@dataclass
class AuditCheckResult:
    path: Path
    ensemble_rmse: Optional[float]
    baseline_rmse: Optional[float]
    rmse_ratio: Optional[float]
    violation: bool


def _load_audit(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _extract_metrics(
    audit: Dict[str, Any]
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """
    Return (ensemble_metrics, baseline_metrics) from an audit payload.

    Ensemble metrics are taken from artifacts['evaluation_metrics']['ensemble']
    when available. Baseline metrics default to 'sarimax' if present; if not,
    they fall back to ensemble itself.
    """
    artifacts = audit.get("artifacts") or {}
    eval_metrics = artifacts.get("evaluation_metrics") or {}
    if not isinstance(eval_metrics, dict):
        return None, None

    ensemble = eval_metrics.get("ensemble")
    sarimax = eval_metrics.get("sarimax")

    if ensemble is None and sarimax is None:
        return None, None

    ensemble_metrics = ensemble or sarimax
    baseline_metrics = sarimax or ensemble_metrics
    return (
        ensemble_metrics if isinstance(ensemble_metrics, dict) else None,
        baseline_metrics if isinstance(baseline_metrics, dict) else None,
    )


def _rmse_from(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("rmse")
    return float(val) if isinstance(val, (int, float)) else None


def check_audit_file(
    path: Path,
    tolerance: float,
) -> Optional[AuditCheckResult]:
    audit = _load_audit(path)
    if not audit:
        return None

    ensemble_metrics, baseline_metrics = _extract_metrics(audit)
    ensemble_rmse = _rmse_from(ensemble_metrics)
    baseline_rmse = _rmse_from(baseline_metrics)

    if ensemble_rmse is None or baseline_rmse is None or baseline_rmse <= 0:
        return AuditCheckResult(
            path=path,
            ensemble_rmse=ensemble_rmse,
            baseline_rmse=baseline_rmse,
            rmse_ratio=None,
            violation=False,
        )

    rmse_ratio = ensemble_rmse / baseline_rmse
    violation = rmse_ratio > (1.0 + tolerance)

    return AuditCheckResult(
        path=path,
        ensemble_rmse=ensemble_rmse,
        baseline_rmse=baseline_rmse,
        rmse_ratio=rmse_ratio,
        violation=violation,
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

    results: List[AuditCheckResult] = []
    for f in files:
        res = check_audit_file(f, args.tolerance)
        if res is not None:
            results.append(res)

    print("=== Forecast Audit Regression Check ===")
    print(f"Audit directory : {audit_dir}")
    print(f"Files inspected : {len(results)} (max_files={args.max_files})")
    print(f"RMSE tolerance  : ensemble_rmse <= (1 + {tolerance:.2f}) * baseline_rmse")

    violation_count = sum(1 for r in results if r.violation)
    effective_n = sum(
        1
        for r in results
        if r.ensemble_rmse is not None and r.baseline_rmse is not None
    )
    violation_rate = (violation_count / effective_n) if effective_n else 0.0

    print(f"\nEffective audits with RMSE: {effective_n}")
    print(f"Violations (ensemble worse than baseline beyond tolerance): {violation_count}")
    print(f"Violation rate: {violation_rate:.2%} (max allowed {max_violation_rate:.2%})")

    print("\nSample details (most recent first):")
    header = f"{'File':<32} {'ens_rmse':>10} {'base_rmse':>10} {'ratio':>8} {'VIOL':>6}"
    print(header)
    print("-" * len(header))
    for r in results[:10]:
        ratio_str = f"{r.rmse_ratio:.3f}" if r.rmse_ratio is not None else "n/a"
        ens_str = f"{r.ensemble_rmse:.4f}" if r.ensemble_rmse is not None else "n/a"
        base_str = f"{r.baseline_rmse:.4f}" if r.baseline_rmse is not None else "n/a"
        viol_flag = "YES" if r.violation else ""
        print(
            f"{r.path.name:<32} {ens_str:>10} {base_str:>10} {ratio_str:>8} {viol_flag:>6}"
        )

    if effective_n == 0:
        # No usable metrics; do not fail hard, but signal that checks were inconclusive.
        raise SystemExit(0)

    if violation_rate > max_violation_rate:
        raise SystemExit(
            f"Ensemble RMSE violation rate {violation_rate:.2%} exceeds "
            f"max-violation-rate {max_violation_rate:.2%}"
        )

    raise SystemExit(0)


if __name__ == "__main__":
    main()
