#!/usr/bin/env python3
"""
Blocking SARIMAX convergence-budget gate for CI.

Parses warning recorder output for SARIMAX convergence budget events and fails
when rates exceed configured thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any

import yaml


DEFAULT_WARNING_LOG = Path("logs/warnings/warning_events.log")
DEFAULT_MONITOR_CONFIG = Path("config/forecaster_monitoring_ci.yml")
EVENT_CONTEXT = "SARIMAXForecaster.convergence_budget"
EVENT_RE = re.compile(r"event=(?P<event>[a-z_]+)\s+occurrence=(?P<count>\d+)")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _load_thresholds(monitor_config: Path) -> Dict[str, float]:
    raw = _load_yaml(monitor_config)
    fm = raw.get("forecaster_monitoring") or {}
    gate = fm.get("sarimax_convergence_budget") or {}
    return {
        "max_primary_nonconverged_rate": float(
            gate.get("max_primary_nonconverged_rate", 0.50)
        ),
        "max_fallback_nonconverged_rate": float(
            gate.get("max_fallback_nonconverged_rate", 0.20)
        ),
        "max_fallback_usage_rate": float(
            gate.get("max_fallback_usage_rate", 0.50)
        ),
    }


def _load_total_runs(suite_report: Path | None) -> int:
    if suite_report is None or not suite_report.exists():
        return 0
    payload = json.loads(suite_report.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    runs_per_variant = int(meta.get("total_runs_per_variant") or 0)
    variants = meta.get("variants") or []
    return runs_per_variant * len(variants)


def _parse_event_counts(log_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {
        "primary_nonconverged": 0,
        "fallback_converged": 0,
        "fallback_nonconverged": 0,
    }
    if not log_path.exists():
        return counts

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if f"[{EVENT_CONTEXT}]" not in line:
            continue
        match = EVENT_RE.search(line)
        if not match:
            continue
        event = str(match.group("event"))
        occurrence = int(match.group("count"))
        if event in counts:
            counts[event] = max(counts[event], occurrence)
    return counts


def _fmt_rate(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return float(n) / float(d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warning-log", default=str(DEFAULT_WARNING_LOG))
    parser.add_argument("--monitor-config", default=str(DEFAULT_MONITOR_CONFIG))
    parser.add_argument("--suite-report", default="")
    parser.add_argument("--max-primary-nonconverged-rate", type=float, default=None)
    parser.add_argument("--max-fallback-nonconverged-rate", type=float, default=None)
    parser.add_argument("--max-fallback-usage-rate", type=float, default=None)
    args = parser.parse_args()

    warning_log = Path(args.warning_log)
    monitor_config = Path(args.monitor_config)
    suite_report = Path(args.suite_report) if args.suite_report else None

    thresholds = _load_thresholds(monitor_config)
    if args.max_primary_nonconverged_rate is not None:
        thresholds["max_primary_nonconverged_rate"] = float(args.max_primary_nonconverged_rate)
    if args.max_fallback_nonconverged_rate is not None:
        thresholds["max_fallback_nonconverged_rate"] = float(args.max_fallback_nonconverged_rate)
    if args.max_fallback_usage_rate is not None:
        thresholds["max_fallback_usage_rate"] = float(args.max_fallback_usage_rate)

    counts = _parse_event_counts(warning_log)
    total_runs = _load_total_runs(suite_report)
    if total_runs <= 0:
        # Fallback so local runs without report still get deterministic output.
        total_runs = max(1, counts["primary_nonconverged"], counts["fallback_converged"], counts["fallback_nonconverged"])

    rates = {
        "primary_nonconverged_rate": _fmt_rate(counts["primary_nonconverged"], total_runs),
        "fallback_nonconverged_rate": _fmt_rate(counts["fallback_nonconverged"], total_runs),
        "fallback_usage_rate": _fmt_rate(counts["fallback_converged"] + counts["fallback_nonconverged"], total_runs),
    }

    breaches = []
    if rates["primary_nonconverged_rate"] > thresholds["max_primary_nonconverged_rate"]:
        breaches.append(
            "primary_nonconverged_rate="
            f"{rates['primary_nonconverged_rate']:.4f} > {thresholds['max_primary_nonconverged_rate']:.4f}"
        )
    if rates["fallback_nonconverged_rate"] > thresholds["max_fallback_nonconverged_rate"]:
        breaches.append(
            "fallback_nonconverged_rate="
            f"{rates['fallback_nonconverged_rate']:.4f} > {thresholds['max_fallback_nonconverged_rate']:.4f}"
        )
    if rates["fallback_usage_rate"] > thresholds["max_fallback_usage_rate"]:
        breaches.append(
            "fallback_usage_rate="
            f"{rates['fallback_usage_rate']:.4f} > {thresholds['max_fallback_usage_rate']:.4f}"
        )

    payload = {
        "warning_log": str(warning_log),
        "suite_report": str(suite_report) if suite_report else None,
        "total_runs": int(total_runs),
        "counts": counts,
        "rates": rates,
        "thresholds": thresholds,
        "breaches": breaches,
    }
    print(json.dumps(payload, indent=2))

    if breaches:
        print("\n[FAIL] SARIMAX convergence budget exceeded:")
        for breach in breaches:
            print(f"- {breach}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
