#!/usr/bin/env python3
"""
check_quant_validation_health.py
--------------------------------

CI/brutal helper that inspects logs/signals/quant_validation.jsonl and
enforces basic health thresholds derived from config/forecaster_monitoring.yml.

Behaviour:
  - Computes global PASS/FAIL counts.
  - Flags when the fraction of FAIL entries exceeds a configurable ceiling.
  - Flags when a configurable fraction of entries have negative expected_profit.
  - Classifies the global state into GREEN/YELLOW/RED using a softer warning
    band (YELLOW) vs a hard CI gate (RED).
  - Exits non-zero only when the hard RED ceilings are violated so CI can gate
    deployments or further automation.

This script is intentionally lightweight and read-only: it does not modify
any configuration or database state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # Local import; dependency already used elsewhere.
except ImportError:  # pragma: no cover - yaml is an explicit project dependency
    yaml = None  # type: ignore[assignment]

ROOT_PATH = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = ROOT_PATH / "logs" / "signals" / "quant_validation.jsonl"
DEFAULT_MONITORING_CONFIG = ROOT_PATH / "config" / "forecaster_monitoring.yml"


@dataclass
class GlobalHealthSummary:
    total: int
    pass_count: int
    fail_count: int
    negative_expected_profit_count: int

    @property
    def fail_fraction(self) -> float:
        return self.fail_count / self.total if self.total else 0.0

    @property
    def negative_expected_profit_fraction(self) -> float:
        return (
            self.negative_expected_profit_count / self.total if self.total else 0.0
        )


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"quant_validation log not found at {path}")

    entries: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(payload)

    if not entries:
        raise SystemExit(f"No quant validation entries found in {path}")
    return entries


def _load_monitoring_cfg(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists() or yaml is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("forecaster_monitoring") or {}


def _summarize_global(entries: List[Dict[str, Any]]) -> GlobalHealthSummary:
    total = 0
    pass_count = 0
    fail_count = 0
    neg_exp_profit = 0

    for rec in entries:
        total += 1
        status = rec.get("status") or (rec.get("quant_validation") or {}).get(
            "status"
        )
        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1

        # expected_profit may live either on the root or inside quant_validation.
        exp_profit = rec.get("expected_profit")
        if exp_profit is None:
            qv = rec.get("quant_validation") or {}
            exp_profit = qv.get("expected_profit")
        if isinstance(exp_profit, (int, float)) and exp_profit < 0:
            neg_exp_profit += 1

    return GlobalHealthSummary(
        total=total,
        pass_count=pass_count,
        fail_count=fail_count,
        negative_expected_profit_count=neg_exp_profit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce basic health thresholds over logs/signals/quant_validation.jsonl "
            "for CI/brutal gating."
        )
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Path to quant_validation.jsonl "
        "(default: logs/signals/quant_validation.jsonl)",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_MONITORING_CONFIG),
        help=(
            "Optional path to forecaster_monitoring.yml "
            "(default: config/forecaster_monitoring.yml if present)"
        ),
    )
    parser.add_argument(
        "--max-fail-fraction",
        type=float,
        default=None,
        help=(
            "Maximum allowed fraction of FAIL entries before exiting non-zero. "
            "If omitted, tries forecaster_monitoring.quant_validation.max_fail_fraction "
            "or defaults to 0.95."
        ),
    )
    parser.add_argument(
        "--max-negative-expected-profit-fraction",
        type=float,
        default=None,
        help=(
            "Maximum allowed fraction of entries with negative expected_profit. "
            "If omitted, tries "
            "forecaster_monitoring.quant_validation.max_negative_expected_profit_fraction "
            "or defaults to 0.50."
        ),
    )
    parser.add_argument(
        "--warn-fail-fraction",
        type=float,
        default=None,
        help=(
            "Warning band for FAIL fraction. Values above this but below the "
            "hard max_fail_fraction will classify the run as YELLOW "
            "(warning-only). If omitted, tries "
            "forecaster_monitoring.quant_validation.warn_fail_fraction or "
            "defaults to the hard ceiling (no YELLOW band)."
        ),
    )
    parser.add_argument(
        "--warn-negative-expected-profit-fraction",
        type=float,
        default=None,
        help=(
            "Warning band for negative expected_profit fraction. Values above "
            "this but below the hard "
            "max_negative_expected_profit_fraction will classify the run as "
            "YELLOW (warning-only). If omitted, tries "
            "forecaster_monitoring.quant_validation."
            "warn_negative_expected_profit_fraction or defaults to the hard "
            "ceiling (no YELLOW band)."
        ),
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    entries = _load_entries(log_path)

    monitoring_cfg = _load_monitoring_cfg(
        Path(args.config_path) if args.config_path else None
    )
    qv_cfg = monitoring_cfg.get("quant_validation") or {}

    max_fail_frac = (
        args.max_fail_fraction
        if args.max_fail_fraction is not None
        else float(qv_cfg.get("max_fail_fraction", 0.95))
    )
    max_neg_exp_frac = (
        args.max_negative_expected_profit_fraction
        if args.max_negative_expected_profit_fraction is not None
        else float(qv_cfg.get("max_negative_expected_profit_fraction", 0.50))
    )
    warn_fail_frac = (
        args.warn_fail_fraction
        if args.warn_fail_fraction is not None
        else float(qv_cfg.get("warn_fail_fraction", max_fail_frac))
    )
    warn_neg_exp_frac = (
        args.warn_negative_expected_profit_fraction
        if args.warn_negative_expected_profit_fraction is not None
        else float(
            qv_cfg.get(
                "warn_negative_expected_profit_fraction", max_neg_exp_frac
            )
        )
    )

    summary = _summarize_global(entries)

    print("=== Quant Validation Global Health ===")
    print(f"  Total entries                : {summary.total}")
    print(f"  PASS                         : {summary.pass_count}")
    print(f"  FAIL                         : {summary.fail_count}")
    print(
        f"  FAIL fraction                : {summary.fail_fraction:.3f} "
        f"(max allowed={max_fail_frac:.3f})"
    )
    print(
        f"  Negative expected_profit     : {summary.negative_expected_profit_count}"
    )
    print(
        f"  Negative expected_profit frac: "
        f"{summary.negative_expected_profit_fraction:.3f} "
        f"(max allowed={max_neg_exp_frac:.3f})"
    )

    violations: List[str] = []
    if summary.fail_fraction > max_fail_frac:
        violations.append("FAIL_fraction_exceeds_max")
    if summary.negative_expected_profit_fraction > max_neg_exp_frac:
        violations.append("negative_expected_profit_fraction_exceeds_max")

    # Determine GREEN/YELLOW/RED classification using the warning band.
    # RED: any hard ceiling violated -> CI/brutal must fail.
    # YELLOW: within hard ceilings but above warning band -> advisory.
    # GREEN: below warning band on all monitored dimensions.
    status: str
    if violations:
        status = "RED"
    elif (
        summary.fail_fraction > warn_fail_frac
        or summary.negative_expected_profit_fraction > warn_neg_exp_frac
    ):
        status = "YELLOW"
    else:
        status = "GREEN"

    print(f"\nGlobal health classification   : {status}")

    if status == "RED":
        joined = ", ".join(violations)
        print(f"Health check VIOLATIONS: {joined}")
        raise SystemExit(1)
    elif status == "YELLOW":
        print(
            "Health check WARNING: metrics in warning band but below hard CI gate "
            "(treat as research / needs attention)."
        )
        # Advisory-only: return cleanly so callers embedding main() do not have to
        # special-case a SystemExit(0).
        return

    print("Health check OK: quant validation within configured limits.")


if __name__ == "__main__":
    main()
