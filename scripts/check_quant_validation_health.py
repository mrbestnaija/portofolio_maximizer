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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # Local import; dependency already used elsewhere.
except ImportError:  # pragma: no cover - yaml is an explicit project dependency
    yaml = None  # type: ignore[assignment]

ROOT_PATH = Path(__file__).resolve().parent.parent

# Phase 7.13-C1: prefer central path constants; fall back to local computation
try:
    from etl.paths import QUANT_VALIDATION_JSONL as DEFAULT_LOG_PATH
except ImportError:
    DEFAULT_LOG_PATH = ROOT_PATH / "logs" / "signals" / "quant_validation.jsonl"

DEFAULT_MONITORING_CONFIG = ROOT_PATH / "config" / "forecaster_monitoring.yml"


@dataclass
class GlobalHealthSummary:
    total: int
    pass_count: int
    fail_count: int
    negative_expected_profit_count: int
    skipped_action_count: int = 0
    skipped_mode_count: int = 0
    skipped_scope_count: int = 0

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


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        # Date-only fallback (YYYY-MM-DD).
        try:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_run_id(rec: Dict[str, Any]) -> Optional[str]:
    run_id = rec.get("run_id")
    if run_id is not None:
        return str(run_id)
    pipeline_id = rec.get("pipeline_id")
    if pipeline_id is not None:
        return str(pipeline_id)
    qv = rec.get("quant_validation") or {}
    run_id = qv.get("run_id")
    if run_id is not None:
        return str(run_id)
    return None


def _summarize_global(
    entries: List[Dict[str, Any]],
    exclude_modes: Optional[List[str]] = None,
    include_actions: Optional[List[str]] = None,
    run_ids: Optional[List[str]] = None,
    since_ts: Optional[datetime] = None,
) -> GlobalHealthSummary:
    total = 0
    pass_count = 0
    fail_count = 0
    neg_exp_profit = 0
    skipped_action = 0
    skipped_mode = 0
    skipped_scope = 0
    exclude_set = set(m.lower() for m in (exclude_modes or []))
    include_actions_set = {
        action.strip().upper()
        for action in (include_actions or [])
        if action and str(action).strip()
    }
    if "ALL" in include_actions_set:
        include_actions_set = set()
    run_id_set = {
        str(v).strip()
        for v in (run_ids or [])
        if v is not None and str(v).strip()
    }

    for rec in entries:
        if since_ts is not None:
            ts = _parse_iso_timestamp(rec.get("timestamp"))
            if ts is None or ts < since_ts:
                skipped_scope += 1
                continue
        if run_id_set:
            rec_run_id = _extract_run_id(rec)
            if rec_run_id is None or rec_run_id not in run_id_set:
                skipped_scope += 1
                continue

        if include_actions_set:
            action = str(rec.get("action") or "").strip().upper()
            if action not in include_actions_set:
                skipped_action += 1
                continue

        # Skip entries whose execution_mode is in the exclusion list.
        # This allows proof-mode entries (max_holding=5, artificial constraints)
        # to be excluded from the RED gate calculation.
        if exclude_set:
            exec_mode = str(
                rec.get("execution_mode")
                or (rec.get("quant_validation") or {}).get("execution_mode")
                or ""
            ).lower()
            proof_raw = rec.get("proof_mode")
            if proof_raw is None:
                proof_raw = (rec.get("quant_validation") or {}).get("proof_mode")
            if isinstance(proof_raw, str):
                proof_flag = proof_raw.strip().lower() in {"1", "true", "yes", "on"}
            else:
                proof_flag = bool(proof_raw)
            if exec_mode in exclude_set or (proof_flag and "proof" in exclude_set):
                skipped_mode += 1
                continue

        total += 1
        status = str(
            rec.get("status") or (rec.get("quant_validation") or {}).get("status") or ""
        ).upper()
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
        skipped_action_count=skipped_action,
        skipped_mode_count=skipped_mode,
        skipped_scope_count=skipped_scope,
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
    parser.add_argument(
        "--exclude-mode",
        nargs="*",
        default=[],
        metavar="MODE",
        help=(
            "Exclude entries whose execution_mode matches any of these values "
            "from FAIL-rate calculation. Useful to strip proof-mode runs "
            "(max_holding=5) that have structurally inflated FAIL rates. "
            "Example: --exclude-mode proof diagnostic"
        ),
    )
    parser.add_argument(
        "--include-action",
        nargs="*",
        default=["BUY", "SELL"],
        metavar="ACTION",
        help=(
            "Only include these actions when computing health metrics. "
            "Defaults to actionable trades only (BUY/SELL). "
            "Use --include-action ALL to disable action filtering."
        ),
    )
    parser.add_argument(
        "--run-id",
        nargs="*",
        default=[],
        metavar="RUN_ID",
        help=(
            "Only include entries matching these run IDs. "
            "Matches root run_id and pipeline_id fields."
        ),
    )
    parser.add_argument(
        "--since",
        default=None,
        help=(
            "Only include entries with timestamp >= this ISO value "
            "(e.g. 2026-02-19T14:40:46+00:00 or 2026-02-19)."
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

    since_ts = None
    if args.since:
        since_ts = _parse_iso_timestamp(args.since)
        if since_ts is None:
            raise SystemExit(
                f"Invalid --since value '{args.since}'. "
                "Use ISO format, e.g. 2026-02-19T14:40:46+00:00."
            )

    summary = _summarize_global(
        entries,
        exclude_modes=args.exclude_mode,
        include_actions=args.include_action,
        run_ids=args.run_id,
        since_ts=since_ts,
    )

    print("=== Quant Validation Global Health ===")
    print(
        "  Filters                      : "
        f"actions={args.include_action or ['ALL']} "
        f"exclude_mode={args.exclude_mode or []} "
        f"run_id={args.run_id or []} "
        f"since={args.since or 'none'}"
    )
    print(f"  Skipped (action filter)      : {summary.skipped_action_count}")
    print(f"  Skipped (mode filter)        : {summary.skipped_mode_count}")
    print(f"  Skipped (scope filter)       : {summary.skipped_scope_count}")
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

    if summary.total == 0:
        print(
            "Health check VIOLATION: no entries remain after filters "
            "(action/mode/scope)."
        )
        raise SystemExit(1)

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
