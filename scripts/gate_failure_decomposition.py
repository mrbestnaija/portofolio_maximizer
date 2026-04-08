#!/usr/bin/env python3
"""Decompose production gate failures into performance, linkage, and hygiene components."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from scripts.robustness_thresholds import threshold_map as _threshold_map  # type: ignore
except Exception:
    _threshold_map = None

try:
    from scripts.production_gate_contract import (
        legacy_phase3_ready as _legacy_phase3_ready,
        legacy_phase3_reason as _legacy_phase3_reason,
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )
except Exception:
    from production_gate_contract import (  # type: ignore
        legacy_phase3_ready as _legacy_phase3_ready,
        legacy_phase3_reason as _legacy_phase3_reason,
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )


DEFAULT_GATE_ARTIFACT = Path("logs") / "audit_gate" / "production_gate_latest.json"
DEFAULT_OUT_JSON = Path("logs") / "audit_gate" / "production_gate_decomposition_latest.json"
DEFAULT_OUT_MD = Path("logs") / "audit_gate" / "production_gate_decomposition_latest.md"
DEFAULT_SUMMARY_CACHE = Path("logs") / "forecast_audits_cache" / "latest_summary.json"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_mtime_utc(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _metric(value: Any, threshold: str, passed: bool) -> Dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "pass": bool(passed),
    }


def _diagnostic_metric(value: Any) -> Dict[str, Any]:
    return {
        "value": value,
        "threshold": "diagnostic_only",
        "pass": value is not None,
    }


def _top_counts(items: list[str], *, limit: int = 10) -> list[Dict[str, Any]]:
    counts = Counter(items)
    out: list[Dict[str, Any]] = []
    for key, value in counts.most_common(max(int(limit), 0)):
        out.append({"reason_code": key, "count": int(value)})
    return out


def _load_thresholds() -> Dict[str, Any]:
    defaults = {
        "r3_min_trades": 30,
        "r3_min_win_rate": 0.45,
        "r3_min_profit_factor": 1.10,
    }
    if _threshold_map is None:
        return defaults
    try:
        payload = _threshold_map()
    except Exception:
        return defaults
    if not isinstance(payload, dict):
        return defaults
    for key in list(defaults.keys()):
        if key in payload:
            defaults[key] = payload[key]
    return defaults


def _normalize_reason(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text else "UNKNOWN"


def _summary_matches_gate_inputs(
    summary_payload: Dict[str, Any],
    gate_inputs: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    mismatch: Dict[str, Any] = {}

    expected_audit_dir = gate_inputs.get("audit_dir")
    summary_audit_dir = summary_payload.get("audit_dir")
    if expected_audit_dir and summary_audit_dir:
        try:
            if Path(str(expected_audit_dir)).resolve() != Path(str(summary_audit_dir)).resolve():
                mismatch["audit_dir"] = {
                    "expected": str(expected_audit_dir),
                    "observed": str(summary_audit_dir),
                }
        except Exception:
            mismatch["audit_dir"] = {
                "expected": str(expected_audit_dir),
                "observed": str(summary_audit_dir),
            }

    expected_max_files = gate_inputs.get("max_files")
    summary_max_files = summary_payload.get("max_files")
    if expected_max_files is not None and summary_max_files is not None:
        if _safe_int(summary_max_files, -1) != _safe_int(expected_max_files, -1):
            mismatch["max_files"] = {
                "expected": _safe_int(expected_max_files, -1),
                "observed": _safe_int(summary_max_files, -1),
            }

    expected_include_research = gate_inputs.get("include_research")
    summary_scope = summary_payload.get("scope")
    summary_include_research = (
        summary_scope.get("include_research")
        if isinstance(summary_scope, dict)
        else None
    )
    if expected_include_research is not None and summary_include_research is not None:
        if _safe_bool(summary_include_research) != _safe_bool(expected_include_research):
            mismatch["include_research"] = {
                "expected": _safe_bool(expected_include_research),
                "observed": _safe_bool(summary_include_research),
            }

    return (len(mismatch) == 0, mismatch)


def _extract_reason_breakdown(
    summary_payload: Dict[str, Any] | None,
    *,
    readiness: Dict[str, Any],
    gate_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    invalid_total_expected = _safe_int(readiness.get("invalid_context_count"), 0)
    non_trade_total_expected = _safe_int(readiness.get("non_trade_context_count"), 0)

    def _fallback_unattributed(
        *,
        available: bool,
        binding_match: bool,
        mismatch: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        invalid_rows = []
        non_trade_rows = []
        if invalid_total_expected > 0:
            invalid_rows.append(
                {
                    "reason_code": "UNATTRIBUTED_INVALID_CONTEXT",
                    "count": int(invalid_total_expected),
                }
            )
        if non_trade_total_expected > 0:
            non_trade_rows.append(
                {
                    "reason_code": "UNATTRIBUTED_NON_TRADE_CONTEXT",
                    "count": int(non_trade_total_expected),
                }
            )
        return {
            "available": available,
            "binding_match": binding_match,
            "binding_mismatch": mismatch or {},
            "summary_path": None,
            "invalid_context_top_reasons": invalid_rows,
            "non_trade_context_top_reasons": non_trade_rows,
            "invalid_context_total_expected": int(invalid_total_expected),
            "non_trade_context_total_expected": int(non_trade_total_expected),
        }

    if not isinstance(summary_payload, dict):
        return _fallback_unattributed(available=False, binding_match=False)

    matches, mismatch = _summary_matches_gate_inputs(summary_payload, gate_inputs)

    windows = summary_payload.get("dataset_windows")
    if not isinstance(windows, list):
        windows = []

    invalid_counter: Counter[str] = Counter()
    non_trade_counter: Counter[str] = Counter()
    for row in windows:
        if not isinstance(row, dict):
            continue
        outcome_status = str(row.get("outcome_status") or "").strip().upper()
        outcome_reason = _normalize_reason(row.get("outcome_reason"))
        if outcome_status == "INVALID_CONTEXT":
            invalid_counter[outcome_reason] += 1
        elif outcome_status == "NON_TRADE_CONTEXT":
            non_trade_counter[outcome_reason] += 1

    window_counts = (
        summary_payload.get("window_counts")
        if isinstance(summary_payload.get("window_counts"), dict)
        else {}
    )
    if not windows and not window_counts:
        return _fallback_unattributed(
            available=False,
            binding_match=matches,
            mismatch=mismatch,
        )
    missing_exec_count = _safe_int(
        window_counts.get("n_outcome_windows_missing_execution_metadata"),
        0,
    )
    if missing_exec_count > 0:
        invalid_counter["MISSING_EXECUTION_METADATA"] = max(
            invalid_counter["MISSING_EXECUTION_METADATA"],
            missing_exec_count,
        )
    missing_signal_count = _safe_int(window_counts.get("n_outcome_windows_no_signal_id"), 0)
    if missing_signal_count > 0:
        invalid_counter["MISSING_SIGNAL_ID"] = max(
            invalid_counter["MISSING_SIGNAL_ID"],
            missing_signal_count,
        )

    if invalid_total_expected <= 0:
        invalid_total_expected = _safe_int(
            window_counts.get("n_outcome_windows_invalid_context"),
            0,
        )
    if non_trade_total_expected <= 0:
        non_trade_total_expected = _safe_int(
            window_counts.get("n_outcome_windows_non_trade_context"),
            0,
        )

    invalid_attributed = sum(int(v) for v in invalid_counter.values())
    non_trade_attributed = sum(int(v) for v in non_trade_counter.values())
    invalid_remainder = max(int(invalid_total_expected) - int(invalid_attributed), 0)
    non_trade_remainder = max(int(non_trade_total_expected) - int(non_trade_attributed), 0)
    if invalid_remainder > 0:
        invalid_counter["UNATTRIBUTED_INVALID_CONTEXT"] += int(invalid_remainder)
    if non_trade_remainder > 0:
        non_trade_counter["UNATTRIBUTED_NON_TRADE_CONTEXT"] += int(non_trade_remainder)

    return {
        "available": True,
        "binding_match": matches,
        "binding_mismatch": {} if matches else mismatch,
        "summary_path": summary_payload.get("audit_dir"),
        "invalid_context_top_reasons": _top_counts(
            list(invalid_counter.elements()),
            limit=10,
        ),
        "non_trade_context_top_reasons": _top_counts(
            list(non_trade_counter.elements()),
            limit=10,
        ),
        "invalid_context_total_expected": int(invalid_total_expected),
        "non_trade_context_total_expected": int(non_trade_total_expected),
    }


def _build_decomposition(
    payload: Dict[str, Any],
    artifact_path: Path,
    summary_payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    lift_gate = payload.get("lift_gate") if isinstance(payload.get("lift_gate"), dict) else {}
    proof = (
        payload.get("profitability_proof")
        if isinstance(payload.get("profitability_proof"), dict)
        else {}
    )
    waterfall = (
        readiness.get("linkage_waterfall")
        if isinstance(readiness.get("linkage_waterfall"), dict)
        else {}
    )
    gate_inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    thresholds = _load_thresholds()

    violation_rate = _safe_float(lift_gate.get("violation_rate"))
    max_violation_rate = _safe_float(lift_gate.get("max_violation_rate"))
    lift_fraction = _safe_float(lift_gate.get("lift_fraction"))
    min_lift_fraction = _safe_float(lift_gate.get("min_lift_fraction"))
    pf = _safe_float(proof.get("profit_factor"))
    wr = _safe_float(proof.get("win_rate"))
    pnl = _safe_float(proof.get("total_pnl"))
    trading_days = _safe_int(proof.get("trading_days"), 0)
    closed_trades = _safe_int(proof.get("closed_trades"), 0)
    proof_evidence = (
        proof.get("evidence_progress")
        if isinstance(proof.get("evidence_progress"), dict)
        else {}
    )
    min_closed_trades = _safe_int(
        proof_evidence.get("min_closed_trades"),
        _safe_int(thresholds.get("r3_min_trades"), 30),
    )
    min_trading_days = _safe_int(proof_evidence.get("min_trading_days"), 21)
    min_profit_factor = _safe_float(thresholds.get("r3_min_profit_factor"))

    matched = _safe_int(readiness.get("outcome_matched"), 0)
    eligible = _safe_int(readiness.get("outcome_eligible"), 0)
    matched_ratio = _safe_float(readiness.get("matched_over_eligible"))
    non_trade = _safe_int(readiness.get("non_trade_context_count"), 0)
    invalid = _safe_int(readiness.get("invalid_context_count"), 0)

    violation_pass = (
        violation_rate is not None
        and max_violation_rate is not None
        and violation_rate <= max_violation_rate
    )
    lift_fraction_pass = (
        lift_fraction is not None
        and min_lift_fraction is not None
        and lift_fraction >= min_lift_fraction
    )
    matched_count_pass = matched >= 10
    matched_ratio_pass = matched_ratio is not None and matched_ratio >= 0.8
    hygiene_non_trade_pass = non_trade == 0
    hygiene_invalid_pass = invalid == 0

    components = {
        "PERFORMANCE_BLOCKER": {
            "pass": _safe_bool(readiness.get("gates_pass"), default=False),
            "metrics": {
                "lift_violation_rate": _metric(
                    violation_rate,
                    f"<= {max_violation_rate}",
                    violation_pass,
                ),
                "lift_fraction": _metric(
                    lift_fraction,
                    f">= {min_lift_fraction}",
                    lift_fraction_pass,
                ),
                "proof_pass": _metric(
                    _safe_bool(proof.get("pass"), default=False),
                    "must_be_true",
                    _safe_bool(proof.get("pass"), default=False),
                ),
                "profit_factor": _metric(
                    pf,
                    f">= {min_profit_factor}",
                    False if (pf is None or min_profit_factor is None) else pf >= min_profit_factor,
                ),
                "win_rate": _diagnostic_metric(wr),
                "total_pnl": _metric(pnl, "context metric", False if pnl is None else True),
                "closed_trades": _metric(
                    closed_trades,
                    f">= {min_closed_trades} (runway)",
                    closed_trades >= min_closed_trades,
                ),
                "trading_days": _metric(
                    trading_days,
                    f">= {min_trading_days} (runway)",
                    trading_days >= min_trading_days,
                ),
            },
        },
        "LINKAGE_BLOCKER": {
            "pass": _safe_bool(readiness.get("linkage_pass"), default=False),
            "metrics": {
                "outcome_matched": _metric(matched, ">= 10", matched_count_pass),
                "outcome_eligible": _metric(eligible, "context metric", eligible > 0),
                "matched_over_eligible": _metric(matched_ratio, ">= 0.80", matched_ratio_pass),
            },
            "waterfall": {
                "raw_candidates": _safe_int(waterfall.get("raw_candidates"), 0),
                "production_only": _safe_int(waterfall.get("production_only"), 0),
                "linked": _safe_int(waterfall.get("linked"), 0),
                "hygiene_pass": _safe_int(waterfall.get("hygiene_pass"), 0),
                "matched": _safe_int(waterfall.get("matched"), 0),
                "excluded_non_trade_context": _safe_int(
                    waterfall.get("excluded_non_trade_context"),
                    0,
                ),
                "excluded_invalid_context": _safe_int(
                    waterfall.get("excluded_invalid_context"),
                    0,
                ),
            },
        },
        "HYGIENE_BLOCKER": {
            "pass": _safe_bool(readiness.get("evidence_hygiene_pass"), default=False),
            "metrics": {
                "non_trade_context_count": _metric(non_trade, "== 0", hygiene_non_trade_pass),
                "invalid_context_count": _metric(invalid, "== 0", hygiene_invalid_pass),
            },
        },
    }

    table_rows = []
    for name, block in components.items():
        for metric_name, metric_value in block.get("metrics", {}).items():
            table_rows.append(
                {
                    "component": name,
                    "metric": metric_name,
                    "value": metric_value.get("value"),
                    "threshold": metric_value.get("threshold"),
                    "pass": _safe_bool(metric_value.get("pass"), default=False),
                }
            )

    reason_breakdown = _extract_reason_breakdown(
        summary_payload,
        readiness=readiness,
        gate_inputs=gate_inputs,
    )

    visualization = {
        "component_status": [
            {
                "name": component,
                "pass": _safe_bool(block.get("pass"), default=False),
            }
            for component, block in components.items()
        ],
        "linkage_waterfall": {
            "raw_candidates": _safe_int(waterfall.get("raw_candidates"), 0),
            "production_only": _safe_int(waterfall.get("production_only"), 0),
            "linked": _safe_int(waterfall.get("linked"), 0),
            "hygiene_pass": _safe_int(waterfall.get("hygiene_pass"), 0),
            "matched": _safe_int(waterfall.get("matched"), 0),
        },
    }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(artifact_path.resolve()),
        "source_timestamp_utc": payload.get("timestamp_utc"),
        "source_artifact_mtime_utc": _artifact_mtime_utc(artifact_path),
        "phase3_ready": bool(_phase3_strict_ready(payload)),
        "phase3_reason": _phase3_strict_reason(payload),
        "phase3_strict_ready": bool(_phase3_strict_ready(payload)),
        "phase3_strict_reason": _phase3_strict_reason(payload),
        "phase3_legacy_ready": bool(_legacy_phase3_ready(payload)),
        "phase3_legacy_reason": _legacy_phase3_reason(payload),
        "components": components,
        "reason_breakdown": reason_breakdown,
        "visualization": visualization,
        "table": table_rows,
    }


def _report_is_stale(
    *,
    report: Dict[str, Any] | None,
    artifact_path: Path,
    gate_payload: Dict[str, Any],
    output_md_path: Path | None,
) -> tuple[bool, str]:
    if not isinstance(report, dict) or not report:
        return True, "missing_report"

    expected_source = str(artifact_path.resolve())
    if str(report.get("source_artifact") or "").strip() != expected_source:
        return True, "source_artifact_mismatch"

    gate_timestamp = str(gate_payload.get("timestamp_utc") or "").strip()
    report_timestamp = str(report.get("source_timestamp_utc") or "").strip()
    if gate_timestamp and report_timestamp != gate_timestamp:
        return True, "source_timestamp_mismatch"

    gate_mtime = _artifact_mtime_utc(artifact_path)
    report_source_mtime = str(report.get("source_artifact_mtime_utc") or "").strip()
    if gate_mtime and report_source_mtime and report_source_mtime != gate_mtime:
        return True, "source_mtime_mismatch"

    generated_at = str(report.get("generated_utc") or "").strip()
    if not generated_at:
        return True, "missing_generated_utc"

    if output_md_path is not None and not output_md_path.exists():
        return True, "missing_markdown"

    return False, "up_to_date"


def refresh_decomposition_report(
    *,
    artifact_path: Path,
    output_json_path: Path,
    summary_cache_path: Path | None = None,
    output_md_path: Path | None = None,
    force: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    payload = _load_json(artifact_path)
    if payload is None:
        raise ValueError(f"unable_to_load_gate_artifact: {artifact_path}")

    existing_report = _load_json(output_json_path) if output_json_path.exists() else None
    stale, reason = _report_is_stale(
        report=existing_report,
        artifact_path=artifact_path,
        gate_payload=payload,
        output_md_path=output_md_path,
    )
    refresh_reason = "forced" if force else reason
    should_refresh = bool(force or stale)

    if should_refresh:
        summary_payload = (
            _load_json(summary_cache_path)
            if isinstance(summary_cache_path, Path) and summary_cache_path.exists()
            else None
        )
        report = _build_decomposition(
            payload,
            artifact_path=artifact_path,
            summary_payload=summary_payload,
        )
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if isinstance(output_md_path, Path):
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            output_md_path.write_text(_render_markdown(report), encoding="utf-8")
    else:
        report = existing_report or {}
        if isinstance(output_md_path, Path) and not output_md_path.exists():
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            output_md_path.write_text(_render_markdown(report), encoding="utf-8")

    refresh_result = {
        "ok": True,
        "refreshed": should_refresh,
        "reason": refresh_reason,
        "artifact_path": str(artifact_path.resolve()),
        "output_json_path": str(output_json_path.resolve()),
        "output_md_path": str(output_md_path.resolve()) if isinstance(output_md_path, Path) else "",
        "summary_cache_path": str(summary_cache_path.resolve()) if isinstance(summary_cache_path, Path) else "",
    }
    return report, refresh_result


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Production Gate Decomposition")
    lines.append("")
    lines.append(f"- Source artifact: `{report.get('source_artifact')}`")
    lines.append(f"- Source timestamp: `{report.get('source_timestamp_utc')}`")
    lines.append(f"- Phase3 reason: `{report.get('phase3_reason')}`")
    lines.append(f"- Phase3 ready: `{int(_safe_bool(report.get('phase3_ready')))}`")
    if (
        _safe_bool(report.get("phase3_legacy_ready")) != _safe_bool(report.get("phase3_ready"))
        or str(report.get("phase3_legacy_reason") or "").strip() != str(report.get("phase3_reason") or "").strip()
    ):
        lines.append(f"- Phase3 legacy reason: `{report.get('phase3_legacy_reason')}`")
        lines.append(f"- Phase3 legacy ready: `{int(_safe_bool(report.get('phase3_legacy_ready')))}`")
    lines.append("")
    lines.append("## Blockers")
    for component in ("PERFORMANCE_BLOCKER", "LINKAGE_BLOCKER", "HYGIENE_BLOCKER"):
        block = report.get("components", {}).get(component, {})
        status = "PASS" if _safe_bool(block.get("pass")) else "FAIL"
        lines.append(f"### {component} - {status}")
        metrics = block.get("metrics", {})
        if isinstance(metrics, dict):
            for metric_name, metric in metrics.items():
                lines.append(
                    "- "
                    f"{metric_name}: value={metric.get('value')} "
                    f"threshold={metric.get('threshold')} "
                    f"pass={int(_safe_bool(metric.get('pass')))}"
                )
        if component == "LINKAGE_BLOCKER":
            waterfall = block.get("waterfall", {})
            if isinstance(waterfall, dict):
                lines.append("- linkage_waterfall:")
                max_value = max(_safe_int(waterfall.get("raw_candidates"), 0), 1)
                for key in ("raw_candidates", "production_only", "linked", "hygiene_pass", "matched"):
                    value = _safe_int(waterfall.get(key), 0)
                    bar = "#" * max(int((value / max_value) * 30), 0)
                    lines.append(f"  - {key}: {value:>4} {bar}")
        lines.append("")
    rb = report.get("reason_breakdown", {})
    lines.append("## Reason Breakdown")
    if isinstance(rb, dict):
        lines.append(f"- binding_match: `{int(_safe_bool(rb.get('binding_match'), False))}`")
        mismatch = rb.get("binding_mismatch", {})
        if isinstance(mismatch, dict) and mismatch:
            lines.append(f"- binding_mismatch: `{json.dumps(mismatch, sort_keys=True)}`")
        lines.append("- INVALID_CONTEXT top reasons:")
        invalid = rb.get("invalid_context_top_reasons", [])
        if isinstance(invalid, list) and invalid:
            for item in invalid:
                lines.append(f"  - {item.get('reason_code')}: {item.get('count')}")
        else:
            lines.append("  - (none)")
        lines.append("- NON_TRADE_CONTEXT top reasons:")
        non_trade = rb.get("non_trade_context_top_reasons", [])
        if isinstance(non_trade, list) and non_trade:
            for item in non_trade:
                lines.append(f"  - {item.get('reason_code')}: {item.get('count')}")
        else:
            lines.append("  - (none)")
    lines.append("")
    return "\n".join(lines)


def _print_report(report: Dict[str, Any]) -> None:
    print("=== Production Gate Failure Decomposition ===")
    print(f"Source artifact : {report.get('source_artifact')}")
    print(f"Source timestamp: {report.get('source_timestamp_utc')}")
    print(f"Phase3 reason   : {report.get('phase3_reason')}")
    print(f"Phase3 ready    : {int(_safe_bool(report.get('phase3_ready')))}")
    if (
        _safe_bool(report.get("phase3_legacy_ready")) != _safe_bool(report.get("phase3_ready"))
        or str(report.get("phase3_legacy_reason") or "").strip() != str(report.get("phase3_reason") or "").strip()
    ):
        print(f"Phase3 legacy   : {int(_safe_bool(report.get('phase3_legacy_ready')))} ({report.get('phase3_legacy_reason')})")
    print("")
    for component in ("PERFORMANCE_BLOCKER", "LINKAGE_BLOCKER", "HYGIENE_BLOCKER"):
        block = report.get("components", {}).get(component, {})
        print(f"{component}: {'PASS' if _safe_bool(block.get('pass')) else 'FAIL'}")
        metrics = block.get("metrics", {})
        for metric_name, metric in metrics.items():
            print(
                "  - "
                f"{metric_name}: value={metric.get('value')} "
                f"threshold={metric.get('threshold')} "
                f"pass={int(_safe_bool(metric.get('pass')))}"
            )
        if component == "LINKAGE_BLOCKER":
            waterfall = block.get("waterfall", {})
            print("  - linkage_waterfall:")
            print(
                "    "
                f"raw={waterfall.get('raw_candidates')} "
                f"production={waterfall.get('production_only')} "
                f"linked={waterfall.get('linked')} "
                f"hygiene_pass={waterfall.get('hygiene_pass')} "
                f"matched={waterfall.get('matched')} "
                f"excluded_non_trade={waterfall.get('excluded_non_trade_context')} "
                f"excluded_invalid={waterfall.get('excluded_invalid_context')}"
            )
        print("")

    rb = report.get("reason_breakdown", {})
    if isinstance(rb, dict):
        print("Reason-code decomposition:")
        invalid = rb.get("invalid_context_top_reasons", [])
        non_trade = rb.get("non_trade_context_top_reasons", [])
        print("  - INVALID_CONTEXT top reasons:")
        if isinstance(invalid, list) and invalid:
            for item in invalid:
                print(f"    {item.get('reason_code')}: {item.get('count')}")
        else:
            print("    (none)")
        print("  - NON_TRADE_CONTEXT top reasons:")
        if isinstance(non_trade, list) and non_trade:
            for item in non_trade:
                print(f"    {item.get('reason_code')}: {item.get('count')}")
        else:
            print("    (none)")
        print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompose production gate failures by blocker type.")
    parser.add_argument(
        "--gate-artifact",
        default=str(DEFAULT_GATE_ARTIFACT),
        help="Path to production gate artifact JSON.",
    )
    parser.add_argument(
        "--out-json",
        default=str(DEFAULT_OUT_JSON),
        help="Output path for decomposition JSON.",
    )
    parser.add_argument(
        "--summary-cache",
        default=str(DEFAULT_SUMMARY_CACHE),
        help="Optional summary cache path for reason-code decomposition.",
    )
    parser.add_argument(
        "--out-md",
        default=str(DEFAULT_OUT_MD),
        help="Output path for markdown report with waterfall visualization.",
    )
    args = parser.parse_args()

    artifact_path = Path(args.gate_artifact)
    if _load_json(artifact_path) is None:
        print(f"[ERROR] unable_to_load_gate_artifact: {artifact_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out_json)
    md_path = Path(args.out_md)
    summary_path = Path(args.summary_cache)
    try:
        report, _ = refresh_decomposition_report(
            artifact_path=artifact_path,
            output_json_path=out_path,
            summary_cache_path=summary_path,
            output_md_path=md_path,
            force=True,
        )
    except ValueError:
        print(f"[ERROR] unable_to_load_gate_artifact: {artifact_path}", file=sys.stderr)
        return 1

    _print_report(report)
    print(f"Output JSON     : {out_path}")
    print(f"Output MD       : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
