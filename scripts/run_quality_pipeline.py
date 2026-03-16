"""
Run the read-only quality pipeline end-to-end and emit one JSON status contract.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_context_quality import DEFAULT_OUTPUT as DEFAULT_CONTEXT_OUT
from scripts.compute_context_quality import compute_context_quality
from scripts.compute_ticker_eligibility import DEFAULT_OUTPUT as DEFAULT_ELIGIBILITY_OUT
from scripts.compute_ticker_eligibility import compute_eligibility
from scripts.apply_ticker_eligibility_gates import (
    DEFAULT_OUTPUT as DEFAULT_ELIGIBILITY_GATES_OUT,
    apply_eligibility_gates,
)
from scripts.data_sufficiency_monitor import DEFAULT_AUDIT_DIR, DEFAULT_DB, run_data_sufficiency
from scripts.generate_performance_charts import (
    DEFAULT_METRICS_PATH,
    DEFAULT_OUT_DIR,
    generate_performance_artifacts,
)
from scripts.quality_pipeline_common import append_threshold_hash_change_warning
from scripts.robustness_thresholds import threshold_map

log = logging.getLogger(__name__)
PIPELINE_VERSION = "2026.03.03.2"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _step_status_from_flags(*, warnings: list[str], error: str | None = None) -> str:
    if error:
        return "ERROR"
    if warnings:
        return "WARN"
    return "PASS"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def run_quality_pipeline(
    *,
    db_path: Path = DEFAULT_DB,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    eligibility_out: Path = DEFAULT_ELIGIBILITY_OUT,
    eligibility_gates_out: Path = DEFAULT_ELIGIBILITY_GATES_OUT,
    context_out: Path = DEFAULT_CONTEXT_OUT,
    charts_out_dir: Path = DEFAULT_OUT_DIR,
    metrics_out: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    eligibility = compute_eligibility(db_path=db_path)
    append_threshold_hash_change_warning(eligibility_out, eligibility)
    _write_json(eligibility_out, eligibility)
    eligibility_warnings = list(eligibility.get("warnings", []))
    eligibility_errors = list(eligibility.get("errors", []))
    if eligibility.get("n_tickers", 0) == 0:
        eligibility_warnings.append("no_tickers_found")
    if int((eligibility.get("summary") or {}).get("HEALTHY", 0)) == 0:
        eligibility_warnings.append("zero_healthy_tickers")
    eligibility_warnings = _dedupe_preserve_order(eligibility_warnings)
    eligibility_errors = _dedupe_preserve_order(eligibility_errors)
    eligibility_error_flag = "eligibility_error" if eligibility_errors else None
    steps.append(
        {
            "name": "compute_ticker_eligibility",
            "status": _step_status_from_flags(warnings=eligibility_warnings, error=eligibility_error_flag),
            "warnings": eligibility_warnings,
            "errors": eligibility_errors,
            "output": str(eligibility_out),
        }
    )
    warnings.extend(eligibility_warnings)
    errors.extend(eligibility_errors)

    eligibility_gate = apply_eligibility_gates(
        eligibility_path=eligibility_out,
        output_path=eligibility_gates_out,
    )
    eligibility_gate_warnings = _dedupe_preserve_order(list(eligibility_gate.get("warnings", [])))
    eligibility_gate_errors = _dedupe_preserve_order(list(eligibility_gate.get("errors", [])))
    eligibility_gate_error_flag = "eligibility_gate_error" if eligibility_gate_errors else None
    steps.append(
        {
            "name": "apply_ticker_eligibility_gates",
            "status": _step_status_from_flags(
                warnings=eligibility_gate_warnings,
                error=eligibility_gate_error_flag,
            ),
            "warnings": eligibility_gate_warnings,
            "errors": eligibility_gate_errors,
            "lab_only_tickers": list(eligibility_gate.get("lab_only_tickers", [])),
            "gate_written": bool(eligibility_gate.get("gate_written", False)),
            "output": str(eligibility_gates_out),
        }
    )
    warnings.extend(eligibility_gate_warnings)
    errors.extend(eligibility_gate_errors)

    context = compute_context_quality(db_path=db_path)
    append_threshold_hash_change_warning(context_out, context)
    _write_json(context_out, context)
    context_warnings = list(context.get("warnings", []))
    context_errors: list[str] = []
    if isinstance(context.get("schema_used"), dict) and context.get("schema_used", {}).get("db_error"):
        context_errors.append(str(context["schema_used"]["db_error"]))
    for code in ("db_open_failed", "query_failed"):
        if code in context_warnings:
            context_errors.append(code)
    if context.get("partial_data"):
        context_warnings.append("partial_data")
    context_warnings = _dedupe_preserve_order(context_warnings)
    context_errors = _dedupe_preserve_order(context_errors)
    context_error_flag = "context_quality_error" if context_errors else None
    steps.append(
        {
            "name": "compute_context_quality",
            "status": _step_status_from_flags(warnings=context_warnings, error=context_error_flag),
            "warnings": context_warnings,
            "errors": context_errors,
            "output": str(context_out),
        }
    )
    warnings.extend(context_warnings)
    errors.extend(context_errors)

    sufficiency = run_data_sufficiency(db_path=db_path, audit_dir=audit_dir) if db_path.exists() else {
        "status": "DATA_ERROR",
        "sufficient": False,
        "recommendations": [f"DB not found: {db_path}"],
    }
    sufficiency_warnings = []
    sufficiency_error = None
    if sufficiency.get("status") == "DATA_ERROR":
        sufficiency_error = "data_error"
        errors.extend(sufficiency.get("recommendations", []) or [sufficiency_error])
    elif sufficiency.get("status") != "SUFFICIENT":
        sufficiency_warnings.append("insufficient_data")
    sufficiency_warnings = _dedupe_preserve_order(sufficiency_warnings)
    steps.append(
        {
            "name": "data_sufficiency_monitor",
            "status": _step_status_from_flags(warnings=sufficiency_warnings, error=sufficiency_error),
            "warnings": sufficiency_warnings,
            "output": None,
        }
    )
    warnings.extend(sufficiency_warnings)

    chart_result = generate_performance_artifacts(
        db_path=db_path,
        audit_dir=audit_dir,
        out_dir=charts_out_dir,
        eligibility_path=eligibility_out,
        context_quality_path=context_out,
        json_metrics_path=metrics_out,
        sufficiency=sufficiency,
        strict_mode=True,
    )
    chart_warnings = list(chart_result.get("warnings", []))
    chart_errors = list(chart_result.get("errors", []))
    chart_warnings = _dedupe_preserve_order(chart_warnings)
    chart_errors = _dedupe_preserve_order(chart_errors)
    chart_error_flag = "chart_generation_error" if chart_errors else None
    steps.append(
        {
            "name": "generate_performance_charts",
            "status": _step_status_from_flags(warnings=chart_warnings, error=chart_error_flag),
            "warnings": chart_warnings,
            "errors": chart_errors,
            "output": str(metrics_out),
        }
    )
    warnings.extend(chart_warnings)
    errors.extend(chart_errors)

    if errors:
        status = "ERROR"
    elif any(step["status"] == "WARN" for step in steps):
        status = "WARN"
    else:
        status = "PASS"

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "ok": status != "ERROR",
        "status": status,
        "pipeline_version": PIPELINE_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(max(0.0, time.perf_counter() - started_perf), 4),
        "db_path": str(db_path),
        "audit_dir": str(audit_dir),
        "thresholds": threshold_map(),
        "steps": steps,
        "artifacts": {
            "eligibility": str(eligibility_out),
            "eligibility_gates": str(eligibility_gates_out),
            "context_quality": str(context_out),
            "charts_out_dir": str(charts_out_dir),
            "metrics_summary": str(metrics_out),
        },
        "eligibility_summary": eligibility.get("summary", {}),
        "sufficiency": sufficiency,
        "warnings": sorted(set(warnings)),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only quality pipeline.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--eligibility-out", type=Path, default=DEFAULT_ELIGIBILITY_OUT)
    parser.add_argument("--eligibility-gates-out", type=Path, default=DEFAULT_ELIGIBILITY_GATES_OUT)
    parser.add_argument("--context-out", type=Path, default=DEFAULT_CONTEXT_OUT)
    parser.add_argument("--charts-out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    result = run_quality_pipeline(
        db_path=args.db,
        audit_dir=args.audit_dir,
        eligibility_out=args.eligibility_out,
        eligibility_gates_out=args.eligibility_gates_out,
        context_out=args.context_out,
        charts_out_dir=args.charts_out_dir,
        metrics_out=args.metrics_out,
    )

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Quality pipeline status: {result['status']}")
        for step in result["steps"]:
            print(f"  {step['name']}: {step['status']}")
    return 0 if result["status"] in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    sys.exit(main())
