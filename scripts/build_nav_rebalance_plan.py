#!/usr/bin/env python3
"""
build_nav_rebalance_plan.py
---------------------------

Read-only sidecar that turns ticker eligibility, sleeve summaries, and the
existing promotion/demotion plan into a shadow-first NAV rebalance artifact.

The script does not touch live routing config. It emits a versioned JSON plan
that explains:

- which tickers are healthy / weak / lab-only,
- which names are promoted / demoted / held,
- how the current NAV buckets would be split in shadow mode, and
- why live application is still blocked until the evidence gate is green.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import yaml

from risk.nav_allocator import BucketBudgets, apply_nav_allocator, load_budgets
from utils.evidence_io import write_versioned_json_artifact

ROOT_PATH = Path(__file__).resolve().parent.parent

DEFAULT_ELIGIBILITY_PATH = ROOT_PATH / "logs" / "ticker_eligibility.json"
DEFAULT_ELIGIBILITY_GATES_PATH = ROOT_PATH / "logs" / "ticker_eligibility_gates.json"
DEFAULT_SLEEVE_SUMMARY_PATH = ROOT_PATH / "logs" / "automation" / "sleeve_summary.json"
DEFAULT_SLEEVE_PLAN_PATH = ROOT_PATH / "logs" / "automation" / "sleeve_promotion_plan.json"
DEFAULT_METRICS_SUMMARY_PATH = ROOT_PATH / "visualizations" / "performance" / "metrics_summary.json"
DEFAULT_RISK_BUDGETS_PATH = ROOT_PATH / "config" / "risk_buckets.yml"
DEFAULT_OUTPUT_PATH = ROOT_PATH / "logs" / "automation" / "nav_rebalance_plan_latest.json"

TARGET_BUCKET_FOR_STATUS = {
    "HEALTHY": "ts_core",
    "WEAK": "cash_reserve",
    "LAB_ONLY": "research_only",
}

CURRENT_BUCKET_FALLBACK = {
    "safe": "safe",
    "core": "ts_core",
    "speculative": "speculative",
    "crypto": "ml_secondary",
    "other": "ml_secondary",
}

REASON_CODES = {
    "shadow_first": "shadow_first_default",
    "evidence_not_green": "evidence_not_green",
    "live_blocked": "live_apply_blocked",
    "healthy": "rolling_pf_wr_ok",
    "weak": "rolling_pf_wr_below_floor",
    "lab_only": "research_only",
    "promotion_plan": "promotion_plan_match",
    "demotion_plan": "demotion_plan_match",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("summary")
    if not rows:
        rows = payload.get("sleeves") or []
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            result.append(row)
    return result


def _load_bucket_map(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    barbell = raw.get("barbell") or {}
    mapping: dict[str, str] = {}
    for name in ("safe_bucket", "core_bucket", "speculative_bucket"):
        blk = barbell.get(name) or {}
        bucket = name.replace("_bucket", "")
        for sym in blk.get("symbols") or []:
            mapping[str(sym).upper()] = bucket
    return mapping


def _load_promotion_membership(plan_payload: dict[str, Any]) -> tuple[set[str], set[str]]:
    plan = plan_payload.get("plan") if isinstance(plan_payload, dict) else {}
    promotions = plan.get("promotions") if isinstance(plan, dict) else []
    demotions = plan.get("demotions") if isinstance(plan, dict) else []

    def _extract(items: Any) -> set[str]:
        out: set[str] = set()
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    ticker = str(item.get("ticker") or "").strip().upper()
                    if ticker:
                        out.add(ticker)
        return out

    return _extract(promotions), _extract(demotions)


def _score_from_eligibility(status: str, win_rate: float, profit_factor: float) -> float:
    pf = max(0.0, min(float(profit_factor), 3.0))
    wr = max(0.0, float(win_rate))
    base = wr * pf
    if status == "HEALTHY":
        return max(base, 0.1)
    if status == "WEAK":
        return max(base * 0.15, 0.01)
    return 0.0


def _current_bucket_for(
    ticker: str,
    summary_by_ticker: dict[str, dict[str, Any]],
    barbell_map: dict[str, str],
) -> str:
    row = summary_by_ticker.get(ticker.upper()) or {}
    current = str(row.get("sleeve") or row.get("bucket") or "").strip().lower()
    if current:
        return CURRENT_BUCKET_FALLBACK.get(current, current)
    return CURRENT_BUCKET_FALLBACK.get(barbell_map.get(ticker.upper(), "other"), "ml_secondary")


def _validate_plan(payload: dict[str, Any]) -> bool | tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    required = {"meta", "rollout", "bucket_allocations", "targets"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        return False, f"missing_required_keys:{','.join(missing)}"
    return True, "ok"


def build_nav_rebalance_plan(
    *,
    eligibility_path: Path = DEFAULT_ELIGIBILITY_PATH,
    eligibility_gates_path: Path = DEFAULT_ELIGIBILITY_GATES_PATH,
    sleeve_summary_path: Path = DEFAULT_SLEEVE_SUMMARY_PATH,
    sleeve_plan_path: Path = DEFAULT_SLEEVE_PLAN_PATH,
    metrics_summary_path: Path = DEFAULT_METRICS_SUMMARY_PATH,
    risk_buckets_path: Path = DEFAULT_RISK_BUDGETS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    eligibility = _load_json(Path(eligibility_path))
    eligibility_gates = _load_json(Path(eligibility_gates_path))
    sleeve_summary = _load_json(Path(sleeve_summary_path))
    sleeve_plan = _load_json(Path(sleeve_plan_path))
    metrics_summary = _load_json(Path(metrics_summary_path))

    summary_rows = _load_summary_rows(sleeve_summary)
    summary_by_ticker: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker and ticker not in summary_by_ticker:
            summary_by_ticker[ticker] = row

    bucket_map = _load_bucket_map(ROOT_PATH / "config" / "barbell.yml")
    promote_set, demote_set = _load_promotion_membership(sleeve_plan)

    ticker_map = eligibility.get("tickers") if isinstance(eligibility.get("tickers"), dict) else {}
    all_tickers = sorted(
        {
            *(str(t).upper() for t in ticker_map.keys()),
            *summary_by_ticker.keys(),
            *promote_set,
            *demote_set,
        }
    )

    budgets_raw = load_budgets(Path(risk_buckets_path))
    shadow_budgets = BucketBudgets(
        enabled=True,
        base_nav_frac=dict(budgets_raw.base_nav_frac),
        min_nav_frac=dict(budgets_raw.min_nav_frac),
        max_nav_frac=dict(budgets_raw.max_nav_frac),
    )

    target_rows: list[dict[str, Any]] = []
    bucket_inputs: dict[str, dict[str, Any]] = defaultdict(lambda: {"weights": {}, "symbols": []})
    weak_names: list[str] = []
    healthy_names: list[str] = []
    lab_only_names: list[str] = []

    for ticker in all_tickers:
        info = ticker_map.get(ticker, {}) if isinstance(ticker_map, dict) else {}
        status = str(info.get("status") or "LAB_ONLY").upper()
        status = status if status in {"HEALTHY", "WEAK", "LAB_ONLY"} else "LAB_ONLY"
        current_bucket = _current_bucket_for(ticker, summary_by_ticker, bucket_map)
        target_bucket = TARGET_BUCKET_FOR_STATUS[status]
        current_score = _score_from_eligibility(
            status,
            float(info.get("win_rate") or 0.0),
            float(info.get("profit_factor") or 0.0),
        )

        reason_codes: list[str] = []
        eligibility_reasons = info.get("reasons")
        if isinstance(eligibility_reasons, list):
            reason_codes.extend(str(reason) for reason in eligibility_reasons if reason)
        if status == "HEALTHY":
            reason_codes.append(REASON_CODES["healthy"])
            healthy_names.append(ticker)
        elif status == "WEAK":
            reason_codes.append(REASON_CODES["weak"])
            weak_names.append(ticker)
        else:
            reason_codes.append(REASON_CODES["lab_only"])
            lab_only_names.append(ticker)

        if ticker in promote_set:
            reason_codes.append(REASON_CODES["promotion_plan"])
        if ticker in demote_set:
            reason_codes.append(REASON_CODES["demotion_plan"])

        if status == "HEALTHY" and current_bucket != "ts_core":
            action = "PROMOTE"
        elif status == "WEAK":
            action = "DEMOTE"
        elif status == "LAB_ONLY":
            action = "RESEARCH_ONLY"
        else:
            action = "HOLD"

        if status == "HEALTHY":
            bucket_inputs["ts_core"]["weights"][ticker] = current_score
            bucket_inputs["ts_core"]["symbols"].append(ticker)

        target_rows.append(
            {
                "ticker": ticker,
                "status": status,
                "current_bucket": current_bucket,
                "target_bucket": target_bucket,
                "action": action,
                "target_nav_frac": 0.0,
                "score": round(current_score, 6),
                "reason_codes": sorted(set(reason_codes)),
            }
        )

    active_weights = bucket_inputs.get("ts_core", {}).get("weights", {})
    active_allocations = apply_nav_allocator(
        active_weights,
        {ticker: "ts_core" for ticker in active_weights},
        shadow_budgets,
        nav=1.0,
    )

    for row in target_rows:
        if row["status"] == "HEALTHY":
            row["target_nav_frac"] = round(float(active_allocations.get(row["ticker"], 0.0)), 8)
        else:
            row["target_nav_frac"] = 0.0

    bucket_allocations: list[dict[str, Any]] = []
    for bucket in ("safe", "ts_core", "speculative", "ml_secondary", "llm_fallback"):
        configured = float(shadow_budgets.base_nav_frac.get(bucket, 0.0) or 0.0)
        min_frac = float(shadow_budgets.min_nav_frac.get(bucket, 0.0) or 0.0)
        max_frac = float(shadow_budgets.max_nav_frac.get(bucket, 1.0) or 1.0)
        configured = max(configured, min_frac)
        configured = min(configured, max_frac)
        symbols = [row for row in target_rows if row["target_bucket"] == bucket]
        allocated = sum(float(row["target_nav_frac"]) for row in symbols)
        reserve = max(0.0, configured - allocated)
        bucket_allocations.append(
            {
                "bucket": bucket,
                "configured_nav_frac": round(configured, 8),
                "allocated_nav_frac": round(allocated, 8),
                "reserve_nav_frac": round(reserve, 8),
                "symbols": [row["ticker"] for row in symbols],
            }
        )

    evidence_status = str(metrics_summary.get("sufficiency_status") or metrics_summary.get("status") or "UNKNOWN").upper()
    evidence_warnings = list(metrics_summary.get("warnings") or [])
    evidence_warnings.extend(str(item) for item in (eligibility.get("warnings") or []) if item)
    evidence_gate_allowed = evidence_status in {"SUFFICIENT", "PASS", "GREEN"}
    if eligibility_gates.get("warnings"):
        evidence_warnings.extend(str(item) for item in eligibility_gates.get("warnings") or [])
    gate_lift_candidate = evidence_gate_allowed and not evidence_warnings

    rollout = {
        "mode": "shadow",
        "shadow_first": True,
        "live_apply_allowed": False,
        "gate_lift_candidate": bool(gate_lift_candidate),
        "gate_lift_ready": False,
        "required_green_cycles": 2,
        "live_apply_blockers": [
            REASON_CODES["shadow_first"],
        ]
        + ([] if gate_lift_candidate else [REASON_CODES["evidence_not_green"]]),
        "evidence_status": evidence_status,
        "evidence_warnings": sorted({str(item) for item in evidence_warnings if item}),
        "eligibility_summary": eligibility.get("summary") if isinstance(eligibility.get("summary"), dict) else {},
        "eligibility_gate_summary": eligibility_gates.get("summary") if isinstance(eligibility_gates.get("summary"), dict) else {},
    }

    payload: dict[str, Any] = {
        "meta": {
            "generated_utc": metrics_summary.get("generated_utc")
            or eligibility.get("generated_utc")
            or eligibility_gates.get("generated_utc"),
            "source_paths": {
                "eligibility": str(eligibility_path),
                "eligibility_gates": str(eligibility_gates_path),
                "sleeve_summary": str(sleeve_summary_path),
                "sleeve_promotion_plan": str(sleeve_plan_path),
                "metrics_summary": str(metrics_summary_path),
                "risk_buckets": str(risk_buckets_path),
            },
            "allocator_feature_flag_enabled": bool(budgets_raw.enabled),
            "planner": "build_nav_rebalance_plan",
            "rollout_policy": "shadow_first",
        },
        "rollout": rollout,
        "bucket_allocations": bucket_allocations,
        "targets": sorted(target_rows, key=lambda row: (row["status"], row["ticker"])),
        "evidence": {
            "eligibility": eligibility,
            "eligibility_gates": eligibility_gates,
            "sleeve_summary": sleeve_summary,
            "sleeve_promotion_plan": sleeve_plan,
            "metrics_summary": metrics_summary,
        },
        "summary": {
            "healthy": sorted(healthy_names),
            "weak": sorted(weak_names),
            "lab_only": sorted(lab_only_names),
            "promotions": sorted(promote_set),
            "demotions": sorted(demote_set),
            "total_targets": len(target_rows),
            "allocated_symbol_nav_frac": round(sum(float(row["target_nav_frac"]) for row in target_rows), 8),
        },
    }
    payload["summary"]["unallocated_bucket_nav_frac"] = round(
        sum(float(item["reserve_nav_frac"]) for item in bucket_allocations),
        8,
    )

    write_result = write_versioned_json_artifact(
        latest_path=Path(output_path),
        payload=payload,
        archive_name="nav_rebalance_plan",
        validate_fn=_validate_plan,
    )
    payload["artifact"] = write_result
    payload["output"] = str(output_path)
    return payload


@click.command()
@click.option("--eligibility-path", default=str(DEFAULT_ELIGIBILITY_PATH), show_default=True)
@click.option("--eligibility-gates-path", default=str(DEFAULT_ELIGIBILITY_GATES_PATH), show_default=True)
@click.option("--sleeve-summary-path", default=str(DEFAULT_SLEEVE_SUMMARY_PATH), show_default=True)
@click.option("--sleeve-plan-path", default=str(DEFAULT_SLEEVE_PLAN_PATH), show_default=True)
@click.option("--metrics-summary-path", default=str(DEFAULT_METRICS_SUMMARY_PATH), show_default=True)
@click.option("--risk-buckets-path", default=str(DEFAULT_RISK_BUDGETS_PATH), show_default=True)
@click.option("--output", default=str(DEFAULT_OUTPUT_PATH), show_default=True)
@click.option("--json", "emit_json", is_flag=True, default=False, help="Print the plan as JSON.")
def main(
    eligibility_path: str,
    eligibility_gates_path: str,
    sleeve_summary_path: str,
    sleeve_plan_path: str,
    metrics_summary_path: str,
    risk_buckets_path: str,
    output: str,
    emit_json: bool,
) -> None:
    result = build_nav_rebalance_plan(
        eligibility_path=Path(eligibility_path),
        eligibility_gates_path=Path(eligibility_gates_path),
        sleeve_summary_path=Path(sleeve_summary_path),
        sleeve_plan_path=Path(sleeve_plan_path),
        metrics_summary_path=Path(metrics_summary_path),
        risk_buckets_path=Path(risk_buckets_path),
        output_path=Path(output),
    )

    if emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"NAV rebalance plan written to {output}")
        print(
            "  mode={mode} live_apply_allowed={live_apply_allowed} gate_lift_candidate={candidate}".format(
                mode=result["rollout"]["mode"],
                live_apply_allowed=result["rollout"]["live_apply_allowed"],
                candidate=result["rollout"]["gate_lift_candidate"],
            )
        )
        print(
            "  HEALTHY={healthy} WEAK={weak} LAB_ONLY={lab_only}".format(
                healthy=len(result["summary"]["healthy"]),
                weak=len(result["summary"]["weak"]),
                lab_only=len(result["summary"]["lab_only"]),
            )
        )
    return None


if __name__ == "__main__":
    main()
