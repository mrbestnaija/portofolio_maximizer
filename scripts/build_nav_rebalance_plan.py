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
DEFAULT_CANONICAL_SNAPSHOT_PATH = ROOT_PATH / "logs" / "canonical_snapshot_latest.json"
DEFAULT_METRICS_SUMMARY_PATH = ROOT_PATH / "visualizations" / "performance" / "metrics_summary.json"
DEFAULT_RISK_BUDGETS_PATH = ROOT_PATH / "config" / "risk_buckets.yml"
DEFAULT_OUTPUT_PATH = ROOT_PATH / "logs" / "automation" / "nav_rebalance_plan_latest.json"
DEFAULT_HISTORY_ROOT = ROOT_PATH / "logs" / "automation" / "nav_rebalance_plan_history"

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


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _build_evidence_contract(
    *,
    canonical_snapshot: dict[str, Any],
    metrics_summary: dict[str, Any],
    evidence_status: str,
    evidence_warnings: list[str],
    evidence_gate_allowed: bool,
    gate_lift_candidate: bool,
) -> dict[str, Any]:
    canonical_summary = canonical_snapshot.get("summary") if isinstance(canonical_snapshot.get("summary"), dict) else {}
    source_contract = canonical_snapshot.get("source_contract") if isinstance(canonical_snapshot.get("source_contract"), dict) else {}
    canonical_source = source_contract.get("canonical") if isinstance(source_contract.get("canonical"), dict) else {}
    source_kind = "canonical_closed_trades" if canonical_snapshot else "ui_only_metrics_summary"
    data_source = str(canonical_source.get("closed_pnl") or "production_closed_trades")
    coverage_ratio = _as_float(metrics_summary.get("coverage_ratio"))
    missing_metrics_fraction = _as_float(metrics_summary.get("missing_metrics_fraction"))
    imputed_fraction = _as_float(metrics_summary.get("imputed_fraction"))
    padding_fraction = _as_float(metrics_summary.get("padding_fraction"))
    fallback_class = str(metrics_summary.get("fallback_class") or "")
    if not fallback_class and evidence_warnings:
        fallback_class = str(evidence_warnings[0])
    if not fallback_class:
        fallback_class = "none"
    return {
        "evidence_status": evidence_status,
        "evidence_gate_allowed": bool(evidence_gate_allowed),
        "gate_lift_candidate": bool(gate_lift_candidate),
        "oos_metrics_available": coverage_ratio is not None or bool(metrics_summary),
        "oos_source_kind": "GENUINE_OOS" if canonical_snapshot else "MISSING",
        "source_kind": source_kind,
        "data_source": data_source,
        "provenance_trusted": bool(source_contract),
        "coverage_ratio": coverage_ratio,
        "missing_metrics_fraction": missing_metrics_fraction,
        "imputed_fraction": imputed_fraction,
        "padding_fraction": padding_fraction,
        "fallback_class": fallback_class,
        "canonical_posture": str((canonical_snapshot.get("gate") or {}).get("posture") or "").upper(),
        "canonical_unattended_ready": bool(canonical_summary.get("unattended_ready")),
        "canonical_unattended_gate": str(canonical_summary.get("unattended_gate") or "").upper(),
        "canonical_roi_ann_pct": _as_float(canonical_summary.get("ann_roi_pct")),
    }


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


def _green_cycle_streak(history_root: Path) -> tuple[int, int]:
    if not history_root.exists():
        return 0, 0
    plan_files = sorted(
        p for p in history_root.glob("nav_rebalance_plan_*.json") if p.is_file()
    )
    if not plan_files:
        return 0, 0

    streak = 0
    inspected = 0
    for path in reversed(plan_files):
        payload = _load_json(path)
        rollout = payload.get("rollout") if isinstance(payload.get("rollout"), dict) else {}
        candidate = bool(rollout.get("gate_lift_candidate"))
        warnings = rollout.get("evidence_warnings") if isinstance(rollout.get("evidence_warnings"), list) else []
        inspected += 1
        if candidate and not warnings:
            streak += 1
        else:
            break
    return streak, inspected


def build_nav_rebalance_plan(
    *,
    eligibility_path: Path = DEFAULT_ELIGIBILITY_PATH,
    eligibility_gates_path: Path = DEFAULT_ELIGIBILITY_GATES_PATH,
    sleeve_summary_path: Path = DEFAULT_SLEEVE_SUMMARY_PATH,
    sleeve_plan_path: Path = DEFAULT_SLEEVE_PLAN_PATH,
    canonical_snapshot_path: Path = DEFAULT_CANONICAL_SNAPSHOT_PATH,
    metrics_summary_path: Path = DEFAULT_METRICS_SUMMARY_PATH,
    risk_buckets_path: Path = DEFAULT_RISK_BUDGETS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    history_root = Path(output_path).parent / "nav_rebalance_plan_history"
    eligibility = _load_json(Path(eligibility_path))
    eligibility_gates = _load_json(Path(eligibility_gates_path))
    sleeve_summary = _load_json(Path(sleeve_summary_path))
    sleeve_plan = _load_json(Path(sleeve_plan_path))
    canonical_snapshot = _load_json(Path(canonical_snapshot_path))
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

    canonical_summary = canonical_snapshot.get("summary") if isinstance(canonical_snapshot.get("summary"), dict) else {}
    canonical_gate = canonical_snapshot.get("gate") if isinstance(canonical_snapshot.get("gate"), dict) else {}
    if canonical_snapshot:
        evidence_status = str(canonical_summary.get("unattended_gate") or canonical_summary.get("status") or "UNKNOWN").upper()
        evidence_warnings = []
        if canonical_summary.get("ann_roi_pct") is None:
            evidence_warnings.append("canonical_roi_missing")
        try:
            gap = canonical_summary.get("gap_to_hurdle_pp")
            if gap is not None and float(gap) > 0:
                evidence_warnings.append("ngn_hurdle_gap_positive")
        except Exception:
            evidence_warnings.append("ngn_hurdle_gap_unparseable")
        if not bool(canonical_summary.get("unattended_ready")):
            evidence_warnings.append("unattended_gate_not_ready")
        if str(canonical_gate.get("posture") or "").upper() == "WARMUP_COVERED_PASS":
            evidence_warnings.append("warmup_exemption_active")
        source_contract = canonical_snapshot.get("source_contract") if isinstance(canonical_snapshot.get("source_contract"), dict) else {}
        if not source_contract:
            evidence_warnings.append("canonical_source_contract_missing")
        evidence_gate_allowed = bool(canonical_summary.get("unattended_ready")) and "warmup_exemption_active" not in evidence_warnings
    else:
        evidence_status = str(metrics_summary.get("sufficiency_status") or metrics_summary.get("status") or "UNKNOWN").upper()
        evidence_warnings = list(metrics_summary.get("warnings") or [])
        evidence_gate_allowed = evidence_status in {"SUFFICIENT", "PASS", "GREEN"}

    evidence_warnings.extend(str(item) for item in (eligibility.get("warnings") or []) if item)
    if eligibility_gates.get("warnings"):
        evidence_warnings.extend(str(item) for item in eligibility_gates.get("warnings") or [])
    gate_lift_candidate = evidence_gate_allowed and not evidence_warnings
    evidence_contract = _build_evidence_contract(
        canonical_snapshot=canonical_snapshot,
        metrics_summary=metrics_summary,
        evidence_status=evidence_status,
        evidence_warnings=list(sorted({str(item) for item in evidence_warnings if item})),
        evidence_gate_allowed=evidence_gate_allowed,
        gate_lift_candidate=gate_lift_candidate,
    )
    prior_green_streak, history_files_considered = _green_cycle_streak(history_root)
    current_green_streak = prior_green_streak + 1 if gate_lift_candidate else 0
    live_apply_allowed = bool(gate_lift_candidate and current_green_streak >= 2)
    rollout_mode = "live" if live_apply_allowed else "shadow"
    rollout_blockers: list[str] = []
    if not live_apply_allowed:
        rollout_blockers.append(REASON_CODES["shadow_first"])
    if not gate_lift_candidate:
        rollout_blockers.append(REASON_CODES["evidence_not_green"])
    if gate_lift_candidate and current_green_streak < 2:
        rollout_blockers.append("gate_lift_waiting_for_2_green_cycles")

    for row in target_rows:
        info = ticker_map.get(str(row.get("ticker") or "").upper(), {}) if isinstance(ticker_map, dict) else {}
        row["metrics"] = {
            "ticker": str(row.get("ticker") or "").upper(),
            "status": str(row.get("status") or "LAB_ONLY").upper(),
            "n_trades": int(info.get("n_trades") or 0),
            "win_rate": _as_float(info.get("win_rate")) or 0.0,
            "profit_factor": _as_float(info.get("profit_factor")) or 0.0,
            "total_pnl": _as_float(info.get("total_pnl")) or 0.0,
            "coverage_ratio": evidence_contract["coverage_ratio"],
            "missing_metrics_fraction": evidence_contract["missing_metrics_fraction"],
            "imputed_fraction": evidence_contract["imputed_fraction"],
            "padding_fraction": evidence_contract["padding_fraction"],
            "oos_source_kind": evidence_contract["oos_source_kind"],
            "source_kind": evidence_contract["source_kind"],
            "data_source": evidence_contract["data_source"],
            "provenance_trusted": evidence_contract["provenance_trusted"],
            "fallback_class": evidence_contract["fallback_class"],
            "oos_metrics_available": evidence_contract["oos_metrics_available"],
            "gate_lift_candidate": evidence_contract["gate_lift_candidate"],
            "evidence_status": evidence_contract["evidence_status"],
            "canonical_posture": evidence_contract["canonical_posture"],
        }

    rollout = {
        "mode": rollout_mode,
        "shadow_first": not live_apply_allowed,
        "live_apply_allowed": live_apply_allowed,
        "gate_lift_candidate": bool(gate_lift_candidate),
        "gate_lift_ready": live_apply_allowed,
        "required_green_cycles": 2,
        "live_apply_blockers": rollout_blockers,
        "evidence_status": evidence_status,
        "evidence_warnings": sorted({str(item) for item in evidence_warnings if item}),
        "eligibility_summary": eligibility.get("summary") if isinstance(eligibility.get("summary"), dict) else {},
        "eligibility_gate_summary": eligibility_gates.get("summary") if isinstance(eligibility_gates.get("summary"), dict) else {},
        "evidence_contract": evidence_contract,
        "gate_lift_state": {
            "history_root": str(history_root),
            "prior_consecutive_green_cycles": prior_green_streak,
            "current_consecutive_green_cycles": current_green_streak,
            "required_green_cycles": 2,
            "history_files_considered": history_files_considered,
            "current_cycle_green": bool(gate_lift_candidate),
            "lift_ready": live_apply_allowed,
        },
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
                "canonical_snapshot": str(canonical_snapshot_path),
                "ui_only_metrics_summary": str(metrics_summary_path),
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
            "canonical_snapshot": canonical_snapshot,
            "ui_only_metrics_summary": metrics_summary,
        },
        "evidence_contract": evidence_contract,
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
        archive_root=history_root,
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
@click.option("--canonical-snapshot-path", default=str(DEFAULT_CANONICAL_SNAPSHOT_PATH), show_default=True)
@click.option("--metrics-summary-path", default=str(DEFAULT_METRICS_SUMMARY_PATH), show_default=True)
@click.option("--risk-buckets-path", default=str(DEFAULT_RISK_BUDGETS_PATH), show_default=True)
@click.option("--output", default=str(DEFAULT_OUTPUT_PATH), show_default=True)
@click.option("--json", "emit_json", is_flag=True, default=False, help="Print the plan as JSON.")
def main(
    eligibility_path: str,
    eligibility_gates_path: str,
    sleeve_summary_path: str,
    sleeve_plan_path: str,
    canonical_snapshot_path: str,
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
        canonical_snapshot_path=Path(canonical_snapshot_path),
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
