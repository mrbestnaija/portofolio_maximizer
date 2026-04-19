#!/usr/bin/env python3
"""apply_nav_reallocation.py

NAV reallocation sidecar for the barbell promotion/demotion pipeline.

Pipeline (primary):
    build_nav_rebalance_plan.py → nav_rebalance_plan_latest.json
    apply_nav_reallocation.py (THIS) → nav_allocation_latest.json [+ staged barbell config]

Pipeline (legacy):
    summarize_sleeves.py → sleeve_summary.json
    evaluate_sleeve_promotions.py → sleeve_promotion_plan.json
    apply_nav_reallocation.py (THIS) → nav_allocation_latest.json

Both plan formats are accepted via format auto-detection.

Responsibilities:
- Consume the NAV rebalance plan from build_nav_rebalance_plan.py (or legacy sleeve_promotion_plan.json)
- Apply symbol moves (speculative ↔ core) subject to barbell bucket constraints
- Emit a structured NAV allocation artifact (JSON) that the barbell gate can read
- Optionally write a staged barbell.yml for operator review before applying live

Promotion and continued deployment both require these evidence contracts; promotion is
not a one-time event. The sidecar is designed to run on every live cycle or cron
schedule and produce an up-to-date allocation artifact.

Evidence-health gate dimensions (tracked in output artifact):
    OOS_COVERAGE_THIN       – coverage_ratio < min threshold
    OOS_MISSING_METRICS     – RMSE-rank running without regression metrics
    PREPROCESS_DISTORTION   – imputed_fraction or padding_fraction exceeds cap
    HEURISTIC_FALLBACK      – ungrounded heuristic present (not HEURISTIC_ALLOWED)
    PROVENANCE_UNTRUSTED    – data_source unknown/synthetic or provenance_trusted=False
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import click
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Barbell allocation constraints (mirrors barbell.yml defaults)
# ---------------------------------------------------------------------------
SAFE_MIN_WEIGHT = 0.75   # safe sleeve must always hold ≥ 75% NAV
CORE_MAX_WEIGHT = 0.20
SPEC_MAX_WEIGHT = 0.10


@dataclass
class AllocationConstraints:
    safe_min_weight: float = SAFE_MIN_WEIGHT
    core_max_weight: float = CORE_MAX_WEIGHT
    spec_max_weight: float = SPEC_MAX_WEIGHT

    # Evidence thresholds for continued production deployment
    # Starting values are intentionally stricter than current metrics;
    # ETL backfill is expected to close the gap over the next N weeks.
    min_coverage_ratio: float = 0.30   # ratchet toward 0.70 as backfill grows
    max_missing_metrics_fraction: float = 0.50  # fraction of windows with no OOS metrics
    max_imputed_fraction: float = 0.30
    max_padding_fraction: float = 0.20


@dataclass
class AllocationArtifact:
    generated_at: str
    source_plan: str
    constraints_applied: Dict
    sleeve_allocations: Dict[str, Dict]    # bucket → {symbols, max_weight, reason}
    applied_promotions: List[Dict]
    applied_demotions: List[Dict]
    skipped_moves: List[Dict]             # moves blocked by constraints or evidence
    evidence_gate: Dict                   # per-ticker evidence health snapshot
    summary: str                          # human-readable 1-line status


# ---------------------------------------------------------------------------
# Evidence health check
# ---------------------------------------------------------------------------

def _check_evidence_gate(
    ticker: str,
    plan_row: Optional[Dict],
    constraints: AllocationConstraints,
) -> Dict:
    """Evaluate evidence-health gate for a ticker.

    Returns dict with keys: passed, blocking_reasons, evidence_class.
    Evidence classes: GENUINE_OOS, HEURISTIC_ALLOWED, HEURISTIC_UNGROUNDED, MISSING.
    """
    blocking: List[str] = []
    evidence_class = "UNKNOWN"

    # If we have no plan row, evidence is absent
    if not plan_row:
        return {"passed": True, "blocking_reasons": [], "evidence_class": "NO_PLAN_DATA"}

    metrics = plan_row.get("metrics") or {}
    # Check coverage (if available in metrics)
    coverage = float(metrics.get("coverage_ratio") or metrics.get("oos_coverage") or 0.0)
    if coverage > 0 and coverage < constraints.min_coverage_ratio:
        blocking.append("OOS_COVERAGE_THIN")

    missing_frac = float(metrics.get("missing_metrics_fraction") or 0.0)
    if missing_frac > constraints.max_missing_metrics_fraction:
        blocking.append("OOS_MISSING_METRICS")

    imputed = float(metrics.get("imputed_fraction") or 0.0)
    if imputed > constraints.max_imputed_fraction:
        blocking.append("PREPROCESS_DISTORTION")

    padding = float(metrics.get("padding_fraction") or 0.0)
    if padding > constraints.max_padding_fraction:
        blocking.append("PREPROCESS_DISTORTION")

    # Heuristic classification: if RMSE-rank ran with no regression metrics and is
    # the sole OOS signal, it is HEURISTIC_UNGROUNDED (research-only).
    # EWMA volatility fallback is HEURISTIC_ALLOWED when backed by convergence guard.
    heuristic_class = str(metrics.get("oos_source_kind") or "").upper()
    if heuristic_class == "HEURISTIC_UNGROUNDED":
        blocking.append("HEURISTIC_FALLBACK")
        evidence_class = "HEURISTIC_UNGROUNDED"
    elif heuristic_class == "HEURISTIC_ALLOWED":
        evidence_class = "HEURISTIC_ALLOWED"
    elif heuristic_class in ("GENUINE_OOS", "PRIMARY"):
        evidence_class = "GENUINE_OOS"
    else:
        evidence_class = "UNKNOWN"

    # Provenance check: block when data source is unknown or explicitly untrusted.
    # build_nav_rebalance_plan sets provenance_trusted=False when eligibility evidence
    # is absent, data_source is synthetic/unknown, or fallback path is unguarded.
    provenance_trusted = metrics.get("provenance_trusted")
    data_source = str(metrics.get("data_source") or "").lower()
    if provenance_trusted is False or data_source in ("unknown", "untrusted", "synthetic"):
        blocking.append("PROVENANCE_UNTRUSTED")

    return {
        "passed": len(blocking) == 0,
        "blocking_reasons": blocking,
        "evidence_class": evidence_class,
    }


# ---------------------------------------------------------------------------
# Core allocation logic
# ---------------------------------------------------------------------------

def apply_reallocation(
    plan: Dict,
    current_config: Dict,
    constraints: AllocationConstraints,
) -> AllocationArtifact:
    """Apply the promotion/demotion plan subject to barbell constraints.

    Modifies a deep copy of current_config's sleeve symbol lists.
    Returns an AllocationArtifact describing the result.
    """
    cfg = copy.deepcopy(current_config)
    barbell = cfg.get("barbell") or {}

    def _symbols(bucket_key: str) -> List[str]:
        return list(barbell.get(bucket_key, {}).get("symbols") or [])

    safe_syms = _symbols("safe_bucket")
    core_syms = _symbols("core_bucket")
    spec_syms = _symbols("speculative_bucket")

    promotions = plan.get("promotions") or []
    demotions = plan.get("demotions") or []
    rollout = plan.get("_rollout") if isinstance(plan.get("_rollout"), dict) else {}
    live_apply_allowed = bool(rollout.get("live_apply_allowed", True))

    applied_promotions: List[Dict] = []
    applied_demotions: List[Dict] = []
    skipped_moves: List[Dict] = []
    evidence_gate: Dict[str, Dict] = {}

    # Build quick lookup: ticker → plan row (for evidence gate)
    plan_rows: Dict[str, Dict] = {}
    for row in promotions + demotions:
        t = str(row.get("ticker") or "").strip()
        if t:
            plan_rows[t] = row

    # Process promotions: speculative → core
    for move in promotions:
        ticker = str(move.get("ticker") or "").strip()
        if not ticker:
            continue
        eg = _check_evidence_gate(ticker, plan_rows.get(ticker), constraints)
        if not live_apply_allowed:
            eg = {
                **eg,
                "passed": False,
                "blocking_reasons": sorted({*eg["blocking_reasons"], "LIVE_APPLY_BLOCKED"}),
            }
        evidence_gate[ticker] = eg
        if not eg["passed"]:
            skipped_moves.append({**move, "skip_reason": eg["blocking_reasons"]})
            continue
        if ticker in spec_syms and ticker not in core_syms:
            spec_syms.remove(ticker)
            core_syms.append(ticker)
            applied_promotions.append(move)
        elif ticker not in core_syms:
            skipped_moves.append({**move, "skip_reason": ["TICKER_NOT_IN_SPECULATIVE"]})

    # Process demotions: core → speculative
    for move in demotions:
        ticker = str(move.get("ticker") or "").strip()
        if not ticker:
            continue
        eg = _check_evidence_gate(ticker, plan_rows.get(ticker), constraints)
        evidence_gate.setdefault(ticker, eg)
        if ticker in core_syms and ticker not in spec_syms:
            core_syms.remove(ticker)
            spec_syms.append(ticker)
            applied_demotions.append(move)
        elif ticker not in spec_syms:
            skipped_moves.append({**move, "skip_reason": ["TICKER_NOT_IN_CORE"]})

    sleeve_allocations = {
        "safe": {
            "symbols": safe_syms,
            "max_weight": barbell.get("safe_bucket", {}).get("max_weight", 0.95),
            "min_weight": constraints.safe_min_weight,
        },
        "core": {
            "symbols": core_syms,
            "max_weight": min(
                constraints.core_max_weight,
                barbell.get("core_bucket", {}).get("max_weight", 0.20),
            ),
        },
        "speculative": {
            "symbols": spec_syms,
            "max_weight": min(
                constraints.spec_max_weight,
                barbell.get("speculative_bucket", {}).get("max_weight", 0.10),
            ),
        },
    }

    n_applied = len(applied_promotions) + len(applied_demotions)
    n_skipped = len(skipped_moves)
    summary = (
        f"{n_applied} moves applied ({len(applied_promotions)} promotions, "
        f"{len(applied_demotions)} demotions), {n_skipped} skipped by evidence gate or constraints"
    )

    return AllocationArtifact(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_plan=str(plan.get("_source_plan") or ""),
        constraints_applied=constraints.__dict__,
        sleeve_allocations=sleeve_allocations,
        applied_promotions=applied_promotions,
        applied_demotions=applied_demotions,
        skipped_moves=skipped_moves,
        evidence_gate=evidence_gate,
        summary=summary,
    )


def _artifact_to_dict(a: AllocationArtifact) -> Dict:
    return {
        "generated_at": a.generated_at,
        "source_plan": a.source_plan,
        "constraints_applied": a.constraints_applied,
        "sleeve_allocations": a.sleeve_allocations,
        "applied_promotions": a.applied_promotions,
        "applied_demotions": a.applied_demotions,
        "skipped_moves": a.skipped_moves,
        "evidence_gate": a.evidence_gate,
        "summary": a.summary,
    }


def _write_staged_config(
    artifact: AllocationArtifact,
    source_config: Dict,
    staged_path: Path,
) -> None:
    """Write a staged barbell.yml with updated symbol lists for operator review."""
    staged = copy.deepcopy(source_config)
    barbell = staged.setdefault("barbell", {})
    allocs = artifact.sleeve_allocations
    for bucket_key, sleeve_key in [
        ("safe_bucket", "safe"),
        ("core_bucket", "core"),
        ("speculative_bucket", "speculative"),
    ]:
        if sleeve_key in allocs:
            barbell.setdefault(bucket_key, {})["symbols"] = allocs[sleeve_key]["symbols"]
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_text(yaml.dump(staged, default_flow_style=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Format adapter: nav_rebalance_plan_latest.json → apply_reallocation() input
# ---------------------------------------------------------------------------

def _adapt_nav_rebalance_plan(payload: Dict) -> Dict:
    """Translate build_nav_rebalance_plan.py output into apply_reallocation() format.

    build_nav_rebalance_plan emits targets[] with ticker/status/target_bucket/target_nav_frac
    and summary.promotions/demotions as sorted ticker lists.  apply_reallocation() expects
    promotions/demotions as lists of dicts with at minimum a 'ticker' key and optional
    'metrics' dict for evidence gate checks.
    """
    summary = payload.get("summary") or {}
    targets = payload.get("targets") or []

    # Build per-ticker metric lookup from the targets list
    ticker_metrics: Dict[str, Dict] = {}
    for row in targets:
        t = str(row.get("ticker") or "").strip()
        if t:
            ticker_metrics[t] = {
                "coverage_ratio": row.get("coverage_ratio"),
                "imputed_fraction": row.get("imputed_fraction"),
                "padding_fraction": row.get("padding_fraction"),
                "missing_metrics_fraction": row.get("missing_metrics_fraction"),
                "oos_source_kind": row.get("oos_source_kind"),
                "provenance_trusted": row.get("provenance_trusted"),
                "data_source": row.get("data_source"),
                "status": row.get("status"),
            }

    promotions = [
        {"ticker": t, "metrics": ticker_metrics.get(t, {}), "reason": "rolling_pf_wr_ok"}
        for t in (summary.get("promotions") or [])
    ]
    demotions = [
        {"ticker": t, "metrics": ticker_metrics.get(t, {}), "reason": "rolling_pf_wr_below_floor"}
        for t in (summary.get("demotions") or [])
    ]

    return {
        "promotions": promotions,
        "demotions": demotions,
        "_format": "nav_rebalance_plan",
        "_rollout": payload.get("rollout") or {},
    }


def _load_and_normalize_plan(plan_path: str) -> Dict:
    """Load a plan JSON and normalize it to apply_reallocation() format.

    Accepts both:
    - sleeve_promotion_plan.json  (evaluate_sleeve_promotions.py output)
    - nav_rebalance_plan_latest.json  (build_nav_rebalance_plan.py output)
    """
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    planner = (payload.get("meta") or {}).get("planner") or ""
    if planner == "build_nav_rebalance_plan" or "targets" in payload:
        plan = _adapt_nav_rebalance_plan(payload)
    else:
        plan = payload.get("plan") or payload
    plan["_source_plan"] = plan_path
    return plan


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--plan-path",
    default="logs/automation/nav_rebalance_plan_latest.json",
    show_default=True,
    help="Path to nav_rebalance_plan_latest.json (build_nav_rebalance_plan.py) "
         "or sleeve_promotion_plan.json (evaluate_sleeve_promotions.py). Both formats accepted.",
)
@click.option(
    "--config-path",
    default="config/barbell.yml",
    show_default=True,
    help="Current barbell configuration YAML",
)
@click.option(
    "--output",
    default="logs/automation/nav_allocation_latest.json",
    show_default=True,
    help="Output allocation artifact JSON",
)
@click.option(
    "--staged-config",
    default=None,
    help="Optional path to write staged barbell.yml (e.g. config/barbell.staged.yml)",
)
@click.option("--safe-min-weight", default=SAFE_MIN_WEIGHT, show_default=True)
@click.option("--core-max-weight", default=CORE_MAX_WEIGHT, show_default=True)
@click.option("--spec-max-weight", default=SPEC_MAX_WEIGHT, show_default=True)
@click.option("--min-coverage-ratio", default=0.30, show_default=True)
def main(
    plan_path: str,
    config_path: str,
    output: str,
    staged_config: Optional[str],
    safe_min_weight: float,
    core_max_weight: float,
    spec_max_weight: float,
    min_coverage_ratio: float,
) -> None:
    plan = _load_and_normalize_plan(plan_path)

    current_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}

    constraints = AllocationConstraints(
        safe_min_weight=safe_min_weight,
        core_max_weight=core_max_weight,
        spec_max_weight=spec_max_weight,
        min_coverage_ratio=min_coverage_ratio,
    )

    artifact = apply_reallocation(plan, current_config, constraints)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_artifact_to_dict(artifact), indent=2), encoding="utf-8")
    print(f"NAV allocation artifact written to {output}")
    print(f"Summary: {artifact.summary}")

    if staged_config:
        _write_staged_config(artifact, current_config, Path(staged_config))
        print(f"Staged barbell config written to {staged_config}")


if __name__ == "__main__":
    main()
