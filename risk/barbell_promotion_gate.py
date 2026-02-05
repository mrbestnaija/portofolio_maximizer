"""
risk.barbell_promotion_gate
---------------------------

Promotion gating for barbell sizing.

The goal is to prevent accidental activation of barbell sizing overlays in
production-like runs without evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BarbellPromotionDecision:
    passed: bool
    reason: str
    evidence_source: str
    report_path: Optional[str] = None


def load_barbell_eval_report(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barbell eval report must be a JSON object")
    return payload


def decide_promotion_from_report(payload: Dict[str, Any]) -> BarbellPromotionDecision:
    evidence_source = str(payload.get("evidence_source") or payload.get("source") or "unknown")

    metrics = payload.get("metrics") or {}
    baseline = metrics.get("ts_only") or {}
    barbell = metrics.get("barbell_sized") or {}
    delta = metrics.get("delta") or {}

    try:
        trades = int(barbell.get("total_trades") or 0)
    except Exception:
        trades = 0

    try:
        losing_trades = int(barbell.get("losing_trades") or 0)
    except Exception:
        losing_trades = 0

    if trades < 30:
        return BarbellPromotionDecision(
            passed=False,
            reason=f"insufficient trades (total_trades={trades} < 30)",
            evidence_source=evidence_source,
        )
    if losing_trades < 5:
        return BarbellPromotionDecision(
            passed=False,
            reason=f"insufficient losses for inference (losing_trades={losing_trades} < 5)",
            evidence_source=evidence_source,
        )

    d_return = float(delta.get("total_return_pct") or 0.0)
    d_pf = float(delta.get("profit_factor") or 0.0)
    d_dd = float(delta.get("max_drawdown") or 0.0)

    if (d_return > 0.0 or d_pf > 0.0) and d_dd <= 0.0:
        return BarbellPromotionDecision(
            passed=True,
            reason="meets delta criteria (return/PF up, drawdown not worse)",
            evidence_source=evidence_source,
        )

    base_pf = baseline.get("profit_factor")
    bb_pf = barbell.get("profit_factor")
    base_ret = baseline.get("total_return_pct")
    bb_ret = barbell.get("total_return_pct")
    base_dd = baseline.get("max_drawdown")
    bb_dd = barbell.get("max_drawdown")
    return BarbellPromotionDecision(
        passed=False,
        reason=(
            "delta criteria not met "
            f"(Δreturn_pct={d_return:.6f}, Δpf={d_pf:.6f}, Δdd={d_dd:.6f}; "
            f"TS_ONLY pf={base_pf}, ret={base_ret}, dd={base_dd}; "
            f"BARBELL pf={bb_pf}, ret={bb_ret}, dd={bb_dd})"
        ),
        evidence_source=evidence_source,
    )


def write_promotion_evidence(
    *,
    path: Path,
    decision: BarbellPromotionDecision,
    report_path: Optional[Path] = None,
) -> None:
    out = {
        "passed": bool(decision.passed),
        "reason": str(decision.reason),
        "evidence_source": str(decision.evidence_source),
        "report_path": str(report_path) if report_path else decision.report_path,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def load_promotion_evidence(path: Path) -> BarbellPromotionDecision:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("promotion evidence must be a JSON object")
    return BarbellPromotionDecision(
        passed=bool(payload.get("passed")),
        reason=str(payload.get("reason") or ""),
        evidence_source=str(payload.get("evidence_source") or "unknown"),
        report_path=(str(payload.get("report_path")) if payload.get("report_path") else None),
    )
