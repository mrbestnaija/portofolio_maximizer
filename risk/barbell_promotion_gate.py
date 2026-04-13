"""
risk.barbell_promotion_gate
---------------------------

Promotion gating for barbell sizing.

The goal is to prevent accidental activation of barbell sizing overlays in
production-like runs without evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ROOT_PATH = Path(__file__).resolve().parent.parent
BARBELL_CONFIG_PATH = ROOT_PATH / "config" / "barbell.yml"


@dataclass(frozen=True)
class BarbellPromotionDecision:
    passed: bool
    reason: str
    evidence_source: str
    report_path: Optional[str] = None
    checks: Dict[str, Any] = field(default_factory=dict)


def _load_promotion_thresholds() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "min_total_trades": 30,
        "min_losing_trades": 5,
        "min_omega_robustness_score": 0.45,
        "min_payoff_asymmetry_effective": 1.10,
        "max_winner_concentration_ratio": 0.60,
        "min_path_risk_ok_rate": 0.80,
        "require_path_risk_evidence": True,
    }
    try:
        payload = yaml.safe_load(BARBELL_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return defaults
    barbell = payload.get("barbell") or {}
    promotion_cfg = barbell.get("promotion_gate") or {}
    out = dict(defaults)
    if isinstance(promotion_cfg, dict):
        out.update(promotion_cfg)
    return out


def load_barbell_eval_report(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barbell eval report must be a JSON object")
    return payload


def decide_promotion_from_report(payload: Dict[str, Any]) -> BarbellPromotionDecision:
    evidence_source = str(payload.get("evidence_source") or payload.get("source") or "unknown")
    thresholds = _load_promotion_thresholds()

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

    d_return = float(delta.get("total_return_pct") or 0.0)
    d_pf = float(delta.get("profit_factor") or 0.0)
    d_dd = float(delta.get("max_drawdown") or 0.0)
    checks: Dict[str, Dict[str, Any]] = {}

    def _record(name: str, passed: Optional[bool], detail: str, *, required: bool = True) -> None:
        checks[name] = {
            "passed": passed,
            "detail": detail,
            "required": required,
        }

    min_total_trades = int(thresholds.get("min_total_trades", 30))
    min_losing_trades = int(thresholds.get("min_losing_trades", 5))
    _record("trade_support", trades >= min_total_trades, f"total_trades={trades}, required>={min_total_trades}")
    _record("loss_support", losing_trades >= min_losing_trades, f"losing_trades={losing_trades}, required>={min_losing_trades}")
    _record("pnl_delta", (d_return > 0.0 or d_pf > 0.0), f"delta_return_pct={d_return:.6f}, delta_profit_factor={d_pf:.6f}")
    _record("drawdown_regression", d_dd <= 0.0, f"delta_max_drawdown={d_dd:.6f}")

    omega_robustness = barbell.get("omega_robustness_score")
    if omega_robustness is None:
        _record("omega_robustness", False, "missing omega_robustness_score")
    else:
        threshold = float(thresholds.get("min_omega_robustness_score", 0.45))
        _record(
            "omega_robustness",
            float(omega_robustness) >= threshold,
            f"omega_robustness_score={float(omega_robustness):.4f}, required>={threshold:.4f}",
        )

    support_ok = barbell.get("payoff_asymmetry_support_ok")
    _record(
        "payoff_asymmetry_support",
        bool(support_ok),
        f"payoff_asymmetry_support_ok={support_ok}",
    )

    payoff_effective = barbell.get("payoff_asymmetry_effective")
    if payoff_effective is None:
        _record("payoff_asymmetry_effective", False, "missing payoff_asymmetry_effective")
    else:
        threshold = float(thresholds.get("min_payoff_asymmetry_effective", 1.10))
        _record(
            "payoff_asymmetry_effective",
            float(payoff_effective) >= threshold,
            f"payoff_asymmetry_effective={float(payoff_effective):.4f}, required>={threshold:.4f}",
        )

    winner_concentration = barbell.get("winner_concentration_ratio")
    if winner_concentration is None:
        _record("winner_concentration", False, "missing winner_concentration_ratio")
    else:
        threshold = float(thresholds.get("max_winner_concentration_ratio", 0.60))
        _record(
            "winner_concentration",
            float(winner_concentration) <= threshold,
            f"winner_concentration_ratio={float(winner_concentration):.4f}, required<={threshold:.4f}",
        )

    barbell_es = barbell.get("expected_shortfall")
    baseline_es = baseline.get("expected_shortfall")
    if barbell_es is None or baseline_es is None:
        _record("expected_shortfall_regression", False, "missing expected_shortfall evidence")
    else:
        _record(
            "expected_shortfall_regression",
            float(barbell_es) >= float(baseline_es),
            f"baseline_es={float(baseline_es):.6f}, barbell_es={float(barbell_es):.6f}",
        )

    barbell_dd = barbell.get("max_drawdown")
    baseline_dd = baseline.get("max_drawdown")
    if barbell_dd is None or baseline_dd is None:
        _record("max_drawdown_regression", False, "missing max_drawdown evidence")
    else:
        _record(
            "max_drawdown_regression",
            float(barbell_dd) <= float(baseline_dd),
            f"baseline_dd={float(baseline_dd):.6f}, barbell_dd={float(barbell_dd):.6f}",
        )

    path_risk_count = barbell.get("path_risk_trade_count")
    path_risk_rate = barbell.get("path_risk_ok_rate")
    if path_risk_count in (None, 0) or path_risk_rate is None:
        _record(
            "path_risk_evidence",
            False,
            "missing realized path-risk evidence",
            required=bool(thresholds.get("require_path_risk_evidence", True)),
        )
    else:
        _record("path_risk_evidence", True, f"path_risk_trade_count={int(path_risk_count)}")
        min_ok_rate = float(thresholds.get("min_path_risk_ok_rate", 0.80))
        _record(
            "path_risk_compliance",
            float(path_risk_rate) >= min_ok_rate,
            f"path_risk_ok_rate={float(path_risk_rate):.4f}, required>={min_ok_rate:.4f}",
        )

    _record(
        "safe_nav_floor",
        False,
        "pending_not_wired: NAV allocator safe-floor evidence not yet integrated",
        required=False,  # not a hard gate until NAV allocator is wired
    )

    failing_required = [
        name
        for name, detail in checks.items()
        if detail.get("required", True) and detail.get("passed") is not True
    ]
    if not failing_required:
        return BarbellPromotionDecision(
            passed=True,
            reason="barbell evidence passed robustness, tail-control, and path-risk checks",
            evidence_source=evidence_source,
            checks=checks,
        )

    return BarbellPromotionDecision(
        passed=False,
        reason="promotion evidence insufficient: " + ", ".join(sorted(failing_required)),
        evidence_source=evidence_source,
        checks=checks,
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
        "checks": dict(decision.checks or {}),
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
        checks=dict(payload.get("checks") or {}),
    )
