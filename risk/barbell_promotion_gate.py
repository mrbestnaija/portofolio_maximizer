"""
risk.barbell_promotion_gate
---------------------------

Promotion gating for barbell sizing.

The goal is to prevent accidental activation of barbell sizing overlays in
production-like runs without evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from utils.evidence_io import write_versioned_json_artifact


ROOT_PATH = Path(__file__).resolve().parent.parent
BARBELL_CONFIG_PATH = ROOT_PATH / "config" / "barbell.yml"
_DEFAULT_REGIME_REALISM_THRESHOLDS: Dict[str, Any] = {
    "min_regime_realism_labeled_trades": 10,
    "min_regime_coverage_rate": 0.80,
    "max_regime_dominance_rate": 0.70,
    "min_unique_regimes": 2,
}


@dataclass(frozen=True)
class BarbellPromotionDecision:
    passed: bool
    reason: str
    evidence_source: str
    report_path: Optional[str] = None
    checks: Dict[str, Any] = field(default_factory=dict)


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def _normalize_regime_label(value: Any) -> Optional[str]:
    label = str(value or "").strip().upper()
    if not label or label in {"UNKNOWN", "NONE", "NULL", "N/A", "NA"}:
        return None
    return label


def _regime_realism_thresholds(thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(_DEFAULT_REGIME_REALISM_THRESHOLDS)
    if isinstance(thresholds, dict):
        cfg["min_regime_realism_labeled_trades"] = _safe_int(
            thresholds.get("min_regime_realism_labeled_trades"),
            cfg["min_regime_realism_labeled_trades"],
        )
        cfg["min_regime_coverage_rate"] = _safe_float(
            thresholds.get("min_regime_coverage_rate"),
            cfg["min_regime_coverage_rate"],
        )
        cfg["max_regime_dominance_rate"] = _safe_float(
            thresholds.get("max_regime_dominance_rate"),
            cfg["max_regime_dominance_rate"],
        )
        cfg["min_unique_regimes"] = _safe_int(
            thresholds.get("min_unique_regimes"),
            cfg["min_unique_regimes"],
        )
    return cfg


def evaluate_regime_realism_summary(
    summary: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a regime-realism summary against the configured promotion policy.

    This fails closed: missing counts, missing coverage, or missing dominance
    evidence all count as failures.
    """
    policy = _regime_realism_thresholds(thresholds)
    total = _safe_int(summary.get("regime_realism_trade_count"), 0) or 0
    labeled = _safe_int(summary.get("regime_realism_labeled_trade_count"), 0) or 0
    coverage = _safe_float(summary.get("regime_realism_coverage_rate"))
    dominance_rate = _safe_float(summary.get("regime_realism_dominance_rate"))
    dominant_regime = summary.get("regime_realism_dominant_regime")
    label_counts = summary.get("regime_realism_label_counts")

    counts: Dict[str, int] = {}
    if isinstance(label_counts, dict):
        for raw_label, raw_count in label_counts.items():
            norm_label = _normalize_regime_label(raw_label)
            count = _safe_int(raw_count, 0) or 0
            if norm_label is not None and count > 0:
                counts[norm_label] = count

    if counts:
        dominant_regime, dominant_count = sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[0]
        dominance_rate = float(dominant_count / max(labeled, 1))

    unique_regimes = sorted(counts) if counts else [
        str(label).strip().upper()
        for label in (summary.get("regime_realism_unique_regimes") or [])
        if _normalize_regime_label(label) is not None
    ]

    if coverage is None and total > 0:
        coverage = float(labeled / max(total, 1))

    reasons = []
    if total <= 0:
        reasons.append("missing regime labels")
    if labeled < int(policy["min_regime_realism_labeled_trades"]):
        reasons.append(
            f"labeled_trades={labeled} < {int(policy['min_regime_realism_labeled_trades'])}"
        )
    if coverage is None:
        reasons.append("missing regime coverage rate")
    elif coverage < float(policy["min_regime_coverage_rate"]):
        reasons.append(
            f"coverage_rate={coverage:.4f} < {float(policy['min_regime_coverage_rate']):.4f}"
        )
    if len(unique_regimes) < int(policy["min_unique_regimes"]):
        reasons.append(
            f"unique_regimes={len(unique_regimes)} < {int(policy['min_unique_regimes'])}"
        )
    if dominance_rate is None:
        reasons.append("missing regime dominance rate")
    elif dominance_rate > float(policy["max_regime_dominance_rate"]):
        reasons.append(
            f"dominance_rate={dominance_rate:.4f} > {float(policy['max_regime_dominance_rate']):.4f}"
        )

    ok = not reasons
    reason = "regime realism passed" if ok else "; ".join(reasons)
    return {
        "regime_realism_trade_count": int(total),
        "regime_realism_labeled_trade_count": int(labeled),
        "regime_realism_coverage_rate": float(coverage) if coverage is not None else None,
        "regime_realism_unique_regimes": list(unique_regimes),
        "regime_realism_dominant_regime": dominant_regime,
        "regime_realism_dominance_rate": float(dominance_rate) if dominance_rate is not None else None,
        "regime_realism_label_counts": dict(sorted(counts.items())),
        "regime_realism_ok": bool(ok),
        "regime_realism_reason": reason,
        "regime_realism_thresholds": dict(policy),
    }


def summarize_regime_realism(
    regime_labels: Sequence[Optional[str]],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a regime-realism summary from executed-trade regime labels.

    Raw labels stay visible in the caller's data flow; this helper collapses them
    into audit fields that the promotion gate can consume.
    """
    raw_labels = list(regime_labels or [])
    counts: Counter[str] = Counter()
    for label in raw_labels:
        norm = _normalize_regime_label(label)
        if norm is not None:
            counts[norm] += 1

    total = len(raw_labels)
    labeled = int(sum(counts.values()))
    coverage = float(labeled / total) if total > 0 else None
    unique_regimes = sorted(counts)
    dominant_regime = None
    dominance_rate = None
    if counts:
        dominant_regime, dominant_count = sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[0]
        dominance_rate = float(dominant_count / max(labeled, 1))

    summary: Dict[str, Any] = {
        "regime_realism_trade_count": int(total),
        "regime_realism_labeled_trade_count": int(labeled),
        "regime_realism_coverage_rate": coverage,
        "regime_realism_unique_regimes": unique_regimes,
        "regime_realism_dominant_regime": dominant_regime,
        "regime_realism_dominance_rate": dominance_rate,
        "regime_realism_label_counts": dict(sorted(counts.items())),
    }
    summary.update(evaluate_regime_realism_summary(summary, thresholds))
    return summary


def _load_promotion_thresholds() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "min_total_trades": 30,
        "min_losing_trades": 5,
        "min_omega_robustness_score": 0.45,
        "min_payoff_asymmetry_effective": 1.10,
        "max_winner_concentration_ratio": 0.60,
        "min_path_risk_ok_rate": 0.80,
        "require_path_risk_evidence": True,
        **_DEFAULT_REGIME_REALISM_THRESHOLDS,
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

    omega_monotonicity_ok = barbell.get("omega_monotonicity_ok")
    _record(
        "omega_monotonicity",
        omega_monotonicity_ok is True,
        f"omega_monotonicity_ok={omega_monotonicity_ok}",
    )

    omega_cliff_ok = barbell.get("omega_cliff_ok")
    cliff_drop = barbell.get("omega_cliff_drop_ratio")
    cliff_detail = f"omega_cliff_ok={omega_cliff_ok}"
    if isinstance(cliff_drop, (int, float)):
        cliff_detail += f", omega_cliff_drop_ratio={float(cliff_drop):.4f}"
    _record("omega_cliff", omega_cliff_ok is True, cliff_detail)

    omega_right_tail_ok = barbell.get("omega_right_tail_ok")
    omega_ci_lower = barbell.get("omega_ci_lower")
    right_tail_detail = f"omega_right_tail_ok={omega_right_tail_ok}"
    if isinstance(omega_ci_lower, (int, float)):
        right_tail_detail += f", omega_ci_lower={float(omega_ci_lower):.4f}"
    _record("omega_right_tail", omega_right_tail_ok is True, right_tail_detail)

    es_bounded = barbell.get("es_to_edge_bounded")
    es_ratio = barbell.get("expected_shortfall_to_edge")
    left_tail_detail = f"es_to_edge_bounded={es_bounded}"
    if isinstance(es_ratio, (int, float)):
        left_tail_detail += f", expected_shortfall_to_edge={float(es_ratio):.4f}"
    _record("left_tail_bounded", es_bounded is True, left_tail_detail)

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

    regime_realism = evaluate_regime_realism_summary(barbell, thresholds)
    _record(
        "regime_realism",
        bool(regime_realism.get("regime_realism_ok", False)),
        str(regime_realism.get("regime_realism_reason") or "missing regime realism evidence"),
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
            reason="barbell evidence passed robustness, tail-control, path-risk, and regime-realism checks",
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
) -> Dict[str, Any]:
    out = {
        "passed": bool(decision.passed),
        "reason": str(decision.reason),
        "evidence_source": str(decision.evidence_source),
        "report_path": str(report_path) if report_path else decision.report_path,
        "checks": dict(decision.checks or {}),
    }
    archive_name = path.stem[:-7] if path.stem.endswith("_latest") else path.stem
    archive_root = path.parent / f"{archive_name}_history"
    return write_versioned_json_artifact(
        latest_path=path,
        payload=out,
        archive_root=archive_root,
        archive_name=archive_name,
    )


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
