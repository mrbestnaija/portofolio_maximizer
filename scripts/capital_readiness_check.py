"""Phase 7.30 — Capital Readiness Gate.

One command answers: "Ready to scale capital?"

Composes Layer 1-4 diagnostics + adversarial status into a single PASS/FAIL/INSUFFICIENT_DATA
verdict.  This is a *diagnostic* gate, not a production trading gate.

Verdict rules (R1–R4 are hard gates; R5 is advisory):

  R1  No confirmed CRITICAL or HIGH adversarial findings.
  R2  Gate artifact (logs/gate_status_latest.json) exists, overall_passed=True, age < 26 h.
  R3  Trade quality: n_trades >= 20, win_rate >= 0.45, profit_factor >= 1.30.
  R4  Calibration not inactive: tier != 'inactive', brier < 0.25.
  R5  Lift CI positive (advisory): lift_ci_low > 0.  Emits WARNING, never FAIL.

Usage::

    python scripts/capital_readiness_check.py
    python scripts/capital_readiness_check.py --json
    python scripts/capital_readiness_check.py --db data/portfolio_maximizer.db \\
        --audit-dir logs/forecast_audits

Exit codes:
    0  ready=True  (all R1–R4 passed, no errors)
    1  ready=False (at least one R1–R4 failed)
    2  INSUFFICIENT_DATA (key inputs missing)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS_DIR = str(ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

log = logging.getLogger(__name__)

try:
    from scripts.quality_pipeline_common import (
        compute_lifecycle_integrity_metrics,
        resolve_forecast_audit_dir,
    )
except Exception:  # pragma: no cover - script execution path fallback
    from quality_pipeline_common import (
        compute_lifecycle_integrity_metrics,
        resolve_forecast_audit_dir,
    )

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DB = ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_AUDIT_ROOT = ROOT / "logs" / "forecast_audits"
DEFAULT_AUDIT_PRODUCTION_DIR = DEFAULT_AUDIT_ROOT / "production"
DEFAULT_AUDIT_DIR = DEFAULT_AUDIT_PRODUCTION_DIR
DEFAULT_JSONL = ROOT / "logs" / "signals" / "quant_validation.jsonl"
GATE_ARTIFACT = ROOT / "logs" / "gate_status_latest.json"

# Hard-gate thresholds
R3_MIN_TRADES = 20
R3_MIN_WIN_RATE = 0.45
R3_MIN_PROFIT_FACTOR = 1.30
R4_MAX_BRIER = 0.25
R2_MAX_AGE_HOURS = 26.0


# ---------------------------------------------------------------------------
# Individual rule evaluators
# ---------------------------------------------------------------------------

def _check_r1_adversarial(db_path: Path, audit_dir: Path) -> tuple[bool, str, dict]:
    """R1: 0 confirmed CRITICAL or HIGH adversarial findings."""
    try:
        from adversarial_diagnostic_runner import run_all_checks

        findings = run_all_checks(db_path, audit_dir, None, None)
        critical_high_confirmed = [
            f for f in findings
            if f.severity in ("CRITICAL", "HIGH") and not f.passed
        ]
        n = len(critical_high_confirmed)
        ids = [f.id for f in critical_high_confirmed]
        metrics = {"n_critical_high_confirmed": n, "confirmed_ids": ids}
        if n > 0:
            return False, f"R1: {n} confirmed CRITICAL/HIGH finding(s): {ids}", metrics
        return True, "", metrics
    except Exception as exc:
        log.warning("R1 adversarial check failed: %s", exc)
        return False, f"R1: adversarial check error -- {exc}", {"n_critical_high_confirmed": -1}


def _check_r2_gate_artifact() -> tuple[bool, str, dict]:
    """R2: gate artifact present, overall_passed=True, age < 26 h."""
    metrics: dict[str, Any] = {"gate_overall_passed": None, "gate_age_hours": None}
    if not GATE_ARTIFACT.exists():
        return False, "R2: gate_status_latest.json not found -- run python scripts/run_all_gates.py", metrics
    try:
        data = json.loads(GATE_ARTIFACT.read_text(encoding="utf-8"))
        overall = bool(data.get("overall_passed", False))
        mtime = GATE_ARTIFACT.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600.0
        metrics["gate_overall_passed"] = overall
        metrics["gate_age_hours"] = round(age_hours, 2)
        if not overall:
            return False, f"R2: gate artifact overall_passed=False (age={age_hours:.1f}h)", metrics
        if age_hours >= R2_MAX_AGE_HOURS:
            return False, (
                f"R2: gate artifact is stale ({age_hours:.1f}h >= {R2_MAX_AGE_HOURS}h) -- "
                "re-run python scripts/run_all_gates.py"
            ), metrics
        return True, "", metrics
    except Exception as exc:
        log.warning("R2 gate artifact check failed: %s", exc)
        return False, f"R2: gate artifact unreadable -- {exc}", metrics


def _check_r3_trade_quality(db_path: Path) -> tuple[Optional[bool], str, dict]:
    """R3: n_trades >= 20, win_rate >= 0.45, profit_factor >= 1.30."""
    metrics: dict[str, Any] = {"n_trades": None, "win_rate": None, "profit_factor": None}
    if not db_path.exists():
        return None, f"R3: db not found -- {db_path}", metrics
    try:
        from check_model_improvement import run_layer3_trade_quality

        result = run_layer3_trade_quality(db_path)
        if result.status == "SKIP":
            return None, f"R3: trade quality skipped -- {result.summary}", metrics
        n_trades = result.metrics.get("n_trades", 0)
        win_rate = result.metrics.get("win_rate", 0.0)
        pf = result.metrics.get("profit_factor", 0.0)
        metrics.update({"n_trades": n_trades, "win_rate": win_rate, "profit_factor": pf})
        reasons = []
        if (n_trades or 0) < R3_MIN_TRADES:
            reasons.append(f"only {n_trades} trades (min {R3_MIN_TRADES})")
        if (win_rate or 0.0) < R3_MIN_WIN_RATE:
            reasons.append(f"win_rate={win_rate:.1%} < {R3_MIN_WIN_RATE:.0%}")
        if (pf or 0.0) < R3_MIN_PROFIT_FACTOR:
            reasons.append(f"profit_factor={pf:.2f} < {R3_MIN_PROFIT_FACTOR:.2f}")
        if reasons:
            return False, "R3: " + ", ".join(reasons), metrics
        return True, "", metrics
    except Exception as exc:
        log.warning("R3 trade quality check failed: %s", exc)
        return None, f"R3: error -- {exc}", metrics


def _check_r4_calibration(db_path: Path, jsonl_path: Path) -> tuple[Optional[bool], str, dict]:
    """R4: calibration tier != 'inactive', brier < 0.25."""
    metrics: dict[str, Any] = {"calibration_tier": None, "brier_score": None}
    try:
        from check_model_improvement import run_layer4_calibration

        result = run_layer4_calibration(db_path, jsonl_path)
        if result.status == "SKIP":
            return None, f"R4: calibration skipped -- {result.summary}", metrics
        tier = result.metrics.get("calibration_active_tier")
        brier = result.metrics.get("brier_score")
        metrics.update({"calibration_tier": tier, "brier_score": brier})
        if tier == "inactive":
            return False, "R4: calibration tier is 'inactive' (no trained calibrator)", metrics
        # WIR-02 fix: tier='unknown' means the audit finding was absent or format changed;
        # treat as insufficient data rather than silently passing the gate.
        if tier is None or tier == "unknown":
            return None, "R4: calibration tier could not be determined (audit format mismatch or no finding)", metrics
        if brier is not None and brier >= R4_MAX_BRIER:
            return False, f"R4: brier_score={brier:.3f} >= {R4_MAX_BRIER} (miscalibrated)", metrics
        return True, "", metrics
    except Exception as exc:
        log.warning("R4 calibration check failed: %s", exc)
        return None, f"R4: error -- {exc}", metrics


def _check_r5_lift_ci(db_path: Path, audit_dir: Path) -> tuple[str, dict]:
    """R5 (advisory): lift CI_low > 0.  Returns a warning string or '' if confirmed."""
    metrics: dict[str, Any] = {"lift_ci_low": None}
    try:
        from check_model_improvement import run_layer1_forecast_quality

        result = run_layer1_forecast_quality(audit_dir=audit_dir)
        ci_low = result.metrics.get("lift_ci_low")
        insufficient = result.metrics.get("lift_ci_insufficient_data", True)
        metrics["lift_ci_low"] = ci_low
        if insufficient or ci_low is None:
            return "", metrics  # not enough data to warn
        if ci_low <= 0.0:
            import math
            ci_high = result.metrics.get("lift_ci_high", float("nan"))
            win_frac = result.metrics.get("lift_win_fraction", float("nan"))
            if math.isfinite(ci_high) and ci_high < 0.0:
                return (
                    f"R5: lift CI [{ci_low:.4f}, {ci_high:.4f}] both bounds negative "
                    f"(win_fraction={win_frac:.1%}) -- ensemble definitively underperforming, "
                    f"not merely inconclusive (advisory; see Layer 1 for gate impact)"
                ), metrics
            return (
                f"R5: lift CI [{ci_low:.4f}, {ci_high:.4f}] spans zero "
                f"(win_fraction={win_frac:.1%}) -- lift not statistically confirmed"
            ), metrics
        return "", metrics
    except Exception as exc:
        log.warning("R5 lift CI check failed: %s", exc)
        return "", metrics


def _check_r6_lifecycle_integrity(db_path: Path) -> tuple[Optional[bool], str, dict]:
    """R6: no HIGH lifecycle violations (close_ts < entry_ts, missing exit_reason)."""
    metrics: dict[str, Any] = {
        "close_before_entry_count": None,
        "closed_missing_exit_reason_count": None,
        "high_integrity_violation_count": None,
    }
    if not db_path.exists():
        return None, f"R6: db not found -- {db_path}", metrics
    try:
        shared = compute_lifecycle_integrity_metrics(db_path)
        close_before_entry_count = int(shared.get("close_before_entry_count", 0) or 0)
        missing_exit_reason_count = int(shared.get("closed_missing_exit_reason_count", 0) or 0)
        high_count = int(shared.get("high_integrity_violation_count", 0) or 0)
        metrics.update(
            {
                "close_before_entry_count": close_before_entry_count,
                "closed_missing_exit_reason_count": missing_exit_reason_count,
                "high_integrity_violation_count": high_count,
            }
        )
        query_error = shared.get("query_error")
        if query_error:
            raise RuntimeError(str(query_error))
        if high_count > 0:
            return (
                False,
                (
                    "R6: HIGH lifecycle violation(s): "
                    f"close_before_entry={close_before_entry_count}, "
                    f"closed_missing_exit_reason={missing_exit_reason_count}"
                ),
                metrics,
            )
        return True, "", metrics
    except Exception as exc:
        log.warning("R6 lifecycle integrity check failed: %s", exc)
        return None, f"R6: error -- {exc}", metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_capital_readiness(
    db_path: Path = DEFAULT_DB,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    jsonl_path: Path = DEFAULT_JSONL,
) -> dict:
    """Run all 5 rules and return the readiness verdict dict.

    Returns::

        {
            "ready": bool,
            "verdict": "PASS" | "FAIL" | "INSUFFICIENT_DATA",
            "reasons": [...],
            "warnings": [...],
            "metrics": {...},
        }
    """
    reasons: list[str] = []
    warnings: list[str] = []
    all_metrics: dict[str, Any] = {}
    insufficient = False

    resolved_audit_dir = resolve_forecast_audit_dir(
        Path(audit_dir),
        default_audit_root=DEFAULT_AUDIT_ROOT,
        default_audit_production_dir=DEFAULT_AUDIT_PRODUCTION_DIR,
    )

    # R1: adversarial
    r1_ok, r1_msg, r1_m = _check_r1_adversarial(db_path, resolved_audit_dir)
    all_metrics.update(r1_m)
    if not r1_ok:
        reasons.append(r1_msg)

    # R2: gate artifact
    r2_ok, r2_msg, r2_m = _check_r2_gate_artifact()
    all_metrics.update(r2_m)
    if not r2_ok:
        reasons.append(r2_msg)

    # R3: trade quality
    r3_ok, r3_msg, r3_m = _check_r3_trade_quality(db_path)
    all_metrics.update(r3_m)
    if r3_ok is None:
        insufficient = True
        reasons.append(r3_msg)
    elif not r3_ok:
        reasons.append(r3_msg)

    # R4: calibration
    r4_ok, r4_msg, r4_m = _check_r4_calibration(db_path, jsonl_path)
    all_metrics.update(r4_m)
    if r4_ok is None:
        insufficient = True
        reasons.append(r4_msg)
    elif not r4_ok:
        reasons.append(r4_msg)

    # R5: lift CI (advisory)
    r5_warn, r5_m = _check_r5_lift_ci(db_path, resolved_audit_dir)
    all_metrics.update(r5_m)
    if r5_warn:
        warnings.append(r5_warn)

    # R6: lifecycle integrity (hard gate)
    r6_ok, r6_msg, r6_m = _check_r6_lifecycle_integrity(db_path)
    all_metrics.update(r6_m)
    if r6_ok is None:
        insufficient = True
        reasons.append(r6_msg)
    elif not r6_ok:
        reasons.append(r6_msg)

    # Verdict
    # INSUFFICIENT_DATA only when ALL failures are missing-data (R3/R4 only) and
    # both hard gates (R1, R2) passed.  If either R1 or R2 failed, always FAIL —
    # adversarial/gate-artifact failures are hard failures regardless of data depth.
    if reasons:
        if insufficient and all(
            msg.startswith(("R3:", "R4:", "R6:")) for msg in reasons
        ) and r1_ok and r2_ok:
            verdict = "INSUFFICIENT_DATA"
        else:
            verdict = "FAIL"
        ready = False
    else:
        verdict = "PASS"
        ready = True

    return {
        "ready": ready,
        "verdict": verdict,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": all_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Capital Readiness Gate — one command to check trading readiness."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to portfolio_maximizer.db (default: data/portfolio_maximizer.db)")
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help=(
            "Forecast audit directory (default: logs/forecast_audits/production "
            "with legacy fallback to logs/forecast_audits)."
        ),
    )
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL,
                        help="Quant validation JSONL (default: logs/signals/quant_validation.jsonl)")
    parser.add_argument("--json", action="store_true", dest="emit_json",
                        help="Emit machine-readable JSON output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    result = run_capital_readiness(
        db_path=args.db,
        audit_dir=args.audit_dir,
        jsonl_path=args.jsonl,
    )

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        ready_str = "[PASS] READY" if result["ready"] else f"[FAIL] {result['verdict']}"
        print(f"\nCapital Readiness: {ready_str}")
        if result["reasons"]:
            print("\nReasons:")
            for r in result["reasons"]:
                print(f"  - {r}")
        if result["warnings"]:
            print("\nWarnings (advisory):")
            for w in result["warnings"]:
                print(f"  ! {w}")
        metrics = result["metrics"]
        print(
            f"\nMetrics: adversarial_confirmed={metrics.get('n_critical_high_confirmed')} "
            f"gate_passed={metrics.get('gate_overall_passed')} "
            f"gate_age_h={metrics.get('gate_age_hours')} "
            f"n_trades={metrics.get('n_trades')} "
            f"win_rate={metrics.get('win_rate')} "
            f"pf={metrics.get('profit_factor')} "
            f"cal_tier={metrics.get('calibration_tier')} "
            f"brier={metrics.get('brier_score')} "
            f"lift_ci_low={metrics.get('lift_ci_low')}"
        )

    sys.exit(0 if result["ready"] else (2 if result["verdict"] == "INSUFFICIENT_DATA" else 1))


if __name__ == "__main__":
    main()
