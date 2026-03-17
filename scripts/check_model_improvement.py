"""
check_model_improvement.py -- Unified 4-layer model improvement checker.

Runs all four diagnostic measurement layers and surfaces a unified health
picture. Each layer tells a different part of the story:

  Layer 1 -- Forecast Quality : ensemble lift, SAMOSSA DA anomaly, data coverage
  Layer 2 -- Gate Status       : run_all_gates.py --json (surface-only, no reinterpretation)
  Layer 3 -- Trade Quality     : win rate, profit factor, exit-reason gap
  Layer 4 -- Calibration       : Platt scaling tier, Brier score, ECE

Usage:
    python scripts/check_model_improvement.py
    python scripts/check_model_improvement.py --layer 1
    python scripts/check_model_improvement.py --json
    python scripts/check_model_improvement.py --save-baseline logs/baseline_YYYYMMDD.json
    python scripts/check_model_improvement.py --baseline logs/baseline_YYYYMMDD.json

Exit codes:
    0 = all run layers are PASS / WARN / SKIP
    1 = at least one layer FAIL
    2 = runtime error

SKIP != PASS. A SKIP layer means 'no measurement data available' (empty audit
dir, DB not found, etc.). It provides no health signal.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure repo root on path for integrity/ and models/ imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure scripts/ on path for ensemble_health_audit and platt_contract_audit imports
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level optional imports (fallback to None enables monkeypatching in tests)
# ---------------------------------------------------------------------------
try:
    from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
except ImportError:
    PnLIntegrityEnforcer = None  # type: ignore[assignment,misc]

try:
    from exit_quality_audit import (  # noqa: PLC0415
        diagnose_direction_gap,
        load_production_trades,
    )
except ImportError:
    diagnose_direction_gap = None  # type: ignore[assignment]
    load_production_trades = None  # type: ignore[assignment]

try:
    from platt_contract_audit import run_audit  # noqa: PLC0415
except ImportError:
    run_audit = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Required metrics schema per layer (used for schema invariant checks in tests)
# ---------------------------------------------------------------------------
LAYER_REQUIRED_KEYS: dict[int, set[str]] = {
    1: {
        "baseline_model",
        "lift_threshold_rmse_ratio",
        "lift_fraction_global",
        "lift_fraction_recent",
        "samossa_da_zero_pct",
        "n_used_windows",
        "n_skipped_malformed",
        "n_skipped_missing_metrics",
        "n_total_files",
        "coverage_ratio",       # Phase 7.19: n_used / n_total; WARN when < 0.20
        "lift_mean",            # Phase 7.25: bootstrap mean(delta), delta=best_single-ensemble
        "lift_ci_low",          # Phase 7.25: lower 95% bootstrap CI bound
        "lift_ci_high",         # Phase 7.25: upper 95% bootstrap CI bound
        "lift_win_fraction",    # Phase 7.25: fraction of windows with positive lift
        "lift_ci_insufficient_data",  # Phase 7.25: True when < 5 valid windows for CI
    },
    2: {"overall_passed", "n_gates_passed", "n_gates_failed"},
    3: {"win_rate", "profit_factor", "total_pnl", "n_trades", "interpretation"},
    4: {"overall_status", "calibration_active_tier", "brier_score", "ece"},
}


# ---------------------------------------------------------------------------
# LayerResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class LayerResult:
    layer: int
    name: str
    status: str          # PASS | WARN | FAIL | SKIP
    metrics: dict
    summary: str
    error: Optional[str] = None


def _empty_metrics(layer: int, **overrides) -> dict:
    """Produce a metrics dict with all required keys set to None, plus any overrides."""
    base = {k: None for k in LAYER_REQUIRED_KEYS[layer]}
    base.update(overrides)
    return base


def _json_safe(value):  # type: ignore[no-untyped-def]
    """Recursively normalize non-finite floats so JSON output stays standards-compliant."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _resolve_layer1_audit_dir(explicit_audit_dir: Optional[str | Path] = None) -> Path:
    """Resolve Layer 1's audit universe from active production/cohort state."""
    if explicit_audit_dir is not None:
        explicit_raw = str(explicit_audit_dir).strip()
        if explicit_raw:
            return Path(explicit_raw)

    env_audit_dir = str(os.environ.get("TS_FORECAST_AUDIT_DIR") or "").strip()
    if env_audit_dir:
        return Path(env_audit_dir)

    cohort_id = str(os.environ.get("PMX_EVIDENCE_COHORT_ID") or "").strip()
    if cohort_id:
        cohort_production_dir = (
            REPO_ROOT / "logs" / "forecast_audits" / "cohorts" / cohort_id / "production"
        )
        if cohort_production_dir.exists():
            return cohort_production_dir

    production_dir = REPO_ROOT / "logs" / "forecast_audits" / "production"
    if production_dir.exists():
        return production_dir

    return REPO_ROOT / "logs" / "forecast_audits"


def _load_layer1_regression_contract() -> tuple[str, float]:
    baseline_model = "BEST_SINGLE"
    min_lift_rmse_ratio = 0.0
    try:
        import yaml as _yaml_layer1_contract

        monitor_cfg_path = REPO_ROOT / "config" / "forecaster_monitoring.yml"
        if monitor_cfg_path.exists():
            monitor_cfg = _yaml_layer1_contract.safe_load(
                monitor_cfg_path.read_text(encoding="utf-8")
            ) or {}
            regression_cfg = monitor_cfg.get("forecaster_monitoring", {}).get(
                "regression_metrics",
                {},
            )
            baseline_model = str(
                regression_cfg.get("baseline_model", baseline_model) or baseline_model
            ).strip().upper() or "BEST_SINGLE"
            min_lift_rmse_ratio = float(regression_cfg.get("min_lift_rmse_ratio", 0.0) or 0.0)
    except Exception:
        pass
    return baseline_model, 1.0 - min_lift_rmse_ratio


# ---------------------------------------------------------------------------
# Layer 1: Forecast Quality
# ---------------------------------------------------------------------------
def run_layer1_forecast_quality(
    audit_dir: Path,
    warn_lift_threshold: float = 0.05,
    fail_lift_threshold: float = 0.01,
    warn_da_zero_pct: float = 0.40,
    min_windows_for_fail: int = 100,
    warn_coverage_threshold: int = 50,
    recent_n: int = 20,
) -> LayerResult:
    """Load forecast audits and compute lift fractions with data-quality tracking.

    Status rules:
      SKIP  -- audit_dir missing/empty or import failure
      FAIL  -- n_used >= min_windows_for_fail AND lift_global < fail_lift_threshold
      WARN  -- lift_global < warn_lift_threshold, OR samossa_da_zero_pct > warn_da_zero_pct,
               OR n_used < warn_coverage_threshold
      PASS  -- none of the above
    """
    audit_dir = Path(audit_dir)

    if not audit_dir.exists():
        return LayerResult(
            layer=1,
            name="Forecast Quality",
            status="SKIP",
            metrics=_empty_metrics(1, n_total_files=0),
            summary=f"SKIP -- audit_dir not found: {audit_dir}",
        )

    try:
        from ensemble_health_audit import (  # noqa: PLC0415
            _window_fingerprint,
            compute_lift_significance,
            compute_per_model_summary,
            extract_window_metrics,
        )
    except ImportError as exc:
        return LayerResult(
            layer=1,
            name="Forecast Quality",
            status="SKIP",
            metrics=_empty_metrics(1),
            summary=f"SKIP -- could not import ensemble_health_audit: {exc}",
            error=str(exc),
        )

    # --- Load all forecast_audit_*.json files and track quality ---
    files = sorted(audit_dir.glob("forecast_audit_*.json"))
    n_total = len(files)

    if n_total == 0:
        return LayerResult(
            layer=1,
            name="Forecast Quality",
            status="SKIP",
            metrics=_empty_metrics(1, n_total_files=0),
            summary="SKIP -- no forecast_audit_*.json files in audit_dir",
        )

    n_malformed = 0
    raw_audits: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_path"] = f.name
            data["_mtime"] = f.stat().st_mtime
            raw_audits.append(data)
        except Exception:
            n_malformed += 1

    # Deduplicate: keep newest per fingerprint
    seen: dict[str, dict] = {}
    for audit in sorted(raw_audits, key=lambda a: (a.get("_mtime", 0.0), a.get("_path", ""))):
        try:
            fp = _window_fingerprint(audit)
        except Exception:
            fp = audit.get("_path", str(id(audit)))
        seen[fp] = audit
    deduped = list(seen.values())

    # Extract metrics, count skipped-missing
    n_missing = 0
    windows: list[dict] = []
    for raw in deduped:
        w = extract_window_metrics(raw)
        if w is None:
            n_missing += 1
        else:
            windows.append(w)

    n_used = len(windows)
    # coverage_ratio = usable windows / total files (after dedup); WARN when < 20%.
    # Low coverage_ratio means most audit files are in old pre-Phase-7.15-F format
    # and should not be silently treated as "no data" — they are structurally missing
    # the evaluation_metrics.ensemble key needed for lift calculations.
    coverage_ratio = n_used / n_total if n_total > 0 else 0.0
    quality: dict = {
        "n_total_files": n_total,
        "n_skipped_malformed": n_malformed,
        "n_skipped_missing_metrics": n_missing,
        "n_used_windows": n_used,
        "coverage_ratio": coverage_ratio,
    }

    baseline_model, lift_threshold_rmse_ratio = _load_layer1_regression_contract()

    if n_used == 0:
        return LayerResult(
            layer=1,
            name="Forecast Quality",
            status="SKIP",
            metrics=_empty_metrics(
                1,
                **quality,
                baseline_model=baseline_model,
                lift_threshold_rmse_ratio=lift_threshold_rmse_ratio,
                lift_fraction_global=0.0,
                lift_fraction_recent=0.0,
                samossa_da_zero_pct=0.0,
                lift_mean=None,
                lift_ci_low=None,
                lift_ci_high=None,
                lift_win_fraction=0.0,
                lift_ci_insufficient_data=True,
            ),
            summary=(
                f"SKIP -- 0 usable windows "
                f"(total={n_total}, malformed={n_malformed}, missing_metrics={n_missing})"
            ),
        )

    # Sort by window_end (timestamp-based recency, not filename)
    windows_sorted = sorted(
        windows, key=lambda w: (w.get("window_end") or "", w.get("window_id") or "")
    )

    # Warn if filename order and timestamp order disagree
    filename_order = [w["window_id"] for w in windows]
    timestamp_order = [w["window_id"] for w in windows_sorted]
    if filename_order != timestamp_order:
        log.warning(
            "Layer 1: filename order and window_end timestamp order disagree -- "
            "using window_end order for recency (may indicate stale file timestamps)"
        )

    # WIRE-01 fix: read min_lift_rmse_ratio from forecaster_monitoring.yml so Layer 1 lift
    # computation is aligned with check_forecast_audits.py (which also uses this value).
    _lift_threshold = lift_threshold_rmse_ratio

    # Lift fractions
    def _lift_frac(ws: list[dict]) -> float:
        if not ws:
            return 0.0
        lift_count = sum(1 for w in ws if (w.get("rmse_ratio") or math.inf) < _lift_threshold)
        return lift_count / len(ws)

    lift_global = _lift_frac(windows_sorted)
    recent_ws = windows_sorted[-recent_n:] if len(windows_sorted) > recent_n else windows_sorted
    lift_recent = _lift_frac(recent_ws)

    # SAMOSSA DA=0 pct
    samossa_da_zero_pct = 0.0
    try:
        model_summary = compute_per_model_summary(windows)
        samossa_stats = model_summary.get("samossa", {})
        samossa_da_zero_windows = samossa_stats.get("da_zero_windows", 0)
        samossa_da_zero_pct = samossa_da_zero_windows / n_used if n_used > 0 else 0.0
    except Exception as exc:
        log.warning("Could not compute per-model summary: %s", exc)

    # Phase 7.25: bootstrap CI for mean lift
    sig = compute_lift_significance(windows_sorted)

    metrics = {
        **quality,
        "baseline_model": baseline_model,
        "lift_threshold_rmse_ratio": lift_threshold_rmse_ratio,
        "lift_fraction_global": lift_global,
        "lift_fraction_recent": lift_recent,
        "samossa_da_zero_pct": samossa_da_zero_pct,
        "lift_mean": sig["mean_lift"],
        "lift_ci_low": sig["ci_low"],
        "lift_ci_high": sig["ci_high"],
        "lift_win_fraction": sig["lift_win_fraction"],
        "lift_ci_insufficient_data": sig["insufficient_data"],
    }

    # Determine status (FAIL > WARN > PASS, checked in order)
    status = "PASS"
    reasons: list[str] = []

    # THR-01 fix: critically-low coverage_ratio escalates to FAIL (not just WARN) when there
    # are enough windows to be statistically meaningful. coverage_ratio < 5% with n_used >= 50
    # means lift evidence is drawn from a non-representative sample -- hard failure.
    if coverage_ratio < 0.05 and n_used >= 50:
        status = "FAIL"
        reasons.append(
            f"coverage_ratio={coverage_ratio:.1%} < 5% FAIL escalation (n_used={n_used} >= 50)"
        )

    if n_used >= min_windows_for_fail and lift_global < fail_lift_threshold:
        status = "FAIL"
        reasons.append(
            f"lift_global={lift_global:.3f} < fail_threshold={fail_lift_threshold} "
            f"(n={n_used} >= {min_windows_for_fail})"
        )

    if status == "PASS":
        if lift_global < warn_lift_threshold:
            status = "WARN"
            reasons.append(f"lift_global={lift_global:.3f} < warn_threshold={warn_lift_threshold}")
        if samossa_da_zero_pct > warn_da_zero_pct:
            status = "WARN"
            reasons.append(
                f"samossa_da_zero={samossa_da_zero_pct:.1%} > {warn_da_zero_pct:.0%}"
            )
        if n_used < warn_coverage_threshold:
            status = "WARN"
            reasons.append(f"n_used={n_used} < coverage_threshold={warn_coverage_threshold}")
        if coverage_ratio < 0.20:
            status = "WARN"
            reasons.append(
                f"coverage_ratio={coverage_ratio:.1%} < 20% "
                f"(only {n_used}/{n_total} files are post-Phase-7.15-F format)"
            )
        # Phase 7.25/7.37: spans-zero CI → advisory WARN (only from PASS, requires ci_high >= 0)
        if not sig["insufficient_data"] and n_used >= 20:
            if sig["ci_low"] <= 0.0 and sig["ci_high"] >= 0.0:
                status = "WARN"
                reasons.append(
                    f"lift CI [{sig['ci_low']:.4f}, {sig['ci_high']:.4f}] spans zero "
                    f"(win_fraction={sig['lift_win_fraction']:.1%}) -- lift not statistically confirmed"
                )

    # Phase 7.37: definitively negative CI → hard FAIL (promotes WARN to FAIL; both bounds < 0)
    if not sig["insufficient_data"] and n_used >= 20 and sig["ci_high"] < 0.0:
        status = "FAIL"
        reasons.append(
            f"lift CI [{sig['ci_low']:.4f}, {sig['ci_high']:.4f}] definitively negative "
            f"(win_fraction={sig['lift_win_fraction']:.1%}) -- ensemble consistently worse than best single"
        )

    reason_str = " | " + "; ".join(reasons) if reasons else ""
    summary = (
        f"{status} | baseline={baseline_model} lift_threshold={lift_threshold_rmse_ratio:.3f} "
        f"lift_global={lift_global:.3f} lift_recent={lift_recent:.3f} "
        f"samossa_da_zero={samossa_da_zero_pct:.1%} n_used={n_used}{reason_str}"
    )
    return LayerResult(layer=1, name="Forecast Quality", status=status, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# Layer 2: Gate Status (surface-only — no reinterpretation of gate logic)
# ---------------------------------------------------------------------------
def run_layer2_gate_status(root: Optional[Path] = None) -> LayerResult:
    """Subprocess run_all_gates.py --json and surface overall_passed.

    CONSTRAINT: Never reinterpret gate results. If the production audit gate
    fails on lift, this layer is FAIL. No softening.

    Status rules:
      SKIP  -- scripts/run_all_gates.py not found
      FAIL  -- overall_passed is False
      PASS  -- overall_passed is True
    """
    if root is None:
        root = REPO_ROOT
    gates_script = Path(root) / "scripts" / "run_all_gates.py"

    if not gates_script.exists():
        return LayerResult(
            layer=2,
            name="Gate Status",
            status="SKIP",
            metrics=_empty_metrics(2),
            summary=f"SKIP -- run_all_gates.py not found: {gates_script}",
        )

    try:
        result = subprocess.run(
            [sys.executable, str(gates_script), "--json"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        raw_output = result.stdout.strip()
        if not raw_output:
            raw_output = result.stderr.strip()

        # run_all_gates may embed JSON after other output; find the JSON object
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to extract the last {...} block from output
            start = raw_output.rfind("{")
            end = raw_output.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw_output[start:end])
            else:
                raise

    except subprocess.TimeoutExpired:
        return LayerResult(
            layer=2,
            name="Gate Status",
            status="FAIL",
            metrics=_empty_metrics(2),
            summary="FAIL -- run_all_gates.py timed out after 180s",
            error="TimeoutExpired",
        )
    except Exception as exc:
        return LayerResult(
            layer=2,
            name="Gate Status",
            status="FAIL",
            metrics=_empty_metrics(2),
            summary=f"FAIL -- could not parse run_all_gates.py output: {exc}",
            error=str(exc),
        )

    # BYP-02 fix: cross-check subprocess exit code with JSON field.
    # If the process exited non-zero, treat the gate as FAIL regardless of JSON content.
    # A JSON field of overall_passed=true but exit_code=1 indicates a discrepancy;
    # we conservatively trust the exit code (more difficult to spoof than JSON output).
    json_overall_passed = bool(data.get("overall_passed", False))
    if result.returncode != 0 and json_overall_passed:
        # Discrepancy: exit code says failure, JSON says pass. Trust exit code.
        overall_passed = False
    elif result.returncode != 0:
        overall_passed = False
    else:
        overall_passed = json_overall_passed

    gates = data.get("gates", [])
    n_passed = sum(1 for g in gates if g.get("passed", False))
    n_failed = sum(1 for g in gates if not g.get("passed", True))
    gate_names = [g.get("label", "?") for g in gates]

    metrics = {
        "overall_passed": overall_passed,
        "n_gates_passed": n_passed,
        "n_gates_failed": n_failed,
        "gate_names": gate_names,
    }
    status = "PASS" if overall_passed else "FAIL"
    summary = (
        f"{status} | overall_passed={overall_passed} "
        f"{n_passed}/{len(gates)} gates passed"
    )
    return LayerResult(layer=2, name="Gate Status", status=status, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# Layer 3: Trade Quality
# ---------------------------------------------------------------------------
def run_layer3_trade_quality(
    db_path: Path,
    tail_n: int = 100,
    win_rate_warn: float = 0.45,
    profit_factor_warn: float = 1.30,
) -> LayerResult:
    """Import PnLIntegrityEnforcer and exit_quality_audit to measure trade health.

    Thresholds aligned with quant_success_config.yml (min_directional_accuracy=0.45).

    Status rules:
      SKIP  -- db_path not found OR no production closed trades
      WARN  -- win_rate < win_rate_warn, OR profit_factor < profit_factor_warn,
               OR interpretation not in ('ok', 'mix')
      PASS  -- none of the above
    """
    db_path = Path(db_path)

    if not db_path.exists():
        return LayerResult(
            layer=3,
            name="Trade Quality",
            status="SKIP",
            metrics=_empty_metrics(3),
            summary=f"SKIP -- db not found: {db_path}; trade health unknown",
        )

    if PnLIntegrityEnforcer is None:
        return LayerResult(
            layer=3,
            name="Trade Quality",
            status="SKIP",
            metrics=_empty_metrics(3),
            summary="SKIP -- could not import PnLIntegrityEnforcer (integrity package missing)",
        )

    if load_production_trades is None or diagnose_direction_gap is None:
        return LayerResult(
            layer=3,
            name="Trade Quality",
            status="SKIP",
            metrics=_empty_metrics(3),
            summary="SKIP -- could not import exit_quality_audit (scripts package missing)",
        )

    # Canonical PnL metrics
    try:
        enforcer = PnLIntegrityEnforcer(str(db_path))
        canonical = enforcer.get_canonical_metrics()
        enforcer.conn.close()
    except FileNotFoundError:
        return LayerResult(
            layer=3,
            name="Trade Quality",
            status="SKIP",
            metrics=_empty_metrics(3),
            summary=f"SKIP -- db not found by enforcer: {db_path}",
        )
    except Exception as exc:
        return LayerResult(
            layer=3,
            name="Trade Quality",
            status="FAIL",
            metrics=_empty_metrics(3),
            summary=f"FAIL -- PnLIntegrityEnforcer error: {exc}",
            error=str(exc),
        )

    n_trades = canonical.total_round_trips
    if n_trades == 0:
        return LayerResult(
            layer=3,
            name="Trade Quality",
            status="SKIP",
            metrics=_empty_metrics(3, n_trades=0),
            summary="SKIP -- no production closed trades; trade health unknown",
        )

    # Exit quality gap diagnosis
    trades_df = load_production_trades(db_path, tail_n=tail_n)
    gap = diagnose_direction_gap(trades_df)

    win_rate = canonical.win_rate
    profit_factor = canonical.profit_factor
    total_pnl = canonical.total_realized_pnl
    interpretation = gap.get("interpretation", "unknown")

    metrics: dict = {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "interpretation": interpretation,
    }
    # Optional enrichment
    if hasattr(canonical, "avg_win") and hasattr(canonical, "avg_loss"):
        avg_win = canonical.avg_win
        avg_loss = abs(canonical.avg_loss) if canonical.avg_loss else 0.0
        if avg_loss > 0:
            metrics["expected_value_per_trade"] = round(
                win_rate * avg_win - (1 - win_rate) * avg_loss, 4
            )

    status = "PASS"
    reasons: list[str] = []
    if win_rate < win_rate_warn:
        status = "WARN"
        reasons.append(f"win_rate={win_rate:.1%} < {win_rate_warn:.0%}")
    if profit_factor < profit_factor_warn:
        status = "WARN"
        reasons.append(f"profit_factor={profit_factor:.2f} < {profit_factor_warn}")
    if interpretation not in ("ok", "mix"):
        status = "WARN"
        reasons.append(f"interpretation={interpretation!r}")

    reason_str = " | " + "; ".join(reasons) if reasons else ""
    summary = (
        f"{status} | win_rate={win_rate:.1%} pf={profit_factor:.2f} "
        f"pnl=${total_pnl:+,.2f} n={n_trades} interp={interpretation!r}{reason_str}"
    )
    return LayerResult(layer=3, name="Trade Quality", status=status, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# Layer 4: Calibration (mirrors platt_contract_audit semantics exactly)
# ---------------------------------------------------------------------------
def run_layer4_calibration(db_path: Path, jsonl_path: Path) -> LayerResult:
    """Call platt_contract_audit.run_audit() and surface tier + quality metrics.

    Status rules (mirror platt_contract_audit findings):
      SKIP  -- both db_path and jsonl_path are missing
      FAIL  -- overall_status == 'FAIL' OR calibration_active_tier == 'inactive'
      WARN  -- overall_status == 'WARN'
      PASS  -- overall_status == 'PASS'

    Structural debt (as of Phase 7.19 — expected, not a failure signal):
      - DB tier (TIER_3_DB) is the PRIMARY tier; it reads from is_close=1
        production trades. This is the most reliable source of truth.
      - JSONL tier (TIER_1_JSONL) is structurally starved: ~50% of entries are
        HOLD signals which can never generate matched outcome pairs. Expect this
        tier to remain underutilised until HOLD filter is applied at write-time.
      - ECE > 0.15 with < 50 pairs is expected (noise-dominated, not model failure).
        WARN is the normal calibration state during early-accumulation.
      - WARN on calibration_quality with n < 50 pairs is not actionable; accumulate
        more trades before treating calibration ECE/Brier as diagnostic signals.
    """
    db_path = Path(db_path)
    jsonl_path = Path(jsonl_path)

    if not db_path.exists() and not jsonl_path.exists():
        return LayerResult(
            layer=4,
            name="Calibration",
            status="SKIP",
            metrics=_empty_metrics(4),
            summary=(
                f"SKIP -- both db ({db_path.name}) and jsonl ({jsonl_path.name}) not found; "
                "calibration health unknown"
            ),
        )

    if run_audit is None:
        return LayerResult(
            layer=4,
            name="Calibration",
            status="SKIP",
            metrics=_empty_metrics(4),
            summary="SKIP -- could not import platt_contract_audit (scripts package missing)",
        )

    try:
        findings = run_audit(db_path=db_path, jsonl_path=jsonl_path)
    except Exception as exc:
        return LayerResult(
            layer=4,
            name="Calibration",
            status="FAIL",
            metrics=_empty_metrics(4),
            summary=f"FAIL -- platt_contract_audit.run_audit() raised: {exc}",
            error=str(exc),
        )

    statuses = [f.status for f in findings]
    overall_status = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    # Extract active tier from calibration_active_tier finding
    tier = "unknown"
    tier_finding = next((f for f in findings if f.check == "calibration_active_tier"), None)
    if tier_finding:
        detail = tier_finding.detail
        if "TIER_1_JSONL" in detail:
            tier = "jsonl"
        elif "TIER_3_DB" in detail or "TIER_3_PARTIAL" in detail:
            tier = "db_local"
        elif "NONE" in detail:
            tier = "inactive"
        elif tier_finding.status == "SKIP":
            tier = "skip"

    # Extract Brier/ECE from calibration_quality finding
    brier: Optional[float] = None
    ece: Optional[float] = None
    quality_finding = next((f for f in findings if f.check == "calibration_quality"), None)
    if quality_finding and quality_finding.status not in ("SKIP", "FAIL"):
        detail = quality_finding.detail
        try:
            if "ECE=" in detail:
                ece = float(detail.split("ECE=")[1].split(" ")[0].rstrip(",)"))
            if "Brier=" in detail:
                brier = float(detail.split("Brier=")[1].split(" ")[0].rstrip(",)"))
        except (IndexError, ValueError):
            pass

    # FAIL override: inactive tier
    if tier == "inactive" and overall_status != "FAIL":
        overall_status = "FAIL"

    metrics = {
        "overall_status": overall_status,
        "calibration_active_tier": tier,
        "brier_score": brier,
        "ece": ece,
    }
    summary = (
        f"{overall_status} | tier={tier!r} "
        f"brier={f'{brier:.4f}' if brier is not None else 'n/a'} "
        f"ece={f'{ece:.4f}' if ece is not None else 'n/a'}"
    )
    return LayerResult(
        layer=4,
        name="Calibration",
        status=overall_status,
        metrics=metrics,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Baseline save / compare
# ---------------------------------------------------------------------------
_SIGNAL_METRICS: dict[int, list[str]] = {
    1: ["lift_fraction_global", "lift_fraction_recent", "samossa_da_zero_pct"],
    3: ["win_rate", "profit_factor"],
    4: ["brier_score", "ece"],
}


def save_baseline(results: list[LayerResult], path: Path) -> None:
    """Persist current layer results as a baseline JSON for later comparison."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "results": [
            {"layer": r.layer, "name": r.name, "status": r.status, "metrics": r.metrics}
            for r in results
        ],
    }
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    log.info("Baseline saved to %s", path)


def compare_baseline(results: list[LayerResult], baseline_path: Path) -> dict:
    """Compare current results to a saved baseline on non-volatile signal metrics.

    Volatile counts (n_trades, n_windows, etc.) are excluded to avoid noisy deltas.
    Returns: {layer_name: {metric: {before, after, delta}}}
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return {}

    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_layer: dict[int, dict] = {
        r["layer"]: r["metrics"] for r in baseline_data.get("results", [])
    }

    comparison: dict[str, dict] = {}
    current_by_layer = {r.layer: r for r in results}

    for layer, metric_names in _SIGNAL_METRICS.items():
        current = current_by_layer.get(layer)
        baseline_metrics = baseline_by_layer.get(layer, {})
        if current is None:
            continue
        layer_diff: dict[str, dict] = {}
        for m in metric_names:
            before = baseline_metrics.get(m)
            after = current.metrics.get(m)
            if before is None and after is None:
                continue
            delta = None
            if before is not None and after is not None:
                try:
                    delta = round(float(after) - float(before), 6)
                except (TypeError, ValueError):
                    pass
            layer_diff[m] = {"before": before, "after": after, "delta": delta}
        if layer_diff:
            comparison[current.name] = layer_diff

    return comparison


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
_STATUS_WIDTH = 6
_NAME_WIDTH = 20
_SUMMARY_WIDTH = 55


def print_summary_table(results: list[LayerResult]) -> None:
    """Print an ASCII summary table of all layer results."""
    header = (
        f"{'Layer':>5}  {'Name':<{_NAME_WIDTH}}  {'Status':<{_STATUS_WIDTH}}  "
        f"{'Key Metrics / Summary'}"
    )
    sep = "-" * (len(header) + 5)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        # Truncate summary for table display
        display_summary = r.summary
        if len(display_summary) > _SUMMARY_WIDTH:
            display_summary = display_summary[:_SUMMARY_WIDTH - 3] + "..."
        print(
            f"{r.layer:>5}  {r.name:<{_NAME_WIDTH}}  {r.status:<{_STATUS_WIDTH}}  "
            f"{display_summary}"
        )
        if r.status == "SKIP":
            print(
                f"{'':>5}  {'':>{_NAME_WIDTH}}  {'':>{_STATUS_WIDTH}}  "
                f"  [!] SKIP != PASS -- no health signal for this layer"
            )
    print(sep)


def print_comparison(comparison: dict) -> None:
    """Print baseline comparison table."""
    if not comparison:
        print("No baseline comparison available.")
        return
    print("\nBaseline comparison (signal metrics only):")
    print("-" * 60)
    for layer_name, metrics in comparison.items():
        print(f"  {layer_name}:")
        for metric, vals in metrics.items():
            before = vals.get("before")
            after = vals.get("after")
            delta = vals.get("delta")
            delta_str = f"  ({delta:+.4f})" if delta is not None else ""
            print(f"    {metric:<35} {before!s:>10}  ->  {after!s:>10}{delta_str}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unified 4-layer model improvement checker.\n"
            "SKIP != PASS: a SKIP layer provides no health signal."
        )
    )
    parser.add_argument(
        "--layer",
        default="all",
        choices=["all", "1", "2", "3", "4"],
        help="Which layer(s) to run (default: all)",
    )
    parser.add_argument(
        "--audit-dir",
        default=None,
        help=(
            "Directory containing forecast_audit_*.json files (Layer 1). "
            "Default resolution: TS_FORECAST_AUDIT_DIR -> active cohort production dir "
            "-> logs/forecast_audits/production -> logs/forecast_audits"
        ),
    )
    parser.add_argument(
        "--db",
        default=str(REPO_ROOT / "data" / "portfolio_maximizer.db"),
        help="Path to portfolio_maximizer.db (Layers 3 and 4)",
    )
    parser.add_argument(
        "--jsonl-path",
        default=str(REPO_ROOT / "logs" / "signals" / "quant_validation.jsonl"),
        help="Path to quant_validation.jsonl (Layer 4)",
    )
    parser.add_argument("--save-baseline", default=None, help="Save results to baseline JSON")
    parser.add_argument("--baseline", default=None, help="Compare to prior baseline JSON")
    parser.add_argument("--json", dest="json_output", action="store_true", help="JSON output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    layers_to_run: list[int] = [1, 2, 3, 4] if args.layer == "all" else [int(args.layer)]
    audit_dir = _resolve_layer1_audit_dir(args.audit_dir)
    db_path = Path(args.db)
    jsonl_path = Path(args.jsonl_path)

    results: list[LayerResult] = []
    try:
        for layer in layers_to_run:
            if layer == 1:
                results.append(run_layer1_forecast_quality(audit_dir))
            elif layer == 2:
                results.append(run_layer2_gate_status())
            elif layer == 3:
                # THR-03 fix: wire win_rate_warn to quant_success_config.yml
                # min_directional_accuracy so Layer 3 and quant_validation_health agree.
                win_rate_warn = 0.45  # default; overridden by config if present
                try:
                    import yaml as _yaml  # type: ignore[import]
                    _qs_path = REPO_ROOT / "config" / "quant_success_config.yml"
                    if _qs_path.exists():
                        _qs_cfg = _yaml.safe_load(_qs_path.read_text(encoding="utf-8")) or {}
                        win_rate_warn = float(
                            _qs_cfg.get("quant_success", {})
                            .get("quant_validation", {})
                            .get("min_directional_accuracy", win_rate_warn)
                        )
                except Exception:
                    pass
                results.append(run_layer3_trade_quality(db_path, win_rate_warn=win_rate_warn))
            elif layer == 4:
                results.append(run_layer4_calibration(db_path, jsonl_path))
    except Exception as exc:
        log.error("Runtime error in measurement layers: %s", exc, exc_info=True)
        if args.json_output:
            print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    if args.json_output:
        output = {
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "results": [asdict(r) for r in results],
        }
        if args.baseline:
            output["comparison"] = compare_baseline(results, Path(args.baseline))
        print(json.dumps(_json_safe(output), indent=2, default=str, allow_nan=False))
    else:
        print_summary_table(results)
        if args.baseline:
            comparison = compare_baseline(results, Path(args.baseline))
            print_comparison(comparison)

    if args.save_baseline:
        save_baseline(results, Path(args.save_baseline))

    # Exit 1 if any layer is FAIL
    has_fail = any(r.status == "FAIL" for r in results)
    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
