#!/usr/bin/env python3
from __future__ import annotations
# Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError for arrow chars)
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
"""
adversarial_diagnostic_runner.py -- Comprehensive adversarial verification CLI.

Surfaces implementation short-circuits, threshold dodges, gate bypass vectors,
data contamination, and metric wiring inconsistencies across the 4-layer
diagnostic stack.

USAGE
-----
    python scripts/adversarial_diagnostic_runner.py
    python scripts/adversarial_diagnostic_runner.py --db data/portfolio_maximizer.db
    python scripts/adversarial_diagnostic_runner.py --json
    python scripts/adversarial_diagnostic_runner.py --severity CRITICAL
    python scripts/adversarial_diagnostic_runner.py --audit-dir logs/forecast_audits
    python scripts/adversarial_diagnostic_runner.py --category integrity
    python scripts/adversarial_diagnostic_runner.py --fix-report

EXIT CODES
----------
    0   no CRITICAL or HIGH findings
    1   one or more CRITICAL or HIGH findings
    2   runtime error

FINDING CATEGORIES
------------------
    integrity    -- PnL canonical view contamination, NULL flag bypass
    bypass       -- Gate skip abuse, exit-code/JSON field mismatch
    leakage      -- Data lookahead, calibration train/test confusion
    threshold    -- Threshold proximity, gradual-drift blindspots
    wiring       -- Metric definition drift, config mismatch
    poisoning    -- Audit file integrity, order cache injection

Phase 7.20 adversarial audit findings (2026-03-01):
    16 CRITICAL / HIGH issues across 3 audit domains.
"""

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
ALL_CATEGORIES = {"integrity", "bypass", "leakage", "threshold", "wiring", "poisoning"}


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    id: str
    severity: str           # CRITICAL | HIGH | MEDIUM | LOW
    category: str           # one of ALL_CATEGORIES
    location: str           # file:line
    title: str
    detail: str
    attack_vector: str
    fix: str
    passed: bool = False    # True = check OK; False = vulnerability confirmed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _db_connect(db_path: Path) -> Optional[sqlite3.Connection]:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=3.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _db_scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Optional[int]:
    try:
        row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return None


def _count_audit_files(audit_dir: Path) -> tuple[int, int]:
    """Return (total_files, usable_files_with_ensemble_metrics)."""
    if not audit_dir.exists():
        return 0, 0
    files = list(audit_dir.glob("forecast_audit_*.json"))
    total = len(files)
    usable = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            arts = data.get("artifacts", {})
            eval_m = arts.get("evaluation_metrics", {})
            if eval_m and data.get("artifacts", {}).get("ensemble_rmse") is not None:
                usable += 1
        except Exception:
            pass
    return total, usable


def _read_config(relative_path: str) -> dict:
    try:
        import yaml  # type: ignore[import]
        p = ROOT / relative_path
        if p.exists():
            with p.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


# ===========================================================================
# CATEGORY: integrity
# ===========================================================================

def chk_null_flag_bypass(conn: Optional[sqlite3.Connection]) -> Finding:
    """A2: COALESCE(is_diagnostic, 0) = 0 treats NULL as non-diagnostic."""
    f = Finding(
        id="INT-01",
        severity="CRITICAL",
        category="integrity",
        location="integrity/pnl_integrity_enforcer.py:231",
        title="NULL is_diagnostic/is_synthetic treated as non-diagnostic (bypass)",
        detail=(
            "get_canonical_metrics() uses COALESCE(is_diagnostic, 0) = 0 which "
            "interprets NULL as 0 (not diagnostic). Trades inserted without explicit "
            "is_diagnostic=0 contaminate production PnL metrics."
        ),
        attack_vector=(
            "Insert trade with is_diagnostic=NULL, is_synthetic=NULL. "
            "get_canonical_metrics() includes it in PnL computation."
        ),
        fix=(
            "Replace COALESCE(is_diagnostic, 0) = 0 with "
            "is_diagnostic = 0 (reject NULL). "
            "Same for is_synthetic. Requires explicit NULL-rejection in the view."
        ),
    )
    if conn is None:
        f.detail = "DB not found -- cannot check. Assume VULNERABLE."
        return f
    try:
        null_diag = _db_scalar(
            conn,
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close=1 AND (is_diagnostic IS NULL OR is_synthetic IS NULL)",
        )
        if null_diag and null_diag > 0:
            f.detail += f" CONFIRMED: {null_diag} trade(s) have NULL is_diagnostic or is_synthetic."
        else:
            f.passed = True
            f.detail = "No NULL-flagged production closes found in DB. Risk is architectural."
    except Exception as exc:
        f.detail += f" [Error checking DB: {exc}]"
    return f


def chk_duplicate_close_null_bypass(conn: Optional[sqlite3.Connection]) -> Finding:
    """A3: DUPLICATE_CLOSE detection bypassed when entry_trade_id IS NULL."""
    f = Finding(
        id="INT-02",
        severity="CRITICAL",
        category="integrity",
        location="integrity/pnl_integrity_enforcer.py:619",
        title="DUPLICATE_CLOSE detection requires entry_trade_id (bypassed when NULL)",
        detail=(
            "DUPLICATE_CLOSE_FOR_ENTRY check uses JOIN ON c.entry_trade_id = o.id. "
            "Two closing legs with entry_trade_id IS NULL cannot be detected as "
            "over-closing the same position. Double-counting is invisible."
        ),
        attack_vector=(
            "Create BUY (id=100), then two SELL closes both with entry_trade_id=NULL. "
            "Both closes are counted as independent PnL; position over-closed with 0 detection."
        ),
        fix=(
            "Add a second query: count unlinked closes (entry_trade_id IS NULL) per ticker+date "
            "where 2+ closes exist for a single opening-day bar. "
            "Promote CLOSE_WITHOUT_ENTRY_LINK to CRITICAL severity."
        ),
    )
    if conn is None:
        f.detail += " DB not found."
        return f
    try:
        # Check how many NULL-linked closes exist
        n_unlinked = _db_scalar(
            conn,
            "SELECT COUNT(*) FROM trade_executions WHERE is_close=1 AND entry_trade_id IS NULL "
            "AND COALESCE(is_diagnostic,0)=0 AND COALESCE(is_synthetic,0)=0",
        )
        enforceable_threshold = 0
        if n_unlinked and n_unlinked > enforceable_threshold:
            f.detail += f" ACTIVE EXPOSURE: {n_unlinked} unlinked close(s) in production view."
        else:
            f.passed = True
            f.detail += " No unlinked closes in current DB. Structural risk remains."
    except Exception as exc:
        f.detail += f" [DB error: {exc}]"
    return f


def chk_proof_raw_table(db_path: Path) -> Finding:
    """validate_profitability_proof.py queries raw trade_executions (not canonical view)."""
    f = Finding(
        id="INT-03",
        severity="CRITICAL",
        category="integrity",
        location="scripts/validate_profitability_proof.py:92-148",
        title="Profitability proof queries raw trade_executions, not production_closed_trades",
        detail=(
            "get_trade_stats() and calculate_win_rate() query trade_executions directly "
            "without filtering is_close=1, is_diagnostic=0, is_synthetic=0. "
            "Opening legs, diagnostic trades, and synthetic trades all contaminate "
            "the profitability proof."
        ),
        attack_vector=(
            "Add synthetic=1 trades with large positive PnL. "
            "calculate_win_rate() includes them, inflating win_rate and profit_factor. "
            "Gate passes falsely."
        ),
        fix=(
            "Replace get_trade_stats() and calculate_win_rate() with "
            "PnLIntegrityEnforcer.get_canonical_metrics(). "
            "Use production_closed_trades view exclusively."
        ),
    )
    src = _read(ROOT / "scripts" / "validate_profitability_proof.py")
    if "production_closed_trades" in src:
        f.passed = True
        f.detail = "production_closed_trades view referenced -- may be partially fixed."
    elif "COALESCE(is_diagnostic" in src or "is_close = 1" in src:
        f.passed = False
        f.detail += " Source queries trade_executions with partial filters only."
    else:
        f.detail += " Source queries raw trade_executions with NO integrity filters."
    return f


def chk_orphan_shorts(src_enforcer: str) -> Finding:
    """A4: Orphan position check only looks at BUY opens, not SELL (shorts)."""
    f = Finding(
        id="INT-04",
        severity="HIGH",
        category="integrity",
        location="integrity/pnl_integrity_enforcer.py:362",
        title="Orphan position detection excludes SHORT positions (action=SELL, is_close=0)",
        detail=(
            "The ORPHANED_POSITION check queries WHERE action='BUY' AND is_close=0. "
            "SELL opens (short positions) are never checked for stale age. "
            "Unlimited stale short exposure accumulates without detection."
        ),
        attack_vector=(
            "Open a SELL position, never cover it. "
            "run_full_integrity_audit() reports 0 orphaned positions. "
            "The position ages for 60+ days undetected."
        ),
        fix=(
            "Add parallel check for action='SELL' AND is_close=0 (short orphans). "
            "Apply the same orphan_threshold_days logic."
        ),
    )
    if "action = 'BUY' AND is_close = 0" in src_enforcer and \
       "action = 'SELL' AND is_close = 0" not in src_enforcer:
        f.passed = False
        f.detail += " CONFIRMED: orphan check is BUY-only in source."
    else:
        f.passed = True
        f.detail = "SELL orphan check found or not BUY-only -- may be addressed."
    return f


def chk_medium_violations_in_ci_gate(src_ci: str) -> Finding:
    """A1 (agent 3): MEDIUM violations pass silently in default (non-strict) mode."""
    f = Finding(
        id="INT-05",
        severity="HIGH",
        category="integrity",
        location="scripts/ci_integrity_gate.py:73",
        title="MEDIUM integrity violations (CLOSE_WITHOUT_ENTRY_LINK) pass in default mode",
        detail=(
            "ci_integrity_gate.py default fail_severities={CRITICAL, HIGH}. "
            "MEDIUM violations (unlinked closes) never block CI. "
            "100+ unlinked closes report [PASS]."
        ),
        attack_vector=(
            "Accumulate unlinked closes (MEDIUM severity). "
            "ci_integrity_gate exits 0 (pass). "
            "run_all_gates reports overall_passed=True."
        ),
        fix=(
            "Add warning count to JSON output. "
            "Fail if MEDIUM violations > configurable limit (e.g., 10). "
            "Or: promote CLOSE_WITHOUT_ENTRY_LINK to HIGH when count > threshold."
        ),
    )
    if 'fail_severities = {"CRITICAL", "HIGH"}' in src_ci or \
       "fail_severities.add" in src_ci:
        f.passed = False
        f.detail += " CONFIRMED: default excludes MEDIUM from blocking set."
    else:
        f.passed = True
    return f


# ===========================================================================
# CATEGORY: bypass
# ===========================================================================

def chk_gate_skip_bypass(src_run_all: str) -> Finding:
    """A2 (agent 1): all 3 optional gates skippable → overall_passed=True with 1/4 gates."""
    f = Finding(
        id="BYP-01",
        severity="CRITICAL",
        category="bypass",
        location="scripts/run_all_gates.py:131-170",
        title="All optional gates skippable simultaneously; overall_passed=True with 1/4 gates",
        detail=(
            "--skip-forecast-gate, --skip-profitability-gate, --skip-institutional-gate "
            "can be combined. Skipped gates are appended with passed=True. "
            "Only ci_integrity_gate (Gate 1) runs; system reports overall_passed=True."
        ),
        attack_vector=(
            "python scripts/run_all_gates.py "
            "--skip-forecast-gate --skip-profitability-gate --skip-institutional-gate\n"
            "→ overall_passed=True with 1/4 gates executed."
        ),
        fix=(
            "Require minimum 3/4 gates to actually run (not skipped). "
            "If more than 1 optional gate is skipped, set overall_passed=False. "
            "Document allowed-skip combinations in comments."
        ),
    )
    skip_flags = [
        "--skip-forecast-gate" in src_run_all,
        "--skip-profitability-gate" in src_run_all,
        "--skip-institutional-gate" in src_run_all,
    ]
    has_all_skip_flags = all(skip_flags)
    # Phase 7.22 enforcement: MAX_SKIPPED_OPTIONAL_GATES = 1 sets overall_passed = False
    # when more than 1 optional gate is skipped, preventing the all-skip bypass.
    has_enforcement = (
        "MAX_SKIPPED_OPTIONAL_GATES" in src_run_all
        and (
            "overall_pass = False" in src_run_all
            or "overall_passed = False" in src_run_all
        )
    )
    if not has_all_skip_flags:
        f.passed = True  # CLEARED: skip flags removed from source
    elif has_enforcement:
        f.passed = True  # CLEARED: MAX_SKIPPED_OPTIONAL_GATES enforcement prevents all-skip bypass
        f.detail += (
            " CLEARED: MAX_SKIPPED_OPTIONAL_GATES enforcement is present and sets "
            "overall_passed=False when > 1 optional gate is skipped."
        )
    else:
        f.passed = False  # CONFIRMED: all 3 skip flags present with no enforcement guard
        f.detail += " CONFIRMED: all 3 skip flags exist with no enforcement to prevent bypass."
    return f


def chk_layer2_exit_code_ignored(src_cmi: str) -> Finding:
    """A1/C2 (agent 1): Layer 2 trusts JSON field exclusively; subprocess exit code ignored."""
    f = Finding(
        id="BYP-02",
        severity="CRITICAL",
        category="bypass",
        location="scripts/check_model_improvement.py:387",
        title="Layer 2 ignores subprocess exit code; JSON overall_passed field trusted exclusively",
        detail=(
            "run_layer2_gate_status() calls run_all_gates.py --json. "
            "subprocess exit code is captured but NEVER checked. "
            "A subprocess that exits 1 but outputs {'overall_passed': true} → Layer 2 PASS."
        ),
        attack_vector=(
            "Craft a wrapper around run_all_gates.py that exits 1 but emits "
            "JSON with overall_passed=true. Layer 2 reports PASS."
        ),
        fix=(
            "After subprocess call: if result.returncode != 0 and not overall_passed_from_json:\n"
            "    return FAIL\n"
            "Trust BOTH exit code AND JSON field. Discrepancy → FAIL."
        ),
    )
    # Check if returncode is checked anywhere near overall_passed
    has_returncode_check = "returncode" in src_cmi and "overall_passed" in src_cmi
    has_combined_check = re.search(r"returncode.*overall_passed|overall_passed.*returncode",
                                   src_cmi, re.DOTALL)
    if has_returncode_check and not has_combined_check:
        f.passed = False
        f.detail += " CONFIRMED: returncode captured but not combined with overall_passed decision."
    elif not has_returncode_check:
        f.passed = False
        f.detail += " returncode not referenced near gate decision."
    else:
        f.passed = True
    return f


def chk_institutional_gate_doesnt_verify_prior_gates(src_inst: str) -> Finding:
    """B1 (agent 3): Institutional gate doesn't verify gates 2/3 passed."""
    f = Finding(
        id="BYP-03",
        severity="CRITICAL",
        category="bypass",
        location="scripts/institutional_unattended_gate.py:254",
        title="Institutional gate doesn't verify gates 2/3 executed or passed",
        detail=(
            "institutional_unattended_gate.py runs P0-P3 checks (source code patterns + "
            "Platt contract). It never checks whether production_audit_gate, "
            "check_quant_validation_health, or the nightly refresh actually ran and passed."
        ),
        attack_vector=(
            "Gates 2 and 3 fail at 8 AM. At noon, run institutional_unattended_gate.py. "
            "It passes P0-P3 (static checks). Overall health appears GREEN. "
            "Operator misses the earlier failures."
        ),
        fix=(
            "Add P4 phase: read logs/audit_gate/production_gate_latest.json and "
            "logs/overnight_refresh.log; verify overall_passed=True and timestamp is recent. "
            "FAIL if last successful gate run is > 26 hours ago."
        ),
    )
    checks_prior_gate_result = (
        "production_gate_latest" in src_inst or
        "run_all_gates" in src_inst or
        "overall_passed" in src_inst
    )
    if not checks_prior_gate_result:
        f.passed = False
        f.detail += " CONFIRMED: no reference to prior gate results in source."
    else:
        f.passed = True
        f.detail = "Prior gate result reference found -- partially addressed."
    return f


def chk_overnight_exit_code(src_refresh: str) -> Finding:
    """G1 (agent 3): run_overnight_refresh.py returns error COUNT, not boolean 0/1."""
    f = Finding(
        id="BYP-04",
        severity="CRITICAL",
        category="bypass",
        location="scripts/run_overnight_refresh.py:630",
        title="run_overnight_refresh.py exits with error COUNT, not boolean 0/1",
        detail=(
            "main() returns the `errors` counter (integer). sys.exit(main()) exits "
            "with code N (number of errors). Cron/CI interpreting exit code > 0 as 'failure' "
            "is correct, but code=1 (one error) is indistinguishable from code=1 (true success). "
            "Any value 1-255 ambiguously maps to 'failure'."
        ),
        attack_vector=(
            "3 errors → exit code 3. Orchestrator with 'if rc > 5: alert' threshold "
            "silently ignores 1-5 errors. 5 failing gates = no alert."
        ),
        fix="Change `return errors` to `return 0 if errors == 0 else 1`.",
    )
    # Look for `return errors` at end of main()
    if re.search(r"return\s+errors\s*$", src_refresh, re.MULTILINE):
        f.passed = False
        f.detail += " CONFIRMED: `return errors` found at end of main()."
    elif re.search(r"sys\.exit.*errors", src_refresh):
        f.passed = False
        f.detail += " CONFIRMED: sys.exit(errors) exits with error count."
    else:
        f.passed = True
    return f


def chk_allow_inconclusive_lift(src_run_all: str) -> Finding:
    """A3 (agent 1): --allow-inconclusive-lift converts INCONCLUSIVE into PASS."""
    f = Finding(
        id="BYP-05",
        severity="HIGH",
        category="bypass",
        location="scripts/run_all_gates.py:146 + scripts/production_audit_gate.py:704",
        title="--allow-inconclusive-lift permanently converts insufficient-evidence INCONCLUSIVE to PASS",
        detail=(
            "run_all_gates.py passes --allow-inconclusive-lift to production_audit_gate.py. "
            "This allows a gate state of 'insufficient audit data' to be treated as PASS "
            "indefinitely, with no auto-expiry or maximum warmup duration enforced."
        ),
        attack_vector=(
            "System stays in warmup (< 20 effective audits) forever if bootstrap doesn't run. "
            "Gate permanently reports PASS via INCONCLUSIVE bypass. "
            "No mechanism forces the gate to eventually make a real decision."
        ),
        fix=(
            "Add --allow-inconclusive-lift-until=<date> CLI arg. "
            "After the date, inconclusive is treated as FAIL. "
            "Alternatively: max_warmup_days config that hard-fails after N days."
        ),
    )
    if "--allow-inconclusive-lift" in src_run_all:
        # Check if there's a time-bound or expiry
        has_expiry = "until" in src_run_all or "expir" in src_run_all or "max_warmup" in src_run_all
        if not has_expiry:
            f.passed = False
            f.detail += " CONFIRMED: unconditional --allow-inconclusive-lift with no expiry."
        else:
            f.passed = True
    else:
        f.passed = True
    return f


# ===========================================================================
# CATEGORY: leakage
# ===========================================================================

def chk_platt_no_train_test_split(src_sig_gen: str) -> Finding:
    """D1 (agent 2): Platt LR trained and evaluated on same pairs -- no holdout."""
    f = Finding(
        id="LEAK-01",
        severity="CRITICAL",
        category="leakage",
        location="models/time_series_signal_generator.py:2264",
        title="Platt LogisticRegression has no train/test split -- trains and predicts on same data",
        detail=(
            "_calibrate_confidence() fits LogisticRegression on ALL outcome pairs "
            "(pairs_conf, pairs_win) and immediately predicts on the same distribution. "
            "No train/test split. Overfitting on injected pairs is undetected."
        ),
        attack_vector=(
            "Inject 30 trades with confidence=0.50, outcome=WIN into DB. "
            "Platt LR trains: 0.50 → P(win)=1.0. "
            "All future signals with raw_conf~0.50 receive calibrated=0.95+. "
            "Position sizing treats them as high-conviction."
        ),
        fix=(
            "Add train/test split (e.g., 70/30) before LogisticRegression.fit(). "
            "Gate: if test_accuracy < 0.55, fall back to raw_conf. "
            "This requires >= 43 pairs (30 for train, 13 for test at current threshold)."
        ),
    )
    # Check for train_test_split usage
    if "train_test_split" in src_sig_gen or "holdout" in src_sig_gen:
        f.passed = True
        f.detail = "train_test_split or holdout reference found."
    else:
        f.passed = False
        f.detail += " CONFIRMED: no train_test_split in Platt calibration path."
    return f


def chk_macro_bfill_lookahead(src_feature: str) -> Finding:
    """A1 (agent 2): macro context bfill fills past features with future values."""
    f = Finding(
        id="LEAK-02",
        severity="HIGH",
        category="leakage",
        location="etl/time_series_feature_builder.py:221",
        title="Macro context bfill() can fill past feature rows with future macro values",
        detail=(
            "Macro context columns (vix_level, yield_spread_10y_2y, sector_momentum_5d) "
            "are aligned with .ffill().bfill(). bfill() fills missing early rows "
            "with future macro values if macro_context extends beyond price_history cutoff."
        ),
        attack_vector=(
            "Provide macro_context with end_date > price_history.index[-1]. "
            "bfill() fills rows before price_history start with future macro values. "
            "Features used in training carry future macro information."
        ),
        fix=(
            "Clip macro_context to price_history date range before alignment:\n"
            "  macro_context = macro_context[macro_context.index <= df.index.max()]\n"
            "Then ffill() only (no bfill). Fill remaining NaN with 0."
        ),
    )
    has_bfill = ".bfill()" in src_feature or "bfill()" in src_feature
    has_macro = "macro_context" in src_feature or "MACRO_COLUMNS" in src_feature
    has_clip = "clip" in src_feature and "macro" in src_feature.lower()
    if has_bfill and has_macro and not has_clip:
        f.passed = False
        f.detail += " CONFIRMED: bfill() used on macro context without date clipping."
    elif has_macro and not has_bfill:
        f.passed = True
        f.detail = "Macro context present but no bfill() detected."
    else:
        f.passed = True
    return f


def chk_audit_file_no_validation(src_health_audit: str) -> Finding:
    """C1 (agent 2): audit JSON loaded without integrity validation -- poisoning attack."""
    f = Finding(
        id="LEAK-03",
        severity="HIGH",
        category="poisoning",
        location="scripts/ensemble_health_audit.py:70",
        title="Forecast audit JSON files loaded without HMAC/signature or range validation",
        detail=(
            "load_audit_windows() reads forecast_audit_*.json from disk with zero "
            "integrity checks. A crafted JSON with ensemble_rmse=0.001, "
            "best_single_rmse=999 inflates lift_fraction and triggers false "
            "adaptive weight rebalancing."
        ),
        attack_vector=(
            "Write `logs/forecast_audits/forecast_audit_20260301_999999.json` with "
            "crafted ensemble_rmse=0.001. lift_fraction inflated from 0.7% to near 100%. "
            "check_model_improvement Layer 1 reports PASS; ensemble weights rebalanced."
        ),
        fix=(
            "Add plausibility gates in extract_window_metrics():\n"
            "  - rmse must be in (0, 100)\n"
            "  - da must be in [0, 1]\n"
            "  - ensemble_rmse must be within 3x of best_single_rmse\n"
            "Log and skip files that fail plausibility checks."
        ),
    )
    # Check for range validation in the audit script
    has_range_check = (
        "rmse > 0" in src_health_audit or
        "rmse < 100" in src_health_audit or
        "plausib" in src_health_audit.lower()
    )
    if not has_range_check:
        f.passed = False
        f.detail += " CONFIRMED: no range validation found in ensemble_health_audit.py."
    else:
        f.passed = True
    return f


# ===========================================================================
# CATEGORY: threshold
# ===========================================================================

def chk_coverage_ratio_warn_not_fail(src_cmi: str) -> Finding:
    """B4 (agent 1): coverage_ratio < 0.20 triggers WARN only, not FAIL."""
    f = Finding(
        id="THR-01",
        severity="MEDIUM",
        category="threshold",
        location="scripts/check_model_improvement.py:303",
        title="coverage_ratio < 20% triggers WARN only -- 99% legacy audit files silently WARN",
        detail=(
            "With 5.7% coverage (138/2441 usable files), Layer 1 returns WARN, not FAIL. "
            "An operator seeing WARN may deprioritize the data accumulation problem. "
            "No escalation path from persistent WARN to FAIL after N days."
        ),
        attack_vector=(
            "System runs with legacy audit files indefinitely. "
            "coverage_ratio stays at 2%. Layer 1 WARN is ignored. "
            "Lift fraction computed from tiny non-representative sample."
        ),
        fix=(
            "Add: if coverage_ratio < 0.05 and n_used >= 50: status = 'FAIL'.\n"
            "Or: track coverage_ratio baseline; FAIL if it doesn't improve within 7 days."
        ),
    )
    # coverage_ratio < 0.20 should only WARN
    has_fail_on_low_coverage = re.search(
        r"coverage_ratio.*FAIL|FAIL.*coverage_ratio", src_cmi
    )
    if not has_fail_on_low_coverage:
        f.passed = False  # WARN only -- as documented
    else:
        f.passed = True
    return f


def chk_quant_health_warn_gap(audit_dir: Path) -> Finding:
    """B1 (agent 1): warn_fail_fraction=0.80 vs current ~27.7% -- 52pp headroom masks drift."""
    f = Finding(
        id="THR-02",
        severity="MEDIUM",
        category="threshold",
        location="config/forecaster_monitoring.yml:22",
        title="warn_fail_fraction=0.80 gives 52pp headroom from current rate -- gradual drift undetectable",
        detail=(
            "Current quant FAIL rate: ~27.7%. Warn threshold: 80%. Hard gate: 85%. "
            "System can drift to 79.9% FAIL with no alert. "
            "No trend-based alerting exists; only point-in-time threshold."
        ),
        attack_vector=(
            "Gradually degrade signal quality over 30 days. "
            "FAIL rate drifts 27% → 80% without any YELLOW alert. "
            "Then crosses 80% briefly → YELLOW. Operator notices."
        ),
        fix=(
            "Add rolling 7-day trend check: if FAIL rate increase > 10pp/week, "
            "trigger WARN regardless of absolute level. "
            "Log historical FAIL rate distribution for trend analysis."
        ),
    )
    cfg = _read_config("config/forecaster_monitoring.yml")
    qv = cfg.get("forecaster_monitoring", {}).get("quant_validation", {})
    warn_frac = qv.get("warn_fail_fraction", 0.80)
    max_frac = qv.get("max_fail_fraction", 0.85)
    if (max_frac - warn_frac) < 0.10:
        f.passed = True
        f.detail = f"warn/max gap = {max_frac - warn_frac:.0%} -- narrow band, acceptable."
    else:
        f.passed = False
        f.detail = (
            f"warn_fail_fraction={warn_frac:.0%}, max_fail_fraction={max_frac:.0%}. "
            f"Gap = {max_frac - warn_frac:.0%}. Drift of >{max_frac - warn_frac:.0%} "
            "without hitting YELLOW zone. No trend detection."
        )
    return f


def chk_layer3_threshold_not_from_config() -> Finding:
    """C1 (agent 1): Layer 3 win_rate_warn hardcoded, not read from quant_success_config."""
    src_cmi = _read(ROOT / "scripts" / "check_model_improvement.py")
    f = Finding(
        id="THR-03",
        severity="HIGH",
        category="wiring",
        location="scripts/check_model_improvement.py:413",
        title="Layer 3 win_rate_warn=0.45 hardcoded -- not wired to quant_success_config.yml",
        detail=(
            "run_layer3_trade_quality(win_rate_warn=0.45) default is hardcoded. "
            "quant_success_config.yml min_directional_accuracy is NOT read. "
            "If config changes, Layer 3 threshold stays stale. "
            "Layers 3 and quant_validation_health can DISAGREE on the same win_rate."
        ),
        attack_vector=(
            "Change quant_success_config.yml min_directional_accuracy to 0.55. "
            "check_quant_validation_health uses new 0.55 threshold (FAIL). "
            "Layer 3 still uses 0.45 (PASS). Same data, conflicting health signals."
        ),
        fix=(
            "In run_layer3_trade_quality(), load quant_success_config.yml and "
            "use quant_validation.min_directional_accuracy as win_rate_warn. "
            "Add config_path param with default."
        ),
    )
    cfg = _read_config("config/quant_success_config.yml")
    qs = cfg.get("quant_success", {}).get("quant_validation", {})
    config_da = qs.get("min_directional_accuracy", None)
    hardcoded = re.search(r"win_rate_warn\s*[:=]\s*0\.\d+", src_cmi)
    if config_da is not None and hardcoded:
        config_val = float(config_da)
        match = hardcoded.group(0)
        hardcoded_val = float(re.search(r"0\.\d+", match).group(0))  # type: ignore[union-attr]
        if abs(config_val - hardcoded_val) > 0.01:
            f.passed = False
            f.detail += (
                f" CONFIRMED: config min_directional_accuracy={config_val}, "
                f"Layer 3 hardcoded={hardcoded_val}. MISMATCH."
            )
        else:
            f.passed = True
            f.detail = f"Values match ({config_val:.2f} vs {hardcoded_val:.2f})."
    else:
        f.passed = False
        f.detail += " Unable to compare (config key not found or no hardcoded value)."
    return f


# ===========================================================================
# CATEGORY: wiring
# ===========================================================================

def chk_lift_computation_mismatch(src_cmi: str, src_gate: str) -> Finding:
    """D2 (agent 1): Layer 1 uses hardcoded <1.0; check_forecast_audits uses configurable threshold."""
    f = Finding(
        id="WIRE-01",
        severity="MEDIUM",
        category="wiring",
        location="scripts/check_model_improvement.py:256 vs scripts/check_forecast_audits.py:784",
        title="Layer 1 lift uses hardcoded rmse_ratio < 1.0; gate uses configurable min_lift_rmse_ratio",
        detail=(
            "check_model_improvement.py Layer 1 computes lift as rmse_ratio < 1.0 (hardcoded). "
            "check_forecast_audits.py computes lift as rmse_ratio < (1 - min_lift_rmse_ratio). "
            "With min_lift_rmse_ratio=0.00, both are equivalent. "
            "If min_lift_rmse_ratio is changed to 0.02 (2% required), Layer 1 and the gate disagree."
        ),
        attack_vector=(
            "Set min_lift_rmse_ratio=0.02 in forecaster_monitoring.yml. "
            "check_forecast_audits uses threshold=0.98 (2% improvement required). "
            "Layer 1 still uses threshold=1.0 (any improvement). "
            "System shows higher lift_fraction in Layer 1 than in the actual gate."
        ),
        fix=(
            "In run_layer1_forecast_quality(), read min_lift_rmse_ratio from "
            "forecaster_monitoring.yml and use lift_threshold = 1.0 - min_lift_rmse_ratio."
        ),
    )
    layer1_hardcoded = "< 1.0" in src_cmi and "rmse_ratio" in src_cmi
    gate_configurable = "min_lift_rmse_ratio" in src_gate
    if layer1_hardcoded and gate_configurable:
        f.passed = False
        f.detail += " CONFIRMED: Layer 1 hardcoded, gate configurable."
    else:
        f.passed = True
    return f


def chk_production_view_integrity(conn: Optional[sqlite3.Connection]) -> Finding:
    """Verify production_closed_trades view correctly filters all contamination sources."""
    f = Finding(
        id="WIRE-02",
        severity="HIGH",
        category="integrity",
        location="integrity/pnl_integrity_enforcer.py (view definition)",
        title="Verify production_closed_trades view excludes synthetic/diagnostic/open legs",
        detail=(
            "production_closed_trades view must enforce: is_close=1, "
            "COALESCE(is_diagnostic,0)=0, COALESCE(is_synthetic,0)=0. "
            "If the view definition is weaker, canonical metrics are contaminated."
        ),
        attack_vector=(
            "View missing is_synthetic filter → synthetic bootstrap trades inflate win_rate. "
            "View using COALESCE (NULL=ok) → NULL-flagged diagnostic trades contaminate PnL."
        ),
        fix=(
            "View definition should use explicit = 0 (not COALESCE) for diagnostic/synthetic. "
            "Add a column-level CHECK constraint on insert to prevent NULL flags."
        ),
    )
    if conn is None:
        f.detail += " DB not available."
        return f
    try:
        # Get view definition
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='production_closed_trades'"
        ).fetchone()
        if row is None:
            f.passed = False
            f.detail = "CRITICAL: production_closed_trades view does not exist in DB."
            return f
        view_sql = str(row[0])
        checks = {
            "is_close = 1": "is_close = 1" in view_sql or "is_close=1" in view_sql,
            "is_diagnostic filter": "is_diagnostic" in view_sql,
            "is_synthetic filter": "is_synthetic" in view_sql,
        }
        missing = [k for k, v in checks.items() if not v]
        if missing:
            f.passed = False
            f.detail = f"View missing filters: {missing}. View SQL: {view_sql[:200]}"
        else:
            f.passed = True
            f.detail = "View contains all required filters."
    except Exception as exc:
        f.detail += f" [DB error: {exc}]"
    return f


def chk_warmup_indefinite(src_gate: str) -> Finding:
    """A4 (agent 1): fail_on_violation_during_holding_period=False allows indefinite suppression."""
    f = Finding(
        id="WIRE-03",
        severity="HIGH",
        category="bypass",
        location="scripts/check_forecast_audits.py:736",
        title="fail_on_violation_during_holding_period=False + no max_warmup_days allows eternal warmup",
        detail=(
            "With fail_on_violation_during_holding_period=False, violation_rate is never "
            "enforced during the holding period. Combined with --allow-inconclusive-lift, "
            "the gate can stay INCONCLUSIVE indefinitely with no hard deadline."
        ),
        attack_vector=(
            "Bootstrap never runs (AUDIT_GATE_BOOTSTRAP=0). "
            "effective_n stays < 20. Gate stays INCONCLUSIVE forever. "
            "Passes via --allow-inconclusive-lift in run_all_gates.py. "
            "System never gets a real lift verdict."
        ),
        fix=(
            "Add max_warmup_days = 30 to forecaster_monitoring.yml. "
            "If now() - first_audit_date > max_warmup_days, treat INCONCLUSIVE as FAIL. "
            "Gate must eventually make a real decision."
        ),
    )
    cfg = _read_config("config/forecaster_monitoring.yml")
    rm = cfg.get("forecaster_monitoring", {}).get("regression_metrics", {})
    has_max_warmup = "max_warmup_days" in rm or "max_warmup" in str(rm)
    fail_on_viol = rm.get("fail_on_violation_during_holding_period", True)
    if not fail_on_viol and not has_max_warmup:
        f.passed = False
        f.detail += " CONFIRMED: fail_on_violation=False, no max_warmup_days in config."
    elif has_max_warmup:
        f.passed = True
    else:
        f.passed = False
    return f


# ===========================================================================
# CATEGORY: poisoning
# ===========================================================================

def chk_order_learner_aic_bounds(src_ol: str) -> Finding:
    """F1 (agent 2): extreme negative AIC can disable grid search."""
    f = Finding(
        id="POI-01",
        severity="MEDIUM",
        category="poisoning",
        location="forcester_ts/order_learner.py:160",
        title="OrderLearner AIC has no lower bound -- extreme negative AIC disables grid search",
        detail=(
            "record_fit() rejects non-finite AIC (nan/inf) but accepts any finite value. "
            "A row with best_aic=-999999 and n_fits >= skip_grid_threshold causes "
            "should_skip_grid() to return True. Grid search is disabled for the ticker."
        ),
        attack_vector=(
            "Insert row: (ticker='AAPL', model_type='GARCH', best_aic=-1e7, n_fits=100). "
            "Next run: suggest() returns the row's order_params (e.g., p=10, q=10). "
            "GARCH fits numerically unstable model without grid validation."
        ),
        fix=(
            "In record_fit(): if not math.isfinite(aic_val) or aic_val < -1e6:\n"
            "    logger.warning('Implausible AIC rejected: %r', aic_val); return\n"
            "AIC below -1e6 is physically implausible for financial time series."
        ),
    )
    has_lower_bound = re.search(r"aic.*-1e6|aic.*< -|lower.*bound", src_ol, re.IGNORECASE)
    has_finite_check = "isfinite" in src_ol and "aic" in src_ol
    if has_finite_check and not has_lower_bound:
        f.passed = False
        f.detail += " CONFIRMED: isfinite check present but no lower bound on AIC value."
    elif has_lower_bound:
        f.passed = True
    else:
        f.passed = False
    return f


def chk_whitelist_divergence(src_enforcer: str, src_prod_gate: str) -> Finding:
    """F1 (agent 3): CLOSE_WITHOUT_ENTRY_LINK whitelist in enforcer not in production audit gate."""
    f = Finding(
        id="POI-02",
        severity="HIGH",
        category="wiring",
        location="integrity/pnl_integrity_enforcer.py:539 vs scripts/production_audit_gate.py:76",
        title="Unlinked-close whitelist in integrity enforcer not reflected in production audit gate",
        detail=(
            "PnLIntegrityEnforcer reads INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS and "
            "excludes those IDs from CLOSE_WITHOUT_ENTRY_LINK violations. "
            "production_audit_gate._count_unlinked_closes() does NOT apply this whitelist. "
            "Enforcer reports 0 violations; gate reports N unlinked closes."
        ),
        attack_vector=(
            "Set INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS=66,75. "
            "ci_integrity_gate: 0 MEDIUM violations (whitelisted). "
            "production_audit_gate: 2 unlinked closes found → reconcile attempted. "
            "Reconcile conflicts with whitelisted intentional state."
        ),
        fix=(
            "production_audit_gate._count_unlinked_closes() should also read "
            "INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS and exclude those IDs. "
            "Or: unify via PnLIntegrityEnforcer.get_unlinked_closes()."
        ),
    )
    enforcer_has_whitelist = "WHITELIST" in src_enforcer.upper()
    gate_has_whitelist = "WHITELIST" in src_prod_gate.upper()
    if enforcer_has_whitelist and not gate_has_whitelist:
        f.passed = False
        f.detail += " CONFIRMED: whitelist in enforcer only, not in gate."
    elif enforcer_has_whitelist and gate_has_whitelist:
        f.passed = True
        f.detail = "Whitelist referenced in both -- likely aligned."
    else:
        f.passed = True
    return f


# ===========================================================================
# Runner
# ===========================================================================

def run_all_checks(
    db_path: Path,
    audit_dir: Path,
    severity_filter: Optional[str],
    category_filter: Optional[str],
) -> list[Finding]:
    findings: list[Finding] = []

    # Load source files once
    src_enforcer = _read(ROOT / "integrity" / "pnl_integrity_enforcer.py")
    src_cmi = _read(ROOT / "scripts" / "check_model_improvement.py")
    src_run_all = _read(ROOT / "scripts" / "run_all_gates.py")
    src_gate = _read(ROOT / "scripts" / "check_forecast_audits.py")
    src_prod_gate = _read(ROOT / "scripts" / "production_audit_gate.py")
    src_inst = _read(ROOT / "scripts" / "institutional_unattended_gate.py")
    src_refresh = _read(ROOT / "scripts" / "run_overnight_refresh.py")
    src_ci = _read(ROOT / "scripts" / "ci_integrity_gate.py")
    src_feature = _read(ROOT / "etl" / "time_series_feature_builder.py")
    src_health_audit = _read(ROOT / "scripts" / "ensemble_health_audit.py")
    src_sig_gen = _read(ROOT / "models" / "time_series_signal_generator.py")
    src_ol = _read(ROOT / "forcester_ts" / "order_learner.py")

    conn = _db_connect(db_path)

    # Run all checks
    findings += [
        # Integrity
        chk_null_flag_bypass(conn),
        chk_duplicate_close_null_bypass(conn),
        chk_proof_raw_table(db_path),
        chk_orphan_shorts(src_enforcer),
        chk_medium_violations_in_ci_gate(src_ci),
        chk_production_view_integrity(conn),
        # Bypass
        chk_gate_skip_bypass(src_run_all),
        chk_layer2_exit_code_ignored(src_cmi),
        chk_institutional_gate_doesnt_verify_prior_gates(src_inst),
        chk_overnight_exit_code(src_refresh),
        chk_allow_inconclusive_lift(src_run_all),
        chk_warmup_indefinite(src_gate),
        # Leakage / Poisoning
        chk_platt_no_train_test_split(src_sig_gen),
        chk_macro_bfill_lookahead(src_feature),
        chk_audit_file_no_validation(src_health_audit),
        chk_order_learner_aic_bounds(src_ol),
        chk_whitelist_divergence(src_enforcer, src_prod_gate),
        # Threshold
        chk_coverage_ratio_warn_not_fail(src_cmi),
        chk_quant_health_warn_gap(audit_dir),
        # Wiring
        chk_layer3_threshold_not_from_config(),
        chk_lift_computation_mismatch(src_cmi, src_gate),
    ]

    if conn:
        conn.close()

    # Apply filters
    if severity_filter:
        target = SEVERITY_ORDER.get(severity_filter.upper(), 9)
        findings = [f for f in findings if SEVERITY_ORDER.get(f.severity, 9) <= target]
    if category_filter:
        findings = [f for f in findings if f.category == category_filter.lower()]

    # Sort: confirmed (not passed) first, then by severity
    findings.sort(key=lambda f: (f.passed, SEVERITY_ORDER.get(f.severity, 9)))
    return findings


def _print_table(findings: list[Finding]) -> None:
    confirmed = [f for f in findings if not f.passed]
    cleared = [f for f in findings if f.passed]

    print(f"\n{'='*80}")
    print("ADVERSARIAL DIAGNOSTIC RUNNER -- Portfolio Maximizer v45")
    print(f"Phase 7.20 | {len(confirmed)} confirmed findings | {len(cleared)} cleared")
    print(f"{'='*80}\n")

    if confirmed:
        print(f"CONFIRMED VULNERABILITIES ({len(confirmed)}):")
        print(f"{'-'*80}")
        for f in confirmed:
            sev_tag = f"[{f.severity}]"
            print(f"\n{sev_tag:<12} {f.id} | {f.category.upper()}")
            print(f"Location : {f.location}")
            print(f"Title    : {f.title}")
            print(f"Detail   : {f.detail}")
            print(f"Vector   : {f.attack_vector}")
            print(f"Fix      : {f.fix}")

    if cleared:
        print(f"\n{'-'*80}")
        print(f"CLEARED CHECKS ({len(cleared)}):")
        for f in cleared:
            print(f"  [OK] {f.id} {f.title[:60]}")

    # Summary counts
    counts: dict[str, int] = {}
    for f in confirmed:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    print(f"\n{'-'*80}")
    print("SUMMARY:")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        n = counts.get(sev, 0)
        if n:
            print(f"  {sev:<10}: {n} confirmed")
    if not confirmed:
        print("  No confirmed vulnerabilities. System adversarially clean.")
    print()

    print("SKIP != PASS. Run with --json for machine-readable output.")
    print("Run with --fix-report for actionable remediation steps.")
    print(f"{'='*80}\n")


def _print_fix_report(findings: list[Finding]) -> None:
    confirmed = [f for f in findings if not f.passed]
    if not confirmed:
        print("No confirmed findings -- no fixes required.")
        return
    print(f"\n{'='*80}")
    print("ADVERSARIAL FIX REPORT")
    print(f"{'='*80}\n")
    for i, f in enumerate(confirmed, 1):
        print(f"[{i}/{len(confirmed)}] {f.id} | {f.severity} | {f.category.upper()}")
        print(f"File     : {f.location}")
        print(f"Problem  : {f.title}")
        print(f"Fix      : {f.fix}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="data/portfolio_maximizer.db",
                        help="Path to SQLite database (default: data/portfolio_maximizer.db)")
    parser.add_argument("--audit-dir", default="logs/forecast_audits",
                        help="Path to forecast audit directory")
    parser.add_argument("--json", action="store_true", dest="emit_json",
                        help="Emit JSON report to stdout")
    parser.add_argument("--severity", default=None,
                        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        help="Only show findings at or above this severity")
    parser.add_argument("--category", default=None,
                        choices=sorted(ALL_CATEGORIES),
                        help="Filter to a single finding category")
    parser.add_argument("--fix-report", action="store_true",
                        help="Emit a concise actionable fix report")
    args = parser.parse_args()

    db_path = ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
    audit_dir = ROOT / args.audit_dir if not Path(args.audit_dir).is_absolute() else Path(args.audit_dir)

    try:
        findings = run_all_checks(db_path, audit_dir, args.severity, args.category)
    except Exception as exc:
        print(f"[ERROR] Runtime error during adversarial checks: {exc}", file=sys.stderr)
        return 2

    if args.emit_json:
        import datetime
        report = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": "adversarial_audit_7.20",
            "db_path": str(db_path),
            "total_checks": len(findings),
            "confirmed": sum(1 for f in findings if not f.passed),
            "cleared": sum(1 for f in findings if f.passed),
            "findings": [asdict(f) for f in findings],
        }
        print(json.dumps(report, indent=2))
    elif args.fix_report:
        _print_fix_report(findings)
    else:
        _print_table(findings)

    # Exit code: 1 if any CRITICAL or HIGH confirmed
    blocking = [f for f in findings if not f.passed and f.severity in ("CRITICAL", "HIGH")]
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
