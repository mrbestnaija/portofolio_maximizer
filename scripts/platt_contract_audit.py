"""
platt_contract_audit.py -- Platt calibration contract audit.

Validates the implementation contract of the Platt confidence calibrator so that
implementation drift (wrong classifier, wrong fallback order, broken bootstrap) is
caught immediately rather than silently producing bad calibration.

Checks:
  1. classifier_identity      -- LogisticRegression (not IsotonicRegression or isotonic)
  2. fallback_chain_order     -- JSONL -> DB-local -> DB-global (priority order)
  3. hold_inflation           -- HOLD entries must not inflate pending/starvation counts
  4. ts_closes_in_db          -- ts_* closed trades exist (bootstrap produced output)
  5. calibration_active_tier  -- which tier is currently active given real data

Exit codes:
  0  All checks PASS (WARN is acceptable)
  1  One or more FAIL findings
  2  Runtime error (import failure, unexpected crash)

Usage:
  python scripts/platt_contract_audit.py
  python scripts/platt_contract_audit.py --db data/other.db --jsonl logs/signals/qv.jsonl
  python scripts/platt_contract_audit.py --json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_CLASSIFIER = "LogisticRegression"
FORBIDDEN_CLASSIFIERS = ["IsotonicRegression", "CalibratedClassifierCV", "isotonic"]
PLATT_MIN_PAIRS = 30


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

class Finding:
    __slots__ = ("check", "status", "detail")

    def __init__(self, check: str, status: str, detail: str) -> None:
        self.check = check
        self.status = status   # PASS | FAIL | WARN | SKIP
        self.detail = detail

    def as_dict(self) -> Dict[str, str]:
        return {"check": self.check, "status": self.status, "detail": self.detail}


# ---------------------------------------------------------------------------
# Check 1: Classifier identity
# ---------------------------------------------------------------------------

def check_classifier_identity() -> Finding:
    """Assert _calibrate_confidence uses LogisticRegression, not isotonic or other."""
    try:
        from models.time_series_signal_generator import TimeSeriesSignalGenerator  # noqa: PLC0415
        source = inspect.getsource(TimeSeriesSignalGenerator._calibrate_confidence)
    except Exception as exc:
        return Finding("classifier_identity", "FAIL", f"Cannot inspect source: {exc}")

    if EXPECTED_CLASSIFIER not in source:
        return Finding(
            "classifier_identity",
            "FAIL",
            f"'{EXPECTED_CLASSIFIER}' not found in _calibrate_confidence. "
            f"Implementation changed -- update docs and this audit.",
        )

    source_lower = source.lower()
    for forbidden in FORBIDDEN_CLASSIFIERS:
        forbidden_norm = forbidden.lower()
        if forbidden_norm in source_lower:
            return Finding(
                "classifier_identity",
                "FAIL",
                f"Forbidden classifier token '{forbidden}' found in _calibrate_confidence. "
                f"If intentional, update PHASE_7.14_GATE_RECALIBRATION.md and AGENTS.md.",
            )

    return Finding(
        "classifier_identity",
        "PASS",
        f"_calibrate_confidence uses {EXPECTED_CLASSIFIER} (classic Platt scaling).",
    )


# ---------------------------------------------------------------------------
# Check 2: Fallback chain order (JSONL -> DB-local -> DB-global)
# ---------------------------------------------------------------------------

def check_fallback_chain_order() -> Finding:
    """Assert fallback chain priority: JSONL before DB-local before DB-global."""
    try:
        from models.time_series_signal_generator import TimeSeriesSignalGenerator  # noqa: PLC0415
        source = inspect.getsource(TimeSeriesSignalGenerator._calibrate_confidence)
    except Exception as exc:
        return Finding("fallback_chain_order", "FAIL", f"Cannot inspect source: {exc}")

    jsonl_pos = source.find("_load_jsonl_outcome_pairs")
    db_local_pos = source.find("db_local")
    db_global_pos = source.find("db_global")

    if any(p < 0 for p in (jsonl_pos, db_local_pos, db_global_pos)):
        return Finding(
            "fallback_chain_order",
            "FAIL",
            f"Tier markers not found -- jsonl={jsonl_pos}, db_local={db_local_pos}, "
            f"db_global={db_global_pos}. Fallback chain refactored without updating audit.",
        )

    if not (jsonl_pos < db_local_pos < db_global_pos):
        return Finding(
            "fallback_chain_order",
            "FAIL",
            f"Tier order wrong: jsonl_pos={jsonl_pos}, db_local_pos={db_local_pos}, "
            f"db_global_pos={db_global_pos}. Expected jsonl < db_local < db_global.",
        )

    return Finding(
        "fallback_chain_order",
        "PASS",
        "Priority order JSONL -> DB-local -> DB-global confirmed.",
    )


# ---------------------------------------------------------------------------
# Check 3: HOLD entries do not inflate starvation metrics
# ---------------------------------------------------------------------------

def check_hold_inflation(jsonl_path: Path) -> Finding:
    """Report HOLD fraction of pending entries.

    HOLD signals cannot produce is_close=1 trades. If reconciler counts them as
    pending, starvation figures are structurally inflated and misleading.
    """
    if not jsonl_path.exists():
        return Finding("hold_inflation", "SKIP", f"JSONL not found: {jsonl_path}")

    try:
        raw_lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        entries = [json.loads(l) for l in raw_lines]
    except Exception as exc:
        return Finding("hold_inflation", "FAIL", f"Cannot parse JSONL: {exc}")

    total = len(entries)
    if total == 0:
        return Finding("hold_inflation", "SKIP", "JSONL is empty.")

    pending = [e for e in entries if e.get("outcome") is None]
    hold_pending = [e for e in pending if str(e.get("action", "")).upper() == "HOLD"]
    n_pending = len(pending)
    n_hold = len(hold_pending)
    hold_pct = (n_hold / n_pending * 100) if n_pending > 0 else 0.0

    if hold_pct > 30:
        return Finding(
            "hold_inflation",
            "WARN",
            f"{n_hold}/{n_pending} pending entries ({hold_pct:.1f}%) are HOLD action -- "
            f"structurally unreconcilable. Reconciler should filter to BUY/SELL only so "
            f"still_pending reflects only actionable candidates.",
        )

    return Finding(
        "hold_inflation",
        "PASS",
        f"HOLD entries are {hold_pct:.1f}% of pending ({n_hold}/{n_pending}). "
        f"Total JSONL entries: {total}.",
    )


# ---------------------------------------------------------------------------
# Check 4: ts_* closed trades exist in DB
# ---------------------------------------------------------------------------

def check_ts_closes_in_db(db_path: Path) -> Finding:
    """Verify bootstrap produced ts_* closed trades.

    If bootstrap ran but produced 0 ts_* closed trades, the JSONL Platt path can
    never accumulate pairs from bootstrap runs. Root cause: cycles-vs-bars mismatch
    where proof-mode max_holding never fires when bar date doesn't advance.
    """
    if not db_path.exists():
        return Finding(
            "ts_closes_in_db",
            "WARN",
            f"DB not found (CI/fresh environment — no bootstrap data): {db_path}",
        )

    try:
        conn = sqlite3.connect(str(db_path), timeout=3.0)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE ts_signal_id LIKE 'ts_%' AND is_close = 1 AND realized_pnl IS NOT NULL"
        )
        ts_closes = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE ts_signal_id LIKE 'legacy_%' AND is_close = 1"
        )
        legacy_closes = cur.fetchone()[0]
        conn.close()
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return Finding(
                "ts_closes_in_db",
                "WARN",
                f"DB present but schema not initialised (bootstrap not run yet): {exc}",
            )
        return Finding("ts_closes_in_db", "FAIL", f"DB query failed: {exc}")

    if ts_closes == 0:
        if legacy_closes > 0:
            return Finding(
                "ts_closes_in_db",
                "WARN",
                f"0 ts_* closed trades in DB. Bootstrap never produced closes -- "
                f"likely cycles-vs-bars mismatch (bar date does not advance across --cycles N "
                f"at a fixed --as-of-date, so max_holding never fires). "
                f"DB calibration tier uses {legacy_closes} legacy_* closes as fallback.",
            )
        return Finding(
            "ts_closes_in_db",
            "FAIL",
            "0 ts_* AND 0 legacy_* closed trades in DB. Calibration cannot fit any tier.",
        )

    return Finding(
        "ts_closes_in_db",
        "PASS",
        f"{ts_closes} ts_* closed trades in DB (+ {legacy_closes} legacy_*).",
    )


# ---------------------------------------------------------------------------
# Check 5: Active calibration tier
# ---------------------------------------------------------------------------

def _count_jsonl_outcome_pairs(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0

    pairs = 0
    try:
        lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return 0

    for line in lines:
        try:
            entry = json.loads(line)
        except Exception:
            continue
        action = str(entry.get("action", "")).upper()
        if action not in {"BUY", "SELL"}:
            continue
        if entry.get("outcome") is None:
            continue
        pairs += 1
    return pairs


def check_calibration_active_tier(db_path: Path, jsonl_path: Path) -> Finding:
    """Report which fallback tier is active given current data volumes."""
    jsonl_pairs = _count_jsonl_outcome_pairs(jsonl_path)

    db_pairs = 0
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=3.0)
            cur = conn.cursor()
            # Must exactly mirror _load_realized_outcome_pairs query so the reported
            # pair count matches what LogisticRegression actually trains on.
            # Only closing legs (is_close=1) are eligible — opening legs have no
            # settled PnL outcome and were never part of calibration training data.
            cols = {row[1] for row in cur.execute("PRAGMA table_info(trade_executions)").fetchall()}
            where = [
                "realized_pnl IS NOT NULL",
                "action IN ('BUY', 'SELL')",
                "COALESCE(confidence_calibrated, effective_confidence, base_confidence) IS NOT NULL",
                "is_close = 1",
                "is_diagnostic = 0",
                "is_synthetic = 0",
            ]
            if "is_contaminated" in cols:
                where.append("is_contaminated = 0")
            cur.execute("SELECT COUNT(*) FROM trade_executions WHERE " + " AND ".join(where))
            db_pairs = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass

    if jsonl_pairs >= PLATT_MIN_PAIRS:
        return Finding(
            "calibration_active_tier",
            "PASS",
            f"[TIER_1_JSONL] Active: {jsonl_pairs} JSONL outcome pairs.",
        )
    if db_pairs >= PLATT_MIN_PAIRS:
        return Finding(
            "calibration_active_tier",
            "PASS",
            f"[TIER_3_DB_GLOBAL] Active: {db_pairs} DB pairs (JSONL only {jsonl_pairs}). "
            f"DB is the correct primary source -- JSONL Platt is supplementary.",
        )
    if db_pairs >= 10:
        return Finding(
            "calibration_active_tier",
            "WARN",
            f"[TIER_3_PARTIAL] DB has {db_pairs} pairs, JSONL has {jsonl_pairs}. "
            f"Both below {PLATT_MIN_PAIRS} threshold -- calibrator may use raw confidence.",
        )
    if not db_path.exists() and not jsonl_path.exists():
        return Finding(
            "calibration_active_tier",
            "WARN",
            "[NONE] No active tier — CI/fresh environment: neither DB nor JSONL file present. "
            "Run bootstrap cycles locally to seed calibration data.",
        )
    if jsonl_pairs == 0 and db_pairs == 0:
        return Finding(
            "calibration_active_tier",
            "WARN",
            "[NONE] No active tier — DB/JSONL present but 0 usable pairs "
            "(fresh environment or schema not initialised). "
            "Run bootstrap cycles to seed calibration data.",
        )
    return Finding(
        "calibration_active_tier",
        "FAIL",
        f"[NONE] No active tier. JSONL={jsonl_pairs}, DB={db_pairs}. "
        f"All below {PLATT_MIN_PAIRS} floor. Signals pass through uncalibrated.",
    )


# ---------------------------------------------------------------------------
# Check 6: Calibration quality (ECE + Brier score)
# ---------------------------------------------------------------------------

def check_calibration_quality(jsonl_path: Path, n_bins: int = 10) -> Finding:
    """Compute Expected Calibration Error (ECE) and Brier score from JSONL outcome pairs.

    ECE measures reliability: whether stated confidence matches empirical win rates.
    Brier score measures overall probabilistic accuracy (0.25 = no-skill baseline).

    Both require JSONL tier to have >= PLATT_MIN_PAIRS actionable outcome pairs.
    Uses confidence_calibrated (pure Platt) when available, else blended confidence.
    """
    if not jsonl_path.exists():
        return Finding("calibration_quality", "SKIP", f"JSONL not found: {jsonl_path}")

    try:
        lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as exc:
        return Finding("calibration_quality", "FAIL", f"Cannot read JSONL: {exc}")

    pairs_conf: List[float] = []
    pairs_win: List[float] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except Exception:
            continue
        action = str(entry.get("action", "")).upper()
        if action not in {"BUY", "SELL"}:
            continue
        outcome = entry.get("outcome")
        if not isinstance(outcome, dict):
            continue
        win_raw = outcome.get("win")
        if win_raw is None:
            continue
        # Prefer pure Platt probability; fall back to blended confidence.
        conf_val = entry.get("confidence_calibrated") or entry.get("confidence")
        if conf_val is None:
            continue
        try:
            pairs_conf.append(float(conf_val))
            pairs_win.append(1.0 if win_raw else 0.0)
        except (TypeError, ValueError):
            continue

    n = len(pairs_conf)
    if n < PLATT_MIN_PAIRS:
        return Finding(
            "calibration_quality",
            "SKIP",
            f"Insufficient outcome pairs for quality check (n={n}, need {PLATT_MIN_PAIRS}).",
        )

    try:
        # Brier score: mean((p - y)^2). Lower is better; 0.25 = no-skill (p=0.5 always).
        brier = sum((c - w) ** 2 for c, w in zip(pairs_conf, pairs_win)) / n

        # ECE: bin confidences into n_bins buckets, compare mean_conf vs win_rate per bin.
        bins: List[List] = [[] for _ in range(n_bins)]
        for c, w in zip(pairs_conf, pairs_win):
            idx = min(int(c * n_bins), n_bins - 1)
            bins[idx].append((c, w))

        ece = 0.0
        for bucket in bins:
            if not bucket:
                continue
            bucket_n = len(bucket)
            mean_conf = sum(c for c, _ in bucket) / bucket_n
            win_rate = sum(w for _, w in bucket) / bucket_n
            ece += (bucket_n / n) * abs(mean_conf - win_rate)

        detail = (
            f"n={n}, ECE={ece:.4f} (threshold 0.15), Brier={brier:.4f} (no-skill=0.25)."
        )
        # Brier threshold = 0.25 (no-skill baseline: always predicting 0.5 gives exactly 0.25).
        # Accepting Brier > 0.25 would mean the calibrator is WORSE than random — a threshold
        # dodge.  Previously set to 0.30 which silently allowed below-random calibrators.
        if ece > 0.15 or brier > 0.25:
            return Finding(
                "calibration_quality",
                "WARN",
                f"Poor calibration: {detail} "
                f"Consider more training data or lower raw_weight to allow stronger correction.",
            )
        return Finding("calibration_quality", "PASS", detail)
    except Exception as exc:
        return Finding("calibration_quality", "FAIL", f"Quality computation failed: {exc}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_audit(*, db_path: Path, jsonl_path: Path) -> List[Finding]:
    return [
        check_classifier_identity(),
        check_fallback_chain_order(),
        check_hold_inflation(jsonl_path),
        check_ts_closes_in_db(db_path),
        check_calibration_active_tier(db_path, jsonl_path),
        check_calibration_quality(jsonl_path),
    ]


def print_report(findings: List[Finding]) -> None:
    width = 72
    print("=" * width)
    print("  PLATT CALIBRATION CONTRACT AUDIT")
    print("=" * width)
    for f in findings:
        print(f"\n[{f.status:4s}] {f.check}")
        # Simple word-wrap at 68 chars
        words = f.detail.split()
        line = "       "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "       " + word
            else:
                line += (" " if line.strip() else "") + word
        if line.strip():
            print(line)

    statuses = [f.status for f in findings]
    n_fail = statuses.count("FAIL")
    n_warn = statuses.count("WARN")
    n_pass = statuses.count("PASS")
    n_skip = statuses.count("SKIP")
    print("\n" + "-" * width)
    print(f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIP={n_skip}")
    if n_fail:
        print("[AUDIT RESULT] FAIL -- contract violations found")
    elif n_warn:
        print("[AUDIT RESULT] WARN -- review recommended")
    else:
        print("[AUDIT RESULT] PASS")
    print("=" * width)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Platt calibration contract audit -- validates implementation vs docs."
    )
    parser.add_argument("--db", default=None, help="Path to portfolio_maximizer.db")
    parser.add_argument("--jsonl", default=None, help="Path to quant_validation.jsonl")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output JSON array")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else ROOT / "data" / "portfolio_maximizer.db"
    jsonl_path = (
        Path(args.jsonl) if args.jsonl
        else ROOT / "logs" / "signals" / "quant_validation.jsonl"
    )

    try:
        findings = run_audit(db_path=db_path, jsonl_path=jsonl_path)
    except Exception as exc:
        print(f"[ERROR] Audit runtime failure: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        print_report(findings)

    return 1 if any(f.status == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
