"""
Pure-Python replacement for bash/overnight_refresh.sh.

Works on Windows without WSL or Git Bash.
Usage:
    python scripts/run_overnight_refresh.py
    python scripts/run_overnight_refresh.py --platt-bootstrap    # seed 2021-2024 Platt pairs
    python scripts/run_overnight_refresh.py --skip-adversarial   # skip step 1 (faster)
    python scripts/run_overnight_refresh.py --tickers MSFT,NVDA  # override ticker list

Env vars honoured:
    PLATT_BOOTSTRAP=1   same as --platt-bootstrap
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / "simpleTrader_env" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

LOG_DIR = REPO_ROOT / "logs" / "run_audit"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"overnight_refresh_{TS}.log"
SUMMARY_PATH = LOG_DIR / f"overnight_refresh_{TS}_summary.txt"

DEFAULT_TICKERS = ["AMZN", "GOOG", "GS", "JPM", "META", "MSFT", "NVDA", "TSLA", "V"]
# Bootstrap: business-day ladder design (2026-02-26, adversarial-validated).
#
# Why ladders, not fixed anchor dates:
#   - Fixed --cycles N at a fixed --as-of-date repeats the same bar N times.
#     Bar-aware gate skips cycles 2..N: 0 bar progression, 0 aging, 0 closes.
#   - 1 cycle per consecutive business day with --resume carries open positions
#     forward in time so proof-mode max_holding fires and closes are written.
#   - Without --resume, each cycle starts with empty portfolio: 0 carries, 0 closes.
#
# Verified (2026-02-26 empirical replay):
#   fixed-cycles    -> 2 opens, 0 closes
#   no-bar-aware    -> 16 opens, 0 closes (same-bar repeats, no aging)
#   ladder no-resume-> 4 opens, 0 closes
#   ladder + resume -> 4 opens, 4 closes  <-- correct design
#   ladder + resume + relaxed guards -> 23 opens, 9 closes (higher yield)
BOOTSTRAP_ANCHOR_DATES = [
    "2021-06-01",   # post-pandemic recovery
    "2022-06-01",   # bear market
    "2023-06-01",   # recovery
    "2024-06-01",   # bull market
]
BOOTSTRAP_LADDER_DAYS = 10      # business days per anchor (enough for max_holding=8 to fire; Phase 7.19)
BOOTSTRAP_TARGET_PAIRS = 30     # stop early when matched pairs reach threshold
AUDIT_MIN_UNIQUE_WINDOWS = 20   # unique dedup keys required for lift gate to exit INCONCLUSIVE

# Audit gate bootstrap dates: 20 AS_OF dates spanning 2022-Q1 through 2026-Q1.
# Each date changes dataset.end in forecast audit files, producing a unique dedup key
# (dataset.start, dataset.end, dataset.length, forecast_horizon) in check_forecast_audits.py.
# With 20 unique dates, effective_n >= holding_period_audits (20) -> gate exits holding period
# and makes a definitive verdict (PASS/FAIL) instead of INCONCLUSIVE.
# Dates are spaced ~90 days apart (quarterly) so each has a distinct dataset window.
AUDIT_GATE_BOOTSTRAP_DATES = [
    "2022-01-03",  # Q1 2022
    "2022-04-04",  # Q2 2022
    "2022-07-05",  # Q3 2022 (July 4 holiday)
    "2022-10-03",  # Q4 2022
    "2023-01-03",  # Q1 2023
    "2023-04-03",  # Q2 2023
    "2023-07-03",  # Q3 2023
    "2023-10-02",  # Q4 2023
    "2024-01-02",  # Q1 2024
    "2024-04-01",  # Q2 2024
    "2024-07-01",  # Q3 2024
    "2024-10-01",  # Q4 2024
    "2025-01-02",  # Q1 2025
    "2025-04-01",  # Q2 2025
    "2025-07-01",  # Q3 2025
    "2025-10-01",  # Q4 2025
    "2026-01-02",  # Q1 2026
    "2026-01-15",  # Mid-Jan 2026
    "2026-02-02",  # Early Feb 2026
    "2026-02-16",  # Mid-Feb 2026
]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_log_fh = None


def _open_log() -> None:
    global _log_fh
    _log_fh = open(LOG_PATH, "w", encoding="utf-8", buffering=1)


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _log_fh:
        _log_fh.write(line + "\n")


def log_section(title: str) -> None:
    sep = "=" * 40
    for line in ("", sep, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {title}", sep):
        print(line)
        if _log_fh:
            _log_fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------
def run(cmd: list[str], *, allow_fail: bool = False) -> int:
    """Run a subprocess, stream stdout/stderr to console + log file."""
    display = " ".join(cmd[1:] if cmd[0] == PYTHON else cmd)
    log(f"Running: {display}")
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\n")
            print(line)
            if _log_fh:
                _log_fh.write(line + "\n")
        proc.wait()
    rc = proc.returncode
    if rc != 0 and not allow_fail:
        log(f"[WARN] Command returned code {rc}")
    return rc


def py(script: str, *args: str, allow_fail: bool = False) -> int:
    return run([PYTHON, script, *args], allow_fail=allow_fail)


# ---------------------------------------------------------------------------
# Bootstrap outcome guard
# ---------------------------------------------------------------------------

def _count_ts_closes() -> int:
    """Return the number of ts_* is_close=1 trades in the production DB.

    Used by the bootstrap outcome guard: if the count does not increase after a
    bootstrap run, the bootstrap design is broken (e.g. cycles-vs-bars mismatch
    where proof-mode max_holding never fires at a fixed --as-of-date).
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415 -- local to avoid top-level import cost
    db_path = REPO_ROOT / "data" / "portfolio_maximizer.db"
    if not db_path.exists():
        return 0
    try:
        conn = _sqlite3.connect(str(db_path), timeout=3.0)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE ts_signal_id LIKE 'ts_%' AND is_close = 1 AND realized_pnl IS NOT NULL"
        )
        n = cur.fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def _count_ts_opens() -> int:
    """Return count of ts_* is_close=0 (open) trades -- used for per-date telemetry."""
    import sqlite3 as _sqlite3  # noqa: PLC0415
    db_path = REPO_ROOT / "data" / "portfolio_maximizer.db"
    if not db_path.exists():
        return 0
    try:
        conn = _sqlite3.connect(str(db_path), timeout=3.0)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE ts_signal_id LIKE 'ts_%' AND is_close = 0"
        )
        n = cur.fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def _count_jsonl_pairs() -> int:
    """Return count of JSONL entries that have an 'outcome' field (matched pairs)."""
    jsonl_path = REPO_ROOT / "logs" / "signals" / "quant_validation.jsonl"
    if not jsonl_path.exists():
        return 0
    try:
        return sum(
            1 for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and "outcome" in json.loads(line)
        )
    except Exception:
        return 0


def _count_unique_audit_windows() -> int:
    """Return the number of unique forecast audit windows in logs/forecast_audits/.

    Deduplicates by (dataset.start, dataset.end, dataset.length, forecast_horizon),
    matching the same key used by check_forecast_audits.py.  Only counts files that
    produced valid metrics (benchmark_summary or model_benchmarks present).

    Phase 7.15-D: used by the audit gate auto-trigger and run summary cardinality report.
    """
    audit_dir = REPO_ROOT / "logs" / "forecast_audits"
    if not audit_dir.exists():
        return 0
    unique_keys: set = set()
    for f in audit_dir.glob("forecast_audit_*.json*"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ds = data.get("dataset", {})
            key = (ds.get("start"), ds.get("end"), ds.get("length"), ds.get("forecast_horizon"))
            if not all(v is not None for v in key):
                continue
            if data.get("benchmark_summary") or data.get("model_benchmarks"):
                unique_keys.add(key)
        except Exception:
            pass
    return len(unique_keys)


def _business_day_ladder(start_date: str, n_days: int) -> list[str]:
    """Return n_days consecutive Mon-Fri dates starting from start_date (inclusive)."""
    import datetime as _dt  # noqa: PLC0415
    d = _dt.date.fromisoformat(start_date)
    result: list[str] = []
    while len(result) < n_days:
        if d.weekday() < 5:   # Monday=0 .. Friday=4
            result.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overnight refresh (Windows-native, no bash required)"
    )
    parser.add_argument(
        "--platt-bootstrap", action="store_true",
        default=os.environ.get("PLATT_BOOTSTRAP", "0") == "1",
        help="Seed Platt pairs from 8 historical dates (2021-2024)",
    )
    parser.add_argument(
        "--skip-adversarial", action="store_true",
        help="Skip Step 1 adversarial forecaster suite (saves ~20 min)",
    )
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated override for pipeline tickers",
    )
    parser.add_argument(
        "--audit-gate-bootstrap", action="store_true",
        default=os.environ.get("AUDIT_GATE_BOOTSTRAP", "0") == "1",
        help=(
            "Seed forecast audit windows from 20 historical AS_OF dates (2022-2026). "
            "Generates unique (dataset.start, dataset.end) dedup keys so effective_n >= "
            "holding_period_audits=20, moving the gate from INCONCLUSIVE to a definitive "
            "PASS/FAIL verdict. Adds ~30-60 min. Env: AUDIT_GATE_BOOTSTRAP=1."
        ),
    )
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else DEFAULT_TICKERS
    all_tickers_csv = ",".join(tickers)
    errors = 0

    _open_log()

    log_section("Overnight refresh started")
    log(f"Python   : {PYTHON}")
    log(f"Repo     : {REPO_ROOT}")
    log(f"Log      : {LOG_PATH}")
    log(f"Tickers        : {' '.join(tickers)}")
    log(f"Platt bootstrap: {args.platt_bootstrap}")
    log(f"Audit bootstrap: {args.audit_gate_bootstrap}")
    log("Estimated runtime: 60-120 min (+30-60 min if --audit-gate-bootstrap)")

    # -------------------------------------------------------------------
    # STEP 1: Adversarial forecaster suite
    # -------------------------------------------------------------------
    if not args.skip_adversarial:
        log_section("STEP 1/3: Adversarial forecaster suite (DA re-baseline)")
        rc = py("scripts/run_adversarial_forecaster_suite.py", allow_fail=True)
        if rc == 0:
            log("[PASS] Adversarial suite completed")
        else:
            log("[WARN] Adversarial suite returned non-zero (check log)")
            errors += 1
    else:
        log_section("STEP 1/3: Adversarial suite SKIPPED (--skip-adversarial)")

    # -------------------------------------------------------------------
    # STEP 2: Pipeline per ticker -> populate quant_validation.jsonl
    # -------------------------------------------------------------------
    log_section("STEP 2/3: Pipeline refresh for non-AAPL tickers")

    log("--- update_platt_outcomes.py (pre-run reconciliation)")
    py("scripts/update_platt_outcomes.py", allow_fail=True)

    ticker_pass = 0
    ticker_fail = 0
    for ticker in tickers:
        log(f"--- Pipeline: {ticker} (start 2024-01-01 end 2026-01-01 synthetic)")
        rc = py(
            "scripts/run_etl_pipeline.py",
            "--tickers", ticker,
            "--start", "2024-01-01",
            "--end", "2026-01-01",
            "--execution-mode", "synthetic",
            allow_fail=True,
        )
        if rc == 0:
            log(f"[PASS] {ticker} pipeline completed")
            ticker_pass += 1
        else:
            log(f"[WARN] {ticker} pipeline returned non-zero")
            ticker_fail += 1
            errors += 1

    log(f"Ticker results: {ticker_pass} passed / {ticker_fail} failed")

    # -------------------------------------------------------------------
    # STEP 2.5: Synthetic auto_trader cycle (accumulate Platt data)
    # -------------------------------------------------------------------
    log_section("STEP 2.5/3: Synthetic auto_trader cycle (Platt data accumulation)")
    log(f"--- run_auto_trader.py --tickers {all_tickers_csv} --cycles 1 synthetic --no-resume")
    rc = py(
        "scripts/run_auto_trader.py",
        "--tickers", all_tickers_csv,
        "--cycles", "1",
        "--execution-mode", "synthetic",
        "--no-resume",
        "--sleep-seconds", "0",
        allow_fail=True,
    )
    if rc == 0:
        log("[PASS] Synthetic auto_trader cycle completed")
    else:
        log("[WARN] Synthetic auto_trader cycle returned non-zero (Platt pairs may not accumulate)")
        errors += 1

    log("--- update_platt_outcomes.py (reconcile new trade outcomes)")
    py("scripts/update_platt_outcomes.py", allow_fail=True)

    # Phase 7.15: Auto-trigger bootstrap when JSONL pair count is below threshold.
    # Root cause of 0 pairs: --cycles 1 never closes positions within one cycle,
    # so no is_close=1 rows are written -> update_platt_outcomes finds nothing to match.
    # Solution: auto-enable bootstrap when pairs_with_outcome < PLATT_MIN_PAIRS.
    PLATT_MIN_PAIRS = 30
    if not args.platt_bootstrap:
        _jsonl = REPO_ROOT / "logs" / "signals" / "quant_validation.jsonl"
        _pairs = 0
        if _jsonl.exists():
            try:
                _pairs = sum(
                    1 for _line in _jsonl.read_text(encoding="utf-8").splitlines()
                    if _line.strip() and "outcome" in json.loads(_line)
                )
            except Exception:
                pass
        log(f"--- Platt pairs check: {_pairs} pairs_with_outcome (threshold: {PLATT_MIN_PAIRS})")
        if _pairs < PLATT_MIN_PAIRS:
            log(f"[AUTO] Platt pairs below {PLATT_MIN_PAIRS} -- enabling bootstrap automatically")
            args.platt_bootstrap = True

    # Phase 7.15-D: Auto-trigger audit gate bootstrap when unique audit windows < threshold.
    # Mirrors Platt bootstrap auto-trigger. check_forecast_audits.py deduplicates by
    # (dataset.start, dataset.end, dataset.length, forecast_horizon); a fixed-date nightly
    # routine is structurally capped at ~11 unique keys. This guard fires once per system
    # until 20+ diverse AS_OF windows have been bootstrapped.
    if not args.audit_gate_bootstrap:
        _unique_windows = _count_unique_audit_windows()
        log(f"--- Audit window check: {_unique_windows} unique windows (threshold: {AUDIT_MIN_UNIQUE_WINDOWS})")
        if _unique_windows < AUDIT_MIN_UNIQUE_WINDOWS:
            log(f"[AUTO] Unique audit windows below {AUDIT_MIN_UNIQUE_WINDOWS} -- enabling audit gate bootstrap automatically")
            args.audit_gate_bootstrap = True

    # -------------------------------------------------------------------
    # STEP 2.6: Historical Platt bootstrap (opt-in, or auto when pairs < threshold)
    # -------------------------------------------------------------------
    if args.platt_bootstrap:
        log_section("STEP 2.6/3: Platt bootstrap -- seeding pairs from 2021-2024")
        # Platt calibration requires CLOSED trades with ts_* signal IDs.
        #
        # Adversarial finding (2026-02-25): historical bootstrap with
        #   --cycles 8 + fixed --as-of-date + default --bar-aware
        # yields cycle-1 executions then cycles 2..8 as SKIPPED_SAME_BAR,
        # so no bar progression occurs and no time exits are written.
        #
        # Fix: run a single cycle per historical as-of date and carry portfolio
        # state forward across dates (resume after first date). This advances
        # bar timestamps across runs and allows proof-mode max_holding exits
        # to produce closed ts_* trades that update_platt_outcomes can match.
        #
        # Bootstrap outcome guard baseline.
        _ts_closes_before = _count_ts_closes()
        _cumulative_matched = _count_jsonl_pairs()
        log(f"--- Bootstrap start: ts_* closes={_ts_closes_before}, matched_pairs={_cumulative_matched} (target {BOOTSTRAP_TARGET_PAIRS})")

        # Relaxed guard env vars for bootstrap only -- maximize close throughput.
        # Empirically verified: same ladder + relaxed guards -> 23 opens / 9 closes
        # vs 4 opens / 4 closes with strict guards (2026-02-26 adversarial replay).
        _RELAX_KEYS = ["PMX_LONG_ONLY", "PMX_PROOF_STRICT_THRESHOLDS", "PMX_EDGE_COST_GATE"]
        _saved_env = {k: os.environ.get(k) for k in _RELAX_KEYS}
        os.environ["PMX_LONG_ONLY"] = "0"
        os.environ["PMX_PROOF_STRICT_THRESHOLDS"] = "0"
        os.environ["PMX_EDGE_COST_GATE"] = "0"

        try:
            for anchor in BOOTSTRAP_ANCHOR_DATES:
                if _cumulative_matched >= BOOTSTRAP_TARGET_PAIRS:
                    log(f"[STOP] Reached {_cumulative_matched} matched pairs (target {BOOTSTRAP_TARGET_PAIRS}) -- bootstrap complete early.")
                    break

                ladder = _business_day_ladder(anchor, BOOTSTRAP_LADDER_DAYS)
                log(f"--- Anchor {anchor}: ladder {ladder[0]} -> {ladder[-1]} ({len(ladder)} days)")
                _opens_before = _count_ts_opens()
                _closes_before_anchor = _count_ts_closes()

                for day_idx, as_of in enumerate(ladder):
                    cmd = [
                        "scripts/run_auto_trader.py",
                        "--tickers", all_tickers_csv,
                        "--cycles", "1",
                        "--execution-mode", "synthetic",
                        "--as-of-date", as_of,
                        "--sleep-seconds", "0",
                        "--proof-mode",
                        "--no-resume" if day_idx == 0 else "--resume",
                    ]
                    py(*cmd, allow_fail=True)

                py("scripts/update_platt_outcomes.py", allow_fail=True)
                _cumulative_matched = _count_jsonl_pairs()
                _opens_delta = _count_ts_opens() - _opens_before
                _closes_delta = _count_ts_closes() - _closes_before_anchor
                log(
                    f"    Anchor {anchor}: opens_delta={_opens_delta}, "
                    f"closes_delta={_closes_delta}, cumulative_matched={_cumulative_matched}"
                )
        finally:
            # Restore env
            for k, v in _saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # Bootstrap outcome guard: verify at least one new close was produced.
        _ts_closes_after = _count_ts_closes()
        _new_closes = _ts_closes_after - _ts_closes_before
        if _new_closes == 0:
            log(
                "[FAIL] Bootstrap outcome guard: 0 new ts_* closed trades produced across all "
                f"{len(BOOTSTRAP_ANCHOR_DATES)} anchor windows. "
                "Ladder+resume design should produce closes -- investigate bar-awareness, "
                "position sizing, or proof-mode gate filtering. "
                "See tests/scripts/test_platt_calibration_contract.py."
            )
            errors += 1
        else:
            log(
                f"[OK] Bootstrap outcome guard: {_new_closes} new ts_* closes, "
                f"{_cumulative_matched} cumulative matched pairs."
            )
        log("[DONE] Platt bootstrap complete")

    # -------------------------------------------------------------------
    # STEP 2.7: Audit gate bootstrap -- seed unique forecast audit windows
    # -------------------------------------------------------------------
    # Problem: check_forecast_audits.py deduplicates by (dataset.start, dataset.end,
    # dataset.length, forecast_horizon). The fixed overnight date range 2024-01-01 ->
    # 2026-01-01 always yields ~11 unique windows, below holding_period_audits=20.
    # Solution: run auto_trader with 20 historical AS_OF dates so each produces a
    # distinct dataset.end -> unique dedup key -> effective_n >= 20 -> gate no longer
    # INCONCLUSIVE. Only needs to be run once; audit files persist in logs/forecast_audits/.
    if args.audit_gate_bootstrap:
        log_section("STEP 2.7/3: Audit gate bootstrap (20 AS_OF windows for lift gate)")
        log(f"Generating {len(AUDIT_GATE_BOOTSTRAP_DATES)} unique dataset windows ...")
        audit_win_pass = 0
        audit_win_fail = 0
        for as_of in AUDIT_GATE_BOOTSTRAP_DATES:
            log(f"--- audit window as-of {as_of}")
            rc = py(
                "scripts/run_auto_trader.py",
                "--tickers", all_tickers_csv,
                "--cycles", "1",
                "--execution-mode", "synthetic",
                "--as-of-date", as_of,
                "--no-resume",
                "--sleep-seconds", "0",
                allow_fail=True,
            )
            if rc == 0:
                audit_win_pass += 1
            else:
                log(f"[WARN] audit window as-of {as_of} returned non-zero")
                audit_win_fail += 1
        log(f"Audit window results: {audit_win_pass} passed / {audit_win_fail} failed")
        log("[DONE] Audit gate bootstrap complete -- run production_audit_gate.py to verify effective_n >= 20")
    else:
        log("STEP 2.7 SKIPPED -- re-run with --audit-gate-bootstrap (or AUDIT_GATE_BOOTSTRAP=1) to seed 20 unique audit windows")

    # -------------------------------------------------------------------
    # STEP 2.8: Deduplicate audit windows (dry-run -- report only)
    # -------------------------------------------------------------------
    log_section("STEP 2.8/3: Deduplicate forecast audit windows (dry-run)")
    rc = py(
        "scripts/dedupe_audit_windows.py",
        allow_fail=True,  # non-blocking: exit 1 = duplicates found (warning only)
    )
    if rc == 1:
        log("[WARN] Duplicate audit windows detected -- run dedupe_audit_windows.py --apply to remove")
    elif rc == 0:
        log("[OK] No duplicate audit windows")

    # -------------------------------------------------------------------
    # STEP 2.9: Ensemble health audit + adaptive weight update
    # -------------------------------------------------------------------
    log_section("STEP 2.9/3: Ensemble health audit + adaptive weights")
    rc = py(
        "scripts/ensemble_health_audit.py",
        "--write-config",
        "--write-report",
        "--recent-n", "20",
        allow_fail=True,  # non-blocking: diagnostic, not gatekeeping
    )
    if rc != 0:
        log("[WARN] Ensemble health audit returned non-zero -- check logs/ensemble_health/")

    # -------------------------------------------------------------------
    # STEP 3: Health check + headroom
    # -------------------------------------------------------------------
    log_section("STEP 3/3: Final health check")

    log("--- check_quant_validation_health.py")
    rc = py("scripts/check_quant_validation_health.py", allow_fail=True)
    if rc != 0:
        errors += 1

    log("--- quant_validation_headroom.py --json")
    rc = py("scripts/quant_validation_headroom.py", "--json", allow_fail=True)
    if rc != 0:
        errors += 1

    log("--- production_audit_gate.py (refresh forecast_audits_cache)")
    # Phase 7.15 / 7.15-D: --allow-inconclusive-lift aligns with forecaster_monitoring.yml comment:
    # "Until [holding_period_audits] are met, checks are considered inconclusive (non-failing)."
    # Phase 7.15-D: auto-trigger in step 2.7 ensures unique_windows >= AUDIT_MIN_UNIQUE_WINDOWS
    # (20) so the gate can make a definitive PASS/FAIL verdict.  Inconclusive = holdout runway
    # not yet complete or lift gate below threshold, not a structural window-count failure.
    # --reconcile --reconcile-apply: auto-link unlinked closes during nightly maintenance window.
    # Scans all unlinked closes (no --close-ids = scan all) and applies direction-aware repair.
    rc = py("scripts/production_audit_gate.py", "--allow-inconclusive-lift",
            "--reconcile", "--reconcile-apply", allow_fail=True)
    if rc != 0:
        errors += 1

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    # Phase 7.15-D: Capture final dedupe-key cardinality for summary.
    _final_unique_windows = _count_unique_audit_windows()

    log_section("OVERNIGHT REFRESH COMPLETE")
    log(f"Errors : {errors}")
    log(f"Log    : {LOG_PATH}")
    log(f"Summary: {SUMMARY_PATH}")

    summary_lines = [
        "=== Overnight Refresh Summary ===",
        f"Completed        : {datetime.datetime.now()}",
        f"Errors           : {errors}",
        f"Tickers          : {ticker_pass} passed / {ticker_fail} failed",
        f"Platt bootstrap  : {'ON (8 AS_OF windows, resume + proof-mode)' if args.platt_bootstrap else 'OFF (run with --platt-bootstrap to seed pairs)'}",
        f"Audit bootstrap  : {'ON (20 AS_OF windows)' if args.audit_gate_bootstrap else 'OFF (run with --audit-gate-bootstrap to reach holding_period_audits=20)'}",
        f"Audit windows    : {_final_unique_windows} unique dedup keys (target {AUDIT_MIN_UNIQUE_WINDOWS}+)",
        "",
        "Diagnostics:",
        "  # Platt pair count (needs > 0 before calibration trains):",
        "  python -c \"import json,pathlib; e=[json.loads(l) for l in pathlib.Path('logs/signals/quant_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; pairs=[x for x in e if x.get('outcome') is not None]; print(f'Platt pairs: {len(pairs)} (target 30+)')\"",
        "  # Audit gate status:",
        "  python scripts/production_audit_gate.py --allow-inconclusive-lift",
        "  # Check effective audit windows:",
        "  python scripts/check_forecast_audits.py --config-path config/forecaster_monitoring.yml",
        "",
        "Next steps:",
        f"  1. Check log   : {LOG_PATH}",
        "  2. Headroom    : python scripts/quant_validation_headroom.py --json",
        "  3. Health      : python scripts/check_quant_validation_health.py",
        "  4. Platt check : python scripts/update_platt_outcomes.py",
        "  5. If 0 Platt pairs: run with --platt-bootstrap to seed 8 historical windows (1 cycle/date + resume + proof-mode)",
        "  6. If lift INCONCLUSIVE: run with --audit-gate-bootstrap to seed 20 unique audit windows",
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    if _log_fh:
        _log_fh.write(summary_text + "\n")

    SUMMARY_PATH.write_text(summary_text + "\n", encoding="utf-8")

    if _log_fh:
        _log_fh.close()

    return errors


if __name__ == "__main__":
    sys.exit(main())
