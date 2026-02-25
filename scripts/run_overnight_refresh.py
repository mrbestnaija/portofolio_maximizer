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
BOOTSTRAP_DATES = [
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
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
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else DEFAULT_TICKERS
    all_tickers_csv = ",".join(tickers)
    errors = 0

    _open_log()

    log_section("Overnight refresh started")
    log(f"Python   : {PYTHON}")
    log(f"Repo     : {REPO_ROOT}")
    log(f"Log      : {LOG_PATH}")
    log(f"Tickers  : {' '.join(tickers)}")
    log(f"Bootstrap: {args.platt_bootstrap}")
    log("Estimated runtime: 60-120 minutes")

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

    # -------------------------------------------------------------------
    # STEP 2.6: Historical Platt bootstrap (opt-in)
    # -------------------------------------------------------------------
    if args.platt_bootstrap:
        log_section("STEP 2.6/3: Platt bootstrap -- seeding pairs from 2021-2024")
        for as_of in BOOTSTRAP_DATES:
            log(f"--- bootstrap as-of {as_of}")
            py(
                "scripts/run_auto_trader.py",
                "--tickers", all_tickers_csv,
                "--cycles", "1",
                "--execution-mode", "synthetic",
                "--as-of-date", as_of,
                "--no-resume",
                "--sleep-seconds", "0",
                allow_fail=True,
            )
            py("scripts/update_platt_outcomes.py", allow_fail=True)
        log("[DONE] Platt bootstrap complete")

    # -------------------------------------------------------------------
    # STEP 3: Health check + headroom
    # -------------------------------------------------------------------
    log_section("STEP 3/3: Final health check")

    log("--- check_quant_validation_health.py")
    py("scripts/check_quant_validation_health.py", allow_fail=True)

    log("--- quant_validation_headroom.py --json")
    py("scripts/quant_validation_headroom.py", "--json", allow_fail=True)

    log("--- production_audit_gate.py (refresh forecast_audits_cache)")
    # Phase 7.15: --allow-inconclusive-lift aligns with forecaster_monitoring.yml comment:
    # "Until [holding_period_audits] are met, checks are considered inconclusive (non-failing)."
    # Holding period requires 20 unique audit windows (different AS_OF dates); overnight refresh
    # with a fixed date range (2024-01-01 -> 2026-01-01) is structurally capped at ~11 unique
    # audit windows after deduplication.  Inconclusive = holdout runway not yet complete, not
    # a model quality failure.
    py("scripts/production_audit_gate.py", "--allow-inconclusive-lift", allow_fail=True)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    log_section("OVERNIGHT REFRESH COMPLETE")
    log(f"Errors : {errors}")
    log(f"Log    : {LOG_PATH}")
    log(f"Summary: {SUMMARY_PATH}")

    summary_lines = [
        "=== Overnight Refresh Summary ===",
        f"Completed : {datetime.datetime.now()}",
        f"Errors    : {errors}",
        f"Tickers   : {ticker_pass} passed / {ticker_fail} failed",
        "",
        "Next steps:",
        f"  1. Check log   : {LOG_PATH}",
        "  2. Headroom    : python scripts/quant_validation_headroom.py --json",
        "  3. Health      : python scripts/check_quant_validation_health.py",
        "  4. Platt check : python scripts/update_platt_outcomes.py",
        "",
        "Platt pair count:",
        "  python -c \"import json,pathlib; e=[json.loads(l) for l in pathlib.Path('logs/signals/quant_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; print(f'Platt pairs: {len([x for x in e if x.get(\\\"outcome\\\") is not None])}')\"",
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
