"""
EXP-R5-001 Phase 3 re-accumulation under RC1-RC4 redesigned residual model.

Runs 10 ETL pipeline passes with distinct --end dates, then chains:
  1. residual_experiment_phase3_backfill.py  (realized-price patching)
  2. run_quality_pipeline.py --enable-residual-experiment  (refresh summary JSON)
  3. residual_experiment_truth.py  (contradiction check)
  4. M3 promotion verdict (printed to terminal)

USAGE
-----
    # Activate venv first, then:
    python scripts/run_exp_r5_phase3_accumulate.py
    python scripts/run_exp_r5_phase3_accumulate.py --ticker MSFT
    python scripts/run_exp_r5_phase3_accumulate.py --dry-run   # skip passes, only backfill+truth
    python scripts/run_exp_r5_phase3_accumulate.py --start 2019-01-01  # wider history

KEY FACTS
---------
- All end dates fall within 2020-2024 so the widest checkpoint
  (pipeline_20260308_184327_data_extraction_*.parquet) covers realized prices.
- Each run produces one uniquely fingerprinted audit (SHA1 of ticker+start+end+len+horizon).
  Runs with identical --start/--end produce the same fingerprint and are deduplicated
  by phase3_backfill.py -- distinct end dates are required for new windows.

M3 PROMOTION CRITERIA
---------------------
  PROMOTE   : n_windows >= 10  AND  mean_rmse_ratio < 1.0  AND  mean_corr >= 0.30
  Otherwise : INCONCLUSIVE / REDESIGN_REQUIRED
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("phase3_accumulate")

ROOT = Path(__file__).resolve().parent.parent

# 10 end dates spanning 2021-2024 in ~3-month steps.
# All within checkpoint coverage (2020-01-01 to 2024-01-01).
DEFAULT_END_DATES = [
    "2021-03-01",
    "2021-06-01",
    "2021-09-01",
    "2021-12-01",
    "2022-03-01",
    "2022-06-01",
    "2022-09-01",
    "2023-01-01",
    "2023-06-01",
    "2024-01-01",
]

M3_RMSE_THRESHOLD = 1.0
M3_CORR_THRESHOLD = 0.30
M3_MIN_WINDOWS = 10

SUMMARY_JSON = ROOT / "visualizations" / "performance" / "residual_experiment_summary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], step: str) -> int:
    """Run a subprocess, stream output live, return exit code."""
    log.info("[%s] Running: %s", step, " ".join(args))
    result = subprocess.run(args, cwd=str(ROOT))
    if result.returncode != 0:
        log.error("[%s] Exited with code %d", step, result.returncode)
    return result.returncode


def _python_bin() -> str:
    """Resolve venv Python binary (Windows or Unix)."""
    candidates = [
        ROOT / "simpleTrader_env" / "Scripts" / "python.exe",
        ROOT / "simpleTrader_env" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back to the interpreter running this script
    return sys.executable


def _print_m3_verdict() -> None:
    """Read summary JSON and print M3 promotion verdict."""
    if not SUMMARY_JSON.exists():
        log.warning("Summary JSON not found at %s", SUMMARY_JSON)
        log.warning("Run scripts/run_quality_pipeline.py --enable-residual-experiment manually.")
        return

    try:
        s = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Cannot read summary JSON: %s", exc)
        return

    n_windows = s.get(
        "n_windows_with_residual_metrics",
        s.get("n_windows_with_realized_residual_metrics", 0),
    )
    rmse_ratio = s.get("rmse_ratio_mean", s.get("mean_rmse_ratio"))
    corr = s.get("corr_mean", s.get("mean_corr"))
    status = s.get("status", "UNKNOWN")

    sep = "=" * 60
    print(sep)
    print("[EXP-R5-001 Phase 3] M3 Promotion Summary")
    print(sep)
    print(f"  n_windows_with_realized_metrics : {n_windows}")
    print(f"  mean_rmse_ratio                 : {rmse_ratio}")
    print(f"  mean_corr(epsilon, epsilon_hat) : {corr}")
    print(f"  summary status                  : {status}")
    print()

    if n_windows is None or n_windows < M3_MIN_WINDOWS:
        print(f"[M3] INSUFFICIENT DATA: {n_windows} windows (need >= {M3_MIN_WINDOWS})")
        print("[M3] Run more passes with additional --end dates.")
        return

    promote = (
        rmse_ratio is not None and rmse_ratio < M3_RMSE_THRESHOLD
        and corr is not None and corr >= M3_CORR_THRESHOLD
    )

    if promote:
        print(
            f"[M3] PROMOTE: mean_rmse_ratio={rmse_ratio:.4f} < 1.0  AND  "
            f"mean_corr={corr:.4f} >= 0.30 over {n_windows} windows"
        )
        print("[M3] RC1-RC4 redesign is an improvement. Ready for Agent C promotion decision.")
    else:
        reasons = []
        if rmse_ratio is not None and rmse_ratio >= M3_RMSE_THRESHOLD:
            reasons.append(f"rmse_ratio={rmse_ratio:.4f} >= {M3_RMSE_THRESHOLD}")
        if corr is not None and corr < M3_CORR_THRESHOLD:
            reasons.append(f"corr={corr:.4f} < {M3_CORR_THRESHOLD}")
        if not reasons:
            reasons.append("rmse_ratio or corr is None (backfill may be incomplete)")
        print(f"[M3] INCONCLUSIVE / REDESIGN_REQUIRED: {'; '.join(reasons)}")
        print("[M3] Investigate root causes before re-accumulating.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-R5-001 Phase 3 accumulation")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to run (default: AAPL)")
    parser.add_argument("--start", default="2020-01-01", help="Pipeline --start date")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip ETL passes; only run backfill + truth check",
    )
    parser.add_argument(
        "--end-dates",
        nargs="+",
        default=DEFAULT_END_DATES,
        metavar="DATE",
        help="Custom list of --end dates (default: 10 dates 2021-2024)",
    )
    args = parser.parse_args()

    python = _python_bin()
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    sep = "=" * 60
    print(sep)
    print(f"[EXP-R5-001 Phase 3] Starting accumulation  {run_ts}")
    print(f"  Ticker    : {args.ticker}")
    print(f"  Start     : {args.start}")
    print(f"  End dates : {args.end_dates}")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  Python    : {python}")
    print(sep)

    # -----------------------------------------------------------------------
    # Step 1: ETL pipeline passes
    # -----------------------------------------------------------------------
    if args.dry_run:
        log.info("[Step 1] DRY RUN -- skipping ETL pipeline passes")
    else:
        total = len(args.end_dates)
        for i, end_date in enumerate(args.end_dates, 1):
            print()
            print(f"[Pass {i}/{total}] --tickers {args.ticker} --start {args.start} --end {end_date}")
            rc = _run(
                [
                    python,
                    str(ROOT / "scripts" / "run_etl_pipeline.py"),
                    "--tickers", args.ticker,
                    "--start", args.start,
                    "--end", end_date,
                    "--execution-mode", "synthetic",
                ],
                step=f"ETL pass {i}/{total}",
            )
            if rc != 0:
                log.error("Pass %d/%d failed (--end %s). Aborting.", i, total, end_date)
                return rc
            print(f"[Pass {i}/{total}] DONE")

        print()
        log.info("[Step 1] All %d pipeline passes completed.", total)

    # -----------------------------------------------------------------------
    # Step 2: Phase 3 backfill
    # -----------------------------------------------------------------------
    print()
    print(sep)
    print("[Step 2] Phase 3 backfill (realized-price patch)")
    print(sep)
    rc = _run(
        [python, str(ROOT / "scripts" / "residual_experiment_phase3_backfill.py")],
        step="backfill",
    )
    if rc != 0:
        log.error("Phase 3 backfill failed with exit code %d.", rc)
        return rc

    # -----------------------------------------------------------------------
    # Step 3: Refresh residual summary JSON via quality pipeline
    # -----------------------------------------------------------------------
    print()
    print(sep)
    print("[Step 3] Refreshing residual summary (quality pipeline)")
    print(sep)
    _run(
        [python, str(ROOT / "scripts" / "run_quality_pipeline.py"),
         "--enable-residual-experiment"],
        step="quality_pipeline",
    )
    # Non-fatal: other gate failures don't block the residual summary refresh

    # -----------------------------------------------------------------------
    # Step 4: Truth check
    # -----------------------------------------------------------------------
    print()
    print(sep)
    print("[Step 4] EXP-R5-001 truth check")
    print(sep)
    truth_rc = _run(
        [python, str(ROOT / "scripts" / "residual_experiment_truth.py")],
        step="truth_check",
    )
    if truth_rc != 0:
        log.warning("Truth check exited %d (possible contradiction in summary).", truth_rc)

    # -----------------------------------------------------------------------
    # Step 5: M3 promotion verdict
    # -----------------------------------------------------------------------
    print()
    _print_m3_verdict()

    print()
    print(sep)
    print(f"[EXP-R5-001 Phase 3] Complete  {run_ts}")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
