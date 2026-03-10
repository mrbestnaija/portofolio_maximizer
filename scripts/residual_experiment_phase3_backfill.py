"""EXP-R5-001 Phase 3 Backfill.

Computes realized-price metrics for all active Phase-2 audit windows:
  rmse_anchor, rmse_residual_ensemble, rmse_ratio,
  da_anchor, da_residual_ensemble, corr(epsilon, epsilon_hat)

Usage:
    python scripts/residual_experiment_phase3_backfill.py [--dry-run]

Realized price source (priority):
  1. data/checkpoints/*data_extraction*.parquet covering [dataset.start, dataset.end+60d]
  2. data/testing/test_YYYYMMDD*.parquet covering [dataset.end+1, ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("phase3_backfill")

AUDIT_DIR = pathlib.Path("logs/forecast_audits")
CHECKPOINT_DIR = pathlib.Path("data/checkpoints")
TESTING_DIR = pathlib.Path("data/testing")

# Reason codes that indicate a residual correction was intentionally blocked by
# model-safety gates (not a runtime failure).
SKIPPED_BAD_SIGNAL_REASON_CODES = {
    "PHI_TOO_SMALL",
    "PHI_TOO_PERSISTENT",
    "TOO_FEW_OOS_POINTS",
    "NO_ANCHOR_RMSE_PROXY",
    "LONG_RUN_MEAN_TOO_LARGE",
    "POOR_TRAIN_DIRECTIONAL_USEFULNESS",
    "LOCAL_USEFULNESS_FAIL",
    "PREDICTED_RESIDUAL_TOO_CONSTANT",
    "PREDICTED_MEAN_TOO_LARGE",
    "ANCHOR_MISMATCH",
}


# ---------------------------------------------------------------------------
# Realized-price loading
# ---------------------------------------------------------------------------

def _load_all_realized_series() -> pd.Series:
    """Build a single Close series covering 2020-2024 from the widest checkpoint."""
    # Prefer the widest data extraction checkpoint
    best: Optional[Tuple[int, pathlib.Path]] = None
    for f in sorted(CHECKPOINT_DIR.glob("*data_extraction*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["Close"])
            span = (df.index[-1] - df.index[0]).days
            if best is None or span > best[0]:
                best = (span, f)
        except Exception:
            pass

    if best is not None:
        log.info("Using realized prices from checkpoint: %s", best[1].name)
        df = pd.read_parquet(best[1], columns=["Close"])
        df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)
        return df["Close"].sort_index()

    # Fallback: stitch non-normalized test parquets
    frames = []
    for f in sorted(p for p in TESTING_DIR.glob("test_2026*.parquet") if "norm" not in p.name):
        try:
            df = pd.read_parquet(f, columns=["Close"])
            df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)
            frames.append(df["Close"])
        except Exception:
            pass
    if frames:
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        log.info("Fallback: stitched %d test parquets -> %d rows", len(frames), len(combined))
        return combined

    raise RuntimeError("No realized price source found in checkpoints or testing/")


def load_realized_prices(
    realized_series: pd.Series,
    dataset_end: str,
    n_steps: int,
) -> np.ndarray:
    """Return n_steps Close values immediately after dataset_end."""
    cutoff = pd.Timestamp(dataset_end).normalize()
    future = realized_series[realized_series.index > cutoff]
    if len(future) < n_steps:
        raise ValueError(
            f"Only {len(future)} realized prices after {dataset_end}; need {n_steps}"
        )
    return future.values[:n_steps]


# ---------------------------------------------------------------------------
# Phase-3 metrics
# ---------------------------------------------------------------------------

def compute_phase3_metrics(
    y_hat_anchor: List[float],
    y_hat_resid_ens: List[float],
    realized: np.ndarray,
) -> dict:
    y_a = np.array(y_hat_anchor)
    y_r = np.array(y_hat_resid_ens)
    y_true = realized[: len(y_a)]

    # RMSE
    rmse_anchor = float(np.sqrt(np.mean((y_a - y_true) ** 2)))
    rmse_resid = float(np.sqrt(np.mean((y_r - y_true) ** 2)))
    rmse_ratio = float(rmse_resid / rmse_anchor) if rmse_anchor > 0 else None

    # Directional accuracy (fraction where sign(forecast_change) == sign(actual_change))
    actual_delta = np.diff(np.concatenate([[y_true[0]], y_true]))
    anchor_delta = np.diff(np.concatenate([[y_a[0]], y_a]))
    resid_delta = np.diff(np.concatenate([[y_r[0]], y_r]))
    da_anchor = float(np.mean(np.sign(anchor_delta) == np.sign(actual_delta)))
    da_resid = float(np.mean(np.sign(resid_delta) == np.sign(actual_delta)))

    # corr(epsilon[t], epsilon_hat[t])
    epsilon = y_true - y_a         # realized anchor errors
    epsilon_hat = y_r - y_a        # residual model's predicted corrections
    if len(epsilon) >= 2 and np.std(epsilon) > 0 and np.std(epsilon_hat) > 0:
        corr_eps = float(np.corrcoef(epsilon, epsilon_hat)[0, 1])
    else:
        corr_eps = None

    return {
        "rmse_anchor": rmse_anchor,
        "rmse_residual_ensemble": rmse_resid,
        "rmse_ratio": rmse_ratio,
        "da_anchor": da_anchor,
        "da_residual_ensemble": da_resid,
        "corr_anchor_residual": corr_eps,  # OVERWRITES phase-2 proxy
        "phase": 3,
    }


# ---------------------------------------------------------------------------
# Audit file handling
# ---------------------------------------------------------------------------

def _fingerprint(ds: dict) -> str:
    s = f"{ds.get('ticker','')}{ds.get('start','')}{ds.get('end','')}{ds.get('length','')}{ds.get('forecast_horizon','')}"
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def _is_skipped_bad_signal(residual_block: dict) -> bool:
    """Return True when the window was skipped by residual-signal safety gates."""
    reason_code = str(residual_block.get("reason_code") or "").upper()
    if reason_code in SKIPPED_BAD_SIGNAL_REASON_CODES:
        return True
    # Keep backward compatibility for artifacts that only emit free-text reason.
    skip_reason = str(residual_block.get("skip_reason") or "").lower()
    return any(
        token in skip_reason
        for token in (
            "phi_too_small",
            "high_phi_near_unit_root",
            "bias_dominated_long_run_mean",
            "poor_train_directional_usefulness",
            "local_usefulness_fail",
            "predicted residual too constant",
            "predicted residual mean too large",
            "anchor_mismatch",
        )
    )


def collect_unique_active_audits() -> Dict[str, Tuple[pathlib.Path, dict]]:
    """Dedup residual audits by fingerprint; keep most recent file per fingerprint."""
    best: Dict[str, Tuple[pathlib.Path, dict]] = {}
    for f in sorted(AUDIT_DIR.glob("forecast_audit_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            re = d.get("artifacts", {}).get("residual_experiment", {})
            if not isinstance(re, dict):
                continue
            status = str(re.get("residual_status") or "")
            if status == "active":
                if not re.get("y_hat_anchor"):
                    continue
            elif not _is_skipped_bad_signal(re):
                continue
            ds = d.get("dataset", {})
            fp = _fingerprint(ds)
            if fp not in best or f.name > best[fp][0].name:
                best[fp] = (f, d)
        except Exception:
            pass
    return best


def patch_audit_file(
    audit_path: pathlib.Path,
    metrics: dict,
    dry_run: bool = False,
) -> None:
    """Patch residual_experiment fields in-place."""
    d = json.loads(audit_path.read_text(encoding="utf-8"))
    re = d.setdefault("artifacts", {}).setdefault("residual_experiment", {})
    re.update(metrics)
    if dry_run:
        log.info(
            "[DRY-RUN] PATCHED_PHASE3 %s  rmse_ratio=%.4f",
            audit_path.name,
            metrics.get("rmse_ratio") or 0,
        )
        return
    audit_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    log.info(
        "PATCHED_PHASE3 %s  rmse_ratio=%.4f  corr=%.3f",
        audit_path.name,
        metrics.get("rmse_ratio") or 0,
        metrics.get("corr_anchor_residual") or 0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> int:
    log.info("=== EXP-R5-001 Phase 3 Backfill ===")

    # Load realized price series once
    try:
        realized_series = _load_all_realized_series()
        if len(realized_series) > 0:
            log.info("Realized price series: %d rows, %s to %s, Close=%.2f-%.2f",
                     len(realized_series),
                     realized_series.index[0].date(),
                     realized_series.index[-1].date(),
                     realized_series.min(), realized_series.max())
        else:
            log.info("Realized price series: 0 rows (empty - all windows will SKIP_PENDING_REALIZED)")
    except RuntimeError as exc:
        log.error("Cannot load realized prices: %s", exc)
        return 1

    # Collect unique active audits
    unique = collect_unique_active_audits()
    log.info("Unique active windows: %d", len(unique))
    if not unique:
        log.warning("No active audit windows found; nothing to patch.")
        return 0

    success = 0
    failed = 0
    skipped_no_realized = 0
    skipped_bad_signal = 0
    already_done = 0

    for fp, (audit_path, d) in sorted(unique.items(), key=lambda x: x[1][1].get("dataset", {}).get("end", "")):
        re = d["artifacts"]["residual_experiment"]
        ds = d.get("dataset", {})
        dataset_end = ds.get("end", "")
        forecast_horizon = ds.get("forecast_horizon", 30)
        y_hat_anchor = re.get("y_hat_anchor", [])
        y_hat_ens = re.get("y_hat_residual_ensemble", [])

        if _is_skipped_bad_signal(re):
            reason_code = str(re.get("reason_code") or "UNKNOWN")
            log.info(
                "  SKIPPED_BAD_SIGNAL: fp=%s end=%s reason_code=%s",
                fp, dataset_end[:10], reason_code,
            )
            skipped_bad_signal += 1
            continue

        if re.get("residual_status") != "active" or not y_hat_anchor:
            # Non-active windows that are not bad-signal skips are out of scope
            # for Phase 3 metric patching.
            continue

        # Skip if already Phase 3
        if re.get("phase") == 3 and re.get("rmse_anchor") is not None:
            log.info("ALREADY_PHASE3: fp=%s  %s", fp, audit_path.name)
            already_done += 1
            continue

        log.info("Processing fp=%s  end=%s  len=%d  horizon=%d",
                 fp, dataset_end[:10], ds.get("length", 0), forecast_horizon)

        try:
            realized = load_realized_prices(realized_series, dataset_end, forecast_horizon)
        except (ValueError, KeyError, IndexError) as exc:
            # No realized prices yet for this window - expected for recent end dates.
            # Recorded as SKIP_PENDING_REALIZED, not failure.
            log.info(
                "  SKIP_PENDING_REALIZED: fp=%s end=%s - %s",
                fp, dataset_end[:10], exc,
            )
            skipped_no_realized += 1
            continue

        try:
            metrics = compute_phase3_metrics(y_hat_anchor, y_hat_ens, realized)
            patch_audit_file(audit_path, metrics, dry_run=dry_run)
            success += 1
        except Exception as exc:
            log.error("  FAIL: fp=%s end=%s - %s", fp, dataset_end[:10], exc)
            failed += 1

    log.info(
        "=== Summary: %d patched, %d already_phase3, "
        "%d skip_pending_realized, %d skipped_bad_signal, %d true_failures ===",
        success, already_done, skipped_no_realized, skipped_bad_signal, failed,
    )
    if skipped_no_realized > 0:
        log.info(
            "  %d window(s) have no realized prices yet (end date beyond checkpoint). "
            "Re-run after fetching data past those dates.",
            skipped_no_realized,
        )

    if not dry_run and success > 0:
        log.info("Run quality pipeline to refresh summary:")
        log.info("  python scripts/run_quality_pipeline.py --audit-dir logs/forecast_audits --enable-residual-experiment")

    # Exit 0 when the only non-success outcomes are SKIP_PENDING_REALIZED.
    # True failures (unexpected exceptions in compute_phase3_metrics or patch_audit_file)
    # are the only reason to exit non-zero.
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-R5-001 Phase 3 realized-price backfill")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics but do not write to disk")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
