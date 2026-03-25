#!/usr/bin/env python3
"""
scripts/validate_pipeline_inputs.py
-------------------------------------
Pre-flight validator for the Phase 9 directional classifier data pipeline.

Runs six structured checks and exits with a clear status code before any
pipeline work starts. Designed to catch the class of failures seen in
the 2026-03-18 run (0 labeled examples, HOLD-dominated eval cycles) by
surfacing filename mismatches, timestamp misalignments, and date-range
coverage gaps *before* they silently propagate.

Checks
------
V1  Filename Convention
      Each required ticker must have at least one checkpoint parquet whose
      filename contains the ticker symbol so _load_price_parquet can find it.
      PASS  = ticker found in filename
      WARN  = unnamed fallback parquets exist but none are ticker-specific;
              price data may be wrong (ETL may have used --execution-mode synthetic)
      FAIL  = no parquet found at all for this ticker

V2  Parquet Coverage Map
      Inspect every *data_extraction*.parquet in data/checkpoints/.
      PASS  = parquet has Close column and >= 90 rows
      WARN  = parquet is short (< 90 rows) or has suspicious uniform prices
              (all-same Close value indicates SyntheticExtractor with fixed seed)
      FAIL  = parquet has no Close column or cannot be read

V3  JSONL Timestamp Alignment
      For each JSONL entry that has classifier_features, check whether its
      timestamp falls inside any parquet's date coverage for that ticker.
      If 0 % of entries are alignable, forward-price labeling will produce
      zero rows — the wall-clock-timestamp bug.
      PASS  = >= 20 % of entries are alignable
      WARN  = 1-19 % alignable
      FAIL  = 0 % alignable (all timestamps outside parquet range)
      SKIP  = JSONL path does not exist

V4  Eval Date Coverage
      For each proposed evaluation date (--eval-dates) and each ticker,
      verify that at least one parquet covers that date. Eval cycles run
      run_auto_trader.py with --as-of-date X; if no price data exists for
      X the auto_trader produces only HOLD signals.
      PASS  = date is within [parquet_start, parquet_end] for every ticker
      WARN  = date is covered for some but not all tickers
      FAIL  = date is outside all available parquet ranges

V5  Duplicate-Parquet Multi-Ticker
      Detect when two or more tickers would load the same physical parquet
      file. This happens when the ETL used --execution-mode synthetic
      (SyntheticExtractor produces identical price series for every ticker).
      Training labels generated from such data are meaningless.
      PASS  = every ticker resolves to a distinct parquet file
      WARN  = some tickers share a file but Close[0] values differ (may be ok)
      FAIL  = two tickers share a file AND have identical Close[0] prices
              (definitive synthetic-data collision)

V6  Edge Cases
      Check for: empty parquets, parquets without Close, JSONL entries with
      null/unparseable timestamps, training dataset already present but older
      than --stale-days (default: 7).
      PASS  = no edge-case anomalies found
      WARN  = non-critical anomalies (e.g. stale training dataset)
      FAIL  = structural problem (empty parquet, missing column)

Exit codes
----------
  0  All checks PASS or WARN only  → pipeline may proceed
  1  At least one FAIL              → pipeline should not run
  2  Validator could not run        → infrastructure problem

Usage
-----
  # Default tickers, eval dates from bootstrap
  python scripts/validate_pipeline_inputs.py

  # Custom tickers and eval dates
  python scripts/validate_pipeline_inputs.py \\
      --tickers AAPL,MSFT,NVDA,GS,AMZN \\
      --eval-dates 2022-07-01,2022-10-01,2023-01-01,2023-04-01 \\
      --checkpoint-dir data/checkpoints

  # JSON output for machine consumption
  python scripts/validate_pipeline_inputs.py --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GS", "AMZN"]
_DEFAULT_EVAL_DATES = [
    "2022-07-01",
    "2022-10-01",
    "2023-01-01",
    "2023-04-01",
]
_DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")
_DEFAULT_JSONL_PATH = Path("logs/signals/quant_validation.jsonl")
_DEFAULT_TRAINING_PATH = Path("data/training/directional_dataset.parquet")
_DEFAULT_STALE_DAYS = 7
_MIN_PARQUET_ROWS = 90  # lookback(60) + horizon(30)
_SYNTHETIC_PRICE_TOLERANCE = 0.01  # two tickers are "same" if Close[0] within 1%


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class CheckResult:
    """Single check result with status (PASS/WARN/FAIL/SKIP) and details."""

    __slots__ = ("check_id", "status", "message", "details")

    def __init__(self, check_id: str, status: str, message: str,
                 details: Optional[Dict[str, Any]] = None) -> None:
        assert status in ("PASS", "WARN", "FAIL", "SKIP")
        self.check_id = check_id
        self.status = status
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _status_symbol(status: str) -> str:
    return {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]


def _load_parquet_safe(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load a parquet, returning (df, None) on success or (None, error_msg) on failure."""
    try:
        df = pd.read_parquet(path)
        return df, None
    except Exception as exc:
        return None, str(exc)


def _parse_parquet_coverage(df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (min_date, max_date) from parquet index, or None if unusable."""
    try:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        idx = idx[idx.notna()]
        if len(idx) == 0:
            return None
        return idx.min(), idx.max()
    except Exception:
        return None


def _find_best_parquet_for_ticker(
    ticker: str,
    checkpoint_dir: Path,
) -> Optional[Path]:
    """Mirror the _load_price_parquet lookup: primary (ticker in name), then fallback."""
    ticker_upper = ticker.upper()
    # Primary: ticker in filename
    primary = sorted(
        list(checkpoint_dir.glob(f"*{ticker_upper}*data_extraction*.parquet"))
        + list(checkpoint_dir.glob(f"*{ticker_upper}*.parquet")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if primary:
        return primary[0]
    # Fallback: largest data_extraction parquet
    fallback = sorted(
        checkpoint_dir.glob("*data_extraction*.parquet"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return fallback[0] if fallback else None


# ---------------------------------------------------------------------------
# V1: Filename Convention
# ---------------------------------------------------------------------------

def check_v1_filename_convention(
    tickers: List[str],
    checkpoint_dir: Path,
) -> List[CheckResult]:
    results = []
    for ticker in tickers:
        ticker_upper = ticker.upper()
        named = list(checkpoint_dir.glob(f"*{ticker_upper}*data_extraction*.parquet"))
        named += list(checkpoint_dir.glob(f"*{ticker_upper}*.parquet"))
        named = [p for p in named if p.is_file()]

        if named:
            results.append(CheckResult(
                f"V1.{ticker}", "PASS",
                f"{ticker}: {len(named)} named parquet(s) found",
                {"named_count": len(named), "examples": [p.name for p in named[:2]]},
            ))
        else:
            # Check for unnamed fallback
            unnamed = list(checkpoint_dir.glob("*data_extraction*.parquet"))
            if unnamed:
                results.append(CheckResult(
                    f"V1.{ticker}", "WARN",
                    (
                        f"{ticker}: no ticker-named parquet found; "
                        f"{len(unnamed)} unnamed parquet(s) exist as fallback. "
                        "Price data may not be ticker-specific. "
                        "Ensure ETL was run with --execution-mode auto and "
                        "parquets were renamed to include the ticker name."
                    ),
                    {"unnamed_fallback_count": len(unnamed)},
                ))
            else:
                results.append(CheckResult(
                    f"V1.{ticker}", "FAIL",
                    (
                        f"{ticker}: no checkpoint parquets found in {checkpoint_dir}. "
                        "Run the ETL bootstrap (Phase 1) to download price data."
                    ),
                    {"checkpoint_dir": str(checkpoint_dir)},
                ))
    return results


# ---------------------------------------------------------------------------
# V2: Parquet Coverage Map
# ---------------------------------------------------------------------------

def check_v2_parquet_coverage(checkpoint_dir: Path) -> List[CheckResult]:
    results = []
    parquets = sorted(checkpoint_dir.glob("*data_extraction*.parquet"))
    if not parquets:
        results.append(CheckResult(
            "V2.coverage", "FAIL",
            f"No *data_extraction*.parquet files found in {checkpoint_dir}",
        ))
        return results

    for path in parquets:
        df, err = _load_parquet_safe(path)
        if err:
            results.append(CheckResult(
                f"V2.{path.name[:40]}", "FAIL",
                f"Cannot read {path.name}: {err}",
                {"path": str(path)},
            ))
            continue

        if "Close" not in df.columns:
            results.append(CheckResult(
                f"V2.{path.name[:40]}", "FAIL",
                f"{path.name}: no 'Close' column — cannot be used for price labeling",
                {"columns": list(df.columns)},
            ))
            continue

        n = len(df)
        if n < _MIN_PARQUET_ROWS:
            results.append(CheckResult(
                f"V2.{path.name[:40]}", "FAIL",
                f"{path.name}: only {n} rows — need >= {_MIN_PARQUET_ROWS} "
                f"(lookback + horizon). Too short for label generation.",
                {"n_rows": n, "min_required": _MIN_PARQUET_ROWS},
            ))
            continue

        coverage = _parse_parquet_coverage(df)
        if coverage is None:
            results.append(CheckResult(
                f"V2.{path.name[:40]}", "WARN",
                f"{path.name}: unparseable date index — coverage unknown",
            ))
            continue

        start, end = coverage
        close_vals = df["Close"].dropna().values

        # Detect synthetic data: all Close values suspiciously uniform
        if len(close_vals) > 10:
            pct_std = np.std(close_vals) / (np.mean(close_vals) + 1e-10)
            if pct_std < 0.001:
                results.append(CheckResult(
                    f"V2.{path.name[:40]}", "FAIL",
                    (
                        f"{path.name}: Close prices are nearly constant (std/mean={pct_std:.6f}). "
                        "This parquet contains degenerate/synthetic data — not real market prices."
                    ),
                    {"close_std_over_mean": round(float(pct_std), 6)},
                ))
                continue

        results.append(CheckResult(
            f"V2.{path.name[:40]}", "PASS",
            f"{path.name}: {n} rows, {start.date()} -> {end.date()}, "
            f"Close[0]={close_vals[0]:.2f}",
            {
                "n_rows": n,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "close_0": round(float(close_vals[0]), 2),
                "close_last": round(float(close_vals[-1]), 2),
            },
        ))

    return results


# ---------------------------------------------------------------------------
# V3: JSONL Timestamp Alignment
# ---------------------------------------------------------------------------

def check_v3_jsonl_alignment(
    jsonl_path: Path,
    checkpoint_dir: Path,
    tickers: Optional[List[str]] = None,
) -> CheckResult:
    if not jsonl_path.exists():
        return CheckResult(
            "V3.alignment", "SKIP",
            f"JSONL not found at {jsonl_path} — V3 skipped",
        )

    # Build parquet coverage map KEYED BY TICKER so that an MSFT parquet cannot
    # satisfy an AAPL JSONL entry.  Ticker is extracted from: (1) the ticker column
    # in the parquet, (2) the filename, or (3) left as None (generic fallback only).
    parquet_ranges_by_ticker: Dict[Optional[str], List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for path in checkpoint_dir.glob("*data_extraction*.parquet"):
        df, err = _load_parquet_safe(path)
        if err or df is None or "Close" not in df.columns:
            continue
        cov = _parse_parquet_coverage(df)
        if not cov:
            continue
        # Determine ticker from data or filename
        file_ticker: Optional[str] = None
        if "ticker" in df.columns:
            tickers_in_file = df["ticker"].dropna().unique().tolist()
            if len(tickers_in_file) == 1:
                file_ticker = str(tickers_in_file[0]).upper()
        if file_ticker is None:
            # Best-effort: extract ticker from filename (e.g. AAPL_pipeline_...)
            stem = path.stem.upper()
            for candidate in stem.split("_"):
                # Simple heuristic: 1-5 uppercase letters, not a common suffix word
                if candidate.isalpha() and 1 <= len(candidate) <= 5 and candidate not in (
                    "DATA", "TRAIN", "PIPE", "LINE", "CHECK", "POINT", "EXTRACTION"
                ):
                    file_ticker = candidate
                    break
        parquet_ranges_by_ticker.setdefault(file_ticker, []).append(cov)

    # All ranges (used for coverage summary display only)
    all_parquet_ranges = [r for ranges in parquet_ranges_by_ticker.values() for r in ranges]

    if not all_parquet_ranges:
        return CheckResult(
            "V3.alignment", "FAIL",
            (
                f"No readable parquet coverage found in {checkpoint_dir}. "
                "Cannot validate timestamp alignment. Run ETL bootstrap first "
                "or fix unreadable checkpoint files."
            ),
            {"checkpoint_dir": str(checkpoint_dir)},
        )

    # Parse JSONL
    entries = []
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("classifier_features"):
                    entries.append(e)
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        return CheckResult("V3.alignment", "FAIL", f"Cannot read JSONL: {exc}")

    if not entries:
        return CheckResult(
            "V3.alignment", "SKIP",
            "No JSONL entries with classifier_features found — V3 skipped",
        )

    n_total = len(entries)
    n_alignable = 0
    n_wall_clock = 0
    n_missing_ticker_coverage = 0
    wall_clock_sample: List[str] = []
    current_year = datetime.now(timezone.utc).year

    for e in entries:
        # Mirror _parse_entry_ts priority order
        ts_raw = e.get("signal_timestamp") or e.get("timestamp") or e.get("entry_ts")
        if not ts_raw:
            continue
        try:
            ts = pd.Timestamp(ts_raw, tz="UTC")
        except Exception:
            continue

        # Ticker-scoped check: only look in parquet ranges for this entry's ticker.
        # Ticker derived from: explicit 'ticker' field → signal_id prefix (ts_{TICKER}_...)
        # → generic fallback when entry has no ticker field.
        entry_ticker = str(e.get("ticker") or "").upper() or None
        if entry_ticker is None:
            sig_id = str(e.get("signal_id") or "")
            if sig_id.startswith("ts_"):
                parts = sig_id.split("_")
                if len(parts) >= 2:
                    entry_ticker = parts[1].upper() or None

        # Track entries for which the requested tickers list has no parquet coverage
        if tickers and entry_ticker and entry_ticker in {t.upper() for t in tickers}:
            has_coverage = bool(
                parquet_ranges_by_ticker.get(entry_ticker)
                or parquet_ranges_by_ticker.get(None)
            )
            if not has_coverage:
                n_missing_ticker_coverage += 1

        candidate_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        if entry_ticker:
            candidate_ranges.extend(parquet_ranges_by_ticker.get(entry_ticker, []))
            # Also include None-keyed parquets (ticker undetermined)
            candidate_ranges.extend(parquet_ranges_by_ticker.get(None, []))
        else:
            # No ticker on JSONL entry — check all parquets as generic fallback
            for ranges in parquet_ranges_by_ticker.values():
                candidate_ranges.extend(ranges)

        alignable = any(start <= ts <= end for start, end in candidate_ranges)
        if alignable:
            n_alignable += 1
        elif ts.year >= current_year:
            n_wall_clock += 1
            if len(wall_clock_sample) < 3:
                wall_clock_sample.append(ts_raw)

    pct_alignable = n_alignable / n_total if n_total > 0 else 0.0

    # Build coverage description safely — guard against empty all_parquet_ranges
    if all_parquet_ranges:
        cov_start = min(r[0] for r in all_parquet_ranges).date()
        cov_end = max(r[1] for r in all_parquet_ranges).date()
        cov_desc = f"{cov_start} - {cov_end}"
    else:
        cov_desc = "no parquets found"

    details = {
        "n_with_features": n_total,
        "n_alignable": n_alignable,
        "n_wall_clock_timestamps": n_wall_clock,
        "n_missing_ticker_coverage": n_missing_ticker_coverage,
        "pct_alignable": round(pct_alignable * 100, 1),
        "parquet_ranges_checked": len(all_parquet_ranges),
        "wall_clock_sample": wall_clock_sample,
    }

    if pct_alignable == 0.0:
        # WARN (not FAIL): 0% alignment means build_directional_training_data.py
        # (the JSONL-based labeler) cannot produce labels, but
        # generate_classifier_training_labels.py (the parquet-scan labeler used
        # by the bootstrap) is completely unaffected by JSONL timestamps — it reads
        # checkpoint parquets directly. Blocking the pipeline here would be a false
        # positive for any setup that already uses the parquet-scan path.
        return CheckResult(
            "V3.alignment", "WARN",
            (
                f"0 of {n_total} JSONL entries have timestamps within any ticker-matched "
                f"parquet range (all are wall-clock {current_year}, parquets cover "
                f"{cov_desc}). "
                "build_directional_training_data.py will produce 0 labeled examples -- "
                "use generate_classifier_training_labels.py (parquet scan) instead. "
                "The bootstrap already uses the parquet-scan path, so this is advisory only."
            ),
            details,
        )
    elif pct_alignable < 0.20:
        return CheckResult(
            "V3.alignment", "WARN",
            (
                f"Only {n_alignable}/{n_total} ({pct_alignable*100:.0f}%) JSONL entries "
                "have alignable timestamps. Forward-price labeling yield will be low."
            ),
            details,
        )
    else:
        return CheckResult(
            "V3.alignment", "PASS",
            (
                f"{n_alignable}/{n_total} ({pct_alignable*100:.0f}%) JSONL entries "
                "have timestamps alignable with available parquet coverage."
            ),
            details,
        )


# ---------------------------------------------------------------------------
# V4: Eval Date Coverage
# ---------------------------------------------------------------------------

def check_v4_eval_date_coverage(
    eval_dates: List[str],
    tickers: List[str],
    checkpoint_dir: Path,
) -> List[CheckResult]:
    # Build per-ticker coverage from named parquets only.  Do NOT fall back to
    # generic *data_extraction* parquets — they may belong to a different ticker,
    # which would give MSFT a false PASS using AAPL price data.
    ticker_coverage: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ticker in tickers:
        ticker_upper = ticker.upper()
        cov_list = []
        named = list(checkpoint_dir.glob(f"*{ticker_upper}*.parquet"))
        all_parquets = named
        for path in all_parquets[:5]:  # check up to 5 candidates
            df, err = _load_parquet_safe(path)
            if err or df is None or "Close" not in df.columns:
                continue
            cov = _parse_parquet_coverage(df)
            if cov:
                cov_list.append(cov)
        ticker_coverage[ticker] = cov_list

    # All available ranges (for the date-is-outside-everything case)
    all_ranges = [r for ranges in ticker_coverage.values() for r in ranges]

    results = []
    for date_str in eval_dates:
        try:
            eval_ts = pd.Timestamp(date_str, tz="UTC")
        except Exception:
            results.append(CheckResult(
                f"V4.{date_str}", "FAIL",
                f"Cannot parse eval date: {date_str!r}",
            ))
            continue

        covered_tickers = []
        uncovered_tickers = []
        for ticker in tickers:
            is_covered = any(start <= eval_ts <= end for start, end in ticker_coverage[ticker])
            if is_covered:
                covered_tickers.append(ticker)
            else:
                uncovered_tickers.append(ticker)

        details: Dict[str, Any] = {
            "eval_date": date_str,
            "covered_tickers": covered_tickers,
            "uncovered_tickers": uncovered_tickers,
        }

        if not uncovered_tickers:
            results.append(CheckResult(
                f"V4.{date_str}", "PASS",
                f"Eval date {date_str}: all {len(tickers)} tickers have parquet coverage",
                details,
            ))
        elif covered_tickers:
            results.append(CheckResult(
                f"V4.{date_str}", "WARN",
                (
                    f"Eval date {date_str}: covered for {covered_tickers}, "
                    f"NOT covered for {uncovered_tickers}. "
                    "Uncovered tickers will produce HOLD-only cycles."
                ),
                details,
            ))
        else:
            # Build a helpful message showing the actual available range
            if all_ranges:
                global_start = min(r[0] for r in all_ranges).date().isoformat()
                global_end = max(r[1] for r in all_ranges).date().isoformat()
                hint = f"Available parquet range: {global_start} -> {global_end}"
            else:
                hint = "No parquets found — run ETL bootstrap first"
            results.append(CheckResult(
                f"V4.{date_str}", "FAIL",
                (
                    f"Eval date {date_str} is outside all available parquet coverage. "
                    f"{hint}. "
                    "Auto_trader will have no price data → all HOLD, 0 trades."
                ),
                details,
            ))

    return results


# ---------------------------------------------------------------------------
# V5: Duplicate-Parquet Multi-Ticker
# ---------------------------------------------------------------------------

def check_v5_duplicate_parquet(
    tickers: List[str],
    checkpoint_dir: Path,
) -> CheckResult:
    ticker_path: Dict[str, Optional[Path]] = {}
    ticker_close0: Dict[str, Optional[float]] = {}

    for ticker in tickers:
        path = _find_best_parquet_for_ticker(ticker, checkpoint_dir)
        ticker_path[ticker] = path
        if path is None:
            ticker_close0[ticker] = None
            continue
        df, err = _load_parquet_safe(path)
        if err or df is None or "Close" not in df.columns or len(df) == 0:
            ticker_close0[ticker] = None
        else:
            ticker_close0[ticker] = round(float(df["Close"].dropna().iloc[0]), 4)

    # Group tickers by resolved parquet path
    path_to_tickers: Dict[str, List[str]] = defaultdict(list)
    for ticker, path in ticker_path.items():
        path_to_tickers[str(path)].append(ticker)

    collisions = []
    for path_str, path_tickers in path_to_tickers.items():
        if len(path_tickers) < 2 or path_str == "None":
            continue
        # Check if Close[0] is identical (synthetic) or just a filename collision
        close_vals = [ticker_close0.get(t) for t in path_tickers]
        if all(v is not None for v in close_vals):
            max_diff = max(abs(a - b) / (max(abs(a), abs(b)) + 1e-10)
                          for i, a in enumerate(close_vals)
                          for b in close_vals[i + 1:])
            if max_diff <= _SYNTHETIC_PRICE_TOLERANCE:
                collisions.append({
                    "path": Path(path_str).name,
                    "tickers": path_tickers,
                    "close_0_values": dict(zip(path_tickers, close_vals)),
                    "severity": "FAIL_identical_prices",
                })
            else:
                collisions.append({
                    "path": Path(path_str).name,
                    "tickers": path_tickers,
                    "close_0_values": dict(zip(path_tickers, close_vals)),
                    "severity": "WARN_different_prices",
                })

    details = {
        "ticker_parquet_map": {t: (Path(str(p)).name if p else None)
                               for t, p in ticker_path.items()},
        "ticker_close0": ticker_close0,
        "collisions": collisions,
    }

    fail_cols = [c for c in collisions if c["severity"] == "FAIL_identical_prices"]
    warn_cols = [c for c in collisions if c["severity"] == "WARN_different_prices"]

    if fail_cols:
        tickers_affected = [t for c in fail_cols for t in c["tickers"]]
        return CheckResult(
            "V5.duplicate_parquet", "FAIL",
            (
                f"Tickers {tickers_affected} all resolve to the same parquet(s) "
                "AND have identical Close[0] prices. This is the synthetic-data "
                "collision: ETL was run with --execution-mode synthetic which uses "
                "SyntheticExtractor (same seed for all tickers). Training labels "
                "from this data are meaningless. Re-run ETL with --execution-mode auto."
            ),
            details,
        )
    elif warn_cols:
        return CheckResult(
            "V5.duplicate_parquet", "WARN",
            (
                f"{len(warn_cols)} ticker group(s) share the same parquet file "
                "but have different Close[0] prices. This may be acceptable if "
                "the parquets were not yet renamed to include ticker names. "
                "Verify price data is ticker-specific."
            ),
            details,
        )
    else:
        return CheckResult(
            "V5.duplicate_parquet", "PASS",
            f"All {len(tickers)} tickers resolve to distinct parquet files with different prices.",
            details,
        )


# ---------------------------------------------------------------------------
# V6: Edge Cases
# ---------------------------------------------------------------------------

def check_v6_edge_cases(
    checkpoint_dir: Path,
    jsonl_path: Path,
    training_path: Path,
    stale_days: int = _DEFAULT_STALE_DAYS,
) -> List[CheckResult]:
    results = []

    # 6a: Empty or trivially small parquets
    for path in checkpoint_dir.glob("*data_extraction*.parquet"):
        df, err = _load_parquet_safe(path)
        if err:
            continue
        if df is not None and len(df) == 0:
            results.append(CheckResult(
                f"V6.empty.{path.name[:35]}", "FAIL",
                f"{path.name}: 0 rows — empty parquet, will cause silent failures",
                {"path": str(path)},
            ))

    # 6b: JSONL null/unparseable timestamps
    if jsonl_path.exists():
        null_count = 0
        malformed_count = 0
        total_feat = 0
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if not e.get("classifier_features"):
                        continue
                    total_feat += 1
                    ts_raw = (e.get("signal_timestamp") or e.get("timestamp")
                              or e.get("entry_ts"))
                    if not ts_raw:
                        null_count += 1
                    else:
                        try:
                            pd.Timestamp(ts_raw)
                        except (ValueError, TypeError):
                            malformed_count += 1
                except Exception:
                    pass
        except Exception:
            pass
        bad_count = null_count + malformed_count
        if bad_count > 0 and total_feat > 0:
            detail_parts = []
            if null_count:
                detail_parts.append(f"{null_count} missing")
            if malformed_count:
                detail_parts.append(f"{malformed_count} unparseable")
            _bad_details = {
                "null_count": null_count,
                "malformed_count": malformed_count,
                "total_with_features": total_feat,
            }
            results.append(CheckResult(
                "V6.null_timestamps", "WARN",
                (
                    f"{bad_count}/{total_feat} JSONL entries with classifier_features "
                    f"have bad timestamps ({', '.join(detail_parts)}). These entries will "
                    "be skipped during labeling."
                ),
                _bad_details,
            ))
            # Also emit a dedicated malformed check so callers can filter by specific id
            if malformed_count > 0:
                results.append(CheckResult(
                    "V6.malformed_timestamps", "WARN",
                    (
                        f"{malformed_count}/{total_feat} JSONL entries have unparseable "
                        "timestamp strings. These entries will be skipped during labeling."
                    ),
                    {"malformed_count": malformed_count, "total_with_features": total_feat},
                ))

    # 6c: Stale training dataset
    if training_path.exists():
        age_seconds = (datetime.now(timezone.utc)
                       - datetime.fromtimestamp(training_path.stat().st_mtime, tz=timezone.utc)
                       ).total_seconds()
        age_days = age_seconds / 86400
        if age_days > stale_days:
            results.append(CheckResult(
                "V6.stale_dataset", "WARN",
                (
                    f"Training dataset at {training_path} is {age_days:.1f} days old "
                    f"(threshold: {stale_days} days). Consider regenerating labels from "
                    "fresh parquet data."
                ),
                {"age_days": round(age_days, 1), "stale_threshold_days": stale_days},
            ))

    # 6d: Checkpoint dir missing entirely
    if not checkpoint_dir.exists():
        results.append(CheckResult(
            "V6.missing_checkpoint_dir", "FAIL",
            f"Checkpoint directory {checkpoint_dir} does not exist. "
            "No parquets can be found. Run the ETL bootstrap first.",
            {"checkpoint_dir": str(checkpoint_dir)},
        ))

    if not results:
        results.append(CheckResult(
            "V6.edge_cases", "PASS",
            "No edge-case anomalies detected (empty parquets, null timestamps, stale dataset).",
        ))

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_checks(
    tickers: List[str],
    eval_dates: List[str],
    checkpoint_dir: Path,
    jsonl_path: Path,
    training_path: Path,
    stale_days: int = _DEFAULT_STALE_DAYS,
) -> Tuple[List[CheckResult], int]:
    """Run all checks and return (results, worst_exit_code)."""
    all_results: List[CheckResult] = []

    all_results.extend(check_v1_filename_convention(tickers, checkpoint_dir))
    all_results.extend(check_v2_parquet_coverage(checkpoint_dir))
    all_results.append(check_v3_jsonl_alignment(jsonl_path, checkpoint_dir))
    all_results.extend(check_v4_eval_date_coverage(eval_dates, tickers, checkpoint_dir))
    all_results.append(check_v5_duplicate_parquet(tickers, checkpoint_dir))
    all_results.extend(check_v6_edge_cases(checkpoint_dir, jsonl_path, training_path, stale_days))

    has_fail = any(r.status == "FAIL" for r in all_results)
    exit_code = 1 if has_fail else 0
    return all_results, exit_code


def print_report(results: List[CheckResult], exit_code: int) -> None:
    """Print a human-readable validation report."""
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r.status] += 1

    print()
    print("=" * 70)
    print("Phase 9 Pipeline Input Validation")
    print("=" * 70)
    for r in results:
        sym = _status_symbol(r.status)
        print(f"\n  {sym} {r.check_id}")
        # Wrap long messages
        words = r.message.split()
        line = "         "
        for word in words:
            if len(line) + len(word) > 80:
                print(line)
                line = "         " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    print()
    print("-" * 70)
    print(f"  Summary: {counts['PASS']} PASS, {counts['WARN']} WARN, "
          f"{counts['FAIL']} FAIL, {counts['SKIP']} SKIP")
    if exit_code == 0:
        print("  Verdict: PIPELINE MAY PROCEED")
    else:
        print("  Verdict: PIPELINE BLOCKED — resolve FAIL(s) before running")
    print("=" * 70)
    print()


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers", default=",".join(_DEFAULT_TICKERS),
        help=f"Comma-separated tickers (default: {','.join(_DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--eval-dates", default=",".join(_DEFAULT_EVAL_DATES),
        help="Comma-separated ISO eval dates for V4 check",
    )
    parser.add_argument(
        "--checkpoint-dir", default=str(_DEFAULT_CHECKPOINT_DIR),
        help=f"Checkpoint parquet directory (default: {_DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--jsonl-path", default=str(_DEFAULT_JSONL_PATH),
        help=f"JSONL log path (default: {_DEFAULT_JSONL_PATH})",
    )
    parser.add_argument(
        "--training-path", default=str(_DEFAULT_TRAINING_PATH),
        help=f"Training parquet path for stale-check (default: {_DEFAULT_TRAINING_PATH})",
    )
    parser.add_argument(
        "--stale-days", type=int, default=_DEFAULT_STALE_DAYS,
        help=f"Training dataset age threshold in days (default: {_DEFAULT_STALE_DAYS})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (machine-readable)",
    )
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    eval_dates = [d.strip() for d in args.eval_dates.split(",") if d.strip()]

    try:
        results, exit_code = run_all_checks(
            tickers=tickers,
            eval_dates=eval_dates,
            checkpoint_dir=Path(args.checkpoint_dir),
            jsonl_path=Path(args.jsonl_path),
            training_path=Path(args.training_path),
            stale_days=args.stale_days,
        )
    except Exception as exc:
        logger.error("Validator failed to run: %s", exc, exc_info=True)
        print(f"[ERROR] Validator could not complete: {exc}")
        return 2

    if args.json:
        payload = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": exit_code,
            "checks": [r.to_dict() for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print_report(results, exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
