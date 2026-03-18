"""
accumulate_classifier_labels.py
--------------------------------
JSONL-path classifier label accumulator.

Reads quant_validation.jsonl for BUY/SELL entries that have classifier_features,
joins them to production_closed_trades in the DB to get realized_pnl as ground truth,
and appends new outcome-linked labeled rows to directional_dataset.parquet.

Unlike the parquet-scan labeler (generate_classifier_training_labels.py) which uses
forward return sign as a proxy label, this script uses actual realized_pnl — meaning
slippage, stops, and holding period are all reflected in the label.

Label encoding:
  y_directional = 1  if  realized_pnl > 0  (profitable round-trip)
  y_directional = 0  if  realized_pnl <= 0 (loss or scratch)

label_source = "outcome_linked"  (distinguishes from parquet-scan rows)

Usage:
  python scripts/accumulate_classifier_labels.py [--dry-run] [--json]
  python scripts/accumulate_classifier_labels.py --db data/portfolio_maximizer.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_JSONL_PATH = _REPO_ROOT / "logs" / "signals" / "quant_validation.jsonl"
_DATASET_PATH = _REPO_ROOT / "data" / "training" / "directional_dataset.parquet"
_DB_PATH = _REPO_ROOT / "data" / "portfolio_maximizer.db"

_FEATURE_NAMES: List[str] = [
    "ensemble_pred_return",
    "ci_width_normalized",
    "snr",
    "model_agreement",
    "directional_vote_fraction",
    "garch_conf",
    "samossa_conf",
    "mssa_rl_conf",
    "igarch_fallback_flag",
    "samossa_evr",
    "hurst_exponent",
    "trend_strength",
    "realized_vol_annualized",
    "adf_pvalue",
    "regime_liquid_rangebound",
    "regime_moderate_trending",
    "regime_high_vol_trending",
    "regime_crisis",
    "recent_return_5d",
    "recent_vol_ratio",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_outcome_map(db_path: Path) -> Tuple[Dict[str, float], str]:
    """
    Returns (outcome_map, status) where status is one of:
      "ok"         — DB found and queried successfully (map may be empty if no closed trades)
      "db_missing" — DB file does not exist (map is empty; live cycles needed)
      "db_error"   — DB exists but query failed (map is empty; investigate)

    Callers MUST check status to distinguish "no trades yet" from "DB broken".
    """
    if not db_path.exists():
        logger.warning("DB not found: %s — no outcome-linked labels possible until live cycles run", db_path)
        return {}, "db_missing"
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            """
            SELECT ts_signal_id, realized_pnl
            FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_diagnostic, 0) = 0
              AND COALESCE(is_synthetic, 0) = 0
              AND ts_signal_id IS NOT NULL
              AND ts_signal_id NOT LIKE 'legacy_%'
              AND realized_pnl IS NOT NULL
            """
        ).fetchall()
        conn.close()
        return {r[0]: float(r[1]) for r in rows}, "ok"
    except Exception as exc:
        logger.error("DB query failed (%s): %s", db_path, exc)
        return {}, "db_error"


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def _load_jsonl_candidates(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Returns JSONL entries that are BUY/SELL signals with classifier_features
    and a non-null signal_id.
    """
    if not jsonl_path.exists():
        logger.warning("JSONL not found: %s", jsonl_path)
        return []
    candidates: List[Dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("action") not in ("BUY", "SELL"):
                continue
            if not entry.get("classifier_features"):
                continue
            if not entry.get("signal_id"):
                continue
            candidates.append(entry)
    return candidates


# ---------------------------------------------------------------------------
# Core accumulation
# ---------------------------------------------------------------------------

def accumulate(
    jsonl_path: Path = _JSONL_PATH,
    dataset_path: Path = _DATASET_PATH,
    db_path: Path = _DB_PATH,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Join JSONL classifier_features to DB outcomes and append to dataset.

    Returns a summary dict with keys:
      n_candidates, n_matched, n_new, n_total, n_by_source, feature_fill_rates
    """
    outcome_map, db_status = _load_outcome_map(db_path)
    candidates = _load_jsonl_candidates(jsonl_path)

    logger.info(
        "Loaded %d outcome-linked trades, %d JSONL candidates with features",
        len(outcome_map),
        len(candidates),
    )

    # Load existing dataset to dedup
    existing_ids: set = set()
    existing_df: Optional[pd.DataFrame] = None
    if dataset_path.exists():
        try:
            existing_df = pd.read_parquet(dataset_path)
            if "ts_signal_id" in existing_df.columns:
                existing_ids = set(existing_df["ts_signal_id"].dropna().tolist())
        except Exception as exc:
            logger.warning("Could not load existing dataset: %s", exc)

    # Build new rows
    new_rows: List[Dict[str, Any]] = []
    n_matched = 0
    n_skipped_no_outcome = 0
    n_skipped_duplicate = 0

    for entry in candidates:
        signal_id = entry["signal_id"]

        if signal_id in existing_ids:
            n_skipped_duplicate += 1
            continue

        if signal_id not in outcome_map:
            n_skipped_no_outcome += 1
            continue

        n_matched += 1
        realized_pnl = outcome_map[signal_id]
        y_directional = 1 if realized_pnl > 0 else 0

        feat = entry["classifier_features"]
        row: Dict[str, Any] = {
            "ts_signal_id": signal_id,
            "ticker": entry.get("ticker", "UNKNOWN"),
            "entry_ts": entry.get("timestamp", ""),
            "action": entry.get("action", ""),
            "y_directional": y_directional,
            "label_source": "outcome_linked",
            "realized_pnl": realized_pnl,
        }
        for fname in _FEATURE_NAMES:
            row[fname] = feat.get(fname, float("nan"))

        new_rows.append(row)
        existing_ids.add(signal_id)

    logger.info(
        "Matched %d new outcome-linked labels "
        "(%d skipped: no outcome, %d duplicate)",
        n_matched,
        n_skipped_no_outcome,
        n_skipped_duplicate,
    )

    # Append to dataset
    if new_rows and not dry_run:
        new_df = pd.DataFrame(new_rows)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        if existing_df is not None:
            # Ensure column alignment
            for col in new_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = float("nan")
            merged = pd.concat(
                [existing_df, new_df[existing_df.columns]], ignore_index=True
            )
        else:
            merged = new_df

        merged.to_parquet(dataset_path, index=False)
        logger.info("Dataset updated: %d total rows", len(merged))
        n_total = len(merged)
    elif existing_df is not None:
        n_total = len(existing_df)
    else:
        n_total = 0

    # Feature fill rates (on full dataset after merge)
    fill_rates: Dict[str, float] = {}
    if dataset_path.exists() and not dry_run:
        try:
            final_df = pd.read_parquet(dataset_path)
            for fname in _FEATURE_NAMES:
                if fname in final_df.columns:
                    fill_rates[fname] = float(1.0 - final_df[fname].isna().mean())
        except Exception:
            pass

    # Source breakdown
    n_by_source: Dict[str, int] = {}
    if dataset_path.exists() and not dry_run:
        try:
            final_df = pd.read_parquet(dataset_path)
            if "label_source" in final_df.columns:
                n_by_source = final_df["label_source"].value_counts().to_dict()
        except Exception:
            pass

    return {
        "n_candidates": len(candidates),
        "n_outcome_map": len(outcome_map),
        "db_status": db_status,
        "n_matched": n_matched,
        "n_skipped_no_outcome": n_skipped_no_outcome,
        "n_skipped_duplicate": n_skipped_duplicate,
        "n_new": n_matched if not dry_run else 0,
        "n_total": n_total,
        "n_by_source": n_by_source,
        "feature_fill_rates": fill_rates,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Accumulate outcome-linked classifier labels")
    parser.add_argument("--jsonl-path", type=Path, default=_JSONL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=_DATASET_PATH)
    parser.add_argument("--db", type=Path, default=_DB_PATH)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report matches without writing to dataset",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary to stdout")
    args = parser.parse_args(argv)

    result = accumulate(
        jsonl_path=args.jsonl_path,
        dataset_path=args.dataset_path,
        db_path=args.db,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        dr = " [DRY RUN]" if args.dry_run else ""
        print(f"[accumulate_classifier_labels]{dr}")
        print(f"  JSONL candidates (BUY/SELL with features): {result['n_candidates']}")
        print(f"  DB outcome trades available:               {result['n_outcome_map']}")
        print(f"  New labels matched:                        {result['n_new']}")
        print(f"  Skipped (no outcome yet):                  {result['n_skipped_no_outcome']}")
        print(f"  Skipped (already in dataset):              {result['n_skipped_duplicate']}")
        print(f"  Dataset total rows:                        {result['n_total']}")
        if result["n_by_source"]:
            print(f"  By source: {result['n_by_source']}")
        if result["feature_fill_rates"]:
            low = {k: f"{v:.0%}" for k, v in result["feature_fill_rates"].items() if v < 0.5}
            if low:
                print(f"  Features < 50% fill: {low}")
            else:
                print("  All features >= 50% fill rate")

    return 0


if __name__ == "__main__":
    sys.exit(main())
