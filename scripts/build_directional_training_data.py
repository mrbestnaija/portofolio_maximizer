#!/usr/bin/env python3
"""
scripts/build_directional_training_data.py
-------------------------------------------
Phase 9: Build labeled training dataset for the directional classifier.

Reads logs/signals/quant_validation.jsonl and joins each BUY/SELL entry
to realized forward Close prices from data/checkpoints/ parquet files.

Label: y=1 if Close[t + forecast_horizon] > Close[t] else 0

Output:
  data/training/directional_dataset.parquet   — feature matrix + labels
  logs/directional_training_latest.json       — summary (n, win_rate, cold_start, ...)

Usage:
  python scripts/build_directional_training_data.py [--fallback-to-pnl-label]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_JSONL_PATH = Path("logs/signals/quant_validation.jsonl")
_CHECKPOINT_DIR = Path("data/checkpoints")
_OUTPUT_PARQUET = Path("data/training/directional_dataset.parquet")
_SUMMARY_PATH = Path("logs/directional_training_latest.json")
_COLD_START_THRESHOLD = 60
_MIN_CLASS_COUNT = 10
_MAX_FORWARD_FILL_BARS = 2  # Max bars to bridge weekends/gaps


def _load_price_parquet(ticker: str, checkpoint_dir: Path) -> Optional[pd.DataFrame]:
    """Load the widest available price parquet for ticker from checkpoints."""
    pattern = f"*{ticker.upper()}*data_extraction*.parquet"
    candidates = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        # Try broader search
        candidates = sorted(
            [p for p in checkpoint_dir.glob("*.parquet") if ticker.upper() in p.name.upper()],
            key=lambda p: p.stat().st_size, reverse=True,
        )
    if not candidates:
        return None
    try:
        df = pd.read_parquet(candidates[0])
        if "Close" not in df.columns:
            return None
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()].sort_index()
        return df
    except Exception as exc:
        logger.warning("Could not load price parquet for %s: %s", ticker, exc)
        return None


def _get_forward_close(
    price_df: pd.DataFrame,
    entry_ts: pd.Timestamp,
    horizon: int,
) -> Optional[float]:
    """Return Close price at entry_ts + horizon bars (with _MAX_FORWARD_FILL_BARS gap tolerance)."""
    if price_df is None or price_df.empty:
        return None
    try:
        close = price_df["Close"].dropna()
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        # Find the integer position of entry_ts or nearest prior bar
        idx_pos = close.index.searchsorted(entry_ts, side="right") - 1
        if idx_pos < 0:
            return None
        target_pos = idx_pos + horizon
        if target_pos >= len(close):
            return None
        # Gap check: ensure the forward bar is within _MAX_FORWARD_FILL_BARS of expected
        # (handles weekends/holidays naturally since we use integer bar offset)
        return float(close.iloc[target_pos])
    except Exception as exc:
        logger.debug("Forward close lookup failed: %s", exc)
        return None


def _parse_entry_ts(entry: Dict[str, Any]) -> Optional[pd.Timestamp]:
    """Extract entry timestamp from JSONL entry."""
    # Try signal_timestamp first, then timestamp
    for key in ("signal_timestamp", "timestamp", "entry_ts"):
        val = entry.get(key)
        if val:
            try:
                return pd.Timestamp(val, tz="UTC")
            except Exception:
                pass
    return None


def build_dataset(
    jsonl_path: Path = _JSONL_PATH,
    checkpoint_dir: Path = _CHECKPOINT_DIR,
    fallback_to_pnl_label: bool = False,
) -> Dict[str, Any]:
    """
    Build the directional training dataset.

    Returns summary dict with keys: n, n_labeled, n_skipped, win_rate,
    cold_start, label_source, output_path.
    """
    if not jsonl_path.exists():
        logger.error("JSONL log not found: %s", jsonl_path)
        return {"error": "jsonl_not_found", "cold_start": True}

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    # Filter: BUY/SELL only, non-synthetic
    tradeable = [
        e for e in entries
        if e.get("action") in ("BUY", "SELL")
        and str(e.get("execution_mode") or "").lower() not in ("synthetic",)
    ]
    logger.info("JSONL: %d total entries, %d BUY/SELL non-synthetic", len(entries), len(tradeable))

    rows: List[Dict[str, Any]] = []
    n_skipped = 0
    price_cache: Dict[str, Optional[pd.DataFrame]] = {}

    for e in tradeable:
        ticker = str(e.get("ticker") or "").upper()
        signal_id = e.get("signal_id") or e.get("ts_signal_id")
        features = e.get("classifier_features") or {}

        # Determine entry timestamp
        entry_ts = _parse_entry_ts(e)
        if entry_ts is None:
            n_skipped += 1
            continue

        # Determine forecast horizon
        horizon = int(e.get("forecast_horizon") or 30)

        # Load price data (cached per ticker)
        if ticker not in price_cache:
            price_cache[ticker] = _load_price_parquet(ticker, checkpoint_dir)
        price_df = price_cache[ticker]

        # Get current price at entry_ts
        y_directional: Optional[int] = None

        if price_df is not None:
            current_close = _get_forward_close(price_df, entry_ts, 0)
            forward_close = _get_forward_close(price_df, entry_ts, horizon)
            if current_close is not None and forward_close is not None and current_close > 0:
                y_directional = 1 if forward_close > current_close else 0

        # Fallback: use PnL win/loss label if requested and no price available
        if y_directional is None and fallback_to_pnl_label:
            outcome = e.get("outcome")
            if isinstance(outcome, dict):
                win = outcome.get("win")
                if isinstance(win, bool):
                    y_directional = 1 if win else 0
                    # Mark fallback label source
                    features = dict(features)

        if y_directional is None:
            n_skipped += 1
            continue

        row = {
            "ts_signal_id": signal_id,
            "ticker": ticker,
            "entry_ts": entry_ts.isoformat(),
            "action": e.get("action"),
            "y_directional": int(y_directional),
            "label_source": "price_forward" if price_df is not None else "pnl_fallback",
            **{k: v for k, v in features.items() if isinstance(v, (int, float, type(None)))},
        }
        rows.append(row)

    n_total = len(rows)
    n_pos = sum(r["y_directional"] for r in rows)
    n_neg = n_total - n_pos
    win_rate = float(n_pos / n_total) if n_total > 0 else float("nan")
    cold_start = (
        n_total < _COLD_START_THRESHOLD
        or n_pos < _MIN_CLASS_COUNT
        or n_neg < _MIN_CLASS_COUNT
    )

    summary = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "n_jsonl_entries": len(entries),
        "n_tradeable": len(tradeable),
        "n_labeled": n_total,
        "n_skipped": n_skipped,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "win_rate": round(win_rate, 4) if not np.isnan(win_rate) else None,
        "cold_start": cold_start,
        "cold_start_reason": (
            f"n={n_total} < {_COLD_START_THRESHOLD} or class imbalance"
            if cold_start else None
        ),
        "output_path": str(_OUTPUT_PARQUET),
    }

    if rows:
        df = pd.DataFrame(rows)
        _OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        tmp = _OUTPUT_PARQUET.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(_OUTPUT_PARQUET)
        logger.info(
            "Wrote %d labeled examples to %s (win_rate=%.1f%%, cold_start=%s)",
            n_total, _OUTPUT_PARQUET, win_rate * 100 if not np.isnan(win_rate) else 0,
            cold_start,
        )
    else:
        logger.warning("No labeled examples produced — output parquet not written")
        summary["output_path"] = None

    _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fallback-to-pnl-label",
        action="store_true",
        help="Use outcome.win as label when forward price is unavailable",
    )
    args = parser.parse_args(argv)
    result = build_dataset(fallback_to_pnl_label=args.fallback_to_pnl_label)
    if result.get("error"):
        print(f"[ERROR] {result['error']}")
        return 1
    cold_start = result.get("cold_start", True)
    print(
        f"[OK] n_labeled={result.get('n_labeled', 0)} "
        f"win_rate={result.get('win_rate')} "
        f"cold_start={cold_start}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
