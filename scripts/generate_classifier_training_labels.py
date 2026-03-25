#!/usr/bin/env python3
"""
scripts/generate_classifier_training_labels.py
------------------------------------------------
Phase 9: Generate directional classifier training labels directly from
checkpoint price parquets, bypassing the JSONL timestamp mismatch.

Problem this solves
-------------------
When ETL runs on historical data (2019-2022), signals are logged to
quant_validation.jsonl with wall-clock timestamps (today's date), not the
historical bar date. build_directional_training_data.py cannot join those
entries to forward prices because the timestamps fall outside the parquet
window. This script sidesteps the issue entirely by reading parquets directly.

What it does
------------
1. Load a checkpoint price parquet (OHLCV) for a given ticker.
2. Walk a rolling window (--lookback bars of history) through the parquet in
   strides of --step bars.
3. At each step, compute classifier features from price data alone:
   - realized_vol_annualized, recent_return_5d, recent_vol_ratio   (price math)
   - adf_pvalue                                                      (statsmodels)
   - hurst_exponent, trend_strength, regime one-hots               (RegimeDetector)
   Features that require a live forecast (ensemble_pred_return, ci_width, snr,
   model_agreement, etc.) are written as NaN and handled by the SimpleImputer
   in the classifier pipeline.
4. Compute forward-price label: y=1 if Close[t+horizon] > Close[t] else 0.
5. Append labeled rows to data/training/directional_dataset.parquet
   (atomic write; deduplicates on entry_ts + ticker).

Usage
-----
  # Single parquet, explicit ticker
  python scripts/generate_classifier_training_labels.py \\
      --ticker AAPL \\
      --parquet data/checkpoints/pipeline_20260318_AAPL.parquet

  # Auto-detect newest parquet per ticker from data/checkpoints/
  python scripts/generate_classifier_training_labels.py \\
      --ticker AAPL --auto-parquet

  # Full bootstrap: 5 tickers, auto-detect parquets
  python scripts/generate_classifier_training_labels.py \\
      --ticker AAPL,MSFT,NVDA,GS,AMZN --auto-parquet

Output
------
  data/training/directional_dataset.parquet   (appended, deduplicated)
  logs/directional_training_latest.json        (updated summary)
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

_OUTPUT_PARQUET = Path("data/training/directional_dataset.parquet")
_SUMMARY_PATH = Path("logs/directional_training_latest.json")
_CHECKPOINT_DIR = Path("data/checkpoints")
_MIN_LOOKBACK = 60          # minimum bars of history before generating a signal
_DEFAULT_LOOKBACK = 252     # bars used for feature computation
_DEFAULT_STEP = 10          # stride between signal generation points
_DEFAULT_HORIZON = 30       # forward bars for label

# Feature names must match forcester_ts.directional_classifier._FEATURE_NAMES
_FEATURE_NAMES = [
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


# ---------------------------------------------------------------------------
# Feature extraction (price-based only)
# ---------------------------------------------------------------------------

def _compute_price_features(close: pd.Series) -> Dict[str, float]:
    """
    Compute the price-derivable subset of classifier features.
    Forecasting features (ensemble_pred_return, ci_width_normalized, etc.)
    are returned as NaN and will be imputed by the classifier's SimpleImputer.
    """
    nan = float("nan")
    feat: Dict[str, float] = {name: nan for name in _FEATURE_NAMES}
    n = len(close)

    # recent_return_5d
    if n >= 6:
        p6 = float(close.iloc[-6])
        if p6 > 0:
            feat["recent_return_5d"] = float(close.iloc[-1] / p6) - 1.0

    # recent_vol_ratio  (5-bar vol / 60-bar vol)
    if n >= 60:
        ret = close.pct_change().dropna()
        vol_5 = float(ret.iloc[-5:].std()) if len(ret) >= 5 else nan
        vol_60 = float(ret.iloc[-60:].std()) if len(ret) >= 60 else nan
        if np.isfinite(vol_60) and vol_60 > 0:
            feat["recent_vol_ratio"] = vol_5 / vol_60

    # realized_vol_annualized
    if n >= 20:
        ret = close.pct_change().dropna()
        feat["realized_vol_annualized"] = float(ret.std() * np.sqrt(252))

    # ADF p-value
    if n >= 20:
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, pval, *_ = adfuller(close.values, maxlag=min(5, n // 5), autolag=None)
            feat["adf_pvalue"] = float(np.clip(pval, 0.0, 1.0))
        except Exception as exc:
            logger.debug("ADF test failed: %s", exc)

    # Hurst exponent + trend strength + regime one-hots.
    # Implemented inline (pure numpy/statsmodels) so the script can be run as
    # 'python scripts/generate_classifier_training_labels.py' without the project
    # root on sys.path — avoids ModuleNotFoundError for forcester_ts.
    if n >= 30:
        try:
            prices = close.values.astype(float)

            # --- Hurst exponent (rescaled range / R-S analysis) ---
            def _hurst(x: np.ndarray) -> float:
                m = len(x)
                lags, rs_vals = [], []
                for lag in range(2, min(20, m // 2)):
                    nw = m // lag
                    if nw < 2:
                        continue
                    rs_per_w = []
                    for i in range(nw):
                        w = x[i * lag:(i + 1) * lag]
                        dev = np.cumsum(w - w.mean())
                        s = w.std(ddof=1)
                        if s > 0:
                            rs_per_w.append((dev.max() - dev.min()) / s)
                    if rs_per_w:
                        rs_vals.append(np.log(np.mean(rs_per_w)))
                        lags.append(np.log(lag))
                if len(lags) < 2:
                    return 0.5
                h = np.polyfit(lags, rs_vals, 1)[0]
                return float(np.clip(h, 0.0, 1.0))

            feat["hurst_exponent"] = _hurst(prices)

            # --- Trend strength (linear-regression R²) ---
            x = np.arange(n, dtype=float)
            coeffs = np.polyfit(x, prices, 1)
            y_hat = np.polyval(coeffs, x)
            ss_res = float(np.sum((prices - y_hat) ** 2))
            ss_tot = float(np.sum((prices - prices.mean()) ** 2))
            feat["trend_strength"] = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)) \
                if ss_tot > 1e-12 else 0.0

            # --- Regime one-hots (aligned with RegimeDetector thresholds) ---
            vol_ann = feat.get("realized_vol_annualized", nan)
            ts = feat["trend_strength"]
            if np.isfinite(vol_ann):
                if vol_ann > 0.40:
                    regime = "crisis"
                elif vol_ann > 0.20 and ts > 0.30:
                    regime = "high_vol_trending"
                elif ts > 0.40:
                    regime = "moderate_trending"
                else:
                    regime = "liquid_rangebound"
                for r in ("liquid_rangebound", "moderate_trending", "high_vol_trending", "crisis"):
                    feat[f"regime_{r}"] = 1.0 if regime == r else 0.0
        except Exception as exc:
            logger.debug("Inline regime features failed: %s", exc)

    # Sanitize: replace inf with nan
    for k, v in feat.items():
        if isinstance(v, float) and not np.isfinite(v) and not np.isnan(v):
            feat[k] = nan

    return feat


# ---------------------------------------------------------------------------
# Parquet discovery
# ---------------------------------------------------------------------------

def _find_parquets_for_ticker(
    ticker: str,
    checkpoint_dir: Path,
    strict: bool = False,
) -> List[Path]:
    """
    Return candidate parquets for ticker in descending size order.

    First tries the original filename-based pattern (ticker in name),
    then falls back to scanning all parquets and selecting those whose
    ticker column contains this ticker.

    *strict=True* (multi-ticker runs): generic fallback is disabled.
    Returns [] if no ticker-specific parquet is found, so the caller
    can signal an ambiguous/missing error instead of silently training
    on another ticker's prices.

    When multiple generic parquets with no ticker column exist (ambiguous),
    returns [] regardless of strict mode.
    """
    ticker_upper = ticker.upper()

    # Primary: ticker in filename
    primary = sorted(
        list(checkpoint_dir.glob(f"*{ticker_upper}*data_extraction*.parquet"))
        + list(checkpoint_dir.glob(f"*{ticker_upper}*.parquet")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if primary:
        return primary

    # Strict mode: refuse generic fallback for multi-ticker runs
    if strict:
        return []

    # Fallback: scan all data_extraction parquets and keep only those that contain
    # a 'ticker' column whose values include this ticker.  This prevents one ticker
    # from silently training on another ticker's prices when parquet filenames are
    # generic (e.g. pipeline_20260101_data_extraction_*.parquet).
    # Files whose ticker column is absent or ambiguous are still included as last-
    # resort candidates so we don't break setups with single-ticker generic parquets.
    # EXCEPT: if multiple generic parquets exist with no ticker column, return [] —
    # it's ambiguous which one to use.
    with_match: List[Path] = []
    without_ticker_col: List[Path] = []

    for path in sorted(
        checkpoint_dir.glob("*data_extraction*.parquet"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    ):
        try:
            df_peek = pd.read_parquet(path, columns=["ticker"]) if True else None
            if "ticker" in df_peek.columns:
                tickers_present = {str(t).upper() for t in df_peek["ticker"].dropna().unique()}
                if ticker_upper in tickers_present:
                    with_match.append(path)
                # If ticker column exists but this ticker isn't in it, skip the file —
                # it belongs to a different ticker.
            else:
                without_ticker_col.append(path)
        except Exception:
            # Parquet may not have a ticker column at the column-metadata level;
            # fall through to the generic bucket.
            without_ticker_col.append(path)

    if with_match:
        return with_match
    # Ambiguous: multiple generic parquets with no ticker column → cannot pick safely
    if len(without_ticker_col) > 1:
        return []
    return without_ticker_col


def _load_best_parquet(
    ticker: str,
    checkpoint_dir: Path,
    parquet_path: Optional[Path] = None,
    known_tickers: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Load price DataFrame for ticker. Returns None if not found or ambiguous.

    When *known_tickers* contains more than one ticker (multi-ticker run) we
    require a ticker-specific parquet and refuse to fall back to a generic
    *data_extraction* file — that file belongs to some other ticker and would
    silently contaminate labels.
    """
    if parquet_path is not None:
        candidates = [parquet_path]
    else:
        # Strict mode: multi-ticker runs must have a ticker-named parquet.
        multi_ticker_run = known_tickers is not None and len(known_tickers) > 1
        candidates = _find_parquets_for_ticker(
            ticker, checkpoint_dir, strict=multi_ticker_run
        )

    for path in candidates[:5]:  # try up to 5 candidates
        try:
            df = pd.read_parquet(path)
            if "Close" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[df.index.notna()].sort_index()
            if len(df) < _MIN_LOOKBACK + _DEFAULT_HORIZON:
                logger.debug("Parquet %s too short (%d rows), skipping", path.name, len(df))
                continue
            logger.info("Loaded parquet for %s: %s (%d rows, %s→%s)",
                        ticker, path.name, len(df), df.index.min().date(), df.index.max().date())
            return df
        except Exception as exc:
            logger.warning("Could not load %s: %s", path.name, exc)

    logger.warning("No usable price parquet found for %s in %s", ticker, checkpoint_dir)
    return None


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_labels(
    ticker: str,
    price_df: pd.DataFrame,
    lookback: int = _DEFAULT_LOOKBACK,
    step: int = _DEFAULT_STEP,
    horizon: int = _DEFAULT_HORIZON,
    min_lookback: int = _MIN_LOOKBACK,
) -> List[Dict[str, Any]]:
    """
    Generate labeled training rows from a price DataFrame.

    Each row contains:
      - entry_ts: ISO timestamp of the signal bar (historical, not wall-clock)
      - ticker, action, y_directional, label_source
      - classifier_features (price-based subset; forecasting features are NaN)
    """
    close = price_df["Close"].dropna()
    n = len(close)
    rows: List[Dict[str, Any]] = []
    n_skipped = 0

    effective_lookback = min(lookback, n - horizon - 1)
    if effective_lookback < min_lookback:
        logger.warning(
            "%s: parquet too short for lookback=%d (n=%d, horizon=%d, min_lookback=%d)",
            ticker, lookback, n, horizon, min_lookback,
        )
        return rows

    start_idx = effective_lookback
    generated = 0

    for bar_idx in range(start_idx, n - horizon, step):
        entry_ts = close.index[bar_idx]
        fwd_idx = bar_idx + horizon

        current_close = float(close.iloc[bar_idx])
        if current_close <= 0:
            n_skipped += 1
            continue

        forward_close = float(close.iloc[fwd_idx])
        if forward_close <= 0:
            n_skipped += 1
            continue

        y_directional = 1 if forward_close > current_close else 0

        # Compute features on the lookback window ending at bar_idx (inclusive)
        window = close.iloc[max(0, bar_idx - effective_lookback):bar_idx + 1]
        features = _compute_price_features(window)

        row = {
            "ts_signal_id": f"gen_{ticker}_{entry_ts.strftime('%Y%m%d')}_{bar_idx:05d}",
            "ticker": ticker,
            "entry_ts": entry_ts.isoformat(),
            "action": "BUY" if y_directional == 1 else "SELL",  # majority direction as proxy
            "y_directional": y_directional,
            "label_source": "price_parquet_scan",
            **{k: v for k, v in features.items() if isinstance(v, (int, float, type(None)))},
        }
        rows.append(row)
        generated += 1

    logger.info(
        "%s: generated %d labeled rows from %d bars (step=%d, horizon=%d, skipped=%d)",
        ticker, generated, n, step, horizon, n_skipped,
    )
    return rows


# ---------------------------------------------------------------------------
# Dataset merge (deduplicate on entry_ts + ticker)
# ---------------------------------------------------------------------------

def _append_to_dataset(
    new_rows: List[Dict[str, Any]],
    output_path: Path = _OUTPUT_PARQUET,
) -> int:
    """Append new rows to existing dataset, deduplicating on ts_signal_id."""
    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)

    if output_path.exists():
        try:
            existing = pd.read_parquet(output_path)
            existing_ids = set(existing["ts_signal_id"].dropna()) if "ts_signal_id" in existing.columns else set()
            new_df = new_df[~new_df["ts_signal_id"].isin(existing_ids)]
            if new_df.empty:
                logger.info("All %d new rows already in dataset — nothing to append", len(new_rows))
                return 0
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as exc:
            logger.warning("Could not read existing dataset (%s); starting fresh", exc)
            combined = new_df
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, index=False)
    tmp.replace(output_path)
    logger.info("Wrote %d total rows to %s (%d new)", len(combined), output_path, len(new_df))
    return len(new_df)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(output_path: Path, summary_path: Path) -> Dict[str, Any]:
    """Read the output parquet and write a summary JSON."""
    summary: Dict[str, Any] = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "label_source": "price_parquet_scan",
    }
    if not output_path.exists():
        summary.update({"n_labeled": 0, "cold_start": True, "error": "output_not_written"})
    else:
        df = pd.read_parquet(output_path)
        n = len(df)
        n_pos = int(df["y_directional"].sum()) if "y_directional" in df.columns else 0
        n_neg = n - n_pos
        win_rate = float(n_pos / n) if n > 0 else float("nan")
        cold_start = n < 60 or n_pos < 10 or n_neg < 10
        summary.update({
            "n_labeled": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "win_rate": round(win_rate, 4) if np.isfinite(win_rate) else None,
            "cold_start": cold_start,
            "cold_start_reason": (
                f"n={n} < 60 or class imbalance" if cold_start else None
            ),
            "output_path": str(output_path),
        })
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker", required=True,
        help="Comma-separated ticker(s), e.g. AAPL or AAPL,MSFT,NVDA",
    )
    parser.add_argument(
        "--parquet",
        help="Explicit parquet path. If omitted with --auto-parquet, scans data/checkpoints/",
    )
    parser.add_argument(
        "--auto-parquet", action="store_true",
        help="Auto-find parquets in data/checkpoints/ for each ticker",
    )
    parser.add_argument(
        "--checkpoint-dir", default=str(_CHECKPOINT_DIR),
        help=f"Checkpoint directory to scan (default: {_CHECKPOINT_DIR})",
    )
    parser.add_argument("--lookback", type=int, default=_DEFAULT_LOOKBACK,
                        help=f"Bars of history per signal window (default: {_DEFAULT_LOOKBACK})")
    parser.add_argument("--step", type=int, default=_DEFAULT_STEP,
                        help=f"Stride between signal points (default: {_DEFAULT_STEP})")
    parser.add_argument("--horizon", type=int, default=_DEFAULT_HORIZON,
                        help=f"Forward bars for price label (default: {_DEFAULT_HORIZON})")
    parser.add_argument("--output", default=str(_OUTPUT_PARQUET),
                        help=f"Output parquet path (default: {_OUTPUT_PARQUET})")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help=(
            "Allow partial runs where some requested tickers have no parquet. "
            "When set, missing tickers are skipped (status=missing_parquet in "
            "ticker_results) instead of causing a hard failure (rc=1)."
        ),
    )
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)

    if not args.parquet and not args.auto_parquet:
        parser.error("Provide --parquet <path> or --auto-parquet to scan data/checkpoints/")

    # Detect duplicate-parquet collisions before generating labels.
    # When two tickers resolve to the same physical file the ETL was likely run
    # with --execution-mode synthetic (SyntheticExtractor, same seed → identical prices).
    # Labels from such data are meaningless; warn loudly so operators can fix it.
    if args.auto_parquet and len(tickers) > 1:
        from collections import defaultdict as _defaultdict
        _path_to_tickers: dict = _defaultdict(list)
        for _tk in tickers:
            _candidates = _find_parquets_for_ticker(_tk, checkpoint_dir)
            if _candidates:
                _path_to_tickers[str(_candidates[0].resolve())].append(_tk)
        for _pstr, _shared in _path_to_tickers.items():
            if len(_shared) > 1:
                logger.warning(
                    "[V5] Tickers %s all resolve to the same parquet %s. "
                    "Price data may not be ticker-specific (was ETL run with "
                    "--execution-mode synthetic?). Training labels from shared price "
                    "data are meaningless. Re-run ETL with --execution-mode auto and "
                    "rename parquets to include the ticker name (e.g. AAPL_pipeline_*).",
                    _shared, Path(_pstr).name,
                )

    total_new = 0
    tickers_satisfied = 0
    ticker_results: Dict[str, Dict[str, Any]] = {}
    multi_ticker_run = len(tickers) > 1

    for ticker in tickers:
        parquet_path = Path(args.parquet) if args.parquet else None
        price_df = _load_best_parquet(
            ticker, checkpoint_dir, parquet_path,
            known_tickers=tickers if multi_ticker_run else None,
        )
        if price_df is None:
            # Determine why: ambiguous (multiple generic) vs simply missing
            generic_candidates = list(checkpoint_dir.glob("*data_extraction*.parquet"))
            ticker_named = list(checkpoint_dir.glob(f"*{ticker.upper()}*.parquet"))
            if not ticker_named and len(generic_candidates) > 1:
                selection_status = "ambiguous_parquet"
            else:
                selection_status = "missing_parquet"
            logger.warning("Skipping %s — %s", ticker, selection_status)
            ticker_results[ticker] = {
                "status": selection_status,
                "generated_rows": 0,
                "new_rows_added": 0,
            }
            continue

        rows = generate_labels(
            ticker=ticker,
            price_df=price_df,
            lookback=args.lookback,
            step=args.step,
            horizon=args.horizon,
        )
        added = _append_to_dataset(rows, output_path)
        total_new += added
        if added > 0:
            tickers_satisfied += 1
        ticker_results[ticker] = {
            "status": "ok",
            "generated_rows": len(rows),
            "new_rows_added": added,
        }

    summary = _write_summary(output_path, _SUMMARY_PATH)
    summary["ticker_results"] = ticker_results
    summary["requested_tickers"] = tickers
    summary["allow_partial"] = bool(getattr(args, "allow_partial", False))
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    n_labeled = summary.get("n_labeled", 0)
    cold_start = summary.get("cold_start", True)

    print(
        f"[OK] n_labeled={n_labeled} "
        f"win_rate={summary.get('win_rate')} "
        f"cold_start={cold_start} "
        f"new_rows_added={total_new}"
    )

    # Hard fail when any ticker has a parquet problem and --allow-partial not set.
    failed_tickers = [
        t for t, r in ticker_results.items()
        if r["status"] in ("missing_parquet", "ambiguous_parquet")
    ]
    if failed_tickers and not getattr(args, "allow_partial", False):
        logger.error(
            "Parquet selection failed for ticker(s) %s — use --allow-partial to skip. "
            "Returning rc=1.",
            failed_tickers,
        )
        return 1

    # Fail when every requested ticker produced 0 new rows and no prior dataset
    # exists.  Without this guard, callers receive rc=0 while training on stale
    # data from a previous run — silently misrepresenting coverage.
    if total_new == 0 and not output_path.exists():
        logger.error(
            "No new rows added for any of the requested tickers (%s) and no prior "
            "dataset exists at %s — cannot continue.",
            ", ".join(tickers), output_path,
        )
        return 1

    return 0 if not cold_start else 2


if __name__ == "__main__":
    sys.exit(main())
