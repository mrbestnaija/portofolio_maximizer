"""
Validate regime detection + ensemble weights against fresh parquet data.

This script prefers fresh parquet snapshots (e.g. data/raw/AAPL_fresh_*.parquet)
to avoid duplicate/corrupted DB rows affecting regime selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.regime_detector import RegimeDetector, RegimeConfig


DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
DEFAULT_LOOKBACK_MIN_LEN = 200
DEFAULT_HORIZON = 5
DEFAULT_REGIMES = ["MODERATE_TRENDING", "HIGH_VOL_TRENDING", "CRISIS"]
DEFAULT_RAW_DIR = Path("data/raw")


def _parse_tickers(raw: Optional[str]) -> List[str]:
    if not raw:
        return DEFAULT_TICKERS
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _latest_fresh_file(output_dir: Path, ticker: str) -> Optional[Path]:
    patterns = [
        f"{ticker}_fresh_*.parquet",
        f"{ticker}_fresh_*.csv",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(output_dir.glob(pat))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame.sort_index()


def _dedupe_series(series: pd.Series) -> Tuple[pd.Series, int]:
    dupes = int(series.index.duplicated().sum())
    if dupes:
        series = series[~series.index.duplicated(keep="last")]
    return series, dupes


def _build_forecaster_config(forecasting_cfg: Dict, horizon: int) -> TimeSeriesForecasterConfig:
    sarimax_cfg = forecasting_cfg.get("sarimax", {})
    garch_cfg = forecasting_cfg.get("garch", {})
    samossa_cfg = forecasting_cfg.get("samossa", {})
    mssa_rl_cfg = forecasting_cfg.get("mssa_rl", {})
    ensemble_cfg = forecasting_cfg.get("ensemble", {})
    regime_cfg = forecasting_cfg.get("regime_detection", {})

    return TimeSeriesForecasterConfig(
        forecast_horizon=int(horizon),
        sarimax_enabled=bool(sarimax_cfg.get("enabled", True)),
        garch_enabled=bool(garch_cfg.get("enabled", True)),
        samossa_enabled=bool(samossa_cfg.get("enabled", True)),
        mssa_rl_enabled=bool(mssa_rl_cfg.get("enabled", True)),
        ensemble_enabled=bool(ensemble_cfg.get("enabled", True)),
        sarimax_kwargs={k: v for k, v in sarimax_cfg.items() if k != "enabled"},
        garch_kwargs={k: v for k, v in garch_cfg.items() if k != "enabled"},
        samossa_kwargs={k: v for k, v in samossa_cfg.items() if k != "enabled"},
        mssa_rl_kwargs={k: v for k, v in mssa_rl_cfg.items() if k != "enabled"},
        ensemble_kwargs={k: v for k, v in ensemble_cfg.items() if k != "enabled"},
        regime_detection_enabled=bool(regime_cfg.get("enabled", False)),
        regime_detection_kwargs={k: v for k, v in regime_cfg.items() if k != "enabled"},
    )


def _build_regime_detector(regime_cfg: Dict) -> RegimeDetector:
    config = RegimeConfig(
        enabled=True,
        lookback_window=int(regime_cfg.get("lookback_window", 60)),
        vol_threshold_low=float(regime_cfg.get("vol_threshold_low", 0.15)),
        vol_threshold_high=float(regime_cfg.get("vol_threshold_high", 0.30)),
        trend_threshold_weak=float(regime_cfg.get("trend_threshold_weak", 0.30)),
        trend_threshold_strong=float(regime_cfg.get("trend_threshold_strong", 0.60)),
    )
    return RegimeDetector(config)


def _scan_regime_hits(
    series: pd.Series,
    forecaster: TimeSeriesForecaster,
    detector: RegimeDetector,
    *,
    min_len: int,
) -> Dict[str, Dict]:
    lookback = detector.config.lookback_window
    hits: Dict[str, Dict] = {}

    for i in range(min_len, len(series)):
        raw_prefix = series.iloc[: i + 1]
        ensured = forecaster._ensure_series(raw_prefix)
        if len(ensured) < lookback + 5:
            continue
        returns = ensured.pct_change().dropna()
        if len(returns) < lookback:
            continue
        rp = ensured.iloc[-lookback:]
        rr = returns.iloc[-lookback:]
        feats = detector._extract_regime_features(rp, rr)
        regime = detector._classify_regime(feats)
        if regime not in hits:
            hits[regime] = {
                "date": ensured.index[-1],
                "features": feats,
                "raw_len": len(raw_prefix),
                "ens_len": len(ensured),
            }
    return hits


def _log_forecaster_result(
    label: str,
    forecaster: TimeSeriesForecaster,
    series: pd.Series,
    horizon: int,
) -> None:
    returns = series.pct_change().dropna()
    forecaster.fit(price_series=series, returns_series=returns)
    result = forecaster.forecast(steps=horizon)

    regime = result.get("regime")
    confidence = result.get("regime_confidence")
    feats = result.get("regime_features") or {}
    weights = (result.get("ensemble_forecast") or {}).get("weights") or {}

    print(f"\n[{label}]")
    print(f"  Detected regime: {regime} (confidence={confidence})")
    if feats:
        print(
            "  Features: vol={:.3f}, trend={:.3f}, hurst={:.3f}".format(
                float(feats.get("realized_volatility", float("nan"))),
                float(feats.get("trend_strength", float("nan"))),
                float(feats.get("hurst_exponent", float("nan"))),
            )
        )
    print(f"  Ensemble weights: {weights}")


def _resolve_files(output_dir: Path, tickers: Iterable[str], files: Optional[List[str]]) -> List[Path]:
    if files:
        return [Path(f) for f in files]
    resolved: List[Path] = []
    for ticker in tickers:
        path = _latest_fresh_file(output_dir, ticker)
        if path:
            resolved.append(path)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate regime + weights on fresh parquet files.")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--files", nargs="*", default=None, help="Explicit parquet/csv files to validate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--min-len", type=int, default=DEFAULT_LOOKBACK_MIN_LEN)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--regimes",
        type=str,
        default=",".join(DEFAULT_REGIMES),
        help="Comma-separated regimes to report (default: MODERATE_TRENDING,HIGH_VOL_TRENDING,CRISIS).",
    )
    args = parser.parse_args()

    tickers = _parse_tickers(args.tickers)
    target_regimes = [r.strip().upper() for r in args.regimes.split(",") if r.strip()]

    files = _resolve_files(args.output_dir, tickers, args.files)
    if not files:
        raise SystemExit("No fresh files found. Provide --files or place *_fresh_*.parquet in data/raw.")

    with Path("config/forecasting_config.yml").open("r", encoding="utf-8") as f:
        forecasting_cfg = yaml.safe_load(f)["forecasting"]

    forecaster_cfg = _build_forecaster_config(forecasting_cfg, args.horizon)
    detector = _build_regime_detector(forecasting_cfg.get("regime_detection", {}))

    np.random.seed(7)

    for path in files:
        df = _load_frame(path)
        if "Close" not in df.columns:
            print(f"[WARN] {path} missing Close column; skipping.")
            continue

        series = df["Close"].astype(float)
        series, dupes = _dedupe_series(series)
        print("\n" + "=" * 80)
        print(f"File: {path}")
        print(f"Rows: {len(series)} | Range: {series.index.min().date()} -> {series.index.max().date()}")
        if dupes:
            print(f"[WARN] {dupes} duplicate timestamps found; deduped by keeping last.")

        forecaster = TimeSeriesForecaster(config=forecaster_cfg)
        hits = _scan_regime_hits(series, forecaster, detector, min_len=args.min_len)

        for regime in target_regimes:
            hit = hits.get(regime)
            if not hit:
                print(f"\n[{regime}] No qualifying window found (min_len={args.min_len}).")
                continue
            asof = pd.Timestamp(hit["date"])
            prefix = series.loc[:asof]
            label = f"{regime} @ {asof.date()} (raw_len={hit['raw_len']} ensured_len={hit['ens_len']})"
            _log_forecaster_result(label, TimeSeriesForecaster(config=forecaster_cfg), prefix, args.horizon)

        # Always log latest as-of.
        _log_forecaster_result("LATEST", TimeSeriesForecaster(config=forecaster_cfg), series, args.horizon)


if __name__ == "__main__":
    main()
