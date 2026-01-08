#!/usr/bin/env python
"""Generate and persist a synthetic OHLCV dataset using the synthetic extractor.

Phase 0 scaffolding: config-driven generation, deterministic dataset_id, parquet
artifacts + manifest for cron/brutal/offline regression runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import pandas as pd

from etl.synthetic_extractor import SyntheticConfig, SyntheticExtractor


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset and persist it to parquet + manifest.")
    parser.add_argument("--config", default="config/synthetic_data_config.yml", help="Path to synthetic config YAML.")
    parser.add_argument("--tickers", help="Comma-separated tickers override (defaults to config).")
    parser.add_argument("--start-date", help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Override end date (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, help="Override RNG seed.")
    parser.add_argument("--frequency", help="Override frequency (e.g., B, 1min).")
    parser.add_argument("--output-root", help="Optional override for dataset output root (default: config persistence root).")
    parser.add_argument("--dataset-id", help="Optional dataset_id override; otherwise computed from config hash/seed/time window.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def append_run_log(entry: dict, log_dir: Path = Path("logs/automation")) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    entry_with_ts = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    log_path = log_dir / "synthetic_runs.log"
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry_with_ts) + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to append synthetic run log: %s", exc)


def build_feature_frames(data: pd.DataFrame, features_cfg: dict) -> dict:
    """Build lightweight feature tables per ticker (optional)."""
    if not features_cfg or not features_cfg.get("enabled", False):
        return {}

    sma_windows = features_cfg.get("sma_windows", [5, 20]) or []
    vol_windows = features_cfg.get("vol_windows", [10]) or []
    return_horizons = features_cfg.get("return_horizons", [1, 5]) or []
    rsi_windows = features_cfg.get("rsi_windows", [14]) or []
    macd_cfg = features_cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9}) or {}
    boll_cfg = features_cfg.get("bollinger", {"window": 20, "num_std": 2}) or {}
    zscore_windows = features_cfg.get("zscore_windows", [20]) or []
    factor_exposures = features_cfg.get("factor_exposures", []) or []

    features_by_ticker = {}
    for ticker, df_t in data.groupby("ticker"):
        feats = pd.DataFrame(index=df_t.index)
        close = df_t["Close"].astype(float)
        returns = close.pct_change()

        for w in sma_windows:
            feats[f"sma_{int(w)}"] = close.rolling(window=int(w), min_periods=max(1, int(w) // 2)).mean()
        for w in vol_windows:
            feats[f"vol_{int(w)}"] = returns.rolling(window=int(w), min_periods=max(1, int(w) // 2)).std()
        for h in return_horizons:
            feats[f"return_fwd_{int(h)}"] = close.pct_change(periods=int(h)).shift(-int(h))
        for w in rsi_windows:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(w, min_periods=max(1, w // 2)).mean()
            loss = -delta.clip(upper=0).rolling(w, min_periods=max(1, w // 2)).mean()
            rs = gain / loss.replace(0, np.nan)
            feats[f"rsi_{int(w)}"] = 100 - (100 / (1 + rs))

        fast = int(macd_cfg.get("fast", 12))
        slow = int(macd_cfg.get("slow", 26))
        signal = int(macd_cfg.get("signal", 9))
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        feats["macd"] = macd_line
        feats["macd_signal"] = macd_line.ewm(span=signal, adjust=False).mean()
        feats["macd_hist"] = feats["macd"] - feats["macd_signal"]

        boll_w = int(boll_cfg.get("window", 20))
        boll_std = float(boll_cfg.get("num_std", 2))
        ma = close.rolling(boll_w, min_periods=max(1, boll_w // 2)).mean()
        std = close.rolling(boll_w, min_periods=max(1, boll_w // 2)).std()
        feats["boll_upper"] = ma + boll_std * std
        feats["boll_lower"] = ma - boll_std * std
        feats["boll_mid"] = ma

        for w in zscore_windows:
            feats[f"zscore_{int(w)}"] = (close - close.rolling(w, min_periods=max(1, w // 2)).mean()) / close.rolling(w, min_periods=max(1, w // 2)).std()

        # Simple factor exposure stub: market beta approximated by rolling regression on returns if available
        if factor_exposures:
            market_ret = returns if "market" in factor_exposures else None
            if market_ret is not None:
                beta = (
                    returns.rolling(60, min_periods=20).cov(market_ret) / returns.rolling(60, min_periods=20).var()
                )
                feats["beta_market"] = beta

        feats["ticker"] = ticker
        features_by_ticker[ticker] = feats

    return features_by_ticker


def persist_features(features_by_ticker: dict, dataset_dir: Path) -> Path:
    features_dir = dataset_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for ticker, frame in features_by_ticker.items():
        dest = features_dir / f"{ticker}.parquet"
        frame.to_parquet(dest)
    return features_dir


def compute_calibration_stats(data: pd.DataFrame) -> dict:
    stats = {"generated_at": datetime.now(timezone.utc).isoformat(), "tickers": {}}
    for ticker, df_t in data.groupby("ticker"):
        rets = df_t["Close"].pct_change().dropna()
        log_rets = np.log(df_t["Close"]).diff().dropna()
        stats["tickers"][ticker] = {
            "observations": int(len(rets)),
            "mean_return": float(rets.mean()) if len(rets) else 0.0,
            "volatility": float(rets.std()) if len(rets) else 0.0,
            "log_return_mean": float(log_rets.mean()) if len(log_rets) else 0.0,
            "log_return_variance": float(log_rets.var()) if len(log_rets) else 0.0,
        }
    return stats


def calibrate_from_real_data(calib_cfg: Dict[str, Any], stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_path = calib_cfg.get("source_path")
    if not source_path:
        return None
    path = Path(source_path)
    if not path.exists():
        return None
    try:
        if path.suffix in {".parquet", ".pq"}:
            real = pd.read_parquet(path)
        else:
            real = pd.read_csv(path)
    except Exception:
        return None
    if "Close" not in real.columns:
        return None
    rets = real["Close"].pct_change().dropna()
    real_stats = {
        "source_path": str(path),
        "mean_return": float(rets.mean()) if len(rets) else 0.0,
        "volatility": float(rets.std()) if len(rets) else 0.0,
        "observations": int(len(rets)),
    }
    stats["real_data"] = real_stats
    return real_stats


def persist_calibration(stats: dict, dataset_dir: Path) -> Path:
    path = dataset_dir / "calibration.json"
    path.write_text(json.dumps(stats, indent=2))
    return path


def prune_older_datasets(root: Path, keep_last: int, current_id: str) -> None:
    if keep_last <= 0:
        return
    if not root.exists():
        return
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("syn_")], key=lambda p: p.stat().st_mtime, reverse=True)
    to_delete = candidates[keep_last:]
    for path in to_delete:
        if path.name == current_id:
            continue
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    item.unlink()
            path.rmdir()
            logger.info("Pruned old synthetic dataset: %s", path)
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to prune %s: %s", path, exc)


def write_latest_pointer(
    dataset_id: str,
    dataset_dir: Path,
    manifest_path: Path,
    cfg: SyntheticConfig,
    config_path: Path,
    tickers: List[str],
    start: str,
    end: str,
    features_path: Path | None = None,
    calibration_path: Path | None = None,
) -> Path:
    """Persist a latest-dataset pointer for automation."""
    payload = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "generator_version": cfg.generator_version,
        "config_hash": compute_config_hash(config_path),
        "tickers": tickers,
        "start_date": start,
        "end_date": end,
        "frequency": cfg.frequency,
        "seed": cfg.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if features_path:
        payload["features_path"] = str(features_path)
    if calibration_path:
        payload["calibration_path"] = str(calibration_path)
    pointer_path = cfg.persistence_root / "latest.json"
    pointer_path.write_text(json.dumps(payload, indent=2))
    return pointer_path


def prune_logs(retention_days: int = 14, log_dir: Path = Path("logs/automation")) -> None:
    """Invoke pruning script to enforce retention."""
    try:
        subprocess.run(
            [sys.executable, "scripts/prune_synthetic_logs.py", "--log-dir", str(log_dir), "--retention-days", str(retention_days)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("Log pruning skipped (subprocess failed)")


def compute_config_hash(config_path: Path) -> str:
    try:
        payload = config_path.read_bytes()
        return hashlib.sha1(payload).hexdigest()[:12]  # nosec B303
    except FileNotFoundError:
        return "missing"
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent)
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def persist_dataset(data: pd.DataFrame, dataset_dir: Path, partitioning: str) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if partitioning == "by_ticker" and "ticker" in data.columns:
        for ticker, df_t in data.groupby("ticker", sort=False):
            dest = dataset_dir / f"{ticker}.parquet"
            df_t.to_parquet(dest)
    else:
        data.to_parquet(dataset_dir / "combined.parquet")


def main() -> None:
    args = parse_args()
    configure_logging()

    extractor = SyntheticExtractor(config_path=args.config, name="synthetic")
    cfg: SyntheticConfig = extractor.config

    tickers: List[str] = (
        [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else list(cfg.tickers)
    )
    start = args.start_date or cfg.start_date
    end = args.end_date or cfg.end_date

    # Apply overrides to config so dataset_id and metadata stay consistent.
    cfg = replace(
        cfg,
        seed=args.seed if args.seed is not None else cfg.seed,
        frequency=args.frequency or cfg.frequency,
        persistence_root=Path(args.output_root) if args.output_root else cfg.persistence_root,
    )
    extractor.config = cfg

    data = extractor.extract_ohlcv(tickers=tickers, start_date=start, end_date=end)
    if data is None or data.empty:
        raise RuntimeError("Synthetic generator returned an empty dataset")

    dataset_id = args.dataset_id or data.attrs.get("dataset_id") or extractor._compute_dataset_id(  # type: ignore[attr-defined]
        tickers, start, end, cfg.seed
    )
    dataset_dir = cfg.persistence_root / dataset_id

    validation = extractor.validate_data(data)
    persist_dataset(data, dataset_dir, cfg.partitioning)

    features_cfg = getattr(cfg, "features", {}) or {}
    features_by_ticker = build_feature_frames(data, features_cfg)
    features_dir = persist_features(features_by_ticker, dataset_dir) if features_by_ticker else None
    feature_columns = sorted({c for frame in features_by_ticker.values() for c in frame.columns if c != "ticker"}) if features_by_ticker else []

    calibration_enabled = bool((cfg.calibration or {}).get("emit_stats", True))
    calibration_stats = compute_calibration_stats(data) if calibration_enabled else None
    real_calibration = calibrate_from_real_data(cfg.calibration, calibration_stats) if calibration_stats is not None else None
    if real_calibration:
        calibration_stats["real_data"] = real_calibration
    calibration_path = persist_calibration(calibration_stats, dataset_dir) if calibration_stats else None

    manifest = {
        "dataset_id": dataset_id,
        "generator_version": cfg.generator_version,
        "config_path": str(Path(args.config)),
        "config_hash": compute_config_hash(Path(args.config)),
        "seed": cfg.seed,
        "start_date": start,
        "end_date": end,
        "frequency": cfg.frequency,
        "tickers": tickers,
        "rows": int(len(data)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(cfg.persistence_root),
        "partitioning": cfg.partitioning,
        "validation": validation,
        "features": {
            "enabled": bool(features_by_ticker),
            "path": str(features_dir) if features_dir else None,
            "columns": feature_columns,
        },
        "calibration": {
            "path": str(calibration_path) if calibration_path else None,
            "stats": calibration_stats or {},
        },
        "git_sha": get_git_sha(),
    }

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Enforce retention/rotation
    prune_older_datasets(cfg.persistence_root, cfg.keep_last, dataset_id)
    latest_pointer = write_latest_pointer(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        cfg=cfg,
        config_path=Path(args.config),
        tickers=tickers,
        start=start,
        end=end,
        features_path=features_dir,
        calibration_path=calibration_path,
    )

    append_run_log(
        {
            "event": "generation",
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_dir),
            "manifest_path": str(manifest_path),
            "latest_pointer": str(latest_pointer),
            "rows": manifest["rows"],
            "tickers": tickers,
            "start_date": start,
            "end_date": end,
            "frequency": cfg.frequency,
            "seed": cfg.seed,
            "features_path": str(features_dir) if features_dir else None,
            "calibration_path": str(calibration_path) if calibration_path else None,
        }
    )
    prune_logs()

    logger.info("OK Synthetic dataset generated")
    logger.info("  dataset_id: %s", dataset_id)
    logger.info("  rows: %s", manifest["rows"])
    logger.info("  manifest: %s", manifest_path)
    print(dataset_id)


if __name__ == "__main__":
    main()
