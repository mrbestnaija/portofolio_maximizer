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
from typing import List

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


def write_latest_pointer(
    dataset_id: str,
    dataset_dir: Path,
    manifest_path: Path,
    cfg: SyntheticConfig,
    config_path: Path,
    tickers: List[str],
    start: str,
    end: str,
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
        "git_sha": get_git_sha(),
    }

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    latest_pointer = write_latest_pointer(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        cfg=cfg,
        config_path=Path(args.config),
        tickers=tickers,
        start=start,
        end=end,
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
