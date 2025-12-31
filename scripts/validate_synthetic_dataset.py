#!/usr/bin/env python
"""Validate a persisted synthetic dataset and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from etl.synthetic_extractor import SyntheticExtractor


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a persisted synthetic dataset directory.")
    parser.add_argument("--dataset-path", required=True, help="Path to the synthetic dataset directory.")
    parser.add_argument(
        "--config",
        default="config/synthetic_data_config.yml",
        help="Synthetic config path for validation defaults.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for validation report JSON (default: logs/automation/synthetic_validation_<id>.json).",
    )
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


def prune_logs(retention_days: int = 14, log_dir: Path = Path("logs/automation")) -> None:
    try:
        subprocess.run(
            [sys.executable, "scripts/prune_synthetic_logs.py", "--log-dir", str(log_dir), "--retention-days", str(retention_days)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("Log pruning skipped (subprocess failed)")


def load_manifest(dataset_path: Path) -> Dict:
    manifest_path = dataset_path / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to read manifest at %s", manifest_path)
    return {}


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    combined = dataset_path / "combined.parquet"
    frames: List[pd.DataFrame] = []

    if combined.exists():
        frames.append(pd.read_parquet(combined))
    else:
        for parquet_file in sorted(dataset_path.glob("*.parquet")):
            if parquet_file.name == "manifest.json":
                continue
            frames.append(pd.read_parquet(parquet_file))

    if not frames:
        raise FileNotFoundError(f"No parquet data files found under {dataset_path}")

    return pd.concat(frames).sort_index()


def main() -> None:
    args = parse_args()
    configure_logging()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    manifest = load_manifest(dataset_path)
    dataset_id: Optional[str] = manifest.get("dataset_id") if isinstance(manifest, dict) else None
    if not dataset_id:
        dataset_id = dataset_path.name

    data = load_dataset(dataset_path)
    extractor = SyntheticExtractor(config_path=args.config, name="synthetic")
    validation = extractor.validate_data(data)

    report = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "rows": int(len(data)),
        "columns": list(data.columns),
        "manifest": manifest,
        "validation": validation,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path = (
        Path(args.output)
        if args.output
        else Path("logs/automation") / f"synthetic_validation_{dataset_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    append_run_log(
        {
            "event": "validation",
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "report_path": str(output_path),
            "rows": report["rows"],
            "validation_passed": validation.get("passed", False),
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
        }
    )
    prune_logs()

    logger.info("OK Synthetic dataset validated")
    logger.info("  dataset_id: %s", dataset_id)
    logger.info("  rows: %s", report["rows"])
    logger.info("  report: %s", output_path)
    print(json.dumps({"dataset_id": dataset_id, "validation_passed": validation.get("passed", False)}))


if __name__ == "__main__":
    main()
