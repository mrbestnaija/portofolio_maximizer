#!/usr/bin/env python
"""Brutal synthetic smoke: generate -> validate -> optional short pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic brutal smoke (generate + validate).")
    parser.add_argument("--config", default="config/synthetic_data_config.yml", help="Synthetic config path.")
    parser.add_argument("--tickers", default="AAPL,MSFT", help="Tickers to synthesize.")
    parser.add_argument("--pipeline", action="store_true", help="Run short pipeline smoke in synthetic mode.")
    parser.add_argument("--pipeline-config", default="config/pipeline_config.yml", help="Pipeline config (optional).")
    parser.add_argument("--start-date", default=None, help="Override start date.")
    parser.add_argument("--end-date", default=None, help="Override end date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = {
        **dict(**{"ENABLE_SYNTHETIC_PROVIDER": "1", "SYNTHETIC_ONLY": "1"}),
        **dict(**{"SYNTHETIC_CONFIG_PATH": args.config}),
        **dict(**{"PYTHONPATH": f"{Path(__file__).resolve().parent.parent}:{os.getenv('PYTHONPATH', '')}"}),
    }
    cmd_gen = [
        sys.executable,
        "scripts/generate_synthetic_dataset.py",
        "--config",
        args.config,
        "--tickers",
        args.tickers,
    ]
    if args.start_date:
        cmd_gen += ["--start-date", args.start_date]
    if args.end_date:
        cmd_gen += ["--end-date", args.end_date]
    subprocess.check_call(cmd_gen, env={**env, **dict()})

    latest = Path("data/synthetic/latest.json")
    if latest.exists():
        dataset_path = latest
        try:
            payload = json.loads(latest.read_text())
            dataset_path = Path(payload.get("dataset_path") or latest)
        except Exception:
            dataset_path = latest
        subprocess.check_call(
            [
                sys.executable,
                "scripts/validate_synthetic_dataset.py",
                "--dataset-path",
                str(dataset_path),
                "--config",
                args.config,
            ],
            env=env,
        )

    if args.pipeline:
        subprocess.check_call(
            [
                sys.executable,
                "scripts/run_etl_pipeline.py",
                "--execution-mode",
                "synthetic",
                "--data-source",
                "synthetic",
                "--tickers",
                args.tickers,
                "--start",
                args.start_date or "2020-01-01",
                "--end",
                args.end_date or "2020-03-31",
                "--config",
                args.pipeline_config,
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
