#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

import pandas as pd

# Make the trainer runnable from repo root without manual PYTHONPATH setup.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forcester_ts.mssa_rl import (
    DEFAULT_MSSA_POLICY_PATH,
    MSSARLConfig,
    MSSAOfflinePolicyTrainingConfig,
    build_mssa_offline_policy_artifact,
    generate_mssa_policy_synthetic_curriculum,
    resolve_mssa_policy_path,
)


def _load_csv_series(path: Path, date_column: str, value_column: str) -> pd.Series:
    frame = pd.read_csv(path)
    if date_column not in frame.columns or value_column not in frame.columns:
        raise ValueError(
            f"CSV must contain {date_column!r} and {value_column!r}; got {list(frame.columns)}"
        )
    frame = frame[[date_column, value_column]].dropna()
    frame[date_column] = pd.to_datetime(frame[date_column], utc=False)
    frame = frame.sort_values(date_column)
    series = pd.Series(frame[value_column].astype(float).to_numpy(), index=frame[date_column], name=path.stem)
    return series


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and write a frozen MSSA offline policy artifact."
    )
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_MSSA_POLICY_PATH,
        help="Output JSON artifact path (default: models/mssa_rl_policy.v1.json).",
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Optional CSV price series input. May be passed multiple times.",
    )
    parser.add_argument("--date-column", default="date", help="CSV date column name.")
    parser.add_argument("--value-column", default="close", help="CSV price/value column name.")
    parser.add_argument(
        "--use-synthetic-curriculum",
        action="store_true",
        help="Train from the built-in deterministic curriculum instead of CSV inputs.",
    )
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--change-point-threshold", type=float, default=4.0)
    parser.add_argument("--reward-horizon", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=150)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--max-windows-per-series", type=int, default=None)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    series_collection: List[pd.Series] = []
    if args.use_synthetic_curriculum or not args.input_csv:
        series_collection.extend(generate_mssa_policy_synthetic_curriculum())
    for csv_path in args.input_csv:
        series_collection.append(
            _load_csv_series(Path(csv_path), args.date_column, args.value_column)
        )

    model_cfg = MSSARLConfig(
        window_length=args.window_length,
        change_point_threshold=args.change_point_threshold,
        reward_horizon=args.reward_horizon,
        use_q_strategy_selection=False,
        policy_artifact_path="",
    )
    train_cfg = MSSAOfflinePolicyTrainingConfig(
        reward_horizon=args.reward_horizon,
        min_train_size=args.min_train_size,
        step_size=args.step_size,
        max_windows_per_series=args.max_windows_per_series,
        policy_source="offline_trainer",
    )
    artifact = build_mssa_offline_policy_artifact(
        series_collection,
        model_config=model_cfg,
        training_config=train_cfg,
    )

    output_path = resolve_mssa_policy_path(args.artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_path": str(output_path),
                "window_count": artifact["training_metadata"]["window_count"],
                "series_count": artifact["training_metadata"]["series_count"],
                "policy_version": artifact["policy_version"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
