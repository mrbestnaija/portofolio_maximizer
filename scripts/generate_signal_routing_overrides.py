#!/usr/bin/env python3
"""
generate_signal_routing_overrides.py
------------------------------------

Read config proposals (TS thresholds + cost suggestions) and emit a YAML
overlay that can be reviewed/applied to signal routing configs without
touching them directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import click
import yaml


@click.command()
@click.option(
    "--proposals-path",
    default="logs/automation/config_proposals.json",
    show_default=True,
    help="Path to config_proposals.json produced by generate_config_proposals.py.",
)
@click.option(
    "--output",
    default="logs/automation/signal_routing_overrides.yml",
    show_default=True,
    help="Path to write YAML overlay with per-ticker thresholds and cost buffers.",
)
def main(proposals_path: str, output: str) -> None:
    path = Path(proposals_path)
    if not path.exists():
        raise SystemExit(f"Proposals file not found: {proposals_path}")
    payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    overrides: Dict[str, Any] = {"time_series": {}, "friction_buffers": {}}

    for entry in payload.get("time_series_thresholds") or []:
        ticker = str(entry.get("ticker") or "").strip()
        if not ticker:
            continue
        overrides["time_series"][ticker] = {
            "confidence_threshold": entry.get("confidence_threshold"),
            "min_expected_return": entry.get("min_expected_return"),
            "evidence": {
                "total_trades": entry.get("total_trades"),
                "win_rate": entry.get("win_rate"),
                "profit_factor": entry.get("profit_factor"),
                "annualized_pnl": entry.get("annualized_pnl"),
            },
        }

    for entry in payload.get("transaction_costs") or []:
        group = str(entry.get("group") or "").strip()
        if not group:
            continue
        overrides["friction_buffers"][group] = {
            "min_expected_return": entry.get("suggested_min_expected_return"),
            "friction_buffer": entry.get("suggested_friction_buffer"),
            "median_commission": entry.get("median_commission"),
        }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(overrides, sort_keys=True), encoding="utf-8")
    click.echo(f"Signal routing overrides written to {out_path}")


if __name__ == "__main__":
    main()
