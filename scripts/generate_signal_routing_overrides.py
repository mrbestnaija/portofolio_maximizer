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

    overrides: Dict[str, Any] = {
        "signal_routing": {
            "time_series": {
                "per_ticker": {},
                "cost_model": {"default_roundtrip_cost_bps": {}},
            }
        }
    }

    for entry in payload.get("time_series_thresholds") or []:
        ticker = str(entry.get("ticker") or "").strip()
        if not ticker:
            continue
        overrides["signal_routing"]["time_series"]["per_ticker"][ticker] = {
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
        bps = entry.get("suggested_roundtrip_cost_bps")
        if bps is None:
            bps = entry.get("roundtrip_cost_median_bps")
        try:
            bps_f = float(bps)
        except (TypeError, ValueError):
            bps_f = None
        if bps_f is None:
            continue
        overrides["signal_routing"]["time_series"]["cost_model"]["default_roundtrip_cost_bps"][group] = bps_f

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(overrides, sort_keys=True), encoding="utf-8")
    click.echo(f"Signal routing overrides written to {out_path}")


if __name__ == "__main__":
    main()
