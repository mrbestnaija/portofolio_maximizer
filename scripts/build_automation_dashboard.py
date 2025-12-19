#!/usr/bin/env python3
"""
build_automation_dashboard.py
-----------------------------

Glue script that consolidates automation artefacts into a single JSON snapshot
for dashboards and agents.

Inputs (all optional, best-effort):
  - logs/automation/ts_threshold_sweep.json
  - logs/automation/transaction_costs.json
  - logs/automation/sleeve_summary.json
  - logs/automation/sleeve_promotion_plan.json
  - logs/automation/config_proposals.json
  - best cached strategy config from strategy optimization

Output:
  - visualizations/dashboard_automation.json

This script is intentionally read-only with respect to configs. It surfaces
“what should we change next?” in one place; humans (or higher-level agents)
remain responsible for applying any config diffs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click

from etl.database_manager import DatabaseManager

ROOT_PATH = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw) if raw.strip() else None
    except Exception:
        return None


@click.command()
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite database used for strategy/TS artefacts.",
)
@click.option(
    "--ts-sweep-path",
    default="logs/automation/ts_threshold_sweep.json",
    show_default=True,
    help="Path to TS threshold sweep JSON.",
)
@click.option(
    "--costs-path",
    default="logs/automation/transaction_costs.json",
    show_default=True,
    help="Path to transaction_costs JSON.",
)
@click.option(
    "--ts-candidates-path",
    default="logs/automation/ts_model_candidates_summary.json",
    show_default=True,
    help="Optional path to TS model candidates summary JSON produced by summarize_ts_candidates.py.",
)
@click.option(
    "--sleeve-summary-path",
    default="logs/automation/sleeve_summary.json",
    show_default=True,
    help="Path to sleeve_summary JSON emitted by summarize_sleeves.py.",
)
@click.option(
    "--sleeve-plan-path",
    default="logs/automation/sleeve_promotion_plan.json",
    show_default=True,
    help="Path to sleeve_promotion_plan JSON emitted by evaluate_sleeve_promotions.py.",
)
@click.option(
    "--config-proposals-path",
    default="logs/automation/config_proposals.json",
    show_default=True,
    help="Path to config_proposals JSON emitted by generate_config_proposals.py.",
)
@click.option(
    "--output",
    default="visualizations/dashboard_automation.json",
    show_default=True,
    help="Output JSON path for the consolidated automation snapshot.",
)
def main(
    db_path: str,
    ts_sweep_path: str,
    costs_path: str,
    ts_candidates_path: str,
    sleeve_summary_path: str,
    sleeve_plan_path: str,
    config_proposals_path: str,
    output: str,
) -> None:
    """Build a consolidated automation dashboard snapshot."""
    ts_sweep = _load_json(ROOT_PATH / ts_sweep_path)
    costs = _load_json(ROOT_PATH / costs_path)
    ts_candidates = _load_json(ROOT_PATH / ts_candidates_path)
    sleeve_summary = _load_json(ROOT_PATH / sleeve_summary_path)
    sleeve_plan = _load_json(ROOT_PATH / sleeve_plan_path)
    config_proposals = _load_json(ROOT_PATH / config_proposals_path)

    # Best cached strategy configuration (generic higher-order hyperopt output).
    best_strategy: Optional[Dict[str, Any]] = None
    db = None
    try:
        db = DatabaseManager(db_path=db_path)
        best_strategy = db.get_best_strategy_config()
    except Exception:
        best_strategy = None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "ts_threshold_sweep": ts_sweep,
            "transaction_costs": costs,
            "transaction_costs_synthetic": (costs or {}).get("synthetic") if costs else None,
            "ts_model_candidates_summary": ts_candidates,
            "sleeve_summary": sleeve_summary,
            "sleeve_promotion_plan": sleeve_plan,
            "config_proposals": config_proposals,
        },
        "models": {
            # Placeholder for future TS model/hyper-param candidate summaries.
            "best_strategy_config": best_strategy,
        },
    }

    out_path = ROOT_PATH / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Automation dashboard snapshot written to {out_path}")


if __name__ == "__main__":
    main()
