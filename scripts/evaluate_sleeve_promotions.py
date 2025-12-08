#!/usr/bin/env python3
"""
evaluate_sleeve_promotions.py
-----------------------------

Promotion/demotion helper for sleeve buckets. Consumes the output of
`scripts/summarize_sleeves.py` and emits a JSON plan describing which tickers
should be promoted out of the speculative bucket or demoted from core when
performance slips.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import click
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent


@dataclass
class PromotionRules:
    min_trades: int = 10
    promote_win_rate: float = 0.55
    promote_profit_factor: float = 1.2
    demote_win_rate: float = 0.45
    demote_profit_factor: float = 0.9


def _load_buckets(config_path: Path) -> Dict[str, str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    b = cfg.get("barbell") or {}
    mapping: Dict[str, str] = {}
    for name in ("safe_bucket", "core_bucket", "speculative_bucket"):
        blk = b.get(name) or {}
        bucket = name.replace("_bucket", "")
        for sym in blk.get("symbols") or []:
            mapping[str(sym)] = bucket
    return mapping


def _decide_move(
    entry: Dict[str, float],
    bucket: str,
    rules: PromotionRules,
) -> Tuple[str | None, str | None]:
    # Accept both legacy "total_trades" and the newer "trades" key emitted by
    # scripts/summarize_sleeves.py.
    trades = int(entry.get("total_trades") or entry.get("trades") or 0)
    win_rate = float(entry.get("win_rate") or 0.0)
    profit_factor = float(entry.get("profit_factor") or 0.0)

    if trades < rules.min_trades:
        return None, None

    if bucket == "speculative":
        if win_rate >= rules.promote_win_rate and profit_factor >= rules.promote_profit_factor:
            return "core", f"Promote: win_rate={win_rate:.2f} pf={profit_factor:.2f} trades={trades}"
    elif bucket == "core":
        if win_rate <= rules.demote_win_rate or profit_factor <= rules.demote_profit_factor:
            return "speculative", (
                f"Demote: win_rate={win_rate:.2f} pf={profit_factor:.2f} trades={trades}"
            )
    return None, None


def evaluate_promotions(
    summary: List[Dict[str, float]],
    bucket_map: Dict[str, str],
    rules: PromotionRules,
) -> Dict[str, List[Dict]]:
    promotions: List[Dict] = []
    demotions: List[Dict] = []
    for row in summary:
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        # Prefer explicit sleeve label from summarize_sleeves; fall back to
        # legacy "bucket" or barbell.yml symbols mapping.
        bucket = (
            row.get("sleeve")
            or row.get("bucket")
            or bucket_map.get(ticker)
            or "unassigned"
        )
        target_bucket, reason = _decide_move(row, bucket, rules)
        if target_bucket is None:
            continue
        move = {
            "ticker": ticker,
            "from": bucket,
            "to": target_bucket,
            "reason": reason,
            "metrics": {
                "win_rate": float(row.get("win_rate") or 0.0),
                "profit_factor": float(row.get("profit_factor") or 0.0),
                "total_trades": int(row.get("total_trades") or row.get("trades") or 0),
            },
        }
        if target_bucket == "core":
            promotions.append(move)
        else:
            demotions.append(move)
    return {"promotions": promotions, "demotions": demotions}


@click.command()
@click.option("--summary-path", default="logs/automation/sleeve_summary.json", show_default=True)
@click.option("--config-path", default="config/barbell.yml", show_default=True)
@click.option("--output", default="logs/automation/sleeve_promotion_plan.json", show_default=True)
@click.option("--min-trades", default=10, show_default=True)
@click.option("--promote-win-rate", default=0.55, show_default=True)
@click.option("--promote-profit-factor", default=1.2, show_default=True)
@click.option("--demote-win-rate", default=0.45, show_default=True)
@click.option("--demote-profit-factor", default=0.9, show_default=True)
def main(
    summary_path: str,
    config_path: str,
    output: str,
    min_trades: int,
    promote_win_rate: float,
    promote_profit_factor: float,
    demote_win_rate: float,
    demote_profit_factor: float,
) -> None:
    summary_payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    # Newer summarize_sleeves payload uses "sleeves" with a flat list of
    # per-(sleeve, ticker) metrics. Preserve compatibility with any legacy
    # "summary" key if present.
    summary = summary_payload.get("summary")
    if not summary:
        summary = summary_payload.get("sleeves") or []

    bucket_map = _load_buckets(Path(config_path))
    rules = PromotionRules(
        min_trades=min_trades,
        promote_win_rate=promote_win_rate,
        promote_profit_factor=promote_profit_factor,
        demote_win_rate=demote_win_rate,
        demote_profit_factor=demote_profit_factor,
    )

    plan = evaluate_promotions(summary, bucket_map, rules)
    output_payload = {
        "meta": {
            "generated_at": summary_payload.get("generated_at")
            or summary_payload.get("meta", {}).get("generated_at"),
            "rules": rules.__dict__,
            "source_summary": summary_path,
            "config_path": config_path,
        },
        "plan": plan,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Promotion/demotion plan written to {output}")


if __name__ == "__main__":
    main()
