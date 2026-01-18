#!/usr/bin/env python3
"""
Seed or patch visualizations/dashboard_data.json with price_series and trade_events
so the live dashboard renders trade/price panels without warnings.

Usage:
    python scripts/seed_dashboard_payload.py
    python scripts/seed_dashboard_payload.py --source visualizations/dashboard_data.sample.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_SOURCE = ROOT / "visualizations" / "dashboard_data.sample.json"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_payload(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(target)
    for key in ("price_series", "trade_events", "positions"):
        if not merged.get(key):
            merged[key] = source.get(key, {})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed dashboard_data.json with price_series/trade_events.")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="Dashboard payload to patch (default: visualizations/dashboard_data.json)")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Sample payload to borrow fields from (default: dashboard_data.sample.json)")
    parser.add_argument(
        "--allow-sample",
        action="store_true",
        help="Permit copying sample price_series/trade_events when the payload is empty (otherwise exits).",
    )
    args = parser.parse_args()

    target_path = Path(args.target)
    source_path = Path(args.source)

    if not target_path.exists():
        raise SystemExit(f"Target payload not found: {target_path}")
    if not source_path.exists():
        raise SystemExit(f"Source payload not found: {source_path}")

    target = load_json(target_path)

    has_real_prices = len((target.get("price_series") or {})) > 0
    has_real_trades = len((target.get("trade_events") or [])) > 0
    if has_real_prices or has_real_trades:
        print("Found existing price_series/trade_events; leaving payload untouched.")
        return

    if not args.allow_sample:
        raise SystemExit(
            "Price series/trade events are missing. Run the pipeline/auto-trader to produce real data, "
            "or rerun with --allow-sample to patch from the sample payload."
        )

    source = load_json(source_path)
    patched = merge_payload(target, source)

    target_path.write_text(json.dumps(patched, indent=2), encoding="utf-8")
    print(f"Patched dashboard payload at {target_path} using sample data (--allow-sample was set).")
    print(
        f"Counts -> tickers: {len((patched.get('meta') or {}).get('tickers') or [])}, "
        f"signals: {len(patched.get('signals') or [])}, "
        f"trades: {len(patched.get('trade_events') or [])}, "
        f"price_series: {len(patched.get('price_series') or {})}"
    )


if __name__ == "__main__":
    main()
