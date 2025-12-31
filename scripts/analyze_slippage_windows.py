#!/usr/bin/env python3
"""
analyze_slippage_windows.py
---------------------------

Diagnostic slippage window analysis. Buckets trades by hour-of-day (if a
timestamp column exists) or by trade_date (fallback) and reports median/p90
slippage per asset class grouping.

NOTE: This relies on `commission` as a proxy for cost; extend with mid-price
logging for production use.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import click


def _group_key(ticker: str) -> str:
    t = (ticker or "").upper()
    if t.endswith("=X"):
        return "FX"
    if t.endswith("-USD") or t in {"BTC", "ETH"}:
        return "CRYPTO"
    if "^" in t:
        return "INDEX"
    return "US_EQUITY"


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    pos = max(0.0, min(100.0, q)) / 100.0 * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def _load_execution_events(path: Path) -> Tuple[List[Dict], int]:
    if not path.exists():
        return [], 0
    events: List[Dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
            if ev.get("status") == "SKIPPED_NO_TRADE_WINDOW":
                skipped += 1
            events.append(ev)
        except Exception:
            continue
    return events, skipped


def _extract_hour(ts_val: str | None) -> int | None:
    if not ts_val:
        return None
    try:
        return datetime.fromisoformat(ts_val.replace("Z", "+00:00")).hour
    except Exception:
        return None


@click.command()
@click.option("--db-path", default="data/portfolio_maximizer.db", show_default=True)
@click.option("--output", default="logs/automation/slippage_windows.json", show_default=True)
@click.option(
    "--execution-log",
    default="logs/automation/execution_log.jsonl",
    show_default=True,
    help="Execution log with mid-price slippage (emitted by run_auto_trader.py).",
)
def main(db_path: str, output: str, execution_log: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticker, trade_date, commission
        FROM trade_executions
        WHERE commission IS NOT NULL
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    by_group: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        ticker = str(r.get("ticker") or "")
        group = _group_key(ticker)
        comm = float(r.get("commission") or 0.0)
        by_group[group].append(comm)

    events, skipped_windows = _load_execution_events(Path(execution_log))
    mid_by_group: Dict[str, List[float]] = defaultdict(list)
    mid_by_hour: Dict[int, List[float]] = defaultdict(list)
    for ev in events:
        mid = ev.get("mid_price")
        entry = ev.get("entry_price")
        if mid in (None, 0, 0.0) or entry is None:
            continue
        try:
            mid = float(mid)
            entry = float(entry)
        except Exception:
            continue
        if mid == 0:
            continue
        slip_bp = ((entry - mid) / mid) * 1e4
        group = _group_key(str(ev.get("ticker") or ""))
        mid_by_group[group].append(slip_bp)
        hour = _extract_hour(str(ev.get("timestamp") or ev.get("logged_at") or ""))
        if hour is not None:
            mid_by_hour[hour].append(slip_bp)

    summary = []
    for group, vals in by_group.items():
        summary.append(
            {
                "group": group,
                "trades": len(vals),
                "commission_median": _percentile(vals, 50.0),
                "commission_p90": _percentile(vals, 90.0),
            }
        )

    mid_group = []
    for group, vals in mid_by_group.items():
        mid_group.append(
            {
                "group": group,
                "records": len(vals),
                "mid_slippage_median_bp": _percentile(vals, 50.0),
                "mid_slippage_p90_bp": _percentile(vals, 90.0),
            }
        )
    mid_hour = []
    for hour, vals in sorted(mid_by_hour.items()):
        mid_hour.append(
            {
                "hour_utc": hour,
                "records": len(vals),
                "mid_slippage_median_bp": _percentile(vals, 50.0),
                "mid_slippage_p90_bp": _percentile(vals, 90.0),
            }
        )

    out = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "db_path": db_path,
            "execution_log": execution_log,
            "note": "Commission still included as proxy; mid-price slippage used when present.",
        },
        "groups": summary,
        "mid_price_slippage": {
            "by_group": mid_group,
            "by_hour_utc": mid_hour,
            "skipped_no_trade_window": skipped_windows,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Slippage window summary written to {output} ({len(summary)} groups)")


if __name__ == "__main__":
    main()
