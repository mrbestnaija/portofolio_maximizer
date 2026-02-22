#!/usr/bin/env python3
"""
Quant validation fail-rate/headroom summary.

This script is a robust replacement for ad-hoc inline Python one-liners over
`logs/signals/quant_validation.jsonl`, especially on Windows PowerShell.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_PATH = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = ROOT_PATH / "logs" / "signals" / "quant_validation.jsonl"


@dataclass(frozen=True)
class QuantHeadroomSummary:
    total: int
    fail_count: int
    fail_rate_pct: float
    red_gate_pct: float
    warn_gate_pct: float
    headroom_to_red_gate_pct: float
    status: str
    per_ticker: list[dict[str, Any]]
    window: int
    used_entries: int


def _status_from_entry(rec: dict[str, Any]) -> str:
    if not isinstance(rec, dict):
        return ""
    raw = (
        rec.get("overall_result")
        or rec.get("status")
        or (rec.get("quant_validation") or {}).get("status")
        or ""
    )
    return str(raw).strip().upper()


def _ticker_from_entry(rec: dict[str, Any]) -> str:
    if not isinstance(rec, dict):
        return "?"
    for key in ("ticker", "symbol", "asset", "instrument"):
        raw = rec.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip().upper()
    return "?"


def _load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"quant_validation log not found at {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    if not rows:
        raise SystemExit(f"No quant validation entries found in {path}")
    return rows


def summarize_headroom(
    *,
    entries: list[dict[str, Any]],
    window: int,
    red_gate_pct: float,
    warn_gate_pct: float,
) -> QuantHeadroomSummary:
    if window <= 0:
        scoped = list(entries)
    else:
        scoped = list(entries[-window:])

    fail_count = 0
    ticker_fail = Counter()
    ticker_total = Counter()
    for rec in scoped:
        ticker = _ticker_from_entry(rec)
        ticker_total[ticker] += 1
        if _status_from_entry(rec) == "FAIL":
            fail_count += 1
            ticker_fail[ticker] += 1

    total = len(scoped)
    fail_rate_pct = (float(fail_count) / float(total) * 100.0) if total else 0.0
    headroom = float(red_gate_pct) - float(fail_rate_pct)
    if fail_rate_pct >= float(red_gate_pct):
        status = "RED"
    elif fail_rate_pct >= float(warn_gate_pct):
        status = "YELLOW"
    else:
        status = "GREEN"

    per_ticker: list[dict[str, Any]] = []
    for ticker in sorted(ticker_total):
        t_total = int(ticker_total[ticker])
        t_fail = int(ticker_fail.get(ticker, 0))
        t_rate = (float(t_fail) / float(t_total) * 100.0) if t_total else 0.0
        per_ticker.append(
            {
                "ticker": ticker,
                "fail_count": t_fail,
                "total": t_total,
                "fail_rate_pct": round(t_rate, 3),
            }
        )

    return QuantHeadroomSummary(
        total=total,
        fail_count=fail_count,
        fail_rate_pct=round(fail_rate_pct, 3),
        red_gate_pct=round(float(red_gate_pct), 3),
        warn_gate_pct=round(float(warn_gate_pct), 3),
        headroom_to_red_gate_pct=round(headroom, 3),
        status=status,
        per_ticker=per_ticker,
        window=int(window),
        used_entries=total,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Path to quant_validation.jsonl (default: logs/signals/quant_validation.jsonl)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Use only the most recent N entries (<=0 means all). Default: 120",
    )
    parser.add_argument(
        "--red-gate-pct",
        type=float,
        default=95.0,
        help="RED gate fail-rate threshold in percent. Default: 95.0",
    )
    parser.add_argument(
        "--warn-gate-pct",
        type=float,
        default=90.0,
        help="YELLOW warning fail-rate threshold in percent. Default: 90.0",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output.",
    )
    args = parser.parse_args(argv)

    log_path = Path(str(args.log_path))
    rows = _load_entries(log_path)
    summary = summarize_headroom(
        entries=rows,
        window=int(args.window),
        red_gate_pct=float(args.red_gate_pct),
        warn_gate_pct=float(args.warn_gate_pct),
    )

    payload = {
        "status": summary.status,
        "fail_count": summary.fail_count,
        "total": summary.total,
        "fail_rate_pct": summary.fail_rate_pct,
        "red_gate_pct": summary.red_gate_pct,
        "warn_gate_pct": summary.warn_gate_pct,
        "headroom_to_red_gate_pct": summary.headroom_to_red_gate_pct,
        "window": summary.window,
        "used_entries": summary.used_entries,
        "per_ticker": summary.per_ticker,
        "log_path": str(log_path),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(
            f"Quant validation: {summary.fail_count}/{summary.total} FAIL "
            f"({summary.fail_rate_pct:.1f}%), headroom to {summary.red_gate_pct:.0f}% gate: "
            f"{summary.headroom_to_red_gate_pct:.1f}%"
        )
        for row in summary.per_ticker:
            print(
                f"  {row['ticker']}: {row['fail_count']}/{row['total']} FAIL "
                f"({float(row['fail_rate_pct']):.0f}%)"
            )
        print(f"Global status: {summary.status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
