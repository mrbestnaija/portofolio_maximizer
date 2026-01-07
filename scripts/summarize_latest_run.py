#!/usr/bin/env python3
"""
Summarize the latest (or specified) auto-trader run from JSONL logs.

Phase 10 helper: turn `logs/automation/run_summary.jsonl` + execution events into a
compact, actionable report for profitability/liquidity/forecast health iteration.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_RUN_SUMMARY_PATH = Path("logs/automation/run_summary.jsonl")
DEFAULT_EXECUTION_LOG_PATH = Path("logs/automation/execution_log.jsonl")


@dataclass(frozen=True)
class RunSelection:
    run_id: str
    record: Dict[str, Any]


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def load_run_summary(path: Path, *, run_id: Optional[str] = None) -> Optional[RunSelection]:
    """Return the latest run summary record (or latest matching run_id)."""
    selected: Optional[Dict[str, Any]] = None
    selected_id: Optional[str] = None

    for payload in _iter_jsonl(path):
        payload_id = payload.get("run_id")
        if not payload_id:
            continue
        payload_id_str = str(payload_id)
        if run_id is None:
            selected = payload
            selected_id = payload_id_str
            continue
        if payload_id_str == run_id:
            selected = payload
            selected_id = payload_id_str

    if selected is None or not selected_id:
        return None
    return RunSelection(run_id=selected_id, record=selected)


def summarize_execution_log(path: Path, *, run_id: str, limit: int = 200) -> Dict[str, Any]:
    events = deque(maxlen=max(int(limit), 1))
    for payload in _iter_jsonl(path):
        if str(payload.get("run_id") or "") != run_id:
            continue
        events.append(payload)

    status_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    executed = 0
    slippage_bps: list[float] = []
    confidences: list[float] = []
    expected_returns: list[float] = []

    for event in events:
        status = str(event.get("status") or "UNKNOWN")
        status_counts[status] += 1
        if status == "REJECTED":
            reason = str(event.get("reason") or "unknown")
            rejection_reasons[reason] += 1
        if status == "EXECUTED":
            executed += 1
            slip = event.get("mid_slippage_bp")
            if isinstance(slip, (int, float)):
                slippage_bps.append(float(slip))
            conf = event.get("signal_confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
            exp = event.get("expected_return")
            if isinstance(exp, (int, float)):
                expected_returns.append(float(exp))

    def _avg(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    return {
        "events_considered": len(events),
        "status_counts": dict(status_counts),
        "top_rejections": rejection_reasons.most_common(5),
        "executed": executed,
        "avg_mid_slippage_bp": _avg(slippage_bps),
        "avg_signal_confidence": _avg(confidences),
        "avg_expected_return": _avg(expected_returns),
    }


def build_summary(run_record: Dict[str, Any], *, execution_stats: Dict[str, Any]) -> Dict[str, Any]:
    profitability = run_record.get("profitability") if isinstance(run_record.get("profitability"), dict) else {}
    liquidity = run_record.get("liquidity") if isinstance(run_record.get("liquidity"), dict) else {}
    forecaster = run_record.get("forecaster") if isinstance(run_record.get("forecaster"), dict) else {}
    quant_validation = (
        run_record.get("quant_validation") if isinstance(run_record.get("quant_validation"), dict) else {}
    )

    return {
        "run": {
            "run_id": run_record.get("run_id"),
            "started_at": run_record.get("started_at"),
            "ended_at": run_record.get("ended_at"),
            "duration_seconds": run_record.get("duration_seconds"),
            "tickers": run_record.get("tickers"),
            "cycles": run_record.get("cycles"),
            "execution_mode": run_record.get("execution_mode"),
            "data_source": run_record.get("data_source"),
        },
        "profitability": profitability,
        "liquidity": liquidity,
        "forecaster": forecaster,
        "quant_validation": quant_validation,
        "execution": execution_stats,
        "next_actions": run_record.get("next_actions") if isinstance(run_record.get("next_actions"), list) else [],
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    run = summary.get("run") or {}
    profitability = summary.get("profitability") or {}
    liquidity = summary.get("liquidity") or {}
    forecaster = summary.get("forecaster") or {}
    quant = summary.get("quant_validation") or {}
    execution = summary.get("execution") or {}
    actions = summary.get("next_actions") or []

    lines = [
        f"# Run Summary: {run.get('run_id')}",
        "",
        f"- Started: {run.get('started_at')}",
        f"- Ended: {run.get('ended_at')}",
        f"- Duration (s): {run.get('duration_seconds')}",
        f"- Tickers: {run.get('tickers')}",
        f"- Cycles: {run.get('cycles')}",
        f"- Mode: {run.get('execution_mode')} (source={run.get('data_source')})",
        "",
        "## Profitability",
        f"- PnL: {profitability.get('pnl_dollars')} ({profitability.get('pnl_pct')})",
        f"- Profit factor: {profitability.get('profit_factor')}",
        f"- Win rate: {profitability.get('win_rate')}",
        f"- Trades: {profitability.get('trades')} (realized={profitability.get('realized_trades')})",
        "",
        "## Liquidity",
        f"- Cash: {liquidity.get('cash')}",
        f"- Total value: {liquidity.get('total_value')}",
        f"- Cash ratio: {liquidity.get('cash_ratio')}",
        f"- Open positions: {liquidity.get('open_positions')}",
        "",
        "## Forecaster Health",
        f"- Metrics: {forecaster.get('metrics')}",
        f"- Status: {forecaster.get('status')}",
        "",
        "## Quant Validation",
        f"- Summary: {quant}",
        "",
        "## Execution Log (Run-Scoped)",
        f"- Events considered: {execution.get('events_considered')}",
        f"- Status counts: {execution.get('status_counts')}",
        f"- Executed: {execution.get('executed')}",
        f"- Avg mid slippage (bp): {execution.get('avg_mid_slippage_bp')}",
        f"- Avg confidence: {execution.get('avg_signal_confidence')}",
        f"- Avg expected return: {execution.get('avg_expected_return')}",
    ]

    top_rejects = execution.get("top_rejections") or []
    if top_rejects:
        lines.append(f"- Top rejections: {top_rejects}")

    lines.extend(["", "## Next Actions"])
    if actions:
        for action in actions:
            lines.append(f"- {action}")
    else:
        lines.append("- (none recorded)")

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the latest run_summary/execution_log entry.")
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY_PATH)
    parser.add_argument("--execution-log", type=Path, default=DEFAULT_EXECUTION_LOG_PATH)
    parser.add_argument("--run-id", default=None, help="Optional run_id (defaults to latest record).")
    parser.add_argument(
        "--execution-events",
        type=int,
        default=200,
        help="Max execution_log events (run-scoped) to consider.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write markdown summary (stdout when omitted).",
    )
    parser.add_argument("--json", dest="emit_json", action="store_true", help="Emit JSON summary instead of markdown.")

    args = parser.parse_args(argv)
    selection = load_run_summary(args.run_summary, run_id=args.run_id)
    if selection is None:
        raise SystemExit(f"No run summary found in {args.run_summary}")

    execution = summarize_execution_log(args.execution_log, run_id=selection.run_id, limit=args.execution_events)
    summary = build_summary(selection.record, execution_stats=execution)

    if args.emit_json:
        output = json.dumps(summary, indent=2, sort_keys=True, default=str)
    else:
        output = render_markdown(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

