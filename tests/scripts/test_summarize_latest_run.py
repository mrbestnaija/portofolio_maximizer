from __future__ import annotations

import json
from pathlib import Path

from scripts import summarize_latest_run


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_run_summary_selects_latest(tmp_path: Path) -> None:
    run_summary = tmp_path / "run_summary.jsonl"
    _write_jsonl(
        run_summary,
        [
            {"run_id": "run1", "profitability": {"pnl_dollars": 0.0}},
            {"run_id": "run2", "profitability": {"pnl_dollars": 1.0}, "next_actions": ["do thing"]},
        ],
    )

    selection = summarize_latest_run.load_run_summary(run_summary)
    assert selection is not None
    assert selection.run_id == "run2"
    assert selection.record["profitability"]["pnl_dollars"] == 1.0


def test_summarize_execution_log_filters_and_averages(tmp_path: Path) -> None:
    execution_log = tmp_path / "execution_log.jsonl"
    _write_jsonl(
        execution_log,
        [
            {"run_id": "run1", "status": "EXECUTED", "mid_slippage_bp": 5.0},
            {
                "run_id": "run2",
                "status": "EXECUTED",
                "mid_slippage_bp": 10.0,
                "signal_confidence": 0.8,
                "expected_return": 0.01,
            },
            {"run_id": "run2", "status": "REJECTED", "reason": "Non-actionable signal"},
        ],
    )

    summary = summarize_latest_run.summarize_execution_log(execution_log, run_id="run2", limit=100)
    assert summary["events_considered"] == 2
    assert summary["status_counts"]["EXECUTED"] == 1
    assert summary["status_counts"]["REJECTED"] == 1
    assert summary["executed"] == 1
    assert summary["avg_mid_slippage_bp"] == 10.0
    assert summary["avg_signal_confidence"] == 0.8
    assert summary["avg_expected_return"] == 0.01

