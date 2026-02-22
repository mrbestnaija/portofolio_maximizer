from __future__ import annotations

import json
from pathlib import Path

from scripts import quant_validation_headroom as qh


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_summarize_headroom_uses_recent_window() -> None:
    rows = [
        {"ticker": "AAPL", "overall_result": "PASS"},
        {"ticker": "AAPL", "overall_result": "FAIL"},
        {"ticker": "MSFT", "overall_result": "FAIL"},
    ]
    summary = qh.summarize_headroom(
        entries=rows,
        window=2,
        red_gate_pct=95.0,
        warn_gate_pct=90.0,
    )
    assert summary.total == 2
    assert summary.fail_count == 2
    assert summary.fail_rate_pct == 100.0
    assert summary.status == "RED"


def test_status_extraction_supports_nested_quant_validation() -> None:
    rec = {"ticker": "BTC-USD", "quant_validation": {"status": "FAIL"}}
    assert qh._status_from_entry(rec) == "FAIL"


def test_main_json_output(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "quant_validation.jsonl"
    _write_jsonl(
        log_path,
        [
            {"ticker": "AAPL", "status": "PASS"},
            {"ticker": "AAPL", "status": "FAIL"},
            {"ticker": "MSFT", "status": "FAIL"},
        ],
    )
    rc = qh.main(
        [
            "--log-path",
            str(log_path),
            "--window",
            "120",
            "--red-gate-pct",
            "95",
            "--warn-gate-pct",
            "90",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["total"] == 3
    assert payload["fail_count"] == 2
    assert payload["status"] == "GREEN"
    assert isinstance(payload["per_ticker"], list)
