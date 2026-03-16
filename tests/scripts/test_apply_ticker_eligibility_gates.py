"""
Tests for scripts/apply_ticker_eligibility_gates.py
"""
from __future__ import annotations

import json


def test_apply_eligibility_gates_writes_expected_lists(tmp_path):
    from scripts.apply_ticker_eligibility_gates import apply_eligibility_gates

    eligibility_path = tmp_path / "eligibility.json"
    output_path = tmp_path / "gates.json"
    eligibility_path.write_text(
        json.dumps(
            {
                "tickers": {
                    "AAPL": {"status": "HEALTHY"},
                    "TSLA": {"status": "LAB_ONLY"},
                    "NVDA": {"status": "WEAK"},
                }
            }
        ),
        encoding="utf-8",
    )

    result = apply_eligibility_gates(
        eligibility_path=eligibility_path,
        output_path=output_path,
    )

    assert result["status"] == "PASS"
    assert result["healthy_tickers"] == ["AAPL"]
    assert result["weak_tickers"] == ["NVDA"]
    assert result["lab_only_tickers"] == ["TSLA"]
    assert result["gate_written"] is True
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"] == {"HEALTHY": 1, "WEAK": 1, "LAB_ONLY": 1}


def test_apply_eligibility_gates_missing_input_is_warn_and_empty(tmp_path):
    from scripts.apply_ticker_eligibility_gates import apply_eligibility_gates

    output_path = tmp_path / "gates.json"
    result = apply_eligibility_gates(
        eligibility_path=tmp_path / "missing.json",
        output_path=output_path,
    )

    assert result["status"] == "WARN"
    assert "eligibility_missing" in result["warnings"]
    assert result["healthy_tickers"] == []
    assert result["weak_tickers"] == []
    assert result["lab_only_tickers"] == []
    assert output_path.exists()


def test_cli_json_output(tmp_path, capsys):
    from scripts.apply_ticker_eligibility_gates import main

    eligibility_path = tmp_path / "eligibility.json"
    output_path = tmp_path / "gates.json"
    eligibility_path.write_text(
        json.dumps(
            {
                "tickers": {
                    "MSFT": {"status": "LAB_ONLY"},
                }
            }
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "--eligibility",
            str(eligibility_path),
            "--output",
            str(output_path),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["lab_only_tickers"] == ["MSFT"]
