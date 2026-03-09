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


def test_apply_eligibility_gates_missing_input_is_fail_and_empty(tmp_path):
    """Missing evidence must be a hard FAIL (fail-closed governance)."""
    from scripts.apply_ticker_eligibility_gates import apply_eligibility_gates

    output_path = tmp_path / "gates.json"
    result = apply_eligibility_gates(
        eligibility_path=tmp_path / "missing.json",
        output_path=output_path,
    )

    assert result["status"] == "FAIL"
    assert result["reason"] == "missing_eligibility_evidence"
    assert "eligibility_missing" in result["errors"]
    assert result["warnings"] == []
    assert result["healthy_tickers"] == []
    assert result["weak_tickers"] == []
    assert result["lab_only_tickers"] == []
    assert output_path.exists()


def test_apply_eligibility_gates_corrupt_input_is_fail(tmp_path):
    """Unreadable/corrupt eligibility JSON must be a hard FAIL."""
    from scripts.apply_ticker_eligibility_gates import apply_eligibility_gates

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json!!", encoding="utf-8")
    output_path = tmp_path / "gates.json"
    result = apply_eligibility_gates(eligibility_path=bad_path, output_path=output_path)

    assert result["status"] == "FAIL"
    assert result["reason"] == "missing_eligibility_evidence"
    assert "eligibility_unreadable" in result["errors"]


def test_cli_missing_input_returns_nonzero_exit(tmp_path):
    """CLI must return exit code 1 when eligibility evidence is absent."""
    from scripts.apply_ticker_eligibility_gates import main

    rc = main(
        [
            "--eligibility",
            str(tmp_path / "no_such_file.json"),
            "--output",
            str(tmp_path / "gates.json"),
        ]
    )
    assert rc == 1


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
