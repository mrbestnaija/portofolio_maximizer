import json
from pathlib import Path

import pytest

from scripts import check_quant_validation_health as qv_health


@pytest.fixture
def tmp_quant_log(tmp_path: Path) -> Path:
    """Create a temporary quant_validation.jsonl-style log file."""
    log_path = tmp_path / "quant_validation.jsonl"
    entries = [
        {
            "ticker": "AAPL",
            "status": "PASS",
            "expected_profit": 100.0,
        },
        {
            "ticker": "AAPL",
            "status": "FAIL",
            "expected_profit": -50.0,
        },
        {
            "ticker": "MSFT",
            "quant_validation": {"status": "FAIL", "expected_profit": -10.0},
        },
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        for rec in entries:
            handle.write(json.dumps(rec) + "\n")
    return log_path


def test_summarize_global_counts_and_fractions(tmp_quant_log: Path) -> None:
    entries = qv_health._load_entries(tmp_quant_log)
    summary = qv_health._summarize_global(entries)

    assert summary.total == 3
    assert summary.pass_count == 1
    assert summary.fail_count == 2
    # Two entries with negative expected_profit
    assert summary.negative_expected_profit_count == 2
    assert summary.fail_fraction == pytest.approx(2 / 3)
    assert summary.negative_expected_profit_fraction == pytest.approx(2 / 3)


def test_health_check_passes_when_within_limits(tmp_quant_log: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure generous thresholds so health check passes.
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_quant_validation_health",
            "--log-path",
            str(tmp_quant_log),
            "--max-fail-fraction",
            "0.9",
            "--max-negative-expected-profit-fraction",
            "0.9",
        ],
        raising=False,
    )

    # Should not raise SystemExit when within limits.
    qv_health.main()


def test_health_check_fails_when_limits_exceeded(tmp_quant_log: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Tight thresholds so our synthetic log violates both.
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_quant_validation_health",
            "--log-path",
            str(tmp_quant_log),
            "--max-fail-fraction",
            "0.5",
            "--max-negative-expected-profit-fraction",
            "0.5",
        ],
        raising=False,
    )

    with pytest.raises(SystemExit) as exc:
        qv_health.main()

    # Non-zero exit code signals failure for CI.
    assert exc.value.code == 1
