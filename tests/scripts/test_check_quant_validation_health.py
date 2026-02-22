import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from scripts import check_quant_validation_health as qv_health


@pytest.fixture
def tmp_quant_log(tmp_path: Path) -> Path:
    """Create a temporary quant_validation.jsonl-style log file."""
    log_path = tmp_path / "quant_validation.jsonl"
    entries = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "status": "PASS",
            "expected_profit": 100.0,
            "timestamp": datetime(2026, 2, 19, 15, 0, tzinfo=timezone.utc).isoformat(),
            "run_id": "run-a",
        },
        {
            "ticker": "AAPL",
            "action": "SELL",
            "status": "FAIL",
            "expected_profit": -50.0,
            "timestamp": datetime(2026, 2, 19, 16, 0, tzinfo=timezone.utc).isoformat(),
            "run_id": "run-a",
        },
        {
            "ticker": "MSFT",
            "action": "HOLD",
            "timestamp": datetime(2026, 2, 18, 12, 0, tzinfo=timezone.utc).isoformat(),
            "run_id": "run-b",
            "execution_mode": "proof",
            "proof_mode": True,
            "quant_validation": {"status": "FAIL", "expected_profit": -10.0},
        },
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        for rec in entries:
            handle.write(json.dumps(rec) + "\n")
    return log_path


def test_summarize_global_counts_and_fractions(tmp_quant_log: Path) -> None:
    entries = qv_health._load_entries(tmp_quant_log)
    summary = qv_health._summarize_global(entries, include_actions=["BUY", "SELL"])

    assert summary.total == 2
    assert summary.pass_count == 1
    assert summary.fail_count == 1
    assert summary.skipped_action_count == 1
    assert summary.negative_expected_profit_count == 1
    assert summary.fail_fraction == pytest.approx(1 / 2)
    assert summary.negative_expected_profit_fraction == pytest.approx(1 / 2)


def test_summarize_global_supports_run_id_and_since_filters(tmp_quant_log: Path) -> None:
    entries = qv_health._load_entries(tmp_quant_log)
    since = datetime(2026, 2, 19, 15, 30, tzinfo=timezone.utc)
    summary = qv_health._summarize_global(
        entries,
        include_actions=["BUY", "SELL"],
        run_ids=["run-a"],
        since_ts=since,
    )
    assert summary.total == 1
    assert summary.pass_count == 0
    assert summary.fail_count == 1
    assert summary.skipped_scope_count == 2


def test_summarize_global_exclude_mode_filters_proof_entries(tmp_quant_log: Path) -> None:
    entries = qv_health._load_entries(tmp_quant_log)
    summary = qv_health._summarize_global(
        entries,
        include_actions=["ALL"],
        exclude_modes=["proof"],
    )
    assert summary.total == 2
    assert summary.skipped_mode_count == 1


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
            "--include-action",
            "BUY",
            "SELL",
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
            "0.49",
            "--max-negative-expected-profit-fraction",
            "0.49",
            "--include-action",
            "BUY",
            "SELL",
        ],
        raising=False,
    )

    with pytest.raises(SystemExit) as exc:
        qv_health.main()

    # Non-zero exit code signals failure for CI.
    assert exc.value.code == 1


def test_health_check_fails_when_filters_remove_all_entries(tmp_quant_log: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_quant_validation_health",
            "--log-path",
            str(tmp_quant_log),
            "--run-id",
            "does-not-exist",
        ],
        raising=False,
    )
    with pytest.raises(SystemExit) as exc:
        qv_health.main()
    assert exc.value.code == 1
