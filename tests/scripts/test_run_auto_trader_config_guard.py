from __future__ import annotations

import pytest

from scripts.run_auto_trader import ConfigurationError, validate_production_ensemble_config


def test_validate_production_ensemble_config_blocks_live_false() -> None:
    with pytest.raises(ConfigurationError):
        validate_production_ensemble_config(
            ensemble_kwargs={"confidence_scaling": False},
            execution_mode="live",
        )


def test_validate_production_ensemble_config_allows_non_live_false() -> None:
    validate_production_ensemble_config(
        ensemble_kwargs={"confidence_scaling": False},
        execution_mode="synthetic",
    )



# ---------------------------------------------------------------------------
# P1-B: funnel audit JSONL logging for blocked (HOLD) signals
# ---------------------------------------------------------------------------

def test_write_funnel_audit_entry_creates_jsonl(tmp_path: "pytest.MonkeyPatch") -> None:
    """_write_funnel_audit_entry must write a JSONL entry with required fields."""
    import json
    import pytest
    from pathlib import Path
    from unittest.mock import patch

    funnel_log = tmp_path / "funnel_audit.jsonl"
    with patch("scripts.run_auto_trader.FUNNEL_AUDIT_LOG_PATH", funnel_log):
        from scripts.run_auto_trader import _write_funnel_audit_entry
        _write_funnel_audit_entry(
            ticker="AAPL",
            ts_signal_id="ts_AAPL_20260405_abc_0001",
            reason="confidence_below_threshold",
            confidence=0.42,
            snr=0.8,
            expected_return=0.0015,
        )

    assert funnel_log.exists(), "funnel_audit.jsonl must be created on first blocked signal"
    lines = funnel_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "Exactly one entry written"
    entry = json.loads(lines[0])
    assert entry["ticker"] == "AAPL"
    assert entry["ts_signal_id"] == "ts_AAPL_20260405_abc_0001"
    assert entry["reason"] == "confidence_below_threshold"
    assert abs(entry["confidence"] - 0.42) < 1e-6
    assert abs(entry["snr"] - 0.8) < 1e-6
    assert "logged_at" in entry
