from __future__ import annotations

import json
from pathlib import Path

from scripts.sanitize_production_forecast_audits import (
    classify_audit,
    sanitize_production_forecast_audits,
)


def _write_audit(
    path: Path,
    *,
    ticker: str = "AAPL",
    dataset_end: str = "2026-02-11 00:00:00",
    entry_ts: str = "2026-03-25T00:00:00+00:00",
    expected_close_source: str | None = None,
) -> None:
    payload = {
        "dataset": {
            "ticker": ticker,
            "start": "2025-03-25 00:00:00",
            "end": dataset_end,
            "length": 232,
            "forecast_horizon": 30,
        },
        "signal_context": {
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "ts_signal_id": f"ts_{ticker}_0001",
            "entry_ts": entry_ts,
        },
        "event_type": "TRADE_FORECAST_AUDIT",
    }
    if expected_close_source is not None:
        payload["signal_context"]["expected_close_source"] = expected_close_source
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_classify_audit_flags_large_gap_only_when_source_missing(tmp_path: Path) -> None:
    audit_path = tmp_path / "forecast_audit_large_gap.json"
    _write_audit(audit_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))

    flagged = classify_audit(
        path=audit_path,
        payload=payload,
        max_positive_gap_days=7.0,
        max_negative_gap_days=1.0,
        require_missing_expected_close_source=True,
    )
    assert flagged["suspect"] is True
    assert "ENTRY_AFTER_DATASET_END_EXCESSIVE" in flagged["reason_codes"]
    assert "MISSING_EXPECTED_CLOSE_SOURCE" in flagged["reason_codes"]

    _write_audit(audit_path, expected_close_source="forecast_index")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    clean = classify_audit(
        path=audit_path,
        payload=payload,
        max_positive_gap_days=7.0,
        max_negative_gap_days=1.0,
        require_missing_expected_close_source=True,
    )
    assert clean["suspect"] is False


def test_sanitize_production_forecast_audits_quarantines_suspects_and_rebuilds_manifest(
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "production"
    audit_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = audit_dir / "quarantine"
    manifest_path = audit_dir / "forecast_audit_manifest.jsonl"

    _write_audit(audit_dir / "forecast_audit_bad.json")
    _write_audit(
        audit_dir / "forecast_audit_good.json",
        dataset_end="2026-03-24 00:00:00",
        entry_ts="2026-03-25T00:00:00+00:00",
        expected_close_source="forecast_index",
    )

    summary = sanitize_production_forecast_audits(
        audit_dir=audit_dir,
        quarantine_dir=quarantine_dir,
        manifest_path=manifest_path,
        apply=True,
    )

    assert summary["totals"]["audits_scanned"] == 2
    assert summary["totals"]["suspects"] == 1
    assert summary["totals"]["quarantined"] == 1
    assert not (audit_dir / "forecast_audit_bad.json").exists()
    assert (audit_dir / "forecast_audit_good.json").exists()
    quarantine_files = list(quarantine_dir.glob("forecast_audit_bad_*.json"))
    assert quarantine_files, "suspect audit should be moved to quarantine"

    manifest_lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_records = [json.loads(line) for line in manifest_lines]
    assert {record["file"] for record in manifest_records} == {"forecast_audit_good.json"}

