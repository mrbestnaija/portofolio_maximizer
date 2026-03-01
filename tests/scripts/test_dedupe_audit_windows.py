"""Tests for scripts/dedupe_audit_windows.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.dedupe_audit_windows import _window_fingerprint, apply_removals, main, scan


def _write_audit(
    directory: Path,
    name: str,
    start: str = "2025-01-01",
    end: str = "2025-06-01",
    length: int = 100,
    horizon: int = 10,
    ticker: str | None = None,
) -> Path:
    data = {
        "dataset": {
            "start": start,
            "end": end,
            "length": length,
            "ticker": ticker,
        },
        "summary": {"forecast_horizon": horizon},
    }
    p = directory / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestDedupeAuditWindows:
    def test_identical_metrics_detected_as_duplicate(self, tmp_path):
        _write_audit(tmp_path, "forecast_audit_20260101_000000.json")
        _write_audit(tmp_path, "forecast_audit_20260102_000000.json")
        dups = scan(tmp_path)
        assert len(dups) == 1
        assert sum(len(v["remove"]) for v in dups.values()) == 1

    def test_different_windows_not_flagged(self, tmp_path):
        _write_audit(tmp_path, "forecast_audit_20260101_000000.json", end="2025-06-01")
        _write_audit(tmp_path, "forecast_audit_20260102_000000.json", end="2025-12-01")
        dups = scan(tmp_path)
        assert len(dups) == 0

    def test_keeps_newest_file(self, tmp_path):
        p1 = _write_audit(tmp_path, "forecast_audit_20260101_000000.json")
        time.sleep(0.05)
        p2 = _write_audit(tmp_path, "forecast_audit_20260102_000000.json")
        dups = scan(tmp_path)
        assert len(dups) == 1
        entry = list(dups.values())[0]
        # Newer file is kept
        assert entry["keep"].name == p2.name
        assert p1.name in [p.name for p in entry["remove"]]

    def test_apply_removes_older_file(self, tmp_path):
        p1 = _write_audit(tmp_path, "forecast_audit_20260101_000000.json")
        time.sleep(0.05)
        p2 = _write_audit(tmp_path, "forecast_audit_20260102_000000.json")
        rc = main(["--audit-dir", str(tmp_path), "--apply"])
        assert not p1.exists(), "Older duplicate should be removed"
        assert p2.exists(), "Newer file should be kept"

    def test_dry_run_does_not_delete(self, tmp_path):
        p1 = _write_audit(tmp_path, "forecast_audit_20260101_000000.json")
        p2 = _write_audit(tmp_path, "forecast_audit_20260102_000000.json")
        rc = main(["--audit-dir", str(tmp_path)])  # no --apply
        assert p1.exists(), "Dry-run must not delete files"
        assert p2.exists()

    def test_exit_code_1_when_duplicates_found(self, tmp_path):
        _write_audit(tmp_path, "forecast_audit_20260101_000000.json")
        _write_audit(tmp_path, "forecast_audit_20260102_000000.json")
        rc = main(["--audit-dir", str(tmp_path)])
        assert rc == 1

    def test_empty_dir_exits_cleanly(self, tmp_path):
        rc = main(["--audit-dir", str(tmp_path)])
        assert rc == 0

    def test_malformed_json_skipped_with_warning(self, tmp_path):
        (tmp_path / "forecast_audit_20260101_000000.json").write_text(
            "not valid json", encoding="utf-8"
        )
        rc = main(["--audit-dir", str(tmp_path)])
        assert rc == 0  # no duplicates among valid files

    def test_different_tickers_different_fingerprints(self, tmp_path):
        _write_audit(
            tmp_path, "forecast_audit_20260101_000000.json", ticker="AAPL"
        )
        _write_audit(
            tmp_path, "forecast_audit_20260102_000000.json", ticker="MSFT"
        )
        dups = scan(tmp_path)
        assert len(dups) == 0, "Different tickers must have different fingerprints"

    def test_write_summary_creates_file(self, tmp_path, monkeypatch):
        import scripts.dedupe_audit_windows as mod

        monkeypatch.setattr(mod, "ENSEMBLE_HEALTH_DIR", tmp_path / "health")
        mod.write_summary(duplicate_count=3, removed_count=1)
        summaries = list((tmp_path / "health").glob("dedupe_summary_*.json"))
        assert len(summaries) == 1
        data = json.loads(summaries[0].read_text(encoding="utf-8"))
        assert data["duplicate_count"] == 3
        assert data["removed_count"] == 1
