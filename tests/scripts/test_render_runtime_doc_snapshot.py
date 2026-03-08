from __future__ import annotations

import json
from pathlib import Path

from scripts import render_runtime_doc_snapshot as mod


def test_build_snapshot_includes_key_fields(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "visualizations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "overnight_denominator").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "audit_gate").mkdir(parents=True, exist_ok=True)

    (tmp_path / "visualizations" / "dashboard_data.json").write_text(
        json.dumps(
            {
                "checks": ["performance_metrics missing"],
                "performance_unknown": True,
                "positions_stale": True,
                "positions_source": "trade_executions_fallback_stale",
                "meta": {"data_origin": "mixed", "payload_schema_version": 2},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "logs" / "overnight_denominator" / "live_denominator_latest.json").write_text(
        json.dumps({"cycles": [{"fresh_trade_rows": 1, "fresh_linkage_included": 2, "fresh_production_valid_matched": 1}]}),
        encoding="utf-8",
    )
    (tmp_path / "logs" / "audit_gate" / "production_gate_latest.json").write_text(
        json.dumps({"phase3_ready": False, "phase3_reason": "THIN_LINKAGE"}),
        encoding="utf-8",
    )

    def _fake_run_json(cmd: list[str]) -> dict:
        if "project_runtime_status.py" in cmd[1]:
            return {"status": "degraded", "failed_checks": ["production_gate"]}
        return {"ready": False, "verdict": "FAIL", "reasons": ["R5 fail"]}

    monkeypatch.setattr(mod, "_run_json", _fake_run_json)

    snapshot = mod.build_snapshot(root=tmp_path)

    assert "Doc Type: status_snapshot" in snapshot
    assert "- overall status: `degraded`" in snapshot
    assert "- data_origin: `mixed`" in snapshot
    assert "- fresh_production_valid_matched: `1`" in snapshot
