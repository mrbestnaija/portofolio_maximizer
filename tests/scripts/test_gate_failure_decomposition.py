from __future__ import annotations

import json
import sys
from pathlib import Path


def _write_gate_artifact(path: Path) -> None:
    payload = {
        "timestamp_utc": "2026-03-09T21:31:07Z",
        "phase3_ready": False,
        "phase3_reason": "GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL",
        "inputs": {
            "audit_dir": "logs/forecast_audits/production",
            "max_files": 50,
            "include_research": False,
        },
        "lift_gate": {
            "pass": False,
            "violation_rate": 0.7222,
            "max_violation_rate": 0.35,
            "lift_fraction": 0.0833,
            "min_lift_fraction": 0.25,
        },
        "profitability_proof": {
            "pass": False,
            "is_proof_valid": True,
            "is_profitable": False,
            "profit_factor": 0.60,
            "win_rate": 0.39,
            "total_pnl": -986.14,
            "closed_trades": 41,
            "trading_days": 11,
        },
        "readiness": {
            "gates_pass": False,
            "linkage_pass": False,
            "evidence_hygiene_pass": False,
            "outcome_matched": 0,
            "outcome_eligible": 1,
            "matched_over_eligible": 0.0,
            "non_trade_context_count": 33,
            "invalid_context_count": 30,
            "linkage_waterfall": {
                "raw_candidates": 82,
                "production_only": 49,
                "linked": 1,
                "hygiene_pass": 19,
                "matched": 0,
                "excluded_non_trade_context": 33,
                "excluded_invalid_context": 30,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_summary_cache(path: Path) -> None:
    payload = {
        "audit_dir": "logs/forecast_audits/production",
        "max_files": 50,
        "scope": {"include_research": False},
        "dataset_windows": [
            {"outcome_status": "INVALID_CONTEXT", "outcome_reason": "MISSING_EXECUTION_METADATA"},
            {"outcome_status": "INVALID_CONTEXT", "outcome_reason": "MISSING_EXECUTION_METADATA"},
            {"outcome_status": "INVALID_CONTEXT", "outcome_reason": "HORIZON_MISMATCH"},
            {"outcome_status": "NON_TRADE_CONTEXT", "outcome_reason": "MISSING_TICKER"},
            {"outcome_status": "NON_TRADE_CONTEXT", "outcome_reason": "MISSING_TICKER"},
            {"outcome_status": "NON_TRADE_CONTEXT", "outcome_reason": "NON_TRADE_CONTEXT"},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_gate_failure_decomposition_preserves_linkage_counts(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "production_gate_latest.json"
    _write_gate_artifact(artifact)
    output = tmp_path / "decomposition.json"
    output_md = tmp_path / "decomposition.md"
    summary_cache = tmp_path / "latest_summary.json"
    _write_summary_cache(summary_cache)

    import scripts.gate_failure_decomposition as mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate_failure_decomposition.py",
            "--gate-artifact",
            str(artifact),
            "--out-json",
            str(output),
            "--summary-cache",
            str(summary_cache),
            "--out-md",
            str(output_md),
        ],
    )
    rc = mod.main()
    assert rc == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    linkage = data["components"]["LINKAGE_BLOCKER"]
    assert linkage["pass"] is False
    assert linkage["metrics"]["outcome_matched"]["value"] == 0
    assert linkage["metrics"]["outcome_eligible"]["value"] == 1
    assert linkage["waterfall"]["raw_candidates"] == 82
    assert linkage["waterfall"]["linked"] == 1
    assert linkage["waterfall"]["matched"] == 0
    assert linkage["waterfall"]["excluded_non_trade_context"] == 33
    assert linkage["waterfall"]["excluded_invalid_context"] == 30
    reason_breakdown = data["reason_breakdown"]
    invalid = reason_breakdown["invalid_context_top_reasons"]
    non_trade = reason_breakdown["non_trade_context_top_reasons"]
    invalid_map = {str(item["reason_code"]): int(item["count"]) for item in invalid}
    non_trade_map = {str(item["reason_code"]): int(item["count"]) for item in non_trade}
    assert invalid_map["MISSING_EXECUTION_METADATA"] == 2
    assert non_trade_map["MISSING_TICKER"] == 2
    assert invalid_map["UNATTRIBUTED_INVALID_CONTEXT"] == 27
    assert non_trade_map["UNATTRIBUTED_NON_TRADE_CONTEXT"] == 30
    assert output_md.exists()


def test_gate_failure_decomposition_returns_nonzero_when_missing_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.gate_failure_decomposition as mod

    output = tmp_path / "decomposition.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate_failure_decomposition.py",
            "--gate-artifact",
            str(tmp_path / "missing.json"),
            "--out-json",
            str(output),
            "--out-md",
            str(tmp_path / "decomposition.md"),
        ],
    )
    rc = mod.main()
    assert rc == 1
    assert not output.exists()


def test_gate_failure_decomposition_marks_summary_binding_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "production_gate_latest.json"
    _write_gate_artifact(artifact)
    output = tmp_path / "decomposition.json"
    output_md = tmp_path / "decomposition.md"
    summary_cache = tmp_path / "latest_summary.json"
    _write_summary_cache(summary_cache)

    payload = json.loads(summary_cache.read_text(encoding="utf-8"))
    payload["audit_dir"] = str(tmp_path / "different_audit_dir")
    summary_cache.write_text(json.dumps(payload), encoding="utf-8")

    import scripts.gate_failure_decomposition as mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate_failure_decomposition.py",
            "--gate-artifact",
            str(artifact),
            "--out-json",
            str(output),
            "--summary-cache",
            str(summary_cache),
            "--out-md",
            str(output_md),
        ],
    )
    rc = mod.main()
    assert rc == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    reason_breakdown = data["reason_breakdown"]
    assert reason_breakdown["available"] is False
    assert reason_breakdown["binding_match"] is False
    assert "audit_dir" in reason_breakdown["binding_mismatch"]
    assert reason_breakdown["invalid_context_top_reasons"][0]["reason_code"] == "UNATTRIBUTED_INVALID_CONTEXT"
    assert reason_breakdown["invalid_context_top_reasons"][0]["count"] == 30
    assert reason_breakdown["non_trade_context_top_reasons"][0]["reason_code"] == "UNATTRIBUTED_NON_TRADE_CONTEXT"
    assert reason_breakdown["non_trade_context_top_reasons"][0]["count"] == 33
    assert output_md.exists()


def test_refresh_decomposition_report_regenerates_stale_latest_report(tmp_path: Path) -> None:
    artifact = tmp_path / "production_gate_latest.json"
    _write_gate_artifact(artifact)
    summary_cache = tmp_path / "latest_summary.json"
    _write_summary_cache(summary_cache)
    output = tmp_path / "decomposition.json"
    output_md = tmp_path / "decomposition.md"
    output.write_text(
        json.dumps(
            {
                "generated_utc": "2026-03-01T00:00:00+00:00",
                "source_artifact": str(artifact.resolve()),
                "source_timestamp_utc": "2026-03-01T00:00:00Z",
                "phase3_ready": True,
                "phase3_reason": "STALE",
            }
        ),
        encoding="utf-8",
    )

    import scripts.gate_failure_decomposition as mod

    report, refresh = mod.refresh_decomposition_report(
        artifact_path=artifact,
        output_json_path=output,
        summary_cache_path=summary_cache,
        output_md_path=output_md,
        force=False,
    )

    assert refresh["ok"] is True
    assert refresh["refreshed"] is True
    assert refresh["reason"] == "source_timestamp_mismatch"
    assert report["source_timestamp_utc"] == "2026-03-09T21:31:07Z"
    assert report["source_artifact_mtime_utc"]
    assert output_md.exists()
