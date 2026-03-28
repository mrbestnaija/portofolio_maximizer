from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from scripts import clean_cohort_manager


def test_freeze_clean_cohort_writes_identity_and_activation_script(tmp_path: Path) -> None:
    payload = clean_cohort_manager.freeze_clean_cohort(
        cohort_id="2026Q1_cleanroom",
        cohort_root=tmp_path,
        build_fingerprint="commit_abc123",
    )

    cohort_root = tmp_path / "2026Q1_cleanroom"
    identity_path = cohort_root / "cohort_identity.json"
    activation_path = cohort_root / "activate_clean_cohort.ps1"

    assert identity_path.exists()
    assert activation_path.exists()
    assert payload["cohort_identity"]["cohort_id"] == "2026Q1_cleanroom"
    assert payload["cohort_identity"]["build_fingerprint"] == "commit_abc123"
    assert payload["cohort_identity"]["routing_mode"] == "explicit_env"
    assert "TS_FORECAST_AUDIT_DIR" in activation_path.read_text(encoding="utf-8")


def test_freeze_clean_cohort_rejects_fingerprint_drift_without_force(tmp_path: Path) -> None:
    clean_cohort_manager.freeze_clean_cohort(
        cohort_id="2026Q1_cleanroom",
        cohort_root=tmp_path,
        build_fingerprint="commit_abc123",
    )

    try:
        clean_cohort_manager.freeze_clean_cohort(
            cohort_id="2026Q1_cleanroom",
            cohort_root=tmp_path,
            build_fingerprint="commit_def456",
        )
    except ValueError as exc:
        assert "fingerprint differs" in str(exc)
    else:
        raise AssertionError("expected fingerprint drift to be rejected")


def test_summarize_evidence_provenance_distinguishes_producer_native(tmp_path: Path) -> None:
    audit_dir = tmp_path / "production"
    audit_dir.mkdir(parents=True)
    (audit_dir / "forecast_audit_one.json").write_text(
        json.dumps(
            {
                "evidence_source_classification": "producer-native",
                "semantic_admission": {"gate_eligible": True},
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "forecast_audit_two.json").write_text(
        json.dumps(
            {
                "evidence_source_classification": "legacy-derived",
                "semantic_admission": {"gate_eligible": False},
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "audit_failure_one.json").write_text(
        json.dumps(
            {
                "artifact_type": "AUDIT_PATCH_FAILURE",
                "reconciliation_bucket": "EXPLICIT_FAILED",
                "evidence_source_classification": "producer-native",
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "forecast_audit_manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"file": "forecast_audit_one.json", "evidence_source_classification": "producer-native"}),
                json.dumps({"file": "forecast_audit_two.json", "evidence_source_classification": "legacy-derived"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = clean_cohort_manager.summarize_evidence_provenance(audit_dir)

    assert summary["total_records"] == 2
    assert summary["producer_native_records"] == 1
    assert summary["legacy_derived_records"] == 1
    assert summary["eligible_records"] == 1
    assert summary["eligible_producer_native_records"] == 1
    assert summary["explicit_failed_records"] == 1
    assert summary["manifest_records"] == 2
    assert summary["manifest_producer_native_records"] == 1


def test_summarize_cohort_funnel_counts_producer_native_matched_round_trip(tmp_path: Path) -> None:
    audit_dir = tmp_path / "production"
    audit_dir.mkdir(parents=True)
    payload = {
        "cohort_id": "terminal_b_exec_round",
        "evidence_source_classification": "producer-native",
        "signal_context": {
            "context_type": "TRADE",
            "ts_signal_id": "ts_AAPL_20260314_000001",
            "run_id": "replay_20260314",
            "entry_ts": "2026-01-02T09:30:00+00:00",
            "expected_close_ts": "2026-02-03T09:30:00+00:00",
        },
        "lineage_v2": {
            "context_type": "TRADE",
            "ts_signal_id": "ts_AAPL_20260314_000001",
            "run_id": "replay_20260314",
            "entry_ts": "2026-01-02T09:30:00+00:00",
            "expected_close_ts": "2026-02-03T09:30:00+00:00",
        },
        "semantic_admission": {
            "gate_eligible": True,
            "gate_bucket": "ELIGIBLE",
            "reason_code": "READY",
            "source_classification": "producer-native",
            "missing_execution_metadata": False,
            "missing_execution_metadata_fields": [],
            "quarantined": False,
        },
    }
    (audit_dir / "forecast_audit_exec.json").write_text(json.dumps(payload), encoding="utf-8")
    (audit_dir / "forecast_audit_manifest.jsonl").write_text(
        json.dumps({"file": "forecast_audit_exec.json", "evidence_source_classification": "producer-native"}) + "\n",
        encoding="utf-8",
    )
    db_path = audit_dir / "proof.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ts_signal_id TEXT,
                is_close INTEGER,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0
            );
            CREATE VIEW production_closed_trades AS
            SELECT *
            FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_diagnostic, 0) = 0
              AND COALESCE(is_synthetic, 0) = 0;
            """
        )
        conn.execute(
            "INSERT INTO trade_executions (id, ts_signal_id, is_close, is_diagnostic, is_synthetic) VALUES (202, 'ts_AAPL_20260314_000001', 1, 0, 0)"
        )
        conn.commit()
    finally:
        conn.close()

    summary = clean_cohort_manager.summarize_cohort_funnel(audit_dir, db_path=db_path)

    assert summary["accepted_records"] == 1
    assert summary["eligible_records"] == 1
    assert summary["matched_records"] == 1
    assert summary["outcome_missing_records"] == 0
    assert summary["producer_native_records"] == 1
    assert summary["producer_native_funnel"]["accepted"] == 1
    assert summary["producer_native_funnel"]["eligible"] == 1
    assert summary["producer_native_funnel"]["matched"] == 1
    assert summary["bucket_exclusivity_ok"] is True
    assert summary["manifest_producer_native_records"] == 1


def test_run_clean_cohort_proof_loop_uses_cohort_specific_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run(cmd, cwd, env, capture_output, text, check):  # noqa: ANN001
        calls.append(
            {
                "cmd": list(cmd),
                "cwd": cwd,
                "env": {
                    "PMX_EVIDENCE_COHORT_ID": env.get("PMX_EVIDENCE_COHORT_ID"),
                    "PMX_BUILD_FINGERPRINT": env.get("PMX_BUILD_FINGERPRINT"),
                    "TS_FORECAST_AUDIT_DIR": env.get("TS_FORECAST_AUDIT_DIR"),
                },
            }
        )
        stdout = "{}"
        if "replay_trade_evidence_chain.py" in cmd[1]:
            stdout = json.dumps({"status": "PASS"})
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(clean_cohort_manager.subprocess, "run", _fake_run)

    summary = clean_cohort_manager.run_clean_cohort_proof_loop(
        cohort_id="2026Q1_cleanroom",
        cohort_root=tmp_path / "cohorts",
        replay_root=tmp_path / "replay",
        include_global_gates=True,
    )

    production_dir = tmp_path / "cohorts" / "2026Q1_cleanroom" / "production"
    gate_output = tmp_path / "cohorts" / "2026Q1_cleanroom" / "production_gate_latest.json"
    proof_output = tmp_path / "cohorts" / "2026Q1_cleanroom" / "proof_loop_latest.json"

    assert summary["overall_passed"] is True
    assert proof_output.exists()
    assert any(
        call["env"]["TS_FORECAST_AUDIT_DIR"] == str(production_dir)  # type: ignore[index]
        for call in calls
    )
    replay_calls = [call for call in calls if "replay_trade_evidence_chain.py" in str(call["cmd"])]  # type: ignore[index]
    assert replay_calls
    replay_cmd = replay_calls[0]["cmd"]  # type: ignore[index]
    assert "--output-dir" in replay_cmd
    assert str(production_dir) in replay_cmd
    assert "--cohort-id" in replay_cmd
    assert "2026Q1_cleanroom" in replay_cmd
    gate_calls = [call for call in calls if "production_audit_gate.py" in str(call["cmd"])]  # type: ignore[index]
    assert gate_calls
    gate_cmd = gate_calls[0]["cmd"]  # type: ignore[index]
    assert "--audit-dir" in gate_cmd
    assert str(production_dir) in gate_cmd
    assert "--output" in gate_cmd
    assert str(gate_output) in gate_cmd
    assert any("run_all_gates.py" in str(call["cmd"]) for call in calls)  # type: ignore[index]
    assert "provenance_summary" in summary
    assert "funnel_summary" in summary


def test_freeze_clean_cohort_reuses_existing_identity_when_build_not_provided(tmp_path: Path) -> None:
    first = clean_cohort_manager.freeze_clean_cohort(
        cohort_id="2026Q1_cleanroom",
        cohort_root=tmp_path,
        build_fingerprint="commit_abc123",
    )

    second = clean_cohort_manager.freeze_clean_cohort(
        cohort_id="2026Q1_cleanroom",
        cohort_root=tmp_path,
    )

    assert second["cohort_identity"]["contract_fingerprint"] == first["cohort_identity"]["contract_fingerprint"]
    assert second["cohort_identity"]["build_fingerprint"] == "commit_abc123"
