from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import scripts.replay_trade_evidence_chain as mod


def test_replay_trade_evidence_chain_happy_path_promotes_latest(tmp_path: Path) -> None:
    summary = mod.run_replay(output_dir=tmp_path, scenario="happy_path")

    assert summary["status"] == "PASS"
    assert summary["reason_code"] == "READY"
    latest = tmp_path / "latest.json"
    manifest = tmp_path / "forecast_audit_manifest.jsonl"
    assert latest.exists()
    assert manifest.exists()
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["semantic_admission"]["gate_eligible"] is True
    assert payload["evidence_source"] == "producer"
    assert payload["semantic_admission"]["source"] == "producer"
    manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
    assert manifest_lines
    manifest_entry = json.loads(manifest_lines[0])
    assert manifest_entry["evidence_source_classification"] == "producer-native"


def test_replay_trade_evidence_chain_duplicate_conflict_quarantines_conflict(tmp_path: Path) -> None:
    summary = mod.run_replay(output_dir=tmp_path, scenario="duplicate_conflict")

    assert summary["status"] == "PASS"
    quarantine_meta = list((tmp_path / "quarantine").glob("*.meta.json"))
    assert quarantine_meta, "duplicate conflict should quarantine the conflicting artifact"
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["semantic_admission"]["reason_code"] == "READY"


def test_replay_trade_evidence_chain_manifest_failure_never_promotes_latest(tmp_path: Path) -> None:
    summary = mod.run_replay(output_dir=tmp_path, scenario="manifest_registration_failure")

    assert summary["status"] == "FAIL"
    assert summary["reason_code"] == "MANIFEST_REGISTRATION_FAILED"
    assert not (tmp_path / "latest.json").exists()


def test_replay_trade_evidence_chain_executed_round_trip_writes_proof_db(tmp_path: Path) -> None:
    summary = mod.run_replay(
        output_dir=tmp_path / "cohorts" / "terminal_b_exec_round" / "production",
        scenario="executed_round_trip",
    )

    assert summary["status"] == "PASS"
    assert summary["cohort_id"] == "terminal_b_exec_round"
    proof_db_path = Path(summary["proof_db_path"])
    canonical_artifact_path = Path(summary["canonical_artifact_path"])
    assert proof_db_path.exists()
    assert canonical_artifact_path.exists()

    latest = json.loads((tmp_path / "cohorts" / "terminal_b_exec_round" / "production" / "latest.json").read_text(encoding="utf-8"))
    assert latest["cohort_id"] == "terminal_b_exec_round"
    assert latest["semantic_admission"]["gate_eligible"] is True
    assert latest["lineage_v2"]["close_trade_id"] == 202

    conn = sqlite3.connect(str(proof_db_path))
    try:
        rows = list(
            conn.execute(
                "SELECT id, entry_trade_id, ts_signal_id FROM production_closed_trades"
            )
        )
    finally:
        conn.close()
    assert rows == [(202, 101, "ts_AAPL_20260314_000001")]


def test_main_scopes_default_output_dir_by_scenario(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    happy_rc = mod.main(["--scenario", "happy_path", "--json"])
    missing_rc = mod.main(["--scenario", "missing_entry_link", "--json"])

    assert happy_rc == 0
    assert missing_rc == 1
    assert not (tmp_path / "logs" / "evidence_replay" / "latest.json").exists()
    assert (tmp_path / "logs" / "evidence_replay" / "happy_path" / "latest.json").exists()
    assert not (tmp_path / "logs" / "evidence_replay" / "missing_entry_link" / "latest.json").exists()


def test_main_respects_explicit_production_output_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    production_dir = tmp_path / "logs" / "forecast_audits" / "cohorts" / "terminal_b_exec_round" / "production"

    rc = mod.main(
        [
            "--scenario",
            "executed_round_trip",
            "--output-dir",
            str(production_dir),
            "--cohort-id",
            "terminal_b_exec_round",
            "--json",
        ]
    )

    assert rc == 0
    assert (production_dir / "latest.json").exists()
    assert not (production_dir / "executed_round_trip" / "latest.json").exists()
