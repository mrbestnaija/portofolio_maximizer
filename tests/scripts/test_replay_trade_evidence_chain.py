from __future__ import annotations

import json
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


def test_replay_trade_evidence_chain_duplicate_conflict_quarantines_conflict(tmp_path: Path) -> None:
    summary = mod.run_replay(output_dir=tmp_path, scenario="duplicate_conflict")

    assert summary["status"] == "PASS"
    quarantine_meta = list((tmp_path / "quarantine").glob("*.meta.json"))
    assert quarantine_meta, "duplicate conflict should quarantine the conflicting artifact"
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["semantic_admission"]["reason_code"] == "READY"


def test_replay_trade_evidence_chain_missing_entry_link_is_preserved_noneligible(tmp_path: Path) -> None:
    summary = mod.run_replay(output_dir=tmp_path, scenario="missing_entry_link")

    assert summary["status"] == "PASS"
    assert summary["reason_code"] == "MISSING_ENTRY_TRADE_ID"
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["semantic_admission"]["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
    assert latest["semantic_admission"]["gate_eligible"] is False


def test_replay_trade_evidence_chain_blocked_by_net_edge_is_preserved_noneligible(tmp_path: Path) -> None:
    summary = mod.run_replay(output_dir=tmp_path, scenario="blocked_by_net_edge")

    assert summary["status"] == "PASS"
    assert summary["reason_code"] == "NON_POSITIVE_NET_EDGE"
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["semantic_admission"]["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
    assert latest["semantic_admission"]["reason_codes"] == ["NON_POSITIVE_NET_EDGE"]
    assert latest["execution_decision"]["execution_policy_blocked"] is True
    assert latest["execution_decision"]["source_classification"] == "producer-native"


def test_replay_trade_evidence_chain_manifest_failure_never_promotes_latest(tmp_path: Path) -> None:
    summary = mod.run_replay(output_dir=tmp_path, scenario="manifest_registration_failure")

    assert summary["status"] == "FAIL"
    assert summary["reason_code"] == "MANIFEST_REGISTRATION_FAILED"
    assert not (tmp_path / "latest.json").exists()
