from __future__ import annotations

import json
from pathlib import Path

from utils.evidence_io import build_manifest_entry, upsert_jsonl_record, write_promoted_json_artifact


def test_write_promoted_json_artifact_quarantines_invalid_payload(tmp_path: Path) -> None:
    stamped = tmp_path / "runs" / "artifact_20260314.json"
    latest = tmp_path / "artifact_latest.json"

    result = write_promoted_json_artifact(
        stamped_path=stamped,
        latest_path=latest,
        payload={"status": "BAD"},
        validate_fn=lambda payload: (False, "semantic_invalid"),
        quarantine_dir=tmp_path / "quarantine",
    )

    assert result["ok"] is False
    assert not stamped.exists()
    assert not latest.exists()
    quarantine_meta = Path(result["quarantine"]["meta_path"])
    assert quarantine_meta.exists()
    meta = json.loads(quarantine_meta.read_text(encoding="utf-8"))
    assert meta["reason"] == "semantic_invalid"


def test_upsert_jsonl_record_is_idempotent_for_same_file_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "audit.json"
    artifact.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    manifest = tmp_path / "forecast_audit_manifest.jsonl"
    entry = build_manifest_entry(
        artifact,
        source="test",
        extra={"audit_id": "audit_1", "event_type": "FORECAST_AUDIT"},
    )
    assert entry is not None

    first = upsert_jsonl_record(manifest, entry, key_field="file")
    second = upsert_jsonl_record(manifest, entry, key_field="file")

    assert first["updated"] is True
    assert second["updated"] is False
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
