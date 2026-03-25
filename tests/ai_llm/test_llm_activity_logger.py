from __future__ import annotations

from ai_llm.llm_activity_logger import LLMActivityLogger


def test_llm_activity_logger_tracks_metadata_and_resolution_counts(tmp_path) -> None:
    logger = LLMActivityLogger(log_dir=tmp_path)

    logger.log_tool_call(
        orchestrator="qwen3:8b",
        tool="review_changed_files",
        arguments={"paths": ["scripts/example.py"]},
        result="PASS",
        metadata={"gate_name": "production_gate", "test_ids": ["tests/scripts/test_example.py::test_case"]},
    )
    logger.log_self_improvement(
        action="propose_change",
        target_file="scripts/example.py",
        description="Tighten operator review flow",
        approved=True,
        applied=True,
        resolved=True,
        metadata={"incident_id": "INC-7", "artifact_path": "logs/audit_gate/production_gate_latest.json"},
    )

    recent = logger.get_recent(hours=24)
    summary = logger.get_summary()

    assert any(entry.get("metadata", {}).get("gate_name") == "production_gate" for entry in recent)
    assert summary["approved_self_improvements"] == 1
    assert summary["applied_self_improvements"] == 1
    assert summary["resolved_events"] == 1
