from __future__ import annotations

import scripts.prepare_llm_finetune_dataset as mod


def test_extract_record_supports_tool_call_with_labels() -> None:
    record = mod._extract_record(
        {
            "type": "tool_call",
            "orchestrator": "qwen3:8b",
            "tool": "triage_gate_failure",
            "arguments": {"gate_artifact": "logs/audit_gate/production_gate_latest.json"},
            "result_preview": "gate triage result",
            "metadata": {
                "gate_name": "production_gate",
                "artifact_path": "logs/audit_gate/production_gate_latest.json",
                "resolved": False,
            },
        }
    )

    assert record is not None
    assert record["source"] == "tool_call"
    assert record["task_type"] == "tool_call:triage_gate_failure"
    assert record["labels"]["gate_name"] == "production_gate"


def test_extract_record_supports_self_improvement_outcomes() -> None:
    record = mod._extract_record(
        {
            "type": "self_improvement",
            "action": "propose_change",
            "target_file": "scripts/example.py",
            "description": "Improve operator review prompts",
            "diff_preview": "+ added safe review helper",
            "approved": True,
            "applied": False,
            "resolved": True,
            "metadata": {
                "incident_id": "INC-42",
                "test_ids": ["tests/scripts/test_llm_operator_tools.py::test_gate_triage_fast_path_request_matches_prompt"],
            },
        }
    )

    assert record is not None
    assert record["source"] == "self_improvement"
    assert record["labels"]["resolved"] is True
    assert record["labels"]["incident_id"] == "INC-42"
    assert record["labels"]["test_ids"][0].endswith("test_gate_triage_fast_path_request_matches_prompt")
