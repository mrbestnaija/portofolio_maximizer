from __future__ import annotations

import json
from pathlib import Path

from scripts import openclaw_production_readiness as mod


def test_gate_truth_posture_detects_skip_policy_and_phase3_drift(tmp_path: Path) -> None:
    gate_artifact = tmp_path / "gate_status_latest.json"
    gate_artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-14T17:20:45Z",
                "overall_passed": False,
                "phase3_ready": True,
                "phase3_reason": "READY",
                "skipped_optional_gates": 2,
                "max_skipped_optional_gates": 1,
                "skipped_gate_labels": ["check_quant_validation_health", "institutional_unattended_gate"],
            }
        ),
        encoding="utf-8",
    )
    production_artifact = tmp_path / "production_gate_latest.json"
    production_artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-14T17:30:28Z",
                "pass_semantics_version": 3,
                "warmup_expired": False,
                "phase3_ready": False,
                "phase3_reason": "GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL",
                "production_profitability_gate": {
                    "gate_semantics_status": "FAIL",
                },
            }
        ),
        encoding="utf-8",
    )

    truth, blockers, warnings = mod._gate_truth_posture(
        gate_artifact_path=gate_artifact,
        production_gate_artifact_path=production_artifact,
        refresh_production_gate=False,
        timeout_seconds=5.0,
    )

    codes = {row["code"] for row in blockers}
    assert "gate_skip_policy_failed" in codes
    assert "stale_gate_artifact_phase3_drift" in codes
    assert truth["freshest_phase3_source"] == "production_gate_latest"
    assert truth["effective_phase3_ready"] is False
    assert warnings == []


def test_collect_openclaw_production_readiness_suppresses_noisy_helper_output(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(mod, "_gate_truth_posture", lambda **kwargs: ({"gate_artifact": {}, "production_gate_artifact": {}, "freshest_phase3_source": "gate_status_latest", "effective_phase3_ready": None, "effective_phase3_reason": "", "drift_detected": False}, [], []))
    monkeypatch.setattr(mod, "_openclaw_model_posture", lambda path: ({}, [], []))
    monkeypatch.setattr(mod, "_security_posture", lambda: ({}, [], []))
    monkeypatch.setattr(mod, "_openclaw_exec_env_posture", lambda: ({"ok": True}, []))
    monkeypatch.setattr(mod, "_openclaw_regression_posture", lambda timeout_seconds: ({"status": "PASS"}, [], []))

    def _chatty_capital():
        print("NOISY STDOUT FROM HELPER")
        print("NOISY STDERR FROM HELPER", file=__import__("sys").stderr)
        return {
            "ready": False,
            "verdict": "FAIL",
            "reasons": ["R2: gate artifact overall_passed=False"],
            "warnings": [],
            "metrics": {},
        }

    monkeypatch.setattr(mod.capital_mod, "run_capital_readiness", _chatty_capital)

    rc = mod.main(["--json"])
    assert rc == 1

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["action"] == "assess_production_readiness"
    captured = payload["capital_readiness"]["captured_output"]
    assert "NOISY STDOUT FROM HELPER" in "\n".join(captured["stdout_tail"])
    assert "NOISY STDERR FROM HELPER" in "\n".join(captured["stderr_tail"])


def test_action_guide_cli_outputs_requested_human_steps(monkeypatch, capsys) -> None:
    payload = {
        "readiness_status": "FAIL",
        "human_action_guides": [
            {
                "id": "approval_token",
                "title": "Set a non-default approval token",
                "cli_hint": "python scripts/openclaw_production_readiness.py --action-guide approval_token",
                "steps": ["step one"],
                "commands": ["cmd one"],
            },
            {
                "id": "capital_readiness",
                "title": "Triage economics and evidence depth",
                "cli_hint": "python scripts/openclaw_production_readiness.py --action-guide capital_readiness",
                "steps": ["step two"],
                "commands": ["cmd two"],
            },
        ],
    }
    monkeypatch.setattr(mod, "collect_openclaw_production_readiness", lambda **kwargs: payload)

    rc = mod.main(["--action-guide", "approval_token", "--json"])
    assert rc == 0

    out = capsys.readouterr().out
    guide_payload = json.loads(out)
    assert guide_payload["action"] == "production_readiness_action_guide"
    assert guide_payload["requested_guide"] == "approval_token"
    assert len(guide_payload["guides"]) == 1
    assert guide_payload["guides"][0]["id"] == "approval_token"
