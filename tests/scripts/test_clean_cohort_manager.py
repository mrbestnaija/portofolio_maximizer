from __future__ import annotations

import json
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
    gate_calls = [call for call in calls if "production_audit_gate.py" in str(call["cmd"])]  # type: ignore[index]
    assert gate_calls
    gate_cmd = gate_calls[0]["cmd"]  # type: ignore[index]
    assert "--audit-dir" in gate_cmd
    assert str(production_dir) in gate_cmd
    assert "--output" in gate_cmd
    assert str(gate_output) in gate_cmd
    assert any("run_all_gates.py" in str(call["cmd"]) for call in calls)  # type: ignore[index]


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
