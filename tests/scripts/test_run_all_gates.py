from __future__ import annotations

import json

import pytest

import scripts.run_all_gates as mod


def test_run_all_gates_includes_institutional_gate_by_default(monkeypatch, capsys) -> None:
    seen: list[str] = []

    def _fake_run(cmd, label):  # noqa: ANN001
        seen.append(label)
        return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(mod, "_run", _fake_run)

    with monkeypatch.context() as m:
        m.setattr(mod.sys, "argv", ["run_all_gates.py", "--json"])
        try:
            mod.main()
        except SystemExit as exc:
            assert exc.code == 0

    _ = capsys.readouterr()
    assert "institutional_unattended_gate" in seen


def test_run_all_gates_skip_institutional_gate(monkeypatch, capsys) -> None:
    seen: list[str] = []

    def _fake_run(cmd, label):  # noqa: ANN001
        seen.append(label)
        return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(mod, "_run", _fake_run)

    with monkeypatch.context() as m:
        m.setattr(mod.sys, "argv", ["run_all_gates.py", "--json", "--skip-institutional-gate"])
        try:
            mod.main()
        except SystemExit as exc:
            assert exc.code == 0

    _ = capsys.readouterr()
    assert "institutional_unattended_gate" not in seen


class TestSkipLimitEnforcement:
    """Phase 7.29 / BYP-01: MAX_SKIPPED_OPTIONAL_GATES blocks all-skip bypass."""

    def test_all_three_skips_yields_overall_false(self, monkeypatch, capsys):
        """Skipping all 3 optional gates must set overall_passed=False (exit 1)."""
        def _fake_run(cmd, label):  # noqa: ANN001
            return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}

        monkeypatch.setattr(mod, "_run", _fake_run)

        argv = [
            "run_all_gates.py", "--json",
            "--skip-forecast-gate",
            "--skip-profitability-gate",
            "--skip-institutional-gate",
        ]
        with monkeypatch.context() as m:
            m.setattr(mod.sys, "argv", argv)
            with pytest.raises(SystemExit) as exc_info:
                mod.main()

        assert exc_info.value.code == 1, (
            "Skipping all 3 optional gates must exit 1 (overall_passed=False)"
        )
        out = capsys.readouterr().out
        import json as _json
        summary = _json.loads(out)
        assert summary["overall_passed"] is False, (
            f"overall_passed should be False when all 3 optional gates skipped; got {summary}"
        )

    def test_one_skip_does_not_force_overall_false(self, monkeypatch, capsys):
        """Skipping exactly 1 optional gate is within the limit; result depends on Gate 1."""
        def _fake_run(cmd, label):  # noqa: ANN001
            return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}

        monkeypatch.setattr(mod, "_run", _fake_run)

        argv = ["run_all_gates.py", "--json", "--skip-institutional-gate"]
        with monkeypatch.context() as m:
            m.setattr(mod.sys, "argv", argv)
            with pytest.raises(SystemExit) as exc_info:
                mod.main()

        # 1 skip is within MAX_SKIPPED_OPTIONAL_GATES=1 — exit depends on actual gate results.
        # Since _fake_run always returns passed=True, overall should be PASS.
        assert exc_info.value.code == 0, (
            "1 optional gate skip should not force overall_passed=False"
        )
        out = capsys.readouterr().out
        import json as _json
        summary = _json.loads(out)
        assert summary["overall_passed"] is True


def test_run_all_gates_uses_unattended_profile_and_emits_phase3_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    artifact = tmp_path / "production_gate_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "pass_semantics_version": 3,
                "lift_inconclusive_allowed": False,
                "proof_profitable_required": True,
                "warmup_expired": True,
                "phase3_ready": False,
                "phase3_reason": "THIN_LINKAGE",
                "production_profitability_gate": {"gate_semantics_status": "INCONCLUSIVE_BLOCKED"},
                "readiness": {
                    "gates_pass": False,
                    "linkage_pass": False,
                    "evidence_hygiene_pass": True,
                    "integrity_pass": True,
                },
            }
        ),
        encoding="utf-8",
    )

    seen_cmds: list[list[str]] = []

    def _fake_run(cmd, label):  # noqa: ANN001
        seen_cmds.append(cmd)
        return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(mod, "_run", _fake_run)
    monkeypatch.setattr(mod, "PRODUCTION_GATE_ARTIFACT", artifact)

    with monkeypatch.context() as m:
        m.setattr(mod.sys, "argv", ["run_all_gates.py", "--json"])
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
    assert exc_info.value.code == 0

    production_cmd = next(cmd for cmd in seen_cmds if "scripts/production_audit_gate.py" in cmd)
    assert "--unattended-profile" in production_cmd

    summary = json.loads(capsys.readouterr().out)
    assert summary["pass_semantics_version"] == 3
    assert summary["lift_inconclusive_allowed"] is False
    assert summary["proof_profitable_required"] is True
    assert summary["warmup_expired"] is True
    assert summary["phase3_ready"] is False
    assert summary["phase3_reason"] == "THIN_LINKAGE"


def test_run_all_gates_writes_current_run_artifact_before_institutional_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    seen_overall_before_institutional: list[bool] = []

    def _fake_run(cmd, label):  # noqa: ANN001
        if label == "ci_integrity_gate":
            return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}
        if label == "check_quant_validation_health":
            return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}
        if label == "production_audit_gate":
            return {"label": label, "exit_code": 1, "passed": False, "stdout": "", "stderr": ""}
        if label == "institutional_unattended_gate":
            if mod.GATE_STATUS_ARTIFACT.exists():
                payload = json.loads(mod.GATE_STATUS_ARTIFACT.read_text(encoding="utf-8"))
                seen_overall_before_institutional.append(bool(payload.get("overall_passed")))
            return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}
        return {"label": label, "exit_code": 0, "passed": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(mod, "_run", _fake_run)
    monkeypatch.setattr(mod, "GATE_STATUS_ARTIFACT", tmp_path / "logs" / "gate_status_latest.json")
    monkeypatch.setattr(mod, "PRODUCTION_GATE_ARTIFACT", tmp_path / "logs" / "audit_gate" / "production_gate_latest.json")

    with monkeypatch.context() as m:
        m.setattr(mod.sys, "argv", ["run_all_gates.py", "--json"])
        with pytest.raises(SystemExit) as exc_info:
            mod.main()

    assert exc_info.value.code == 1
    assert seen_overall_before_institutional == [False], (
        "Institutional gate must observe current-run status artifact, not stale prior-run PASS."
    )
