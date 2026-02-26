from __future__ import annotations

import json

import scripts.institutional_unattended_gate as mod


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_gate_returns_structured_findings() -> None:
    findings = mod.run_gate()
    assert isinstance(findings, list)
    assert findings, "Gate must produce at least one finding."
    assert all(isinstance(f, mod.Finding) for f in findings)


def test_main_json_exit_code_fails_on_blocking_findings(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "run_gate",
        lambda: [mod.Finding("P0", "example", "FAIL", "blocking")],
    )
    rc = mod.main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 1
    assert payload[0]["status"] == "FAIL"


def test_main_json_exit_code_zero_when_no_failures(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "run_gate",
        lambda: [mod.Finding("P0", "example", "PASS", "ok")],
    )
    rc = mod.main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload[0]["status"] == "PASS"


def test_phase_p2_fails_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: _Proc(0, stdout="{not-json}"))
    findings = mod._phase_p2_platt_data()
    assert findings
    assert findings[0].status == "FAIL"
    assert "Unable to parse" in findings[0].detail


def test_phase_p2_fails_on_empty_findings(monkeypatch) -> None:
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: _Proc(0, stdout="[]"))
    findings = mod._phase_p2_platt_data()
    assert findings
    assert findings[0].status == "FAIL"
    assert "empty findings list" in findings[0].detail
