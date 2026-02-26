from __future__ import annotations

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

