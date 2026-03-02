from __future__ import annotations

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

