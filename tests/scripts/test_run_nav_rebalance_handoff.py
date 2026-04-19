from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import run_nav_rebalance_handoff as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_handoff_noops_when_live_apply_is_not_allowed(tmp_path: Path, monkeypatch) -> None:
    plan_path = tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest.json"
    config_path = tmp_path / "config" / "barbell.yml"
    output_path = tmp_path / "logs" / "automation" / "nav_allocation_latest.json"
    status_path = tmp_path / "logs" / "automation" / "nav_rebalance_handoff_latest.json"

    _write_json(
        plan_path,
        {
            "rollout": {
                "mode": "shadow",
                "live_apply_allowed": False,
                "gate_lift_candidate": False,
                "gate_lift_ready": False,
                "live_apply_blockers": ["shadow_first_default", "evidence_not_green"],
            },
            "summary": {"weak": ["AAPL", "GS"]},
        },
    )
    _write_yaml(
        config_path,
        """
barbell:
  safe_bucket:
    symbols: ["CASH"]
  core_bucket:
    symbols: ["AAPL"]
  speculative_bucket:
    symbols: ["NVDA"]
""".strip(),
    )

    called = []

    def _fail_if_called(*args, **kwargs):  # pragma: no cover - defensive
        called.append((args, kwargs))
        raise AssertionError("apply_nav_reallocation subprocess must not run in shadow mode")

    monkeypatch.setattr(mod.subprocess, "run", _fail_if_called)

    result = mod.run_nav_rebalance_handoff(
        plan_path=plan_path,
        config_path=config_path,
        output_path=output_path,
        staged_config_path=tmp_path / "config" / "barbell.staged.yml",
        status_path=status_path,
    )

    assert called == []
    assert result["status"] == "NO_OP"
    assert result["action_taken"] == "NO_OP"
    assert result["exit_code"] == 0
    assert result["live_apply_allowed"] is False
    assert result["blockers"] == ["shadow_first_default", "evidence_not_green"]
    assert status_path.exists()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "NO_OP"
    assert payload["action_taken"] == "NO_OP"
    assert payload["live_apply_allowed"] is False


def test_handoff_invokes_apply_when_live_apply_allowed(tmp_path: Path, monkeypatch) -> None:
    plan_path = tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest.json"
    config_path = tmp_path / "config" / "barbell.yml"
    output_path = tmp_path / "logs" / "automation" / "nav_allocation_latest.json"
    status_path = tmp_path / "logs" / "automation" / "nav_rebalance_handoff_latest.json"
    staged_config_path = tmp_path / "config" / "barbell.staged.yml"

    _write_json(
        plan_path,
        {
            "rollout": {
                "mode": "live",
                "live_apply_allowed": True,
                "gate_lift_candidate": True,
                "gate_lift_ready": True,
                "live_apply_blockers": [],
            },
            "summary": {"healthy": ["NVDA"], "weak": ["AAPL", "GS"]},
        },
    )
    _write_yaml(
        config_path,
        """
barbell:
  safe_bucket:
    symbols: ["CASH"]
  core_bucket:
    symbols: ["AAPL"]
  speculative_bucket:
    symbols: ["NVDA"]
""".strip(),
    )

    def _fake_run(cmd, capture_output, text):
        assert cmd[0] == str(sys.executable)
        assert cmd[1].endswith("scripts\\apply_nav_reallocation.py") or cmd[1].endswith("scripts/apply_nav_reallocation.py")
        assert "--plan-path" in cmd
        assert "--config-path" in cmd
        assert "--output" in cmd
        assert "--staged-config" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="apply ok\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    result = mod.run_nav_rebalance_handoff(
        plan_path=plan_path,
        config_path=config_path,
        output_path=output_path,
        staged_config_path=staged_config_path,
        status_path=status_path,
    )

    assert result["status"] == "APPLIED"
    assert result["action_taken"] == "APPLY"
    assert result["exit_code"] == 0
    assert result["apply_rc"] == 0
    assert "apply_nav_reallocation.py" in " ".join(result["apply_command"])
    assert "apply ok" in result["apply_stdout_tail"]
    assert status_path.exists()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "APPLIED"
    assert payload["action_taken"] == "APPLY"
    assert payload["apply_rc"] == 0


def test_direct_cli_invocation_does_not_raise_module_not_found_error() -> None:
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_nav_rebalance_handoff.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        env={k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"},
    )
    assert result.returncode == 0, (
        f"Direct CLI invocation failed (exit={result.returncode}).\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "ModuleNotFoundError" not in result.stderr, (
        f"sys.path bootstrap missing — first-party imports failed:\n{result.stderr[:500]}"
    )
    assert "Usage:" in result.stdout, f"Expected Click help output; got: {result.stdout[:200]}"
