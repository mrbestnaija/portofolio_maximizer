from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import scripts.audit_gate_defaults as defaults


def test_forecast_gate_max_files_default_is_shared_contract() -> None:
    assert defaults.FORECAST_AUDIT_MAX_FILES_DEFAULT == 500

    check_src = Path("scripts/check_forecast_audits.py").read_text(encoding="utf-8")
    prod_src = Path("scripts/production_audit_gate.py").read_text(encoding="utf-8")
    assert "FORECAST_AUDIT_MAX_FILES_DEFAULT" in check_src
    assert "FORECAST_AUDIT_MAX_FILES_DEFAULT" in prod_src


def test_overnight_final_health_failures_increment_error_counter() -> None:
    source = Path("scripts/run_overnight_refresh.py").read_text(encoding="utf-8")
    checks = [
        "check_quant_validation_health.py",
        "quant_validation_headroom.py",
        "production_audit_gate.py",
    ]
    for script_name in checks:
        pattern = (
            rf'rc = py\("scripts/{re.escape(script_name)}".*?\n'
            rf'\s*if rc != 0:\n'
            rf'\s*errors \+= 1'
        )
        assert re.search(pattern, source, re.DOTALL), (
            f"Final health command '{script_name}' does not increment errors on non-zero exit."
        )


def test_platt_contract_audit_runs_without_manual_pythonpath() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/platt_contract_audit.py", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    merged = f"{proc.stdout}\n{proc.stderr}"
    assert "No module named 'models'" not in merged
    assert proc.returncode in {0, 1}


def test_repo_has_no_shadow_duplicate_runtime_files() -> None:
    assert not Path("Dockerfile (1)").exists()
    assert not Path("execution/lob_simulator (1).py").exists()


def test_run_all_gates_wires_institutional_gate() -> None:
    source = Path("scripts/run_all_gates.py").read_text(encoding="utf-8")
    assert "institutional_unattended_gate.py" in source
    assert "--skip-institutional-gate" in source


def test_ci_workflow_runs_institutional_gate() -> None:
    source = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Institutional unattended hardening gate" in source
    assert "python scripts/institutional_unattended_gate.py --json" in source
