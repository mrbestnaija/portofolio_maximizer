from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "repo_python.ps1"


@pytest.mark.skipif(os.name != "nt", reason="PowerShell wrapper is Windows-specific")
def test_repo_python_wrapper_uses_simpletrader_env() -> None:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WRAPPER),
            "-c",
            "import sys; print(sys.executable)",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    executable = completed.stdout.strip().splitlines()[-1]
    assert executable.lower().endswith(r"simpletrader_env\scripts\python.exe")
