from __future__ import annotations

from pathlib import Path

import scripts.source_contract_guard as mod


def test_source_contract_guard_scans_python_and_shell_and_exempts_helper(tmp_path, monkeypatch) -> None:
    scripts_root = tmp_path / "scripts"
    bash_root = tmp_path / "bash"
    helpers_root = tmp_path / "helpers"
    scripts_root.mkdir(parents=True)
    bash_root.mkdir(parents=True)
    helpers_root.mkdir(parents=True)

    (scripts_root / "robustness_thresholds.py").write_text(
        'FORBIDDEN = "linkage_min_matched"\n',
        encoding="utf-8",
    )
    (scripts_root / "bad_reader.py").write_text(
        'value = "metrics_summary.json"\nkey = "linkage_min_ratio"\n',
        encoding="utf-8",
    )
    (bash_root / "helper.sh").write_text(
        '#!/usr/bin/env bash\ncat metrics_summary.json\n',
        encoding="utf-8",
    )
    (bash_root / "entry.sh").write_text(
        '#!/usr/bin/env bash\nsource ../helpers/indirect.sh\n',
        encoding="utf-8",
    )
    (helpers_root / "indirect.sh").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" metrics_summary.json\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "PYTHON_SCAN_ROOTS", (scripts_root,))
    monkeypatch.setattr(mod, "SHELL_SCAN_ROOTS", (bash_root,))
    monkeypatch.setattr(mod, "PYTHON_ALLOWLIST", set())

    report = mod.run_source_contract_guard(tmp_path)

    assert report["ok"] is False
    kinds = {violation["kind"] for violation in report["violations"]}
    assert "metrics_summary_reference" in kinds
    assert "threshold_key_reference" in kinds
    assert any(v["file"] == "scripts/bad_reader.py" for v in report["violations"])
    assert any(v["file"] == "bash/helper.sh" for v in report["violations"])
    assert any(v["file"] == "helpers/indirect.sh" for v in report["violations"])
