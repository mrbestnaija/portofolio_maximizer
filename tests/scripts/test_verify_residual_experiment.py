from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_emits_summary_payload_even_without_audits(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs" / "forecast_audits").mkdir(parents=True, exist_ok=True)

    from scripts.verify_residual_experiment import main

    rc = main(["--audit-dir", "logs/forecast_audits", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload["summary"]["summary_path"].replace("\\", "/") == (
        "visualizations/performance/residual_experiment_summary.json"
    )
    assert payload["summary"]["summary_exists"] is False


def test_verify_reports_active_and_canonical_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _write_json(
        tmp_path / "logs" / "forecast_audits" / "forecast_audit_20260308_000001.json",
        {
            "artifacts": {
                "residual_experiment": {
                    "residual_status": "active",
                    "residual_active": True,
                }
            }
        },
    )
    _write_json(
        tmp_path / "visualizations" / "performance" / "residual_experiment_summary.json",
        {
            "status": "PASS",
            "reason_code": "RESIDUAL_EXPERIMENT_AVAILABLE",
            "n_windows_with_residual_metrics": 2,
        },
    )

    from scripts.verify_residual_experiment import main

    rc = main(["--audit-dir", "logs/forecast_audits", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["active"] is True
    assert payload["summary"]["status"] == "PASS"
    assert payload["summary"]["n_windows_with_residual_metrics"] == 2
