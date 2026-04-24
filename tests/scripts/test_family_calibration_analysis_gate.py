from __future__ import annotations

import json
from pathlib import Path

from scripts import family_calibration_analysis_gate as mod


def test_analysis_gate_refuses_non_actionable_latest_row(tmp_path: Path) -> None:
    path = tmp_path / "family_calibration.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"schema_version": 1, "analysis_gate_passed": False, "analysis_gate_reasons": ["window_cycles_below_min"]}),
                json.dumps({"schema_version": 1, "analysis_gate_passed": False, "analysis_gate_reasons": ["regime_diversity_insufficient"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.evaluate_family_calibration_analysis_gate(path)

    assert report["analysis_gate_passed"] is False
    assert report["row_present"] is True
    assert report["analysis_gate_reasons"] == ["regime_diversity_insufficient"]
    assert mod.main(["--input", str(path), "--json"]) == 1


def test_analysis_gate_accepts_latest_actionable_row(tmp_path: Path) -> None:
    path = tmp_path / "family_calibration.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"schema_version": 1, "analysis_gate_passed": False, "analysis_gate_reasons": ["window_cycles_below_min"]}),
                json.dumps({"schema_version": 1, "analysis_gate_passed": True, "analysis_gate_reasons": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.evaluate_family_calibration_analysis_gate(path)

    assert report["analysis_gate_passed"] is True
    assert report["latest_schema_version"] == 1
    assert report["latest_window_cycles"] is None
    assert mod.main(["--input", str(path), "--json"]) == 0
