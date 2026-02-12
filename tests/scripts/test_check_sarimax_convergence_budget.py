from __future__ import annotations

import json
from pathlib import Path

from scripts.check_sarimax_convergence_budget import (
    _load_total_runs,
    _parse_event_counts,
)


def test_parse_event_counts_uses_highest_occurrence(tmp_path: Path) -> None:
    log_path = tmp_path / "warning_events.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-02-12 - WARNING - [SARIMAXForecaster.convergence_budget] event=primary_nonconverged occurrence=1",
                "2026-02-12 - WARNING - [SARIMAXForecaster.convergence_budget] event=primary_nonconverged occurrence=2",
                "2026-02-12 - WARNING - [SARIMAXForecaster.convergence_budget] event=fallback_converged occurrence=1",
            ]
        ),
        encoding="utf-8",
    )

    counts = _parse_event_counts(log_path)
    assert counts["primary_nonconverged"] == 2
    assert counts["fallback_converged"] == 1
    assert counts["fallback_nonconverged"] == 0


def test_load_total_runs_from_suite_report(tmp_path: Path) -> None:
    report_path = tmp_path / "suite.json"
    report_path.write_text(
        json.dumps(
            {
                "meta": {
                    "total_runs_per_variant": 18,
                    "variants": ["prod_like_conf_on", "sarimax_augmented_conf_on"],
                }
            }
        ),
        encoding="utf-8",
    )

    assert _load_total_runs(report_path) == 36
