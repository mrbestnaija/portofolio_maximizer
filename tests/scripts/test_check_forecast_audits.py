from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest


def _write_audit(
    path: Path,
    *,
    start: str,
    end: str,
    length: int,
    horizon: int,
    weights: dict,
    eval_metrics: dict,
) -> None:
    payload = {
        "dataset": {
            "start": start,
            "end": end,
            "length": length,
            "forecast_horizon": horizon,
        },
        "artifacts": {
            "ensemble_weights": weights,
            "evaluation_metrics": eval_metrics,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_forecast_audits_dedupes_by_dataset_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two audits with the same dataset window; only the most recent should count.
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        start="2024-01-01",
        end="2024-06-01",
        length=180,
        horizon=30,
    )

    older = audit_dir / "forecast_audit_older.json"
    newer = audit_dir / "forecast_audit_newer.json"

    _write_audit(
        older,
        **common,
        weights={"samossa": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 4.0},
            "samossa": {"rmse": 6.0},
            "ensemble": {"rmse": 6.0},
        },
    )
    # Ensure distinct mtimes.
    time.sleep(0.02)
    _write_audit(
        newer,
        **common,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 4.0},
            "samossa": {"rmse": 6.0},
            "ensemble": {"rmse": 4.0},
        },
    )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    holding_period_audits: 1",
                "    disable_ensemble_if_no_lift: false",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    # Run the CLI; it should pass (no violations) and not double-count.
    import scripts.check_forecast_audits as mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_forecast_audits.py",
            "--audit-dir",
            str(audit_dir),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0


def test_check_audit_file_uses_requested_baseline(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    _write_audit(
        audit,
        start="2024-01-01",
        end="2024-01-31",
        length=200,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 5.0},
            "samossa": {"rmse": 2.0},
            "ensemble": {"rmse": 3.0},
        },
    )

    import scripts.check_forecast_audits as mod

    res = mod.check_audit_file(audit, tolerance=0.1, baseline_model="SAMOSSA")
    assert res is not None
    assert res.baseline_model == "SAMOSSA"
    assert res.rmse_ratio == pytest.approx(1.5)


def test_check_forecast_audits_no_lift_gate_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Three effective audits, none beating BEST_SINGLE baseline.
    for i in range(3):
        _write_audit(
            audit_dir / f"forecast_audit_{i}.json",
            start=f"2024-01-{i+1:02d}",
            end=f"2024-02-{i+1:02d}",
            length=200,
            horizon=30,
            weights={"samossa": 1.0},
            eval_metrics={
                "sarimax": {"rmse": 5.0},
                "samossa": {"rmse": 4.0},
                "ensemble": {"rmse": 4.0},  # equals baseline => no lift when min_lift_rmse_ratio > 0
            },
        )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    holding_period_audits: 3",
                "    disable_ensemble_if_no_lift: true",
                "    min_lift_rmse_ratio: 0.01",
                "    min_lift_fraction: 0.10",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    import scripts.check_forecast_audits as mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_forecast_audits.py",
            "--audit-dir",
            str(audit_dir),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code != 0


def test_check_forecast_audits_fail_during_holding_period(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Two effective audits, both violating the tolerance. With strict warmup,
    # the gate should fail even though holding_period_audits is not met yet.
    for i in range(2):
        _write_audit(
            audit_dir / f"forecast_audit_{i}.json",
            start=f"2024-03-{i+1:02d}",
            end=f"2024-04-{i+1:02d}",
            length=200,
            horizon=30,
            weights={"samossa": 1.0},
            eval_metrics={
                "sarimax": {"rmse": 10.0},
                "samossa": {"rmse": 10.0},
                "ensemble": {"rmse": 15.0},  # 50% worse than baseline => violation
            },
        )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    holding_period_audits: 5",
                "    fail_on_violation_during_holding_period: true",
                "    disable_ensemble_if_no_lift: false",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    import scripts.check_forecast_audits as mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_forecast_audits.py",
            "--audit-dir",
            str(audit_dir),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code != 0
