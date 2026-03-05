from __future__ import annotations

import hashlib
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
    ticker: str | None = None,
    regime: str | None = None,
) -> None:
    payload = {
        "dataset": {
            "start": start,
            "end": end,
            "length": length,
            "forecast_horizon": horizon,
            "ticker": ticker,
            "detected_regime": regime,
        },
        "artifacts": {
            "ensemble_weights": weights,
            "evaluation_metrics": eval_metrics,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_manifest(audit_dir: Path, audit_paths: list[Path]) -> Path:
    manifest = audit_dir / "forecast_audit_manifest.jsonl"
    lines = []
    for path in audit_paths:
        lines.append(
            json.dumps(
                {
                    "file": path.name,
                    "sha256": _sha256(path),
                }
            )
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


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


def test_check_forecast_audits_reports_parse_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    valid = audit_dir / "forecast_audit_valid.json"
    invalid = audit_dir / "forecast_audit_invalid.json"
    _write_audit(
        valid,
        start="2024-01-01",
        end="2024-01-31",
        length=180,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
    )
    invalid.write_text("{not-json", encoding="utf-8")

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

    output = capsys.readouterr().out
    assert "parseable=1" in output
    assert "parse_errors=1" in output


def test_check_forecast_audits_emits_window_counts_and_diversity_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_1.json",
        start="2024-01-01",
        end="2024-01-05",
        length=180,
        horizon=30,
        ticker="GOOG",
        regime="HIGH_VOL_TRENDING",
        weights={"samossa": 1.0},
        eval_metrics={
            "samossa": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
    )
    _write_audit(
        audit_dir / "forecast_audit_2.json",
        start="2024-01-08",
        end="2024-01-10",
        length=185,
        horizon=30,
        ticker="MSFT",
        regime="LOW_VOL",
        weights={"garch": 1.0},
        eval_metrics={
            "garch": {"rmse": 3.0},
            "ensemble": {"rmse": 3.0},
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

    import scripts.check_forecast_audits as mod

    monkeypatch.chdir(tmp_path)
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

    output = capsys.readouterr().out
    assert "RMSE coverage  : raw=2 parseable=2 deduped=2 processed=2 usable=2" in output
    assert "Outcome cov.   : outcomes_loaded=0 join_attempted=0 eligible=0 matched=0" in output
    assert "Diversity      : regimes=2 healthy_tickers=2 trading_days=2" in output

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["window_counts"] == {
        "n_raw_windows": 2,
        "n_parseable_windows": 2,
        "n_deduped_windows": 2,
        "n_rmse_windows_processed": 2,
        "n_rmse_windows_usable": 2,
        "n_outcome_windows_eligible": 0,
        "n_outcome_windows_matched": 0,
    }
    assert summary["telemetry_contract"]["schema_version"] == 2
    assert summary["telemetry_contract"]["outcomes_loaded"] is False
    assert "cache_status" in summary
    assert summary["window_diversity"] == {
        "regime_count": 2,
        "healthy_ticker_count": 2,
        "distinct_trading_days": 2,
    }


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


def test_check_audit_file_best_single_includes_garch(tmp_path: Path) -> None:
    audit = tmp_path / "audit_garch_best.json"
    _write_audit(
        audit,
        start="2024-02-01",
        end="2024-02-29",
        length=200,
        horizon=30,
        weights={"garch": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 5.0},
            "samossa": {"rmse": 4.0},
            "garch": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
    )

    import scripts.check_forecast_audits as mod

    res = mod.check_audit_file(audit, tolerance=0.1, baseline_model="BEST_SINGLE")
    assert res is not None
    assert res.baseline_model == "GARCH"
    assert res.rmse_ratio == pytest.approx(1.0)


def test_check_audit_file_missing_ensemble_metrics_no_fallback(tmp_path: Path) -> None:
    audit = tmp_path / "audit_missing_ensemble.json"
    _write_audit(
        audit,
        start="2024-03-01",
        end="2024-03-31",
        length=180,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 3.0},
            "samossa": {"rmse": 2.0},
        },
    )

    import scripts.check_forecast_audits as mod

    res = mod.check_audit_file(audit, tolerance=0.1, baseline_model="BEST_SINGLE")
    assert res is not None
    assert res.ensemble_rmse is None
    assert res.baseline_rmse == pytest.approx(2.0)
    assert res.rmse_ratio is None
    assert res.ensemble_missing is True


def test_check_forecast_audits_manifest_fail_mode_blocks_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_path = audit_dir / "forecast_audit_0.json"
    _write_audit(
        audit_path,
        start="2024-04-01",
        end="2024-04-30",
        length=200,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 3.0},
            "ensemble": {"rmse": 3.0},
        },
    )
    _write_manifest(audit_dir, [audit_path])
    # Tamper after manifest generation.
    _write_audit(
        audit_path,
        start="2024-04-01",
        end="2024-04-30",
        length=200,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 3.0},
            "ensemble": {"rmse": 9.0},
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
                "    manifest_integrity_mode: fail",
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


def test_check_forecast_audits_manifest_fail_mode_accepts_valid_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_path = audit_dir / "forecast_audit_0.json"
    _write_audit(
        audit_path,
        start="2024-04-01",
        end="2024-04-30",
        length=200,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 3.0},
            "ensemble": {"rmse": 3.0},
        },
    )
    _write_manifest(audit_dir, [audit_path])

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
                "    manifest_integrity_mode: fail",
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
    assert excinfo.value.code == 0


def test_check_forecast_audits_fails_on_missing_ensemble_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_0.json",
        start="2024-05-01",
        end="2024-05-31",
        length=180,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 3.0},
            "samossa": {"rmse": 2.0},
            # ensemble intentionally missing
        },
    )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    max_missing_ensemble_rate: 0.0",
                "    holding_period_audits: 1",
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


def test_check_forecast_audits_missing_ensemble_rate_not_deferred_by_effective_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_0.json",
        start="2024-06-01",
        end="2024-06-30",
        length=160,
        horizon=30,
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 2.5},
            "samossa": {"rmse": 2.0},
            # ensemble intentionally missing
        },
    )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    max_missing_ensemble_rate: 0.0",
                "    min_effective_for_missing_rate_check: 20",
                "    holding_period_audits: 1",
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


def test_check_forecast_audits_no_lift_soft_mode_sets_disable_default_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

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
                "    holding_period_audits: 3",
                "    disable_ensemble_if_no_lift: false",
                "    min_lift_rmse_ratio: 0.01",
                "    min_lift_fraction: 0.10",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.25",
                "    promotion_margin: 0.0",
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
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Decision: DISABLE_DEFAULT" in out


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


def test_check_forecast_audits_require_holding_period_fails_when_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_0.json",
        start="2024-05-01",
        end="2024-06-01",
        length=200,
        horizon=30,
        weights={"samossa": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 10.0},
            "samossa": {"rmse": 10.0},
            "ensemble": {"rmse": 10.0},  # within tolerance => no violation
        },
    )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    holding_period_audits: 20",
                "    disable_ensemble_if_no_lift: false",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    import scripts.check_forecast_audits as mod

    # Default behavior: inconclusive warmup exits 0.
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

    # Strict behavior: require holding period exits non-zero when insufficient.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_forecast_audits.py",
            "--audit-dir",
            str(audit_dir),
            "--config-path",
            str(cfg),
            "--require-holding-period",
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo2:
        mod.main()
    assert excinfo2.value.code != 0


def test_check_forecast_audits_recent_window_gate_catches_fresh_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Older 3 audits: clean.
    for i in range(3):
        _write_audit(
            audit_dir / f"forecast_audit_old_{i}.json",
            start=f"2024-07-{i+1:02d}",
            end=f"2024-08-{i+1:02d}",
            length=220,
            horizon=30,
            weights={"samossa": 1.0},
            eval_metrics={
                "sarimax": {"rmse": 10.0},
                "samossa": {"rmse": 9.8},
                "ensemble": {"rmse": 9.9},
            },
        )

    # Most recent 3 audits: 2/3 violating.
    for i in range(3):
        rmse = 11.5 if i < 2 else 9.8
        _write_audit(
            audit_dir / f"forecast_audit_new_{i}.json",
            start=f"2024-09-{i+1:02d}",
            end=f"2024-10-{i+1:02d}",
            length=220,
            horizon=30,
            weights={"samossa": 1.0},
            eval_metrics={
                "sarimax": {"rmse": 10.0},
                "samossa": {"rmse": 10.0},
                "ensemble": {"rmse": rmse},
            },
        )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.40",
                "    recent_window_audits: 3",
                "    recent_window_max_violation_rate: 0.50",
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


def test_check_forecast_audits_min_forecast_horizon_filters_short_horizon_regressions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Short-horizon artifacts (typical of tests/research) are regressions.
    _write_audit(
        audit_dir / "forecast_audit_short_0.json",
        start="2024-11-01",
        end="2024-12-01",
        length=180,
        horizon=5,
        weights={"samossa": 1.0},
        eval_metrics={
            "samossa": {"rmse": 10.0},
            "ensemble": {"rmse": 15.0},
        },
    )
    _write_audit(
        audit_dir / "forecast_audit_short_1.json",
        start="2024-11-02",
        end="2024-12-02",
        length=180,
        horizon=10,
        weights={"samossa": 1.0},
        eval_metrics={
            "samossa": {"rmse": 10.0},
            "ensemble": {"rmse": 14.0},
        },
    )

    # Production-like horizon remains healthy.
    _write_audit(
        audit_dir / "forecast_audit_prod_0.json",
        start="2024-11-03",
        end="2024-12-03",
        length=220,
        horizon=30,
        weights={"samossa": 1.0},
        eval_metrics={
            "samossa": {"rmse": 10.0},
            "ensemble": {"rmse": 10.0},
        },
    )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    min_forecast_horizon: 20",
                "    holding_period_audits: 1",
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
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Horizon filter : forecast_horizon >= 20" in out


def test_check_forecast_audits_min_forecast_horizon_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Same structure as the filter test: two short-horizon regressions + one clean long horizon.
    for i, horizon, ens_rmse in (
        (0, 5, 15.0),
        (1, 10, 14.0),
        (2, 30, 10.0),
    ):
        _write_audit(
            audit_dir / f"forecast_audit_{i}.json",
            start=f"2024-11-{i+1:02d}",
            end=f"2024-12-{i+1:02d}",
            length=220,
            horizon=horizon,
            weights={"samossa": 1.0},
            eval_metrics={
                "samossa": {"rmse": 10.0},
                "ensemble": {"rmse": ens_rmse},
            },
        )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    min_forecast_horizon: 20",
                "    holding_period_audits: 1",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    import scripts.check_forecast_audits as mod

    # Override the config filter and include all horizons; should fail on violation rate.
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
            "--min-forecast-horizon",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code != 0
