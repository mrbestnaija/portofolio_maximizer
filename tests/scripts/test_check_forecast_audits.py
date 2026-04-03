from __future__ import annotations

import hashlib
import json
import sqlite3
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
    signal_context: dict | None = None,
    semantic_admission: dict | None = None,
    residual_diagnostics: dict | None = None,
    effective_default_model: str | None = None,
    ensemble_index_mismatch: bool | None = None,
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
    if residual_diagnostics is not None:
        payload["artifacts"]["residual_diagnostics"] = residual_diagnostics
    if effective_default_model is not None:
        payload["artifacts"]["effective_default_model"] = effective_default_model
    if ensemble_index_mismatch is not None:
        payload["artifacts"]["ensemble_index_mismatch"] = ensemble_index_mismatch
    if signal_context is not None:
        payload["signal_context"] = signal_context
    if semantic_admission is not None:
        payload["semantic_admission"] = semantic_admission
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


def _write_closed_trades_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_signal_id TEXT,
                is_close INTEGER DEFAULT 1,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0
            )
            """
        )
        for row in rows:
            conn.execute(
                """
                INSERT INTO trade_executions (ts_signal_id, is_close, is_diagnostic, is_synthetic)
                VALUES (?, ?, ?, ?)
                """,
                (
                    row.get("ts_signal_id"),
                    int(row.get("is_close", 1)),
                    int(row.get("is_diagnostic", 0)),
                    int(row.get("is_synthetic", 0)),
                ),
            )
        conn.execute(
            """
            CREATE VIEW production_closed_trades AS
            SELECT *
            FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_diagnostic, 0) = 0
              AND COALESCE(is_synthetic, 0) = 0
            """
        )
        conn.commit()
    finally:
        conn.close()


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


def test_check_forecast_audits_include_research_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    production_dir = tmp_path / "logs" / "forecast_audits" / "production"
    research_dir = tmp_path / "logs" / "forecast_audits" / "research"
    production_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        production_dir / "forecast_audit_prod.json",
        start="2024-01-01",
        end="2024-01-02",
        length=120,
        horizon=1,
        ticker="AAPL",
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
    )
    _write_audit(
        research_dir / "forecast_audit_research.json",
        start="2024-01-03",
        end="2024-01-04",
        length=120,
        horizon=1,
        ticker="MSFT",
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
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
            str(production_dir),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0
    summary_prod = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary_prod["window_counts"]["n_raw_windows"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_forecast_audits.py",
            "--audit-dir",
            str(production_dir),
            "--include-research",
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0
    summary_all = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary_all["window_counts"]["n_raw_windows"] == 2


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
    assert (
        "Outcome join   : outcomes_loaded=0 join_attempted=0 accepted=2 accepted_noneligible=0 "
        "eligible=2 quarantined=0 due_eligible=0 matched=0 missing=0 ambiguous=0 "
        "not_due=0 invalid_context=0 not_yet_eligible=0 duplicate_conflicts=0 "
        "contract_drift=0 cohort_drift=0 no_signal_id=0 non_trade_context=0 missing_exec_meta=2"
    ) in output
    assert "Diversity      : regimes=2 healthy_tickers=2 trading_days=2" in output

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["window_counts"]["n_raw_windows"] == 2
    assert summary["window_counts"]["n_parseable_windows"] == 2
    assert summary["window_counts"]["n_deduped_windows"] == 2
    assert summary["window_counts"]["n_outcome_deduped_windows"] == 2
    assert summary["window_counts"]["n_rmse_windows_processed"] == 2
    assert summary["window_counts"]["n_rmse_windows_usable"] == 2
    assert summary["window_counts"]["n_outcome_windows_eligible"] == 0
    assert summary["window_counts"]["n_outcome_windows_matched"] == 0
    assert summary["window_counts"]["n_outcome_windows_missing"] == 0
    assert summary["window_counts"]["n_outcome_windows_ambiguous"] == 0
    assert summary["window_counts"]["n_outcome_windows_not_due"] == 0
    assert summary["window_counts"]["n_outcome_windows_not_yet_eligible"] == 0
    assert summary["window_counts"]["n_outcome_windows_invalid_context"] == 0
    assert summary["window_counts"]["n_outcome_windows_outcomes_not_loaded"] == 0
    assert summary["window_counts"]["n_outcome_windows_no_signal_id"] == 0
    assert summary["window_counts"]["n_outcome_windows_non_trade_context"] == 0
    assert summary["window_counts"]["n_outcome_windows_missing_execution_metadata"] == 0
    assert summary["window_counts"]["n_accepted_records"] == 2
    assert summary["window_counts"]["n_accepted_noneligible_records"] == 0
    assert summary["window_counts"]["n_eligible_records"] == 2
    assert summary["window_counts"]["n_quarantined_records"] == 0
    assert summary["window_counts"]["n_duplicate_conflicts"] == 0
    assert summary["window_counts"]["n_admission_missing_execution_metadata_records"] == 2
    assert summary["window_counts"]["n_readiness_denominator_included"] == 0
    assert summary["window_counts"]["n_linkage_denominator_included"] == 0
    assert summary["measurement_contract_version"] == 1
    assert summary["baseline_model"] == "BEST_SINGLE"
    assert summary["lift_threshold_rmse_ratio"] == pytest.approx(1.0)
    assert summary["admission_summary"]["missing_execution_metadata_records"] == 2
    assert summary["telemetry_contract"]["schema_version"] == 3
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


def test_check_forecast_audits_fails_on_residual_diag_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    for idx, white_noise in enumerate([False, False, True], start=1):
        _write_audit(
            audit_dir / f"forecast_audit_{idx}.json",
            start=f"2024-01-0{idx}",
            end=f"2024-01-1{idx}",
            length=180,
            horizon=30,
            ticker="AAPL",
            weights={"samossa": 1.0},
            eval_metrics={
                "samossa": {"rmse": 2.0},
                "ensemble": {"rmse": 2.0},
            },
            effective_default_model="SAMOSSA",
            residual_diagnostics={
                "samossa": {
                    "white_noise": white_noise,
                    "n": 30,
                    "lb_pvalue": 0.50 if white_noise else 0.001,
                    "jb_pvalue": 0.50 if white_noise else 0.001,
                }
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
                "    max_violation_rate: 1.0",
                "    max_non_white_noise_rate: 0.50",
                "    min_residual_diagnostics_n: 10",
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
    assert excinfo.value.code == 1


def test_check_forecast_audits_ignores_residual_diag_below_min_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_small_n.json",
        start="2024-01-01",
        end="2024-01-31",
        length=180,
        horizon=30,
        ticker="MSFT",
        weights={"samossa": 1.0},
        eval_metrics={
            "samossa": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
        effective_default_model="SAMOSSA",
        residual_diagnostics={
            "samossa": {
                "white_noise": False,
                "n": 5,
                "lb_pvalue": 0.001,
                "jb_pvalue": 0.001,
            }
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
                "    max_violation_rate: 1.0",
                "    max_non_white_noise_rate: 0.0",
                "    min_residual_diagnostics_n: 10",
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


def test_check_forecast_audits_fails_on_missing_residual_diag_after_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_missing_resid.json",
        start="2024-01-01",
        end="2024-01-31",
        length=180,
        horizon=30,
        ticker="NVDA",
        weights={"samossa": 1.0},
        eval_metrics={
            "samossa": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
        effective_default_model="SAMOSSA",
    )

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    min_effective_audits: 1",
                "    holding_period_audits: 1",
                "    disable_ensemble_if_no_lift: false",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 1.0",
                "    fail_on_missing_residual_diagnostics: true",
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
    assert excinfo.value.code == 1


def test_check_forecast_audits_fails_on_index_mismatch_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_mismatch.json",
        start="2024-01-01",
        end="2024-01-31",
        length=180,
        horizon=30,
        ticker="AMD",
        weights={"samossa": 1.0},
        eval_metrics={
            "samossa": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
        effective_default_model="SAMOSSA",
        ensemble_index_mismatch=True,
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
                "    max_violation_rate: 1.0",
                "    max_index_mismatch_rate: 0.0",
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
    assert excinfo.value.code == 1


def test_outcome_join_happy_path_ts_signal_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts_signal_id = "legacy_2026-02-13_91"

    _write_audit(
        audit_dir / "forecast_audit_join_ok.json",
        start="2024-01-01",
        end="2024-01-05",
        length=180,
        horizon=1,
        ticker="GS",
        regime="LOW_VOL",
        signal_context={
            "ts_signal_id": ts_signal_id,
            "ticker": "GS",
            "run_id": "20260214_202138",
            "entry_ts": "2024-01-05T00:00:00Z",
            "forecast_horizon": 1,
            "signal_context_missing": False,
        },
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
    )
    db_path = tmp_path / "portfolio.db"
    _write_closed_trades_db(db_path, [{"ts_signal_id": ts_signal_id}])

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
            "--db",
            str(db_path),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["telemetry_contract"]["outcomes_loaded"] is True
    assert summary["telemetry_contract"]["outcome_join_attempted"] is True
    assert summary["window_counts"]["n_outcome_windows_eligible"] == 1
    assert summary["window_counts"]["n_outcome_windows_matched"] == 1
    assert summary["window_counts"]["n_outcome_windows_missing"] == 0
    assert summary["window_counts"]["n_outcome_windows_ambiguous"] == 0
    assert summary["effective_outcome_audits"] == 1
    assert summary["dataset_windows"][0]["outcome_status"] == "MATCHED"


def test_outcome_join_ambiguous_duplicate_ts_signal_id_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts_signal_id = "legacy_2026-02-13_DUP"

    _write_audit(
        audit_dir / "forecast_audit_join_ambiguous.json",
        start="2024-01-01",
        end="2024-01-05",
        length=180,
        horizon=1,
        ticker="GS",
        regime="LOW_VOL",
        signal_context={
            "ts_signal_id": ts_signal_id,
            "ticker": "GS",
            "run_id": "20260214_202138",
            "entry_ts": "2024-01-05T00:00:00Z",
            "forecast_horizon": 1,
            "signal_context_missing": False,
        },
        weights={"sarimax": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 2.0},
            "ensemble": {"rmse": 2.0},
        },
    )
    db_path = tmp_path / "portfolio.db"
    _write_closed_trades_db(
        db_path,
        [
            {"ts_signal_id": ts_signal_id},
            {"ts_signal_id": ts_signal_id},
        ],
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
            "--db",
            str(db_path),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["window_counts"]["n_outcome_windows_eligible"] == 0
    assert summary["window_counts"]["n_outcome_windows_matched"] == 0
    assert summary["window_counts"]["n_outcome_windows_ambiguous"] == 1
    assert summary["window_counts"]["n_outcome_windows_invalid_context"] == 1
    assert summary["effective_outcome_audits"] == 0
    assert summary["dataset_windows"][0]["outcome_status"] == "INVALID_CONTEXT"
    assert summary["dataset_windows"][0]["outcome_reason"] == "AMBIGUOUS_MATCH"


def test_outcome_join_uses_ticker_aware_dedupe_for_linkage_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    common_dataset = {
        "start": "2024-01-01",
        "end": "2024-01-05",
        "length": 180,
        "horizon": 1,
        "regime": "LOW_VOL",
    }
    _write_audit(
        audit_dir / "forecast_audit_aapl.json",
        ticker="AAPL",
        signal_context={
            "ts_signal_id": "ts_AAPL_1",
            "ticker": "AAPL",
            "run_id": "run_aapl",
            "entry_ts": "2024-01-05T00:00:00Z",
            "forecast_horizon": 1,
        },
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
        **common_dataset,
    )
    _write_audit(
        audit_dir / "forecast_audit_msft.json",
        ticker="MSFT",
        signal_context={
            "ts_signal_id": "ts_MSFT_1",
            "ticker": "MSFT",
            "run_id": "run_msft",
            "entry_ts": "2024-01-05T00:00:00Z",
            "forecast_horizon": 1,
        },
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
        **common_dataset,
    )

    db_path = tmp_path / "portfolio.db"
    _write_closed_trades_db(db_path, [{"ts_signal_id": "ts_AAPL_1"}])

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
            "--db",
            str(db_path),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["window_counts"]["n_deduped_windows"] == 1
    assert summary["window_counts"]["n_outcome_deduped_windows"] == 2
    assert summary["window_counts"]["n_outcome_windows_eligible"] == 2
    assert summary["window_counts"]["n_outcome_windows_matched"] == 1
    assert summary["window_counts"]["n_outcome_windows_missing"] == 1
    statuses_by_ticker = {
        str(entry.get("ticker")): entry.get("outcome_status")
        for entry in summary["dataset_windows"]
    }
    assert statuses_by_ticker["AAPL"] == "MATCHED"
    assert statuses_by_ticker["MSFT"] == "OUTCOME_MISSING"


def test_outcome_join_flags_causality_violation_as_invalid_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_causality_invalid.json",
        start="2026-01-01",
        end="2026-02-01",
        length=180,
        horizon=1,
        ticker="AAPL",
        regime="LOW_VOL",
        signal_context={
            "ts_signal_id": "ts_AAPL_2",
            "ticker": "AAPL",
            "run_id": "run_aapl_2",
            "entry_ts": "2026-03-04T00:00:00Z",
            "forecast_horizon": 1,
            "signal_context_missing": True,
        },
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
    )
    db_path = tmp_path / "portfolio.db"
    _write_closed_trades_db(db_path, [])

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
            "--db",
            str(db_path),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["window_counts"]["n_outcome_windows_invalid_context"] == 1
    assert summary["dataset_windows"][0]["outcome_status"] == "INVALID_CONTEXT"
    assert summary["dataset_windows"][0]["outcome_reason"] == "CAUSALITY_VIOLATION"


def test_outcome_join_flags_horizon_mismatch_as_invalid_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_horizon_mismatch.json",
        start="2026-01-01",
        end="2026-01-31",
        length=180,
        horizon=30,
        ticker="AAPL",
        regime="LOW_VOL",
        signal_context={
            "ts_signal_id": "ts_AAPL_3",
            "ticker": "AAPL",
            "run_id": "run_aapl_3",
            "entry_ts": "2026-01-31T00:00:00Z",
            "forecast_horizon": 1,
            "signal_context_missing": False,
        },
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
    )
    db_path = tmp_path / "portfolio.db"
    _write_closed_trades_db(db_path, [])

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
            "--db",
            str(db_path),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["window_counts"]["n_outcome_windows_invalid_context"] == 1
    assert summary["window_counts"]["n_outcome_windows_eligible"] == 0
    assert summary["dataset_windows"][0]["outcome_status"] == "INVALID_CONTEXT"
    assert summary["dataset_windows"][0]["outcome_reason"] == "HORIZON_MISMATCH"


def test_outcome_join_marks_future_window_as_not_due_not_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_not_due.json",
        start="2099-01-01",
        end="2099-01-31",
        length=180,
        horizon=7,
        ticker="AAPL",
        regime="LOW_VOL",
        signal_context={
            "ts_signal_id": "ts_AAPL_4",
            "ticker": "AAPL",
            "run_id": "run_aapl_4",
            "entry_ts": "2099-01-31T00:00:00Z",
            "forecast_horizon": 7,
            "signal_context_missing": False,
        },
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
    )
    db_path = tmp_path / "portfolio.db"
    _write_closed_trades_db(db_path, [])

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
            "--db",
            str(db_path),
            "--config-path",
            str(cfg),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 0

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    assert summary["window_counts"]["n_outcome_windows_not_due"] == 1
    assert summary["window_counts"]["n_outcome_windows_missing"] == 0
    assert summary["window_counts"]["n_readiness_denominator_included"] == 0
    assert summary["window_counts"]["n_readiness_excluded_not_due"] == 1
    assert summary["dataset_windows"][0]["outcome_status"] == "NOT_DUE"


def test_check_forecast_audits_preserves_producer_admission_fields_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_admission_contract.json",
        start="2024-01-01",
        end="2024-01-31",
        length=180,
        horizon=5,
        ticker="AAPL",
        regime="LOW_VOL",
        signal_context={
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "ts_signal_id": "ts_missing_meta",
            "ticker": "AAPL",
            "run_id": None,
            "entry_ts": None,
            "forecast_horizon": 5,
            "signal_context_missing": False,
        },
        semantic_admission={
            "admission_contract_version": 1,
            "accepted_for_audit_history": True,
            "admissible_for_readiness": False,
            "gate_eligible": False,
            "gate_bucket": "ACCEPTED_NONELIGIBLE",
            "reason_code": "MISSING_RUN_ID,MISSING_ENTRY_TS",
            "reason_codes": ["MISSING_RUN_ID", "MISSING_ENTRY_TS"],
            "production_labeled": True,
            "duplicate_conflict": False,
            "quarantined": False,
            "not_quarantined": True,
            "missing_execution_metadata": True,
            "missing_execution_metadata_fields": ["run_id", "entry_ts"],
        },
        weights={"sarimax": 1.0},
        eval_metrics={"sarimax": {"rmse": 2.0}, "ensemble": {"rmse": 2.0}},
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
    with pytest.raises(SystemExit):
        mod.main()

    summary = json.loads((tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text())
    entry = summary["dataset_windows"][0]
    assert entry["semantic_admission_source"] == "producer"
    assert entry["semantic_admission_preserved"] is True
    assert entry["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
    assert entry["admission_reason_code"] == "MISSING_RUN_ID,MISSING_ENTRY_TS"
    assert entry["admission_reason_codes"] == ["MISSING_RUN_ID", "MISSING_ENTRY_TS"]
    assert entry["missing_execution_metadata"] is True
    assert entry["missing_execution_metadata_fields"] == ["run_id", "entry_ts"]
    assert summary["admission_summary"]["accepted_noneligible_records"] == 1
    assert summary["admission_summary"]["missing_execution_metadata_records"] == 1
    assert summary["admission_summary"]["source_counts"]["producer"] == 1


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


def test_check_forecast_audits_failure_writes_latest_summary_with_generated_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_0.json",
        start="2024-01-01",
        end="2024-02-01",
        length=220,
        horizon=30,
        weights={"samossa": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 5.0},
            "samossa": {"rmse": 5.0},
            "ensemble": {"rmse": 8.0},
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
    assert excinfo.value.code != 0

    summary_path = tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["exit_code"] == 1
    assert isinstance(summary.get("generated_utc"), str) and summary["generated_utc"]
    assert summary["audit_dir"] == str(audit_dir)
    assert summary["max_files"] == 50
    assert summary["scope"]["include_research"] is False


def test_check_forecast_audits_failure_summary_preserves_outcome_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    ts_signal_id = "ts_AAPL_20240101T000000Z_dead_0001"
    _write_audit(
        audit_dir / "forecast_audit_0.json",
        start="2023-11-01",
        end="2024-01-01",
        length=220,
        horizon=30,
        weights={"samossa": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 5.0},
            "samossa": {"rmse": 5.0},
            "ensemble": {"rmse": 8.0},
        },
        ticker="AAPL",
        signal_context={
            "context_type": "TRADE",
            "ts_signal_id": ts_signal_id,
            "run_id": "20240101_000000",
            "entry_ts": "2024-01-01T00:00:00+00:00",
            "forecast_horizon": 30,
        },
    )

    db_path = tmp_path / "portfolio_maximizer.db"
    _write_closed_trades_db(
        db_path,
        [
            {
                "ts_signal_id": ts_signal_id,
                "is_close": 1,
                "is_diagnostic": 0,
                "is_synthetic": 0,
            }
        ],
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
            "--db",
            str(db_path),
            "--max-files",
            "50",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code != 0

    summary_path = tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert len(summary["dataset_windows"]) == 1
    assert summary["dataset_windows"][0]["outcome_status"] == "MATCHED"
    assert bool(summary["dataset_windows"][0]["counts_toward_linkage_denominator"]) is True
    assert summary["window_counts"]["n_outcome_windows_matched"] == 1


def test_check_forecast_audits_failure_summary_retains_rmse_and_lift_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_audit(
        audit_dir / "forecast_audit_0.json",
        start="2024-01-01",
        end="2024-02-01",
        length=220,
        horizon=30,
        weights={"samossa": 1.0},
        eval_metrics={
            "sarimax": {"rmse": 5.0},
            "samossa": {"rmse": 5.0},
            "ensemble": {"rmse": 8.0},
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
    assert excinfo.value.code != 0

    summary = json.loads(
        (tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text(encoding="utf-8")
    )
    assert summary["effective_audits"] == 1
    assert summary["violation_count"] == 1
    assert summary["violation_rate"] == pytest.approx(1.0)
    assert summary["lift_fraction"] == pytest.approx(0.0)
    assert summary["measurement_contract_version"] == 1
    assert summary["baseline_model"] == "BEST_SINGLE"
    assert summary["lift_threshold_rmse_ratio"] == pytest.approx(1.0)
    assert summary["window_counts"]["n_rmse_windows_processed"] == 1
    assert summary["window_counts"]["n_rmse_windows_usable"] == 1


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


def test_effective_default_baseline_uses_ensemble_selection_primary_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_extract_metrics resolves EFFECTIVE_DEFAULT via ensemble_selection.primary_model.

    Audit has garch RMSE=2.0 (oracle best single) and samossa RMSE=3.0.
    ensemble_selection.primary_model=samossa → baseline = samossa (RMSE 3.0).
    ensemble RMSE=3.3 → ratio = 3.3/3.0 = 1.1 → VIOLATION (threshold 1.05).
    Under BEST_SINGLE the ratio would be 3.3/2.0 = 1.65, also a violation, but
    the resolved_baseline logged should be "SAMOSSA", not "GARCH".
    """
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": {
            "start": "2024-01-01",
            "end": "2024-06-01",
            "length": 150,
            "forecast_horizon": 30,
            "ticker": "AAPL",
        },
        "artifacts": {
            "ensemble_weights": {"garch": 0.5, "samossa": 0.5},
            "evaluation_metrics": {
                "garch":   {"rmse": 2.0, "directional_accuracy": 0.55},
                "samossa": {"rmse": 3.0, "directional_accuracy": 0.50},
                "ensemble": {"rmse": 3.3, "directional_accuracy": 0.52},
            },
            "ensemble_selection": {"primary_model": "samossa"},
        },
    }
    (audit_dir / "forecast_audit_20240101.json").write_text(json.dumps(payload), encoding="utf-8")

    cfg = tmp_path / "forecaster_monitoring.yml"
    cfg.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: EFFECTIVE_DEFAULT",
                "    holding_period_audits: 1",
                "    disable_ensemble_if_no_lift: false",
                "    max_rmse_ratio_vs_baseline: 1.05",
                "    max_violation_rate: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    # Directly test _extract_metrics to verify resolved_baseline is "SAMOSSA".
    import scripts.check_forecast_audits as mod
    ensemble_m, baseline_m, resolved = mod._extract_metrics(payload, baseline_model="EFFECTIVE_DEFAULT")
    assert resolved == "SAMOSSA", f"Expected SAMOSSA, got {resolved}"
    assert baseline_m is not None
    assert abs(baseline_m["rmse"] - 3.0) < 1e-9

    # Also verify that BEST_SINGLE resolves to "GARCH" (oracle min).
    _, baseline_bs, resolved_bs = mod._extract_metrics(payload, baseline_model="BEST_SINGLE")
    assert resolved_bs == "GARCH"
    assert abs(baseline_bs["rmse"] - 2.0) < 1e-9
