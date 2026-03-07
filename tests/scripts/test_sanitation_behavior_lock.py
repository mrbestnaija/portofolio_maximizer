from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, raw in value.items():
            if key in {
                "timestamp_utc",
                "generated_utc",
                "mtime_utc",
                "ctime_utc",
                "_mtime",
            }:
                continue
            out[key] = _normalize_payload(raw)
        return out
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    return value


def _write_audit(path: Path) -> None:
    payload = {
        "dataset": {
            "start": "2025-01-01T00:00:00+00:00",
            "end": "2025-01-02T00:00:00+00:00",
            "length": 128,
            "forecast_horizon": 1,
            "ticker": "AAPL",
            "detected_regime": "MODERATE_TRENDING",
        },
        "signal_context": {
            "context_type": "TRADE",
            "ts_signal_id": "ts_AAPL_TEST_0001",
            "entry_ts": "2025-01-01T00:00:00+00:00",
            "forecast_horizon": 1,
            "execution_metadata": {"route": "ts"},
        },
        "artifacts": {
            "ensemble_weights": {"samossa": 1.0},
            "evaluation_metrics": {
                "samossa": {"rmse": 1.0, "directional_accuracy": 0.6, "smape": 1.1},
                "garch": {"rmse": 1.1, "directional_accuracy": 0.55, "smape": 1.2},
                "ensemble": {"rmse": 1.0, "directional_accuracy": 0.6},
            },
        },
        "runs": [{"model": "regime", "metadata": {"regime": "MODERATE_TRENDING"}}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_monitor_cfg(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.35",
                "    holding_period_audits: 1",
                "    fail_on_violation_during_holding_period: false",
                "    disable_ensemble_if_no_lift: false",
                "    min_lift_fraction: 0.0",
                "    manifest_integrity_mode: off",
            ]
        ),
        encoding="utf-8",
    )


def _write_proof_cfg(path: Path) -> None:
    path.write_text("profitability_proof_requirements: {}\n", encoding="utf-8")


def _write_closed_trades_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_signal_id TEXT,
                entry_trade_id INTEGER,
                trade_date TEXT,
                bar_timestamp TEXT,
                is_close INTEGER DEFAULT 1,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0,
                realized_pnl REAL,
                exit_reason TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions
            (ts_signal_id, entry_trade_id, trade_date, bar_timestamp, is_close, is_diagnostic, is_synthetic, realized_pnl, exit_reason)
            VALUES
            ('ts_AAPL_TEST_0001', NULL, '2025-01-02', '2025-01-02T00:00:00+00:00', 1, 0, 0, 10.0, 'signal_exit')
            """
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


def test_behavior_lock_check_forecast_audits_json_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.check_forecast_audits as mod

    audit_dir = tmp_path / "logs" / "forecast_audits"
    cfg = tmp_path / "forecaster_monitoring.yml"
    db = tmp_path / "portfolio.db"
    _write_audit(audit_dir / "forecast_audit_20250102_000000.json")
    _write_monitor_cfg(cfg)
    _write_closed_trades_db(db)

    monkeypatch.chdir(tmp_path)
    argv = [
        "check_forecast_audits.py",
        "--audit-dir",
        str(audit_dir),
        "--config-path",
        str(cfg),
        "--db",
        str(db),
        "--max-files",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as first_exit:
        mod.main()
    assert first_exit.value.code == 0
    first_summary = json.loads(
        (tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text(
            encoding="utf-8"
        )
    )

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as second_exit:
        mod.main()
    assert second_exit.value.code == 0
    second_summary = json.loads(
        (tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert _normalize_payload(first_summary) == _normalize_payload(second_summary)


def test_behavior_lock_production_audit_gate_json_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    audit_dir = tmp_path / "logs" / "forecast_audits"
    db = tmp_path / "portfolio.db"
    monitor_cfg = tmp_path / "forecaster_monitoring.yml"
    proof_cfg = tmp_path / "profitability_proof_requirements.yml"
    output_json = tmp_path / "logs" / "audit_gate" / "production_gate_latest.json"
    _write_audit(audit_dir / "forecast_audit_20250102_000000.json")
    _write_monitor_cfg(monitor_cfg)
    _write_proof_cfg(proof_cfg)
    _write_closed_trades_db(db)

    def _fake_run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        del cwd
        joined = " ".join(cmd)
        if "check_forecast_audits.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="\n".join(
                    [
                        "Effective audits with RMSE: 1",
                        "Violations (ensemble worse than baseline beyond tolerance): 0",
                        "Violation rate: 0.00% (max allowed 35.00%)",
                        "Decision: KEEP (lift demonstrated during holding period)",
                    ]
                ),
                stderr="",
            )
        if "validate_profitability_proof.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps(
                    {
                        "is_proof_valid": True,
                        "is_profitable": True,
                        "metrics": {
                            "total_pnl": 10.0,
                            "profit_factor": 1.2,
                            "win_rate": 0.6,
                            "winning_trades": 3,
                            "losing_trades": 2,
                            "trading_days": 3,
                        },
                    }
                ),
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    monkeypatch.setattr(mod, "_run_command", _fake_run)
    monkeypatch.setattr(
        mod,
        "_run_reconcile_step",
        lambda **kwargs: {
            "requested": False,
            "status": "PASS",
            "status_reason": "not_requested",
            "remaining_unlinked_closes": 0,
            "remaining_unlinked_close_ids": [],
        },
    )

    argv = [
        "production_audit_gate.py",
        "--db",
        str(db),
        "--audit-dir",
        str(audit_dir),
        "--monitor-config",
        str(monitor_cfg),
        "--proof-requirements",
        str(proof_cfg),
        "--output-json",
        str(output_json),
        "--max-files",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    first_payload = json.loads(output_json.read_text(encoding="utf-8"))

    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    second_payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert _normalize_payload(first_payload) == _normalize_payload(second_payload)


def test_behavior_lock_capital_readiness_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.capital_readiness_check as mod

    monkeypatch.setattr(
        mod,
        "_check_r1_adversarial",
        lambda *_: (True, "", {"n_critical_high_confirmed": 0, "confirmed_ids": []}),
    )
    monkeypatch.setattr(
        mod,
        "_check_r2_gate_artifact",
        lambda: (True, "", {"gate_overall_passed": True, "gate_age_hours": 1.0}),
    )
    monkeypatch.setattr(
        mod,
        "_check_r3_trade_quality",
        lambda *_: (True, "", {"n_trades": 30, "win_rate": 0.6, "profit_factor": 1.6}),
    )
    monkeypatch.setattr(
        mod,
        "_check_r4_calibration",
        lambda *_: (True, "", {"calibration_tier": "db_local", "brier_score": 0.12}),
    )
    monkeypatch.setattr(
        mod,
        "_check_r5_lift_ci",
        lambda *_: ("", {"lift_ci_low": 0.01}),
    )
    monkeypatch.setattr(
        mod,
        "_check_r6_lifecycle_integrity",
        lambda *_: (
            True,
            "",
            {
                "close_before_entry_count": 0,
                "closed_missing_exit_reason_count": 0,
                "high_integrity_violation_count": 0,
            },
        ),
    )

    first = mod.run_capital_readiness(
        db_path=tmp_path / "x.db",
        audit_dir=tmp_path / "audits",
        jsonl_path=tmp_path / "quant_validation.jsonl",
    )
    second = mod.run_capital_readiness(
        db_path=tmp_path / "x.db",
        audit_dir=tmp_path / "audits",
        jsonl_path=tmp_path / "quant_validation.jsonl",
    )

    assert _normalize_payload(first) == _normalize_payload(second)


def test_behavior_lock_check_model_improvement_json_layer1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scripts.check_model_improvement as mod

    audit_dir = tmp_path / "logs" / "forecast_audits"
    _write_audit(audit_dir / "forecast_audit_20250102_000000.json")

    argv = [
        "check_model_improvement.py",
        "--layer",
        "1",
        "--audit-dir",
        str(audit_dir),
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc1 = mod.main()
    out1 = capsys.readouterr().out
    assert rc1 in (0, 1)

    monkeypatch.setattr(sys, "argv", argv)
    rc2 = mod.main()
    out2 = capsys.readouterr().out
    assert rc2 in (0, 1)

    payload1 = json.loads(out1)
    payload2 = json.loads(out2)
    assert _normalize_payload(payload1) == _normalize_payload(payload2)
