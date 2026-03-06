from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
import sys

import pytest


def test_extract_lift_output_metrics_inconclusive() -> None:
    import scripts.production_audit_gate as mod

    output = "\n".join(
        [
            "Effective audits with RMSE: 7",
            "Violations (ensemble worse than baseline beyond tolerance): 2",
            "Violation rate: 28.57% (max allowed 35.00%)",
            "RMSE gate inconclusive: effective_audits=7 < required_audits=20.",
        ]
    )
    metrics = mod._extract_lift_output_metrics(output)
    assert metrics["effective_audits"] == 7
    assert metrics["violation_count"] == 2
    assert metrics["violation_rate"] == pytest.approx(0.2857, abs=1e-6)
    assert metrics["max_violation_rate"] == pytest.approx(0.35, abs=1e-6)
    assert metrics["decision"] is None
    assert metrics["decision_reason"] is None


def test_extract_lift_output_metrics_decision_and_lift_fraction() -> None:
    import scripts.production_audit_gate as mod

    output = "\n".join(
        [
            "Ensemble lift fraction: 28.57% (required >= 25.00%)",
            "Decision: KEEP (lift demonstrated during holding period)",
        ]
    )
    metrics = mod._extract_lift_output_metrics(output)
    assert metrics["lift_fraction"] == pytest.approx(0.2857, abs=1e-6)
    assert metrics["min_lift_fraction"] == pytest.approx(0.25, abs=1e-6)
    assert metrics["decision"] == "KEEP"
    assert metrics["decision_reason"] == "lift demonstrated during holding period"


def test_summary_matches_invocation_max_files_match() -> None:
    import scripts.production_audit_gate as mod

    audit_dir = Path.cwd() / "logs" / "forecast_audits"
    summary = {"audit_dir": str(audit_dir), "max_files": 500}
    assert mod._summary_matches_invocation(summary, audit_dir=audit_dir, max_files=500)


def test_summary_matches_invocation_max_files_mismatch() -> None:
    import scripts.production_audit_gate as mod

    audit_dir = Path.cwd() / "logs" / "forecast_audits"
    summary = {"audit_dir": str(audit_dir), "max_files": 500}
    assert not mod._summary_matches_invocation(summary, audit_dir=audit_dir, max_files=50)


def _seed_trade_exec_table(db_path: Path, *, close_id: int, entry_trade_id: int | None) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            is_close INTEGER NOT NULL,
            entry_trade_id INTEGER,
            realized_pnl REAL
        )
        """
    )
    conn.execute(
        "INSERT INTO trade_executions (id, is_close, entry_trade_id, realized_pnl) VALUES (?, 1, ?, 10.0)",
        (int(close_id), entry_trade_id),
    )
    conn.commit()
    conn.close()


def test_run_reconcile_step_apply_fails_when_unlinked_remains_even_if_command_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    db_path = tmp_path / "portfolio.db"
    _seed_trade_exec_table(db_path, close_id=66, entry_trade_id=None)

    def _fake_run_command(cmd: list[str], cwd: Path):  # noqa: ANN001
        del cmd, cwd
        return subprocess.CompletedProcess(
            args=["python", "scripts/repair_unlinked_closes.py"],
            returncode=0,
            stdout="[WARNING] No repairs identified\nRemaining unlinked closes: 1\n",
            stderr="",
        )

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)

    res = mod._run_reconcile_step(
        python_bin="python",
        repo_root=Path.cwd(),
        db_path=db_path,
        close_ids=[66],
        apply=True,
    )
    assert res["exit_code"] == 0
    assert res["status"] == "FAIL"
    assert res["status_reason"] == "remaining_unlinked_after_apply"
    assert res["remaining_unlinked_closes"] == 1
    assert res["remaining_unlinked_close_ids"] == [66]


def test_run_reconcile_step_apply_passes_when_verified_zero_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    db_path = tmp_path / "portfolio.db"
    _seed_trade_exec_table(db_path, close_id=66, entry_trade_id=12345)

    def _fake_run_command(cmd: list[str], cwd: Path):  # noqa: ANN001
        del cmd, cwd
        return subprocess.CompletedProcess(
            args=["python", "scripts/repair_unlinked_closes.py"],
            returncode=0,
            stdout="[SUCCESS] Repairs applied\nRemaining unlinked closes: 0\n",
            stderr="",
        )

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)

    res = mod._run_reconcile_step(
        python_bin="python",
        repo_root=Path.cwd(),
        db_path=db_path,
        close_ids=[66],
        apply=True,
    )
    assert res["status"] == "PASS"
    assert res["status_reason"] == "verified_zero_unlinked_after_apply"
    assert res["remaining_unlinked_closes"] == 0
    assert res["remaining_unlinked_close_ids"] == []


def test_run_reconcile_step_dry_run_fails_when_unlinked_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    db_path = tmp_path / "portfolio.db"
    _seed_trade_exec_table(db_path, close_id=66, entry_trade_id=None)

    def _fake_run_command(cmd: list[str], cwd: Path):  # noqa: ANN001
        del cmd, cwd
        return subprocess.CompletedProcess(
            args=["python", "scripts/repair_unlinked_closes.py"],
            returncode=0,
            stdout="[DRY RUN] No changes applied\n",
            stderr="",
        )

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)

    res = mod._run_reconcile_step(
        python_bin="python",
        repo_root=Path.cwd(),
        db_path=db_path,
        close_ids=[66],
        apply=False,
    )
    assert res["status"] == "FAIL"
    assert res["status_reason"] == "remaining_unlinked_detected"
    assert res["remaining_unlinked_closes"] == 1


def test_main_fails_gate_when_reconcile_status_is_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scripts.production_audit_gate as mod

    output_json = tmp_path / "production_gate.json"
    monitor_cfg = tmp_path / "monitor.yml"
    monitor_cfg.write_text("forecaster_monitoring: {}\n", encoding="utf-8")
    proof_cfg = tmp_path / "proof.yml"
    proof_cfg.write_text("profitability_proof_requirements: {}\n", encoding="utf-8")
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "portfolio.db"
    sqlite3.connect(str(db_path)).close()

    def _fake_run_command(cmd: list[str], cwd: Path):  # noqa: ANN001
        del cwd
        joined = " ".join(cmd)
        if "check_forecast_audits.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="\n".join(
                    [
                        "Effective audits with RMSE: 20",
                        "Violations (ensemble worse than baseline beyond tolerance): 0",
                        "Violation rate: 0.00% (max allowed 35.00%)",
                        "Ensemble lift fraction: 30.00% (required >= 25.00%)",
                        "Decision: KEEP (lift demonstrated during holding period)",
                    ]
                ),
                stderr="",
            )
        if "validate_profitability_proof.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 100.0, "profit_factor": 1.5, "win_rate": 0.6, "winning_trades": 6, "losing_trades": 4, "trading_days": 10}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(
        mod,
        "_run_reconcile_step",
        lambda **kwargs: {
            "requested": True,
            "apply": True,
            "close_ids": [66],
            "exit_code": 0,
            "status": "FAIL",
            "status_reason": "remaining_unlinked_after_apply",
            "remaining_unlinked_closes": 1,
            "remaining_unlinked_close_ids": [66],
            "output_tail": "[WARNING] No repairs identified",
        },
    )
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setenv("PMX_NOTIFY_OPENCLAW", "0")
    monkeypatch.setenv("OPENCLAW_TARGETS", "")
    monkeypatch.setenv("OPENCLAW_TO", "")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_audit_gate.py",
            "--db",
            str(db_path),
            "--proof-requirements",
            str(proof_cfg),
            "--audit-dir",
            str(audit_dir),
            "--monitor-config",
            str(monitor_cfg),
            "--output-json",
            str(output_json),
            "--reconcile",
            "66",
            "--reconcile-apply",
        ],
    )

    rc = mod.main()
    assert rc == 1

    payload = mod._safe_load_json(output_json)
    assert payload is not None
    assert payload["reconciliation"]["status"] == "FAIL"
    assert payload["production_profitability_gate"]["pass"] is False
    assert payload["production_profitability_gate"]["reconcile_pass"] is False

    out = capsys.readouterr().out
    assert "Reconcile step : FAIL" in out


def test_unattended_profile_requires_profitable_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    output_json = tmp_path / "production_gate.json"
    monitor_cfg = tmp_path / "monitor.yml"
    monitor_cfg.write_text("forecaster_monitoring: {}\n", encoding="utf-8")
    proof_cfg = tmp_path / "proof.yml"
    proof_cfg.write_text("profitability_proof_requirements: {}\n", encoding="utf-8")
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "portfolio.db"
    sqlite3.connect(str(db_path)).close()

    def _fake_run_command(cmd: list[str], cwd: Path):  # noqa: ANN001
        del cwd
        joined = " ".join(cmd)
        if "check_forecast_audits.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="\n".join(
                    [
                        "Effective audits with RMSE: 21",
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
                stdout='{"is_proof_valid": true, "is_profitable": false, "metrics": {"total_pnl": -10.0, "profit_factor": 0.9, "win_rate": 0.4, "winning_trades": 4, "losing_trades": 6, "trading_days": 10}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(
        mod,
        "_compute_warmup_window",
        lambda **kwargs: {
            "max_warmup_days": 30,
            "first_audit_ts_utc": "2026-01-01T00:00:00+00:00",
            "allow_inconclusive_until_utc": "2026-01-31T00:00:00+00:00",
            "warmup_expired": False,
        },
    )
    monkeypatch.setenv("PMX_NOTIFY_OPENCLAW", "0")
    monkeypatch.setenv("OPENCLAW_TARGETS", "")
    monkeypatch.setenv("OPENCLAW_TO", "")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_audit_gate.py",
            "--db",
            str(db_path),
            "--proof-requirements",
            str(proof_cfg),
            "--audit-dir",
            str(audit_dir),
            "--monitor-config",
            str(monitor_cfg),
            "--output-json",
            str(output_json),
            "--unattended-profile",
        ],
    )

    rc = mod.main()
    assert rc == 1
    payload = mod._safe_load_json(output_json)
    assert payload is not None
    assert payload["proof_profitable_required"] is True
    assert payload["profitability_proof"]["is_proof_valid"] is True
    assert payload["profitability_proof"]["is_profitable"] is False
    assert payload["production_profitability_gate"]["status"] == "FAIL"


def test_unattended_profile_blocks_inconclusive_after_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    output_json = tmp_path / "production_gate.json"
    monitor_cfg = tmp_path / "monitor.yml"
    monitor_cfg.write_text("forecaster_monitoring: {}\n", encoding="utf-8")
    proof_cfg = tmp_path / "proof.yml"
    proof_cfg.write_text("profitability_proof_requirements: {}\n", encoding="utf-8")
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "portfolio.db"
    sqlite3.connect(str(db_path)).close()

    def _fake_run_command(cmd: list[str], cwd: Path):  # noqa: ANN001
        del cwd
        joined = " ".join(cmd)
        if "check_forecast_audits.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="\n".join(
                    [
                        "Effective audits with RMSE: 7",
                        "Violations (ensemble worse than baseline beyond tolerance): 0",
                        "Violation rate: 0.00% (max allowed 35.00%)",
                        "RMSE gate inconclusive: effective_audits=7 < required_audits=20.",
                    ]
                ),
                stderr="",
            )
        if "validate_profitability_proof.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 10.0, "profit_factor": 1.5, "win_rate": 0.6, "winning_trades": 6, "losing_trades": 4, "trading_days": 10}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(
        mod,
        "_compute_warmup_window",
        lambda **kwargs: {
            "max_warmup_days": 30,
            "first_audit_ts_utc": "2026-01-01T00:00:00+00:00",
            "allow_inconclusive_until_utc": "2026-01-31T00:00:00+00:00",
            "warmup_expired": True,
        },
    )
    monkeypatch.setenv("PMX_NOTIFY_OPENCLAW", "0")
    monkeypatch.setenv("OPENCLAW_TARGETS", "")
    monkeypatch.setenv("OPENCLAW_TO", "")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_audit_gate.py",
            "--db",
            str(db_path),
            "--proof-requirements",
            str(proof_cfg),
            "--audit-dir",
            str(audit_dir),
            "--monitor-config",
            str(monitor_cfg),
            "--output-json",
            str(output_json),
            "--unattended-profile",
        ],
    )

    rc = mod.main()
    assert rc == 1
    payload = mod._safe_load_json(output_json)
    assert payload is not None
    assert payload["lift_inconclusive_allowed"] is False
    assert payload["warmup_expired"] is True
    assert payload["lift_gate"]["inconclusive"] is True
    assert payload["lift_gate"]["pass"] is False
    assert payload["production_profitability_gate"]["gate_semantics_status"] == "INCONCLUSIVE_BLOCKED"
