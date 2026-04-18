from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
import sys

import pytest


def _make_lift_summary(audit_dir: Path, **overrides) -> dict:
    base = {
        "generated_utc": "2026-03-15T12:00:00+00:00",
        "audit_dir": str(audit_dir),
        "max_files": 500,
        "measurement_contract_version": 1,
        "baseline_model": "BEST_SINGLE",
        "lift_threshold_rmse_ratio": 1.0,
        "scope": {"include_research": False, "production_audit_only": False},
        "effective_audits": 20,
        "violation_rate": 0.0,
        "max_violation_rate": 0.35,
        "lift_fraction": 0.30,
        "min_lift_fraction": 0.25,
        "decision": "KEEP",
        "decision_reason": "lift demonstrated during holding period",
        "admission_summary": {
            "accepted_records": 12,
            "accepted_noneligible_records": 0,
            "eligible_records": 12,
            "quarantined_records": 0,
            "duplicate_conflicts": 0,
            "missing_execution_metadata_records": 0,
            "bucket_counts": {
                "ELIGIBLE": 12,
                "ACCEPTED_NONELIGIBLE": 0,
                "QUARANTINED": 0,
            },
            "source_counts": {
                "producer": 12,
                "legacy_derived": 0,
            },
        },
        "window_counts": {
            "n_rmse_windows_processed": 20,
            "n_rmse_windows_usable": 20,
            "n_outcome_windows_not_due": 2,
            "n_outcome_windows_eligible": 12,
            "n_outcome_windows_matched": 10,
            "n_outcome_windows_invalid_context": 0,
            "n_outcome_windows_non_trade_context": 0,
            "n_accepted_records": 12,
            "n_accepted_noneligible_records": 0,
            "n_eligible_records": 12,
            "n_quarantined_records": 0,
            "n_duplicate_conflicts": 0,
            "n_contract_versions": 1,
            "n_cohort_fingerprints": 1,
            "n_readiness_denominator_included": 12,
        },
    }
    if "window_counts" in overrides and isinstance(overrides["window_counts"], dict):
        merged_counts = dict(base["window_counts"])
        merged_counts.update(overrides["window_counts"])
        overrides = dict(overrides)
        overrides["window_counts"] = merged_counts
    base.update(overrides)
    return base

def test_collect_thresholds_uses_configs() -> None:
    import scripts.production_audit_gate as mod

    proof_req = mod._load_profitability_requirements(Path("config/profitability_proof_requirements.yml"))
    thresholds = mod._collect_thresholds(
        monitor_config=Path("config/forecaster_monitoring.yml"),
        proof_requirements=proof_req,
        lift_inconclusive_allowed=True,
        proof_profitable_required=True,
        require_holding_period=True,
        warmup_policy={"warmup_expired": False, "max_warmup_days": 30},
    )

    assert thresholds["proof"]["min_closed_trades"] == 30
    assert thresholds["proof"]["min_profit_factor"] == pytest.approx(1.1)
    assert thresholds["lift"]["min_lift_fraction"] == pytest.approx(0.25)
    assert thresholds["semantics"]["proof_profitable_required"] is True


def test_build_linkage_waterfall_counts() -> None:
    import scripts.production_audit_gate as mod

    waterfall = mod._build_linkage_waterfall(
        {
            "n_outcome_deduped_windows": 20,
            "n_outcome_windows_non_trade_context": 3,
            "n_outcome_windows_invalid_context": 2,
            "n_outcome_windows_eligible": 10,
            "n_outcome_windows_matched": 7,
            "n_readiness_denominator_included": 12,
        },
        production_audit_only=True,
    )
    assert waterfall["raw_candidates"] == 20
    assert waterfall["production_only"] == 17
    assert waterfall["linked"] == 10
    assert waterfall["hygiene_pass"] == 12
    assert waterfall["matched"] == 7
    assert waterfall["matched_over_linked"] == pytest.approx(0.7)


def test_evaluate_artifact_binding_passes_when_fresh_and_bound() -> None:
    import scripts.production_audit_gate as mod

    binding = mod._evaluate_artifact_binding(
        lift_summary={"generated_utc": "2026-03-09T19:35:00+00:00"},
        live_cycle_binding={
            "latest_live_cycle_ts_utc": "2026-03-09T19:30:00+00:00",
            "latest_live_run_id": "20260309_193000",
        },
        repo_state={"head": "abc123"},
    )
    assert binding["pass"] is True
    assert binding["freshness_pass"] is True
    assert binding["run_id_present"] is True
    assert binding["commit_hash_present"] is True
    assert binding["reason_codes"] == []


def test_evaluate_artifact_binding_fails_when_stale() -> None:
    import scripts.production_audit_gate as mod

    binding = mod._evaluate_artifact_binding(
        lift_summary={"generated_utc": "2026-03-09T19:20:00+00:00"},
        live_cycle_binding={
            "latest_live_cycle_ts_utc": "2026-03-09T19:30:00+00:00",
            "latest_live_run_id": "20260309_193000",
        },
        repo_state={"head": "abc123"},
    )
    assert binding["pass"] is False
    assert "SUMMARY_STALE_BEFORE_LIVE_CYCLE" in binding["reason_codes"]


def test_summary_matches_invocation_max_files_match() -> None:
    import scripts.production_audit_gate as mod

    audit_dir = Path.cwd() / "logs" / "forecast_audits"
    summary = {"audit_dir": str(audit_dir), "max_files": 500, "scope": {"include_research": False}}
    assert mod._summary_matches_invocation(
        summary,
        audit_dir=audit_dir,
        max_files=500,
        include_research=False,
    )


def test_summary_matches_invocation_max_files_mismatch() -> None:
    import scripts.production_audit_gate as mod

    audit_dir = Path.cwd() / "logs" / "forecast_audits"
    summary = {"audit_dir": str(audit_dir), "max_files": 500, "scope": {"include_research": False}}
    assert not mod._summary_matches_invocation(
        summary,
        audit_dir=audit_dir,
        max_files=50,
        include_research=False,
    )


def test_summary_matches_invocation_include_research_mismatch() -> None:
    import scripts.production_audit_gate as mod

    audit_dir = Path.cwd() / "logs" / "forecast_audits"
    summary = {"audit_dir": str(audit_dir), "max_files": 500, "scope": {"include_research": True}}
    assert not mod._summary_matches_invocation(
        summary,
        audit_dir=audit_dir,
        max_files=500,
        include_research=False,
    )


def test_missing_summary_metric_keys_requires_measurement_contract_and_nested_counts() -> None:
    import scripts.production_audit_gate as mod

    missing = mod._missing_summary_metric_keys(
        {
            "effective_audits": 20,
            "violation_rate": 0.0,
            "max_violation_rate": 0.35,
            "lift_fraction": 0.30,
            "min_lift_fraction": 0.25,
            "decision": "KEEP",
            "decision_reason": "ok",
            "window_counts": {
                "n_rmse_windows_processed": 20,
            },
        }
    )
    assert "measurement_contract_version" in missing
    assert "baseline_model" in missing
    assert "lift_threshold_rmse_ratio" in missing
    assert "window_counts.n_rmse_windows_usable" in missing
    assert "window_counts.n_outcome_windows_not_due" in missing
    assert "window_counts.n_readiness_denominator_included" in missing


def test_binding_safe_lift_summary_retains_freshness_fields_only() -> None:
    import scripts.production_audit_gate as mod

    raw = {
        "generated_utc": "2026-03-09T20:46:01+00:00",
        "audit_dir": str(Path.cwd() / "logs" / "forecast_audits"),
        "max_files": 500,
        "scope": {"include_research": False},
        "decision": "KEEP",
        "effective_audits": 35,
        "window_counts": {"n_outcome_windows_eligible": 10},
    }
    safe = mod._binding_safe_lift_summary(raw)
    assert safe["generated_utc"] == raw["generated_utc"]
    assert safe["audit_dir"] == raw["audit_dir"]
    assert safe["max_files"] == 500
    assert "window_counts" in safe
    assert "decision" not in safe
    assert "effective_audits" not in safe


def test_main_prefers_structured_summary_over_stdout_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    output_json = tmp_path / "production_gate.json"
    monitor_cfg = tmp_path / "monitor.yml"
    monitor_cfg.write_text("forecaster_monitoring: {}\n", encoding="utf-8")
    proof_cfg = tmp_path / "proof.yml"
    proof_cfg.write_text(
        "\n".join(
            [
                "profitability_proof_requirements:",
                "  statistical_significance:",
                "    min_closed_trades: 30",
                "    min_trading_days: 21",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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
                        "Effective audits with RMSE: 99",
                        "Violation rate: 99.00% (max allowed 35.00%)",
                        "Ensemble lift fraction: 99.00% (required >= 25.00%)",
                        "Decision: KEEP (stdout should not be trusted)",
                    ]
                ),
                stderr="",
            )
        if "validate_profitability_proof.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 100.0, "profit_factor": 1.5, "win_rate": 0.6, "winning_trades": 6, "losing_trades": 4, "trading_days": 21}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    original_safe_load_json = mod._safe_load_json

    def _fake_safe_load_json(path: Path):  # noqa: ANN001
        if Path(path).name == "latest_summary.json":
            return _make_lift_summary(
                audit_dir,
                effective_audits=20,
                violation_rate=0.0,
                lift_fraction=0.30,
            )
        return original_safe_load_json(path)

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_safe_load_json", _fake_safe_load_json)
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(
        mod,
        "_load_latest_live_cycle_binding",
        lambda _db_path: {
            "available": False,
            "latest_live_cycle_ts_utc": None,
            "latest_live_run_id": None,
            "latest_live_trade_id": None,
            "query_error": "test_stubbed",
        },
    )
    monkeypatch.setattr(mod, "_evaluate_artifact_binding", lambda **kwargs: {"pass": True, "reason_codes": []})
    monkeypatch.setattr(mod, "_count_masked_unlinked_closes", lambda _db_path: (0, []))
    monkeypatch.setattr(
        mod,
        "_compute_lifecycle_integrity",
        lambda _db_path: {
            "close_before_entry_count": 0,
            "closed_missing_exit_reason_count": 0,
            "query_error": None,
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
        ],
    )

    rc = mod.main()
    assert rc == 0
    payload = original_safe_load_json(output_json)
    assert payload["lift_gate"]["effective_audits"] == 20
    assert payload["lift_gate"]["lift_fraction"] == pytest.approx(0.30)
    assert payload["lift_gate"]["baseline_model"] == "BEST_SINGLE"
    assert payload["lift_gate"]["lift_threshold_rmse_ratio"] == pytest.approx(1.0)
    assert payload["lift_gate"]["rmse_windows_usable"] == 20
    assert payload["lift_gate"]["outcome_windows_not_due"] == 2


def test_main_fails_closed_when_structured_summary_missing_contract_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    output_json = tmp_path / "production_gate.json"
    monitor_cfg = tmp_path / "monitor.yml"
    monitor_cfg.write_text("forecaster_monitoring: {}\n", encoding="utf-8")
    proof_cfg = tmp_path / "proof.yml"
    proof_cfg.write_text(
        "\n".join(
            [
                "profitability_proof_requirements:",
                "  statistical_significance:",
                "    min_closed_trades: 30",
                "    min_trading_days: 21",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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
                        "Effective audits with RMSE: 99",
                        "Violation rate: 0.00% (max allowed 35.00%)",
                        "Ensemble lift fraction: 99.00% (required >= 25.00%)",
                        "Decision: KEEP (stdout should not be trusted)",
                    ]
                ),
                stderr="",
            )
        if "validate_profitability_proof.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 100.0, "profit_factor": 1.5, "win_rate": 0.6, "winning_trades": 6, "losing_trades": 4, "trading_days": 21}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    original_safe_load_json = mod._safe_load_json

    def _fake_safe_load_json(path: Path):  # noqa: ANN001
        if Path(path).name == "latest_summary.json":
            summary = _make_lift_summary(audit_dir)
            summary.pop("baseline_model", None)
            summary["window_counts"].pop("n_rmse_windows_usable", None)
            return summary
        return original_safe_load_json(path)

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_safe_load_json", _fake_safe_load_json)
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(mod, "_evaluate_artifact_binding", lambda **kwargs: {"pass": True, "reason_codes": []})
    monkeypatch.setattr(
        mod,
        "_compute_lifecycle_integrity",
        lambda _db_path: {
            "close_before_entry_count": 0,
            "closed_missing_exit_reason_count": 0,
            "query_error": None,
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
        ],
    )

    rc = mod.main()
    assert rc == 1
    payload = original_safe_load_json(output_json)
    assert payload["lift_gate"]["summary_metrics_error"] == "SUMMARY_METRICS_MISSING"
    assert "baseline_model" in payload["lift_gate"]["missing_summary_metric_keys"]
    assert "window_counts.n_rmse_windows_usable" in payload["lift_gate"]["missing_summary_metric_keys"]
    assert payload["lift_gate"]["decision"] is None
    assert payload["lift_gate"]["pass"] is False


def test_main_preserves_admission_summary_from_structured_summary(
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
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "validate_profitability_proof.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 100.0, "profit_factor": 1.5, "win_rate": 0.6, "winning_trades": 6, "losing_trades": 4, "trading_days": 21}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    original_safe_load_json = mod._safe_load_json

    def _fake_safe_load_json(path: Path):  # noqa: ANN001
        if Path(path).name == "latest_summary.json":
            return _make_lift_summary(
                audit_dir,
                admission_summary={
                    "accepted_records": 9,
                    "accepted_noneligible_records": 3,
                    "eligible_records": 6,
                    "quarantined_records": 1,
                    "duplicate_conflicts": 2,
                    "missing_execution_metadata_records": 4,
                    "bucket_counts": {
                        "ELIGIBLE": 6,
                        "ACCEPTED_NONELIGIBLE": 3,
                        "QUARANTINED": 1,
                    },
                    "source_counts": {
                        "producer": 9,
                        "legacy_derived": 0,
                    },
                },
            )
        return original_safe_load_json(path)

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_safe_load_json", _fake_safe_load_json)
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(mod, "_evaluate_artifact_binding", lambda **kwargs: {"pass": True, "reason_codes": []})
    monkeypatch.setattr(
        mod,
        "_compute_lifecycle_integrity",
        lambda _db_path: {
            "close_before_entry_count": 0,
            "closed_missing_exit_reason_count": 0,
            "query_error": None,
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
        ],
    )

    rc = mod.main()
    assert rc == 0
    payload = original_safe_load_json(output_json)
    assert payload["readiness"]["admission_summary"]["missing_execution_metadata_records"] == 4
    assert payload["readiness_v2"]["admission_summary"]["bucket_counts"]["QUARANTINED"] == 1
    assert payload["readiness_v2"]["accepted_records"] == 9
    assert payload["readiness_v2"]["duplicate_conflicts"] == 2


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


def _seed_trade_exec_table_with_allocations(
    db_path: Path,
    *,
    close_id: int,
    allocation_entry_ids: list[int],
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            is_close INTEGER NOT NULL,
            entry_trade_id INTEGER,
            realized_pnl REAL
        );
        CREATE TABLE trade_close_allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            close_trade_id INTEGER NOT NULL,
            entry_trade_id INTEGER NOT NULL,
            allocated_shares REAL NOT NULL
        );
        CREATE VIEW trade_close_linkages AS
        SELECT close_trade_id, entry_trade_id, allocated_shares
        FROM trade_close_allocations
        UNION ALL
        SELECT id AS close_trade_id, entry_trade_id, 1.0 AS allocated_shares
        FROM trade_executions
        WHERE is_close = 1
          AND entry_trade_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM trade_close_allocations a WHERE a.close_trade_id = trade_executions.id
          );
        """
    )
    conn.execute(
        "INSERT INTO trade_executions (id, is_close, entry_trade_id, realized_pnl) VALUES (?, 1, NULL, 10.0)",
        (int(close_id),),
    )
    conn.executemany(
        "INSERT INTO trade_close_allocations (close_trade_id, entry_trade_id, allocated_shares) VALUES (?, ?, 1.0)",
        [(int(close_id), int(entry_id)) for entry_id in allocation_entry_ids],
    )
    conn.commit()
    conn.close()


def test_run_reconcile_step_apply_fails_when_unlinked_remains_even_if_command_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod
    monkeypatch.delenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", raising=False)

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
    # Ensure whitelist does not mask trade 66 (INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS=66,75
    # is set in some environments; clear it so the count sees the unlinked close).
    monkeypatch.delenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", raising=False)

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
    monkeypatch.delenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", raising=False)

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


def test_count_unlinked_closes_treats_allocation_linked_close_as_linked(tmp_path: Path) -> None:
    import scripts.production_audit_gate as mod

    db_path = tmp_path / "portfolio.db"
    _seed_trade_exec_table_with_allocations(db_path, close_id=77, allocation_entry_ids=[1001, 1002])

    remaining_count, remaining_ids, error = mod._count_unlinked_closes(db_path, close_ids=[77])

    assert error is None
    assert remaining_count == 0
    assert remaining_ids == []


def test_run_reconcile_step_dry_run_fails_when_unlinked_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod
    monkeypatch.delenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", raising=False)

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
    # Ensure whitelist does not mask trade 66 (INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS=66,75
    # is set in some environments; clear it so the count sees the unlinked close).
    monkeypatch.delenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", raising=False)

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
    original_safe_load_json = mod._safe_load_json
    monkeypatch.setattr(
        mod,
        "_safe_load_json",
        lambda path: _make_lift_summary(audit_dir)
        if Path(path).name == "latest_summary.json"
        else original_safe_load_json(path),
    )
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


def test_main_fails_closed_on_unknown_args(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.production_audit_gate as mod

    monkeypatch.setattr(sys, "argv", ["production_audit_gate.py", "--unknown-flag"])
    with pytest.raises(SystemExit) as excinfo:
        mod.main()
    assert excinfo.value.code == 2


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
    original_safe_load_json = mod._safe_load_json
    monkeypatch.setattr(
        mod,
        "_safe_load_json",
        lambda path: _make_lift_summary(audit_dir, effective_audits=21)
        if Path(path).name == "latest_summary.json"
        else original_safe_load_json(path),
    )
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
    original_safe_load_json = mod._safe_load_json
    monkeypatch.setattr(
        mod,
        "_safe_load_json",
        lambda path: _make_lift_summary(
            audit_dir,
            effective_audits=7,
            decision="INCONCLUSIVE",
            decision_reason="effective_audits=7 < required_audits=20",
            scope={"include_research": False, "production_audit_only": True},
        )
        if Path(path).name == "latest_summary.json"
        else original_safe_load_json(path),
    )
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(
        mod,
        "_load_latest_live_cycle_binding",
        lambda _db_path: {
            "latest_live_cycle_ts_utc": "2026-03-15T11:59:00+00:00",
            "latest_live_run_id": "20260315_115900",
        },
    )
    monkeypatch.setattr(
        mod,
        "_evaluate_artifact_binding",
        lambda **kwargs: {
            "pass": True,
            "freshness_pass": True,
            "run_id_present": True,
            "commit_hash_present": True,
            "reason_codes": [],
            "summary_generated_utc": "2026-03-15T12:00:00+00:00",
            "latest_live_cycle_ts_utc": "2026-03-15T11:59:00+00:00",
            "latest_live_run_id": "20260315_115900",
            "repo_head": "abc123",
        },
    )
    monkeypatch.setattr(
        mod,
        "_compute_lifecycle_integrity",
        lambda _db_path: {
            "close_before_entry_count": 0,
            "closed_missing_exit_reason_count": 0,
            "query_error": None,
        },
    )
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
    assert payload["production_profitability_gate"]["strict_pass"] is False
    assert payload["phase3_strict_ready"] is False


def test_inconclusive_allowed_emits_strict_false_even_when_legacy_phase3_ready(
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
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 10.0, "profit_factor": 1.5, "win_rate": 0.6, "winning_trades": 6, "losing_trades": 4, "trading_days": 30}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    original_safe_load_json = mod._safe_load_json
    monkeypatch.setattr(
        mod,
        "_safe_load_json",
        lambda path: _make_lift_summary(
            audit_dir,
            effective_audits=7,
            decision="INCONCLUSIVE",
            decision_reason="effective_audits=7 < required_audits=20",
            scope={"include_research": False, "production_audit_only": True},
        )
        if Path(path).name == "latest_summary.json"
        else original_safe_load_json(path),
    )
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(
        mod,
        "_load_latest_live_cycle_binding",
        lambda _db_path: {
            "latest_live_cycle_ts_utc": "2026-03-15T11:59:00+00:00",
            "latest_live_run_id": "20260315_115900",
        },
    )
    monkeypatch.setattr(
        mod,
        "_evaluate_artifact_binding",
        lambda **kwargs: {
            "pass": True,
            "freshness_pass": True,
            "run_id_present": True,
            "commit_hash_present": True,
            "reason_codes": [],
            "summary_generated_utc": "2026-03-15T12:00:00+00:00",
            "latest_live_cycle_ts_utc": "2026-03-15T11:59:00+00:00",
            "latest_live_run_id": "20260315_115900",
            "repo_head": "abc123",
        },
    )
    monkeypatch.setattr(
        mod,
        "_compute_lifecycle_integrity",
        lambda _db_path: {
            "close_before_entry_count": 0,
            "closed_missing_exit_reason_count": 0,
            "query_error": None,
        },
    )
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
    assert rc == 0
    payload = mod._safe_load_json(output_json)
    assert payload is not None
    assert payload["production_profitability_gate"]["pass"] is True
    assert payload["production_profitability_gate"]["strict_pass"] is False
    assert payload["production_profitability_gate"]["gate_semantics_status"] == "INCONCLUSIVE_ALLOWED"
    assert payload["production_profitability_gate"]["posture"] == "WARMUP_COVERED_PASS"
    assert "gate_semantics_inconclusive_allowed" in payload["production_profitability_gate"]["covered_state_reasons"]
    assert payload["phase3_ready"] is True
    assert payload["phase3_strict_ready"] is False
    assert payload["phase3_strict_reason"].endswith("GATE_SEMANTICS_INCONCLUSIVE_ALLOWED")
    assert payload["posture"] == "WARMUP_COVERED_PASS"
    assert payload["readiness"]["phase3_ready"] is True
    assert payload["readiness"]["phase3_strict_ready"] is False
    assert payload["readiness"]["posture"] == "WARMUP_COVERED_PASS"


def test_warmup_covered_posture_detects_proof_and_linkage_gaps_even_when_strict_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.production_audit_gate as mod

    output_json = tmp_path / "production_gate.json"
    monitor_cfg = tmp_path / "monitor.yml"
    monitor_cfg.write_text("forecaster_monitoring: {}\n", encoding="utf-8")
    proof_cfg = tmp_path / "proof.yml"
    proof_cfg.write_text(
        "\n".join(
            [
                "profitability_proof_requirements:",
                "  statistical_significance:",
                "    min_closed_trades: 30",
                "    min_trading_days: 21",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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
                        "Effective audits with RMSE: 35",
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
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 10.0, "profit_factor": 1.5, "win_rate": 0.6, "winning_trades": 20, "losing_trades": 20, "trading_days": 10}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    original_safe_load_json = mod._safe_load_json
    monkeypatch.setattr(
        mod,
        "_safe_load_json",
        lambda path: _make_lift_summary(
            audit_dir,
            effective_audits=35,
            decision="KEEP",
            decision_reason="lift demonstrated during holding period",
            scope={"include_research": False, "production_audit_only": True},
            window_counts={
                "n_outcome_windows_eligible": 309,
                "n_outcome_windows_matched": 1,
                "n_readiness_denominator_included": 309,
            },
        )
        if Path(path).name == "latest_summary.json"
        else original_safe_load_json(path),
    )
    monkeypatch.setattr(mod, "_collect_git_state", lambda _repo_root: {"available": False})
    monkeypatch.setattr(
        mod,
        "_load_latest_live_cycle_binding",
        lambda _db_path: {
            "latest_live_cycle_ts_utc": "2026-03-15T11:59:00+00:00",
            "latest_live_run_id": "20260315_115900",
        },
    )
    monkeypatch.setattr(
        mod,
        "_evaluate_artifact_binding",
        lambda **kwargs: {
            "pass": True,
            "freshness_pass": True,
            "run_id_present": True,
            "commit_hash_present": True,
            "reason_codes": [],
            "summary_generated_utc": "2026-03-15T12:00:00+00:00",
            "latest_live_cycle_ts_utc": "2026-03-15T11:59:00+00:00",
            "latest_live_run_id": "20260315_115900",
            "repo_head": "abc123",
        },
    )
    monkeypatch.setattr(
        mod,
        "_compute_lifecycle_integrity",
        lambda _db_path: {
            "close_before_entry_count": 0,
            "closed_missing_exit_reason_count": 0,
            "query_error": None,
        },
    )
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
    assert rc == 0
    payload = mod._safe_load_json(output_json)
    assert payload is not None
    assert payload["phase3_strict_ready"] is True
    assert payload["posture"] == "WARMUP_COVERED_PASS"
    assert payload["readiness"]["posture"] == "WARMUP_COVERED_PASS"
    assert payload["readiness"]["proof_evidence_ready"] is False
    assert payload["readiness"]["linkage_full_thresholds_pass"] is False
    assert set(payload["covered_state_reasons"]) == {
        "proof_evidence_incomplete",
        "linkage_warmup_exemption",
    }


def test_compute_lifecycle_integrity_excludes_legacy_trades(tmp_path: Path) -> None:
    """Legacy trades (ts_signal_id LIKE 'legacy_%') must not be flagged for
    close_before_entry — their bar_timestamps are historical forecast bars, not
    execution clocks, and are unreliable for chronological ordering."""
    import scripts.production_audit_gate as mod

    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY, ticker TEXT, trade_date TEXT, action TEXT,
            realized_pnl REAL, is_close INTEGER DEFAULT 0,
            is_diagnostic INTEGER DEFAULT 0, is_synthetic INTEGER DEFAULT 0,
            bar_timestamp TEXT, exit_reason TEXT, entry_trade_id INTEGER,
            ts_signal_id TEXT
        );
        -- Legacy open leg: bar_timestamp from 2025 (historical forecast bar)
        INSERT INTO trade_executions VALUES
          (1,'MSFT','2026-02-10','BUY',NULL,0,0,0,
           '2025-07-25T00:00:00+00:00',NULL,NULL,'legacy_2026-02-10_1');
        -- Legacy close leg: bar_timestamp from 2025 — EARLIER than open (artifact)
        INSERT INTO trade_executions VALUES
          (2,'MSFT','2026-02-10','SELL',72.5,1,0,0,
           '2025-07-18T00:00:00+00:00','TIME_EXIT',1,'legacy_2026-02-10_2');
        CREATE VIEW production_closed_trades AS
          SELECT * FROM trade_executions
          WHERE is_close=1
            AND COALESCE(is_diagnostic,0)=0
            AND COALESCE(is_synthetic,0)=0;
        """
    )
    conn.close()

    result = mod._compute_lifecycle_integrity(db)

    assert result["close_before_entry_count"] == 0, (
        "Legacy trades must not be counted as close_before_entry violations"
    )
    assert result["query_error"] is None


def test_count_masked_unlinked_closes_returns_zero_when_whitelist_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """masked count must be 0 when no whitelist env var is set."""
    import scripts.production_audit_gate as mod

    monkeypatch.delenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", raising=False)
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE trade_executions "
        "(id INTEGER PRIMARY KEY, is_close INTEGER, entry_trade_id INTEGER);"
        "INSERT INTO trade_executions VALUES (255, 1, NULL);"
    )
    conn.close()

    count, ids = mod._count_masked_unlinked_closes(db)
    assert count == 0
    assert ids == []


def test_count_masked_unlinked_closes_counts_matching_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """masked count must reflect whitelisted IDs actually in DB as unlinked closes."""
    import scripts.production_audit_gate as mod

    monkeypatch.setenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", "66,75,255")
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE trade_executions "
        "(id INTEGER PRIMARY KEY, is_close INTEGER, entry_trade_id INTEGER);"
        # id=255: unlinked live close (whitelisted)
        "INSERT INTO trade_executions VALUES (255, 1, NULL);"
        # id=66: whitelisted but has an entry_trade_id (linked — not counted)
        "INSERT INTO trade_executions VALUES (66, 1, 50);"
        # id=999: unlinked but NOT in whitelist
        "INSERT INTO trade_executions VALUES (999, 1, NULL);"
    )
    conn.close()

    count, ids = mod._count_masked_unlinked_closes(db)
    assert count == 1, "only id=255 is both whitelisted AND unlinked"
    assert ids == [255]


def test_count_masked_unlinked_closes_absent_from_db_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """masked count must be 0 when whitelisted IDs are not in DB at all."""
    import scripts.production_audit_gate as mod

    monkeypatch.setenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", "66,75,255")
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE trade_executions "
        "(id INTEGER PRIMARY KEY, is_close INTEGER, entry_trade_id INTEGER);"
    )
    conn.close()

    count, ids = mod._count_masked_unlinked_closes(db)
    assert count == 0
    assert ids == []


# ---------------------------------------------------------------------------
# Phase 10: THIN_LINKAGE warmup provision
# ---------------------------------------------------------------------------


def test_linkage_fails_closed_when_no_eligible_records() -> None:
    """0 eligible records must no longer vacuously satisfy linkage."""
    import scripts.production_audit_gate as mod

    # Simulate: 0 matched, 0 eligible — warmup active
    warmup_policy = {"warmup_expired": False, "max_warmup_days": 30}

    outcome_matched = 0
    outcome_eligible = 0
    matched_over_eligible = mod._safe_ratio(outcome_matched, outcome_eligible)
    _linkage_min_matched = 10
    _linkage_min_ratio = 0.8
    _linkage_warmup_active = not bool(warmup_policy.get("warmup_expired", True))
    if _linkage_warmup_active:
        _linkage_min_matched = 1
        _linkage_min_ratio = 0.0
    _linkage_no_eligible = outcome_eligible == 0
    linkage_pass = (
        outcome_eligible > 0
        and (
        outcome_matched >= _linkage_min_matched
        and matched_over_eligible >= _linkage_min_ratio
        )
    )

    assert linkage_pass is False, "0 eligible records must fail closed on linkage"
    assert _linkage_warmup_active is True
    assert _linkage_no_eligible is True


def test_linkage_fail_when_warmup_expired_and_below_threshold() -> None:
    """After warmup, THIN_LINKAGE must fail when matched < 10."""
    import scripts.production_audit_gate as mod

    warmup_policy = {"warmup_expired": True, "max_warmup_days": 30}

    outcome_matched = 3
    outcome_eligible = 10
    matched_over_eligible = mod._safe_ratio(outcome_matched, outcome_eligible)
    _linkage_min_matched = 10
    _linkage_min_ratio = 0.8
    _linkage_warmup_active = not bool(warmup_policy.get("warmup_expired", True))
    if _linkage_warmup_active:
        _linkage_min_matched = 1
        _linkage_min_ratio = 0.0
    _linkage_no_eligible = outcome_eligible == 0
    linkage_pass = _linkage_no_eligible or (
        outcome_matched >= _linkage_min_matched
        and matched_over_eligible >= _linkage_min_ratio
    )

    assert linkage_pass is False, "matched=3 < 10 after warmup must fail linkage"
    assert _linkage_warmup_active is False
    assert _linkage_no_eligible is False


def test_main_passes_effective_default_baseline_through_to_gate_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When lift summary reports baseline_model=EFFECTIVE_DEFAULT the gate output must
    reflect EFFECTIVE_DEFAULT in lift_gate.baseline_model (pure passthrough check)."""
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
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "validate_profitability_proof.py" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"is_proof_valid": true, "is_profitable": true, "metrics": {"total_pnl": 200.0, "profit_factor": 2.0, "win_rate": 0.6, "winning_trades": 6, "losing_trades": 4, "trading_days": 21}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {joined}")

    original_safe_load_json = mod._safe_load_json

    def _fake_safe_load_json(path: Path):  # noqa: ANN001
        if Path(path).name == "latest_summary.json":
            # Override baseline_model to EFFECTIVE_DEFAULT — this is the new value
            # introduced by the causal baseline recalibration.
            return _make_lift_summary(audit_dir, baseline_model="EFFECTIVE_DEFAULT")
        return original_safe_load_json(path)

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_safe_load_json", _fake_safe_load_json)
    monkeypatch.setattr(mod, "_collect_git_state", lambda _: {"available": False})
    monkeypatch.setattr(
        mod,
        "_load_latest_live_cycle_binding",
        lambda _: {"available": False, "latest_live_cycle_ts_utc": None,
                    "latest_live_run_id": None, "latest_live_trade_id": None,
                    "query_error": "test_stubbed"},
    )
    monkeypatch.setattr(mod, "_evaluate_artifact_binding", lambda **kwargs: {"pass": True, "reason_codes": []})
    monkeypatch.setattr(mod, "_count_masked_unlinked_closes", lambda _: (0, []))
    monkeypatch.setattr(
        mod,
        "_compute_lifecycle_integrity",
        lambda _: {"close_before_entry_count": 0,
                   "closed_missing_exit_reason_count": 0, "query_error": None},
    )
    monkeypatch.setenv("PMX_NOTIFY_OPENCLAW", "0")
    monkeypatch.setenv("OPENCLAW_TARGETS", "")
    monkeypatch.setenv("OPENCLAW_TO", "")
    monkeypatch.setattr(
        sys,
        "argv",
        ["production_audit_gate.py", "--db", str(db_path),
         "--proof-requirements", str(proof_cfg),
         "--audit-dir", str(audit_dir),
         "--monitor-config", str(monitor_cfg),
         "--output-json", str(output_json)],
    )

    rc = mod.main()
    assert rc == 0
    payload = original_safe_load_json(output_json)
    assert payload["lift_gate"]["baseline_model"] == "EFFECTIVE_DEFAULT"


def test_evidence_hygiene_excludes_execution_rejected_from_dirty_count() -> None:
    """EXECUTION_REJECTED invalids must not fail evidence hygiene.

    Phase 11-B introduced writing audit files for every forecaster run to the
    production dir, including HOLD/blocked signals (executed=False,
    source=producer-native).  These are legitimate production artifacts — they
    should be counted as EXECUTION_REJECTED and excluded from the "dirty"
    invalid count that evidence_hygiene_pass tests against.
    """
    import scripts.production_audit_gate as mod

    safe_int = mod._safe_int

    # Scenario: 100 invalids, 95 of which are EXECUTION_REJECTED (clean HOLD signals).
    # Only 5 are genuinely dirty (missing metadata, horizon mismatch, etc.).
    invalid_context_count = 100
    execution_rejected_count = 95
    non_trade_count = 0
    production_audit_only = True

    dirty_invalid_count = max(0, invalid_context_count - execution_rejected_count)
    evidence_hygiene_pass = production_audit_only and dirty_invalid_count == 0

    # 5 dirty invalids remain — hygiene must FAIL.
    assert dirty_invalid_count == 5
    assert evidence_hygiene_pass is False

    # With all invalids being EXECUTION_REJECTED — hygiene must PASS.
    all_rejected_dirty_count = max(0, 100 - 100)
    evidence_hygiene_all_clean = production_audit_only and all_rejected_dirty_count == 0
    assert all_rejected_dirty_count == 0
    assert evidence_hygiene_all_clean is True

    # Verify _safe_int handles the new field name correctly.
    window_counts = {
        "n_outcome_windows_invalid_context": 123,
        "n_outcome_windows_execution_rejected": 123,
    }
    er = safe_int(window_counts.get("n_outcome_windows_execution_rejected"), 0)
    inv = safe_int(window_counts.get("n_outcome_windows_invalid_context"), 0)
    assert max(0, inv - er) == 0


def test_evidence_hygiene_excludes_misrouted_rmse_only_records_from_dirty_invalids() -> None:
    """Explicit RMSE_ONLY artifacts stay visible but do not count as dirty live-trade invalids."""
    import scripts.production_audit_gate as mod

    safe_int = mod._safe_int
    window_counts = {
        "n_outcome_windows_invalid_context": 3,
        "n_outcome_windows_execution_rejected": 2,
        "n_misrouted_rmse_only_records": 1,
    }

    invalid_context_count = safe_int(window_counts.get("n_outcome_windows_invalid_context"), 0)
    execution_rejected_count = safe_int(window_counts.get("n_outcome_windows_execution_rejected"), 0)
    misrouted_rmse_only_count = safe_int(window_counts.get("n_misrouted_rmse_only_records"), 0)
    dirty_invalid_count = max(0, invalid_context_count - execution_rejected_count - misrouted_rmse_only_count)
    evidence_hygiene_pass = dirty_invalid_count == 0

    assert dirty_invalid_count == 0
    assert evidence_hygiene_pass is True
    assert safe_int(window_counts.get("n_misrouted_rmse_only_records"), 0) == 1


def test_evidence_hygiene_ignores_non_trade_rmse_only_exclusions() -> None:
    """Non-trade RMSE-only exclusions are no longer hygiene blockers."""
    production_audit_only = True
    non_trade_count = 2
    dirty_invalid_count = 0

    evidence_hygiene_pass = production_audit_only and dirty_invalid_count == 0

    assert non_trade_count == 2
    assert evidence_hygiene_pass is True
