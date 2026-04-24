from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts import build_nav_rebalance_plan as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_v4_canonical_snapshot(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 4,
            "summary": {
                "ann_roi_pct": 30.2,
                "roi_ann_pct": 30.2,
                "deployment_pct": 1.83,
                "objective_score": 55.266,
                "objective_valid": True,
                "ngn_hurdle_pct": 28.0,
                "gap_to_hurdle_pp": -2.2,
                "evidence_health": "clean",
                "unattended_gate": "PASS",
                "unattended_ready": True,
            },
            "gate": {
                "posture": "GENUINE_PASS",
                "freshness_status": {
                    "status": "fresh",
                    "age_minutes": 15.0,
                    "expected_max_age_minutes": 1440.0,
                    "last_expected_emission_utc": "2026-04-18T20:00:00Z",
                    "last_actual_emission_utc": "2026-04-18T19:45:00Z",
                },
                "warmup_state": {
                    "posture": "expired",
                    "deadline_utc": "2026-04-24T20:00:00Z",
                    "matched_needed": 0,
                },
                "trajectory_alarm": {
                    "active": False,
                    "days_to_deadline": 5.0,
                    "matched_needed": 0,
                    "expected_closes_remaining": 0.0,
                    "shortfall": 0.0,
                },
                "post_deadline_time_to_10_estimate": {
                    "status": "inactive",
                    "estimated_days": None,
                    "covered_lot_term_days": None,
                    "new_round_trip_term_days": None,
                    "covered_lots_remaining": 0,
                    "matched_needed": 0,
                    "covered_lot_daily_close_rate": 0.0,
                    "new_round_trip_daily_rate": 0.0,
                },
                "gate_artifact_age_minutes": 15.0,
            },
            "utilization": {"roi_ann_pct": 30.2, "deployment_pct": 1.83},
            "alpha_objective": {
                "roi_ann_pct": 30.2,
                "deployment_pct": 1.83,
                "objective_score": 55.266,
                "objective_valid": True,
            },
            "thin_linkage": {
                "matched_current": 10,
                "matched_needed": 0,
                "trajectory_alarm": {
                    "active": False,
                    "days_to_deadline": 5.0,
                    "matched_needed": 0,
                    "expected_closes_remaining": 0.0,
                    "shortfall": 0.0,
                },
                "post_deadline_time_to_10_estimate": {
                    "status": "inactive",
                    "estimated_days": None,
                    "covered_lot_term_days": None,
                    "new_round_trip_term_days": None,
                    "covered_lots_remaining": 0,
                    "matched_needed": 0,
                    "covered_lot_daily_close_rate": 0.0,
                    "new_round_trip_daily_rate": 0.0,
                },
            },
            "source_contract": {
                "status": "clean",
                "canonical_sources": [
                    {"metric": "closed_pnl", "source_file": "production_closed_trades", "query_or_key": "production_closed_trades"},
                    {"metric": "capital", "source_file": "portfolio_cash_state", "query_or_key": "portfolio_cash_state.initial_capital"},
                    {"metric": "open_risk", "source_file": "trade_executions", "query_or_key": "trade_executions WHERE is_close=0"},
                    {"metric": "utilization", "source_file": "scripts/compute_capital_utilization.py", "query_or_key": "scripts.compute_capital_utilization.compute_utilization"},
                ],
                "allowlisted_readers": [
                    "scripts/build_nav_rebalance_plan.py",
                    "scripts/build_automation_dashboard.py",
                ],
                "violations_found": [],
                "scan_timestamp_utc": "2026-04-18T12:00:00Z",
                "canonical": {"closed_pnl": "production_closed_trades"},
                "ui_only": {"metrics_summary": "visualizations/performance/metrics_summary.json"},
            },
        },
    )


def test_shadow_first_plan_demotes_weak_and_concentrates_healthy(tmp_path: Path) -> None:
    eligibility_path = tmp_path / "logs" / "ticker_eligibility.json"
    eligibility_gates_path = tmp_path / "logs" / "ticker_eligibility_gates.json"
    sleeve_summary_path = tmp_path / "logs" / "automation" / "sleeve_summary.json"
    sleeve_plan_path = tmp_path / "logs" / "automation" / "sleeve_promotion_plan.json"
    canonical_snapshot_path = tmp_path / "logs" / "canonical_snapshot_latest.json"
    risk_buckets_path = tmp_path / "config" / "risk_buckets.yml"
    output_path = tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest.json"

    _write_json(
        eligibility_path,
        {
            "generated_utc": "2026-04-18T12:00:00Z",
            "summary": {"HEALTHY": 3, "WEAK": 2, "LAB_ONLY": 1},
            "warnings": [],
            "tickers": {
                "AAPL": {
                    "status": "WEAK",
                    "n_trades": 8,
                    "win_rate": 0.125,
                    "profit_factor": 0.0132,
                    "total_pnl": -376.02,
                    "reasons": ["win_rate_below_weak_floor(0.13<0.30)"],
                },
                "GS": {
                    "status": "WEAK",
                    "n_trades": 5,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_pnl": -91.26,
                    "reasons": ["win_rate_below_weak_floor(0.00<0.30)"],
                },
                "NVDA": {
                    "status": "HEALTHY",
                    "n_trades": 8,
                    "win_rate": 0.625,
                    "profit_factor": 8.4616,
                    "omega_ratio": 1.9,
                    "payoff_asymmetry_effective": 3.8,
                    "take_profit_frequency": 0.20,
                    "total_pnl": 706.97,
                    "reasons": ["meets_r3_thresholds(n>=20, wr>=0.45, pf>=1.30)"],
                },
                "MSFT": {
                    "status": "HEALTHY",
                    "n_trades": 6,
                    "win_rate": 0.6667,
                    "profit_factor": 2.2562,
                    "omega_ratio": 1.4,
                    "payoff_asymmetry_effective": 2.6,
                    "take_profit_frequency": 0.12,
                    "total_pnl": 188.21,
                    "reasons": ["meets_r3_thresholds(n>=20, wr>=0.45, pf>=1.30)"],
                },
                "GOOG": {
                    "status": "HEALTHY",
                    "n_trades": 10,
                    "win_rate": 0.5,
                    "profit_factor": 2.4283,
                    "omega_ratio": 1.1,
                    "payoff_asymmetry_effective": 2.1,
                    "take_profit_frequency": 0.08,
                    "total_pnl": 180.29,
                    "reasons": ["meets_r3_thresholds(n>=20, wr>=0.45, pf>=1.30)"],
                },
                "TSLA": {
                    "status": "LAB_ONLY",
                    "n_trades": 1,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_pnl": -1.09,
                    "reasons": ["manual_research_only"],
                },
            },
        },
    )
    _write_json(
        eligibility_gates_path,
        {
            "generated_utc": "2026-04-18T12:00:00Z",
            "status": "PASS",
            "gate_written": True,
            "summary": {"HEALTHY": 3, "WEAK": 2, "LAB_ONLY": 1},
            "warnings": [],
            "errors": [],
        },
    )
    _write_json(
        sleeve_summary_path,
        {
            "generated_at": "2026-04-18",
            "sleeves": [
                {"sleeve": "core", "ticker": "AAPL", "trades": 8, "win_rate": 0.125, "profit_factor": 0.01},
                {"sleeve": "core", "ticker": "GS", "trades": 5, "win_rate": 0.0, "profit_factor": 0.0},
                {"sleeve": "speculative", "ticker": "NVDA", "trades": 8, "win_rate": 0.625, "profit_factor": 8.46},
                {"sleeve": "speculative", "ticker": "MSFT", "trades": 6, "win_rate": 0.6667, "profit_factor": 2.25},
                {"sleeve": "speculative", "ticker": "GOOG", "trades": 10, "win_rate": 0.5, "profit_factor": 2.43},
                {"sleeve": "other", "ticker": "TSLA", "trades": 1, "win_rate": 0.0, "profit_factor": 0.0},
            ],
        },
    )
    _write_json(
        sleeve_plan_path,
        {
            "meta": {"generated_at": "2026-04-18"},
            "plan": {
                "promotions": [
                    {"ticker": "NVDA", "from": "speculative", "to": "core", "reason": "Promote"},
                    {"ticker": "MSFT", "from": "speculative", "to": "core", "reason": "Promote"},
                    {"ticker": "GOOG", "from": "speculative", "to": "core", "reason": "Promote"},
                ],
                "demotions": [
                    {"ticker": "AAPL", "from": "core", "to": "speculative", "reason": "Demote"},
                    {"ticker": "GS", "from": "core", "to": "speculative", "reason": "Demote"},
                ],
            },
        },
    )
    _write_v4_canonical_snapshot(canonical_snapshot_path)
    snapshot_payload = json.loads(canonical_snapshot_path.read_text(encoding="utf-8"))
    snapshot_payload.setdefault("gate", {})["coverage_ratio_alarm"] = {
        "active": True,
        "ratio": 0.12,
        "matched_needed": 0,
        "expected_closes_remaining": 0.0,
        "shortfall": 0.0,
    }
    canonical_snapshot_path.write_text(json.dumps(snapshot_payload, indent=2), encoding="utf-8")
    _write_yaml(
        risk_buckets_path,
        """
risk_buckets:
  enabled: false
  base_nav_frac:
    safe: 0.80
    ts_core: 0.12
    speculative: 0.05
    ml_secondary: 0.02
    llm_fallback: 0.01
  min_nav_frac:
    safe: 0.65
    ts_core: 0.08
    speculative: 0.02
    ml_secondary: 0.0
    llm_fallback: 0.0
  max_nav_frac:
    safe: 0.95
    ts_core: 0.20
    speculative: 0.10
    ml_secondary: 0.05
    llm_fallback: 0.02
""".strip(),
    )

    runner = CliRunner()
    result = runner.invoke(
        mod.main,
        [
            "--eligibility-path",
            str(eligibility_path),
            "--eligibility-gates-path",
            str(eligibility_gates_path),
            "--sleeve-summary-path",
            str(sleeve_summary_path),
            "--sleeve-plan-path",
            str(sleeve_plan_path),
            "--canonical-snapshot-path",
            str(canonical_snapshot_path),
            "--risk-buckets-path",
            str(risk_buckets_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["rollout"]["mode"] == "shadow"
    assert payload["rollout"]["live_apply_allowed"] is False
    assert payload["rollout"]["gate_lift_candidate"] is True
    assert "shadow_first_default" in payload["rollout"]["live_apply_blockers"]
    assert "evidence_not_green" not in payload["rollout"]["live_apply_blockers"]
    assert set(payload["rollout"]["live_apply_blockers"]) == {
        "shadow_first_default",
        "gate_lift_waiting_for_2_green_cycles",
    }
    assert payload["rollout"]["evidence_status"] in {"PASS", "CLEAN"}
    assert payload["rollout"]["gate_lift_state"]["current_consecutive_green_cycles"] == 1
    assert payload["evidence_contract"]["source_kind"] == "canonical_closed_trades"
    assert payload["evidence_contract"]["oos_source_kind"] == "GENUINE_OOS"
    assert payload["evidence_contract"]["data_source"] == "production_closed_trades"
    assert payload["evidence_contract"]["canonical_coverage_ratio_alarm_active"] is True

    target_map = {row["ticker"]: row for row in payload["targets"]}
    assert target_map["AAPL"]["action"] == "DEMOTE"
    assert target_map["GS"]["action"] == "DEMOTE"
    assert target_map["NVDA"]["action"] == "PROMOTE"
    assert target_map["MSFT"]["action"] == "PROMOTE"
    assert target_map["GOOG"]["action"] == "PROMOTE"
    assert target_map["AAPL"]["target_nav_frac"] == 0.0
    assert target_map["GS"]["target_nav_frac"] == 0.0
    assert target_map["NVDA"]["target_nav_frac"] > target_map["MSFT"]["target_nav_frac"] > target_map["GOOG"]["target_nav_frac"]
    assert target_map["AAPL"]["metrics"]["coverage_ratio"] is None
    assert target_map["AAPL"]["metrics"]["oos_source_kind"] == "GENUINE_OOS"
    assert target_map["AAPL"]["metrics"]["provenance_trusted"] is True
    assert payload["evidence_contract"]["coverage_ratio"] is None
    assert payload["evidence_contract"]["oos_metrics_available"] is False

    bucket_map = {row["bucket"]: row for row in payload["bucket_allocations"]}
    assert bucket_map["safe"]["reserve_nav_frac"] == pytest.approx(0.80, abs=1e-8)
    assert bucket_map["ts_core"]["allocated_nav_frac"] == pytest.approx(0.12, abs=1e-8)
    assert bucket_map["speculative"]["reserve_nav_frac"] == pytest.approx(0.05, abs=1e-8)
    assert payload["summary"]["allocated_symbol_nav_frac"] == pytest.approx(0.12, abs=1e-8)
    assert payload["summary"]["unallocated_bucket_nav_frac"] > 0.0


def test_build_nav_rebalance_plan_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    common_files = {
        "eligibility_path": tmp_path / "logs" / "ticker_eligibility.json",
        "eligibility_gates_path": tmp_path / "logs" / "ticker_eligibility_gates.json",
        "sleeve_summary_path": tmp_path / "logs" / "automation" / "sleeve_summary.json",
        "sleeve_plan_path": tmp_path / "logs" / "automation" / "sleeve_promotion_plan.json",
        "canonical_snapshot_path": tmp_path / "logs" / "canonical_snapshot_latest.json",
        "risk_buckets_path": tmp_path / "config" / "risk_buckets.yml",
    }
    for key, path in common_files.items():
        if key.endswith("_path") and "risk_buckets" not in key:
            _write_json(path, {"status": "PASS", "summary": {}, "warnings": [], "tickers": {}})
    _write_yaml(
        common_files["risk_buckets_path"],
        """
risk_buckets:
  enabled: false
  base_nav_frac:
    safe: 0.80
    ts_core: 0.12
    speculative: 0.05
    ml_secondary: 0.02
    llm_fallback: 0.01
""".strip(),
    )
    # Overwrite with a small but complete eligible data set.
    _write_json(
        common_files["eligibility_path"],
        {
            "generated_utc": "2026-04-18T12:00:00Z",
            "summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0},
            "warnings": [],
            "tickers": {
                "NVDA": {
                    "status": "HEALTHY",
                    "n_trades": 8,
                    "win_rate": 0.625,
                    "profit_factor": 8.4616,
                    "total_pnl": 706.97,
                    "reasons": ["meets_r3_thresholds"],
                }
            },
        },
    )
    _write_json(
        common_files["eligibility_gates_path"],
        {
            "generated_utc": "2026-04-18T12:00:00Z",
            "status": "PASS",
            "gate_written": True,
            "summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0},
            "warnings": [],
            "errors": [],
        },
    )
    _write_json(
        common_files["sleeve_summary_path"],
        {"generated_at": "2026-04-18", "sleeves": [{"sleeve": "other", "ticker": "NVDA", "trades": 8, "win_rate": 0.625, "profit_factor": 8.46}]},
    )
    _write_json(
        common_files["sleeve_plan_path"],
        {"plan": {"promotions": [{"ticker": "NVDA", "from": "speculative", "to": "core"}], "demotions": []}},
    )
    _write_v4_canonical_snapshot(common_files["canonical_snapshot_path"])

    first = mod.build_nav_rebalance_plan(
        eligibility_path=common_files["eligibility_path"],
        eligibility_gates_path=common_files["eligibility_gates_path"],
        sleeve_summary_path=common_files["sleeve_summary_path"],
        sleeve_plan_path=common_files["sleeve_plan_path"],
        canonical_snapshot_path=common_files["canonical_snapshot_path"],
        risk_buckets_path=common_files["risk_buckets_path"],
        output_path=tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest.json",
    )
    second = mod.build_nav_rebalance_plan(
        eligibility_path=common_files["eligibility_path"],
        eligibility_gates_path=common_files["eligibility_gates_path"],
        sleeve_summary_path=common_files["sleeve_summary_path"],
        sleeve_plan_path=common_files["sleeve_plan_path"],
        canonical_snapshot_path=common_files["canonical_snapshot_path"],
        risk_buckets_path=common_files["risk_buckets_path"],
        output_path=tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest_2.json",
    )

    assert first["summary"]["allocated_symbol_nav_frac"] == pytest.approx(second["summary"]["allocated_symbol_nav_frac"])
    assert first["targets"] == second["targets"]
    assert first["rollout"]["gate_lift_candidate"] is True


def test_build_nav_rebalance_plan_lifts_after_two_consecutive_green_cycles(tmp_path: Path) -> None:
    eligibility_path = tmp_path / "logs" / "ticker_eligibility.json"
    eligibility_gates_path = tmp_path / "logs" / "ticker_eligibility_gates.json"
    sleeve_summary_path = tmp_path / "logs" / "automation" / "sleeve_summary.json"
    sleeve_plan_path = tmp_path / "logs" / "automation" / "sleeve_promotion_plan.json"
    canonical_snapshot_path = tmp_path / "logs" / "canonical_snapshot_latest.json"
    risk_buckets_path = tmp_path / "config" / "risk_buckets.yml"
    output_path = tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest.json"
    history_root = tmp_path / "logs" / "automation" / "nav_rebalance_plan_history"

    _write_json(
        eligibility_path,
        {
            "generated_utc": "2026-04-18T12:00:00Z",
            "summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0},
            "warnings": [],
            "tickers": {
                "NVDA": {
                    "status": "HEALTHY",
                    "n_trades": 8,
                    "win_rate": 0.625,
                    "profit_factor": 8.4616,
                    "total_pnl": 706.97,
                    "reasons": ["meets_r3_thresholds"],
                }
            },
        },
    )
    _write_json(
        eligibility_gates_path,
        {
            "generated_utc": "2026-04-18T12:00:00Z",
            "status": "PASS",
            "gate_written": True,
            "summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0},
            "warnings": [],
            "errors": [],
        },
    )
    _write_json(
        sleeve_summary_path,
        {"generated_at": "2026-04-18", "sleeves": [{"sleeve": "other", "ticker": "NVDA", "trades": 8, "win_rate": 0.625, "profit_factor": 8.46}]},
    )
    _write_json(
        sleeve_plan_path,
        {"plan": {"promotions": [{"ticker": "NVDA", "from": "speculative", "to": "core"}], "demotions": []}},
    )
    _write_v4_canonical_snapshot(canonical_snapshot_path)
    _write_yaml(
        risk_buckets_path,
        """
risk_buckets:
  enabled: false
  base_nav_frac:
    safe: 0.80
    ts_core: 0.12
    speculative: 0.05
    ml_secondary: 0.02
    llm_fallback: 0.01
""".strip(),
    )
    _write_json(
        history_root / "nav_rebalance_plan_20260417_120000_000000.json",
        {
            "rollout": {
                "gate_lift_candidate": True,
                "evidence_warnings": [],
                "evidence_status": "PASS",
            }
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        mod.main,
        [
            "--eligibility-path",
            str(eligibility_path),
            "--eligibility-gates-path",
            str(eligibility_gates_path),
            "--sleeve-summary-path",
            str(sleeve_summary_path),
            "--sleeve-plan-path",
            str(sleeve_plan_path),
            "--canonical-snapshot-path",
            str(canonical_snapshot_path),
            "--risk-buckets-path",
            str(risk_buckets_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["rollout"]["gate_lift_candidate"] is True
    assert payload["rollout"]["gate_lift_state"]["prior_consecutive_green_cycles"] == 1
    assert payload["rollout"]["gate_lift_state"]["current_consecutive_green_cycles"] == 2
    assert payload["rollout"]["live_apply_allowed"] is True
    assert payload["rollout"]["gate_lift_ready"] is True
    assert payload["rollout"]["mode"] == "live"
    assert payload["rollout"]["live_apply_blockers"] == []


def test_direct_cli_invocation_does_not_raise_module_not_found_error() -> None:
    """Direct subprocess invocation (as cron would call it) must not raise ModuleNotFoundError.

    The CliRunner path injects sys.path via pytest conftest, so it cannot catch the
    real failure mode. This test invokes the script as an external process — no PYTHONPATH
    set — to reproduce the cron environment and confirm the sys.path bootstrap works.
    """
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "build_nav_rebalance_plan.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        # Deliberately omit PYTHONPATH to mirror the cron/shell invocation environment
        env={k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"},
    )
    assert result.returncode == 0, (
        f"Direct CLI invocation failed (exit={result.returncode}).\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "ModuleNotFoundError" not in result.stderr, (
        f"sys.path bootstrap missing — first-party imports failed:\n{result.stderr[:500]}"
    )
    assert "Usage:" in result.stdout, (
        f"Expected Click help output; got: {result.stdout[:200]}"
    )


def test_evidence_contract_suppresses_objective_score_when_invalid() -> None:
    canonical_snapshot = {
        "schema_version": 4,
        "summary": {
            "roi_ann_pct": -2.5,
            "ann_roi_pct": -2.5,
            "deployment_pct": 1.0,
            "objective_score": None,
            "objective_valid": False,
            "evidence_health": "clean",
            "unattended_ready": False,
            "gap_to_hurdle_pp": 30.5,
        },
        "gate": {
            "freshness_status": {"status": "fresh", "age_minutes": 15.0, "expected_max_age_minutes": 1440.0},
            "warmup_state": {"posture": "expired", "deadline_utc": "2026-04-24T20:00:00Z", "matched_needed": 0},
            "trajectory_alarm": {"active": False},
        },
        "alpha_objective": {
            "roi_ann_pct": -2.5,
            "deployment_pct": 1.0,
            "objective_score": None,
            "objective_valid": False,
        },
        "source_contract": {
            "status": "clean",
            "canonical_sources": [],
            "allowlisted_readers": [],
            "violations_found": [],
            "scan_timestamp_utc": "2026-04-18T12:00:00Z",
        },
        "thin_linkage": {"matched_current": 10, "matched_needed": 0},
    }

    evidence_contract = mod._build_evidence_contract(
        canonical_snapshot=canonical_snapshot,
        metrics_summary={},
        evidence_status="CLEAN",
        evidence_warnings=[],
        evidence_gate_allowed=True,
        gate_lift_candidate=False,
    )

    assert evidence_contract["canonical_objective_valid"] is False
    assert evidence_contract["canonical_objective_score"] is None
    assert evidence_contract["gate_lift_candidate"] is False
    assert evidence_contract["canonical_objective_valid"] is False
