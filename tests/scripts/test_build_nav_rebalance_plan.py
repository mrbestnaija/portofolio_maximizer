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


def test_shadow_first_plan_demotes_weak_and_concentrates_healthy(tmp_path: Path) -> None:
    eligibility_path = tmp_path / "logs" / "ticker_eligibility.json"
    eligibility_gates_path = tmp_path / "logs" / "ticker_eligibility_gates.json"
    sleeve_summary_path = tmp_path / "logs" / "automation" / "sleeve_summary.json"
    sleeve_plan_path = tmp_path / "logs" / "automation" / "sleeve_promotion_plan.json"
    metrics_summary_path = tmp_path / "visualizations" / "performance" / "metrics_summary.json"
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
                    "total_pnl": 706.97,
                    "reasons": ["meets_r3_thresholds(n>=20, wr>=0.45, pf>=1.30)"],
                },
                "MSFT": {
                    "status": "HEALTHY",
                    "n_trades": 6,
                    "win_rate": 0.6667,
                    "profit_factor": 2.2562,
                    "total_pnl": 188.21,
                    "reasons": ["meets_r3_thresholds(n>=20, wr>=0.45, pf>=1.30)"],
                },
                "GOOG": {
                    "status": "HEALTHY",
                    "n_trades": 10,
                    "win_rate": 0.5,
                    "profit_factor": 2.4283,
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
    _write_json(
        metrics_summary_path,
        {
            "generated_utc": "2026-04-18T12:00:00Z",
            "status": "WARN",
            "sufficiency_status": "INSUFFICIENT",
            "coverage_ratio": 0.139,
            "warnings": ["context_partial_data", "sufficiency_not_green"],
        },
    )
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
            "--metrics-summary-path",
            str(metrics_summary_path),
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
    assert payload["rollout"]["gate_lift_candidate"] is False
    assert "shadow_first_default" in payload["rollout"]["live_apply_blockers"]
    assert "evidence_not_green" in payload["rollout"]["live_apply_blockers"]
    assert payload["rollout"]["evidence_status"] == "INSUFFICIENT"

    target_map = {row["ticker"]: row for row in payload["targets"]}
    assert target_map["AAPL"]["action"] == "DEMOTE"
    assert target_map["GS"]["action"] == "DEMOTE"
    assert target_map["NVDA"]["action"] == "PROMOTE"
    assert target_map["MSFT"]["action"] == "PROMOTE"
    assert target_map["GOOG"]["action"] == "PROMOTE"
    assert target_map["AAPL"]["target_nav_frac"] == 0.0
    assert target_map["GS"]["target_nav_frac"] == 0.0
    assert target_map["NVDA"]["target_nav_frac"] > target_map["MSFT"]["target_nav_frac"] > target_map["GOOG"]["target_nav_frac"]

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
        "metrics_summary_path": tmp_path / "visualizations" / "performance" / "metrics_summary.json",
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
    _write_json(
        common_files["metrics_summary_path"],
        {"generated_utc": "2026-04-18T12:00:00Z", "status": "PASS", "sufficiency_status": "SUFFICIENT", "warnings": []},
    )

    first = mod.build_nav_rebalance_plan(
        eligibility_path=common_files["eligibility_path"],
        eligibility_gates_path=common_files["eligibility_gates_path"],
        sleeve_summary_path=common_files["sleeve_summary_path"],
        sleeve_plan_path=common_files["sleeve_plan_path"],
        metrics_summary_path=common_files["metrics_summary_path"],
        risk_buckets_path=common_files["risk_buckets_path"],
        output_path=tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest.json",
    )
    second = mod.build_nav_rebalance_plan(
        eligibility_path=common_files["eligibility_path"],
        eligibility_gates_path=common_files["eligibility_gates_path"],
        sleeve_summary_path=common_files["sleeve_summary_path"],
        sleeve_plan_path=common_files["sleeve_plan_path"],
        metrics_summary_path=common_files["metrics_summary_path"],
        risk_buckets_path=common_files["risk_buckets_path"],
        output_path=tmp_path / "logs" / "automation" / "nav_rebalance_plan_latest_2.json",
    )

    assert first["summary"]["allocated_symbol_nav_frac"] == pytest.approx(second["summary"]["allocated_symbol_nav_frac"])
    assert first["targets"] == second["targets"]
    assert first["rollout"]["gate_lift_candidate"] is True
