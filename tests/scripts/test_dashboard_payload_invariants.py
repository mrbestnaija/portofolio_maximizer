import json
from pathlib import Path

import pytest

from scripts import dashboard_db_bridge as bridge_mod
from scripts import run_auto_trader
from scripts.run_auto_trader import _emit_dashboard_json


def test_dashboard_payload_pnl_matches_equity(tmp_path: Path) -> None:
    path = tmp_path / "run_auto_trader_latest.json"

    meta = {
        "run_id": "test_run",
        "ts": "2025-01-01T00:00:00Z",
        "tickers": ["AAPL"],
        "cycles": 2,
        "llm_enabled": False,
        "strategy": {},
    }

    # Initial capital 100.0 -> final equity 120.0 => PnL +20%, 0.2
    summary = {
        "cash": 0.0,
        "positions": {},
        "total_value": 120.0,
        "trades": 3,
        "pnl_dollars": 20.0,
        "pnl_pct": 0.2,
    }

    routing_stats = {
        "time_series_signals": 2,
        "llm_fallback_signals": 1,
    }

    equity_points = [
        {"t": "start", "v": 100.0},
        {"t": "cycle_1", "v": 110.0},
        {"t": "end", "v": 120.0},
    ]
    realized_equity_points = list(equity_points)

    latencies = {"ts_ms": 10.0, "llm_ms": 25.0}
    quality_summary = {
        "average": 0.9,
        "minimum": 0.8,
        "records": [
            {
                "ticker": "AAPL",
                "quality_score": 0.9,
                "missing_pct": 0.0,
                "coverage": 1.0,
                "outlier_frac": 0.0,
                "source": "test",
            }
        ],
    }

    forecaster_health = {
        "thresholds": {"profit_factor_min": 1.1},
        "metrics": {"profit_factor": 1.5},
        "status": {"profit_factor_ok": True},
    }
    orchestration_health = {
        "status": "WARN",
        "summary": {
            "snapshots": 1,
            "oos_source_counts": {"latest_metrics": 1},
            "mssa_white_noise_failures": 0,
            "garch_unstable_runs": 1,
            "rmse_rank_active_runs": 1,
            "allow_as_default_runs": 1,
        },
        "latest": {
            "oos_source": "latest_metrics",
            "oos_quality": "observed_holdout",
            "mssa_eligible": True,
            "mssa_white_noise": True,
            "garch_fallback_mode": "exploding_variance_ratio",
            "garch_unstable": True,
        },
        "issues": ["garch_unstable_reliance"],
    }
    quant_validation_health = {
        "total": 10,
        "pass_count": 8,
        "fail_count": 2,
        "fail_fraction": 0.2,
        "negative_expected_profit_fraction": 0.1,
    }

    _emit_dashboard_json(
        path=path,
        meta=meta,
        summary=summary,
        routing_stats=routing_stats,
        equity_points=equity_points,
        realized_equity_points=realized_equity_points,
        recent_signals=[],
        win_rate=0.6,
        latencies=latencies,
        quality_summary=quality_summary,
        forecaster_health=forecaster_health,
        quant_validation_health=quant_validation_health,
        orchestration_health=orchestration_health,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    # PnL block should mirror the summary.
    assert payload["pnl"]["absolute"] == pytest.approx(summary["pnl_dollars"])
    assert payload["pnl"]["pct"] == pytest.approx(summary["pnl_pct"])
    assert payload["trade_count"] == summary["trades"]

    # Equity curve should be arithmetically consistent with PnL.
    start_val = payload["equity"][0]["v"]
    end_val = payload["equity"][-1]["v"]
    assert end_val - start_val == pytest.approx(payload["pnl"]["absolute"])
    if start_val:
        assert (end_val / start_val) - 1.0 == pytest.approx(payload["pnl"]["pct"])

    # Routing, latency, and health blocks should be present and structurally intact.
    assert payload["routing"]["ts_signals"] == routing_stats["time_series_signals"]
    assert payload["routing"]["llm_signals"] == routing_stats["llm_fallback_signals"]
    assert "forecaster_health" in payload and payload["forecaster_health"]
    assert "quant_validation_health" in payload and payload["quant_validation_health"]
    assert "orchestration_health" in payload and payload["orchestration_health"]


def test_run_auto_trader_snapshot_path_is_separate_from_canonical_dashboard() -> None:
    assert run_auto_trader.RUN_AUTO_TRADER_ARTIFACT_PATH != bridge_mod.DEFAULT_OUTPUT_PATH
    assert run_auto_trader.RUN_AUTO_TRADER_ARTIFACT_PATH.name == "run_auto_trader_latest.json"


def test_dashboard_payload_serializes_nan_as_null(tmp_path: Path) -> None:
    path = tmp_path / "run_auto_trader_latest.json"

    _emit_dashboard_json(
        path=path,
        meta={"run_id": "nan_test", "ts": "2026-04-09T00:00:00Z", "tickers": ["AAPL"], "cycles": 1, "llm_enabled": False, "strategy": {}},
        summary={"cash": 100.0, "positions": {}, "total_value": float("nan"), "trades": 1, "pnl_dollars": float("nan"), "pnl_pct": float("nan")},
        routing_stats={"time_series_signals": 1, "llm_fallback_signals": 0},
        equity_points=[{"t": "end", "v": float("nan")}],
        recent_signals=[{"ticker": "AAPL", "entry_price": float("nan"), "portfolio_value": float("nan")}],
        trade_events=[{"ticker": "AAPL", "price": float("nan"), "slippage": float("nan")}],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pnl"]["absolute"] is None
    assert payload["pnl"]["pct"] is None
    assert payload["equity"][0]["v"] is None
    assert payload["signals"][0]["entry_price"] is None
    assert payload["trade_events"][0]["price"] is None


def test_build_action_plan_includes_orchestration_health_followups() -> None:
    from scripts.run_auto_trader import _build_action_plan

    actions = _build_action_plan(
        pnl_dollars=12.0,
        profit_factor=1.3,
        win_rate=0.55,
        realized_trades=4,
        cash_ratio=0.2,
        forecaster_health={"status": {}, "metrics": {}},
        quant_health={},
        orchestration_health={
            "status": "WARN",
            "summary": {
                "snapshots": 2,
                "oos_source_counts": {"heuristic_fallback": 1, "latest_metrics": 1},
                "mssa_white_noise_failures": 1,
                "garch_unstable_runs": 1,
                "rmse_rank_active_runs": 1,
                "allow_as_default_runs": 1,
            },
            "latest": {
                "oos_source": "heuristic_fallback",
                "oos_quality": "heuristic_fallback",
                "mssa_eligible": False,
                "mssa_white_noise": False,
                "mssa_policy_status": "ready",
                "garch_fallback_mode": "exploding_variance_ratio",
                "garch_unstable": True,
            },
        },
    )

    assert any("Refresh trailing OOS evidence" in action for action in actions)
    assert any("Keep MSSA-RL containment-only" in action for action in actions)
    assert any("Reduce GARCH reliance" in action for action in actions)
