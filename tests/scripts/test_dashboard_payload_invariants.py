import json
from pathlib import Path

import pytest

from scripts.run_auto_trader import _emit_dashboard_json


def test_dashboard_payload_pnl_matches_equity(tmp_path: Path) -> None:
    path = tmp_path / "dashboard_data.json"

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

