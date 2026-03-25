from __future__ import annotations

import json

import scripts.llm_multi_model_orchestrator as orch


def test_read_gate_status_reports_no_data_when_latest_run_has_no_gate_logs(tmp_path, monkeypatch) -> None:
    latest = tmp_path / "logs" / "audit_sprint" / "2026-03-25T00-00-00Z"
    latest.mkdir(parents=True)
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    payload = json.loads(orch._read_gate_status("all"))

    assert payload["status"] == "NO_DATA"
    assert payload["no_data_reason"] == "missing_gate_logs"


def test_validate_trade_metrics_summary_marks_no_round_trip_data_limited() -> None:
    raw = json.dumps(
        {
            "total_pnl": "$+0.00",
            "win_rate": "0.0%",
            "profit_factor": "0.00",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
        }
    )

    validated = json.loads(orch._validate_tool_result("query_trade_metrics", {"metric": "summary"}, raw))

    assert validated["status"] == "LIMITED"
    assert validated["no_data_reason"] == "no_round_trip_data"


def test_validate_trade_metrics_summary_fails_closed_on_out_of_range_values() -> None:
    raw = json.dumps(
        {
            "total_pnl": "$+15.00",
            "win_rate": "140.0%",
            "profit_factor": "1.20",
            "total_trades": 3,
            "wins": 4,
            "losses": 0,
        }
    )

    validated = json.loads(orch._validate_tool_result("query_trade_metrics", {"metric": "summary"}, raw))

    assert validated["status"] == "FAIL"
    assert validated["validation_code"] == "trade_metrics_out_of_range"


def test_validate_search_web_tavily_marks_empty_success_payload_as_no_data() -> None:
    raw = json.dumps(
        {
            "status": "PASS",
            "provider": "tavily",
            "attempts": [{"provider": "tavily"}],
            "answer": "",
            "results": [],
            "latency_seconds": 0.42,
        }
    )

    validated = json.loads(orch._validate_tool_result("search_web_tavily", {"query": "gate status"}, raw))

    assert validated["status"] == "NO_DATA"
    assert validated["no_data_reason"] == "empty_search_result"


def test_validate_quant_validation_headroom_fails_on_impossible_percentages() -> None:
    raw = json.dumps(
        {
            "status": "PASS",
            "summary": {
                "fail_count": 5,
                "total": 2,
                "fail_rate_pct": 250.0,
                "red_gate_pct": 95.0,
                "warn_gate_pct": 90.0,
                "headroom_to_red_gate_pct": -155.0,
                "per_ticker": [{"ticker": "AAPL", "fail_count": 5, "total": 2, "fail_rate_pct": 250.0}],
            },
        }
    )

    validated = json.loads(
        orch._validate_tool_result("check_quant_validation_headroom", {}, raw)
    )

    assert validated["status"] == "FAIL"
    assert validated["validation_code"] == "quant_validation_out_of_range"


def test_tool_result_guidance_tells_model_not_to_infer_when_data_is_missing() -> None:
    guidance = orch._tool_result_guidance_message(
        json.dumps({"status": "NO_DATA", "message": "Search completed but returned no answer or search results."})
    )

    assert "Do not infer unavailable values" in guidance
