from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts.run_auto_trader import (
    _build_execution_policy_metrics,
    _prepare_execution_candidate,
    _rank_prepared_execution_candidates,
)


def test_build_execution_policy_metrics_passes_positive_net_edge_after_uncertainty() -> None:
    policy = _build_execution_policy_metrics(
        {
            "action": "BUY",
            "expected_return": 0.02,
            "expected_return_net": 0.015,
            "net_trade_return": 0.015,
            "signal_to_noise": 3.0,
            "roundtrip_cost_fraction": 0.005,
        },
        current_position=0,
        proof_mode=False,
        llm_primary_takeover=False,
    )

    assert policy["blocked"] is False
    assert policy["entry_like"] is True
    assert policy["edge_over_risk_score"] == pytest.approx(2.25)
    assert policy["net_edge_after_uncertainty"] == pytest.approx(0.011666666666666665)


def test_build_execution_policy_metrics_blocks_llm_fallback_without_explicit_metrics() -> None:
    policy = _build_execution_policy_metrics(
        {
            "action": "BUY",
        },
        current_position=0,
        proof_mode=False,
        llm_primary_takeover=True,
    )

    assert policy["blocked"] is True
    assert "LLM_FALLBACK_MISSING_EDGE_METRICS" in policy["reason_codes"]
    assert "LLM_FALLBACK_MISSING_UNCERTAINTY" in policy["reason_codes"]


def test_prepare_execution_candidate_promotes_routing_truth_and_nested_metrics() -> None:
    router = MagicMock()
    router.route_signal.return_value = SimpleNamespace(
        primary_signal={
            "signal_id": "sig-1",
            "ts_signal_id": "sig-1",
            "source": "LLM",
            "action": "BUY",
            "confidence": 0.82,
            "expected_return": 0.02,
            "entry_price": 100.0,
            "lower_ci": 99.0,
            "upper_ci": 103.0,
            "provenance": {
                "decision_context": {
                    "expected_return_net": 0.015,
                    "net_trade_return": 0.015,
                    "roundtrip_cost_fraction": 0.005,
                    "signal_to_noise": 3.0,
                }
            },
        },
        metadata={
            "primary_source": "LLM",
            "llm_primary_takeover": True,
            "fallback_trigger": "TS_UNAVAILABLE",
        },
    )
    market_data = pd.DataFrame(
        {
            "High": [101.0] * 20,
            "Low": [99.0] * 20,
            "Close": [100.0] * 20,
        },
        index=pd.date_range("2026-01-01", periods=20, freq="D"),
    )

    prepared = _prepare_execution_candidate(
        router=router,
        ticker="AAPL",
        forecast_bundle={"horizon": 5},
        current_price=100.0,
        market_data=market_data,
        quality={"quality_score": 0.9},
        data_source="yfinance",
        mid_price=100.0,
        run_id="run-1",
        execution_mode="live",
        proof_mode=False,
        current_position=0,
    )

    assert prepared is not None
    payload = prepared["primary_payload"]
    assert payload["expected_return_net"] == pytest.approx(0.015)
    assert payload["net_trade_return"] == pytest.approx(0.015)
    assert payload["execution_source"] == "LLM"
    assert payload["llm_primary_takeover"] is True
    assert payload["fallback_reason"] == "TS_UNAVAILABLE"
    assert payload["execution_policy_blocked"] is False


def test_rank_prepared_execution_candidates_blocks_entries_outside_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMX_EXECUTION_TOP_K", "1")
    monkeypatch.setenv("PMX_MIN_EDGE_OVER_RISK_SCORE", "0.0")
    prepared = [
        {
            "ticker": "AAPL",
            "policy": {"entry_like": True, "edge_over_risk_score": 2.0, "net_trade_return": 0.02},
            "primary_payload": {"confidence": 0.80, "action": "BUY"},
        },
        {
            "ticker": "MSFT",
            "policy": {"entry_like": True, "edge_over_risk_score": 1.0, "net_trade_return": 0.01},
            "primary_payload": {"confidence": 0.70, "action": "BUY"},
        },
    ]

    ranked = _rank_prepared_execution_candidates(prepared, proof_mode=False)

    assert ranked[0]["primary_payload"]["execution_rank"] == 1
    assert ranked[0]["primary_payload"].get("execution_policy_blocked") is not True
    assert ranked[1]["primary_payload"]["execution_rank"] == 2
    assert ranked[1]["primary_payload"]["execution_policy_blocked"] is True
    assert ranked[1]["primary_payload"]["execution_policy_reason_code"] == "EDGE_RANK_OUTSIDE_TOP_K"
