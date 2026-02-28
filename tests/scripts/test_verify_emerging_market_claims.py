from __future__ import annotations

import json

from scripts.verify_emerging_market_claims import audit_claims, main


def test_audit_claims_reports_expected_repo_statuses() -> None:
    payload = audit_claims()
    claims = payload["claims"]
    summary = payload["summary"]

    assert claims["frontier_market_data"]["status"] == "implemented"
    assert claims["commodity_fx_execution"]["status"] == "implemented"
    assert claims["emerging_market_equity_execution"]["status"] == "partial"
    assert "no XTB execution adapter" in claims["emerging_market_equity_execution"]["summary"]
    assert claims["sentiment_news_monitoring"]["status"] == "dormant"
    assert claims["weather_risk_overlay"]["status"] == "implemented"
    assert any("utils/weather_context.py" in item for item in claims["weather_risk_overlay"]["evidence"])
    assert any("execution/paper_trading_engine.py" in item for item in claims["weather_risk_overlay"]["evidence"])
    assert claims["backtesting_and_optimization"]["status"] == "implemented"
    assert claims["liquidity_and_slippage_controls"]["status"] == "implemented"
    assert claims["geopolitical_policy_monitoring"]["status"] == "unsupported"
    assert claims["cross_border_arbitrage"]["status"] == "unsupported"
    assert claims["esg_screening"]["status"] == "unsupported"
    assert summary["implemented"] == 5
    assert summary["partial"] == 1


def test_main_json_emits_machine_readable_payload(capsys) -> None:
    exit_code = main(["--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "claims" in payload
    assert "summary" in payload
    assert payload["summary"]["implemented"] >= 1
    assert payload["claims"]["emerging_market_equity_execution"]["status"] == "partial"
