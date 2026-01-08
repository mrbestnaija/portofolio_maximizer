"""Tests for the cTrader Open API client."""

from __future__ import annotations

from typing import Any, Dict, Optional

import json

import pytest

from execution.ctrader_client import (
    CTraderClient,
    CTraderClientConfig,
    CTraderOrder,
    CTraderOrderError,
    CTraderAuthError,
)


class _StubResponse:
    def __init__(self, *, status: int = 200, payload: Optional[Dict[str, Any]] = None):
        self.status_code = status
        self._payload = payload or {}
        self.text = json.dumps(self._payload)
        self.content = self.text.encode()

    def json(self) -> Dict[str, Any]:
        return self._payload


class _StubSession:
    def __init__(self):
        self.post_calls: list[Dict[str, Any]] = []
        self.request_calls: list[Dict[str, Any]] = []
        self.headers: Dict[str, Any] = {}

    def post(self, url: str, data: Dict[str, Any], timeout=None):  # noqa: D401
        self.post_calls.append({"url": url, "data": data, "timeout": timeout})
        if "connect/token" in url:
            return _StubResponse(
                payload={
                    "access_token": "demo-token",
                    "refresh_token": "refresh",
                    "expires_in": 3600,
                }
            )
        return _StubResponse()

    def request(self, method: str, url: str, headers=None, params=None, json=None, timeout=None):
        self.request_calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "json": json,
                "timeout": timeout,
            }
        )
        return _StubResponse(
            payload={
                "orderId": "ABC123",
                "status": "FILLED",
                "filledVolume": json.get("volume", 0),
                "averagePrice": 101.25,
            }
        )

    def close(self):  # pragma: no cover - not used in tests
        pass


def test_config_from_env_reads_required_values(monkeypatch):
    monkeypatch.setenv("USERNAME_CTRADER", "demo_user")
    monkeypatch.setenv("PASSWORD_CTRADER", "demo_pass")
    monkeypatch.setenv("APPLICATION_NAME_CTRADER", "demo_app")
    monkeypatch.setenv("CTRADER_ACCOUNT_ID", "123456")

    config = CTraderClientConfig.from_env(environment="demo")

    assert config.username == "demo_user"
    assert config.password == "demo_pass"
    assert config.application_id == "demo_app"
    assert config.account_id == 123456
    assert config.is_demo is True


def test_config_from_env_raises_when_missing(monkeypatch):
    for key in ["USERNAME_CTRADER", "PASSWORD_CTRADER", "APPLICATION_NAME_CTRADER", "EMAIL_CTRADER"]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr("execution.ctrader_client._load_env_pair", lambda: {})

    with pytest.raises(CTraderAuthError):
        CTraderClientConfig.from_env(environment="demo")


def test_place_order_uses_bearer_token(monkeypatch):
    config = CTraderClientConfig(
        username="demo_user",
        password="demo_pass",
        application_id="demo_app",
        account_id=987654,
        environment="demo",
    )
    session = _StubSession()
    client = CTraderClient(config=config, session=session)

    placement = client.place_order(
        CTraderOrder(symbol="AAPL", side="BUY", volume=2, order_type="MARKET")
    )

    assert placement.order_id == "ABC123"
    assert placement.filled_volume == 2
    assert placement.avg_price == 101.25

    assert session.post_calls, "Expected authentication call"
    request_call = session.request_calls[0]
    assert request_call["headers"]["Authorization"].startswith("Bearer ")
    assert request_call["json"]["symbolName"] == "AAPL"


def test_place_order_requires_account_id():
    config = CTraderClientConfig(
        username="demo_user",
        password="demo_pass",
        application_id="demo_app",
        account_id=None,
    )
    client = CTraderClient(config=config, session=_StubSession())

    with pytest.raises(CTraderOrderError):
        client.place_order(CTraderOrder(symbol="AAPL", side="BUY", volume=1))
