"""Tests for the order management system."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from execution.ctrader_client import (
    AccountSnapshot,
    CTraderClientConfig,
    CTraderOrder,
    OrderPlacement,
)
from execution.order_manager import OrderManager
from risk.real_time_risk_manager import RiskReport


class _FakeCTraderClient:
    def __init__(self):
        self.config = CTraderClientConfig(
            username="demo",
            password="pass",
            application_id="app",
            account_id=1111,
        )
        self._snapshot = AccountSnapshot(
            balance=10000.0,
            equity=10000.0,
            free_margin=9500.0,
            currency="USD",
        )
        self.placed_orders: List[CTraderOrder] = []

    def get_account_overview(self) -> AccountSnapshot:
        return self._snapshot

    def get_positions(self) -> Dict[str, Any]:
        return {"positions": []}

    def place_order(self, order: CTraderOrder) -> OrderPlacement:
        self.placed_orders.append(order)
        return OrderPlacement(
            order_id="ORDER123",
            status="FILLED",
            filled_volume=order.volume,
            avg_price=order.limit_price or 100.0,
            submitted_at=datetime.utcnow(),
            raw_response={},
        )


class _FakeRiskManager:
    def __init__(self, status: str = "HEALTHY") -> None:
        self.status = status

    def monitor_portfolio_risk(self, portfolio_value, positions, position_prices) -> RiskReport:
        return RiskReport(
            current_drawdown=0.0,
            volatility=0.1,
            var_95=0.02,
            portfolio_value=portfolio_value,
            alerts=[],
            status=self.status,
        )


class _FakeDB:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def save_trade_execution(self, **kwargs):
        self.calls.append(kwargs)
        return 1


def _build_manager(status: str = "HEALTHY") -> tuple[OrderManager, _FakeCTraderClient, _FakeDB]:
    client = _FakeCTraderClient()
    db = _FakeDB()
    risk = _FakeRiskManager(status=status)
    manager = OrderManager(
        mode="demo",
        client=client,
        database_manager=db,
        risk_manager=risk,
        max_position_risk=0.02,
        min_confidence=0.6,
        max_trades_per_day=10,
    )
    return manager, client, db


def test_order_manager_rejects_low_confidence_signal():
    manager, client, _ = _build_manager()
    signal = {
        "ticker": "AAPL",
        "action": "BUY",
        "confidence_score": 0.2,
        "current_price": 100.0,
    }

    result = manager.submit_signal(signal)

    assert result.status == "REJECTED"
    assert not client.placed_orders
    assert "confidence" in result.pre_trade.reasons


def test_order_manager_executes_and_persists_trade():
    manager, client, db = _build_manager()
    signal = {
        "ticker": "MSFT",
        "action": "BUY",
        "confidence_score": 0.85,
        "current_price": 320.0,
        "signal_id": 42,
    }

    result = manager.submit_signal(signal)

    assert result.status == "EXECUTED"
    assert client.placed_orders, "Expected order to reach broker client"
    assert len(db.calls) == 1
    saved = db.calls[0]
    assert saved["ticker"] == "MSFT"
    assert saved["signal_id"] == 42


def test_order_manager_blocks_when_risk_status_not_healthy():
    manager, client, _ = _build_manager(status="CRITICAL")
    signal = {
        "ticker": "TSLA",
        "action": "SELL",
        "confidence_score": 0.9,
        "current_price": 250.0,
    }

    result = manager.submit_signal(signal)

    assert result.status == "REJECTED"
    assert "risk_status" in result.pre_trade.reasons
    assert not client.placed_orders
