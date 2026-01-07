"""Tests for the order management system."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

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


def test_order_manager_persists_provenance_and_mid_slippage():
    manager, client, db = _build_manager()
    signal = {
        "ticker": "MSFT",
        "action": "BUY",
        "confidence_score": 0.9,
        "current_price": 100.0,
        "signal_id": 99,
        "data_source": "synthetic",
        "execution_mode": "synthetic",
        "synthetic_dataset_id": "syn_test",
        "synthetic_generator_version": "v1",
        "run_id": "run_prov",
    }

    market_data = pd.DataFrame({"Bid": [99.0], "Ask": [101.0], "Close": [100.0]})
    result = manager.submit_signal(signal, market_data=market_data)

    assert result.status == "EXECUTED"
    call = db.calls[-1]
    assert call["data_source"] == "synthetic"
    assert call["execution_mode"] == "synthetic"
    assert call["synthetic_dataset_id"] == "syn_test"
    assert call["synthetic_generator_version"] == "v1"
    assert call["run_id"] == "run_prov"
    assert call["mid_price"] == 100.0
    assert call["mid_slippage_bps"] is not None


def test_order_manager_prefers_broker_mid_slippage():
    class _FakeClientWithMid(_FakeCTraderClient):
        def place_order(self, order: CTraderOrder) -> OrderPlacement:
            self.placed_orders.append(order)
            return OrderPlacement(
                order_id="ORDER999",
                status="FILLED",
                filled_volume=order.volume,
                avg_price=101.0,
                submitted_at=datetime.utcnow(),
                mid_price=100.0,
                mid_slippage_bps=10.0,
                raw_response={},
            )

    client = _FakeClientWithMid()
    db = _FakeDB()
    manager = OrderManager(
        mode="demo",
        client=client,
        database_manager=db,
        risk_manager=_FakeRiskManager(),
    )
    signal = {"ticker": "AAPL", "action": "BUY", "confidence_score": 0.9, "current_price": 100.0}
    result = manager.submit_signal(signal)
    assert result.status == "EXECUTED"
    call = db.calls[-1]
    assert call["mid_price"] == 100.0
    assert call["mid_slippage_bps"] == 10.0
