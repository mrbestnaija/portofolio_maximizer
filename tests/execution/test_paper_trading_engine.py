from datetime import datetime
from typing import List

import pandas as pd

from execution.paper_trading_engine import ExecutionResult, PaperTradingEngine, Trade
from etl.database_manager import DatabaseManager


class DummyValidationResult:
    def __init__(self, is_valid: bool, recommendation: str, confidence_score: float, warnings: List[str] = None):
        self.is_valid = is_valid
        self.recommendation = recommendation
        self.confidence_score = confidence_score
        self.warnings = warnings or []


class DummyValidator:
    def __init__(self, result: DummyValidationResult):
        self._result = result

    def validate_llm_signal(self, signal, market_data, portfolio_value):
        return self._result


def make_market_data(close_price: float = 100.0) -> pd.DataFrame:
    prices = [close_price * (1 + 0.001 * i) for i in range(30)]
    return pd.DataFrame({"Close": prices})


def test_execute_signal_rejected_when_validation_fails():
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(False, "REJECT", 0.2, ["low_confidence"]))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    signal = {"ticker": "AAPL", "action": "BUY", "confidence": 0.4}
    result = engine.execute_signal(signal, make_market_data())

    assert isinstance(result, ExecutionResult)
    assert result.status == "REJECTED"
    assert "Validation failed" in (result.reason or "")

    db.close()


def test_execute_signal_executes_and_persists_trade():
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    signal = {"ticker": "MSFT", "action": "BUY", "confidence": 0.8, "signal_id": 42}
    market_data = make_market_data(120.0)
    result = engine.execute_signal(signal, market_data)

    assert result.status == "EXECUTED"
    assert isinstance(result.trade, Trade)
    assert result.trade.ticker == "MSFT"
    assert len(engine.trades) == 1
    assert engine.portfolio.cash < 10_000.0

    row = db.cursor.execute("SELECT ticker, action, commission, signal_id FROM trade_executions").fetchone()
    assert row["ticker"] == "MSFT"
    assert row["action"] == "BUY"
    assert row["signal_id"] == 42

    db.close()
