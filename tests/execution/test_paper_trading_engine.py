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


def test_regime_state_risk_multiplier_scales_position_size(tmp_path, monkeypatch):
    """Risk multiplier from regime_state.yml should affect max position value."""
    # Create an in-memory DB and engine
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    # Write a temporary regime_state.yml marking AAPL as exploration (0.25x risk)
    regime_payload = {
        "regime_state": {
            "AAPL": {"n_trades": 5, "sharpe_N": None, "mode": "exploration", "state": "neutral"}
        }
    }
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    regime_file = cfg_dir / "regime_state.yml"
    import yaml

    regime_file.write_text(yaml.safe_dump(regime_payload, sort_keys=False), encoding="utf-8")

    # Point the engine helper to the temp config dir by monkeypatching Path resolution
    import execution.paper_trading_engine as pte
    original_get_regime = engine._get_regime_risk_multiplier

    def _patched_get_regime_risk_multiplier(ticker: str) -> float:
        # Mimic logic but read from our temp file
        raw = yaml.safe_load(regime_file.read_text(encoding="utf-8")) or {}
        rs = raw.get("regime_state") or {}
        info = rs.get(ticker)
        if not isinstance(info, dict):
            return 1.0
        mode = info.get("mode")
        state = info.get("state")
        if mode == "exploration":
            return 0.25
        if state == "red":
            return 0.3
        if state == "green":
            return 1.2
        return 1.0

    engine._get_regime_risk_multiplier = _patched_get_regime_risk_multiplier  # type: ignore[assignment]

    signal = {"ticker": "AAPL", "action": "BUY", "confidence": 0.8}
    market_data = make_market_data(100.0)

    # Capture position size under exploration (0.25x)
    pos_size_exploration = engine._calculate_position_size(
        signal, confidence_score=0.9, market_data=market_data, current_position=0
    )

    # Now simulate neutral regime (1.0x)
    regime_payload["regime_state"]["AAPL"]["mode"] = "exploitation"
    regime_payload["regime_state"]["AAPL"]["state"] = "neutral"
    regime_file.write_text(yaml.safe_dump(regime_payload, sort_keys=False), encoding="utf-8")

    pos_size_neutral = engine._calculate_position_size(
        signal, confidence_score=0.9, market_data=market_data, current_position=0
    )

    assert pos_size_exploration < pos_size_neutral

    db.close()
