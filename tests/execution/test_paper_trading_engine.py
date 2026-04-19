from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execution.paper_trading_engine import ExecutionResult, PaperTradingEngine, Trade
from etl.database_manager import DatabaseManager
from execution.lob_simulator import LOBConfig


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


def make_microstructure_market_data(
    *,
    mid_price: float = 100.0,
    half_spread: float = 0.10,
    depth_notional: float = 200.0,
    txn_cost_bps: float = 10.0,
    points: int = 5,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [mid_price] * points,
            "Spread": [half_spread] * points,
            "Depth": [depth_notional] * points,
            "TxnCostBps": [txn_cost_bps] * points,
        }
    )


def make_indexed_market_data(close_price: float, ts: datetime) -> pd.DataFrame:
    return pd.DataFrame({"Close": [close_price]}, index=pd.DatetimeIndex([ts]))


def make_indexed_market_data_with_trailing_nan(close_price: float, ts_valid: datetime, ts_nan: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [close_price, float("nan")]},
        index=pd.DatetimeIndex([ts_valid, ts_nan]),
    )


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


def test_execute_signal_uses_last_valid_market_row_when_terminal_close_is_nan():
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    signal = {"ticker": "MSFT", "action": "BUY", "confidence": 0.8}
    market_data = make_indexed_market_data_with_trailing_nan(
        120.0,
        datetime(2026, 4, 8, tzinfo=timezone.utc),
        datetime(2026, 4, 9, tzinfo=timezone.utc),
    )
    result = engine.execute_signal(signal, market_data)

    assert result.status == "EXECUTED"
    assert result.trade is not None
    assert result.trade.entry_price == pytest.approx(120.0, rel=1e-3)
    assert result.trade.bar_timestamp == datetime(2026, 4, 8, tzinfo=timezone.utc)

    db.close()


def test_execute_signal_applies_high_snr_holding_override(monkeypatch):
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )
    monkeypatch.setenv("MAX_HOLDING_DAYS_CAP", "10")

    signal = {
        "ticker": "MSFT",
        "action": "BUY",
        "confidence": 0.8,
        "forecast_horizon": 30,
        "max_holding_days_override": 15,
    }
    result = engine.execute_signal(signal, make_market_data(120.0))

    assert result.status == "EXECUTED"
    assert engine.portfolio.max_holding_days["MSFT"] == 15

    db.close()


def test_execute_signal_uses_default_holding_cap_without_override(monkeypatch):
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )
    monkeypatch.setenv("MAX_HOLDING_DAYS_CAP", "10")

    signal = {
        "ticker": "MSFT",
        "action": "BUY",
        "confidence": 0.8,
        "forecast_horizon": 30,
    }
    result = engine.execute_signal(signal, make_market_data(120.0))

    assert result.status == "EXECUTED"
    assert engine.portfolio.max_holding_days["MSFT"] == 10

    db.close()


def test_edge_cost_gate_uses_nested_expected_return_net(monkeypatch):
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )
    monkeypatch.setenv("PMX_EDGE_COST_GATE", "1")

    signal = {
        "ticker": "AAPL",
        "action": "BUY",
        "confidence": 0.8,
        "expected_return": 0.02,
        "provenance": {
            "decision_context": {
                "expected_return_net": 0.0,
            }
        },
    }

    result = engine.execute_signal(signal, make_market_data())

    assert result.status == "REJECTED"
    assert "Edge/cost gate" in (result.reason or "")
    assert len(engine.trades) == 0

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


def test_candidate_sizing_knobs_reduce_new_exposure_but_not_exits():
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=100_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )
    engine.portfolio.total_value = 100_000.0

    market_data = make_market_data(100.0)
    base_open = engine._calculate_position_size(
        {"ticker": "AAPL", "action": "BUY", "confidence": 0.9},
        confidence_score=0.9,
        market_data=market_data,
        current_position=0,
    )
    capped_open = engine._calculate_position_size(
        {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 0.9,
            "sizing_kelly_fraction_cap": 0.25,
        },
        confidence_score=0.9,
        market_data=market_data,
        current_position=0,
    )
    assert capped_open < base_open

    baseline_add = engine._calculate_position_size(
        {"ticker": "AAPL", "action": "BUY", "confidence": 0.9},
        confidence_score=0.9,
        market_data=market_data,
        current_position=500,
    )
    penalized_add = engine._calculate_position_size(
        {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 0.9,
            "diversification_penalty": 1.0,
        },
        confidence_score=0.9,
        market_data=market_data,
        current_position=500,
    )
    assert penalized_add < baseline_add

    baseline_exit = engine._calculate_position_size(
        {"ticker": "AAPL", "action": "SELL", "confidence": 0.9},
        confidence_score=0.9,
        market_data=market_data,
        current_position=500,
    )
    penalized_exit = engine._calculate_position_size(
        {
            "ticker": "AAPL",
            "action": "SELL",
            "confidence": 0.9,
            "sizing_kelly_fraction_cap": 0.25,
            "diversification_penalty": 1.0,
        },
        confidence_score=0.9,
        market_data=market_data,
        current_position=500,
    )
    assert penalized_exit == baseline_exit

    db.close()


def test_lob_execution_price_moves_with_order_size():
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=100_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    market_data = make_microstructure_market_data()
    buy_small = engine._simulate_entry_price(100.0, "BUY", 1, market_data=market_data)
    buy_large = engine._simulate_entry_price(100.0, "BUY", 25, market_data=market_data)
    assert buy_large >= buy_small
    sell_small = engine._simulate_entry_price(100.0, "SELL", 1, market_data=market_data)
    sell_large = engine._simulate_entry_price(100.0, "SELL", 25, market_data=market_data)
    assert sell_large <= sell_small

    db.close()


def test_execute_signal_marks_returned_trade_contaminated_when_closing_synthetic_opener():
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    opener_id = db.save_trade_execution(
        ticker="AAPL",
        trade_date=datetime(2026, 4, 8, tzinfo=timezone.utc).date(),
        action="BUY",
        shares=1,
        price=100.0,
        total_value=100.0,
        commission=0.0,
        execution_mode="synthetic",
        is_synthetic=1,
    )
    engine.portfolio.positions["AAPL"] = 1
    engine.portfolio.entry_prices["AAPL"] = 100.0
    engine.portfolio.entry_timestamps["AAPL"] = datetime(2026, 4, 8, tzinfo=timezone.utc)
    engine.portfolio.entry_trade_ids["AAPL"] = opener_id
    engine.portfolio.total_value = 10_000.0

    result = engine.execute_signal(
        {"ticker": "AAPL", "action": "SELL", "confidence": 0.9, "execution_mode": "live"},
        make_market_data(110.0),
        proof_mode=True,
    )

    assert result.status == "EXECUTED"
    assert result.trade is not None
    assert result.trade.is_contaminated == 1

    row = db.cursor.execute(
        "SELECT COALESCE(is_contaminated, 0) AS is_contaminated "
        "FROM trade_executions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert int(row["is_contaminated"]) == 1

    db.close()


def test_execute_signal_persists_multi_lot_close_allocations_for_same_ticker():
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    opener_1 = db.save_trade_execution(
        ticker="NVDA",
        trade_date=datetime(2026, 4, 2, tzinfo=timezone.utc).date(),
        action="BUY",
        shares=1.0,
        price=175.0,
        total_value=175.0,
        commission=0.0,
        execution_mode="live",
    )
    opener_2 = db.save_trade_execution(
        ticker="NVDA",
        trade_date=datetime(2026, 4, 2, tzinfo=timezone.utc).date(),
        action="BUY",
        shares=1.0,
        price=177.0,
        total_value=177.0,
        commission=0.0,
        execution_mode="live",
    )
    engine.portfolio.positions["NVDA"] = 2
    engine.portfolio.entry_prices["NVDA"] = 176.0
    engine.portfolio.entry_timestamps["NVDA"] = datetime(2026, 4, 2, tzinfo=timezone.utc)
    engine.portfolio.entry_lots["NVDA"] = [
        {"trade_id": opener_1, "action": "BUY", "remaining_shares": 1.0, "is_synthetic": 0},
        {"trade_id": opener_2, "action": "BUY", "remaining_shares": 1.0, "is_synthetic": 0},
    ]
    engine._sync_entry_trade_id_map(engine.portfolio)
    engine._calculate_position_size = lambda signal, confidence_score, market_data, current_position: 2  # type: ignore[method-assign]

    close_signal = {"ticker": "NVDA", "action": "SELL", "confidence": 0.9, "execution_mode": "live"}
    close_result = engine.execute_signal(close_signal, make_market_data(165.0), proof_mode=True)

    assert close_result.status == "EXECUTED"
    assert engine.portfolio.positions.get("NVDA", 0) == 0
    assert "NVDA" not in engine.portfolio.entry_lots

    close_row = db.cursor.execute(
        "SELECT id, entry_trade_id, is_close, close_size FROM trade_executions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert close_row is not None
    assert int(close_row["is_close"]) == 1
    assert close_row["entry_trade_id"] is not None
    assert abs(float(close_row["close_size"]) - 2.0) < 1e-9

    allocation_rows = db.cursor.execute(
        """
        SELECT entry_trade_id, allocated_shares
        FROM trade_close_allocations
        WHERE close_trade_id = ?
        ORDER BY entry_trade_id
        """,
        (int(close_row["id"]),),
    ).fetchall()
    assert len(allocation_rows) == 2
    assert abs(sum(float(row["allocated_shares"]) for row in allocation_rows) - 2.0) < 1e-9

    db.close()


def test_execute_signal_reverse_through_flat_persists_residual_open_lot_and_resume_linkage(tmp_path):
    db_path = tmp_path / "reverse_through_flat.db"
    db = DatabaseManager(str(db_path))
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )
    engine._calculate_position_size = lambda signal, confidence_score, market_data, current_position: 2 if str(signal.get("action") or "").upper() == "BUY" else 5  # type: ignore[method-assign]

    entry_result = engine.execute_signal(
        {"ticker": "AAPL", "action": "BUY", "confidence": 0.9, "execution_mode": "live"},
        make_market_data(100.0),
    )
    assert entry_result.status == "EXECUTED"

    reverse_result = engine.execute_signal(
        {"ticker": "AAPL", "action": "SELL", "confidence": 0.9, "execution_mode": "live"},
        make_market_data(110.0),
    )
    assert reverse_result.status == "EXECUTED"
    assert engine.portfolio.positions.get("AAPL") == -3
    assert "AAPL" in engine.portfolio.entry_lots
    assert len(engine.portfolio.entry_lots["AAPL"]) == 1
    assert engine.portfolio.entry_lots["AAPL"][0]["remaining_shares"] == pytest.approx(3.0)

    reverse_row = db.cursor.execute(
        "SELECT id, entry_trade_id, is_close, position_before, position_after "
        "FROM trade_executions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert reverse_row is not None
    assert int(reverse_row["is_close"]) == 1
    assert int(reverse_row["entry_trade_id"]) > 0
    assert float(reverse_row["position_before"]) == pytest.approx(2.0)
    assert float(reverse_row["position_after"]) == pytest.approx(-3.0)
    assert engine.portfolio.entry_lots["AAPL"][0]["trade_id"] == int(reverse_row["id"])
    assert engine.portfolio.entry_trade_ids["AAPL"] == int(reverse_row["id"])

    # Persist and ensure the residual opener can be reconstructed after restart.
    engine.save_state()
    resumed = PaperTradingEngine(
        initial_capital=10_000.0,
        db_path=str(db_path),
        resume_from_db=True,
    )
    try:
        assert resumed.portfolio.positions.get("AAPL") == -3
        assert resumed.portfolio.entry_trade_ids["AAPL"] == int(reverse_row["id"])
        assert resumed.portfolio.entry_lots["AAPL"][0]["trade_id"] == int(reverse_row["id"])
        assert resumed.portfolio.entry_lots["AAPL"][0]["remaining_shares"] == pytest.approx(3.0)
    finally:
        resumed.db_manager.close()
        db.close()


def test_lob_fallback_uses_depth_profiles_when_depth_missing(monkeypatch):
    """When Depth/Spread are missing, LOB simulator should fall back to configured profiles."""
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=100_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    lob_cfg = LOBConfig(
        levels=5,
        tick_size_bps=1.0,
        alpha=0.3,
        depth_profiles={
            "US_EQUITY": {"depth_notional": 100000.0, "half_spread_bps": 0.5},
            "CRYPTO": {"depth_notional": 30000.0, "half_spread_bps": 5.0},
        },
    )
    monkeypatch.setattr(engine, "_load_lob_config", lambda: lob_cfg)

    market_data = pd.DataFrame({"Close": [100.0]})

    equity_buy = engine._simulate_entry_price(100.0, "BUY", 50, market_data=market_data, ticker="AAPL")
    crypto_buy = engine._simulate_entry_price(100.0, "BUY", 50, market_data=market_data, ticker="BTC-USD")
    assert crypto_buy > equity_buy

    equity_sell = engine._simulate_entry_price(100.0, "SELL", 50, market_data=market_data, ticker="AAPL")
    crypto_sell = engine._simulate_entry_price(100.0, "SELL", 50, market_data=market_data, ticker="BTC-USD")
    assert crypto_sell < equity_sell

    db.close()


def test_time_exit_uses_bar_count_not_calendar_days():
    """Intraday horizons should exit after N bars even within the same day."""
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.95))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    t0 = datetime(2024, 1, 2, 9, 30)
    t1 = t0 + timedelta(hours=1)
    t2 = t1 + timedelta(hours=1)

    entry = {"ticker": "AAPL", "action": "BUY", "confidence": 0.9, "forecast_horizon": 2}
    opened = engine.execute_signal(entry, make_indexed_market_data(100.0, t0))
    assert opened.status == "EXECUTED"
    assert engine.portfolio.positions.get("AAPL") == 1

    hold = {"ticker": "AAPL", "action": "HOLD", "confidence": 0.6}
    mid = engine.execute_signal(hold, make_indexed_market_data(100.1, t1))
    assert mid.status == "REJECTED"
    assert engine.portfolio.positions.get("AAPL") == 1

    final = engine.execute_signal(hold, make_indexed_market_data(100.2, t2))
    assert final.status == "EXECUTED"
    assert final.trade is not None
    assert final.trade.exit_reason == "TIME_EXIT"
    assert engine.portfolio.positions.get("AAPL", 0) == 0

    db.close()


def test_build_close_allocations_live_lots_consumed_before_synthetic():
    """INT-06: live (is_synthetic=0) lots must be consumed BEFORE synthetic lots.

    Without the sort fix, evidence-sprint synthetic opens have older trade_dates
    and win FIFO, causing every live close to be marked is_contaminated=1 and
    excluded from THIN_LINKAGE counting.  This test verifies that when the
    portfolio has a synthetic lot and a live lot for the same ticker, a live
    close consumes the live lot first and is NOT contaminated.
    """
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    # Synthetic opener (older trade_date — would win legacy FIFO)
    synthetic_id = db.save_trade_execution(
        ticker="AAPL",
        trade_date=datetime(2021, 2, 19, tzinfo=timezone.utc).date(),
        action="BUY",
        shares=1,
        price=80.0,
        total_value=80.0,
        commission=0.0,
        execution_mode="synthetic",
        is_synthetic=1,
    )
    # Live opener (newer trade_date — should win with INT-06 sort)
    live_id = db.save_trade_execution(
        ticker="AAPL",
        trade_date=datetime(2026, 4, 14, tzinfo=timezone.utc).date(),
        action="BUY",
        shares=1,
        price=200.0,
        total_value=200.0,
        commission=0.0,
        execution_mode="live",
        is_synthetic=0,
    )

    # Populate entry_lots with BOTH synthetic (first/older) and live (second/newer)
    engine.portfolio.entry_lots["AAPL"] = [
        {
            "trade_id": synthetic_id,
            "action": "BUY",
            "remaining_shares": 1.0,
            "is_synthetic": 1,
        },
        {
            "trade_id": live_id,
            "action": "BUY",
            "remaining_shares": 1.0,
            "is_synthetic": 0,
        },
    ]
    engine.portfolio.positions["AAPL"] = 2
    engine.portfolio.entry_prices["AAPL"] = 140.0
    engine.portfolio.entry_timestamps["AAPL"] = datetime(2026, 4, 14, tzinfo=timezone.utc)
    engine.portfolio.entry_trade_ids["AAPL"] = live_id

    result = engine.execute_signal(
        {"ticker": "AAPL", "action": "SELL", "confidence": 0.9, "execution_mode": "live"},
        make_market_data(210.0),
        proof_mode=True,
    )

    assert result.status == "EXECUTED"
    assert result.trade is not None
    # The live close must have consumed the LIVE lot — NOT contaminated
    assert result.trade.is_contaminated == 0, (
        "INT-06: live close consumed synthetic opener; entry_lots sort is broken"
    )

    row = db.cursor.execute(
        "SELECT COALESCE(is_contaminated, 0) AS is_contaminated, entry_trade_id "
        "FROM trade_executions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert int(row["is_contaminated"]) == 0, (
        "INT-06: DB record is_contaminated=1; live lot was not consumed first"
    )
    assert row["entry_trade_id"] == live_id, (
        f"INT-06: expected entry_trade_id={live_id} (live), got {row['entry_trade_id']}"
    )

    db.close()


def test_confidence_calibrated_saved_to_db():
    """Phase 7.14-E: confidence_calibrated flows from signal dict to trade_executions DB."""
    db = DatabaseManager(":memory:")
    validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.95))
    engine = PaperTradingEngine(
        initial_capital=10_000.0,
        slippage_pct=0.0,
        transaction_cost_pct=0.0,
        database_manager=db,
        signal_validator=validator,
    )

    # Use the same market data format as the working test_execute_signal_executes_and_persists_trade
    signal = {
        "ticker": "MSFT",
        "action": "BUY",
        "confidence": 0.78,
        "confidence_calibrated": 0.62,  # Platt-scaled value
    }

    result = engine.execute_signal(signal, make_market_data(120.0))
    assert result.status == "EXECUTED"

    # Query DB to confirm confidence_calibrated was saved
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT confidence_calibrated FROM trade_executions WHERE ticker = 'MSFT' ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] is not None
    assert abs(row[0] - 0.62) < 1e-6

    db.close()


# ---------------------------------------------------------------------------
# FIX 1: Forced exits inherit OPEN leg's is_synthetic, not current cycle mode
# ---------------------------------------------------------------------------

class TestForcedExitSyntheticInheritance:
    """Verify that forced-exit CLOSE legs inherit is_synthetic from the DB opener,
    preventing execution_mode drift from contaminating live positions."""

    @staticmethod
    def _make_engine_with_open_position(
        db, is_synthetic_open: int, entry_price: float = 100.0, stop_loss: float = 130.0
    ):
        """Create engine with an open BUY position and a stop_loss that market data will breach.

        stop_loss=130.0 is above entry=100.0 so that make_market_data(120.0) (last bar ≈123.5)
        satisfies current_price <= stop_loss → STOP_LOSS forced exit fires.
        """
        validator = DummyValidator(DummyValidationResult(True, "EXECUTE", 0.9))
        engine = PaperTradingEngine(
            initial_capital=10_000.0,
            slippage_pct=0.0,
            transaction_cost_pct=0.0,
            database_manager=db,
            signal_validator=validator,
        )
        exec_mode = "synthetic" if is_synthetic_open else "live"
        opener_id = db.save_trade_execution(
            ticker="AAPL",
            trade_date=datetime(2026, 4, 1, tzinfo=timezone.utc).date(),
            action="BUY",
            shares=1,
            price=entry_price,
            total_value=entry_price,
            commission=0.0,
            execution_mode=exec_mode,
            is_synthetic=is_synthetic_open,
        )
        engine.portfolio.positions["AAPL"] = 1
        engine.portfolio.entry_prices["AAPL"] = entry_price
        engine.portfolio.entry_timestamps["AAPL"] = datetime(2026, 4, 1, tzinfo=timezone.utc)
        engine.portfolio.entry_trade_ids["AAPL"] = opener_id
        engine.portfolio.stop_losses["AAPL"] = stop_loss  # price that triggers STOP_LOSS
        engine.portfolio.entry_lots["AAPL"] = [
            {"trade_id": opener_id, "action": "BUY", "remaining_shares": 1.0,
             "is_synthetic": is_synthetic_open}
        ]
        return engine, opener_id

    def test_forced_exit_of_live_position_stays_live(self):
        """CLOSE forced by stop_loss breach of a live OPEN must be is_synthetic=0,
        even when the current cycle's execution_mode is 'synthetic'."""
        db = DatabaseManager(":memory:")
        # stop_loss=130.0 → market data last price ≈123.5 ≤ 130.0 → STOP_LOSS fires
        engine, _ = self._make_engine_with_open_position(db, is_synthetic_open=0)

        # BUY signal with synthetic execution_mode (simulates data-source fallback cycle)
        signal = {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 0.9,
            "execution_mode": "synthetic",  # current cycle fell back to synthetic
        }
        result = engine.execute_signal(signal, make_market_data(120.0))
        assert result.status == "EXECUTED"
        assert result.trade is not None, "No trade returned"
        assert result.trade.is_forced_exit == 1 or result.trade.exit_reason is not None, (
            "Expected forced exit (STOP_LOSS) but got normal execution — "
            "check stop_loss and market data price"
        )
        assert result.trade.is_synthetic == 0, (
            "Live opener forced-exit was tagged synthetic — THIN_LINKAGE would miss this close"
        )

        row = db.cursor.execute(
            "SELECT is_synthetic FROM trade_executions WHERE is_close=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert int(row["is_synthetic"]) == 0, "DB close leg has is_synthetic=1 for live opener"
        db.close()

    def test_forced_exit_of_synthetic_position_stays_synthetic(self):
        """CLOSE forced by stop_loss of a synthetic OPEN must remain is_synthetic=1."""
        db = DatabaseManager(":memory:")
        engine, _ = self._make_engine_with_open_position(db, is_synthetic_open=1)

        signal = {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 0.9,
            "execution_mode": "synthetic",
        }
        result = engine.execute_signal(signal, make_market_data(120.0))
        assert result.status == "EXECUTED"
        assert result.trade is not None
        assert result.trade.is_forced_exit == 1 or result.trade.exit_reason is not None, (
            "Expected forced exit (STOP_LOSS)"
        )
        assert result.trade.is_synthetic == 1, (
            "Synthetic opener forced-exit was incorrectly tagged is_synthetic=0"
        )
        db.close()

    def test_forced_exit_falls_back_gracefully_when_db_unavailable(self):
        """When opener DB lookup fails, fall back to current execution_mode — no crash."""
        db = DatabaseManager(":memory:")
        engine, _ = self._make_engine_with_open_position(db, is_synthetic_open=0)

        # Corrupt entry_trade_ids to a non-existent ID to trigger the fallback path
        engine.portfolio.entry_trade_ids["AAPL"] = 99999

        signal = {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 0.9,
            "execution_mode": "live",
        }
        # Must not raise
        result = engine.execute_signal(signal, make_market_data(120.0))
        assert result.status == "EXECUTED", "Graceful fallback path crashed"
        db.close()
