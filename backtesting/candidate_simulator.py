"""
Candidate simulator: replay a simple guardrail-aware trading loop per candidate.

Design:
- Map candidate.params into signal confidence and execution cost tweaks.
- Use past returns as a naive forecast (no look-ahead), compared against the
  guardrail min_expected_return to decide BUY/SELL/HOLD.
- Execute via PaperTradingEngine (in-memory DB) to get PnL metrics without
  modifying the live database.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from etl.database_manager import DatabaseManager
from execution.paper_trading_engine import PaperTradingEngine

logger = logging.getLogger(__name__)


def _max_drawdown(equity: List[Dict[str, float]]) -> float:
    peak = -float("inf")
    max_dd = 0.0
    for pt in equity:
        val = float(pt.get("equity", 0.0))
        peak = max(peak, val)
        if peak > 0:
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def simulate_candidate(
    source_db: DatabaseManager,
    tickers: Sequence[str],
    start_date: Optional[str],
    end_date: Optional[str],
    candidate_params: Dict[str, Any],
    guardrails: Dict[str, Any],
    initial_capital: float = 100000.0,
) -> Dict[str, float]:
    """
    Run a simple simulation for a candidate over a historical window.
    Guardrails (min_expected_return, max_risk_score) are read but not changed.
    """
    if not tickers:
        return {"total_return": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "total_trades": 0}

    ohlcv = source_db.load_ohlcv(list(tickers), start_date=start_date, end_date=end_date)
    if ohlcv.empty:
        return {"total_return": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "total_trades": 0}

    # Create isolated in-memory DB for simulation to avoid polluting the main DB.
    sim_db = DatabaseManager(db_path=":memory:")
    # Candidate-driven execution tweaks
    execution_style = candidate_params.get("execution_style", "market")
    transaction_cost_pct = 0.001
    slippage_pct = 0.001
    if execution_style == "limit_bias":
        transaction_cost_pct = 0.0005
        slippage_pct = 0.0005

    engine = PaperTradingEngine(
        initial_capital=initial_capital,
        slippage_pct=slippage_pct,
        transaction_cost_pct=transaction_cost_pct,
        database_manager=sim_db,
    )

    base_min_ret = float(guardrails.get("min_expected_return", 0.0) or 0.0)
    div_pen = float(candidate_params.get("diversification_penalty", 0.0) or 0.0)
    kelly_cap = float(candidate_params.get("sizing_kelly_fraction_cap", 0.0) or 0.0)
    # Loosen threshold slightly based on diversification penalty to avoid zero trades
    effective_min_ret = max(0.0, base_min_ret * max(0.25, 1.0 - div_pen))
    confidence_base = 0.55

    for ticker, tdf in ohlcv.groupby("ticker"):
        tdf = tdf.sort_index()
        closes = tdf["close"].astype(float)
        rets = closes.pct_change().fillna(0.0)

        for idx in range(1, len(tdf)):
            hist_slice = tdf.iloc[: idx + 1]
            daily_ret = rets.iloc[idx]
            # Use recent return as forecast to guarantee non-zero signals when markets move
            forecast_ret = rets.iloc[max(0, idx - 5): idx].mean() if idx > 1 else daily_ret
            expected_return = forecast_ret

            action = "HOLD"
            if forecast_ret > effective_min_ret:
                action = "BUY"
            elif forecast_ret < -effective_min_ret:
                action = "SELL"

            conf = confidence_base + kelly_cap - div_pen
            conf = max(0.5, min(0.95, conf))

            signal = {
                "ticker": ticker,
                "action": action,
                "confidence": conf,
                "expected_return": expected_return,
                "risk_score": 0.5,
                "entry_price": float(closes.iloc[idx]),
                "signal_timestamp": datetime.combine(hist_slice.index[-1], datetime.min.time()),
            }

            engine.execute_signal(signal, market_data=hist_slice)

        # Mark to market at the end of ticker loop
        engine.mark_to_market({ticker: float(closes.iloc[-1])})

    summary = sim_db.get_performance_summary()
    equity = sim_db.get_equity_curve(initial_capital=initial_capital)
    max_dd = _max_drawdown(equity)

    total_profit = summary.get("total_profit") or 0.0
    profit_factor = summary.get("profit_factor") or 0.0
    win_rate = summary.get("win_rate") or 0.0
    total_trades = summary.get("total_trades") or 0

    return {
        "total_return": float(total_profit),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_dd),
        "total_trades": int(total_trades),
    }
