"""
Candidate backtester (lightweight, placeholder).

Replays a simple rule-based strategy over historical OHLCV data while honoring
guardrails and candidate parameters. This is intentionally conservative and
does not hardcode any strategies; it provides a deterministic way to score
candidate configurations without modifying global thresholds (min_expected_return,
max_risk_score).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from etl.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    total_profit: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (roll_max - equity) / roll_max.replace(0, np.nan)
    return float(dd.max() if not dd.empty else 0.0)


def backtest_candidate(
    db_manager: DatabaseManager,
    tickers: Sequence[str],
    start: Optional[str],
    end: Optional[str],
    candidate_params: Dict[str, Any],
    guardrails: Dict[str, Any],
) -> BacktestResult:
    """
    Simple candidate backtest:
    - Loads OHLCV from DB within [start, end].
    - For each ticker, uses a naive momentum rule:
        if daily return > min_expected_return -> BUY 1 * sizing_factor
        else if daily return < -min_expected_return -> SELL 1 * sizing_factor
    - sizing_factor is influenced by candidate parameters:
        sizing_factor = (1 - diversification_penalty) * (1 + kelly_cap / 10)
      and is clipped to [0, 2].
    - Guardrails (min_expected_return, max_risk_score) are read from config;
      this backtest does not modify them.
    """
    if not tickers:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0)

    df = db_manager.load_ohlcv(list(tickers), start_date=start, end_date=end)
    if df.empty:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0)

    # Extract guardrails
    min_ret = float(guardrails.get("min_expected_return", 0.0) or 0.0)

    # Candidate-driven sizing factor (bounded)
    div_pen = float(candidate_params.get("diversification_penalty", 0.0) or 0.0)
    kelly_cap = float(candidate_params.get("sizing_kelly_fraction_cap", 0.0) or 0.0)
    sizing_factor = max(0.0, min(2.0, (1.0 - div_pen) * (1.0 + kelly_cap / 10.0)))

    equity_curve: List[float] = []
    pnl_events: List[float] = []

    # Group by ticker
    for ticker, tdf in df.groupby("ticker"):
        closes = tdf["close"].astype(float)
        rets = closes.pct_change().fillna(0.0)

        # Simple rule: buy on positive above threshold, sell on negative below -threshold
        positions = []
        pnl = 0.0
        for r in rets:
            if r > min_ret:
                pnl += r * sizing_factor
                positions.append("BUY")
            elif r < -min_ret:
                pnl -= r * sizing_factor  # short equivalent
                positions.append("SELL")
            else:
                positions.append("HOLD")
        pnl_events.append(pnl)

        # Equity curve contribution (normalized)
        equity_component = (1 + rets * sizing_factor).cumprod()
        equity_curve.append(equity_component.values)

    total_profit = float(np.sum(pnl_events)) if pnl_events else 0.0

    # Approximate trades and win rate from per-ticker PnL events
    total_trades = len(pnl_events)
    wins = sum(1 for ev in pnl_events if ev > 0)
    win_rate = wins / len(pnl_events) if pnl_events else 0.0

    # Profit factor approximation
    gross_profit = float(np.sum([ev for ev in pnl_events if ev > 0])) if pnl_events else 0.0
    gross_loss = float(np.abs(np.sum([ev for ev in pnl_events if ev < 0]))) if pnl_events else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Equity aggregation for drawdown
    max_len = max((len(eq) for eq in equity_curve), default=0)
    agg = np.zeros(max_len)
    for eq in equity_curve:
        if len(eq) < max_len:
            padded = np.pad(eq, (0, max_len - len(eq)), constant_values=eq[-1] if len(eq) else 1.0)
        else:
            padded = eq
        agg += padded
    agg = agg / max(1, len(equity_curve)) if len(equity_curve) else agg
    max_drawdown = _max_drawdown(pd.Series(agg))

    return BacktestResult(
        total_profit=total_profit,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        total_trades=len(pnl_events),
    )
