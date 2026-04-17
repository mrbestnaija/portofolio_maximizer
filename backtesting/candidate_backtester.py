"""
Compatibility adapter for causal candidate evaluation.

Historically this module held a same-bar look-ahead placeholder. The optimizer
now uses the walk-forward simulator directly; this adapter remains only so older
callers can keep importing ``backtest_candidate`` without getting a non-causal
evaluation path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from etl.database_manager import DatabaseManager
from backtesting.candidate_simulator import simulate_candidate

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    total_profit: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    total_return: float = 0.0
    strategy_returns: Optional[pd.Series] = None


def backtest_candidate(
    db_manager: DatabaseManager,
    tickers: Sequence[str],
    start: Optional[str],
    end: Optional[str],
    candidate_params: Dict[str, Any],
    guardrails: Dict[str, Any],
) -> BacktestResult:
    """
    Run the causal simulator and map its evidence bundle into the historical
    result dataclass expected by older call sites.
    """
    if not tickers:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0, total_return=0.0, strategy_returns=pd.Series(dtype=float))

    metrics = simulate_candidate(
        source_db=db_manager,
        tickers=tickers,
        start_date=start,
        end_date=end,
        candidate_params=candidate_params,
        guardrails=guardrails,
        include_strategy_returns=True,
    )

    strategy_returns_raw = metrics.pop("strategy_returns", None)
    if isinstance(strategy_returns_raw, pd.Series):
        strategy_returns = strategy_returns_raw.astype(float)
    elif isinstance(strategy_returns_raw, (list, tuple)):
        strategy_returns = pd.Series(strategy_returns_raw, dtype=float)
    else:
        strategy_returns = pd.Series(dtype=float)

    return BacktestResult(
        total_profit=float(metrics.get("total_profit", 0.0) or 0.0),
        win_rate=float(metrics.get("win_rate", 0.0) or 0.0),
        profit_factor=float(metrics.get("profit_factor", 0.0) or 0.0),
        max_drawdown=float(metrics.get("max_drawdown", 0.0) or 0.0),
        total_trades=int(metrics.get("total_trades", 0) or 0),
        total_return=float(metrics.get("total_return", 0.0) or 0.0),
        strategy_returns=strategy_returns,
    )
