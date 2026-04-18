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
    alpha: float = 0.0
    information_ratio: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    r_squared: float = 0.0
    benchmark_proxy: str = "equal_weight_universe"
    benchmark_metrics_status: str = "unavailable"
    benchmark_observations: int = 0
    anti_barbell_ok: bool = False
    anti_barbell_reason: Optional[str] = None
    anti_barbell_evidence: Optional[Dict[str, Any]] = None
    candidate_invalid_reason: Optional[str] = None
    strategy_returns: Optional[pd.Series] = None


def backtest_candidate(
    db_manager: DatabaseManager,
    tickers: Sequence[str],
    start: Optional[str],
    end: Optional[str],
    candidate_params: Dict[str, Any],
    guardrails: Dict[str, Any],
    forecasting_config_path: Optional[str] = None,
) -> BacktestResult:
    """
    Run the causal simulator and map its evidence bundle into the historical
    result dataclass expected by older call sites.
    """
    if not tickers:
        return BacktestResult(
            total_profit=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            total_trades=0,
            total_return=0.0,
            alpha=0.0,
            information_ratio=0.0,
            beta=0.0,
            tracking_error=0.0,
            r_squared=0.0,
            benchmark_proxy="equal_weight_universe",
            benchmark_metrics_status="unavailable",
            benchmark_observations=0,
            anti_barbell_ok=False,
            anti_barbell_reason="no_tickers",
            anti_barbell_evidence={},
            candidate_invalid_reason="no_tickers",
            strategy_returns=pd.Series(dtype=float),
        )

    metrics = simulate_candidate(
        source_db=db_manager,
        tickers=tickers,
        start_date=start,
        end_date=end,
        candidate_params=candidate_params,
        guardrails=guardrails,
        include_strategy_returns=True,
        forecasting_config_path=forecasting_config_path,
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
        alpha=float(metrics.get("alpha", 0.0) or 0.0),
        information_ratio=float(metrics.get("information_ratio", 0.0) or 0.0),
        beta=float(metrics.get("beta", 0.0) or 0.0),
        tracking_error=float(metrics.get("tracking_error", 0.0) or 0.0),
        r_squared=float(metrics.get("r_squared", 0.0) or 0.0),
        benchmark_proxy=str(metrics.get("benchmark_proxy") or "equal_weight_universe"),
        benchmark_metrics_status=str(metrics.get("benchmark_metrics_status") or "unavailable"),
        benchmark_observations=int(metrics.get("benchmark_observations", 0) or 0),
        anti_barbell_ok=bool(metrics.get("anti_barbell_ok", False)),
        anti_barbell_reason=(
            str(metrics.get("anti_barbell_reason"))
            if metrics.get("anti_barbell_reason") is not None
            else None
        ),
        anti_barbell_evidence=dict(metrics.get("anti_barbell_evidence") or {}),
        candidate_invalid_reason=(
            str(metrics.get("candidate_invalid_reason"))
            if metrics.get("candidate_invalid_reason") is not None
            else None
        ),
        strategy_returns=strategy_returns,
    )
