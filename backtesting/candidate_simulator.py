"""
Candidate simulator: replay a simple guardrail-aware trading loop per candidate.

Design:
- Map candidate.params into signal confidence and execution cost tweaks.
- Walk-forward harness: fit forecaster -> generate TS signal -> execute stepwise
  across a historical window (no look-ahead).
- Execute via PaperTradingEngine (in-memory DB) to get PnL metrics without
  modifying the live database.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from etl.time_series_forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from etl.database_manager import DatabaseManager
from execution.paper_trading_engine import PaperTradingEngine
from models.time_series_signal_generator import TimeSeriesSignalGenerator

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


def _to_engine_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mapped = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).copy()
    mapped.index = pd.to_datetime(mapped.index)
    mapped.sort_index(inplace=True)
    return mapped


def _ts_signal_to_dict(signal: Any) -> Dict[str, Any]:
    risk_level = "medium"
    try:
        rs = float(getattr(signal, "risk_score", 0.5))
        if rs <= 0.40:
            risk_level = "low"
        elif rs <= 0.60:
            risk_level = "medium"
        elif rs <= 0.80:
            risk_level = "high"
        else:
            risk_level = "extreme"
    except Exception:
        risk_level = "medium"

    ts = getattr(signal, "signal_timestamp", None)
    if isinstance(ts, datetime):
        ts = ts.isoformat()

    return {
        "ticker": getattr(signal, "ticker", None),
        "action": getattr(signal, "action", "HOLD"),
        "confidence": float(getattr(signal, "confidence", 0.5)),
        "entry_price": float(getattr(signal, "entry_price", 0.0)),
        "target_price": getattr(signal, "target_price", None),
        "stop_loss": getattr(signal, "stop_loss", None),
        "signal_timestamp": ts,
        "model_type": getattr(signal, "model_type", "ENSEMBLE"),
        "forecast_horizon": int(getattr(signal, "forecast_horizon", 30)),
        "expected_return": float(getattr(signal, "expected_return", 0.0)),
        "risk_score": float(getattr(signal, "risk_score", 0.5)),
        "risk_level": risk_level,
        "reasoning": getattr(signal, "reasoning", ""),
        "provenance": getattr(signal, "provenance", {}),
        "signal_type": getattr(signal, "signal_type", "TIME_SERIES"),
        "volatility": getattr(signal, "volatility", None),
        "lower_ci": getattr(signal, "lower_ci", None),
        "upper_ci": getattr(signal, "upper_ci", None),
        "signal_id": getattr(signal, "signal_id", None),
        "source": "TIME_SERIES",
        "is_primary": True,
    }


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

    forecast_horizon = int(candidate_params.get("forecast_horizon", 30) or 30)
    history_bars = int(candidate_params.get("history_bars", 120) or 120)
    min_bars = int(candidate_params.get("min_bars", 60) or 60)

    ts_generator = TimeSeriesSignalGenerator(
        confidence_threshold=float(guardrails.get("confidence_threshold", 0.55) or 0.55),
        min_expected_return=float(guardrails.get("min_expected_return", 0.003) or 0.003),
        max_risk_score=float(guardrails.get("max_risk_score", 0.7) or 0.7),
        use_volatility_filter=bool(guardrails.get("use_volatility_filter", True)),
        per_ticker_thresholds=guardrails.get("per_ticker") if isinstance(guardrails.get("per_ticker"), dict) else None,
        cost_model=guardrails.get("cost_model") if isinstance(guardrails.get("cost_model"), dict) else None,
    )

    frames_by_ticker: Dict[str, pd.DataFrame] = {
        str(sym): df.sort_index()
        for sym, df in ohlcv.groupby("ticker")
    }
    all_dates: set[pd.Timestamp] = set()
    for df in frames_by_ticker.values():
        all_dates.update(pd.to_datetime(df.index).to_list())
    for date in sorted(all_dates):
        price_map: Dict[str, float] = {}
        for ticker in tickers:
            tdf = frames_by_ticker.get(str(ticker))
            if tdf is None or tdf.empty:
                continue
            if date not in tdf.index:
                continue

            hist = tdf.loc[:date]
            if hist.empty:
                continue
            if history_bars and len(hist) > history_bars:
                hist = hist.tail(history_bars)
            if len(hist) < min_bars:
                continue

            engine_frame = _to_engine_frame(hist)
            if engine_frame.empty or "Close" not in engine_frame.columns:
                continue

            close_series = engine_frame["Close"].astype(float)
            returns_series = close_series.pct_change().dropna()
            if returns_series.empty:
                continue

            current_price = float(close_series.iloc[-1])
            price_map[str(ticker)] = current_price

            forecast_bundle = None
            try:
                forecaster = TimeSeriesForecaster(
                    config=TimeSeriesForecasterConfig(forecast_horizon=forecast_horizon)
                )
                forecaster.fit(price_series=close_series, returns_series=returns_series)
                forecast_bundle = forecaster.forecast()
            except Exception as exc:
                logger.debug("Forecaster failed for %s @ %s: %s", ticker, date, exc)
                continue

            if not isinstance(forecast_bundle, dict) or not forecast_bundle:
                continue

            try:
                ts_signal = ts_generator.generate_signal(
                    forecast_bundle=forecast_bundle,
                    current_price=current_price,
                    ticker=str(ticker),
                    market_data=engine_frame,
                )
            except Exception as exc:
                logger.debug("Signal generation failed for %s @ %s: %s", ticker, date, exc)
                continue

            signal_dict = _ts_signal_to_dict(ts_signal)
            # Provide a mid-price hint for fill telemetry (high/low proxy).
            try:
                last = engine_frame.iloc[-1]
                high = float(last.get("High")) if pd.notna(last.get("High")) else None
                low = float(last.get("Low")) if pd.notna(last.get("Low")) else None
                if high is not None and low is not None:
                    signal_dict["mid_price_hint"] = (high + low) / 2.0
                else:
                    signal_dict["mid_price_hint"] = current_price
            except Exception:
                signal_dict["mid_price_hint"] = current_price

            engine.execute_signal(signal_dict, market_data=engine_frame)

        if price_map:
            engine.mark_to_market(price_map)

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
