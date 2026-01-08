#!/usr/bin/env python3
"""
sweep_ts_thresholds.py
----------------------

Lightweight helper to sweep Time Series signal generator thresholds
over a simple grid and summarise realised performance per ticker.

Design goals:
- Read-only with respect to configs and schema (no migrations).
- Work directly off the existing trade_executions table and OHLCV data
  via etl.database_manager.DatabaseManager.
- Output a JSON summary that higher-level orchestration (e.g. bash
  helpers or notebooks) can use to decide which (confidence_threshold,
  min_expected_return) pairs to promote.

CURRENT SCOPE (SCAFFOLD):
- Does NOT re-simulate trades. It slices realised performance by
  ticker and uses filtering heuristics (min_trades) to score regimes.
- Selection / persistence of “best” thresholds is intentionally left
  to callers so this script stays safe and composable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in __import__("sys").modules["sys"].path:
    __import__("sys").modules["sys"].path.insert(0, str(ROOT_PATH))

from etl.database_manager import DatabaseManager  # noqa: E402

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class GridPointResult:
    ticker: str
    confidence_threshold: float
    min_expected_return: float
    total_trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    annualized_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "confidence_threshold": self.confidence_threshold,
            "min_expected_return": self.min_expected_return,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_profit": self.total_profit,
            "annualized_pnl": self.annualized_pnl,
        }


def _parse_float_list(raw: str) -> List[float]:
    raw = raw.strip()
    if not raw:
        return []
    # Allow simple JSON lists as well as comma-separated strings.
    if raw.startswith("["):
        try:
            values = json.loads(raw)
            return [float(v) for v in values]
        except Exception:
            pass
    return [float(part) for part in raw.split(",") if part.strip()]


def _iter_tickers(db: DatabaseManager, tickers: Optional[List[str]]) -> List[str]:
    if tickers:
        return tickers
    # Fall back to tickers that actually have trades to keep the
    # grid focused on sleeves with realised history.
    db.cursor.execute(
        "SELECT DISTINCT ticker FROM trade_executions WHERE realized_pnl IS NOT NULL ORDER BY ticker"
    )
    rows = db.cursor.fetchall()
    return [row[0] for row in rows] if rows else []


def _load_ticker_trades(
    db: DatabaseManager,
    ticker: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    query = """
        SELECT trade_date, realized_pnl
        FROM trade_executions
        WHERE ticker = ? AND realized_pnl IS NOT NULL
    """
    params: List[Any] = [ticker]
    if start_date:
        query += " AND trade_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND trade_date <= ?"
        params.append(end_date)
    query += " ORDER BY trade_date ASC, id ASC"

    db.cursor.execute(query, params)
    rows = db.cursor.fetchall()
    return [dict(row) for row in rows]


def _summarise_trades(
    trades: Iterable[Dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> Tuple[int, float, float, float, float]:
    total_trades = 0
    winning = 0
    gross_profit = 0.0
    gross_loss = 0.0
    first_trade_date: Optional[datetime] = None

    for row in trades:
        pnl = float(row.get("realized_pnl") or 0.0)
        td = row.get("trade_date")
        if td:
            try:
                dt = datetime.fromisoformat(str(td))
                first_trade_date = dt if first_trade_date is None else min(first_trade_date, dt)
            except Exception:
                pass
        total_trades += 1
        if pnl > 0:
            winning += 1
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)

    if total_trades == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    win_rate = winning / total_trades
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    total_profit = gross_profit - gross_loss
    # Annualise realised PnL over the observed window to stabilise selection.
    window_start = first_trade_date or window_start
    span_days = max((window_end - window_start).days, 1)
    annualized_pnl = total_profit * (365.0 / span_days)
    return total_trades, win_rate, profit_factor, total_profit, annualized_pnl


def _select_best_by_rules(
    results: List[GridPointResult],
    min_trades: int,
    min_profit_factor: float,
    min_win_rate: float,
) -> Dict[str, Dict[str, Any]]:
    by_ticker: Dict[str, List[GridPointResult]] = {}
    for row in results:
        by_ticker.setdefault(row.ticker, []).append(row)

    selection: Dict[str, Dict[str, Any]] = {}
    for ticker, rows in by_ticker.items():
        candidates = [
            r
            for r in rows
            if r.total_trades >= int(min_trades)
            and r.profit_factor >= float(min_profit_factor)
            and r.win_rate >= float(min_win_rate)
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda r: (
                r.annualized_pnl,
                r.total_profit,
                r.profit_factor,
            ),
        )
        selection[ticker] = {
            "confidence_threshold": best.confidence_threshold,
            "min_expected_return": best.min_expected_return,
            "total_trades": best.total_trades,
            "win_rate": best.win_rate,
            "profit_factor": best.profit_factor,
            "annualized_pnl": best.annualized_pnl,
            "total_profit": best.total_profit,
        }
    return selection


@click.command()
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite database path used by the auto trader.",
)
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated list of tickers to sweep. "
    "If omitted, uses tickers with realised trades in trade_executions.",
)
@click.option(
    "--grid-confidence",
    default="0.50,0.55,0.60",
    show_default=True,
    help="Grid of confidence_threshold values (comma-separated or JSON list).",
)
@click.option(
    "--grid-min-return",
    default="0.001,0.002,0.003",
    show_default=True,
    help="Grid of min_expected_return values (decimal, e.g. 0.002 = 0.2%).",
)
@click.option(
    "--lookback-days",
    default=365,
    show_default=True,
    help="Lookback window in days for realised trades.",
)
@click.option(
    "--min-trades",
    default=10,
    show_default=True,
    help="Minimum number of trades required for a gridpoint to be considered.",
)
@click.option(
    "--selection-min-profit-factor",
    default=1.1,
    show_default=True,
    help="Minimum profit factor required for a gridpoint to qualify for selection.",
)
@click.option(
    "--selection-min-win-rate",
    default=0.5,
    show_default=True,
    help="Minimum win rate required for a gridpoint to qualify for selection.",
)
@click.option(
    "--output",
    default="logs/automation/ts_threshold_sweep.json",
    show_default=True,
    help="Path to write JSON summary (directories are created as needed).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    db_path: str,
    tickers: Optional[str],
    grid_confidence: str,
    grid_min_return: str,
    lookback_days: int,
    min_trades: int,
    selection_min_profit_factor: float,
    selection_min_win_rate: float,
    output: str,
    verbose: bool,
) -> None:
    """
    Sweep TS thresholds over a simple grid and summarise realised performance.

    NOTE: This scaffold operates purely on realised trades. It does not
    mutate configs or choose/persist “best” parameters; callers are expected
    to consume the JSON output and decide how to apply any recommendations.
    """
    _configure_logging(verbose)

    confidences = _parse_float_list(grid_confidence)
    min_returns = _parse_float_list(grid_min_return)
    if not confidences or not min_returns:
        raise SystemExit("Grid parameters are empty; please provide at least one value for each axis.")

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=int(lookback_days))
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)

    results: List[GridPointResult] = []
    with DatabaseManager(db_path=db_path) as db:
        ticker_list = _iter_tickers(db, [t.strip() for t in tickers.split(",")] if tickers else None)
        if not ticker_list:
            logger.info("No tickers with realised trades found; nothing to sweep.")
            return

        for ticker in ticker_list:
            trades = _load_ticker_trades(db, ticker, start_iso, end_iso)
            if not trades:
                logger.debug("No trades for %s in window %s -> %s", ticker, start_iso, end_iso)
                continue

            for c in confidences:
                for mret in min_returns:
                    # In a future iteration, trade-level metadata (e.g., the
                    # thresholds active when the signal was generated) could
                    # be used to filter here. For now, we treat all trades
                    # for the ticker as one sleeve and attach the gridpoint
                    # parameters purely as labels.
                    total_trades, win_rate, profit_factor, total_profit, annualized_pnl = _summarise_trades(
                        trades, start_dt, end_dt
                    )
                    if total_trades < int(min_trades):
                        continue
                    results.append(
                        GridPointResult(
                            ticker=ticker,
                            confidence_threshold=float(c),
                            min_expected_return=float(mret),
                            total_trades=total_trades,
                            win_rate=win_rate,
                            profit_factor=profit_factor,
                            total_profit=total_profit,
                            annualized_pnl=annualized_pnl,
                        )
                    )

    selection = _select_best_by_rules(
        results=results,
        min_trades=min_trades,
        min_profit_factor=selection_min_profit_factor,
        min_win_rate=selection_min_win_rate,
    )

    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "db_path": db_path,
            "lookback_days": lookback_days,
            "min_trades": min_trades,
            "grid_confidence": confidences,
            "grid_min_return": min_returns,
            "window_start": start_iso,
            "window_end": end_iso,
            "selection_min_profit_factor": selection_min_profit_factor,
            "selection_min_win_rate": selection_min_win_rate,
        },
        "results": [r.to_dict() for r in results],
        "selection": selection,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Threshold sweep summary written to %s (%s gridpoints)", out_path, len(results))


if __name__ == "__main__":
    main()
