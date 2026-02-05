#!/usr/bin/env python3
"""
Horizon-consistent TS backtest (walk-forward).

Replays TimeSeriesForecaster -> TimeSeriesSignalGenerator -> PaperTradingEngine
across a fixed historical window so entry, exits, and evaluation share the same
forecast horizon semantics.

This is the Phase 3.1 verification harness referenced by
Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md.
"""

from __future__ import annotations

import json
import logging
import os
import site
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent
site.addsitedir(str(ROOT_PATH))
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from backtesting.candidate_simulator import simulate_candidate
from etl.database_manager import DatabaseManager

logger = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass(frozen=True)
class Window:
    start_date: str
    end_date: str


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _split_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in (raw or "").split(",") if t.strip()]


def _db_max_ohlcv_date(db: DatabaseManager) -> Optional[str]:
    try:
        db.cursor.execute("SELECT MAX(date) FROM ohlcv_data")
        row = db.cursor.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _resolve_window(db: DatabaseManager, start_date: Optional[str], end_date: Optional[str], lookback_days: int) -> Window:
    end = end_date or _db_max_ohlcv_date(db)
    if not end:
        end = datetime.now(UTC).date().isoformat()

    if start_date:
        start = start_date
    else:
        end_dt = datetime.fromisoformat(end)
        start_dt = (end_dt - timedelta(days=max(int(lookback_days), 1))).date()
        start = start_dt.isoformat()

    return Window(start_date=str(start), end_date=str(end))


def _load_ts_guardrails(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    sr = raw.get("signal_routing") or {}
    if not isinstance(sr, dict):
        return {}
    ts = sr.get("time_series") or {}
    return ts if isinstance(ts, dict) else {}


def run_horizon_backtest(
    *,
    db_manager: DatabaseManager,
    tickers: List[str],
    window: Window,
    candidate_params: Dict[str, Any],
    guardrails: Dict[str, Any],
    initial_capital: float,
    report_path: Path,
) -> Dict[str, Any]:
    metrics = simulate_candidate(
        source_db=db_manager,
        tickers=tickers,
        start_date=window.start_date,
        end_date=window.end_date,
        candidate_params=candidate_params,
        guardrails=guardrails,
        initial_capital=initial_capital,
    )

    payload = {
        "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        "window": {"start_date": window.start_date, "end_date": window.end_date},
        "tickers": tickers,
        "candidate_params": candidate_params,
        "guardrails": guardrails,
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _default_report_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return ROOT_PATH / "reports" / f"horizon_backtest_{ts}.json"


@click.command()
@click.option(
    "--tickers",
    default="AAPL,MSFT",
    show_default=True,
    help="Comma separated tickers to simulate.",
)
@click.option(
    "--db-path",
    default=None,
    show_default=False,
    help="Optional SQLite DB path with OHLCV in ohlcv_data (defaults to PORTFOLIO_DB_PATH or data/portfolio_maximizer.db).",
)
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD).")
@click.option(
    "--lookback-days",
    default=120,
    show_default=True,
    help="Window size (days) when --start-date is not provided.",
)
@click.option(
    "--forecast-horizon",
    default=14,
    show_default=True,
    help="Forecast horizon in bars (daily bars == days).",
)
@click.option(
    "--history-bars",
    default=120,
    show_default=True,
    help="How many bars to feed into each re-fit.",
)
@click.option(
    "--min-bars",
    default=60,
    show_default=True,
    help="Minimum bars required before fitting/forecasting.",
)
@click.option(
    "--execution-style",
    default="market",
    show_default=True,
    type=click.Choice(["market", "limit_bias"], case_sensitive=False),
    help="Execution style to map into simulator slippage/fees.",
)
@click.option(
    "--initial-capital",
    default=25000.0,
    show_default=True,
    help="Initial capital for the simulation engine.",
)
@click.option(
    "--signal-routing-config",
    default="config/signal_routing_config.yml",
    show_default=True,
    help="Path to signal routing config used for TS guardrails.",
)
@click.option(
    "--candidate-json",
    default=None,
    show_default=False,
    help="Optional JSON string merged into candidate params (overrides flags).",
)
@click.option(
    "--report-path",
    default=None,
    show_default=False,
    help="Optional report JSON path (defaults to reports/horizon_backtest_<ts>.json).",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    tickers: str,
    db_path: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    lookback_days: int,
    forecast_horizon: int,
    history_bars: int,
    min_bars: int,
    execution_style: str,
    initial_capital: float,
    signal_routing_config: str,
    candidate_json: Optional[str],
    report_path: Optional[str],
    verbose: bool,
) -> None:
    _configure_logging(verbose)

    ticker_list = _split_tickers(tickers)
    if not ticker_list:
        raise click.UsageError("At least one ticker is required.")

    resolved_db_path = db_path or os.getenv("PORTFOLIO_DB_PATH") or str(ROOT_PATH / "data" / "portfolio_maximizer.db")
    db_manager = DatabaseManager(db_path=resolved_db_path)

    window = _resolve_window(db_manager, start_date, end_date, lookback_days)
    guardrails = _load_ts_guardrails(ROOT_PATH / signal_routing_config)

    candidate_params: Dict[str, Any] = {
        "forecast_horizon": int(forecast_horizon),
        "history_bars": int(history_bars),
        "min_bars": int(min_bars),
        "execution_style": str(execution_style),
    }
    if candidate_json:
        try:
            extra = json.loads(candidate_json)
            if isinstance(extra, dict):
                candidate_params.update(extra)
        except json.JSONDecodeError as exc:
            raise click.UsageError(f"candidate-json is not valid JSON: {exc}") from exc

    out_path = Path(report_path).expanduser() if report_path else _default_report_path()
    payload = run_horizon_backtest(
        db_manager=db_manager,
        tickers=ticker_list,
        window=window,
        candidate_params=candidate_params,
        guardrails=guardrails,
        initial_capital=float(initial_capital),
        report_path=out_path,
    )

    metrics = payload.get("metrics") or {}
    logger.info(
        "Horizon backtest complete (%s -> %s) trades=%s PF=%.3f WR=%.3f PnL=%.2f maxDD=%.3f",
        window.start_date,
        window.end_date,
        metrics.get("total_trades", 0),
        float(metrics.get("profit_factor", 0.0)),
        float(metrics.get("win_rate", 0.0)),
        float(metrics.get("total_return", 0.0)),
        float(metrics.get("max_drawdown", 0.0)),
    )
    logger.info("Report written: %s", out_path)


if __name__ == "__main__":
    main()
