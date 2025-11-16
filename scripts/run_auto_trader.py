#!/usr/bin/env python3
"""
run_auto_trader.py
------------------

Autonomous trading loop that turns Portfolio Maximizer into a profit-focused
machine by wiring extraction → validation → forecasting → signal routing →
execution into a single continuously running workflow.
"""

from __future__ import annotations

from pathlib import Path
import logging
import os
import site
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent
site.addsitedir(str(ROOT_PATH))
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from etl.data_source_manager import DataSourceManager
from etl.data_validator import DataValidator
from etl.preprocessor import Preprocessor
from etl.time_series_forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from execution.paper_trading_engine import PaperTradingEngine
from models.signal_router import SignalRouter

try:  # Optional Ollama dependency
    from ai_llm.ollama_client import OllamaClient, OllamaConnectionError
    from ai_llm.signal_generator import LLMSignalGenerator
except Exception:  # pragma: no cover - optional path
    OllamaClient = None  # type: ignore
    LLMSignalGenerator = None  # type: ignore
    OllamaConnectionError = Exception  # type: ignore

logger = logging.getLogger(__name__)
AI_COMPANION_CONFIG_PATH = ROOT_PATH / "config" / "ai_companion.yml"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _initialize_llm_generator(model: str) -> Optional[LLMSignalGenerator]:
    """Try to boot an LLM fallback for redundancy."""
    if not OllamaClient or not LLMSignalGenerator:
        logger.info("LLM modules not available; running without LLM fallback.")
        return None

    try:
        client = OllamaClient(model=model)
        logger.info("LLM fallback READY (%s)", model)
        return LLMSignalGenerator(ollama_client=client)
    except OllamaConnectionError as err:
            logger.warning("LLM fallback disabled: %s", err)
        return None


def _split_ticker_frame(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Extract a single ticker slice from combined OHLCV data."""
    if data is None or data.empty:
        return None

    ticker_col = "ticker" if "ticker" in data.columns else "Ticker" if "Ticker" in data.columns else None
    if ticker_col is None:
        logger.warning("Ticker column missing; cannot isolate %s", ticker)
        return None

    mask = data[ticker_col].astype(str).str.upper() == ticker.upper()
    ticker_frame = data.loc[mask].copy()
    if ticker_frame.empty:
        return None

    ticker_frame.index = pd.to_datetime(ticker_frame.index)
    ticker_frame.sort_index(inplace=True)
    ticker_frame.drop(columns=[ticker_col], inplace=True, errors="ignore")
    return ticker_frame


def _prepare_market_window(
    manager: DataSourceManager,
    tickers: List[str],
    lookback_days: int,
) -> pd.DataFrame:
    """Fetch the latest OHLCV window for all tickers."""
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=lookback_days)
    logger.info("Fetching OHLCV window: %s → %s", start_date, end_date)
    return manager.extract_ohlcv(
        tickers=tickers,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )


def _generate_time_series_forecast(
    price_frame: pd.DataFrame,
    horizon: int,
) -> Tuple[Optional[Dict], Optional[float]]:
    """Fit the ensemble forecaster and return the forecast bundle + latest price."""
    if "Close" not in price_frame.columns:
        logger.warning("Close column missing; skipping forecasting.")
        return None, None

    close_series = price_frame["Close"].astype(float)
    returns_series = close_series.pct_change().dropna()

    try:
        forecaster = TimeSeriesForecaster(
            config=TimeSeriesForecasterConfig(forecast_horizon=horizon)
        )
        forecaster.fit(price_series=close_series, returns_series=returns_series)
        forecast_bundle = forecaster.forecast()
    except Exception as exc:
        logger.error("Forecasting failed: %s", exc)
        return None, None

    current_price = float(close_series.iloc[-1])
    return forecast_bundle, current_price


def _validate_market_window(validator: DataValidator, data: pd.DataFrame) -> bool:
    """Ensure OHLCV window passes statistical validation."""
    report = validator.validate_ohlcv(data)
    if report["passed"]:
        return True

    logger.warning("Validation failed: %s", report["errors"])
    return False


def _execute_signal(
    router: SignalRouter,
    trading_engine: PaperTradingEngine,
    ticker: str,
    forecast_bundle: Dict,
    current_price: float,
    market_data: pd.DataFrame,
) -> Optional[Dict]:
    """Route signals and push the primary decision through the execution engine."""
    bundle = router.route_signal(
        ticker=ticker,
        forecast_bundle=forecast_bundle,
        current_price=current_price,
        market_data=market_data,
    )

    primary = bundle.primary_signal
    if not primary:
        logger.info("No actionable signal produced for %s", ticker)
        return None

    result = trading_engine.execute_signal(primary, market_data)
    logger.info(
        "Execution result for %s: %s",
        ticker,
        result.status,
    )

    if result.status != "EXECUTED":
        return {
            "ticker": ticker,
            "status": result.status,
            "reason": result.reason,
            "warnings": result.validation_warnings,
        }

    return {
        "ticker": ticker,
        "status": result.status,
        "shares": result.trade.shares if result.trade else 0,
        "action": result.trade.action if result.trade else primary.get("action", "HOLD"),
        "entry_price": result.trade.entry_price if result.trade else current_price,
        "portfolio_value": result.portfolio.total_value if result.portfolio else None,
    }


def _summarize_portfolio(engine: PaperTradingEngine) -> Dict:
    """Return a snapshot of the current automated book."""
    positions = {
        ticker: {
            "shares": shares,
            "entry_price": engine.portfolio.entry_prices.get(ticker),
        }
        for ticker, shares in engine.portfolio.positions.items()
    }

    return {
        "cash": engine.portfolio.cash,
        "positions": positions,
        "total_value": engine.portfolio.total_value,
        "trades": len(engine.trades),
    }


def _load_ai_companion_config(config_path: Path = AI_COMPANION_CONFIG_PATH) -> Dict[str, Any]:
    """Load the AI companion guardrail file so launchers inherit the approved stack."""
    if not config_path.exists():
        logger.warning("AI companion config missing at %s", config_path)
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        logger.error("Failed to parse AI companion config: %s", exc)
        return {}

    return payload


def _activate_ai_companion_guardrails(companion_config: Dict[str, Any]) -> None:
    """Expose tier + knowledge base guardrails via env vars for downstream agents."""
    if not companion_config:
        return

    settings = companion_config.get("ai_companion") or {}
    stack_meta = settings.get("recommended_stack") or {}
    tier = stack_meta.get("tier")
    knowledge_base = settings.get("knowledge_base") or []

    if tier:
        os.environ["AI_COMPANION_STACK_TIER"] = str(tier)
    if knowledge_base:
        resolved = [
            str((ROOT_PATH / kb_entry).resolve())
            for kb_entry in knowledge_base
        ]
        os.environ["AI_COMPANION_KB"] = os.pathsep.join(resolved)

    logger.info(
        "AI companion guardrails active (tier=%s, kb_entries=%s)",
        tier or "unknown",
        len(knowledge_base),
    )


@click.command()
@click.option(
    "--tickers",
    default="AAPL,MSFT",
    show_default=True,
    help="Comma separated list of tickers to trade automatically.",
)
@click.option(
    "--lookback-days",
    default=365,
    show_default=True,
    help="Historical window (days) for building forecasts.",
)
@click.option(
    "--forecast-horizon",
    default=30,
    show_default=True,
    help="Forecast horizon (days) for the ensemble.",
)
@click.option(
    "--initial-capital",
    default=25000.0,
    show_default=True,
    help="Starting capital for the automated engine.",
)
@click.option(
    "--cycles",
    default=1,
    show_default=True,
    help="Number of trading cycles to run before exiting.",
)
@click.option(
    "--sleep-seconds",
    default=300,
    show_default=True,
    help="Delay between cycles when running continuously.",
)
@click.option(
    "--enable-llm",
    is_flag=True,
    help="Enable LLM fallback routing (requires local Ollama).",
)
@click.option(
    "--llm-model",
    default="deepseek-coder:6.7b-instruct-q4_K_M",
    show_default=True,
    help="Ollama model to use when LLM fallback is enabled.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    tickers: str,
    lookback_days: int,
    forecast_horizon: int,
    initial_capital: float,
    cycles: int,
    sleep_seconds: int,
    enable_llm: bool,
    llm_model: str,
    verbose: bool,
) -> None:
    """Entry point for the automated profit engine."""
    _configure_logging(verbose)
    companion_config = _load_ai_companion_config()
    _activate_ai_companion_guardrails(companion_config)

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise click.UsageError("At least one ticker symbol is required.")

    logger.info("Autonomous trading loop booting for tickers: %s", ", ".join(ticker_list))

    data_source_manager = DataSourceManager()
    data_validator = DataValidator()
    preprocessor = Preprocessor()
    trading_engine = PaperTradingEngine(initial_capital=initial_capital)

    llm_generator = _initialize_llm_generator(llm_model) if enable_llm else None
    router_config = {
        "time_series_primary": True,
        "llm_fallback": enable_llm and llm_generator is not None,
        "llm_redundancy": False,
    }
    signal_router = SignalRouter(config=router_config, llm_generator=llm_generator)

    for cycle in range(1, cycles + 1):
        logger.info("=== Trading Cycle %s/%s ===", cycle, cycles)
        try:
            raw_window = _prepare_market_window(data_source_manager, ticker_list, lookback_days)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Market window extraction failed: %s", exc)
            if cycle < cycles:
                time.sleep(sleep_seconds)
            continue

        cycle_results = []
        for ticker in ticker_list:
            ticker_frame = _split_ticker_frame(raw_window, ticker)
            if ticker_frame is None or ticker_frame.empty:
                logger.warning("No data for %s; skipping.", ticker)
                continue

            ticker_frame = preprocessor.handle_missing(ticker_frame)

            if not _validate_market_window(data_validator, ticker_frame):
                logger.warning("Validation rejected %s window; skipping.", ticker)
                continue

            forecast_bundle, current_price = _generate_time_series_forecast(
                ticker_frame,
                forecast_horizon,
            )

            if not forecast_bundle or current_price is None:
                logger.warning("Forecasting failed for %s; skipping.", ticker)
                continue

            execution_report = _execute_signal(
                router=signal_router,
                trading_engine=trading_engine,
                ticker=ticker,
                forecast_bundle=forecast_bundle,
                current_price=current_price,
                market_data=ticker_frame,
            )

            if execution_report:
                cycle_results.append(execution_report)

        summary = _summarize_portfolio(trading_engine)
        logger.info(
            "Cycle %s complete: %s trades | Cash $%.2f | Portfolio $%.2f",
            cycle,
            len(cycle_results),
            summary["cash"],
            summary["total_value"],
        )

        if verbose:
            logger.debug("Positions: %s", summary["positions"])

        if cycle < cycles:
            logger.info("Sleeping %s seconds before next cycle...", sleep_seconds)
            time.sleep(sleep_seconds)

    final_summary = _summarize_portfolio(trading_engine)
    logger.info("=== Automated Trading Complete ===")
    logger.info("Total trades executed: %s", final_summary["trades"])
    logger.info("Final cash: $%.2f | Portfolio value: $%.2f", final_summary["cash"], final_summary["total_value"])
    if final_summary["positions"]:
        logger.info("Open positions: %s", final_summary["positions"])


if __name__ == "__main__":
    main()
