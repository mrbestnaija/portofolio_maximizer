#!/usr/bin/env python3
"""
run_auto_trader.py
------------------

Autonomous trading loop that turns Portfolio Maximizer into a profit-focused
machine by wiring extraction -> validation -> forecasting -> signal routing ->
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
import json
import matplotlib.pyplot as plt

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
from etl.data_universe import resolve_ticker_universe

try:  # Optional Ollama dependency
    from ai_llm.ollama_client import OllamaClient, OllamaConnectionError
    from ai_llm.signal_generator import LLMSignalGenerator
except Exception:  # pragma: no cover - optional path
    OllamaClient = None  # type: ignore
    LLMSignalGenerator = None  # type: ignore
    OllamaConnectionError = Exception  # type: ignore

logger = logging.getLogger(__name__)
AI_COMPANION_CONFIG_PATH = ROOT_PATH / "config" / "ai_companion.yml"
DASHBOARD_DATA_PATH = ROOT_PATH / "visualizations" / "dashboard_data.json"
MIN_LOOKBACK_DAYS = 180
MIN_SERIES_POINTS = 90
MIN_QUALITY_SCORE = 0.50


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


def _ensure_min_length(frame: pd.DataFrame, min_points: int = MIN_SERIES_POINTS) -> pd.DataFrame:
    """Pad short series forward with the last known row to stabilize forecasting."""
    if frame is None or frame.empty or len(frame) >= min_points:
        return frame

    idx = pd.DatetimeIndex(frame.index)

    # Infer frequency defensively; if too few points, fall back to business day
    try:
        inferred = pd.infer_freq(idx)
    except ValueError:
        inferred = None
    freq = inferred or "B"

    # If fewer than 3 points, fabricate a tiny scaffold to allow padding
    if len(idx) < 3:
        start = idx.min()
        scaffold = pd.date_range(start=start, periods=3, freq=freq)
        frame = frame.reindex(idx.union(scaffold)).sort_index().ffill()
        idx = pd.DatetimeIndex(frame.index)

    end = idx[-1]
    padded_index = pd.date_range(end=end, periods=min_points, freq=freq)
    padded = frame.reindex(idx.union(padded_index)).sort_index().ffill()
    return padded.tail(min_points)


def _prepare_market_window(
    manager: DataSourceManager,
    tickers: List[str],
    lookback_days: int,
) -> pd.DataFrame:
    """Fetch the latest OHLCV window for all tickers."""
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=max(lookback_days, MIN_LOOKBACK_DAYS))
    logger.info("Fetching OHLCV window: %s -> %s", start_date, end_date)
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
    quality: Optional[Dict[str, Any]] = None,
    data_source: Optional[str] = None,
) -> Optional[Dict]:
    """Route signals and push the primary decision through the execution engine."""
    bundle = router.route_signal(
        ticker=ticker,
        forecast_bundle=forecast_bundle,
        current_price=current_price,
        market_data=market_data,
        quality=quality,
        data_source=data_source,
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
            "quality": quality,
            "data_source": data_source,
        }

    realized_pnl = getattr(result.trade, "realized_pnl", None)
    realized_pnl_pct = getattr(result.trade, "realized_pnl_pct", None)
    executed_at = result.trade.timestamp.isoformat() if result.trade else None

    return {
        "ticker": ticker,
        "status": result.status,
        "shares": result.trade.shares if result.trade else 0,
        "action": result.trade.action if result.trade else primary.get("action", "HOLD"),
        "entry_price": result.trade.entry_price if result.trade else current_price,
        "portfolio_value": result.portfolio.total_value if result.portfolio else None,
        "signal_source": primary.get("source", "TIME_SERIES"),
        "signal_confidence": primary.get("confidence"),
        "expected_return": primary.get("expected_return"),
        "quality": quality,
        "data_source": data_source,
        "timestamp": executed_at,
        "realized_pnl": realized_pnl,
        "realized_pnl_pct": realized_pnl_pct,
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
    pnl_dollars = engine.portfolio.total_value - engine.initial_capital
    pnl_pct = pnl_dollars / engine.initial_capital if engine.initial_capital else 0.0

    return {
        "cash": engine.portfolio.cash,
        "positions": positions,
        "total_value": engine.portfolio.total_value,
        "trades": len(engine.trades),
        "pnl_dollars": pnl_dollars,
        "pnl_pct": pnl_pct,
    }


def _compute_quality_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute simple window quality metrics for forecasting/LLM routing."""
    if frame is None or frame.empty:
        return {"length": 0, "missing_pct": 1.0, "coverage": 0.0, "outlier_frac": 0.0, "quality_score": 0.0}

    length = len(frame)
    missing_pct = float(frame.isna().sum().sum()) / float(frame.size) if frame.size else 1.0

    idx = pd.DatetimeIndex(frame.index)
    if len(idx) > 1:
        inferred = pd.infer_freq(idx)
        expected_len = len(pd.date_range(idx[0], idx[-1], freq=inferred or "B"))
        coverage = min(1.0, length / expected_len) if expected_len else 0.0
    else:
        coverage = 1.0

    outlier_frac = 0.0
    if "Close" in frame.columns and length > 5:
        rets = pd.Series(frame["Close"]).astype(float).pct_change().dropna()
        if not rets.empty and rets.std() > 0:
            z = (rets - rets.mean()) / rets.std()
            outlier_frac = (z.abs() > 4).mean()

    # Simple scoring: penalize missingness and outliers, reward coverage
    quality_score = max(0.0, min(1.0, 1.0 - missing_pct * 2 - outlier_frac - max(0, 1 - coverage)))

    return {
        "length": length,
        "missing_pct": missing_pct,
        "coverage": coverage,
        "outlier_frac": outlier_frac,
        "quality_score": quality_score,
    }


def _emit_dashboard_json(
    path: Path,
    meta: Dict[str, Any],
    summary: Dict[str, Any],
    routing_stats: Dict[str, Any],
    equity_points: list[Dict[str, Any]],
    realized_equity_points: Optional[list[Dict[str, Any]]] = None,
    recent_signals: Optional[list[Dict[str, Any]]] = None,
    win_rate: float = 0.0,
    latencies: Optional[Dict[str, float]] = None,
    quality_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the latest run snapshot for the HTML dashboard."""
    def _json_safe(obj: Any) -> Any:
        """Recursively convert datetimes/pd.Timestamps to ISO strings for JSON dump."""
        from pandas import Timestamp  # lazy import to avoid circulars

        if isinstance(obj, (datetime, Timestamp)):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj

    payload = {
        "meta": meta,
        "pnl": {"absolute": summary["pnl_dollars"], "pct": summary["pnl_pct"]},
        "win_rate": win_rate,
        "trade_count": summary["trades"],
        "latency": latencies or {},
        "routing": {
            "ts_signals": routing_stats.get("time_series_signals", 0),
            "llm_signals": routing_stats.get("llm_fallback_signals", 0),
            "fallback_used": routing_stats.get("llm_fallback_signals", 0),
        },
        "quality": quality_summary or {},
        "equity": equity_points,
        "equity_realized": realized_equity_points or [],
        "signals": (recent_signals or [])[-20:],  # keep it small
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2)
        logger.info("Dashboard data emitted to %s", path)
    except Exception as exc:
        logger.warning("Unable to emit dashboard data: %s", exc)


def _emit_dashboard_png(path: Path, equity_points: list[Dict[str, Any]], pnl_pct: float) -> None:
    """Render a simple equity curve PNG for quick previews."""
    try:
        if not equity_points:
            return
        xs = [p["t"] for p in equity_points]
        ys = [p["v"] for p in equity_points]
        plt.figure(figsize=(6, 3))
        plt.plot(xs, ys, color="#25c2a0", linewidth=2)
        plt.fill_between(xs, ys, color="#25c2a0", alpha=0.1)
        plt.title(f"Equity Curve (PnL {pnl_pct*100:.2f}%)", color="#102235")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        plt.close()
        logger.info("Dashboard PNG emitted to %s", path)
    except Exception as exc:
        logger.debug("Unable to emit dashboard PNG: %s", exc)


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
    "--include-frontier-tickers",
    is_flag=True,
    help="Append curated frontier market tickers to the provided --tickers list.",
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
    default=True,
    show_default=True,
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
    include_frontier_tickers: bool,
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _configure_logging(verbose)
    companion_config = _load_ai_companion_config()
    _activate_ai_companion_guardrails(companion_config)

    base_tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    data_source_manager = DataSourceManager()
    universe = resolve_ticker_universe(
        base_tickers=base_tickers,
        include_frontier=include_frontier_tickers,
        active_source=data_source_manager.get_active_source(),
        use_discovery=True,
    )
    ticker_list = universe.tickers
    if not ticker_list:
        raise click.UsageError("At least one ticker symbol is required (no tickers resolved).")

    logger.info(
        "Autonomous trading loop booting for tickers from %s (source=%s): %s",
        universe.universe_source or "unknown",
        universe.active_source,
        ", ".join(ticker_list),
    )
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
    equity_points: list[Dict[str, Any]] = [{"t": "start", "v": initial_capital}]
    executed_signals: list[Dict[str, Any]] = []
    quality_records: list[Dict[str, Any]] = []

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
        price_map: Dict[str, float] = {}
        recent_signals: list[Dict[str, Any]] = []
        for ticker in ticker_list:
            ticker_frame = _split_ticker_frame(raw_window, ticker)
            if ticker_frame is None or ticker_frame.empty:
                logger.warning("No data for %s; skipping.", ticker)
                continue

            raw_frame = ticker_frame.copy()
            quality = _compute_quality_metrics(raw_frame)
            try:
                data_source = getattr(data_source_manager.active_extractor, "name", None)
            except Exception:
                data_source = None
            try:
                trading_engine.db_manager.save_quality_snapshot(
                    ticker=ticker,
                    window_start=raw_frame.index.min(),
                    window_end=raw_frame.index.max(),
                    length=quality["length"],
                    missing_pct=quality["missing_pct"],
                    coverage=quality["coverage"],
                    outlier_frac=quality["outlier_frac"],
                    quality_score=quality["quality_score"],
                    source=data_source,
                )
            except Exception:
                logger.debug("Skipping quality snapshot persistence for %s", ticker)

            quality_records.append(
                {
                    "ticker": ticker,
                    "quality_score": quality["quality_score"],
                    "missing_pct": quality["missing_pct"],
                    "coverage": quality["coverage"],
                    "outlier_frac": quality["outlier_frac"],
                    "source": data_source,
                }
            )

            if quality["quality_score"] < MIN_QUALITY_SCORE:
                logger.info(
                    "Quality gate blocked %s (score=%.2f < %.2f); skipping signal.",
                    ticker,
                    quality["quality_score"],
                    MIN_QUALITY_SCORE,
                )
                continue

            ticker_frame = _ensure_min_length(preprocessor.handle_missing(ticker_frame))
            try:
                price_map[ticker] = float(ticker_frame["Close"].iloc[-1])
            except Exception:
                price_map[ticker] = None

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
                quality=quality,
                data_source=data_source,
            )

            if execution_report:
                execution_report["quality"] = quality
                execution_report["data_source"] = data_source
                cycle_results.append(execution_report)
                recent_signals.append(execution_report)
                if execution_report.get("status") == "EXECUTED":
                    executed_signals.append(execution_report)

        if price_map:
            trading_engine.mark_to_market({k: v for k, v in price_map.items() if v is not None})
        summary = _summarize_portfolio(trading_engine)
        logger.info(
            "Cycle %s complete: %s trades | Cash $%.2f | Equity $%.2f | PnL $%.2f (%.2f%%)",
            cycle,
            len(cycle_results),
            summary["cash"],
            summary["total_value"],
            summary["pnl_dollars"],
            summary["pnl_pct"] * 100,
        )

        if verbose:
            logger.debug("Positions: %s", summary["positions"])

        equity_points.append({"t": f"cycle_{cycle}", "v": summary["total_value"]})

        if cycle < cycles:
            logger.info("Sleeping %s seconds before next cycle...", sleep_seconds)
            time.sleep(sleep_seconds)

    final_summary = _summarize_portfolio(trading_engine)
    logger.info("=== Automated Trading Complete ===")
    logger.info("Total trades executed: %s", final_summary["trades"])
    logger.info(
        "Final cash: $%.2f | Portfolio value: $%.2f | PnL $%.2f (%.2f%%)",
        final_summary["cash"],
        final_summary["total_value"],
        final_summary["pnl_dollars"],
        final_summary["pnl_pct"] * 100,
    )
    if final_summary["trades"] == 0:
        logger.warning(
            "No trades were executed. Consider lowering confidence/min_return thresholds "
            "or enabling LLM fallback for redundancy."
        )

    perf_summary = trading_engine.get_performance_metrics()
    win_rate = perf_summary.get("win_rate", 0.0)

    quality_summary = {}
    if quality_records:
        scores = [q["quality_score"] for q in quality_records if q.get("quality_score") is not None]
        if scores:
            quality_summary = {
                "average": sum(scores) / len(scores),
                "minimum": min(scores),
                "records": quality_records,
            }

    # Emit dashboard snapshot for visualization
    equity_points.append({"t": "end", "v": final_summary["total_value"]})
    realized_equity_points: list[Dict[str, Any]] = []
    realized_value = trading_engine.initial_capital
    for idx, exec_sig in enumerate(executed_signals):
        rpnl = exec_sig.get("realized_pnl")
        if rpnl is None:
            continue
        realized_value += rpnl
        realized_equity_points.append(
            {
                "t": exec_sig.get("timestamp") or f"trade_{idx+1}",
                "v": realized_value,
            }
        )
    if realized_equity_points:
        realized_equity_points.insert(0, {"t": "start", "v": trading_engine.initial_capital})
        realized_equity_points.append({"t": "end", "v": realized_value})
    else:
        try:
            history = trading_engine.db_manager.get_realized_pnl_history(limit=500)
            if history:
                val = trading_engine.initial_capital
                realized_equity_points.append({"t": "start", "v": val})
                for rec in history:
                    rp = rec.get("realized_pnl") or 0.0
                    val += rp
                    realized_equity_points.append({"t": str(rec.get("trade_date")), "v": val})
                realized_equity_points.append({"t": "end", "v": val})
        except Exception:
            logger.debug("Skipping realized PnL history aggregation")
    ts_lat = signal_router.latencies.get("ts_ms", [])
    llm_lat = signal_router.latencies.get("llm_ms", [])
    latencies = {
        "ts_ms": sum(ts_lat) / len(ts_lat) if ts_lat else None,
        "llm_ms": sum(llm_lat) / len(llm_lat) if llm_lat else None,
    }
    strategy_config = None
    try:
        strategy_config = trading_engine.db_manager.get_best_strategy_config(regime="default")
    except Exception:
        logger.debug("No cached strategy configuration available for dashboard.")
    meta = {
        "run_id": run_id,
        "ts": datetime.now(UTC).isoformat(),
        "tickers": ticker_list,
        "cycles": cycles,
        "llm_enabled": bool(llm_generator),
        "strategy": strategy_config or {},
    }
    _emit_dashboard_json(
        path=DASHBOARD_DATA_PATH,
        meta=meta,
        summary=final_summary,
        routing_stats=signal_router.routing_stats,
        equity_points=equity_points,
        realized_equity_points=realized_equity_points,
        recent_signals=recent_signals if 'recent_signals' in locals() else [],
        win_rate=win_rate,
        latencies=latencies,
        quality_summary=quality_summary,
    )
    _emit_dashboard_png(
        path=ROOT_PATH / "visualizations" / "dashboard_snapshot.png",
        equity_points=equity_points,
        pnl_pct=final_summary["pnl_pct"],
    )

    # Persist per-ticker latency metrics
    for ticker, lats in signal_router.ticker_latencies.items():
        try:
            trading_engine.db_manager.save_latency_metrics(
                ticker=ticker,
                run_id=run_id,
                stage="routing",
                ts_ms=lats.get("ts_ms"),
                llm_ms=lats.get("llm_ms"),
            )
        except Exception:
            logger.debug("Skipping latency persistence for %s", ticker)
    if final_summary["positions"]:
        logger.info("Open positions: %s", final_summary["positions"])


if __name__ == "__main__":
    main()
