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
import atexit
import logging
import os
import site
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
import yaml
import json

os.environ.setdefault("MPLBACKEND", "Agg")

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
from risk.barbell_policy import BarbellConfig

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
MIN_LOOKBACK_DAYS = 365
MIN_SERIES_POINTS = 120
MIN_QUALITY_SCORE = 0.50
EXECUTION_LOG_PATH = ROOT_PATH / "logs" / "automation" / "execution_log.jsonl"
_NO_TRADE_WINDOWS = None

UTC = timezone.utc


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


def _asset_class(ticker: str) -> str:
    t = (ticker or "").upper()
    if t.endswith("=X"):
        return "FX"
    if t.endswith("-USD") or t in {"BTC", "ETH"}:
        return "CRYPTO"
    if "^" in t:
        return "INDEX"
    return "US_EQUITY"


def _get_no_trade_windows() -> dict:
    global _NO_TRADE_WINDOWS
    if _NO_TRADE_WINDOWS is not None:
        return _NO_TRADE_WINDOWS
    cfg_path = ROOT_PATH / "config" / "signal_routing_config.yml"
    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        sr = raw.get("signal_routing") or {}
        exec_cfg = sr.get("execution") or {}
        _NO_TRADE_WINDOWS = exec_cfg.get("no_trade_windows") or {}
    except Exception:
        _NO_TRADE_WINDOWS = {}
    return _NO_TRADE_WINDOWS


def _in_no_trade_window(ticker: str) -> bool:
    windows = _get_no_trade_windows()
    asset_class = _asset_class(ticker)
    slots = windows.get(asset_class) or []
    if not slots:
        return False
    now_time = datetime.now(UTC).time()
    for slot in slots:
        try:
            start_s, end_s = slot.split("-")
            start_h, start_m = [int(x) for x in start_s.split(":")]
            end_h, end_m = [int(x) for x in end_s.split(":")]
            start = now_time.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
            end = now_time.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
            if start <= now_time < end:
                return True
        except Exception:
            continue
    return False


def _compute_mid_price(frame: pd.DataFrame) -> Optional[float]:
    """Best-effort mid-price using bid/ask -> high/low -> close."""
    if frame is None or frame.empty:
        return None
    last = frame.iloc[-1]
    try:
        bid = last.get("Bid")
        ask = last.get("Ask")
        if bid is not None and ask is not None and pd.notna(bid) and pd.notna(ask):
            return float((float(bid) + float(ask)) / 2.0)
    except Exception:
        pass
    try:
        high = last.get("High")
        low = last.get("Low")
        if high is not None and low is not None and pd.notna(high) and pd.notna(low):
            return float((float(high) + float(low)) / 2.0)
    except Exception:
        pass
    try:
        close = last.get("Close")
        if close is not None and pd.notna(close):
            return float(close)
    except Exception:
        pass
    return None


def _log_execution_event(run_id: str, cycle: int, record: Dict[str, Any]) -> None:
    """Append a compact execution/skip event for slippage + no-trade audits."""
    payload = dict(record or {})
    payload.setdefault("run_id", run_id)
    payload.setdefault("cycle", cycle)
    payload.setdefault("logged_at", datetime.now(UTC).isoformat())
    try:
        EXECUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EXECUTION_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        logger.debug("Unable to log execution event for %s", record.get("ticker"))


def _llm_signals_ready_for_trading(
    tracking_path: Path | None = None,
) -> bool:
    """
    Inspect the LLM signal tracking DB and decide whether LLM fallback
    should be allowed to influence trading decisions.

    Per LLM_PERFORMANCE_REVIEW and AGENT guidance:
    - Until at least one signal has passed validation, LLM outputs are
      considered research-only and must not drive trades.
    - In diagnostic modes we always allow LLM so behaviour can be
      stress‑tested without gating.
    """
    # In diagnostic mode, keep LLM available so test runs can observe
    # behaviour without production guardrails.
    diag_mode = str(
        os.getenv("DIAGNOSTIC_MODE")
        or os.getenv("TS_DIAGNOSTIC_MODE")
        or os.getenv("EXECUTION_DIAGNOSTIC_MODE")
        or "0"
    ) == "1"
    if diag_mode:
        return True

    if tracking_path is None:
        tracking_path = ROOT_PATH / "data" / "llm_signal_tracking.json"

    if not tracking_path.exists():
        logger.info(
            "LLM tracking DB missing at %s; LLM fallback will be disabled for trading.",
            tracking_path,
        )
        return False

    try:
        raw = tracking_path.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # pragma: no cover - guardrail must be robust
        logger.warning(
            "Failed to parse LLM tracking DB at %s (%s); disabling LLM fallback.",
            tracking_path,
            exc,
        )
        return False

    meta = payload.get("metadata") or {}
    total = int(meta.get("total_signals") or 0)
    validated = int(meta.get("validated_signals") or 0)
    if total <= 0 or validated <= 0:
        logger.info(
            "LLM tracking DB reports total_signals=%s, validated_signals=%s; "
            "LLM fallback will be disabled until at least one signal passes "
            "validation.",
            total,
            validated,
        )
        return False

    logger.info(
        "LLM tracking DB healthy for trading use "
        "(total_signals=%s, validated_signals=%s).",
        total,
        validated,
    )
    return True


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
    mid_price: Optional[float] = None,
) -> Optional[Dict]:
    """Route signals and push the primary decision through the execution engine."""
    bundle = router.route_signal(
        ticker=ticker,
        forecast_bundle=forecast_bundle,
        current_price=current_price,
        market_data=market_data,
        quality=quality,
        data_source=data_source,
        mid_price=mid_price,
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

    mid_px = mid_price if mid_price is not None else _compute_mid_price(market_data)
    if result.status != "EXECUTED":
        return {
            "ticker": ticker,
            "status": result.status,
            "reason": result.reason,
            "warnings": result.validation_warnings,
            "quality": quality,
            "data_source": data_source,
            "mid_price": mid_px,
        }

    realized_pnl = getattr(result.trade, "realized_pnl", None)
    realized_pnl_pct = getattr(result.trade, "realized_pnl_pct", None)
    executed_at = result.trade.timestamp.isoformat() if result.trade else None
    entry_price = result.trade.entry_price if result.trade else current_price
    mid_slippage_bp = None
    if mid_px not in (None, 0, 0.0):
        try:
            mid_slippage_bp = ((entry_price - float(mid_px)) / float(mid_px)) * 1e4
        except Exception:
            mid_slippage_bp = None

    return {
        "ticker": ticker,
        "status": result.status,
        "shares": result.trade.shares if result.trade else 0,
        "action": result.trade.action if result.trade else primary.get("action", "HOLD"),
        "entry_price": entry_price,
        "portfolio_value": result.portfolio.total_value if result.portfolio else None,
        "signal_source": primary.get("source", "TIME_SERIES"),
        "signal_confidence": primary.get("confidence"),
        "expected_return": primary.get("expected_return"),
        "quality": quality,
        "data_source": data_source,
        "timestamp": executed_at,
        "realized_pnl": realized_pnl,
        "realized_pnl_pct": realized_pnl_pct,
        "mid_price": mid_px,
        "mid_slippage_bp": mid_slippage_bp,
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
    forecaster_health: Optional[Dict[str, Any]] = None,
    quant_validation_health: Optional[Dict[str, Any]] = None,
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
        "forecaster_health": forecaster_health or {},
        "quant_validation_health": quant_validation_health or {},
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

    # Optional barbell-aware quant gate: when barbell validation is enabled,
    # use aggregate quant_validation health to temporarily disable risk-bucket
    # tickers if recent TS signals look unhealthy. This keeps safe-bucket
    # behaviour unchanged and aligns with BARBELL_OPTIONS_MIGRATION.md.
    try:
        barbell_cfg = BarbellConfig.from_yaml()
    except Exception:
        barbell_cfg = None

    if barbell_cfg and barbell_cfg.enable_barbell_validation:
        risk_symbols = set(barbell_cfg.risk_symbols or [])
        quant_log_path = ROOT_PATH / "logs" / "signals" / "quant_validation.jsonl"
        monitoring_cfg_path = ROOT_PATH / "config" / "forecaster_monitoring.yml"

        if quant_log_path.exists() and monitoring_cfg_path.exists():
            try:
                from scripts.check_quant_validation_health import _load_entries, _summarize_global

                entries = _load_entries(quant_log_path)
                summary = _summarize_global(entries)

                fm_raw = yaml.safe_load(monitoring_cfg_path.read_text()) or {}
                fm_cfg = fm_raw.get("forecaster_monitoring") or {}
                qv_cfg = fm_cfg.get("quant_validation") or {}

                max_fail_fraction = float(qv_cfg.get("max_fail_fraction", 0.98))
                max_neg_exp_frac = float(
                    qv_cfg.get("max_negative_expected_profit_fraction", 0.50)
                )

                fail_frac = summary.fail_fraction
                neg_frac = summary.negative_expected_profit_fraction
                gate_active = (fail_frac > max_fail_fraction) or (
                    neg_frac > max_neg_exp_frac
                )

                if gate_active:
                    original_tickers = list(ticker_list)
                    gated_tickers = [
                        t for t in original_tickers if t not in risk_symbols
                    ]
                    if gated_tickers:
                        disabled = sorted(risk_symbols.intersection(original_tickers))
                        logger.warning(
                            "Barbell quant gate ACTIVE: fail_fraction=%.3f "
                            "(max=%.3f), neg_expected_profit_frac=%.3f (max=%.3f). "
                            "Temporarily disabling risk-bucket tickers: %s",
                            fail_frac,
                            max_fail_fraction,
                            neg_frac,
                            max_neg_exp_frac,
                            ", ".join(disabled) if disabled else "(none)",
                        )
                        ticker_list = gated_tickers
                    else:
                        logger.warning(
                            "Barbell quant gate would remove all tickers; "
                            "leaving universe unchanged."
                        )
            except SystemExit:
                # Helper uses SystemExit for missing/empty logs; treat as no-op.
                logger.debug(
                    "Quant validation health helper exited early; skipping barbell gate."
                )
            except Exception as exc:
                logger.warning(
                    "Failed to evaluate barbell quant gate; proceeding without it: %s",
                    exc,
                )

    data_validator = DataValidator()
    preprocessor = Preprocessor()
    trading_engine = PaperTradingEngine(initial_capital=initial_capital)
    # Ensure the WSL SQLite mirror (if used) is synchronized back to the
    # Windows-mount database even when this script exits without an explicit
    # close (e.g., Ctrl+C, exceptions).
    atexit.register(trading_engine.db_manager.close)

    llm_generator: Optional[LLMSignalGenerator]
    if enable_llm and _llm_signals_ready_for_trading():
        llm_generator = _initialize_llm_generator(llm_model)
    else:
        if enable_llm:
            logger.info(
                "LLM fallback requested but guardrails are not satisfied; "
                "running with Time Series only."
            )
        llm_generator = None

    # Load signal routing config so TS thresholds and flags stay
    # configuration-driven and aligned with the documented contract.
    routing_cfg: Dict[str, Any] = {}
    routing_cfg_path = ROOT_PATH / "config" / "signal_routing_config.yml"
    if routing_cfg_path.exists():
        try:
            with routing_cfg_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            routing_cfg = raw.get("signal_routing") or {}
        except Exception as exc:
            logger.warning("Failed to load signal routing config: %s", exc)

    router_config = {
        "time_series_primary": routing_cfg.get("time_series_primary", True),
        "llm_fallback": routing_cfg.get(
            "llm_fallback", enable_llm and llm_generator is not None
        ),
        "llm_redundancy": routing_cfg.get("llm_redundancy", False),
        "enable_samossa": routing_cfg.get("enable_samossa", True),
        "enable_sarimax": routing_cfg.get("enable_sarimax", True),
        "enable_garch": routing_cfg.get("enable_garch", True),
        "enable_mssa_rl": routing_cfg.get("enable_mssa_rl", True),
        "time_series": routing_cfg.get("time_series") or {},
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
            mid_price = _compute_mid_price(ticker_frame)

            if _in_no_trade_window(ticker):
                window_slots = _get_no_trade_windows().get(_asset_class(ticker)) or []
                skip_report = {
                    "ticker": ticker,
                    "status": "SKIPPED_NO_TRADE_WINDOW",
                    "reason": "no_trade_window",
                    "window_slots": window_slots,
                    "quality": quality,
                    "data_source": data_source,
                    "mid_price": mid_price,
                }
                cycle_results.append(skip_report)
                recent_signals.append(skip_report)
                _log_execution_event(run_id, cycle, skip_report)
                continue

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
                mid_price=mid_price,
            )

            if execution_report:
                execution_report["quality"] = quality
                execution_report["data_source"] = data_source
                if execution_report.get("mid_price") is None and mid_price is not None:
                    execution_report["mid_price"] = mid_price
                cycle_results.append(execution_report)
                recent_signals.append(execution_report)
                if execution_report.get("status") == "EXECUTED":
                    executed_signals.append(execution_report)
                _log_execution_event(run_id, cycle, execution_report)

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
    profit_factor = perf_summary.get("profit_factor", 0.0)

    # Compute forecaster health snapshot using shared monitoring thresholds so
    # dashboards, brutal CLIs, and hyperopt all speak the same language.
    forecaster_health: Dict[str, Any] = {}
    quant_health: Dict[str, Any] = {}
    monitoring_cfg_path = ROOT_PATH / "config" / "forecaster_monitoring.yml"
    if monitoring_cfg_path.exists():
        try:
            fm_raw = yaml.safe_load(monitoring_cfg_path.read_text()) or {}
            fm_cfg = fm_raw.get("forecaster_monitoring") or {}

            qv_cfg = fm_cfg.get("quant_validation") or {}
            rm_cfg = fm_cfg.get("regression_metrics") or {}

            min_pf = qv_cfg.get("min_profit_factor")
            min_wr = qv_cfg.get("min_win_rate")
            max_fail_fraction = qv_cfg.get("max_fail_fraction")
            max_neg_exp_frac = qv_cfg.get("max_negative_expected_profit_fraction")
            max_ratio = rm_cfg.get("max_rmse_ratio_vs_baseline")

            pf_ok = (
                isinstance(min_pf, (int, float))
                and isinstance(profit_factor, (int, float))
                and float(profit_factor) >= float(min_pf)
            )
            wr_ok = (
                isinstance(min_wr, (int, float))
                and isinstance(win_rate, (int, float))
                and float(win_rate) >= float(min_wr)
            )

            # Use the same regression summary helper as the brutal/optimizer path.
            end_date = datetime.now(UTC).date()
            start_date = end_date - timedelta(days=180)
            try:
                reg_window = trading_engine.db_manager.get_forecast_regression_summary(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
                reg_baseline = trading_engine.db_manager.get_forecast_regression_summary(
                    model_type="SAMOSSA"
                )
            except Exception:  # pragma: no cover - dashboard is advisory
                reg_window = {}
                reg_baseline = {}

            ens_win = (reg_window.get("ensemble") or {}) if reg_window else {}
            samossa_base = (reg_baseline.get("samossa") or {}) if reg_baseline else {}
            ensemble_rmse = ens_win.get("rmse")
            baseline_rmse = samossa_base.get("rmse")
            rmse_ratio = None
            rmse_ok = None
            if (
                isinstance(ensemble_rmse, (int, float))
                and isinstance(baseline_rmse, (int, float))
                and baseline_rmse > 0
            ):
                rmse_ratio = float(ensemble_rmse) / float(baseline_rmse)
                if isinstance(max_ratio, (int, float)):
                    rmse_ok = rmse_ratio <= float(max_ratio)

            forecaster_health = {
                "thresholds": {
                    "profit_factor_min": min_pf,
                    "win_rate_min": min_wr,
                    "max_fail_fraction": max_fail_fraction,
                    "max_negative_expected_profit_fraction": max_neg_exp_frac,
                    "rmse_ratio_max": max_ratio,
                },
                "metrics": {
                    "profit_factor": float(profit_factor) if isinstance(profit_factor, (int, float)) else None,
                    "win_rate": float(win_rate) if isinstance(win_rate, (int, float)) else None,
                    "rmse": {
                        "ensemble": ensemble_rmse,
                        "baseline": baseline_rmse,
                        "ratio": rmse_ratio,
                    },
                },
                "status": {
                    "profit_factor_ok": pf_ok if isinstance(pf_ok, bool) else None,
                    "win_rate_ok": wr_ok if isinstance(wr_ok, bool) else None,
                    "rmse_ok": rmse_ok,
                },
            }

            # Attach a lightweight quant validation health summary derived
            # from logs/signals/quant_validation.jsonl so dashboards and
            # monitoring jobs can see aggregate PASS/FAIL behaviour.
            quant_log_path = ROOT_PATH / "logs" / "signals" / "quant_validation.jsonl"
            if quant_log_path.exists():
                try:
                    from scripts.check_quant_validation_health import _load_entries, _summarize_global

                    entries = _load_entries(quant_log_path)
                    summary = _summarize_global(entries)
                    quant_health = {
                        "total": summary.total,
                        "pass_count": summary.pass_count,
                        "fail_count": summary.fail_count,
                        "fail_fraction": summary.fail_fraction,
                        "negative_expected_profit_fraction": summary.negative_expected_profit_fraction,
                        "max_fail_fraction": max_fail_fraction,
                        "max_negative_expected_profit_fraction": max_neg_exp_frac,
                    }
                except SystemExit:
                    # If helper exits early due to empty/missing logs, leave
                    # quant_health empty; the dashboard is advisory only.
                    quant_health = {}
        except Exception:  # pragma: no cover - dashboard is advisory
            forecaster_health = {}
            quant_health = {}

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
        forecaster_health=forecaster_health,
        quant_validation_health=quant_health,
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

    # Close to flush + sync any WSL mirror back to the canonical DB path.
    try:
        trading_engine.db_manager.close()
    except Exception:
        logger.debug("Unable to close trading database cleanly", exc_info=True)


if __name__ == "__main__":
    main()
