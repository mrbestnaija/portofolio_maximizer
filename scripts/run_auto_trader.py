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
import re
import site
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import yaml
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

try:  # Optional GPU acceleration path (torch)
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

ROOT_PATH = Path(__file__).resolve().parent.parent
site.addsitedir(str(ROOT_PATH))
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from etl.data_source_manager import DataSourceManager
from etl.data_validator import DataValidator
from etl.data_storage import DataStorage
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
MIN_LOOKBACK_DAYS_DAILY = 365
MIN_LOOKBACK_DAYS_INTRADAY = 30
MIN_SERIES_POINTS = 120
MIN_QUALITY_SCORE = 0.50
EXECUTION_LOG_PATH = ROOT_PATH / "logs" / "automation" / "execution_log.jsonl"
RUN_SUMMARY_LOG_PATH = ROOT_PATH / "logs" / "automation" / "run_summary.jsonl"
_NO_TRADE_WINDOWS = None
DEFAULT_BAR_STATE_PATH = ROOT_PATH / "logs" / "automation" / "bar_state.json"
PERFORMANCE_LOG_DIR = ROOT_PATH / "logs" / "performance"

UTC = timezone.utc
_GPU_PARALLEL_ENABLED: Optional[bool] = None
_INTRADAY_SARIMAX_PROFILE_LOGGED = False


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _env_flag(name: str) -> Optional[bool]:
    """Parse common truthy/falsey environment flags; returns None when unset."""
    raw = os.getenv(name)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _parse_int_tuple(raw: Optional[str], *, expected: int) -> Optional[Tuple[int, ...]]:
    """Parse comma-delimited integers from env-style strings."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != expected:
        return None
    try:
        values = tuple(int(p) for p in parts)
    except Exception:
        return None
    if any(v < 0 for v in values):
        return None
    return values


def _extract_last_bar_timestamp(frame: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Return the last observed bar timestamp for a ticker frame."""
    if frame is None or frame.empty:
        return None
    try:
        idx = pd.DatetimeIndex(frame.index)
        if idx.empty:
            return None
        return pd.Timestamp(idx[-1])
    except Exception:
        try:
            ts = pd.to_datetime(frame.index[-1], errors="coerce")
            return None if ts is pd.NaT else pd.Timestamp(ts)
        except Exception:
            return None


def _format_bar_timestamp(ts: pd.Timestamp) -> str:
    if ts is None or ts is pd.NaT:  # type: ignore[truthy-bool]
        return ""
    try:
        if ts.tzinfo is not None:
            ts = ts.tz_convert(UTC)
    except Exception:
        pass
    try:
        return ts.to_pydatetime().isoformat()
    except Exception:
        return str(ts)


class BarTimestampGate:
    """Tracks last-seen bars per ticker to keep trading bar-aware."""

    def __init__(self, *, state_path: Path = DEFAULT_BAR_STATE_PATH, persist: bool = False) -> None:
        self.state_path = Path(state_path)
        self.persist = bool(persist)
        self._last_seen: Dict[str, str] = {}
        if self.persist:
            self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        payload = raw.get("tickers") if isinstance(raw.get("tickers"), dict) else raw
        if not isinstance(payload, dict):
            return
        cleaned: Dict[str, str] = {}
        for key, value in payload.items():
            if not key:
                continue
            if isinstance(value, str) and value.strip():
                cleaned[str(key).upper()] = value.strip()
        self._last_seen = cleaned

    def _save(self) -> None:
        if not self.persist:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now(UTC).isoformat(),
                "tickers": dict(sorted(self._last_seen.items())),
            }
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Unable to persist bar-state to %s", self.state_path, exc_info=True)

    def check(self, ticker: str, bar_ts: pd.Timestamp) -> Tuple[bool, Optional[str], str]:
        """
        Return (is_new_bar, previous_bar_timestamp, current_bar_timestamp).
        """
        symbol = (ticker or "").upper()
        current = _format_bar_timestamp(bar_ts)
        previous = self._last_seen.get(symbol)
        if previous == current and current:
            return False, previous, current
        if symbol and current:
            self._last_seen[symbol] = current
            self._save()
        return True, previous, current


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


def _log_run_summary(record: Dict[str, Any]) -> None:
    """Persist a single-line run summary for quick auditing and dashboards."""
    payload = dict(record or {})
    payload.setdefault("logged_at", datetime.now(UTC).isoformat())
    try:
        RUN_SUMMARY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RUN_SUMMARY_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        logger.debug("Unable to append run summary log", exc_info=True)


def _build_action_plan(
    pnl_dollars: float,
    profit_factor: float,
    win_rate: float,
    realized_trades: int,
    cash_ratio: Optional[float],
    forecaster_health: Dict[str, Any],
    quant_health: Dict[str, Any],
) -> list[str]:
    """Translate profitability/liquidity/forecast health into next-step prompts."""
    actions: list[str] = []

    try:
        realized_trades_n = int(realized_trades or 0)
    except (TypeError, ValueError):
        realized_trades_n = 0

    if realized_trades_n <= 0:
        # With no realized exits, profit_factor and win_rate are not meaningful yet.
        if pnl_dollars < 0:
            actions.append(
                "PnL negative (cost/unrealized drag); tighten sizing and ensure exits trigger before judging edge."
            )
        else:
            actions.append(
                "No realized trades yet; extend run or trigger lifecycle exits before judging PF/WR."
            )
    else:
        if pnl_dollars < 0 or (isinstance(profit_factor, (int, float)) and profit_factor < 1.0):
            actions.append(
                "Tighten position sizing and review signal thresholds; profitability below break-even."
            )
        elif isinstance(profit_factor, (int, float)) and profit_factor >= 1.2 and pnl_dollars > 0:
            actions.append(
                "Profitability trending positive; keep current risk budget and monitor for drift."
            )

    if cash_ratio is not None:
        if cash_ratio < 0.10:
            actions.append(
                f"Liquidity low (cash ratio {cash_ratio:.1%}); trim/scale exits to free capital."
            )
        elif cash_ratio > 0.60 and pnl_dollars > 0:
            actions.append(
                f"High idle cash ({cash_ratio:.1%}); redeploy gradually if signals stay healthy."
            )

    status = forecaster_health.get("status") if isinstance(forecaster_health, dict) else {}
    metrics = forecaster_health.get("metrics") if isinstance(forecaster_health, dict) else {}
    thresholds = forecaster_health.get("thresholds") if isinstance(forecaster_health, dict) else {}
    rmse = (metrics.get("rmse") or {}) if isinstance(metrics, dict) else {}

    if isinstance(status, dict):
        if status.get("profit_factor_ok") is False or status.get("win_rate_ok") is False:
            actions.append(
                "Forecast-driven hit-rate below target; rerun hyperopt or tighten quant validation."
            )
        if status.get("rmse_ok") is False:
            ratio = rmse.get("ratio")
            max_ratio = thresholds.get("rmse_ratio_max") if isinstance(thresholds, dict) else None
            actions.append(
                f"Model drift detected (RMSE ratio {ratio} > {max_ratio}); retrain or refresh features."
            )

    if isinstance(quant_health, dict):
        fail_frac = quant_health.get("fail_fraction")
        max_fail = quant_health.get("max_fail_fraction")
        if isinstance(fail_frac, (int, float)) and isinstance(max_fail, (int, float)) and fail_frac > max_fail:
            actions.append(
                f"Quant validation failing {fail_frac:.2f}>{max_fail:.2f}; pause risky buckets until signals improve."
            )

    if not actions:
        actions.append(
            "Metrics within thresholds; continue current playbook and monitor dashboards."
        )

    return actions


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
    frames = _build_ticker_frame_map(data, [ticker])
    return frames.get(ticker.upper())


def _build_ticker_frame_map(data: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Vectorized ticker slicing: build a map of {TICKER: frame} once per window to
    avoid repeated per-ticker scans.
    """
    if data is None or data.empty:
        return {}

    tickers_norm = {t.upper() for t in tickers if t}
    if not tickers_norm:
        return {}

    ticker_col = "ticker" if "ticker" in data.columns else "Ticker" if "Ticker" in data.columns else None
    if ticker_col is None:
        logger.warning("Ticker column missing; cannot isolate %s", ticker)
        return {}

    ticker_upper = data[ticker_col].astype(str).str.upper()
    mask = ticker_upper.isin(tickers_norm)
    if not bool(mask.any()):
        return {}

    sliced = data.loc[mask].copy()
    sliced.index = pd.to_datetime(sliced.index)
    sliced["_TICKER_UPPER"] = ticker_upper[mask].values
    sliced.sort_index(inplace=True)

    frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in sliced.groupby("_TICKER_UPPER", sort=False):
        trimmed = frame.drop(columns=[ticker_col, "_TICKER_UPPER"], errors="ignore")
        frames[str(symbol).upper()] = trimmed
    return frames


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


def _is_intraday_interval(interval: Optional[str]) -> bool:
    if not interval:
        return False
    interval = str(interval).strip().lower()
    # Treat minute/hour intervals as intraday. Everything with day/week/month
    # semantics is treated as daily-or-slower.
    return bool(re.match(r"^\\d+\\s*(m|h|min|hour|hours)$", interval)) or (
        any(token in interval for token in ("m", "h")) and not any(token in interval for token in ("d", "wk", "mo"))
    )


def _effective_min_lookback_days(interval: Optional[str]) -> int:
    return MIN_LOOKBACK_DAYS_INTRADAY if _is_intraday_interval(interval) else MIN_LOOKBACK_DAYS_DAILY


def _sarimax_kwargs_for_interval(interval: Optional[str]) -> Dict[str, Any]:
    """
    Return SARIMAX kwargs appropriate for the current yfinance interval.

    Intraday grids are disabled by default because SARIMAX order search is too
    slow for multi-ticker evaluation runs. Re-enable by setting
    PMX_SARIMAX_AUTO_SELECT=1.
    """
    if not _is_intraday_interval(interval):
        return {}

    auto_select = _env_flag("PMX_SARIMAX_AUTO_SELECT")
    if auto_select is None:
        auto_select = False
    if auto_select:
        return {}

    manual_order = _parse_int_tuple(os.getenv("PMX_SARIMAX_MANUAL_ORDER"), expected=3) or (1, 1, 0)
    manual_seasonal = _parse_int_tuple(os.getenv("PMX_SARIMAX_MANUAL_SEASONAL_ORDER"), expected=4) or (0, 0, 0, 0)

    global _INTRADAY_SARIMAX_PROFILE_LOGGED
    if not _INTRADAY_SARIMAX_PROFILE_LOGGED:
        logger.info(
            "Intraday interval detected (%s): using fixed SARIMAX order=%s seasonal=%s "
            "(set PMX_SARIMAX_AUTO_SELECT=1 to re-enable order search).",
            interval,
            manual_order,
            manual_seasonal,
        )
        _INTRADAY_SARIMAX_PROFILE_LOGGED = True

    return {
        "auto_select": False,
        "manual_order": manual_order,
        "manual_seasonal_order": manual_seasonal,
    }


def _parse_int_env(name: str) -> Optional[int]:
    """Return positive int from env if set; otherwise None."""
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        val = int(str(raw).strip())
        return val if val > 0 else None
    except Exception:
        return None


def _downcast_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory footprint while preserving values.
    Keeps datetime index intact; returns a new DataFrame.
    """
    if frame is None or frame.empty:
        return frame
    df = frame.copy()
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_integer_dtype(series):
            df[col] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            df[col] = pd.to_numeric(series, downcast="float")
    return df


def _estimate_frame_mb(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty:
        return 0.0
    try:
        return float(frame.memory_usage(deep=True).sum()) / 1_000_000.0
    except Exception:
        return 0.0


def _write_performance_artifact(kind: str, payload: Dict[str, Any]) -> None:
    """
    Write a small JSON artifact when PERFORMANCE_MONITORING=1.
    """
    if str(os.getenv("PERFORMANCE_MONITORING") or "0") != "1":
        return
    try:
        PERFORMANCE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = PERFORMANCE_LOG_DIR / f"{kind}_{ts}.json"
        safe = dict(payload or {})
        safe.setdefault("kind", kind)
        safe.setdefault("timestamp", datetime.now(UTC).isoformat())
        path.write_text(json.dumps(safe, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote performance artifact: %s", path)
    except Exception:
        logger.debug("Unable to write performance artifact %s", kind, exc_info=True)


def _gpu_parallel_enabled() -> bool:
    """Return True when torch+CUDA is available and GPU parallel is enabled."""
    global _GPU_PARALLEL_ENABLED
    if _GPU_PARALLEL_ENABLED is not None:
        return _GPU_PARALLEL_ENABLED

    enabled = _env_flag("ENABLE_GPU_PARALLEL")
    if enabled is None:
        enabled = True

    if not enabled or torch is None:
        _GPU_PARALLEL_ENABLED = False
        return False

    try:
        _GPU_PARALLEL_ENABLED = bool(torch.cuda.is_available())
    except Exception:
        _GPU_PARALLEL_ENABLED = False
    return _GPU_PARALLEL_ENABLED


def _prepare_market_window(
    manager: DataSourceManager,
    tickers: List[str],
    lookback_days: int,
) -> pd.DataFrame:
    """Fetch the latest OHLCV window for all tickers."""
    end_date = datetime.now(UTC).date()
    interval = getattr(getattr(manager, "active_extractor", None), "interval", None)
    min_lookback_days = _effective_min_lookback_days(interval)
    start_date = end_date - timedelta(days=max(lookback_days, min_lookback_days))
    logger.info(
        "Fetching OHLCV window (interval=%s): %s -> %s",
        interval or "unknown",
        start_date,
        end_date,
    )
    chunk_size = _parse_int_env("DATA_SOURCE_CHUNK_SIZE")
    t0 = time.perf_counter()
    window = manager.extract_ohlcv(
        tickers=tickers,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        chunk_size=chunk_size,
    )
    t1 = time.perf_counter()
    before_mb = _estimate_frame_mb(window)
    downcasted = _downcast_numeric_frame(window)
    after_mb = _estimate_frame_mb(downcasted)
    _write_performance_artifact(
        "memory_profile",
        {
            "stage": "prepare_market_window",
            "tickers": len(tickers),
            "chunk_size": chunk_size,
            "rows": int(len(window)) if isinstance(window, pd.DataFrame) else 0,
            "fetch_seconds": t1 - t0,
            "memory_before_mb": before_mb,
            "memory_after_mb": after_mb,
        },
    )
    return downcasted


def _generate_time_series_forecast(
    price_frame: pd.DataFrame,
    horizon: int,
    *,
    interval: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[float]]:
    """Fit the ensemble forecaster and return the forecast bundle + latest price."""
    if "Close" not in price_frame.columns:
        logger.warning("Close column missing; skipping forecasting.")
        return None, None

    close_series = price_frame["Close"].astype(float)
    clean_close = close_series.dropna()
    if _gpu_parallel_enabled() and len(clean_close) > 1:
        try:
            values = torch.as_tensor(clean_close.to_numpy(dtype=float, copy=False), device="cuda")  # type: ignore
            returns = (values[1:] / values[:-1]) - 1.0
            returns_series = pd.Series(returns.detach().cpu().numpy(), index=clean_close.index[1:])
        except Exception:
            returns_series = clean_close.pct_change().dropna()
    else:
        returns_series = clean_close.pct_change().dropna()

    try:
        resolved_interval = interval or os.getenv("YFINANCE_INTERVAL")
        mssa_use_gpu = _env_flag("MSSA_RL_USE_GPU")
        if mssa_use_gpu is None:
            mssa_use_gpu = _gpu_parallel_enabled()

        forecaster = TimeSeriesForecaster(
            config=TimeSeriesForecasterConfig(
                forecast_horizon=horizon,
                sarimax_kwargs=_sarimax_kwargs_for_interval(resolved_interval),
                samossa_kwargs={"forecast_horizon": int(horizon)},
                mssa_rl_kwargs={
                    "forecast_horizon": int(horizon),
                    "use_gpu": bool(mssa_use_gpu),
                },
            )
        )
        forecaster.fit(price_series=close_series, returns_series=returns_series)
        forecast_bundle = forecaster.forecast()
    except Exception as exc:
        logger.error("Forecasting failed: %s", exc)
        return None, None

    current_price = float(close_series.iloc[-1])
    return forecast_bundle, current_price


def _generate_forecasts_bulk(
    frames_by_ticker: Dict[str, pd.DataFrame],
    horizon: int,
    *,
    parallel: bool,
    max_workers: Optional[int] = None,
    interval: Optional[str] = None,
) -> Dict[str, Tuple[Optional[Dict], Optional[float]]]:
    """
    Generate forecasts for many tickers, optionally in parallel.
    Returns a map {TICKER: (forecast_bundle, current_price)}.
    """
    tickers = list(frames_by_ticker.keys())
    if not tickers:
        return {}

    if not parallel:
        out: Dict[str, Tuple[Optional[Dict], Optional[float]]] = {}
        for symbol in tickers:
            out[symbol] = _generate_time_series_forecast(
                frames_by_ticker[symbol],
                horizon,
                interval=interval,
            )
        return out

    workers = max_workers or min(4, max(1, len(tickers)))
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _generate_time_series_forecast,
                frames_by_ticker[symbol],
                horizon,
                interval=interval,
            ): symbol
            for symbol in tickers
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                out[symbol] = future.result()
            except Exception:
                out[symbol] = (None, None)
    return out


def _prepare_ticker_candidate(
    *,
    ticker: str,
    frame: pd.DataFrame,
    preprocessor: Preprocessor,
) -> Dict[str, Any]:
    """Compute quality, preprocess, and derive mid-price for a ticker frame."""
    raw_frame = frame.copy()
    quality = _compute_quality_metrics(raw_frame)
    processed = _ensure_min_length(preprocessor.handle_missing(frame))
    mid_price = _compute_mid_price(processed)
    return {
        "ticker": ticker,
        "symbol": ticker.upper(),
        "raw_frame": raw_frame,
        "frame": processed,
        "quality": quality,
        "mid_price": mid_price,
    }


def _build_ticker_candidates(
    entries: List[Dict[str, Any]],
    *,
    preprocessor: Preprocessor,
    parallel: bool,
    max_workers: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Prepare per-ticker candidates (quality + preprocessed frames), optionally in parallel.
    Each entry should include {"ticker": str, "frame": DataFrame, "order": int, ...}.
    """
    if not entries:
        return []

    if not parallel:
        out: List[Dict[str, Any]] = []
        for entry in entries:
            candidate = _prepare_ticker_candidate(
                ticker=entry["ticker"],
                frame=entry["frame"],
                preprocessor=preprocessor,
            )
            candidate.update(entry)
            out.append(candidate)
        return sorted(out, key=lambda item: item["order"])

    workers = max_workers or min(4, max(1, len(entries)))
    out_map: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _prepare_ticker_candidate,
                ticker=entry["ticker"],
                frame=entry["frame"],
                preprocessor=preprocessor,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                candidate = future.result()
            except Exception:
                candidate = {
                    "ticker": entry["ticker"],
                    "symbol": entry["ticker"].upper(),
                    "raw_frame": entry["frame"],
                    "frame": entry["frame"],
                    "quality": {"quality_score": 0.0, "missing_pct": 1.0, "coverage": 0.0, "outlier_frac": 0.0},
                    "mid_price": None,
                }
            candidate.update(entry)
            out_map[entry["order"]] = candidate
    return [out_map[idx] for idx in sorted(out_map)]


def _prepare_candidate_with_forecast(
    *,
    ticker: str,
    frame: pd.DataFrame,
    preprocessor: Preprocessor,
    horizon: int,
) -> Dict[str, Any]:
    candidate = _prepare_ticker_candidate(ticker=ticker, frame=frame, preprocessor=preprocessor)
    forecast_bundle, current_price = _generate_time_series_forecast(candidate["frame"], horizon)
    candidate["forecast_bundle"] = forecast_bundle
    candidate["current_price"] = current_price
    return candidate


def _build_candidates_with_forecasts(
    entries: List[Dict[str, Any]],
    *,
    preprocessor: Preprocessor,
    horizon: int,
    parallel: bool,
    max_workers: Optional[int],
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    if not parallel:
        out: List[Dict[str, Any]] = []
        for entry in entries:
            candidate = _prepare_candidate_with_forecast(
                ticker=entry["ticker"],
                frame=entry["frame"],
                preprocessor=preprocessor,
                horizon=horizon,
            )
            candidate.update(entry)
            out.append(candidate)
        return sorted(out, key=lambda item: item["order"])

    workers = max_workers or min(4, max(1, len(entries)))
    out_map: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _prepare_candidate_with_forecast,
                ticker=entry["ticker"],
                frame=entry["frame"],
                preprocessor=preprocessor,
                horizon=horizon,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                candidate = future.result()
            except Exception:
                candidate = {
                    "ticker": entry["ticker"],
                    "symbol": entry["ticker"].upper(),
                    "raw_frame": entry["frame"],
                    "frame": entry["frame"],
                    "quality": {"quality_score": 0.0, "missing_pct": 1.0, "coverage": 0.0, "outlier_frac": 0.0},
                    "mid_price": None,
                    "forecast_bundle": None,
                    "current_price": None,
                }
            candidate.update(entry)
            out_map[entry["order"]] = candidate
    return [out_map[idx] for idx in sorted(out_map)]


def _extract_forecast_scalar(payload: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (forecast_value, lower_ci, upper_ci) using horizon-end semantics when available."""
    if not isinstance(payload, dict) or not payload:
        return None, None, None
    series = payload.get("forecast")
    forecast_value: Optional[float] = None
    if isinstance(series, pd.Series) and not series.empty:
        cleaned = series.dropna()
        if not cleaned.empty:
            try:
                forecast_value = float(cleaned.iloc[-1])
            except Exception:
                forecast_value = None
    elif isinstance(series, (int, float)):
        forecast_value = float(series)

    def _ci_val(key: str) -> Optional[float]:
        val = payload.get(key)
        if isinstance(val, pd.Series) and not val.empty:
            cleaned = val.dropna()
            if cleaned.empty:
                return None
            try:
                return float(cleaned.iloc[-1])
            except Exception:
                return None
        if isinstance(val, (int, float)):
            return float(val)
        return None

    return forecast_value, _ci_val("lower_ci"), _ci_val("upper_ci")


def _persist_forecast_snapshots(
    *,
    db_manager: DatabaseManager,
    ticker: str,
    bar_ts: pd.Timestamp,
    forecast_bundle: Dict[str, Any],
) -> None:
    """Persist horizon-end forecast snapshots so forecaster health is run-fresh."""
    if not db_manager or not isinstance(forecast_bundle, dict) or not forecast_bundle:
        return
    forecast_date = bar_ts.date().isoformat() if hasattr(bar_ts, "date") else str(bar_ts)
    horizon = int(forecast_bundle.get("horizon") or 0) or None

    ensemble_payload = forecast_bundle.get("ensemble_forecast")
    samossa_payload = forecast_bundle.get("samossa_forecast")
    sarimax_payload = forecast_bundle.get("sarimax_forecast")

    for model_type, payload in (
        ("COMBINED", ensemble_payload),
        ("SAMOSSA", samossa_payload),
        ("SARIMAX", sarimax_payload),
    ):
        if not isinstance(payload, dict) or not payload:
            continue
        forecast_value, lower_ci, upper_ci = _extract_forecast_scalar(payload)
        if forecast_value is None:
            continue
        forecast_data = {
            "model_type": model_type,
            "forecast_horizon": int(horizon or forecast_bundle.get("horizon") or 1),
            "forecast_value": float(forecast_value),
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "model_order": {},
            "diagnostics": {
                "ensemble_metadata": forecast_bundle.get("ensemble_metadata") or {},
                "model_errors": forecast_bundle.get("model_errors") or {},
            },
            # regression_metrics will be populated via lagged evaluation.
            "regression_metrics": forecast_bundle.get("regression_metrics", {}).get(model_type.lower())
            if isinstance(forecast_bundle.get("regression_metrics"), dict)
            else None,
        }
        try:
            db_manager.save_forecast(ticker, forecast_date, forecast_data)
        except Exception:
            logger.debug("Failed to persist %s forecast snapshot for %s", model_type, ticker, exc_info=True)


def _backfill_forecast_regression_metrics(
    *,
    db_manager: DatabaseManager,
    ticker: str,
    close_series: pd.Series,
    model_types: Optional[List[str]] = None,
    max_updates: int = 50,
) -> int:
    """
    Best-effort lagged evaluation of stored horizon-end forecasts.

    When realised prices for (forecast_date + horizon bars) are available in the
    current close_series window, write a per-forecast regression_metrics payload
    so forecaster_health is not stale.
    """
    if not db_manager or close_series is None or close_series.empty:
        return 0
    types = model_types or ["COMBINED", "SAMOSSA"]
    rows = db_manager.get_forecasts(ticker, model_types=types, limit=500)  # type: ignore[arg-type]
    if not rows:
        return 0

    idx = pd.to_datetime(close_series.index, errors="coerce")
    normalized = pd.DatetimeIndex(idx).normalize()
    pos_map: Dict[pd.Timestamp, int] = {}
    for i, ts in enumerate(normalized):
        if ts is pd.NaT:
            continue
        # Keep first occurrence for stability.
        if ts not in pos_map:
            pos_map[ts] = i

    updates = 0
    eps = 1e-9
    for row in rows:
        if updates >= int(max_updates):
            break
        raw_metrics = row.get("regression_metrics")
        if isinstance(raw_metrics, str) and '"rmse"' in raw_metrics:
            continue
        try:
            forecast_id = int(row.get("id") or 0)
        except Exception:
            continue
        if forecast_id <= 0:
            continue
        try:
            horizon = int(row.get("forecast_horizon") or 0)
        except Exception:
            continue
        if horizon <= 0:
            continue
        try:
            forecast_value = float(row.get("forecast_value"))
        except Exception:
            continue
        forecast_date = row.get("forecast_date")
        if not forecast_date:
            continue
        try:
            anchor = pd.to_datetime(forecast_date, errors="coerce").normalize()
        except Exception:
            continue
        if anchor is pd.NaT:
            continue
        pos = pos_map.get(anchor)
        if pos is None:
            continue
        target_pos = pos + horizon
        if target_pos >= len(close_series):
            continue
        try:
            anchor_price = float(close_series.iloc[pos])
            actual_target = float(close_series.iloc[target_pos])
        except Exception:
            continue

        abs_err = abs(forecast_value - actual_target)
        smape = 2.0 * abs_err / max(abs(actual_target) + abs(forecast_value), eps)
        direction_pred = float(np.sign(forecast_value - anchor_price))
        direction_real = float(np.sign(actual_target - anchor_price))
        dir_acc = 1.0 if direction_pred == direction_real else 0.0
        metrics = {
            "rmse": abs_err,
            "smape": smape,
            "tracking_error": abs_err,
            "directional_accuracy": dir_acc,
            "n_observations": 1,
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        try:
            if db_manager.update_forecast_regression_metrics(forecast_id, metrics):
                updates += 1
        except Exception:
            continue

    return updates


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
    run_id: Optional[str] = None,
    execution_mode: Optional[str] = None,
    synthetic_dataset_id: Optional[str] = None,
    synthetic_generator_version: Optional[str] = None,
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

    primary_payload = dict(primary)
    if run_id:
        primary_payload["run_id"] = run_id
    if execution_mode:
        primary_payload["execution_mode"] = execution_mode
    if synthetic_dataset_id:
        primary_payload["synthetic_dataset_id"] = synthetic_dataset_id
    if synthetic_generator_version:
        primary_payload["synthetic_generator_version"] = synthetic_generator_version

    result = trading_engine.execute_signal(primary_payload, market_data)
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
        try:
            inferred = pd.infer_freq(idx)
        except ValueError:
            inferred = None
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
    "--bar-aware/--no-bar-aware",
    default=True,
    show_default=True,
    help="Only generate/execute signals when a new bar timestamp is observed per ticker.",
)
@click.option(
    "--persist-bar-state",
    is_flag=True,
    default=False,
    show_default=True,
    help="Persist last-seen bar timestamps for restart continuity.",
)
@click.option(
    "--bar-state-path",
    default=str(DEFAULT_BAR_STATE_PATH),
    show_default=True,
    help="Path to read/write persisted bar-state when enabled.",
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
@click.option(
    "--yfinance-interval",
    default=None,
    show_default=False,
    help="Optional override for yfinance interval (e.g., 1h, 30m, 1d).",
)
def main(
    tickers: str,
    include_frontier_tickers: bool,
    lookback_days: int,
    forecast_horizon: int,
    initial_capital: float,
    cycles: int,
    sleep_seconds: int,
    bar_aware: bool,
    persist_bar_state: bool,
    bar_state_path: str,
    enable_llm: bool,
    llm_model: str,
    verbose: bool,
    yfinance_interval: Optional[str] = None,
) -> None:
    """Entry point for the automated profit engine."""
    run_started_at = datetime.now(UTC)
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    _configure_logging(verbose)
    companion_config = _load_ai_companion_config()
    _activate_ai_companion_guardrails(companion_config)

    env_bar_aware = _env_flag("BAR_AWARE_TRADING")
    if env_bar_aware is not None:
        bar_aware = env_bar_aware
    env_persist_bar = _env_flag("PERSIST_BAR_STATE")
    if env_persist_bar is None:
        env_persist_bar = _env_flag("BAR_AWARE_PERSIST")
    if env_persist_bar is not None:
        persist_bar_state = env_persist_bar
    env_bar_state_path = os.getenv("BAR_STATE_PATH") or os.getenv("BAR_AWARE_STATE_PATH")
    if env_bar_state_path:
        bar_state_path = env_bar_state_path

    if yfinance_interval:
        os.environ["YFINANCE_INTERVAL"] = str(yfinance_interval).strip()

    base_tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    execution_mode = (os.getenv("EXECUTION_MODE") or "live").strip().lower()
    if execution_mode not in {"live", "synthetic", "auto"}:
        execution_mode = "live"
    enable_data_cache = _env_flag("ENABLE_DATA_CACHE")
    if enable_data_cache is None:
        enable_data_cache = False
    storage = DataStorage() if enable_data_cache else None
    data_source_manager = DataSourceManager(storage=storage, execution_mode=execution_mode)
    active_source = data_source_manager.get_active_source()
    synthetic_only = bool(_env_flag("SYNTHETIC_ONLY"))
    if active_source == "synthetic" and not synthetic_only and execution_mode != "synthetic":
        raise click.UsageError(
            "Synthetic data source selected; live evaluations require a real provider. "
            "Unset ENABLE_SYNTHETIC_PROVIDER/ENABLE_SYNTHETIC_DATA_SOURCE or set EXECUTION_MODE=live."
        )
    universe = resolve_ticker_universe(
        base_tickers=base_tickers,
        include_frontier=include_frontier_tickers,
        active_source=active_source,
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

    # Optional cache warming (no-op unless ENABLE_DATA_CACHE=1 and warming is enabled).
    warm_enabled = _env_flag("ENABLE_CACHE_WARMING")
    warm_tickers_env = os.getenv("CACHE_WARM_TICKERS")
    warm_top_n = _parse_int_env("CACHE_WARM_TOP_N")
    warm_lookback = _parse_int_env("CACHE_WARM_LOOKBACK_DAYS") or 30
    if enable_data_cache and (warm_enabled or warm_tickers_env):
        if warm_tickers_env:
            warm_list = [t.strip() for t in warm_tickers_env.split(",") if t.strip()]
        elif warm_top_n:
            warm_list = ticker_list[:warm_top_n]
        else:
            warm_list = ticker_list[: min(10, len(ticker_list))]
        try:
            logger.info("Cache warming enabled; warming %s tickers", len(warm_list))
            _prepare_market_window(data_source_manager, warm_list, warm_lookback)
        except Exception:
            logger.debug("Cache warming failed; continuing without warmed cache", exc_info=True)

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
    last_dataset_id: Optional[str] = None
    last_generator_version: Optional[str] = None
    last_execution_mode: Optional[str] = None
    bar_gate: Optional[BarTimestampGate] = None
    if bar_aware:
        bar_gate = BarTimestampGate(state_path=Path(bar_state_path), persist=persist_bar_state)
        logger.info(
            "Bar-aware trading ENABLED (persist=%s, state=%s)",
            persist_bar_state,
            Path(bar_state_path),
        )
    else:
        logger.warning("Bar-aware trading DISABLED; loop may trade repeatedly on the same bar.")

    parallel_forecasts = _env_flag("ENABLE_PARALLEL_FORECASTS")
    if parallel_forecasts is None:
        legacy_parallel = _env_flag("ENABLE_PARALLEL_TICKERS")
        parallel_forecasts = legacy_parallel if legacy_parallel is not None else True
    parallel_workers = _parse_int_env("PARALLEL_TICKER_WORKERS")
    parallel_ticker_processing = _env_flag("ENABLE_PARALLEL_TICKER_PROCESSING")
    if parallel_ticker_processing is None:
        legacy_parallel = _env_flag("ENABLE_PARALLEL_TICKERS")
        parallel_ticker_processing = legacy_parallel if legacy_parallel is not None else True
    if _gpu_parallel_enabled():
        logger.info("GPU parallel path available (torch CUDA detected).")
    else:
        logger.info("GPU parallel path unavailable; using CPU threads.")

    for cycle in range(1, cycles + 1):
        logger.info("=== Trading Cycle %s/%s ===", cycle, cycles)
        try:
            raw_window = _prepare_market_window(data_source_manager, ticker_list, lookback_days)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Market window extraction failed: %s", exc)
            if cycle < cycles:
                time.sleep(sleep_seconds)
            continue

        window_dataset_id = raw_window.attrs.get("dataset_id") if hasattr(raw_window, "attrs") else None
        window_generator_version = raw_window.attrs.get("generator_version") if hasattr(raw_window, "attrs") else None
        active_source = data_source_manager.get_active_source()
        synthetic_only = str(os.getenv("SYNTHETIC_ONLY") or "").strip() == "1"
        effective_execution_mode = "synthetic" if active_source == "synthetic" or synthetic_only else "live"
        last_dataset_id = window_dataset_id or last_dataset_id
        last_generator_version = window_generator_version or last_generator_version
        last_execution_mode = effective_execution_mode
        if cycle == 1:
            try:
                trading_engine.db_manager.record_run_provenance(
                    run_id=run_id,
                    execution_mode=effective_execution_mode,
                    data_source=active_source,
                    synthetic_dataset_id=window_dataset_id,
                    synthetic_generator_version=window_generator_version,
                    note="auto_trader",
                )
            except Exception:
                logger.debug("Failed to record run provenance", exc_info=True)

        interval = getattr(getattr(data_source_manager, "active_extractor", None), "interval", None)

        cycle_results = []
        price_map: Dict[str, float] = {}
        recent_signals: list[Dict[str, Any]] = []
        frames_by_ticker = _build_ticker_frame_map(raw_window, ticker_list)
        pre_entries: list[Dict[str, Any]] = []

        for order, ticker in enumerate(ticker_list):
            symbol = ticker.upper()
            ticker_frame = frames_by_ticker.get(symbol)
            if ticker_frame is None or ticker_frame.empty:
                logger.warning("No data for %s; skipping.", ticker)
                continue

            if bar_gate is not None:
                bar_ts = _extract_last_bar_timestamp(ticker_frame)
                if bar_ts is not None:
                    is_new_bar, prev_bar, current_bar = bar_gate.check(ticker, bar_ts)
                    if not is_new_bar:
                        skip_report = {
                            "ticker": ticker,
                            "status": "SKIPPED_SAME_BAR",
                            "reason": "same_bar",
                            "bar_timestamp": current_bar,
                            "last_processed_bar_timestamp": prev_bar,
                            "data_source": getattr(data_source_manager.active_extractor, "name", None),
                        }
                        cycle_results.append(skip_report)
                        recent_signals.append(skip_report)
                        _log_execution_event(run_id, cycle, skip_report)
                        continue

            pre_entries.append(
                {
                    "ticker": ticker,
                    "frame": ticker_frame,
                    "order": order,
                }
            )

        if parallel_ticker_processing and pre_entries:
            logger.info(
                "Parallel ticker processing enabled for %s tickers (workers=%s)",
                len(pre_entries),
                parallel_workers or "auto",
            )

        combined_parallel = bool(parallel_ticker_processing and parallel_forecasts)
        if combined_parallel:
            logger.info(
                "Parallel pipeline enabled (candidates + forecasts) for %s tickers (workers=%s)",
                len(pre_entries),
                parallel_workers or "auto",
            )
            candidates = _build_candidates_with_forecasts(
                pre_entries,
                preprocessor=preprocessor,
                horizon=forecast_horizon,
                parallel=True,
                max_workers=parallel_workers,
            )
        else:
            candidates = _build_ticker_candidates(
                pre_entries,
                preprocessor=preprocessor,
                parallel=parallel_ticker_processing,
                max_workers=parallel_workers,
            )

        forecast_inputs: Dict[str, pd.DataFrame] = {}
        for candidate in candidates:
            ticker = candidate["ticker"]
            symbol = candidate["symbol"]
            raw_frame = candidate["raw_frame"]
            ticker_frame = candidate["frame"]
            quality = candidate["quality"]
            mid_price = candidate["mid_price"]
            try:
                data_source = getattr(data_source_manager.active_extractor, "name", None)
            except Exception:
                data_source = None

            try:
                trading_engine.db_manager.save_quality_snapshot(
                    ticker=ticker,
                    window_start=raw_frame.index.min(),
                    window_end=raw_frame.index.max(),
                    length=quality.get("length", 0),
                    missing_pct=quality.get("missing_pct", 1.0),
                    coverage=quality.get("coverage", 0.0),
                    outlier_frac=quality.get("outlier_frac", 0.0),
                    quality_score=quality.get("quality_score", 0.0),
                    source=data_source,
                )
            except Exception:
                logger.debug("Skipping quality snapshot persistence for %s", ticker)

            quality_records.append(
                {
                    "ticker": ticker,
                    "quality_score": quality.get("quality_score", 0.0),
                    "missing_pct": quality.get("missing_pct", 1.0),
                    "coverage": quality.get("coverage", 0.0),
                    "outlier_frac": quality.get("outlier_frac", 0.0),
                    "source": data_source,
                }
            )

            if quality.get("quality_score", 0.0) < MIN_QUALITY_SCORE:
                logger.info(
                    "Quality gate blocked %s (score=%.2f < %.2f); skipping signal.",
                    ticker,
                    quality.get("quality_score", 0.0),
                    MIN_QUALITY_SCORE,
                )
                continue

            try:
                price_map[ticker] = float(ticker_frame["Close"].iloc[-1])
            except Exception:
                price_map[ticker] = None

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

            candidate["data_source"] = data_source
            if not combined_parallel:
                forecast_inputs[symbol] = ticker_frame

        forecast_map: Dict[str, Tuple[Optional[Dict], Optional[float]]] = {}
        if parallel_forecasts and forecast_inputs:
            logger.info(
                "Parallel forecasting enabled for %s tickers (workers=%s)",
                len(forecast_inputs),
                parallel_workers or "auto",
            )
            forecast_map = _generate_forecasts_bulk(
                forecast_inputs,
                forecast_horizon,
                parallel=True,
                max_workers=parallel_workers,
                interval=interval,
            )

        for candidate in candidates:
            ticker = candidate["ticker"]
            symbol = candidate["symbol"]
            ticker_frame = candidate["frame"]
            quality = candidate["quality"]
            data_source = candidate.get("data_source")
            mid_price = candidate["mid_price"]

            # Lagged evaluation of historical forecasts (best-effort).
            try:
                updated = _backfill_forecast_regression_metrics(
                    db_manager=trading_engine.db_manager,
                    ticker=ticker,
                    close_series=ticker_frame["Close"].astype(float),
                )
                if updated:
                    logger.debug("Updated %s forecast regression rows for %s", updated, ticker)
            except Exception:
                logger.debug("Skipping forecast regression backfill for %s", ticker, exc_info=True)

            if "forecast_bundle" in candidate:
                forecast_bundle = candidate.get("forecast_bundle")
                current_price = candidate.get("current_price")
            elif forecast_map:
                forecast_bundle, current_price = forecast_map.get(symbol, (None, None))
            else:
                forecast_bundle, current_price = _generate_time_series_forecast(
                    ticker_frame,
                    forecast_horizon,
                    interval=interval,
                )

            if not forecast_bundle or current_price is None:
                logger.warning("Forecasting failed for %s; skipping.", ticker)
                continue

            # Persist forecast snapshots so monitoring health uses fresh data.
            try:
                last_bar_ts = _extract_last_bar_timestamp(ticker_frame)
                if last_bar_ts is not None:
                    _persist_forecast_snapshots(
                        db_manager=trading_engine.db_manager,
                        ticker=ticker,
                        bar_ts=last_bar_ts,
                        forecast_bundle=forecast_bundle,
                    )
            except Exception:
                logger.debug("Skipping forecast snapshot persistence for %s", ticker, exc_info=True)

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
                run_id=run_id,
                execution_mode=effective_execution_mode,
                synthetic_dataset_id=window_dataset_id,
                synthetic_generator_version=window_generator_version,
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

    run_perf = {}
    lifetime_perf = {}
    try:
        run_perf = trading_engine.db_manager.get_performance_summary(run_id=run_id)
        lifetime_perf = trading_engine.db_manager.get_performance_summary()
    except Exception:
        logger.debug("Skipping performance summary aggregation", exc_info=True)
    try:
        win_rate = float(run_perf.get("win_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        win_rate = 0.0
    try:
        profit_factor = float(run_perf.get("profit_factor", 0.0) or 0.0)
    except (TypeError, ValueError):
        profit_factor = 0.0
    try:
        realized_trades = int(run_perf.get("total_trades") or 0)
    except (TypeError, ValueError):
        realized_trades = 0

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

    cash_ratio = None
    try:
        total_val = float(final_summary.get("total_value", 0.0))
        if total_val:
            cash_ratio = float(final_summary.get("cash", 0.0)) / total_val
    except Exception:
        cash_ratio = None

    run_completed_at = datetime.now(UTC)
    action_plan = _build_action_plan(
        pnl_dollars=float(final_summary.get("pnl_dollars", 0.0)),
        profit_factor=float(profit_factor) if isinstance(profit_factor, (int, float)) else 0.0,
        win_rate=float(win_rate) if isinstance(win_rate, (int, float)) else 0.0,
        realized_trades=realized_trades,
        cash_ratio=cash_ratio,
        forecaster_health=forecaster_health,
        quant_health=quant_health,
    )
    run_summary_record = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "ended_at": run_completed_at.isoformat(),
        "duration_seconds": (run_completed_at - run_started_at).total_seconds(),
        "tickers": ticker_list,
        "cycles": cycles,
        "execution_mode": last_execution_mode,
        "data_source": data_source_manager.get_active_source(),
        "synthetic_dataset_id": last_dataset_id,
        "synthetic_generator_version": last_generator_version,
        "profitability": {
            "pnl_dollars": final_summary["pnl_dollars"],
            "pnl_pct": final_summary["pnl_pct"],
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "realized_trades": realized_trades,
            "trades": final_summary["trades"],
            "lifetime": {
                "profit_factor": lifetime_perf.get("profit_factor", 0.0) if isinstance(lifetime_perf, dict) else 0.0,
                "win_rate": lifetime_perf.get("win_rate", 0.0) if isinstance(lifetime_perf, dict) else 0.0,
                "total_trades": lifetime_perf.get("total_trades", 0) if isinstance(lifetime_perf, dict) else 0,
                "total_profit": lifetime_perf.get("total_profit", 0.0) if isinstance(lifetime_perf, dict) else 0.0,
            },
        },
        "liquidity": {
            "cash": final_summary["cash"],
            "total_value": final_summary["total_value"],
            "cash_ratio": cash_ratio,
            "open_positions": len(final_summary["positions"]),
        },
        "forecaster": {
            "metrics": forecaster_health.get("metrics") if isinstance(forecaster_health, dict) else {},
            "status": forecaster_health.get("status") if isinstance(forecaster_health, dict) else {},
        },
        "quant_validation": quant_health,
        "next_actions": action_plan,
    }
    _log_run_summary(run_summary_record)
    logger.info(
        "Run summary: PnL $%.2f (%.2f%%) | PF %.2f | Win rate %.1f%% | Cash ratio %s",
        final_summary["pnl_dollars"],
        final_summary["pnl_pct"] * 100,
        float(profit_factor) if isinstance(profit_factor, (int, float)) else 0.0,
        float(win_rate) * 100 if isinstance(win_rate, (int, float)) else 0.0,
        f"{cash_ratio:.1%}" if cash_ratio is not None else "n/a",
    )
    logger.info("Next actions: %s", " | ".join(action_plan))

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

    try:
        trading_engine.db_manager.record_run_provenance(
            run_id=run_id,
            execution_mode=last_execution_mode,
            data_source=data_source_manager.get_active_source(),
            synthetic_dataset_id=last_dataset_id,
            synthetic_generator_version=last_generator_version,
            note="auto_trader_complete",
        )
    except Exception:
        logger.debug("Failed to stamp run provenance for dashboard badge", exc_info=True)

    try:
        prov = trading_engine.db_manager.get_data_provenance_summary()
        prov.update(
            {
                "run_id": run_id,
                "execution_mode": last_execution_mode,
                "dataset_id": last_dataset_id,
                "generator_version": last_generator_version,
            }
        )
        artifact = ROOT_PATH / "logs" / "automation" / f"db_provenance_{run_id}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(prov, indent=2))
        logger.info("Wrote DB provenance artifact: %s", artifact)
    except Exception:
        logger.debug("Failed to emit DB provenance artifact", exc_info=True)

    # Close to flush + sync any WSL mirror back to the canonical DB path.
    try:
        trading_engine.db_manager.close()
    except Exception:
        logger.debug("Unable to close trading database cleanly", exc_info=True)


if __name__ == "__main__":
    main()
