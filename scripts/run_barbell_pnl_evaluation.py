#!/usr/bin/env python3
"""
Barbell PnL evaluation harness (feature-flagged).

Runs the same walk-forward TS forecaster + signal generator + PaperTradingEngine
simulation twice:
- TS_ONLY: baseline sizing (confidence as produced)
- BARBELL_SIZED: applies barbell bucket multipliers to confidence before sizing

This is not a full NAV allocator; it is an evidence step to decide if barbell
sizing is promising before wiring it into production paths.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import click
import pandas as pd

ROOT_PATH = Path(__file__).resolve().parent.parent
site.addsitedir(str(ROOT_PATH))
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from etl.database_manager import DatabaseManager
from etl.time_series_forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from execution.paper_trading_engine import PaperTradingEngine
from models.time_series_signal_generator import TimeSeriesSignalGenerator
from risk.barbell_policy import BarbellConfig
from risk.barbell_promotion_gate import decide_promotion_from_report, write_promotion_evidence
from risk.barbell_sizing import apply_barbell_confidence, barbell_confidence_multipliers

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


def _resolve_window(
    db: DatabaseManager, start_date: Optional[str], end_date: Optional[str], lookback_days: int
) -> Window:
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


def _to_engine_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mapped = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    ).copy()
    mapped.index = pd.to_datetime(mapped.index)
    mapped.sort_index(inplace=True)
    return mapped


def _barbell_multipliers(cfg: BarbellConfig) -> Dict[str, float]:
    # Backward-compatible wrapper. Source of truth lives in risk.barbell_sizing.
    return barbell_confidence_multipliers(cfg)


def _apply_barbell_confidence(
    *,
    ticker: str,
    confidence: float,
    cfg: BarbellConfig,
    multipliers: Dict[str, float],
) -> float:
    _ = multipliers  # signature retained; multipliers are derived from cfg in shared helper
    return apply_barbell_confidence(
        ticker=ticker,
        base_confidence=float(confidence),
        cfg=cfg,
    ).effective_confidence


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


def _simulate_from_trade_history(
    *,
    source_db: DatabaseManager,
    tickers: Sequence[str],
    window: Window,
    initial_capital: float,
    enable_barbell_sizing: bool,
    barbell_cfg: BarbellConfig,
    run_id: Optional[str] = None,
) -> Dict[str, float]:
    multipliers = _barbell_multipliers(barbell_cfg)
    safe_set = {str(s).upper() for s in (barbell_cfg.safe_symbols or [])}
    core_set = {str(s).upper() for s in (barbell_cfg.core_symbols or [])}
    spec_set = {str(s).upper() for s in (barbell_cfg.speculative_symbols or [])}
    ticker_set = {str(t).upper() for t in tickers}

    query = """
        SELECT trade_date, ticker, realized_pnl
        FROM trade_executions
        WHERE realized_pnl IS NOT NULL
          AND trade_date >= ?
          AND trade_date <= ?
    """
    params: List[Any] = [window.start_date, window.end_date]
    if run_id:
        query += " AND run_id = ?"
        params.append(run_id)
    if ticker_set:
        placeholders = ",".join("?" for _ in ticker_set)
        query += f" AND UPPER(ticker) IN ({placeholders})"
        params.extend(sorted(ticker_set))
    query += " ORDER BY trade_date ASC, id ASC"

    source_db.cursor.execute(query, params)
    rows = source_db.cursor.fetchall() or []

    pnl_events: List[Tuple[str, str, float]] = []
    for row in rows:
        try:
            trade_date = str(row[0])
            ticker = str(row[1]).upper()
            pnl = float(row[2] or 0.0)
        except Exception:
            continue

        if enable_barbell_sizing:
            if ticker in safe_set:
                pnl *= float(multipliers.get("safe", 1.0))
            elif ticker in core_set:
                pnl *= float(multipliers.get("core", 1.0))
            elif ticker in spec_set:
                pnl *= float(multipliers.get("spec", 1.0))
        pnl_events.append((trade_date, ticker, pnl))

    total_trades = len(pnl_events)
    gross_profit = sum(max(0.0, pnl) for _, __, pnl in pnl_events)
    gross_loss = sum(max(0.0, -pnl) for _, __, pnl in pnl_events)
    total_profit = gross_profit - gross_loss
    win_trades = sum(1 for _, __, pnl in pnl_events if pnl > 0)
    losing_trades = sum(1 for _, __, pnl in pnl_events if pnl < 0)
    win_rate = (win_trades / total_trades) if total_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    equity: List[Dict[str, float]] = []
    eq = float(initial_capital)
    for trade_date, ticker, pnl in pnl_events:
        eq += float(pnl)
        equity.append({"date": trade_date, "ticker": ticker, "equity": float(eq)})

    max_dd = _max_drawdown(equity)
    total_return_pct = (eq - float(initial_capital)) / float(initial_capital) if initial_capital else 0.0
    return {
        "total_return": float(total_profit),  # alias for backward compat
        "total_profit": float(total_profit),
        "total_return_pct": float(total_return_pct),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_dd),
        "total_trades": int(total_trades),
        "losing_trades": int(losing_trades),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
    }


def _simulate_walk_forward(
    *,
    source_db: DatabaseManager,
    tickers: Sequence[str],
    window: Window,
    initial_capital: float,
    forecast_horizon: int,
    history_bars: int,
    min_bars: int,
    enable_barbell_sizing: bool,
    barbell_cfg: BarbellConfig,
    step_days: int = 1,
    forecaster_profile: str = "full",
    signal_confidence_threshold: float = 0.55,
    signal_min_expected_return: float = 0.003,
    signal_max_risk_score: float = 0.7,
    disable_quant_validation: bool = False,
    disable_volatility_filter: bool = False,
) -> Dict[str, float]:
    ohlcv = source_db.load_ohlcv(list(tickers), start_date=window.start_date, end_date=window.end_date)
    if ohlcv.empty:
        return {"total_return": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "total_trades": 0}

    sim_db = DatabaseManager(db_path=":memory:")
    engine = PaperTradingEngine(
        initial_capital=float(initial_capital),
        slippage_pct=0.001,
        transaction_cost_pct=0.001,
        database_manager=sim_db,
    )

    quant_validation_config = {"enabled": False} if disable_quant_validation else None
    ts_generator = TimeSeriesSignalGenerator(
        confidence_threshold=float(signal_confidence_threshold),
        min_expected_return=float(signal_min_expected_return),
        max_risk_score=float(signal_max_risk_score),
        use_volatility_filter=not bool(disable_volatility_filter),
        quant_validation_config=quant_validation_config,
    )
    multipliers = _barbell_multipliers(barbell_cfg)
    profile = str(forecaster_profile or "full").strip().lower()
    if profile not in {"full", "fast", "mssa_only"}:
        raise ValueError("forecaster_profile must be one of {'full','fast','mssa_only'}")

    frames_by_ticker: Dict[str, pd.DataFrame] = {
        str(sym).upper(): df.sort_index() for sym, df in ohlcv.groupby("ticker")
    }
    all_dates: set[pd.Timestamp] = set()
    for df in frames_by_ticker.values():
        all_dates.update(pd.to_datetime(df.index).to_list())

    ordered_dates = sorted(all_dates)
    stride = max(int(step_days), 1)
    for date in ordered_dates[::stride]:
        price_map: Dict[str, float] = {}
        for ticker in tickers:
            tdf = frames_by_ticker.get(str(ticker).upper())
            if tdf is None or tdf.empty or date not in tdf.index:
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
            price_map[str(ticker).upper()] = current_price

            forecast_bundle: Dict[str, Any]
            try:
                if profile == "full":
                    ts_config = TimeSeriesForecasterConfig(forecast_horizon=int(forecast_horizon))
                elif profile == "fast":
                    ts_config = TimeSeriesForecasterConfig(
                        forecast_horizon=int(forecast_horizon),
                        sarimax_enabled=False,
                        samossa_enabled=False,
                    )
                else:
                    ts_config = TimeSeriesForecasterConfig(
                        forecast_horizon=int(forecast_horizon),
                        sarimax_enabled=False,
                        garch_enabled=False,
                        samossa_enabled=False,
                        ensemble_enabled=False,
                    )
                forecaster = TimeSeriesForecaster(
                    config=ts_config
                )
                forecaster.fit(price_series=close_series, returns_series=returns_series)
                forecast_bundle = forecaster.forecast()
            except Exception:
                continue

            try:
                ts_signal = ts_generator.generate_signal(
                    forecast_bundle=forecast_bundle,
                    current_price=current_price,
                    ticker=str(ticker).upper(),
                    market_data=engine_frame,
                )
            except Exception:
                continue

            signal_dict = {
                "ticker": getattr(ts_signal, "ticker", str(ticker).upper()),
                "action": getattr(ts_signal, "action", "HOLD"),
                "confidence": float(getattr(ts_signal, "confidence", 0.0)),
                "expected_return": float(getattr(ts_signal, "expected_return", 0.0)),
                "forecast_horizon": int(getattr(ts_signal, "forecast_horizon", forecast_horizon)),
                "stop_loss": getattr(ts_signal, "stop_loss", None),
                "target_price": getattr(ts_signal, "target_price", None),
                "source": "TIME_SERIES",
            }

            if enable_barbell_sizing:
                signal_dict["confidence"] = _apply_barbell_confidence(
                    ticker=signal_dict["ticker"],
                    confidence=signal_dict["confidence"],
                    cfg=barbell_cfg,
                    multipliers=multipliers,
                )

            engine.execute_signal(signal_dict, market_data=engine_frame)

        if price_map:
            # PaperTradingEngine expects plain tickers.
            engine.mark_to_market({k: float(v) for k, v in price_map.items()})

    summary = sim_db.get_performance_summary()
    equity = sim_db.get_equity_curve(initial_capital=float(initial_capital))
    max_dd = _max_drawdown(equity)
    final_equity = float((equity[-1] or {}).get("equity", initial_capital)) if equity else float(initial_capital)
    total_profit = float(summary.get("total_profit") or 0.0)
    total_return_pct = (final_equity - float(initial_capital)) / float(initial_capital) if initial_capital else 0.0

    return {
        # Backward-compatible alias: "total_return" historically meant total_profit ($).
        "total_return": total_profit,
        "total_profit": total_profit,
        "total_return_pct": float(total_return_pct),
        "profit_factor": float(summary.get("profit_factor") or 0.0),
        "win_rate": float(summary.get("win_rate") or 0.0),
        "max_drawdown": float(max_dd),
        "total_trades": int(summary.get("total_trades") or 0),
        "losing_trades": int(
            summary.get("losing_trades")
            or max(
                0,
                int(summary.get("total_trades") or 0)
                - int(round(float(summary.get("win_rate") or 0.0) * float(summary.get("total_trades") or 0))),
            )
        ),
    }


def run_barbell_eval(
    *,
    db_path: str,
    tickers: Sequence[str],
    window: Window,
    initial_capital: float,
    forecast_horizon: int,
    history_bars: int,
    min_bars: int,
    step_days: int = 1,
    forecaster_profile: str = "full",
    signal_confidence_threshold: float = 0.55,
    signal_min_expected_return: float = 0.003,
    signal_max_risk_score: float = 0.7,
    disable_quant_validation: bool = False,
    disable_volatility_filter: bool = False,
    evidence_source: str = "walk_forward",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    db = DatabaseManager(db_path=db_path)
    barbell_cfg = BarbellConfig.from_yaml()

    src = str(evidence_source or "walk_forward").strip().lower()
    if src not in {"walk_forward", "trade_history"}:
        raise ValueError("evidence_source must be one of {'walk_forward','trade_history'}")

    if src == "trade_history":
        baseline = _simulate_from_trade_history(
            source_db=db,
            tickers=tickers,
            window=window,
            initial_capital=initial_capital,
            enable_barbell_sizing=False,
            barbell_cfg=barbell_cfg,
            run_id=run_id,
        )
        barbell = _simulate_from_trade_history(
            source_db=db,
            tickers=tickers,
            window=window,
            initial_capital=initial_capital,
            enable_barbell_sizing=True,
            barbell_cfg=barbell_cfg,
            run_id=run_id,
        )
    else:
        baseline = _simulate_walk_forward(
            source_db=db,
            tickers=tickers,
            window=window,
            initial_capital=initial_capital,
            forecast_horizon=forecast_horizon,
            history_bars=history_bars,
            min_bars=min_bars,
            enable_barbell_sizing=False,
            barbell_cfg=barbell_cfg,
            step_days=step_days,
            forecaster_profile=forecaster_profile,
            signal_confidence_threshold=signal_confidence_threshold,
            signal_min_expected_return=signal_min_expected_return,
            signal_max_risk_score=signal_max_risk_score,
            disable_quant_validation=disable_quant_validation,
            disable_volatility_filter=disable_volatility_filter,
        )
        barbell = _simulate_walk_forward(
            source_db=db,
            tickers=tickers,
            window=window,
            initial_capital=initial_capital,
            forecast_horizon=forecast_horizon,
            history_bars=history_bars,
            min_bars=min_bars,
            enable_barbell_sizing=True,
            barbell_cfg=barbell_cfg,
            step_days=step_days,
            forecaster_profile=forecaster_profile,
            signal_confidence_threshold=signal_confidence_threshold,
            signal_min_expected_return=signal_min_expected_return,
            signal_max_risk_score=signal_max_risk_score,
            disable_quant_validation=disable_quant_validation,
            disable_volatility_filter=disable_volatility_filter,
        )

    def _delta(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        keys = set(a) | set(b)
        out: Dict[str, float] = {}
        for k in keys:
            try:
                out[str(k)] = float(b.get(k, 0.0)) - float(a.get(k, 0.0))
            except Exception:
                out[str(k)] = 0.0
        return out

    payload = {
        "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        "window": {"start_date": window.start_date, "end_date": window.end_date},
        "tickers": list(tickers),
        "evidence_source": src,
        "run_filter": {"run_id": run_id} if run_id else None,
        "forecaster_profile": str(forecaster_profile or "full"),
        "signal_policy": {
            "confidence_threshold": float(signal_confidence_threshold),
            "min_expected_return": float(signal_min_expected_return),
            "max_risk_score": float(signal_max_risk_score),
            "quant_validation": "off" if disable_quant_validation else "default",
            "volatility_filter": "off" if disable_volatility_filter else "on",
        },
        "barbell_config": {
            "safe_symbols": list(barbell_cfg.safe_symbols),
            "core_symbols": list(barbell_cfg.core_symbols),
            "speculative_symbols": list(barbell_cfg.speculative_symbols),
            "safe_min": float(barbell_cfg.safe_min),
            "risk_max": float(barbell_cfg.risk_max),
            "core_max_per": float(barbell_cfg.core_max_per),
            "spec_max_per": float(barbell_cfg.spec_max_per),
        },
        "metrics": {"ts_only": baseline, "barbell_sized": barbell, "delta": _delta(baseline, barbell)},
    }
    decision = decide_promotion_from_report(payload)
    payload["promotion_decision"] = {
        "passed": bool(decision.passed),
        "reason": decision.reason,
        "evidence_source": decision.evidence_source,
    }
    try:
        db.close()
    except Exception:
        pass
    return payload


def _default_report_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return ROOT_PATH / "reports" / f"barbell_pnl_eval_{ts}.json"


@click.command()
@click.option("--tickers", default="SHY,BIL,IEF,MSFT,CL=F,MTN,AAPL,BTC-USD", show_default=True)
@click.option("--db-path", default=None, help="DB path (defaults to PORTFOLIO_DB_PATH or data/portfolio_maximizer.db).")
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD).")
@click.option("--lookback-days", default=180, show_default=True, help="Window size in days when --start-date is not provided.")
@click.option("--forecast-horizon", default=14, show_default=True, help="Forecast horizon in bars.")
@click.option("--history-bars", default=120, show_default=True, help="Bars provided into each re-fit.")
@click.option("--min-bars", default=60, show_default=True, help="Minimum bars required before fitting.")
@click.option(
    "--forecaster-profile",
    default="full",
    show_default=True,
    type=click.Choice(["full", "fast", "mssa_only"], case_sensitive=False),
    help="Forecast model profile. 'fast' disables SARIMAX/SAMOSSA; 'mssa_only' uses only MSSA-RL.",
)
@click.option("--signal-confidence-threshold", default=0.55, show_default=True, type=float)
@click.option("--signal-min-expected-return", default=0.003, show_default=True, type=float)
@click.option("--signal-max-risk-score", default=0.7, show_default=True, type=float)
@click.option("--disable-quant-validation", is_flag=True, help="Disable quant validation gating to increase trade count.")
@click.option("--disable-volatility-filter", is_flag=True, help="Disable volatility gating to increase trade count.")
@click.option(
    "--evidence-source",
    default="walk_forward",
    show_default=True,
    type=click.Choice(["walk_forward", "trade_history"], case_sensitive=False),
    help="Evidence source: walk-forward simulation vs counterfactual scaling of realized trade history.",
)
@click.option("--run-id", default=None, help="Optional trade_executions.run_id filter when evidence_source=trade_history.")
@click.option(
    "--step-days",
    default=1,
    show_default=True,
    help="Stride over evaluation dates (e.g. 5 = evaluate every 5th bar for faster runs).",
)
@click.option("--initial-capital", default=25000.0, show_default=True)
@click.option("--report-path", default=None, help="Output JSON path (default: reports/barbell_pnl_eval_<ts>.json).")
@click.option(
    "--write-promotion-evidence",
    default="",
    help="Optional JSON path to write a promotion gate artifact (e.g., reports/barbell_pnl_evidence_latest.json).",
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
    forecaster_profile: str,
    signal_confidence_threshold: float,
    signal_min_expected_return: float,
    signal_max_risk_score: float,
    disable_quant_validation: bool,
    disable_volatility_filter: bool,
    evidence_source: str,
    run_id: Optional[str],
    step_days: int,
    initial_capital: float,
    report_path: Optional[str],
    promotion_evidence_path: str,
    verbose: bool,
) -> None:
    _configure_logging(verbose)
    ticker_list = _split_tickers(tickers)
    if not ticker_list:
        raise click.UsageError("At least one ticker is required.")

    resolved_db_path = db_path or os.getenv("PORTFOLIO_DB_PATH") or str(ROOT_PATH / "data" / "portfolio_maximizer.db")
    db = DatabaseManager(db_path=resolved_db_path)
    window = _resolve_window(db, start_date, end_date, lookback_days)
    try:
        db.close()
    except Exception:
        pass

    payload = run_barbell_eval(
        db_path=resolved_db_path,
        tickers=ticker_list,
        window=window,
        initial_capital=float(initial_capital),
        forecast_horizon=int(forecast_horizon),
        history_bars=int(history_bars),
        min_bars=int(min_bars),
        step_days=int(step_days),
        forecaster_profile=str(forecaster_profile),
        signal_confidence_threshold=float(signal_confidence_threshold),
        signal_min_expected_return=float(signal_min_expected_return),
        signal_max_risk_score=float(signal_max_risk_score),
        disable_quant_validation=bool(disable_quant_validation),
        disable_volatility_filter=bool(disable_volatility_filter),
        evidence_source=str(evidence_source),
        run_id=run_id,
    )

    out_path = Path(report_path).expanduser() if report_path else _default_report_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if promotion_evidence_path:
        evidence_path = Path(promotion_evidence_path).expanduser()
        decision = decide_promotion_from_report(payload)
        write_promotion_evidence(path=evidence_path, decision=decision, report_path=out_path)

    m = payload.get("metrics") or {}
    ts_only = m.get("ts_only") or {}
    bb = m.get("barbell_sized") or {}
    delta = m.get("delta") or {}
    logger.info(
        "Barbell eval complete (%s -> %s) TS_ONLY pnl=%.2f PF=%.3f | BARBELL pnl=%.2f PF=%.3f | Δpnl=%.2f",
        window.start_date,
        window.end_date,
        float(ts_only.get("total_return", 0.0)),
        float(ts_only.get("profit_factor", 0.0)),
        float(bb.get("total_return", 0.0)),
        float(bb.get("profit_factor", 0.0)),
        float(delta.get("total_return", 0.0)),
    )
    logger.info("Report written: %s", out_path)


if __name__ == "__main__":
    main()
