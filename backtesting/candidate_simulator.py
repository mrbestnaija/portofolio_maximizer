"""
Candidate simulator: replay a guardrail-aware trading loop per candidate.

Design:
- Map candidate.params into signal confidence, ensemble weights, and execution
  cost tweaks.
- Walk-forward harness: fit forecaster -> generate TS signal -> execute stepwise
  across a historical window (no look-ahead).
- Execute via PaperTradingEngine (in-memory DB) to get PnL metrics without
  modifying the live database.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from etl.database_manager import DatabaseManager
from etl.portfolio_math import portfolio_metrics_ngn
from etl.time_series_forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from execution.paper_trading_engine import PaperTradingEngine
from models.signal_generator_factory import build_signal_generator
from forcester_ts.regime_detector import RegimeConfig, RegimeDetector
from risk.barbell_policy import BarbellConfig
from risk.barbell_promotion_gate import summarize_regime_realism
from risk.barbell_sizing import build_barbell_market_context, evaluate_barbell_path_risk

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).resolve().parent.parent
_DEFAULT_FORECASTING_CONFIG_PATH = Path("config/forecasting_config.yml")
_DEFAULT_BARBELL_CONFIG_PATH = ROOT_PATH / "config" / "barbell.yml"
_ENSEMBLE_WEIGHT_KEYS = {
    "ensemble_weight_sarimax": "sarimax",
    "ensemble_weight_garch": "garch",
    "ensemble_weight_samossa": "samossa",
    "ensemble_weight_mssa_rl": "mssa_rl",
}


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


def _load_forecasting_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.debug("Forecasting config not found at %s; using forecaster defaults.", path)
        return {}

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Unable to read forecasting config %s: %s", path, exc)
        return {}

    if isinstance(payload, dict):
        nested = payload.get("forecasting")
        if isinstance(nested, dict):
            return dict(nested)
        return dict(payload)
    return {}


def _load_barbell_promotion_thresholds(path: Path = _DEFAULT_BARBELL_CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Unable to read barbell config %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    barbell = payload.get("barbell")
    if not isinstance(barbell, dict):
        return {}
    promotion_gate = barbell.get("promotion_gate")
    return dict(promotion_gate) if isinstance(promotion_gate, dict) else {}


def _load_candidate_regime_detector(forecasting_cfg: Dict[str, Any]) -> Optional[RegimeDetector]:
    regime_cfg = forecasting_cfg.get("regime_detection")
    if not isinstance(regime_cfg, dict):
        return None
    if not bool(regime_cfg.get("enabled", True)):
        return None

    allowed_keys = {
        "enabled",
        "lookback_window",
        "vol_threshold_low",
        "vol_threshold_high",
        "trend_threshold_weak",
        "trend_threshold_strong",
    }
    try:
        return RegimeDetector(
            RegimeConfig(**{k: v for k, v in regime_cfg.items() if k in allowed_keys})
        )
    except Exception as exc:
        logger.warning("Unable to build candidate regime detector: %s", exc)
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_candidate_ensemble_weights(candidate_params: Dict[str, Any]) -> tuple[Optional[Dict[str, float]], Optional[str]]:
    seen_weight_key = False
    candidate_weights: Dict[str, float] = {}

    for source_key, model_key in _ENSEMBLE_WEIGHT_KEYS.items():
        if source_key not in candidate_params:
            continue
        seen_weight_key = True
        raw_value = candidate_params.get(source_key)
        try:
            weight = float(raw_value)
        except (TypeError, ValueError):
            return None, f"invalid_{source_key}"
        if not np.isfinite(weight) or weight < 0.0:
            return None, f"invalid_{source_key}"
        if weight > 0.0:
            candidate_weights[model_key] = weight

    if not seen_weight_key:
        return None, None
    if not candidate_weights:
        return None, "invalid_ensemble_weight_vector"

    total = float(sum(candidate_weights.values()))
    if not np.isfinite(total) or total <= 0.0:
        return None, "invalid_ensemble_weight_vector"

    return {model: weight / total for model, weight in candidate_weights.items()}, None


def _build_candidate_forecaster_config(
    *,
    forecasting_cfg: Dict[str, Any],
    forecast_horizon: int,
    candidate_weights: Optional[Dict[str, float]],
) -> TimeSeriesForecasterConfig:
    config = TimeSeriesForecasterConfig(forecast_horizon=forecast_horizon)

    def _section(name: str) -> Dict[str, Any]:
        section = forecasting_cfg.get(name, {})
        return section if isinstance(section, dict) else {}

    sarimax_cfg = _section("sarimax")
    garch_cfg = _section("garch")
    samossa_cfg = _section("samossa")
    mssa_cfg = _section("mssa_rl")
    ensemble_cfg = _section("ensemble")
    regime_cfg = _section("regime_detection")
    order_learning_cfg = _section("order_learning")
    monte_carlo_cfg = _section("monte_carlo")

    if sarimax_cfg:
        config.sarimax_enabled = bool(sarimax_cfg.get("enabled", config.sarimax_enabled))
        config.sarimax_kwargs = {k: v for k, v in sarimax_cfg.items() if k != "enabled"}
    if garch_cfg:
        config.garch_enabled = bool(garch_cfg.get("enabled", config.garch_enabled))
        config.garch_kwargs = {k: v for k, v in garch_cfg.items() if k != "enabled"}
    if samossa_cfg:
        config.samossa_enabled = bool(samossa_cfg.get("enabled", config.samossa_enabled))
        config.samossa_kwargs = {k: v for k, v in samossa_cfg.items() if k != "enabled"}
    if mssa_cfg:
        config.mssa_rl_enabled = bool(mssa_cfg.get("enabled", config.mssa_rl_enabled))
        config.mssa_rl_kwargs = {k: v for k, v in mssa_cfg.items() if k != "enabled"}
    if ensemble_cfg:
        config.ensemble_enabled = bool(ensemble_cfg.get("enabled", config.ensemble_enabled))
        config.ensemble_kwargs = {k: v for k, v in ensemble_cfg.items() if k != "enabled"}
    if regime_cfg:
        config.regime_detection_enabled = bool(regime_cfg.get("enabled", config.regime_detection_enabled))
        config.regime_detection_kwargs = {k: v for k, v in regime_cfg.items() if k != "enabled"}
    if order_learning_cfg:
        config.order_learning_config = dict(order_learning_cfg)
    if monte_carlo_cfg:
        config.monte_carlo_config = dict(monte_carlo_cfg)

    if candidate_weights is not None:
        # Candidate alpha evaluation must measure the candidate vector itself.
        # Disable adaptive/regime candidate overrides so they cannot mask the
        # candidate under test and create a false alpha signal.
        config.ensemble_enabled = True
        config.ensemble_kwargs = dict(config.ensemble_kwargs or {})
        config.ensemble_kwargs["candidate_weights"] = [candidate_weights]
        config.ensemble_kwargs["adaptive_candidate_weights"] = []
        config.regime_detection_enabled = False
        config.regime_detection_kwargs = {}

    return config


def _equal_weight_benchmark_returns(ohlcv: pd.DataFrame) -> pd.Series:
    if ohlcv is None or ohlcv.empty:
        return pd.Series(dtype=float)

    try:
        price_matrix = (
            ohlcv.reset_index()
            .pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
            .sort_index()
        )
    except Exception as exc:
        logger.warning("Unable to build equal-weight benchmark proxy: %s", exc)
        return pd.Series(dtype=float)

    if price_matrix.empty:
        return pd.Series(dtype=float)

    price_matrix = price_matrix.astype(float).replace([np.inf, -np.inf], np.nan)
    benchmark_returns = (
        price_matrix.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .mean(axis=1, skipna=True)
        .dropna()
        .astype(float)
    )
    benchmark_returns.index = pd.DatetimeIndex(pd.to_datetime(benchmark_returns.index))
    return benchmark_returns


def _evaluate_executed_trade_path_risk(
    *,
    signal_payload: Dict[str, Any],
    trade: Any,
    market_data: pd.DataFrame,
    barbell_cfg: Optional[BarbellConfig],
    detected_regime: Optional[str],
) -> Dict[str, Any]:
    if barbell_cfg is None or trade is None:
        return {
            "path_risk_evidence": False,
            "barbell_path_risk_ok": False,
            "path_risk_checks": {},
            "path_risk_reason": "barbell_config_unavailable",
            "detected_regime": detected_regime,
        }

    try:
        trade_shares = abs(float(getattr(trade, "shares", 0.0) or 0.0))
        trade_entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        trade_shares = 0.0
        trade_entry_price = 0.0

    if trade_shares <= 0.0 or trade_entry_price <= 0.0:
        return {
            "path_risk_evidence": False,
            "barbell_path_risk_ok": False,
            "path_risk_checks": {},
            "path_risk_reason": "missing trade notional for path-risk evaluation",
            "detected_regime": detected_regime,
        }

    context = build_barbell_market_context(
        signal_payload={**signal_payload, "position_value": trade_shares * trade_entry_price},
        market_data=market_data,
        detected_regime=detected_regime,
    )
    assessed = evaluate_barbell_path_risk(context=context, cfg=barbell_cfg)
    checks = dict(assessed.get("path_risk_checks") or {})
    evidence_available = any(value is not None for value in checks.values())
    if not evidence_available:
        return {
            "path_risk_evidence": False,
            "barbell_path_risk_ok": False,
            "path_risk_checks": checks,
            "path_risk_reason": "missing path-risk evidence",
            "detected_regime": detected_regime,
        }

    failed_checks = [name for name, passed in checks.items() if passed is False]
    reason = (
        "path risk passed"
        if bool(assessed.get("barbell_path_risk_ok", False))
        else ("; ".join(failed_checks) if failed_checks else "path risk failed")
    )
    return {
        "path_risk_evidence": True,
        "barbell_path_risk_ok": bool(assessed.get("barbell_path_risk_ok", False)),
        "path_risk_checks": checks,
        "path_risk_reason": reason,
        "detected_regime": detected_regime,
        "path_risk_diagnostics": dict(assessed.get("diagnostics") or {}),
    }


def _summarize_path_risk_records(path_risk_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(path_risk_records)
    evidence_count = 0
    ok_count = 0
    check_summary: Dict[str, Dict[str, int]] = {}
    failure_reasons: List[str] = []

    for record in path_risk_records or []:
        if not isinstance(record, dict):
            continue
        if bool(record.get("path_risk_evidence")):
            evidence_count += 1
        if bool(record.get("barbell_path_risk_ok")):
            ok_count += 1
        else:
            reason = str(record.get("path_risk_reason") or "path risk failed")
            if reason not in failure_reasons:
                failure_reasons.append(reason)

        checks = record.get("path_risk_checks")
        if isinstance(checks, dict):
            for check_name, raw_value in checks.items():
                bucket = check_summary.setdefault(
                    str(check_name),
                    {"passed": 0, "failed": 0, "missing": 0},
                )
                if raw_value is None:
                    bucket["missing"] += 1
                elif bool(raw_value):
                    bucket["passed"] += 1
                else:
                    bucket["failed"] += 1

    return {
        "path_risk_trade_count": int(total),
        "path_risk_evidence_count": int(evidence_count),
        "path_risk_ok_rate": float(ok_count / total) if total > 0 else None,
        "barbell_path_risk_ok": bool(total > 0 and ok_count == total and evidence_count == total),
        "path_risk_check_summary": {
            key: dict(sorted(value.items()))
            for key, value in sorted(check_summary.items())
        },
        "path_risk_failure_reasons": sorted(failure_reasons),
    }


def _summarize_candidate_anti_barbell_evidence(
    *,
    metrics: Dict[str, Any],
    path_risk_records: Sequence[Dict[str, Any]],
    regime_labels: Sequence[Optional[str]],
    promotion_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    regime_summary = summarize_regime_realism(regime_labels, promotion_thresholds)
    path_risk_summary = _summarize_path_risk_records(path_risk_records)

    threshold_sensitivity = {
        "omega_robustness_score": metrics.get("omega_robustness_score"),
        "omega_monotonicity_ok": metrics.get("omega_monotonicity_ok"),
        "omega_cliff_drop_ratio": metrics.get("omega_cliff_drop_ratio"),
        "omega_cliff_ok": metrics.get("omega_cliff_ok"),
    }
    right_tail_confidence = {
        "omega_ci_lower": metrics.get("omega_ci_lower"),
        "omega_ci_upper": metrics.get("omega_ci_upper"),
        "omega_ci_width": metrics.get("omega_ci_width"),
        "omega_right_tail_ok": metrics.get("omega_right_tail_ok"),
    }
    left_tail_containment = {
        "expected_shortfall_raw": metrics.get("expected_shortfall_raw"),
        "expected_shortfall_to_edge": metrics.get("expected_shortfall_to_edge"),
        "es_to_edge_bounded": metrics.get("es_to_edge_bounded"),
    }
    alpha_quality = {
        "alpha": metrics.get("alpha"),
        "information_ratio": metrics.get("information_ratio"),
        "benchmark_proxy": metrics.get("benchmark_proxy"),
        "benchmark_metrics_status": metrics.get("benchmark_metrics_status"),
        "benchmark_observations": metrics.get("benchmark_observations"),
    }

    anti_barbell_ok = bool(
        metrics.get("omega_monotonicity_ok") is True
        and metrics.get("omega_cliff_ok") is True
        and metrics.get("omega_right_tail_ok") is True
        and metrics.get("es_to_edge_bounded") is True
        and path_risk_summary["barbell_path_risk_ok"] is True
        and regime_summary["regime_realism_ok"] is True
    )

    failure_reasons = []
    if metrics.get("omega_monotonicity_ok") is not True:
        failure_reasons.append("threshold_sensitivity")
    if metrics.get("omega_cliff_ok") is not True:
        failure_reasons.append("omega_cliff")
    if metrics.get("omega_right_tail_ok") is not True:
        failure_reasons.append("right_tail_confidence")
    if metrics.get("es_to_edge_bounded") is not True:
        failure_reasons.append("left_tail_containment")
    if path_risk_summary["barbell_path_risk_ok"] is not True:
        failure_reasons.append("path_risk")
    if regime_summary["regime_realism_ok"] is not True:
        failure_reasons.append("regime_realism")

    return {
        "alpha_quality": alpha_quality,
        "threshold_sensitivity": threshold_sensitivity,
        "right_tail_confidence": right_tail_confidence,
        "left_tail_containment": left_tail_containment,
        "path_risk": path_risk_summary,
        "regime_realism": regime_summary,
        "anti_barbell_ok": anti_barbell_ok,
        "anti_barbell_reason": "all anti-barbell checks passed"
        if anti_barbell_ok
        else "; ".join(failure_reasons) if failure_reasons else "anti-barbell checks failed",
    }


def _empty_metrics(
    *,
    reason: str,
    benchmark_proxy: str = "equal_weight_universe",
    benchmark_metrics_status: str = "unavailable",
    include_strategy_returns: bool = False,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "total_profit": 0.0,
        "total_return": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "alpha": 0.0,
        "information_ratio": 0.0,
        "beta": 0.0,
        "tracking_error": 0.0,
        "r_squared": 0.0,
        "benchmark_proxy": benchmark_proxy,
        "benchmark_metrics_status": benchmark_metrics_status,
        "benchmark_observations": 0,
        "path_risk_trade_count": 0,
        "path_risk_evidence_count": 0,
        "path_risk_ok_rate": None,
        "barbell_path_risk_ok": False,
        "barbell_path_risk_checks": {},
        "barbell_path_risk_failure_reasons": [],
        "regime_realism_trade_count": 0,
        "regime_realism_labeled_trade_count": 0,
        "regime_realism_coverage_rate": None,
        "regime_realism_unique_regimes": [],
        "regime_realism_dominant_regime": None,
        "regime_realism_dominance_rate": None,
        "regime_realism_label_counts": {},
        "regime_realism_ok": False,
        "regime_realism_reason": reason,
        "anti_barbell_ok": False,
        "anti_barbell_reason": reason,
        "anti_barbell_evidence": {
            "alpha_quality": {
                "alpha": 0.0,
                "information_ratio": 0.0,
                "benchmark_proxy": benchmark_proxy,
                "benchmark_metrics_status": benchmark_metrics_status,
                "benchmark_observations": 0,
            },
            "threshold_sensitivity": {
                "omega_robustness_score": 0.0,
                "omega_monotonicity_ok": False,
                "omega_cliff_drop_ratio": None,
                "omega_cliff_ok": False,
            },
            "right_tail_confidence": {
                "omega_ci_lower": None,
                "omega_ci_upper": None,
                "omega_ci_width": None,
                "omega_right_tail_ok": False,
            },
            "left_tail_containment": {
                "expected_shortfall_raw": None,
                "expected_shortfall_to_edge": None,
                "es_to_edge_bounded": False,
            },
            "path_risk": {
                "path_risk_trade_count": 0,
                "path_risk_evidence_count": 0,
                "path_risk_ok_rate": None,
                "barbell_path_risk_ok": False,
                "path_risk_check_summary": {},
                "path_risk_failure_reasons": [reason],
            },
            "regime_realism": {
                "regime_realism_trade_count": 0,
                "regime_realism_labeled_trade_count": 0,
                "regime_realism_coverage_rate": None,
                "regime_realism_unique_regimes": [],
                "regime_realism_dominant_regime": None,
                "regime_realism_dominance_rate": None,
                "regime_realism_label_counts": {},
                "regime_realism_ok": False,
                "regime_realism_reason": reason,
                "regime_realism_thresholds": {},
            },
            "anti_barbell_ok": False,
            "anti_barbell_reason": reason,
        },
        "candidate_invalid_reason": reason,
    }
    if include_strategy_returns:
        metrics["strategy_returns"] = []
    return metrics


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
    include_strategy_returns: bool = False,
    forecasting_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a simple simulation for a candidate over a historical window.
    Guardrails (min_expected_return, max_risk_score) are read but not changed.
    """
    if not tickers:
        return _empty_metrics(
            reason="no_tickers",
            benchmark_metrics_status="unavailable",
            include_strategy_returns=include_strategy_returns,
        )

    forecast_horizon = int(candidate_params.get("forecast_horizon", 30) or 30)
    history_bars = int(candidate_params.get("history_bars", 120) or 120)
    min_bars = int(candidate_params.get("min_bars", 60) or 60)

    candidate_weights, invalid_reason = _extract_candidate_ensemble_weights(candidate_params)
    if invalid_reason:
        logger.warning("Candidate rejected before simulation: %s", invalid_reason)
        return _empty_metrics(
            reason=invalid_reason,
            benchmark_metrics_status="skipped_invalid_candidate",
            include_strategy_returns=include_strategy_returns,
        )

    forecasting_cfg_path = (
        Path(forecasting_config_path).expanduser()
        if forecasting_config_path
        else _DEFAULT_FORECASTING_CONFIG_PATH
    )
    forecasting_cfg = _load_forecasting_config(forecasting_cfg_path)
    forecaster_config = _build_candidate_forecaster_config(
        forecasting_cfg=forecasting_cfg,
        forecast_horizon=forecast_horizon,
        candidate_weights=candidate_weights,
    )
    regime_detector = _load_candidate_regime_detector(forecasting_cfg)
    barbell_promotion_thresholds = _load_barbell_promotion_thresholds()
    try:
        barbell_cfg: Optional[BarbellConfig] = BarbellConfig.from_yaml()
    except Exception as exc:
        logger.warning("Unable to load barbell config for anti-barbell evidence: %s", exc)
        barbell_cfg = None

    ohlcv = source_db.load_ohlcv(list(tickers), start_date=start_date, end_date=end_date)
    if ohlcv.empty:
        return _empty_metrics(
            reason="empty_ohlcv",
            benchmark_metrics_status="unavailable",
            include_strategy_returns=include_strategy_returns,
        )

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

    # Phase 7.15: factory provides all 9 params (previously quant_validation_config
    # and forecasting_config_path were missing). guardrails dict keys match the
    # signal_routing.time_series YAML section, so they merge cleanly as overrides.
    ts_generator = build_signal_generator(
        ts_cfg_overrides=guardrails if guardrails else None,
        forecasting_config_path=forecasting_cfg_path,
    )

    frames_by_ticker: Dict[str, pd.DataFrame] = {
        str(sym): df.sort_index()
        for sym, df in ohlcv.groupby("ticker")
    }
    benchmark_returns = _equal_weight_benchmark_returns(ohlcv)
    path_risk_records: List[Dict[str, Any]] = []
    regime_labels: List[Optional[str]] = []
    all_dates: set[pd.Timestamp] = set()
    for df in frames_by_ticker.values():
        all_dates.update(pd.to_datetime(df.index).to_list())

    equity_snapshots: List[tuple[pd.Timestamp, float]] = []

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

            detected_regime: Optional[str] = None
            if regime_detector is not None:
                try:
                    regime_result = regime_detector.detect_regime(
                        price_series=close_series,
                        returns_series=returns_series,
                    )
                    detected_regime = str(regime_result.get("regime") or "").strip().upper() or None
                except Exception as exc:
                    logger.debug("Regime detection failed for %s @ %s: %s", ticker, date, exc)

            forecast_bundle = None
            try:
                forecaster = TimeSeriesForecaster(config=forecaster_config)
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
            signal_dict["sizing_kelly_fraction_cap"] = candidate_params.get("sizing_kelly_fraction_cap")
            signal_dict["diversification_penalty"] = candidate_params.get("diversification_penalty")
            signal_dict["execution_style"] = execution_style
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

            execution_result = engine.execute_signal(signal_dict, market_data=engine_frame)
            if execution_result.status == "EXECUTED" and getattr(execution_result, "trade", None) is not None:
                regime_labels.append(detected_regime)
                path_risk_records.append(
                    _evaluate_executed_trade_path_risk(
                        signal_payload={
                            "expected_return_net": (
                                (signal_dict.get("provenance") or {})
                                .get("decision_context", {})
                                .get("net_trade_return", signal_dict.get("expected_return"))
                            ),
                            "forecast_horizon": signal_dict.get("forecast_horizon"),
                            "roundtrip_cost_bps": (
                                (signal_dict.get("provenance") or {})
                                .get("decision_context", {})
                                .get("roundtrip_cost_bps")
                            ),
                            "leverage": (
                                (signal_dict.get("provenance") or {})
                                .get("decision_context", {})
                                .get("leverage")
                                or (signal_dict.get("provenance") or {}).get("leverage")
                            ),
                        },
                        trade=execution_result.trade,
                        market_data=engine_frame,
                        barbell_cfg=barbell_cfg,
                        detected_regime=detected_regime,
                    )
                )

        if price_map:
            engine.mark_to_market(price_map)
            equity_snapshots.append((pd.Timestamp(date), float(engine.portfolio.total_value)))

    summary = sim_db.get_performance_summary()
    total_profit = summary.get("total_profit") or 0.0
    profit_factor = summary.get("profit_factor") or 0.0
    win_rate = summary.get("win_rate") or 0.0
    total_trades = summary.get("total_trades") or 0

    if equity_snapshots:
        equity_series = pd.Series(
            [float(value) for _, value in equity_snapshots],
            index=pd.DatetimeIndex([pd.Timestamp(ts) for ts, _ in equity_snapshots]),
            dtype=float,
        ).sort_index()
        max_dd = _max_drawdown([{"equity": float(v)} for v in equity_series.tolist()])
        strategy_returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    else:
        equity_series = pd.Series(dtype=float)
        max_dd = 0.0
        strategy_returns = pd.Series(dtype=float)

    benchmark_metrics_status = "unavailable"
    benchmark_observations = 0
    performance: Dict[str, Any] = {}
    if not strategy_returns.empty:
        if not benchmark_returns.empty:
            common_index = strategy_returns.index.intersection(benchmark_returns.index).sort_values()
            benchmark_observations = int(len(common_index))
            if len(common_index) >= 2:
                aligned_strategy = strategy_returns.reindex(common_index).astype(float)
                aligned_benchmark = benchmark_returns.reindex(common_index).astype(float)
                performance = portfolio_metrics_ngn(
                    aligned_strategy,
                    benchmark_returns=aligned_benchmark,
                )
                benchmark_metrics_status = "aligned"
            else:
                performance = portfolio_metrics_ngn(strategy_returns)
                benchmark_metrics_status = "insufficient_aligned_observations"
        else:
            performance = portfolio_metrics_ngn(strategy_returns)
    elif not benchmark_returns.empty:
        benchmark_metrics_status = "insufficient_strategy_returns"

    final_equity = float(equity_series.iloc[-1]) if not equity_series.empty else float(initial_capital)
    normalized_return = (final_equity / max(float(initial_capital), 1e-12)) - 1.0

    metrics: Dict[str, Any] = {
        "total_profit": float(total_profit),
        "total_return": float(normalized_return),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "max_drawdown": float(performance.get("max_drawdown", max_dd)),
        "total_trades": int(total_trades),
        "benchmark_proxy": "equal_weight_universe",
        "benchmark_metrics_status": benchmark_metrics_status,
        "benchmark_observations": int(benchmark_observations),
    }
    metrics.update(performance)

    if include_strategy_returns:
        metrics["strategy_returns"] = [float(x) for x in strategy_returns.tolist()]

    metrics.update(
        _summarize_candidate_anti_barbell_evidence(
            metrics=metrics,
            path_risk_records=path_risk_records,
            regime_labels=regime_labels,
            promotion_thresholds=barbell_promotion_thresholds,
        )
    )

    return metrics
