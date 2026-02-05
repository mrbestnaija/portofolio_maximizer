"""
Time Series Signal Generator - Convert forecasts to trading signals
Line Count: ~350 lines

Converts time series forecasts (SARIMAX, SAMOSSA, GARCH, MSSA-RL) into trading signals.
This is the DEFAULT signal generator, with LLM as fallback/redundancy.

Per refactoring plan:
- Time Series ensemble is PRIMARY signal source
- LLM serves as fallback when Time Series models fail or need validation
- Signals include confidence scores, risk metrics, and provenance
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from etl.time_series_forecaster import (
    TimeSeriesForecaster,
    RollingWindowValidator,
    RollingWindowCVConfig,
    TimeSeriesForecasterConfig,
)
from etl.portfolio_math import (
    DEFAULT_RISK_FREE_RATE,
    bootstrap_confidence_intervals,
    calculate_enhanced_portfolio_metrics,
    calculate_portfolio_metrics,
    test_strategy_significance,
)

try:  # pragma: no cover - optional import for execution references
    from execution.order_manager import request_safe_price
except Exception:  # pragma: no cover - keep generator usable without execution stack
    def request_safe_price(price: float) -> float:
        return price if price and price > 0 else 1.0

try:  # pragma: no cover - optional for environments without execution stack
    from execution.lob_simulator import LOBConfig, simulate_market_order_fill
except Exception:  # pragma: no cover - graceful fallback when execution layer absent
    LOBConfig = None  # type: ignore
    simulate_market_order_fill = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesSignal:
    """Trading signal generated from time series forecast"""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    signal_timestamp: datetime = field(default_factory=datetime.now)
    model_type: str = 'ENSEMBLE'  # 'SARIMAX', 'SAMOSSA', 'GARCH', 'MSSA_RL', 'ENSEMBLE'
    forecast_horizon: int = 30
    expected_return: float = 0.0
    risk_score: float = 0.5
    reasoning: str = ''
    provenance: Dict[str, Any] = field(default_factory=dict)
    signal_type: str = 'TIME_SERIES'
    volatility: Optional[float] = None
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None


class TimeSeriesSignalGenerator:
    """
    Generate trading signals from time series forecasts.

    This is the DEFAULT signal generator. Converts ensemble forecasts
    (SARIMAX, SAMOSSA, GARCH, MSSA-RL) into actionable trading signals
    with confidence scores and risk metrics.
    """

    def __init__(self,
                 confidence_threshold: float = 0.55,
                 min_expected_return: float = 0.003,  # 0.3% minimum to clear costs
                 max_risk_score: float = 0.7,
                 use_volatility_filter: bool = True,
                 quant_validation_config: Optional[Dict[str, Any]] = None,
                 quant_validation_config_path: Optional[str] = None,
                 per_ticker_thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
                 cost_model: Optional[Dict[str, Any]] = None):
        """
        Initialize Time Series signal generator.

        Args:
            confidence_threshold: Minimum confidence to generate signal (0.55 = 55%)
            min_expected_return: Minimum expected return to trigger signal (0.3%)
            max_risk_score: Maximum risk score allowed (0.7 = 70%)
            use_volatility_filter: Filter signals based on volatility forecasts
            quant_validation_config: Optional overrides for quant success helper
            quant_validation_config_path: Optional path to YAML config describing
                quant validation thresholds (defaults to config/quant_success_config.yml)
        """
        # Diagnostic toggle (env DIAGNOSTIC_MODE=1 or TS_DIAGNOSTIC_MODE=1) to force more signals.
        diag_mode = str(os.getenv("TS_DIAGNOSTIC_MODE") or os.getenv("DIAGNOSTIC_MODE") or "0") == "1"
        if diag_mode:
            confidence_threshold = 0.10
            min_expected_return = 0.0
            max_risk_score = 1.0
            use_volatility_filter = False

        self.confidence_threshold = confidence_threshold
        self.min_expected_return = min_expected_return
        self.max_risk_score = max_risk_score
        self.use_volatility_filter = use_volatility_filter
        self._diag_mode = diag_mode
        self._per_ticker_thresholds = per_ticker_thresholds or {}
        self._cost_model = cost_model or self._load_execution_cost_model()
        self._default_roundtrip_cost_bps = {
            "US_EQUITY": 10.0,
            "INTL_EQUITY": 15.0,
            "FX": 5.0,
            "CRYPTO": 25.0,
            "INDEX": 10.0,
            "UNKNOWN": 10.0,
        }
        overrides = self._cost_model.get("default_roundtrip_cost_bps")
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                try:
                    self._default_roundtrip_cost_bps[str(key).upper()] = float(value)
                except (TypeError, ValueError):
                    continue
        try:
            self._min_signal_to_noise = float(self._cost_model.get("min_signal_to_noise", 0.0) or 0.0)
        except (TypeError, ValueError):
            self._min_signal_to_noise = 0.0
        self._quant_validation_config_path = (
            Path(quant_validation_config_path).expanduser()
            if quant_validation_config_path
            else Path("config/quant_success_config.yml")
        )
        self.quant_validation_config = self._load_quant_validation_config(quant_validation_config)
        self._quant_validation_enabled = bool(
            self.quant_validation_config and self.quant_validation_config.get("enabled", False)
        )
        if diag_mode:
            # Disable quant validation in diagnostic mode to avoid gating signals.
            self._quant_validation_enabled = False

        # Cache for expensive forecast-edge validation runs (rolling CV).
        # Keyed by (ticker, last_bar_ts, horizon) so repeated calls on the same
        # bar (or across near-identical windows) stay fast.
        self._forecast_edge_cache: Dict[Tuple[str, str, int], Tuple[Dict[str, Any], Dict[str, bool]]] = {}

        logger.info(
            "Time Series Signal Generator initialized "
            "(confidence_threshold=%.2f, min_return=%.2f%%, max_risk=%.2f, quant_validation=%s)",
            confidence_threshold,
            min_expected_return * 100,
            max_risk_score,
            "on" if self._quant_validation_enabled else "off",
        )

    def _load_quant_validation_config(self, override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Load quant validation configuration from override or disk."""
        if override:
            return override

        path = getattr(self, "_quant_validation_config_path", None)
        if not path or not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - configuration errors are logged
            logger.warning("Unable to read quant validation config %s: %s", path, exc)
            return None

        if isinstance(payload, dict) and "quant_validation" in payload:
            return payload["quant_validation"]
        return payload if isinstance(payload, dict) else None

    def _load_execution_cost_model(self) -> Dict[str, Any]:
        """Load LOB/execution cost model configuration from disk when available."""
        cfg_path = Path("config") / "execution_cost_model.yml"
        if not cfg_path.exists():
            return {}
        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if isinstance(payload, dict):
                return payload.get("execution_cost_model") or payload
        except Exception as exc:  # pragma: no cover - config errors are non-fatal
            logger.debug("Unable to load execution cost model config: %s", exc)
        return {}

    @staticmethod
    def _normalize_forecast_payload(payload: Any) -> Optional[Dict[str, Any]]:
        """Return a dict-based payload or None if the payload is empty."""
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload if payload else None
        if isinstance(payload, pd.Series) and not payload.empty:
            return {"forecast": payload}
        return None

    def _has_payload(self, payload: Any) -> bool:
        """Safely determine whether a forecast payload contains information."""
        return self._normalize_forecast_payload(payload) is not None

    def _resolve_primary_forecast(self, forecast_bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve the primary forecast container with graceful fallback to mean or SARIMAX.
        Returns a dict payload with at least a 'forecast' Series when available.
        """
        for key in ("ensemble_forecast", "mean_forecast", "sarimax_forecast"):
            normalized = self._normalize_forecast_payload(forecast_bundle.get(key))
            if normalized:
                return normalized
        return None

    def generate_signal(self,
                       forecast_bundle: Dict[str, Any],
                       current_price: float,
                       ticker: str,
                       market_data: Optional[pd.DataFrame] = None) -> TimeSeriesSignal:
        """
        Generate trading signal from time series forecast bundle.

        Args:
            forecast_bundle: Output from TimeSeriesForecaster.forecast()
            current_price: Current market price
            ticker: Stock ticker symbol
            market_data: Optional historical data for context

        Returns:
            TimeSeriesSignal with action, confidence, and risk metrics
        """
        try:
            # Extract ensemble forecast (primary signal source)
            ensemble_forecast = self._resolve_primary_forecast(forecast_bundle)
            if ensemble_forecast is None:
                logger.warning(f"No ensemble forecast available for {ticker}, returning HOLD")
                return self._create_hold_signal(ticker, current_price, "No forecast available")

            # Get forecast target value (horizon-consistent: horizon-end by default).
            forecast_value = self._extract_forecast_value(ensemble_forecast)
            if forecast_value is None:
                return self._create_hold_signal(ticker, current_price, "Invalid forecast value")
            if current_price <= 0:
                return self._create_hold_signal(ticker, current_price, "Invalid current price")

            # Calculate expected return
            expected_return = (forecast_value / current_price) - 1
            if abs(expected_return) > 0.1:
                logger.debug(
                    "Clamping extreme expected return for %s (raw=%.2f%%)",
                    ticker,
                    expected_return * 100,
                )
                expected_return = float(np.clip(expected_return, -0.1, 0.1))

            thresholds = self._resolve_thresholds_for_ticker(ticker)
            confidence_threshold = thresholds["confidence_threshold"]
            min_expected_return = thresholds["min_expected_return"]
            max_risk_score = thresholds["max_risk_score"]

            friction = self._estimate_roundtrip_friction(
                ticker=ticker,
                market_data=market_data,
            )
            roundtrip_cost = float(friction["roundtrip_cost_fraction"])
            gross_trade_return = abs(expected_return)
            net_trade_return = max(0.0, gross_trade_return - roundtrip_cost)
            direction = float(np.sign(expected_return))
            net_expected_return = float(direction * net_trade_return) if direction != 0 else 0.0

            # Extract volatility for risk assessment (ensure scalar)
            volatility_forecast = forecast_bundle.get('volatility_forecast') or {}
            if isinstance(volatility_forecast, dict):
                volatility = self._to_scalar(volatility_forecast.get('volatility'))
            else:
                volatility = self._to_scalar(volatility_forecast)

            lower_ci, upper_ci = self._extract_ci_bounds(ensemble_forecast)
            snr = self._estimate_signal_to_noise(
                current_price=current_price,
                expected_return=expected_return,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
            )
            if snr is not None and self._min_signal_to_noise > 0 and snr < self._min_signal_to_noise:
                net_trade_return = 0.0
                net_expected_return = 0.0

            model_agreement = self._check_model_agreement(forecast_bundle)
            diagnostics = self._evaluate_diagnostics_details(forecast_bundle)

            # Calculate confidence score (discriminative: net edge + uncertainty + model quality).
            confidence = self._calculate_confidence(
                expected_return=expected_return,
                net_trade_return=net_trade_return,
                min_expected_return=min_expected_return,
                volatility=volatility,
                model_agreement=model_agreement,
                diagnostics_score=float(diagnostics.get("score", 0.5)),
                snr=snr,
            )

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                expected_return,
                volatility,
                forecast_bundle
            )

            if snr is not None and snr < 0.5:
                risk_score = min(1.0, risk_score + 0.10)

            change_point_info = self._summarize_recent_change_points(
                forecast_bundle=forecast_bundle,
                market_data=market_data,
            )
            if change_point_info:
                # Treat fresh change-points as elevated regime uncertainty.
                recent_days = change_point_info.get("recent_change_point_days")
                if isinstance(recent_days, int) and recent_days <= 10:
                    risk_score = min(1.0, risk_score + 0.10)
                    confidence = max(0.0, confidence - 0.05)

            action = self._determine_action(
                expected_return=expected_return,
                net_trade_return=net_trade_return,
                confidence=confidence,
                risk_score=risk_score,
                confidence_threshold=confidence_threshold,
                min_expected_return=min_expected_return,
                max_risk_score=max_risk_score,
            )

            # Calculate target and stop loss
            target_price, stop_loss = self._calculate_targets(
                current_price,
                forecast_value,
                volatility,
                action
            )

            # Build reasoning
            reasoning = self._build_reasoning(
                action,
                expected_return,
                net_trade_return,
                float(friction.get("roundtrip_cost_bps", roundtrip_cost * 1e4)),
                confidence,
                risk_score,
                forecast_bundle
            )

            # Extract provenance metadata
            provenance = self._extract_provenance(forecast_bundle)

            provenance["execution_friction"] = friction
            provenance["thresholds"] = {
                "confidence_threshold": confidence_threshold,
                "min_expected_return": min_expected_return,
                "max_risk_score": max_risk_score,
            }
            provenance["diagnostics"] = diagnostics
            if isinstance(diagnostics.get("reasons"), list) and diagnostics.get("reasons"):
                score = self._safe_float(diagnostics.get("score"))
                reasons = [r for r in diagnostics.get("reasons") if isinstance(r, str)]
                noisy = {"missing_regression_metrics", "missing_weighted_model_confidence"}
                filtered = [r for r in reasons if r not in noisy]
                if (score is not None and score < 0.45) and filtered:
                    provenance["why_low_quality"] = filtered[:5]
            provenance["model_agreement"] = model_agreement
            if snr is not None:
                provenance["decision_context_snr"] = snr
            if change_point_info:
                provenance["mssa_rl_change_points"] = change_point_info

            provenance['decision_context'] = {
                'expected_return': expected_return,
                'expected_return_net': net_expected_return,
                'gross_trade_return': gross_trade_return,
                'net_trade_return': net_trade_return,
                'roundtrip_cost_fraction': roundtrip_cost,
                'roundtrip_cost_bps': float(friction.get("roundtrip_cost_bps", roundtrip_cost * 1e4)),
                'confidence': confidence,
                'risk_score': risk_score,
                'volatility': volatility,
                'signal_to_noise': snr,
            }

            signal = TimeSeriesSignal(
                ticker=ticker,
                action=action,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                signal_timestamp=datetime.now(),
                model_type=provenance.get('primary_model', 'ENSEMBLE'),
                forecast_horizon=forecast_bundle.get('horizon', 30),
                expected_return=expected_return,
                risk_score=risk_score,
                reasoning=reasoning,
                provenance=provenance,
                signal_type='TIME_SERIES',
                volatility=volatility,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
            )

            quant_profile = self._build_quant_success_profile(
                ticker=ticker,
                market_data=market_data,
                signal=signal,
            )
            if quant_profile:
                status = str(quant_profile.get("status") or "").upper()
                # Attach full profile for downstream inspection.
                signal.provenance["quant_validation"] = quant_profile
                summary = f"QuantValidation={status}"
                failed = quant_profile.get("failed_criteria") or []
                if failed:
                    summary += f" ({','.join(failed)})"
                signal.reasoning = summary if not signal.reasoning else f"{signal.reasoning} | {summary}"

                # When quant validation is enabled, treat FAILED profiles as a hard
                # gate for new TS trades so that only regimes meeting the configured
                # profit_factor / win_rate / expected_profit thresholds can open
                # positions. In diagnostic mode this gate is disabled via
                # _quant_validation_enabled.
                if self._quant_validation_enabled and status == "FAIL" and action != "HOLD":
                    logger.info(
                        "Quant validation FAILED for %s; demoting %s signal to HOLD to protect PnL.",
                        ticker,
                        action,
                    )
                    action = "HOLD"
                    signal.action = "HOLD"

                self._log_quant_validation(
                    ticker=ticker,
                    signal=signal,
                    quant_profile=quant_profile,
                    market_data=market_data,
                )

            if action == 'HOLD':
                logger.info(
                    "Generated HOLD signal for %s: confidence=%.2f, expected_return=%.2f%%, risk=%.2f",
                    ticker,
                    confidence,
                    expected_return * 100,
                    risk_score,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Holding %s due to thresholds (expected_return=%.4f, confidence=%.2f, risk=%.2f)",
                        ticker,
                        expected_return,
                        confidence,
                        risk_score,
                    )
            else:
                logger.info(
                    "Generated %s signal for %s: confidence=%.2f, expected_return=%.2f%%, risk=%.2f",
                    action,
                    ticker,
                    confidence,
                    expected_return * 100,
                    risk_score,
                )

            return signal

        except Exception as e:
            logger.exception("Error generating Time Series signal for %s: %s", ticker, e)
            return self._create_hold_signal(ticker, current_price, f"Error: {str(e)}")


    def _extract_forecast_value(self, forecast: Dict[str, Any]) -> Optional[float]:
        """Extract forecast value from forecast dictionary"""
        if isinstance(forecast, dict):
            # Try different keys
            if 'forecast' in forecast:
                forecast_series = forecast['forecast']
                if isinstance(forecast_series, pd.Series):
                    if forecast_series.empty:
                        return None
                    cleaned = forecast_series.dropna()
                    if cleaned.empty:
                        return None
                    # Horizon-consistent target: use the horizon-end forecast.
                    return float(cleaned.iloc[-1])
                elif isinstance(forecast_series, (int, float)):
                    return float(forecast_series)
            elif 'mean' in forecast:
                return float(forecast['mean'])
            elif 'value' in forecast:
                return float(forecast['value'])

        return None

    def _to_scalar(self, value: Any) -> Optional[float]:
        """Convert Series/array-like values into a scalar float."""
        if value is None:
            return None
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            return float(value.iloc[0])
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) == 0:
                return None
            return float(value[0])
        if isinstance(value, (np.generic, float, int)):
            return float(value)
        return None

    def _resolve_thresholds_for_ticker(self, ticker: str) -> Dict[str, Any]:
        """Resolve per-ticker overrides for routing thresholds."""
        thresholds = {
            "confidence_threshold": float(self.confidence_threshold),
            "min_expected_return": float(self.min_expected_return),
            "max_risk_score": float(self.max_risk_score),
        }
        if self._diag_mode:
            return thresholds

        key = (ticker or "").strip()
        if not key:
            return thresholds

        raw = self._per_ticker_thresholds.get(key) or self._per_ticker_thresholds.get(key.upper())
        if not isinstance(raw, dict):
            return thresholds

        for field, cast in (
            ("confidence_threshold", float),
            ("min_expected_return", float),
            ("max_risk_score", float),
        ):
            if field in raw:
                try:
                    thresholds[field] = cast(raw[field])
                except (TypeError, ValueError):
                    continue
        return thresholds

    @staticmethod
    def _infer_asset_class(ticker: str) -> str:
        """Infer a coarse asset-class for cost defaults and routing diagnostics."""
        sym = (ticker or "").upper()
        if sym.endswith("=X"):
            return "FX"
        if sym.endswith("-USD") or sym in {"BTC", "ETH", "SOL"}:
            return "CRYPTO"
        if "^" in sym:
            return "INDEX"
        if any(sym.endswith(suffix) for suffix in (".NS", ".TW", ".L")):
            return "INTL_EQUITY"
        return "US_EQUITY"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            out = float(value)
        except Exception:
            return None
        return out

    def _estimate_roundtrip_friction(
        self,
        *,
        ticker: str,
        market_data: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Estimate round-trip friction as a fraction of notional.

        Preference order:
        1) Synthetic microstructure columns (TxnCostBps, ImpactBps) when present.
        2) LOB simulation using configured depth/spread profiles when available.
        3) Bid/Ask spread when present.
        4) Asset-class default bps.
        """
        asset_class = self._infer_asset_class(ticker)
        default_bps = float(
            self._default_roundtrip_cost_bps.get(
                asset_class, self._default_roundtrip_cost_bps["UNKNOWN"]
            )
        )

        raw_lob_cfg = self._cost_model.get("lob") if isinstance(self._cost_model, dict) else {}
        lob_cfg = raw_lob_cfg or {}
        lob_enabled = bool(lob_cfg.get("enabled", False)) and simulate_market_order_fill is not None and LOBConfig is not None
        lob_config_obj: Optional[LOBConfig] = None
        if lob_enabled:
            try:
                depth_profiles = lob_cfg.get("depth_profiles")
                if depth_profiles is not None and not isinstance(depth_profiles, dict):
                    depth_profiles = None
                lob_config_obj = LOBConfig(
                    levels=int(lob_cfg.get("levels", 10)),
                    tick_size_bps=float(lob_cfg.get("tick_size_bps", 1.0)),
                    alpha=float(lob_cfg.get("alpha", 0.8)),
                    max_exhaust_levels=int(lob_cfg.get("max_exhaust_levels", 25)),
                    default_order_value=float(lob_cfg.get("default_order_value", 10_000.0)),
                    depth_profiles=depth_profiles,
                    tail_depth_multiplier=float(lob_cfg.get("tail_depth_multiplier", 1.0)),
                )
            except Exception:
                lob_config_obj = None

        if market_data is None or market_data.empty:
            return {
                "source": "default",
                "asset_class": asset_class,
                "roundtrip_cost_bps": default_bps,
                "roundtrip_cost_fraction": default_bps / 1e4,
            }

        last = market_data.iloc[-1]

        txn_bps = None
        impact_bps = 0.0
        if "TxnCostBps" in market_data.columns:
            txn_bps = self._safe_float(last.get("TxnCostBps") if hasattr(last, "get") else None)
        if txn_bps is not None and "ImpactBps" in market_data.columns:
            impact_val = self._safe_float(last.get("ImpactBps") if hasattr(last, "get") else None)
            impact_bps = impact_val if impact_val is not None else 0.0

        if txn_bps is not None:
            per_side_bps = max(0.0, float(txn_bps) + float(impact_bps))
            roundtrip_bps = 2.0 * per_side_bps
            return {
                "source": "microstructure",
                "asset_class": asset_class,
                "txn_cost_bps": float(txn_bps),
                "impact_bps": float(impact_bps),
                "roundtrip_cost_bps": float(roundtrip_bps),
                "roundtrip_cost_fraction": float(roundtrip_bps) / 1e4,
            }

        bid = self._safe_float(last.get("Bid") if hasattr(last, "get") else None)
        ask = self._safe_float(last.get("Ask") if hasattr(last, "get") else None)
        mid_price = None
        half_spread = None
        if bid is not None and ask is not None and bid > 0 and ask > bid:
            mid_price = 0.5 * (bid + ask)
            half_spread = 0.5 * (ask - bid)
        else:
            close_val = self._safe_float(last.get("Close") if hasattr(last, "get") else None)
            high_val = self._safe_float(last.get("High") if hasattr(last, "get") else None)
            low_val = self._safe_float(last.get("Low") if hasattr(last, "get") else None)
            if high_val is not None and low_val is not None and high_val > 0 and low_val > 0:
                mid_price = 0.5 * (high_val + low_val)
            elif close_val is not None:
                mid_price = close_val
            spread_val = self._safe_float(last.get("Spread") if hasattr(last, "get") else None)
            if spread_val is not None and spread_val > 0:
                half_spread = spread_val / 2.0

        depth_notional = self._safe_float(last.get("Depth") if hasattr(last, "get") else None)

        if lob_enabled and lob_config_obj is not None and mid_price is not None:
            try:
                fill = simulate_market_order_fill(
                    side="BUY",
                    mid_price=mid_price,
                    half_spread=half_spread,
                    depth_notional=depth_notional,
                    order_notional=None,
                    asset_class=asset_class,
                    config=lob_config_obj,
                )
                lob_roundtrip_bps = max(default_bps, abs(fill.mid_slippage_bps) * 2.0)
                return {
                    "source": "lob_sim",
                    "asset_class": asset_class,
                    "mid_price": mid_price,
                    "depth_notional": depth_notional,
                    "levels_consumed": fill.levels_consumed,
                    "roundtrip_cost_bps": float(lob_roundtrip_bps),
                    "roundtrip_cost_fraction": float(lob_roundtrip_bps) / 1e4,
                }
            except Exception as exc:
                logger.debug("LOB cost estimation failed, falling back: %s", exc)

        if bid is not None and ask is not None and bid > 0 and ask > bid:
            spread = ask - bid
            spread_bps = (spread / (0.5 * (bid + ask))) * 1e4 if (bid + ask) > 0 else default_bps
            roundtrip_bps = max(spread_bps, default_bps)
            return {
                "source": "bid_ask",
                "asset_class": asset_class,
                "bid": bid,
                "ask": ask,
                "spread_bps": float(spread_bps),
                "roundtrip_cost_bps": float(roundtrip_bps),
                "roundtrip_cost_fraction": float(roundtrip_bps) / 1e4,
            }

        return {
            "source": "default",
            "asset_class": asset_class,
            "roundtrip_cost_bps": default_bps,
            "roundtrip_cost_fraction": default_bps / 1e4,
        }

    def _extract_ci_bounds(self, forecast_payload: Optional[Dict[str, Any]]) -> tuple[Optional[float], Optional[float]]:
        if not forecast_payload:
            return None, None
        lower_ci = self._to_scalar(forecast_payload.get("lower_ci"))
        upper_ci = self._to_scalar(forecast_payload.get("upper_ci"))
        if lower_ci is None or upper_ci is None:
            return lower_ci, upper_ci
        if not np.isfinite(lower_ci) or not np.isfinite(upper_ci):
            return None, None
        return float(lower_ci), float(upper_ci)

    @staticmethod
    def _estimate_signal_to_noise(
        *,
        current_price: float,
        expected_return: float,
        lower_ci: Optional[float],
        upper_ci: Optional[float],
        z_value: float = 1.96,
    ) -> Optional[float]:
        """Estimate signal-to-noise ratio using CI-implied sigma (approximate)."""
        if lower_ci is None or upper_ci is None:
            return None
        if current_price <= 0:
            return None
        width = float(upper_ci) - float(lower_ci)
        if not np.isfinite(width) or width <= 0:
            return None
        sigma_price = (width / 2.0) / max(z_value, 1e-6)
        sigma_return = sigma_price / float(current_price)
        if sigma_return <= 0 or not np.isfinite(sigma_return):
            return None
        return float(abs(expected_return) / sigma_return)

    @staticmethod
    def _summarize_recent_change_points(
        *,
        forecast_bundle: Dict[str, Any],
        market_data: Optional[pd.DataFrame],
    ) -> Optional[Dict[str, Any]]:
        """Summarize MSSA-RL change-point recency for risk adjustments."""
        payload = forecast_bundle.get("mssa_rl_forecast")
        if not isinstance(payload, dict) or not payload:
            return None
        change_points = payload.get("change_points")
        if change_points is None:
            return None
        if market_data is None or market_data.empty:
            return None
        try:
            end_ts = pd.to_datetime(market_data.index[-1])
        except Exception:
            return None

        cps: List[pd.Timestamp] = []
        if isinstance(change_points, pd.DatetimeIndex):
            cps = [pd.to_datetime(ts) for ts in change_points.to_list()]
        elif isinstance(change_points, list):
            for item in change_points:
                try:
                    cps.append(pd.to_datetime(item))
                except Exception:
                    continue
        else:
            try:
                cps = [pd.to_datetime(change_points)]
            except Exception:
                cps = []

        cps = [ts for ts in cps if ts is not None and pd.notna(ts)]
        if not cps:
            return {"count": 0, "recent_change_point_days": None}

        last_cp = max(cps)
        try:
            recent_days = int(abs((end_ts - last_cp).days))
        except Exception:
            recent_days = None

        return {
            "count": int(len(cps)),
            "last_change_point": str(last_cp),
            "recent_change_point_days": recent_days,
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _calculate_confidence(
        self,
        *,
        expected_return: float,
        net_trade_return: float,
        min_expected_return: float,
        volatility: Optional[float],
        model_agreement: float,
        diagnostics_score: float,
        snr: Optional[float],
    ) -> float:
        """
        Discriminative confidence score (0.0..1.0).

        Confidence should track *actionable predictive edge*:
        - penalize tiny net edges after costs
        - downweight high-uncertainty forecasts (CI/SNR)
        - incorporate model quality/diagnostics and agreement
        """

        # 1) Edge score: 0 when net edge is tiny, 1 when it is meaningfully above threshold.
        threshold = max(float(min_expected_return), 1e-6)
        edge_ratio = float(net_trade_return) / threshold if threshold > 0 else float(net_trade_return)
        edge_score = self._clamp01(edge_ratio / 3.0)  # 3x threshold -> full credit

        # 2) Uncertainty score from SNR (CI-implied).
        if snr is None:
            snr_score = 0.5
        else:
            snr_score = self._clamp01((float(snr) - 0.5) / 1.5)  # 0.5 -> 0, 2.0 -> 1

        # 3) Volatility factor (optional soft penalty; don't artificially inflate).
        vol_factor = 1.0
        if volatility is not None and self.use_volatility_filter:
            vol = float(volatility)
            if vol >= 0.60:
                vol_factor = 0.60
            elif vol >= 0.40:
                vol_factor = 0.75
            elif vol <= 0.15:
                vol_factor = 1.05

        # 4) Combine model-quality features with edge/uncertainty.
        # Edge is additive (not multiplicative) so marginal edges can still
        # clear confidence when model agreement/quality is strong, while
        # remaining discriminative across regimes.
        core = (
            0.30 * self._clamp01(diagnostics_score)
            + 0.25 * self._clamp01(model_agreement)
            + 0.20 * self._clamp01(snr_score)
            + 0.25 * self._clamp01(edge_score)
        )
        confidence = (0.05 + 0.95 * core) * vol_factor
        # Very small gross expected_return should be conservative even if other signals look good.
        if abs(float(expected_return)) < float(min_expected_return):
            confidence *= 0.75

        return self._clamp01(confidence)

    def _check_model_agreement(self, forecast_bundle: Dict[str, Any]) -> float:
        """Check agreement between different models (0.0 to 1.0)"""
        forecasts = []

        # Collect forecasts from all models
        sarimax_payload = forecast_bundle.get('sarimax_forecast')
        if self._has_payload(sarimax_payload):
            sarimax_val = self._extract_forecast_value(sarimax_payload)
            if sarimax_val is not None:
                forecasts.append(sarimax_val)

        samossa_payload = forecast_bundle.get('samossa_forecast')
        if self._has_payload(samossa_payload):
            samossa_val = self._extract_forecast_value(samossa_payload)
            if samossa_val is not None:
                forecasts.append(samossa_val)

        mssa_payload = forecast_bundle.get('mssa_rl_forecast')
        if self._has_payload(mssa_payload):
            mssa_val = self._extract_forecast_value(mssa_payload)
            if mssa_val is not None:
                forecasts.append(mssa_val)

        if len(forecasts) < 2:
            return 0.5  # Can't assess agreement with <2 models

        # Calculate coefficient of variation (lower = more agreement)
        mean_forecast = np.mean(forecasts)
        std_forecast = np.std(forecasts)

        if mean_forecast == 0:
            return 0.5

        cv = std_forecast / abs(mean_forecast)

        # Convert CV to agreement score (lower CV = higher agreement)
        # CV < 0.1 = excellent agreement (1.0)
        # CV > 0.5 = poor agreement (0.0)
        agreement = max(0.0, min(1.0, 1.0 - (cv - 0.1) / 0.4))

        return agreement

    def _evaluate_diagnostics_details(self, forecast_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate diagnostics quality and return a score + reasons (for provenance)."""
        reasons: List[str] = []

        ensemble_metadata = forecast_bundle.get("ensemble_metadata") or {}
        weights = ensemble_metadata.get("weights") if isinstance(ensemble_metadata, dict) else None
        model_conf = ensemble_metadata.get("confidence") if isinstance(ensemble_metadata, dict) else None

        weighted_conf = None
        if isinstance(weights, dict) and isinstance(model_conf, dict) and weights:
            num = 0.0
            den = 0.0
            for model_key, weight in weights.items():
                try:
                    w = float(weight)
                except (TypeError, ValueError):
                    continue
                if w <= 0:
                    continue
                c = model_conf.get(str(model_key).lower())
                try:
                    c_val = float(c)
                except (TypeError, ValueError):
                    continue
                num += w * self._clamp01(c_val)
                den += w
            if den > 0:
                weighted_conf = num / den
        if weighted_conf is None:
            reasons.append("missing_weighted_model_confidence")
            weighted_conf = 0.5

        regression_metrics = forecast_bundle.get("regression_metrics") or {}
        rmse_ratio = None
        dir_acc = None
        if isinstance(regression_metrics, dict) and regression_metrics:
            ens = regression_metrics.get("ensemble") or {}
            base = regression_metrics.get("samossa") or regression_metrics.get("sarimax") or {}
            if isinstance(ens, dict):
                try:
                    dir_acc = float(ens.get("directional_accuracy")) if ens.get("directional_accuracy") is not None else None
                except (TypeError, ValueError):
                    dir_acc = None
            try:
                ens_rmse = float(ens.get("rmse")) if isinstance(ens, dict) and ens.get("rmse") is not None else None
                base_rmse = float(base.get("rmse")) if isinstance(base, dict) and base.get("rmse") is not None else None
                if ens_rmse is not None and base_rmse and base_rmse > 0:
                    rmse_ratio = ens_rmse / base_rmse
            except (TypeError, ValueError):
                rmse_ratio = None
        else:
            reasons.append("missing_regression_metrics")

        model_errors = forecast_bundle.get("model_errors") or {}
        if isinstance(model_errors, dict) and any(model_errors.values()):
            reasons.append("model_errors_present")

        # Diagnostics score: start from weighted model confidence, then apply
        # conservative penalties/boosts from regression metrics when available.
        score = float(weighted_conf)
        if rmse_ratio is not None:
            # Penalize when worse than baseline; soft reward when better.
            if rmse_ratio > 1.10:
                reasons.append("rmse_worse_than_baseline")
                score *= 0.70
            elif rmse_ratio < 0.95:
                score = min(1.0, score + 0.05)
        if dir_acc is not None:
            if dir_acc < 0.50:
                reasons.append("directional_accuracy_below_50pct")
                score *= 0.80
            elif dir_acc > 0.55:
                score = min(1.0, score + 0.05)
        if "model_errors_present" in reasons:
            score *= 0.70

        return {
            "score": self._clamp01(score),
            "reasons": reasons,
            "weighted_model_confidence": float(weighted_conf) if weighted_conf is not None else None,
            "rmse_ratio_vs_baseline": rmse_ratio,
            "directional_accuracy": dir_acc,
        }

    def _calculate_risk_score(self,
                              expected_return: float,
                              volatility: Optional[float],
                              forecast_bundle: Dict[str, Any]) -> float:
        """
        Calculate risk score (0.0 to 1.0, higher = riskier).

        Based on:
        - Volatility forecast
        - Confidence interval width
        - Model uncertainty
        """
        risk = 0.5  # Base risk

        # Factor 1: Volatility (higher volatility = higher risk)
        if volatility is not None:
            if volatility > 0.40:  # >40% volatility
                risk += 0.3
            elif volatility > 0.25:  # >25% volatility
                risk += 0.15
            elif volatility < 0.15:  # <15% volatility
                risk -= 0.15

        # Factor 2: Confidence interval width (wider = riskier)
        ensemble_forecast = self._resolve_primary_forecast(forecast_bundle)
        if ensemble_forecast:
            lower_ci, upper_ci = self._extract_ci_bounds(ensemble_forecast)
            forecast_price = self._extract_forecast_value(ensemble_forecast)

            if lower_ci is not None and upper_ci is not None and forecast_price is not None:
                ci_width_price = float(upper_ci - lower_ci)
                # Convert to a dimensionless uncertainty ratio: CI width relative to expected move.
                denom = abs(float(expected_return)) * max(float(forecast_price) / max(1.0 + float(expected_return), 1e-6), 1e-6)
                uncertainty_ratio = ci_width_price / denom if denom > 0 else float("inf")
                if uncertainty_ratio > 2.0:
                    risk += 0.2
                elif uncertainty_ratio < 0.75:
                    risk -= 0.1

        # Factor 3: Expected return magnitude (smaller moves = riskier relative to reward)
        if abs(expected_return) < 0.01:  # <1% expected return
            risk += 0.1

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, risk))

    def _determine_action(self,
                         expected_return: float,
                         net_trade_return: float,
                         confidence: float,
                         risk_score: float,
                         confidence_threshold: float,
                         min_expected_return: float,
                         max_risk_score: float) -> str:
        """
        Determine trading action based on forecast and risk metrics.

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Must meet minimum thresholds
        if confidence < confidence_threshold:
            return 'HOLD'

        # net_trade_return is always non-negative and already clears estimated friction.
        if net_trade_return < min_expected_return:
            return 'HOLD'

        if risk_score > max_risk_score:
            return 'HOLD'

        # Determine direction
        if expected_return > 0:
            return 'BUY'
        if expected_return < 0:
            return 'SELL'
        return 'HOLD'

    def _calculate_targets(self,
                          current_price: float,
                          forecast_price: float,
                          volatility: Optional[float],
                          action: str) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate target price and stop loss.

        Returns:
            (target_price, stop_loss)
        """
        if action == 'HOLD':
            return None, None

        # Target: forecast price (or 2:1 reward/risk ratio)
        target_price = forecast_price

        # Stop loss: based on volatility or 2% default
        if volatility is not None:
            stop_loss_pct = max(0.015, min(0.05, volatility * 0.5))  # 1.5% to 5%
        else:
            stop_loss_pct = 0.02  # 2% default

        if action == 'BUY':
            stop_loss = current_price * (1 - stop_loss_pct)
        else:  # SELL
            stop_loss = current_price * (1 + stop_loss_pct)

        return target_price, stop_loss

    def _build_reasoning(self,
                        action: str,
                        expected_return: float,
                        net_trade_return: float,
                        roundtrip_cost_bps: float,
                        confidence: float,
                        risk_score: float,
                        forecast_bundle: Dict[str, Any]) -> str:
        """Build human-readable reasoning for signal"""
        model_type = forecast_bundle.get('ensemble_metadata', {}).get('primary_model', 'ENSEMBLE')

        reasoning = (
            f"Time Series {model_type} forecast indicates {action} signal. "
            f"Gross move: {expected_return:.2%}, "
            f"Net edge: {net_trade_return:.2%} (est cost: {roundtrip_cost_bps:.1f}bp), "
            f"Confidence: {confidence:.1%}, "
            f"Risk score: {risk_score:.1%}. "
        )

        if self._has_payload(forecast_bundle.get('samossa_forecast')):
            reasoning += "SAMOSSA SSA decomposition confirms trend. "

        if self._has_payload(forecast_bundle.get('mssa_rl_forecast')):
            reasoning += "MSSA-RL change-point detection active. "

        return reasoning.strip()

    def _extract_provenance(self, forecast_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Extract provenance metadata from forecast bundle"""
        provenance = {
            'model_type': 'TIME_SERIES_ENSEMBLE',
            'timestamp': datetime.now().isoformat(),
            'forecast_horizon': forecast_bundle.get('horizon', 30)
        }

        # Add model-specific metadata
        ensemble_metadata = forecast_bundle.get('ensemble_metadata', {})
        if ensemble_metadata:
            provenance.update({
                'primary_model': ensemble_metadata.get('primary_model', 'ENSEMBLE'),
                'model_weights': ensemble_metadata.get('weights', {}),
                'aic': ensemble_metadata.get('aic'),
                'bic': ensemble_metadata.get('bic')
            })

        # Add individual model flags
        provenance['models_used'] = []
        if self._has_payload(forecast_bundle.get('sarimax_forecast')):
            provenance['models_used'].append('SARIMAX')
        if self._has_payload(forecast_bundle.get('samossa_forecast')):
            provenance['models_used'].append('SAMOSSA')
        if self._has_payload(forecast_bundle.get('mssa_rl_forecast')):
            provenance['models_used'].append('MSSA_RL')
        if self._has_payload(forecast_bundle.get('garch_forecast')):
            provenance['models_used'].append('GARCH')

        return provenance

    def _create_hold_signal(self,
                           ticker: str,
                           current_price: float,
                           reason: str) -> TimeSeriesSignal:
        """Create a HOLD signal"""
        timestamp = datetime.now()
        return TimeSeriesSignal(
            ticker=ticker,
            action='HOLD',
            confidence=0.0,
            entry_price=current_price,
            signal_timestamp=timestamp,
            model_type='ENSEMBLE',
            expected_return=0.0,
            risk_score=0.5,
            reasoning=f"HOLD: {reason}",
            provenance={
                'model_type': 'TIME_SERIES_ENSEMBLE',
                'reason': reason,
                'timestamp': timestamp.isoformat(),
            },
            signal_type='TIME_SERIES'
        )

    def _build_quant_success_profile(
        self,
        ticker: str,
        market_data: Optional[pd.DataFrame],
        signal: TimeSeriesSignal,
    ) -> Optional[Dict[str, Any]]:
        """
        Build quantitative success profile referencing institutional criteria.
        """
        if not self._quant_validation_enabled:
            return None
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return None

        config = self.quant_validation_config or {}
        lookback = max(int(config.get('lookback_days', 120)), 2)
        price_series = market_data['Close'].tail(lookback).dropna()
        if len(price_series) < 2:
            return None

        prices = np.maximum(price_series.astype(float).values, 1e-6)
        log_returns = np.diff(np.log(prices))
        if log_returns.size < 2:
            return None

        action = str(getattr(signal, "action", None) or "HOLD").upper()
        if action == "SELL":
            direction = -1.0
        else:
            direction = 1.0
        strategy_returns = log_returns * direction

        try:
            metrics = calculate_enhanced_portfolio_metrics(
                returns=strategy_returns.reshape(-1, 1),
                weights=np.array([1.0]),
                risk_free_rate=float(config.get('risk_free_rate', DEFAULT_RISK_FREE_RATE)),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to compute quant metrics for %s: %s", ticker, exc)
            return None

        bootstrap_cfg = config.get('bootstrap', {})
        try:
            bootstrap_stats = bootstrap_confidence_intervals(
                returns=strategy_returns,
                n_bootstrap=int(bootstrap_cfg.get('n_samples', 500)),
                confidence_level=float(bootstrap_cfg.get('confidence_level', 0.95)),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Bootstrap stats unavailable for %s: %s", ticker, exc)
            bootstrap_stats = {}

        benchmark_col = config.get('benchmark_column')
        significance = None
        if benchmark_col and benchmark_col in market_data.columns:
            benchmark_prices = market_data[benchmark_col].tail(len(price_series)).dropna()
            bench_values = np.maximum(benchmark_prices.astype(float).values, 1e-6)
            bench_returns = np.diff(np.log(bench_values))
            if bench_returns.size >= log_returns.size:
                benchmark_slice = bench_returns[-log_returns.size:]
                try:
                    significance = test_strategy_significance(strategy_returns, benchmark_slice * direction)
                except ValueError:
                    significance = None

        performance_snapshot = self._calculate_return_based_performance(strategy_returns)

        # Allow per-ticker overrides for success criteria so that
        # higher-volatility or structurally weaker assets (e.g. crypto,
        # certain commodities/FX) can demand a higher expected_profit
        # before passing quant validation, without penalising the entire
        # universe.
        per_ticker_cfg = (config.get('per_ticker') or {}).get(ticker, {})
        if isinstance(per_ticker_cfg, dict) and per_ticker_cfg.get('success_criteria'):
            criteria_cfg = per_ticker_cfg['success_criteria']
        else:
            criteria_cfg = config.get('success_criteria', {})

        capital_base = float(criteria_cfg.get('capital_base', config.get('capital_base', 25000.0)))
        allocation = min(max(float(signal.confidence or 0.0), 0.05), 1.0)
        safe_price = request_safe_price(signal.entry_price)
        position_value = capital_base * allocation
        ctx = (signal.provenance or {}).get("decision_context") or {}
        try:
            net_trade_return = float(ctx.get("net_trade_return"))
        except (TypeError, ValueError):
            net_trade_return = max(0.0, abs(float(signal.expected_return or 0.0)))
        expected_profit = position_value * net_trade_return
        criteria = self._evaluate_success_criteria(
            criteria_cfg=criteria_cfg,
            metrics=metrics,
            performance_snapshot=performance_snapshot,
            significance=significance,
            expected_profit=expected_profit,
        )
        drift_criteria = dict(criteria) if isinstance(criteria, dict) else {}

        # Forecast-edge validation (optional): measure incremental edge using
        # rolling-window forecast regression metrics instead of drift proxy.
        validation_mode = str(config.get("validation_mode") or "drift_proxy").lower()
        edge_block: Dict[str, Any] = {}
        if validation_mode == "forecast_edge":
            edge_block, edge_criteria = self._evaluate_forecast_edge(
                price_series=price_series,
                signal=signal,
                config=config,
                criteria_cfg=criteria_cfg,
            )
            if edge_criteria:
                # Edge criteria are the primary gate; keep drift-proxy criteria
                # available in metrics but do not require them unless explicitly requested.
                include_drift = bool(criteria_cfg.get("include_drift_proxy_criteria", False))
                criteria = {"expected_profit": drift_criteria.get("expected_profit", True), **edge_criteria}
                if include_drift:
                    criteria.update({f"drift_{k}": v for k, v in drift_criteria.items() if k != "expected_profit"})

        viz_cfg = config.get('visualization') or {}
        visualization_result = None
        if viz_cfg.get('enabled'):
            visualization_result = self._render_quant_validation_plot(
                ticker=ticker,
                market_data=market_data,
                output_dir=viz_cfg.get('output_dir', 'visualizations/quant_validation'),
                max_points=int(viz_cfg.get('max_points', lookback)),
            )

        failed = [name for name, passed in criteria.items() if not passed]
        return {
            'status': 'PASS' if criteria and all(criteria.values()) else 'FAIL' if criteria else 'SKIPPED',
            'metrics': {
                'annual_return': metrics.get('annual_return'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'sortino_ratio': metrics.get('sortino_ratio'),
                'max_drawdown': metrics.get('max_drawdown'),
                'volatility': metrics.get('volatility'),
                'win_rate': performance_snapshot.get('win_rate'),
                'profit_factor': performance_snapshot.get('profit_factor'),
            },
            'forecast_edge': edge_block,
            'bootstrap': bootstrap_stats,
            'criteria': criteria,
            'failed_criteria': failed,
            'significance': significance,
            'lookback_bars': int(len(price_series)),
            'expected_profit': expected_profit,
            'capital_base': capital_base,
            'position_value': position_value,
            'estimated_shares': position_value / safe_price if safe_price else None,
            'performance_snapshot': performance_snapshot,
            'visualization': {'path': visualization_result} if visualization_result else {},
        }

    def _evaluate_forecast_edge(
        self,
        *,
        price_series: pd.Series,
        signal: TimeSeriesSignal,
        config: Dict[str, Any],
        criteria_cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        """Compute rolling CV regression metrics and evaluate forecast-edge criteria."""
        cv_cfg = config.get("forecast_edge_cv") if isinstance(config, dict) else None
        cv_cfg = cv_cfg if isinstance(cv_cfg, dict) else {}

        horizon = int(cv_cfg.get("horizon") or getattr(signal, "forecast_horizon", 30) or 30)
        horizon = max(horizon, 1)
        min_train_size = int(cv_cfg.get("min_train_size") or 180)
        step_size = int(cv_cfg.get("step_size") or horizon)
        max_folds = cv_cfg.get("max_folds")
        try:
            max_folds_int = int(max_folds) if max_folds is not None else None
        except (TypeError, ValueError):
            max_folds_int = None
        baseline_model = str(
            criteria_cfg.get("baseline_model")
            or cv_cfg.get("baseline_model")
            or "samossa"
        ).lower()

        symbol = str(getattr(signal, "ticker", "") or "").upper()
        last_ts = ""
        try:
            last_ts = pd.to_datetime(price_series.index[-1]).isoformat()
        except Exception:
            last_ts = str(price_series.index[-1]) if len(price_series.index) else ""
        cache_key = (symbol, last_ts, horizon)
        cached = self._forecast_edge_cache.get(cache_key)
        if cached is not None:
            return cached

        edge_payload: Dict[str, Any] = {
            "mode": "forecast_edge",
            "baseline_model": baseline_model,
            "horizon": horizon,
        }
        edge_criteria: Dict[str, bool] = {}

        try:
            returns_series = price_series.pct_change().dropna()
            validator = RollingWindowValidator(
                forecaster_config=TimeSeriesForecasterConfig(forecast_horizon=horizon),
                cv_config=RollingWindowCVConfig(
                    min_train_size=int(min_train_size),
                    horizon=int(horizon),
                    step_size=int(step_size),
                    max_folds=max_folds_int,
                ),
            )
            report = validator.run(price_series=price_series, returns_series=returns_series)
            aggregate = report.get("aggregate_metrics") or {}
            fold_count = int(report.get("fold_count") or 0)
            ens = aggregate.get("ensemble") or {}
            base = aggregate.get(baseline_model) or {}
            edge_payload.update(
                {
                    "fold_count": fold_count,
                    "ensemble": ens,
                    "baseline": base,
                }
            )

            rmse_ratio = None
            try:
                ens_rmse = float(ens.get("rmse")) if isinstance(ens, dict) and ens.get("rmse") is not None else None
                base_rmse = float(base.get("rmse")) if isinstance(base, dict) and base.get("rmse") is not None else None
                if ens_rmse is not None and base_rmse and base_rmse > 0:
                    rmse_ratio = ens_rmse / base_rmse
            except (TypeError, ValueError):
                rmse_ratio = None
            edge_payload["rmse_ratio_vs_baseline"] = rmse_ratio

            dir_acc = None
            try:
                dir_acc = float(ens.get("directional_accuracy")) if isinstance(ens, dict) and ens.get("directional_accuracy") is not None else None
            except (TypeError, ValueError):
                dir_acc = None
            edge_payload["directional_accuracy"] = dir_acc

            max_rmse_ratio = criteria_cfg.get("max_rmse_ratio_vs_baseline")
            if max_rmse_ratio is not None:
                try:
                    thr = float(max_rmse_ratio)
                    edge_criteria["rmse_ratio_vs_baseline"] = rmse_ratio is not None and rmse_ratio <= thr
                except (TypeError, ValueError):
                    pass
            min_dir_acc = criteria_cfg.get("min_directional_accuracy")
            if min_dir_acc is not None:
                try:
                    thr = float(min_dir_acc)
                    edge_criteria["directional_accuracy"] = dir_acc is not None and dir_acc >= thr
                except (TypeError, ValueError):
                    pass
            if "rmse_ratio_vs_baseline" not in edge_criteria and rmse_ratio is not None:
                # Conservative default when no explicit criteria provided.
                edge_criteria["rmse_ratio_vs_baseline"] = rmse_ratio <= 1.10
            if "directional_accuracy" not in edge_criteria and dir_acc is not None:
                edge_criteria["directional_accuracy"] = dir_acc >= 0.50
        except Exception as exc:  # pragma: no cover - best-effort metric
            edge_payload["error"] = str(exc)

        if len(self._forecast_edge_cache) >= 256:
            self._forecast_edge_cache.clear()
        self._forecast_edge_cache[cache_key] = (edge_payload, edge_criteria)
        return edge_payload, edge_criteria

    @staticmethod
    def _calculate_return_based_performance(returns: np.ndarray) -> Dict[str, float]:
        """Return simplified performance stats used for config thresholds."""
        if returns.size == 0:
            return {'win_rate': 0.0, 'profit_factor': 0.0, 'gross_profit': 0.0, 'gross_loss': 0.0,
                    'avg_gain': 0.0, 'avg_loss': 0.0}

        positive = returns[returns > 0]
        negative = returns[returns < 0]
        gross_profit = float(positive.sum()) if positive.size else 0.0
        gross_loss = abs(float(negative.sum())) if negative.size else 0.0
        if gross_loss > 1e-8:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0

        return {
            'win_rate': float(positive.size / returns.size),
            'profit_factor': float(profit_factor),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_gain': float(positive.mean()) if positive.size else 0.0,
            'avg_loss': float(negative.mean()) if negative.size else 0.0,
        }

    @staticmethod
    def _evaluate_success_criteria(
        criteria_cfg: Optional[Dict[str, Any]],
        metrics: Dict[str, float],
        performance_snapshot: Dict[str, float],
        significance: Optional[Dict[str, Any]],
        expected_profit: float,
    ) -> Dict[str, bool]:
        """Evaluate metrics vs configuration thresholds."""
        if not isinstance(criteria_cfg, dict) or not criteria_cfg:
            return {}

        results: Dict[str, bool] = {}

        if 'min_annual_return' in criteria_cfg and 'annual_return' in metrics:
            results['annual_return'] = metrics['annual_return'] >= float(criteria_cfg['min_annual_return'])

        if 'min_sharpe' in criteria_cfg and 'sharpe_ratio' in metrics:
            results['sharpe_ratio'] = metrics['sharpe_ratio'] >= float(criteria_cfg['min_sharpe'])

        if 'min_sortino' in criteria_cfg and 'sortino_ratio' in metrics:
            results['sortino_ratio'] = metrics['sortino_ratio'] >= float(criteria_cfg['min_sortino'])

        if 'max_drawdown' in criteria_cfg and 'max_drawdown' in metrics:
            results['max_drawdown'] = metrics['max_drawdown'] <= float(criteria_cfg['max_drawdown'])

        if 'min_profit_factor' in criteria_cfg:
            pf = performance_snapshot.get('profit_factor', 0.0)
            results['profit_factor'] = pf >= float(criteria_cfg['min_profit_factor'])

        if 'min_win_rate' in criteria_cfg:
            wr = performance_snapshot.get('win_rate', 0.0)
            results['win_rate'] = wr >= float(criteria_cfg['min_win_rate'])

        if 'min_expected_profit' in criteria_cfg:
            results['expected_profit'] = expected_profit >= float(criteria_cfg['min_expected_profit'])

        if criteria_cfg.get('require_significance'):
            if significance is None:
                results['statistical_significance'] = False
            else:
                results['statistical_significance'] = bool(significance.get('significant'))

        if 'min_information_ratio' in criteria_cfg and significance is not None:
            ir = significance.get('information_ratio')
            if ir is not None:
                results['information_ratio'] = float(ir) >= float(criteria_cfg['min_information_ratio'])

        return results

    def _render_quant_validation_plot(
        self,
        ticker: str,
        market_data: pd.DataFrame,
        output_dir: str,
        max_points: int,
    ) -> Optional[str]:
        """Persist optional visualization artifact for dashboards."""
        try:
            from etl.visualizer import TimeSeriesVisualizer  # pylint: disable=import-outside-toplevel
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover
            logger.debug("Quant validation visualization skipped: %s", exc)
            return None

        subset = market_data[['Close']].tail(max(max_points, 2))
        if subset.empty:
            return None

        visualizer = TimeSeriesVisualizer()
        fig = visualizer.plot_time_series_overview(subset, title=f"{ticker} Quant Validation")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / f"{ticker}_quant_validation.png"
        fig.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return str(target)

    def _log_quant_validation(
        self,
        ticker: str,
        signal: TimeSeriesSignal,
        quant_profile: Dict[str, Any],
        market_data: Optional[pd.DataFrame],
    ) -> None:
        """Persist quant validation output for downstream troubleshooting."""
        config = self.quant_validation_config or {}
        logging_cfg = config.get('logging') or {}
        if not logging_cfg.get('enabled'):
            return

        log_dir = Path(logging_cfg.get('log_dir', 'logs/signals'))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / logging_cfg.get('filename', 'quant_validation.jsonl')

        pipeline_id = (
            os.environ.get('PORTFOLIO_MAXIMIZER_PIPELINE_ID')
            or os.environ.get('PIPELINE_ID')
        )
        visualization_path = (
            (quant_profile.get('visualization') or {}).get('path')
            if isinstance(quant_profile, dict)
            else None
        )

        market_context: Dict[str, Any] = {
            'rows': int(market_data.shape[0]) if market_data is not None else 0,
            'start': None,
            'end': None,
            'data_source': None,
        }
        if market_data is not None and hasattr(market_data, 'index') and len(market_data.index) > 0:
            start_idx = market_data.index[0]
            end_idx = market_data.index[-1]
            market_context['start'] = (
                start_idx.isoformat()
                if hasattr(start_idx, 'isoformat')
                else str(start_idx)
            )
            market_context['end'] = (
                end_idx.isoformat()
                if hasattr(end_idx, 'isoformat')
                else str(end_idx)
            )
        if hasattr(market_data, 'attrs'):
            market_context['data_source'] = market_data.attrs.get('source')

        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'pipeline_id': pipeline_id,
            'ticker': ticker,
            'action': signal.action,
            'confidence': signal.confidence,
            'expected_return': signal.expected_return,
            'risk_score': signal.risk_score,
            'volatility': signal.volatility,
            'status': quant_profile.get('status'),
            'failed_criteria': quant_profile.get('failed_criteria'),
            'position_value': quant_profile.get('position_value'),
            'estimated_shares': quant_profile.get('estimated_shares'),
            'visualization_path': visualization_path,
            'quant_validation': quant_profile,
            'market_context': market_context,
        }

        try:
            with log_file.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, default=self._json_serializer) + "\n")
        except Exception as exc:  # pragma: no cover - logging must not break signals
            logger.warning("Unable to persist quant validation log for %s: %s", ticker, exc)

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Serializer for numpy/pandas objects when dumping JSON."""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, (np.ndarray, list, tuple)):
            return list(obj)
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    def generate_signals_batch(self,
                               forecast_bundles: Dict[str, Dict[str, Any]],
                               current_prices: Dict[str, float],
                               market_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, TimeSeriesSignal]:
        """
        Generate signals for multiple tickers.

        Args:
            forecast_bundles: Dict mapping ticker to forecast bundle
            current_prices: Dict mapping ticker to current price
            market_data: Optional dict mapping ticker to market data

        Returns:
            Dict mapping ticker to TimeSeriesSignal
        """
        signals = {}

        for ticker, forecast_bundle in forecast_bundles.items():
            current_price = current_prices.get(ticker)
            if current_price is None:
                logger.warning(f"No current price for {ticker}, skipping")
                continue

            market_data_ticker = market_data.get(ticker) if market_data else None

            signal = self.generate_signal(
                forecast_bundle,
                current_price,
                ticker,
                market_data_ticker
            )

            signals[ticker] = signal

        return signals


# Validation
assert TimeSeriesSignalGenerator.generate_signal.__doc__ is not None
assert TimeSeriesSignalGenerator.generate_signals_batch.__doc__ is not None

logger.info("Time Series Signal Generator module loaded successfully")
