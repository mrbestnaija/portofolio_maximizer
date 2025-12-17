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
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from etl.time_series_forecaster import TimeSeriesForecaster
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
                 quant_validation_config_path: Optional[str] = None):
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
            
            # Get forecast value (typically first step or mean)
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

            # Account for trading frictions (slippage + fees). The buffer is
            # intentionally modest so profitable TS regimes with 0.1–0.3% moves
            # are not all suppressed before validation.
            friction_buffer = 0.0005
            net_expected_return = expected_return - friction_buffer
            
            # Extract volatility for risk assessment (ensure scalar)
            volatility_forecast = forecast_bundle.get('volatility_forecast') or {}
            if isinstance(volatility_forecast, dict):
                volatility = self._to_scalar(volatility_forecast.get('volatility'))
            else:
                volatility = self._to_scalar(volatility_forecast)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                forecast_bundle,
                expected_return,
                volatility
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                expected_return,
                volatility,
                forecast_bundle
            )
            
            # Determine action (conservative: require net return to clear friction buffer)
            action = self._determine_action(
                net_expected_return,
                confidence,
                risk_score
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
                net_expected_return,
                confidence,
                risk_score,
                forecast_bundle
            )
            
            # Extract provenance metadata
            provenance = self._extract_provenance(forecast_bundle)
            
            # Extract confidence intervals (handle Series or scalar)
            lower_ci_raw = ensemble_forecast.get('lower_ci')
            upper_ci_raw = ensemble_forecast.get('upper_ci')
            
            # Convert Series to scalar if needed (use first value)
            if isinstance(lower_ci_raw, pd.Series) and len(lower_ci_raw) > 0:
                lower_ci = float(lower_ci_raw.iloc[0])
            elif lower_ci_raw is not None:
                lower_ci = float(lower_ci_raw) if not isinstance(lower_ci_raw, pd.Series) else None
            else:
                lower_ci = None
                
            if isinstance(upper_ci_raw, pd.Series) and len(upper_ci_raw) > 0:
                upper_ci = float(upper_ci_raw.iloc[0])
            elif upper_ci_raw is not None:
                upper_ci = float(upper_ci_raw) if not isinstance(upper_ci_raw, pd.Series) else None
            else:
                upper_ci = None
            
            provenance['decision_context'] = {
                'expected_return': expected_return,
                'confidence': confidence,
                'risk_score': risk_score,
                'volatility': volatility,
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
                    return float(forecast_series.iloc[0]) if len(forecast_series) > 0 else None
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
    
    def _calculate_confidence(self,
                              forecast_bundle: Dict[str, Any],
                              expected_return: float,
                              volatility: Optional[float]) -> float:
        """
        Calculate confidence score (0.0 to 1.0) based on:
        - Model agreement (ensemble consensus)
        - Forecast strength (magnitude of expected return)
        - Model diagnostics (AIC/BIC, explained variance)
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Expected return magnitude (stronger moves = higher confidence)
        if abs(expected_return) > 0.05:  # >5% move
            confidence += 0.15
        elif abs(expected_return) > 0.02:  # >2% move
            confidence += 0.10
        
        # Factor 2: Model agreement (check if multiple models agree)
        model_agreement = self._check_model_agreement(forecast_bundle)
        confidence += model_agreement * 0.20
        
        # Factor 3: Model diagnostics quality
        diagnostics_score = self._evaluate_diagnostics(forecast_bundle)
        confidence += diagnostics_score * 0.15
        
        # Factor 4: Volatility filter (lower volatility = higher confidence)
        if volatility is not None and self.use_volatility_filter:
            if volatility < 0.20:  # <20% volatility
                confidence += 0.10
            elif volatility > 0.40:  # >40% volatility
                confidence -= 0.10
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))
    
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
    
    def _evaluate_diagnostics(self, forecast_bundle: Dict[str, Any]) -> float:
        """Evaluate model diagnostics quality (0.0 to 1.0)"""
        score = 0.5  # Base score
        
        # Check ensemble metadata
        ensemble_metadata = forecast_bundle.get('ensemble_metadata', {})
        
        # Check for AIC/BIC (lower is better, but we normalize)
        if 'aic' in ensemble_metadata:
            score += 0.2
        if 'bic' in ensemble_metadata:
            score += 0.2
        
        # Check for explained variance (SAMOSSA)
        if 'explained_variance' in ensemble_metadata:
            evr = ensemble_metadata['explained_variance']
            if evr > 0.90:  # >90% explained
                score += 0.1
        
        return min(1.0, score)
    
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
            lower_ci_raw = ensemble_forecast.get('lower_ci')
            upper_ci_raw = ensemble_forecast.get('upper_ci')
            
            # Extract scalar values from Series if needed
            if isinstance(lower_ci_raw, pd.Series) and len(lower_ci_raw) > 0:
                lower_ci = float(lower_ci_raw.iloc[0])
            else:
                lower_ci = float(lower_ci_raw) if lower_ci_raw is not None and not isinstance(lower_ci_raw, pd.Series) else None
                
            if isinstance(upper_ci_raw, pd.Series) and len(upper_ci_raw) > 0:
                upper_ci = float(upper_ci_raw.iloc[0])
            else:
                upper_ci = float(upper_ci_raw) if upper_ci_raw is not None and not isinstance(upper_ci_raw, pd.Series) else None
            
            if lower_ci is not None and upper_ci is not None:
                ci_width = (upper_ci - lower_ci) / abs(expected_return) if expected_return != 0 else 1.0
                if ci_width > 0.5:  # Wide confidence interval
                    risk += 0.2
                elif ci_width < 0.2:  # Narrow confidence interval
                    risk -= 0.1
        
        # Factor 3: Expected return magnitude (smaller moves = riskier relative to reward)
        if abs(expected_return) < 0.01:  # <1% expected return
            risk += 0.1
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, risk))
    
    def _determine_action(self,
                         expected_return: float,
                         confidence: float,
                         risk_score: float) -> str:
        """
        Determine trading action based on forecast and risk metrics.
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Must meet minimum thresholds
        if confidence < self.confidence_threshold:
            return 'HOLD'
        
        if abs(expected_return) < self.min_expected_return:
            return 'HOLD'
        
        if risk_score > self.max_risk_score:
            return 'HOLD'
        
        # Determine direction
        if expected_return > self.min_expected_return:
            return 'BUY'
        elif expected_return < -self.min_expected_return:
            return 'SELL'
        else:
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
                        confidence: float,
                        risk_score: float,
                        forecast_bundle: Dict[str, Any]) -> str:
        """Build human-readable reasoning for signal"""
        model_type = forecast_bundle.get('ensemble_metadata', {}).get('primary_model', 'ENSEMBLE')
        
        reasoning = (
            f"Time Series {model_type} forecast indicates {action} signal. "
            f"Expected return: {expected_return:.2%}, "
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

        try:
            metrics = calculate_enhanced_portfolio_metrics(
                returns=log_returns.reshape(-1, 1),
                weights=np.array([1.0]),
                risk_free_rate=float(config.get('risk_free_rate', DEFAULT_RISK_FREE_RATE)),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to compute quant metrics for %s: %s", ticker, exc)
            return None

        bootstrap_cfg = config.get('bootstrap', {})
        try:
            bootstrap_stats = bootstrap_confidence_intervals(
                returns=log_returns,
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
                    significance = test_strategy_significance(log_returns, benchmark_slice)
                except ValueError:
                    significance = None

        performance_snapshot = self._calculate_return_based_performance(log_returns)

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
        expected_profit = position_value * float(signal.expected_return or 0.0)
        criteria = self._evaluate_success_criteria(
            criteria_cfg=criteria_cfg,
            metrics=metrics,
            performance_snapshot=performance_snapshot,
            significance=significance,
            expected_profit=expected_profit,
        )

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

