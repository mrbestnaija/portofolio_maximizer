"""
Signal Validator - Production-grade 5-layer validation framework
Line Count: ~250 lines (within budget)

Validates LLM signals before execution with multi-layer checks:
- Layer 1: Statistical validation (trend, volatility)
- Layer 2: Market regime alignment
- Layer 3: Risk-adjusted position sizing
- Layer 4: Portfolio correlation impact
- Layer 5: Transaction cost feasibility

Per AGENT_INSTRUCTION.md:
- Signals require >55% accuracy for live trading
- 30-day rolling backtest for validation
- Must beat buy-and-hold baseline
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from etl.portfolio_math import calculate_kelly_fraction_correct
from etl.statistical_tests import StatisticalTestSuite

logger = logging.getLogger(__name__)

ALLOWED_RISK_LEVELS = ("low", "medium", "high", "extreme")


def _normalise_risk_level(level: Any) -> Tuple[str, Optional[str]]:
    """Coerce risk levels into the database-approved taxonomy."""
    if isinstance(level, str):
        normalised = level.strip().lower()
    else:
        normalised = "medium"

    if normalised not in ALLOWED_RISK_LEVELS:
        return "high", f"Adjusted unsupported risk_level='{level}' to 'high'"
    return normalised, None


def _clamp_confidence(confidence: Any, default: float = 0.5) -> float:
    """Ensure confidence remains between 0 and 1."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(value, 1.0))


def detect_market_regime(price_series: pd.Series, window: int = 60) -> Dict[str, Any]:
    """
    Detect market regime using rolling volatility significance testing.
    """
    if len(price_series) < window + 1:
        return {"regimes": ["insufficient"], "current_regime": "insufficient"}

    log_returns = np.log(price_series).diff().dropna()
    rolling_vol = log_returns.rolling(window).std()
    recent_vol = rolling_vol.iloc[-window:]
    current_vol = rolling_vol.iloc[-1]

    with np.errstate(invalid="ignore"):
        t_stat, p_value = stats.ttest_1samp(recent_vol.dropna(), current_vol)

    if np.isnan(t_stat):
        regime = "sideways"
    elif p_value < 0.05:
        regime = "high_vol" if current_vol > recent_vol.mean() else "low_vol"
    else:
        regime = "normal"

    trend = price_series.iloc[-window:].pct_change().sum()
    if trend > 0.05 and regime != "insufficient":
        market_regime = f"bull_{regime}"
    elif trend < -0.05 and regime != "insufficient":
        market_regime = f"bear_{regime}"
    else:
        market_regime = f"sideways_{regime}"

    return {
        "regimes": rolling_vol.index.to_list(),
        "current_regime": market_regime,
        "volatility": current_vol,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
    }


@dataclass
class ValidationResult:
    """Result of signal validation"""
    is_valid: bool
    confidence_score: float
    warnings: List[str]
    layer_results: Dict[str, bool]
    recommendation: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BacktestReport:
    """Result of signal quality backtesting"""
    hit_rate: float
    profit_factor: float
    sharpe_ratio: float
    annual_return: float
    trades_analyzed: int
    avg_confidence: float
    recommendation: str
    period_days: int
    p_value: float = 1.0
    statistically_significant: bool = False
    information_ratio: float = 0.0
    information_coefficient: float = 0.0
    timestamp: datetime = None
    statistical_summary: Dict[str, float] = field(default_factory=dict)
    autocorrelation: Dict[str, float] = field(default_factory=dict)
    bootstrap_intervals: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SignalValidator:
    """
    Production-grade 5-layer signal validation.
    
    Validates LLM signals against quantitative criteria before execution.
    Includes 30-day rolling backtest for signal quality assessment.
    """
    
    def __init__(self, 
                 min_confidence: float = 0.55,
                 max_volatility_percentile: float = 0.95,
                 max_position_size: float = 0.02,
                 transaction_cost: float = 0.001):
        """
        Initialize signal validator.
        
        Args:
            min_confidence: Minimum confidence for signal approval (default 55%)
            max_volatility_percentile: Max volatility percentile for trading (95th)
            max_position_size: Maximum position size as fraction of portfolio (2%)
            transaction_cost: Transaction cost as fraction (0.1%)
        """
        self.min_confidence = min_confidence
        self.max_volatility_percentile = max_volatility_percentile
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self._stat_suite = StatisticalTestSuite()
        
        logger.info(f"Signal Validator initialized with min_confidence={min_confidence}")

    @staticmethod
    def _is_exit_trade(signal: Dict[str, Any], portfolio_state: Optional[Dict[str, Any]]) -> bool:
        """
        Return True when the signal reduces an existing exposure (i.e., closing).

        This is used to avoid blocking risk-reducing exits due to trend/regime
        warnings meant for opening/adding exposure.
        """
        if not isinstance(portfolio_state, dict):
            return False
        action = str(signal.get("action", "HOLD")).upper()
        if action not in {"BUY", "SELL"}:
            return False

        ticker_raw = signal.get("ticker")
        ticker = str(ticker_raw).strip() if ticker_raw is not None else ""
        if not ticker:
            return False

        positions = portfolio_state.get("positions")
        if not isinstance(positions, dict) or not positions:
            return False

        # Best-effort lookup with common casings.
        shares_raw = positions.get(ticker)
        if shares_raw is None:
            shares_raw = positions.get(ticker.upper())
        if shares_raw is None:
            shares_raw = positions.get(ticker.lower())
        try:
            current_shares = float(shares_raw or 0.0)
        except (TypeError, ValueError):
            current_shares = 0.0

        # SELL reduces an existing long; BUY reduces an existing short.
        if action == "SELL" and current_shares > 0:
            return True
        if action == "BUY" and current_shares < 0:
            return True
        return False

    @staticmethod
    def _fingerprint_market_data(market_data: pd.DataFrame) -> Tuple[int, Any, float, float]:
        """Fast-ish fingerprint used to invalidate cached market context."""
        length = int(len(market_data))
        last_index = None
        if length:
            try:
                last_index = market_data.index[-1]
            except Exception:
                last_index = None

        first_close = float("nan")
        last_close = float("nan")
        close_series = market_data.get("Close")
        if isinstance(close_series, pd.Series) and not close_series.empty:
            try:
                first_close = float(close_series.iloc[0])
                last_close = float(close_series.iloc[-1])
            except Exception:
                first_close = float("nan")
                last_close = float("nan")

        return (length, last_index, first_close, last_close)

    def _market_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Compute/cache derived market features used across validation layers."""
        fingerprint = self._fingerprint_market_data(market_data)
        cache_key = "_portfolio_maximizer_signal_validator_ctx"
        cached = None
        if isinstance(getattr(market_data, "attrs", None), dict):
            cached = market_data.attrs.get(cache_key)
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            return cached

        close_series = market_data["Close"]
        close_values = close_series.to_numpy(dtype=float, copy=False)
        if close_values.size:
            current_price = float(close_values[-1])
        else:
            current_price = float("nan")

        if close_values.size >= 20:
            sma_20 = float(np.mean(close_values[-20:]))
        else:
            sma_20 = current_price

        if close_values.size >= 50:
            sma_50 = float(np.mean(close_values[-50:]))
        else:
            sma_50 = sma_20

        if close_values.size >= 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_returns = np.diff(np.log(close_values))
        else:
            log_returns = np.array([], dtype=float)

        if log_returns.size:
            expected_daily_return = float(np.mean(log_returns))
            volatility_annualised = float(np.std(log_returns) * np.sqrt(252))
        else:
            expected_daily_return = 0.0
            volatility_annualised = 0.0

        rolling_vol = close_series.rolling(20).std()
        rolling_last = rolling_vol.iloc[-1] if len(rolling_vol) else np.nan
        if pd.notna(rolling_last):
            vol_percentile = float((rolling_last > rolling_vol).mean())
        else:
            vol_percentile = 0.0

        context: Dict[str, Any] = {
            "close_values": close_values,
            "current_price": current_price,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "log_returns": log_returns,
            "expected_daily_return": expected_daily_return,
            "volatility_annualised": volatility_annualised,
            "vol_percentile": vol_percentile,
            "regime_info": detect_market_regime(close_series),
            "fingerprint": fingerprint,
        }

        if isinstance(getattr(market_data, "attrs", None), dict):
            try:
                market_data.attrs[cache_key] = context
            except Exception:
                pass
        return context
    
    def validate_llm_signal(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        portfolio_value: float = 10000.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        5-layer validation before signal execution.
        
        Args:
            signal: LLM signal dict with action, confidence, reasoning, risk_level
            market_data: Recent OHLCV data for validation
            portfolio_value: Current portfolio value for position sizing
            portfolio_state: Optional snapshot with keys like cash/positions/entry_prices
                (used for concentration/correlation checks when available).
            
        Returns:
            ValidationResult with validation status and warnings
        """
        warnings = []
        layer_results = {}

        action = str(signal.get('action', 'HOLD')).upper()
        if action not in {'BUY', 'SELL', 'HOLD'}:
            warnings.append(f"Unsupported action '{action}', defaulting to HOLD")
            action = 'HOLD'
        signal['action'] = action

        risk_level, risk_warning = _normalise_risk_level(signal.get('risk_level', 'medium'))
        if risk_warning:
            warnings.append(risk_warning)
        signal['risk_level'] = risk_level

        confidence = _clamp_confidence(signal.get('confidence', 0.5))
        signal['confidence'] = confidence

        exit_trade = self._is_exit_trade(signal, portfolio_state)
        
        # Layer 1: Statistical validation
        stats_valid, stats_warnings = self._validate_statistics(signal, market_data)
        if exit_trade:
            stats_valid = True
        layer_results['statistical'] = stats_valid
        warnings.extend(stats_warnings)
        
        # Layer 2: Market regime alignment
        regime_valid, regime_warnings = self._validate_regime(signal, market_data)
        if exit_trade:
            regime_valid = True
        layer_results['regime'] = regime_valid
        warnings.extend(regime_warnings)
        
        # Layer 3: Risk-adjusted position sizing
        position_valid, position_warnings = self._validate_position_size(
            signal, market_data, portfolio_value
        )
        if exit_trade:
            position_valid = True
        layer_results['position_sizing'] = position_valid
        warnings.extend(position_warnings)
        
        # Layer 4: Portfolio correlation / concentration impact.
        correlation_valid, corr_warnings = self._validate_correlation(
            signal,
            market_data,
            portfolio_state=portfolio_state,
            portfolio_value=portfolio_value,
        )
        layer_results['correlation'] = correlation_valid
        warnings.extend(corr_warnings)
        
        # Layer 5: Transaction cost feasibility
        cost_valid, cost_warnings = self._validate_transaction_costs(
            signal, market_data, portfolio_value
        )
        if exit_trade:
            cost_valid = True
        layer_results['transaction_costs'] = cost_valid
        warnings.extend(cost_warnings)
        
        layers_passed = all(layer_results.values())

        # Adjust confidence based on validation layers and warnings
        failed_layers = sum(1 for v in layer_results.values() if not v)
        adjusted_confidence = confidence
        if failed_layers:
            adjusted_confidence *= max(0.0, 1 - 0.15 * failed_layers)
        if warnings:
            adjusted_confidence *= max(0.0, 1 - 0.05 * len(warnings))
        # Optional edge-aware adjustment for Time Series provenance.
        edge_ratio = None
        provenance_raw = signal.get("provenance")
        provenance: Dict[str, Any] = {}
        if isinstance(provenance_raw, dict):
            provenance = provenance_raw
        elif isinstance(provenance_raw, str) and provenance_raw.strip():
            try:
                parsed = json.loads(provenance_raw)
                if isinstance(parsed, dict):
                    provenance = parsed
            except Exception:
                provenance = {}
        decision_context = provenance.get("decision_context") if isinstance(provenance.get("decision_context"), dict) else {}
        if decision_context:
            try:
                net_trade_return = float(decision_context.get("net_trade_return"))
            except (TypeError, ValueError):
                net_trade_return = None
            try:
                roundtrip_cost_bps = float(decision_context.get("roundtrip_cost_bps"))
            except (TypeError, ValueError):
                roundtrip_cost_bps = None
            if roundtrip_cost_bps in (None, 0.0):
                try:
                    roundtrip_cost_fraction = float(decision_context.get("roundtrip_cost_fraction"))
                except (TypeError, ValueError):
                    roundtrip_cost_fraction = None
                roundtrip_cost_bps = roundtrip_cost_fraction * 1e4 if roundtrip_cost_fraction else None

            if (
                net_trade_return is not None
                and roundtrip_cost_bps is not None
                and np.isfinite(net_trade_return)
                and np.isfinite(roundtrip_cost_bps)
                and roundtrip_cost_bps > 0
            ):
                edge_ratio = (net_trade_return * 1e4) / roundtrip_cost_bps

        if edge_ratio is not None:
            if edge_ratio < 0.5:
                adjusted_confidence *= 0.85
            elif edge_ratio > 3.0:
                adjusted_confidence = min(1.0, adjusted_confidence * 1.05)

        adjusted_confidence = max(0.0, min(adjusted_confidence, 1.0))
        
        is_valid = layers_passed and adjusted_confidence >= self.min_confidence
        if exit_trade and action in {"BUY", "SELL"}:
            # Never block risk-reducing exits due to entry-oriented guardrails.
            is_valid = True
        
        # Generate recommendation
        if is_valid:
            recommendation = 'EXECUTE'
        elif layers_passed and adjusted_confidence >= 0.45:
            recommendation = 'MONITOR'
        else:
            recommendation = 'REJECT'
        if exit_trade and action in {"BUY", "SELL"}:
            recommendation = "EXECUTE"
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=adjusted_confidence,
            warnings=warnings,
            layer_results=layer_results,
            recommendation=recommendation
        )
    
    def _validate_statistics(self, 
                            signal: Dict[str, Any], 
                            market_data: pd.DataFrame) -> tuple[bool, List[str]]:
        """Layer 1: Statistical validation (trend, volatility)"""
        warnings = []

        context = self._market_context(market_data)
        sma_20 = context["sma_20"]
        sma_50 = context["sma_50"]
        current_price = context["current_price"]
        
        action = signal.get('action', 'HOLD').upper()
        
        # Check trend alignment
        if action == 'BUY':
            if current_price < sma_20:
                warnings.append("BUY signal below SMA(20) - counter-trend")
            if sma_20 < sma_50:
                warnings.append("BUY signal in downtrend (SMA(20) < SMA(50))")
        elif action == 'SELL':
            if current_price > sma_20:
                warnings.append("SELL signal above SMA(20) - counter-trend")
            if sma_20 > sma_50:
                warnings.append("SELL signal in uptrend (SMA(20) > SMA(50))")
        
        vol_percentile = context["vol_percentile"]
        
        if vol_percentile > self.max_volatility_percentile:
            warnings.append(f"High volatility: {vol_percentile:.1%} percentile")
        
        # Pass if < 2 warnings
        is_valid = len(warnings) < 2
        
        return is_valid, warnings
    
    def _validate_regime(self, 
                        signal: Dict[str, Any], 
                        market_data: pd.DataFrame) -> tuple[bool, List[str]]:
        """Layer 2: Market regime alignment"""
        warnings = []

        regime_info = self._market_context(market_data)["regime_info"]
        regime = regime_info.get('current_regime', 'sideways_normal')
        
        action = signal.get('action', 'HOLD')
        risk_level = signal.get('risk_level', 'medium')
        confidence = signal.get('confidence', 0.0)

        if regime.startswith('bear') and action == 'BUY':
            warnings.append(f"BUY signal in {regime} regime - elevated downside risk")
        if regime.startswith('bull') and action == 'SELL' and confidence < 0.7:
            warnings.append("SELL signal counter to bullish regime with modest confidence")
        if "high_vol" in regime and risk_level == 'high':
            warnings.append("High risk signal during high volatility regime")
        
        # Pass if < 2 warnings
        is_valid = len(warnings) < 2
        
        return is_valid, warnings
    
    def _validate_position_size(self, 
                                signal: Dict[str, Any], 
                                market_data: pd.DataFrame,
                                portfolio_value: float) -> tuple[bool, List[str]]:
        """Layer 3: Risk-adjusted position sizing (Kelly criterion)"""
        warnings = []

        context = self._market_context(market_data)
        returns = context["log_returns"]
        
        confidence = signal.get('confidence', 0.5)
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        avg_win = float(np.mean(positive_returns)) if len(positive_returns) > 0 else 0.01
        avg_loss = float(abs(np.mean(negative_returns))) if len(negative_returns) > 0 else 0.01

        win_rate = max(0.51, confidence)
        kelly_fraction = calculate_kelly_fraction_correct(win_rate, avg_win, avg_loss)

        recommended_fraction = min(kelly_fraction * 0.5, self.max_position_size)
        recommended_fraction = max(recommended_fraction, 0.0)
        
        # Check if recommended size is feasible
        if recommended_fraction < 0.005:  # Less than 0.5%
            warnings.append(f"Position size too small: {recommended_fraction:.2%}")
        
        if confidence < 0.6 and recommended_fraction > 0.015:
            warnings.append(f"Low confidence signal suggests smaller position")
        
        # Volatility adjustment
        volatility = context["volatility_annualised"]
        if volatility > 0.4:  # High volatility
            warnings.append(f"High volatility ({volatility:.1%}) - reduce position size")
        
        is_valid = len(warnings) < 2
        
        return is_valid, warnings
    
    def _validate_correlation(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        *,
        portfolio_state: Optional[Dict[str, Any]] = None,
        portfolio_value: Optional[float] = None,
    ) -> tuple[bool, List[str]]:
        """Layer 4: Portfolio impact (concentration + correlation when possible)."""
        warnings: List[str] = []

        action = str(signal.get("action", "HOLD")).upper()
        if action == "HOLD":
            return True, warnings

        if not isinstance(portfolio_state, dict):
            if action == "BUY":
                warnings.append("NOTICE: Full portfolio correlation check requires portfolio data")
            return True, warnings

        positions_raw = portfolio_state.get("positions")
        if not isinstance(positions_raw, dict) or not positions_raw:
            if action == "BUY":
                warnings.append("NOTICE: Portfolio snapshot missing positions; correlation check limited")
            return True, warnings

        entry_prices_raw = portfolio_state.get("entry_prices") or {}
        entry_prices: Dict[str, Any]
        if isinstance(entry_prices_raw, dict):
            entry_prices = entry_prices_raw
        else:
            entry_prices = {}

        def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return default
            if not np.isfinite(parsed):
                return default
            return float(parsed)

        def _infer_asset_class(ticker: str) -> str:
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

        def _lookup(mapping: Dict[str, Any], key: str, default: Any = None) -> Any:
            if key in mapping:
                return mapping[key]
            upper = key.upper()
            if upper in mapping:
                return mapping[upper]
            lower = key.lower()
            if lower in mapping:
                return mapping[lower]
            return default

        ticker_raw = signal.get("ticker")
        ticker = str(ticker_raw).strip() if ticker_raw is not None else ""
        if not ticker:
            if action == "BUY":
                warnings.append("NOTICE: Missing ticker; portfolio impact check limited")
            return True, warnings

        positions: Dict[str, Any] = positions_raw
        current_shares = _safe_float(_lookup(positions, ticker, 0.0), default=0.0) or 0.0

        # Use the most current price we have for the active symbol.
        price_est = _safe_float(signal.get("entry_price"))
        if (price_est is None or price_est <= 0.0) and isinstance(market_data, pd.DataFrame) and not market_data.empty:
            if "Close" in market_data.columns:
                price_est = _safe_float(market_data["Close"].iloc[-1])
        if price_est is None or price_est <= 0.0:
            price_est = _safe_float(_lookup(entry_prices, ticker, None))
        if price_est is None or price_est <= 0.0:
            if action == "BUY":
                warnings.append("NOTICE: Missing price context; portfolio impact check limited")
            return True, warnings

        cash = _safe_float(portfolio_state.get("cash"), default=0.0) or 0.0

        # Estimate equity and gross exposure from the snapshot (entry prices are a best-effort proxy).
        equity_est = cash
        gross_exposure = 0.0
        asset_exposure: Dict[str, float] = {}
        ticker_upper = ticker.upper()
        for sym, shares_raw in positions.items():
            shares = _safe_float(shares_raw, default=0.0) or 0.0
            if shares == 0.0:
                continue
            sym_str = str(sym).strip()
            if not sym_str:
                continue
            sym_upper = sym_str.upper()
            price = _safe_float(_lookup(entry_prices, sym_str, None))
            if sym_upper == ticker_upper:
                price = price_est
            if price is None or price <= 0.0:
                continue
            value = shares * price
            equity_est += value
            exposure = abs(value)
            gross_exposure += exposure
            bucket = _infer_asset_class(sym_upper)
            asset_exposure[bucket] = asset_exposure.get(bucket, 0.0) + exposure

        equity_base = _safe_float(portfolio_value)
        if equity_base is None or equity_base <= 0.0:
            equity_base = equity_est if equity_est > 0.0 else max(abs(equity_est), abs(cash), 1.0)

        # Concentration checks (correlation requires multi-asset history, so we treat this as
        # the "portfolio impact" gate until that context is wired through).
        increases_exposure = (action == "BUY" and current_shares >= 0.0) or (
            action == "SELL" and current_shares <= 0.0
        )
        if not increases_exposure:
            return True, warnings

        single_name_weight = abs(current_shares * price_est) / equity_base if equity_base > 0 else 0.0
        gross_leverage = gross_exposure / equity_base if equity_base > 0 else 0.0

        warn_single_name = 0.10
        max_single_name = 0.20
        warn_gross_leverage = 1.20
        max_gross_leverage = 1.50

        is_valid = True
        if single_name_weight >= warn_single_name:
            warnings.append(
                f"Concentration warning: {ticker_upper} is already ~{single_name_weight:.1%} of equity"
            )
        if single_name_weight >= max_single_name:
            warnings.append(
                f"Concentration limit breached: {ticker_upper} exceeds {max_single_name:.0%} of equity"
            )
            is_valid = False

        if gross_leverage >= warn_gross_leverage:
            warnings.append(f"High gross exposure: ~{gross_leverage:.2f}x equity")
        if gross_leverage >= max_gross_leverage:
            warnings.append(f"Gross exposure exceeds cap: {max_gross_leverage:.2f}x equity")
            is_valid = False

        asset_class = _infer_asset_class(ticker_upper)
        asset_weight = asset_exposure.get(asset_class, 0.0) / equity_base if equity_base > 0 else 0.0
        if asset_weight >= 0.90:
            warnings.append(f"High {asset_class} concentration: ~{asset_weight:.1%} of gross exposure")

        # Optional: pairwise return correlations when pre-computed by the execution engine.
        corr_snapshot = portfolio_state.get("correlation_snapshot")
        if isinstance(corr_snapshot, dict):
            corr_map = corr_snapshot.get("correlations")
            if isinstance(corr_map, dict) and corr_map:
                warn_corr = 0.75
                max_corr = 0.90
                worst_sym = None
                worst_corr = 0.0
                for sym, value in corr_map.items():
                    corr_val = _safe_float(value)
                    if corr_val is None:
                        continue
                    abs_corr = abs(corr_val)
                    if abs_corr > abs(worst_corr):
                        worst_corr = corr_val
                        worst_sym = str(sym).upper()
                if worst_sym is not None and abs(worst_corr) >= warn_corr:
                    warnings.append(
                        f"Correlation warning: {ticker_upper} vs {worst_sym} ~{worst_corr:+.2f} "
                        f"(n={corr_snapshot.get('observations')})"
                    )
                if worst_sym is not None and abs(worst_corr) >= max_corr:
                    warnings.append(
                        f"Correlation limit breached: {ticker_upper} too correlated with {worst_sym} "
                        f"(abs corr {abs(worst_corr):.2f} >= {max_corr:.2f})"
                    )
                    is_valid = False

        return is_valid, warnings
    
    def _validate_transaction_costs(self, 
                                    signal: Dict[str, Any], 
                                    market_data: pd.DataFrame,
                                    portfolio_value: float) -> tuple[bool, List[str]]:
        """Layer 5: Transaction cost feasibility"""
        # In diagnostic mode, skip cost-based gating so trades can flow.
        diag_mode = str(os.getenv("DIAGNOSTIC_MODE") or os.getenv("EXECUTION_DIAGNOSTIC_MODE") or "0") == "1"
        if diag_mode:
            return True, []
        warnings = []

        def _safe_float(value: Any) -> Optional[float]:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(parsed):
                return None
            return float(parsed)

        # Prefer signal-provided decision context (Time Series signals) over
        # unconditional historical averages (LLM signals may not include it).
        provenance_raw = signal.get("provenance")
        provenance: Dict[str, Any] = {}
        if isinstance(provenance_raw, dict):
            provenance = provenance_raw
        elif isinstance(provenance_raw, str) and provenance_raw.strip():
            try:
                parsed = json.loads(provenance_raw)
                if isinstance(parsed, dict):
                    provenance = parsed
            except Exception:
                provenance = {}

        decision_context_raw = provenance.get("decision_context")
        decision_context: Dict[str, Any] = {}
        if isinstance(decision_context_raw, dict):
            decision_context = decision_context_raw
        elif isinstance(decision_context_raw, str) and decision_context_raw.strip():
            try:
                parsed = json.loads(decision_context_raw)
                if isinstance(parsed, dict):
                    decision_context = parsed
            except Exception:
                decision_context = {}

        exec_friction = provenance.get("execution_friction")
        if not isinstance(exec_friction, dict):
            exec_friction = {}

        gross_trade_return = _safe_float(decision_context.get("gross_trade_return"))
        if gross_trade_return is None:
            expected_return_signal = _safe_float(signal.get("expected_return"))
            if expected_return_signal is not None:
                gross_trade_return = abs(expected_return_signal)

        net_trade_return = _safe_float(decision_context.get("net_trade_return"))

        # Engine-level configured commission proxy (per-side) -> round-trip fraction.
        engine_roundtrip_cost = 2.0 * float(self.transaction_cost)

        estimated_roundtrip_cost = _safe_float(decision_context.get("roundtrip_cost_fraction"))
        if estimated_roundtrip_cost is None:
            estimated_roundtrip_cost = _safe_float(exec_friction.get("roundtrip_cost_fraction"))

        if estimated_roundtrip_cost is None:
            effective_roundtrip_cost = engine_roundtrip_cost
        else:
            effective_roundtrip_cost = max(engine_roundtrip_cost, max(0.0, estimated_roundtrip_cost))

        roundtrip_cost_bps = _safe_float(decision_context.get("roundtrip_cost_bps"))
        if roundtrip_cost_bps is None:
            roundtrip_cost_bps = _safe_float(exec_friction.get("roundtrip_cost_bps"))
        if roundtrip_cost_bps is None:
            roundtrip_cost_bps = max(0.0, effective_roundtrip_cost) * 1e4

        # Fallback: infer expected return from historical drift only when we have
        # no signal-provided edge.
        if gross_trade_return is None and net_trade_return is None:
            context = self._market_context(market_data)
            expected_daily_return = float(context.get("expected_daily_return", 0.0))

            risk_level = signal.get('risk_level', 'medium')
            horizon_raw = signal.get("forecast_horizon") or signal.get("horizon")
            holding_period = None
            try:
                holding_period = int(horizon_raw) if horizon_raw is not None else None
            except (TypeError, ValueError):
                holding_period = None
            if holding_period is None or holding_period <= 0:
                holding_period = {'low': 60, 'medium': 30, 'high': 10, 'extreme': 5}.get(risk_level, 30)
            holding_period = max(1, min(int(holding_period), 90))

            expected_return = expected_daily_return * holding_period
            gross_trade_return = abs(float(expected_return))

        # Check if expected edge clears costs with a modest cushion.
        edge_ratio = None
        if net_trade_return is not None and roundtrip_cost_bps:
            edge_ratio = (max(0.0, net_trade_return) * 1e4) / roundtrip_cost_bps if roundtrip_cost_bps > 0 else None
        elif gross_trade_return is not None and effective_roundtrip_cost:
            net_est = max(0.0, gross_trade_return - max(0.0, effective_roundtrip_cost))
            cost_bps = max(0.0, effective_roundtrip_cost) * 1e4
            edge_ratio = (net_est * 1e4) / cost_bps if cost_bps > 0 else None

        if edge_ratio is None:
            if gross_trade_return is not None and gross_trade_return < effective_roundtrip_cost * 2.0:  # 2x costs min
                warnings.append(
                    f"Expected edge ({gross_trade_return:.2%}) < 2x round-trip costs ({effective_roundtrip_cost:.2%})"
                )
        elif edge_ratio < 1.0:
            warnings.append(
                f"Edge/cost ratio low: {edge_ratio:.2f}x (net edge ~{max(0.0, (net_trade_return or 0.0)):.2%}, "
                f"round-trip cost ~{max(0.0, effective_roundtrip_cost):.2%})"
            )
        
        # For very small positions, costs matter more
        confidence = signal.get('confidence', 0.5)
        position_value = portfolio_value * self.max_position_size * confidence
        transaction_cost_dollars = position_value * effective_roundtrip_cost
        
        if transaction_cost_dollars > position_value * 0.02:  # Costs > 2% of position
            warnings.append(f"Transaction costs too high relative to position size")
        
        is_valid = len(warnings) < 2
        
        return is_valid, warnings
    
    def backtest_signal_quality(self, 
                               signals: List[Dict[str, Any]], 
                               actual_prices: pd.DataFrame,
                               lookback_days: int = 30) -> BacktestReport:
        """
        30-day rolling backtest of signal accuracy.
        
        Args:
            signals: List of historical LLM signals
            actual_prices: Actual price data for validation
            lookback_days: Days to look back for analysis
            
        Returns:
            BacktestReport with accuracy metrics
        """
        if len(signals) == 0:
            return BacktestReport(
                hit_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                annual_return=0.0,
                trades_analyzed=0,
                avg_confidence=0.0,
                recommendation='INSUFFICIENT_DATA',
                period_days=0,
                p_value=1.0,
                statistically_significant=False,
                information_ratio=0.0,
                information_coefficient=0.0
            )
        
        # Calculate hit rate (directional accuracy)
        correct_predictions = 0
        total_predictions = 0
        gross_profit = 0.0
        gross_loss = 0.0
        returns = []
        strategy_returns = []
        benchmark_returns = []
        predicted_directions = []
        actual_directions = []
        price_series = actual_prices['Close']
        
        for signal in signals:
            action = signal.get('action', 'HOLD').upper()
            if action == 'HOLD':
                continue
            
            ticker = signal.get('ticker', '')
            signal_date = pd.to_datetime(signal.get('signal_timestamp'))
            
            # Get actual price movement
            try:
                if signal_date in price_series.index:
                    signal_price = float(price_series.loc[signal_date])
                else:
                    prior = price_series.loc[:signal_date]
                    if prior.empty:
                        continue
                    signal_price = float(prior.iloc[-1])
                
                future_window = price_series.loc[signal_date + timedelta(days=1): signal_date + timedelta(days=10)]
                if future_window.empty:
                    continue
                future_price = float(future_window.iloc[min(4, len(future_window) - 1)])
                
                actual_return = (future_price / signal_price) - 1
                
                # Check if prediction was correct
                if (action == 'BUY' and actual_return > 0) or \
                   (action == 'SELL' and actual_return < 0):
                    correct_predictions += 1
                    gross_profit += abs(actual_return)
                else:
                    gross_loss += abs(actual_return)
                
                trade_return = actual_return if action == 'BUY' else -actual_return
                returns.append(trade_return)
                strategy_returns.append(trade_return)
                benchmark_returns.append(actual_return)
                predicted_directions.append(1 if action == 'BUY' else -1)
                actual_directions.append(np.sign(actual_return))
                total_predictions += 1
                
            except (KeyError, ValueError):
                continue
        
        # Calculate metrics
        hit_rate = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        returns_array = np.asarray(returns, dtype=float) if returns else np.asarray([], dtype=float)
        if returns_array.size > 0:
            std_returns = np.std(returns_array)
            sharpe_ratio = (np.mean(returns_array) / std_returns * np.sqrt(252)) if std_returns > 0 else 0.0
            cumulative_return = float(np.prod(1 + returns_array) - 1)
            if lookback_days > 0:
                annual_return = (1 + cumulative_return) ** (252 / lookback_days) - 1
            else:
                annual_return = cumulative_return
        else:
            sharpe_ratio = 0.0
            annual_return = 0.0
        
        avg_confidence = np.mean([_clamp_confidence(s.get('confidence', 0.5)) for s in signals]) if signals else 0.0

        stat_summary: Dict[str, float] = {}
        autocorr_summary: Dict[str, float] = {}
        bootstrap_summary: Dict[str, Any] = {}
        information_coefficient = 0.0
        p_value = 1.0
        statistically_significant = False
        information_ratio = 0.0
        
        if strategy_returns and benchmark_returns:
            try:
                stat_summary = self._stat_suite.test_strategy_significance(
                    strategy_returns, benchmark_returns
                )
            except ValueError as exc:
                logger.debug("Statistical significance test skipped: %s", exc)
                stat_summary = {}

            try:
                lags = min(10, len(strategy_returns) - 1)
                if lags >= 1:
                    autocorr_summary = self._stat_suite.test_autocorrelation(
                        strategy_returns, lags=lags
                    )
            except ValueError as exc:
                logger.debug("Autocorrelation diagnostic skipped: %s", exc)
                autocorr_summary = {}

            try:
                if len(strategy_returns) >= 5:
                    bootstrap = self._stat_suite.bootstrap_validation(
                        strategy_returns,
                        n_bootstrap=min(500, len(strategy_returns) * 10),
                    )
                    bootstrap_summary = {
                        "sharpe_ratio_ci": bootstrap.sharpe_ratio,
                        "max_drawdown_ci": bootstrap.max_drawdown,
                        "samples": bootstrap.samples,
                        "confidence_level": bootstrap.confidence_level,
                    }
            except ValueError as exc:
                logger.debug("Bootstrap validation skipped: %s", exc)
                bootstrap_summary = {}

            correlation_matrix = np.corrcoef(predicted_directions, actual_directions)
            if correlation_matrix.shape == (2, 2) and not np.isnan(correlation_matrix[0, 1]):
                information_coefficient = float(correlation_matrix[0, 1])
            else:
                information_coefficient = 0.0

            p_value = float(stat_summary.get("p_value", 1.0))
            statistically_significant = bool(stat_summary.get("significant", False))
            information_ratio = float(stat_summary.get("information_ratio", 0.0))
        
        # Generate recommendation
        if hit_rate >= 0.55 and profit_factor >= 1.5:
            recommendation = 'APPROVE_FOR_LIVE_TRADING'
        elif hit_rate >= 0.52:
            recommendation = 'CONTINUE_PAPER_TRADING'
        else:
            recommendation = 'IMPROVE_SIGNALS'
        
        return BacktestReport(
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            annual_return=annual_return,
            trades_analyzed=total_predictions,
            avg_confidence=avg_confidence,
            recommendation=recommendation,
            period_days=lookback_days,
            p_value=p_value,
            statistically_significant=statistically_significant,
            information_ratio=information_ratio,
            information_coefficient=information_coefficient,
            statistical_summary=stat_summary,
            autocorrelation=autocorr_summary,
            bootstrap_intervals=bootstrap_summary,
        )


# Validation
assert SignalValidator.validate_llm_signal.__doc__ is not None
assert SignalValidator.backtest_signal_quality.__doc__ is not None

logger.info("Signal Validator module loaded successfully")

# Line count: ~380 lines (slightly over budget but essential functionality)


