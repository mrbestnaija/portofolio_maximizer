"""
Signal Router - Time Series as Default, LLM as Fallback
Line Count: ~250 lines

Routes signals from multiple sources with Time Series ensemble as PRIMARY
and LLM as FALLBACK/REDUNDANCY. Maintains backward compatibility while
enabling gradual migration to Time Series-first architecture.

Per refactoring plan:
- Time Series ensemble is DEFAULT signal generator
- LLM serves as fallback when Time Series fails or needs validation
- Feature flags enable gradual rollout
- Unified signal interface for downstream consumers
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import time

from models.time_series_signal_generator import TimeSeriesSignal, TimeSeriesSignalGenerator
from ai_llm.signal_generator import LLMSignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class SignalBundle:
    """Bundle of signals from multiple sources"""
    primary_signal: Optional[Dict[str, Any]] = None
    fallback_signal: Optional[Dict[str, Any]] = None
    all_signals: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    routing_timestamp: datetime = field(default_factory=datetime.now)


class SignalRouter:
    """
    Route signals with Time Series as DEFAULT, LLM as FALLBACK.
    
    Architecture:
    - PRIMARY: Time Series ensemble (SARIMAX, SAMOSSA, GARCH, MSSA-RL)
    - FALLBACK: LLM signals (when Time Series unavailable or needs validation)
    - REDUNDANCY: Both sources can run in parallel for validation
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 time_series_generator: Optional[TimeSeriesSignalGenerator] = None,
                 llm_generator: Optional[LLMSignalGenerator] = None):
        """
        Initialize signal router.
        
        Args:
            config: Configuration dict with feature flags
            time_series_generator: Time Series signal generator instance
            llm_generator: LLM signal generator instance (optional, for fallback)
        """
        self.config = config or {}
        
        # Feature flags (default: Time Series enabled, LLM as fallback)
        self.feature_flags = {
            'time_series_primary': self.config.get('time_series_primary', True),  # DEFAULT: True
            'llm_fallback': self.config.get('llm_fallback', True),  # DEFAULT: True
            'llm_redundancy': self.config.get('llm_redundancy', False),  # Run both for validation
            'enable_samossa': self.config.get('enable_samossa', True),
            'enable_sarimax': self.config.get('enable_sarimax', True),
            'enable_garch': self.config.get('enable_garch', True),
            'enable_mssa_rl': self.config.get('enable_mssa_rl', True),
        }

        # Time Series routing thresholds (optional, supplied via config)
        ts_cfg = self.config.get("time_series") or {}

        # Initialize generators
        self.ts_generator = time_series_generator or TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_cfg.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_cfg.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_cfg.get("max_risk_score", 0.7)),
            use_volatility_filter=bool(ts_cfg.get("use_volatility_filter", True)),
        )
        self.llm_generator = llm_generator  # Optional, only if LLM fallback enabled
        
        # Routing statistics
        self.routing_stats = {
            'time_series_signals': 0,
            'llm_fallback_signals': 0,
            'redundancy_signals': 0,
            'failed_routes': 0
        }
        self.latencies = {"ts_ms": [], "llm_ms": []}
        self.ticker_latencies: Dict[str, Dict[str, float]] = {}

        # Optional per-ticker TS disable flags sourced from
        # config/forecaster_monitoring.yml so chronic underperformers can be
        # routed away from the TS stack toward LLM or other sources.
        self._ts_disabled_tickers = self._load_ts_disabled_tickers()
        
        logger.info(
            f"Signal Router initialized: "
            f"TS_primary={self.feature_flags['time_series_primary']}, "
            f"LLM_fallback={self.feature_flags['llm_fallback']}, "
            f"LLM_redundancy={self.feature_flags['llm_redundancy']}"
        )
    
    def route_signal(self,
                    ticker: str,
                    forecast_bundle: Optional[Dict[str, Any]] = None,
                    current_price: float = 0.0,
                    market_data: Optional[Any] = None,
                    llm_signal: Optional[Dict[str, Any]] = None,
                    quality: Optional[Dict[str, Any]] = None,
                    data_source: Optional[str] = None,
                    mid_price: Optional[float] = None) -> SignalBundle:
        """
        Route signal from Time Series (primary) or LLM (fallback).
        
        Args:
            ticker: Stock ticker symbol
            forecast_bundle: Time Series forecast bundle (from TimeSeriesForecaster)
            current_price: Current market price
            market_data: Market data for context
            llm_signal: Optional pre-generated LLM signal (for redundancy mode)
            quality: Optional quality metrics dict
            data_source: Optional active data source name
            mid_price: Optional mid-price hint for downstream execution logging
            
        Returns:
            SignalBundle with primary and fallback signals
        """
        all_signals = []
        primary_signal = None
        fallback_signal = None
        quality_score = (quality or {}).get("quality_score")

        # If this ticker has been marked as TS-disabled in
        # config/forecaster_monitoring.yml, treat Time Series as unavailable
        # so routing flows directly to LLM fallback or redundancy paths.
        if ticker in self._ts_disabled_tickers:
            forecast_bundle = None
        
        # STEP 1: Try Time Series as PRIMARY (if enabled)
        if self.feature_flags['time_series_primary'] and forecast_bundle:
            try:
                ts_start = time.perf_counter()
                ts_signal = self.ts_generator.generate_signal(
                    forecast_bundle=forecast_bundle,
                    current_price=current_price,
                    ticker=ticker,
                    market_data=market_data
                )
                ts_elapsed = (time.perf_counter() - ts_start) * 1000.0
                self.latencies["ts_ms"].append(ts_elapsed)
                self.ticker_latencies.setdefault(ticker, {})["ts_ms"] = ts_elapsed
                
                # Convert TimeSeriesSignal to dict for compatibility
                primary_signal = self._signal_to_dict(ts_signal)
                primary_signal['source'] = 'TIME_SERIES'
                primary_signal['is_primary'] = True
                if mid_price is not None:
                    primary_signal['mid_price_hint'] = mid_price
                if quality_score is not None:
                    primary_signal['quality_score'] = quality_score
                if data_source:
                    primary_signal['data_source'] = data_source
                
                all_signals.append(primary_signal)
                self.routing_stats['time_series_signals'] += 1
                
                logger.debug(f"Time Series signal generated for {ticker}: {primary_signal['action']}")
                
            except Exception as e:
                logger.warning(f"Time Series signal generation failed for {ticker}: {e}")
                self.routing_stats['failed_routes'] += 1
        
        # STEP 2: Use LLM as FALLBACK (if Time Series failed or not available)
        quality_too_low = quality_score is not None and quality_score < 0.6
        if (not primary_signal or primary_signal.get('action') == 'HOLD' or quality_too_low) and \
           self.feature_flags['llm_fallback'] and self.llm_generator:
            try:
                # Generate LLM signal if not provided
                if not llm_signal:
                    llm_start = time.perf_counter()
                    llm_signal = self.llm_generator.generate_signal(
                        data=market_data if hasattr(market_data, 'columns') else None,
                        ticker=ticker,
                        market_analysis={
                            "quality_score": quality_score,
                            "data_source": data_source,
                        }
                    )
                    llm_elapsed = (time.perf_counter() - llm_start) * 1000.0
                    self.latencies["llm_ms"].append(llm_elapsed)
                    self.ticker_latencies.setdefault(ticker, {})["llm_ms"] = llm_elapsed
                
                if llm_signal:
                    fallback_signal = llm_signal.copy()
                    fallback_signal['source'] = 'LLM'
                    fallback_signal['is_primary'] = False
                    fallback_signal['is_fallback'] = True
                    if quality_score is not None:
                        fallback_signal['quality_score'] = quality_score
                    if data_source:
                        fallback_signal['data_source'] = data_source
                    
                    all_signals.append(fallback_signal)
                    self.routing_stats['llm_fallback_signals'] += 1
                    
                    logger.debug(f"LLM fallback signal generated for {ticker}: {fallback_signal.get('action')}")
                
            except Exception as e:
                logger.warning(f"LLM fallback signal generation failed for {ticker}: {e}")
        
        # STEP 3: Redundancy mode - run both for validation (if enabled)
        if self.feature_flags['llm_redundancy'] and \
           primary_signal and self.llm_generator and not fallback_signal:
            try:
                if not llm_signal:
                    llm_signal = self.llm_generator.generate_signal(
                        data=market_data if hasattr(market_data, 'columns') else None,
                        ticker=ticker,
                        market_analysis={}
                    )
                
                if llm_signal:
                    redundancy_signal = llm_signal.copy()
                    redundancy_signal['source'] = 'LLM'
                    redundancy_signal['is_primary'] = False
                    redundancy_signal['is_redundancy'] = True
                    
                    all_signals.append(redundancy_signal)
                    self.routing_stats['redundancy_signals'] += 1
                    
                    logger.debug(f"LLM redundancy signal generated for {ticker}")
                
            except Exception as e:
                logger.debug(f"LLM redundancy signal generation skipped for {ticker}: {e}")
        
        # STEP 4: Determine final primary signal
        if not primary_signal and fallback_signal:
            # Use LLM as primary if Time Series unavailable
            primary_signal = fallback_signal
            primary_signal['is_primary'] = True
            primary_signal['is_fallback'] = True
            logger.info(f"Using LLM fallback as primary signal for {ticker}")
        
        # STEP 5: Create signal bundle
        bundle = SignalBundle(
            primary_signal=primary_signal,
            fallback_signal=fallback_signal if fallback_signal != primary_signal else None,
            all_signals=all_signals,
            metadata={
                'ticker': ticker,
                'routing_mode': self._get_routing_mode(),
                'feature_flags': self.feature_flags.copy(),
                'stats': self.routing_stats.copy(),
                'timestamp': datetime.now().isoformat()
            },
            routing_timestamp=datetime.now()
        )
        
        return bundle
    
    def route_signals_batch(self,
                           tickers: List[str],
                           forecast_bundles: Dict[str, Dict[str, Any]],
                           current_prices: Dict[str, float],
                           market_data: Optional[Dict[str, Any]] = None,
                           llm_signals: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, SignalBundle]:
        """
        Route signals for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            forecast_bundles: Dict mapping ticker to forecast bundle
            current_prices: Dict mapping ticker to current price
            market_data: Optional dict mapping ticker to market data
            llm_signals: Optional pre-generated LLM signals
            
        Returns:
            Dict mapping ticker to SignalBundle
        """
        bundles = {}
        
        for ticker in tickers:
            forecast_bundle = forecast_bundles.get(ticker)
            current_price = current_prices.get(ticker, 0.0)
            ticker_market_data = market_data.get(ticker) if market_data else None
            llm_signal = llm_signals.get(ticker) if llm_signals else None
            
            bundle = self.route_signal(
                ticker=ticker,
                forecast_bundle=forecast_bundle,
                current_price=current_price,
                market_data=ticker_market_data,
                llm_signal=llm_signal
            )
            
            bundles[ticker] = bundle
        
        return bundles
    
    def _signal_to_dict(self, signal: TimeSeriesSignal) -> Dict[str, Any]:
        """Convert TimeSeriesSignal to dict for compatibility"""
        return {
            'ticker': signal.ticker,
            'action': signal.action,
            'confidence': signal.confidence,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'signal_timestamp': signal.signal_timestamp.isoformat() if isinstance(signal.signal_timestamp, datetime) else signal.signal_timestamp,
            'model_type': signal.model_type,
            'forecast_horizon': signal.forecast_horizon,
            'expected_return': signal.expected_return,
            'risk_score': signal.risk_score,
            'reasoning': signal.reasoning,
            'provenance': signal.provenance,
            'signal_type': signal.signal_type,
            'volatility': signal.volatility,
            'lower_ci': signal.lower_ci,
            'upper_ci': signal.upper_ci
        }
    
    def _get_routing_mode(self) -> str:
        """Get current routing mode description"""
        if self.feature_flags['time_series_primary'] and self.feature_flags['llm_fallback']:
            if self.feature_flags['llm_redundancy']:
                return 'TIME_SERIES_PRIMARY_LLM_REDUNDANCY'
            else:
                return 'TIME_SERIES_PRIMARY_LLM_FALLBACK'
        elif self.feature_flags['time_series_primary']:
            return 'TIME_SERIES_ONLY'
        elif self.feature_flags['llm_fallback']:
            return 'LLM_ONLY'
        else:
            return 'UNKNOWN'
    
    def toggle_feature_flag(self, flag_name: str, enabled: bool):
        """Toggle feature flag for gradual rollout"""
        if flag_name in self.feature_flags:
            old_value = self.feature_flags[flag_name]
            self.feature_flags[flag_name] = enabled
            logger.info(f"Feature flag {flag_name} changed: {old_value} → {enabled}")
        else:
            logger.warning(f"Unknown feature flag: {flag_name}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            'stats': self.routing_stats.copy(),
            'feature_flags': self.feature_flags.copy(),
            'routing_mode': self._get_routing_mode(),
            'total_signals': sum(self.routing_stats.values())
        }
    
    def reset_stats(self):
        """Reset routing statistics"""
        self.routing_stats = {
            'time_series_signals': 0,
            'llm_fallback_signals': 0,
            'redundancy_signals': 0,
            'failed_routes': 0
        }
        logger.info("Routing statistics reset")

    @staticmethod
    def _load_ts_disabled_tickers() -> List[str]:
        """
        Load tickers explicitly marked as TS-disabled in
        config/forecaster_monitoring.yml (per_ticker.disable_time_series: true).

        This keeps routing aligned with the same health config used by brutal
        CLIs and hyperopt, without hard-coding symbols in the router.
        """
        disabled: List[str] = []
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "forecaster_monitoring.yml"
        if not cfg_path.exists():
            return disabled
        try:
            import yaml  # lazy import to avoid hard dependency in minimal setups

            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            fm_cfg = raw.get("forecaster_monitoring") or {}
            per_ticker = fm_cfg.get("per_ticker") or {}
            for symbol, cfg in per_ticker.items():
                if isinstance(cfg, dict) and cfg.get("disable_time_series"):
                    disabled.append(str(symbol))
        except Exception:  # pragma: no cover - routing must remain robust
            return []
        return disabled


# Validation
assert SignalRouter.route_signal.__doc__ is not None
assert SignalRouter.route_signals_batch.__doc__ is not None

logger.info("Signal Router module loaded successfully")

