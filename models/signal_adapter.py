"""
Signal Adapter - Unified Signal Interface for Backward Compatibility
Line Count: ~200 lines

Provides a unified interface for signals from different sources (Time Series, LLM)
to ensure backward compatibility with existing downstream consumers.

This adapter:
- Normalizes signal schemas from different sources
- Provides compatibility layer for existing code
- Handles signal conversion and validation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from models.time_series_signal_generator import TimeSeriesSignal

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSignal:
    """Unified signal format for all sources"""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    signal_timestamp: datetime
    source: str  # 'TIME_SERIES', 'LLM', 'HYBRID'
    model_type: Optional[str] = None
    reasoning: str = ''

    # Optional fields (Time Series specific)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    volatility: Optional[float] = None

    # Optional fields (LLM specific)
    llm_model: Optional[str] = None
    fallback: bool = False

    # Metadata
    provenance: Dict[str, Any] = None
    signal_type: str = 'UNIFIED'

    def __post_init__(self):
        if self.provenance is None:
            self.provenance = {}


class SignalAdapter:
    """
    Adapter for converting signals from different sources to unified format.

    Ensures backward compatibility with existing consumers while supporting
    new Time Series signals.
    """

    @staticmethod
    def from_time_series_signal(ts_signal: TimeSeriesSignal) -> UnifiedSignal:
        """
        Convert TimeSeriesSignal to UnifiedSignal.

        Args:
            ts_signal: Time Series signal

        Returns:
            UnifiedSignal with normalized schema
        """
        return UnifiedSignal(
            ticker=ts_signal.ticker,
            action=ts_signal.action,
            confidence=ts_signal.confidence,
            entry_price=ts_signal.entry_price,
            signal_timestamp=ts_signal.signal_timestamp,
            source='TIME_SERIES',
            model_type=ts_signal.model_type,
            reasoning=ts_signal.reasoning,
            target_price=ts_signal.target_price,
            stop_loss=ts_signal.stop_loss,
            expected_return=ts_signal.expected_return,
            risk_score=ts_signal.risk_score,
            volatility=ts_signal.volatility,
            provenance=ts_signal.provenance.copy() if ts_signal.provenance else {},
            signal_type='TIME_SERIES'
        )

    @staticmethod
    def from_llm_signal(llm_signal: Dict[str, Any]) -> UnifiedSignal:
        """
        Convert LLM signal dict to UnifiedSignal.

        Args:
            llm_signal: LLM signal dictionary

        Returns:
            UnifiedSignal with normalized schema
        """
        # Parse timestamp
        signal_timestamp = llm_signal.get('signal_timestamp')
        if isinstance(signal_timestamp, str):
            try:
                signal_timestamp = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                signal_timestamp = datetime.now()
        elif not isinstance(signal_timestamp, datetime):
            signal_timestamp = datetime.now()

        # Extract entry price
        entry_price = llm_signal.get('entry_price', 0.0)
        if entry_price is None:
            entry_price = 0.0

        return UnifiedSignal(
            ticker=llm_signal.get('ticker', ''),
            action=llm_signal.get('action', 'HOLD'),
            confidence=llm_signal.get('confidence', 0.5),
            entry_price=float(entry_price),
            signal_timestamp=signal_timestamp,
            source='LLM',
            model_type=llm_signal.get('llm_model', 'unknown'),
            reasoning=llm_signal.get('reasoning', ''),
            llm_model=llm_signal.get('llm_model'),
            fallback=llm_signal.get('fallback', False),
            provenance={
                'llm_model': llm_signal.get('llm_model'),
                'fallback': llm_signal.get('fallback', False),
                'data_period': llm_signal.get('data_period'),
                'indicators': llm_signal.get('indicators', {})
            },
            signal_type='LLM'
        )

    @staticmethod
    def to_legacy_dict(unified_signal: UnifiedSignal) -> Dict[str, Any]:
        """
        Convert UnifiedSignal to legacy LLM signal format for backward compatibility.

        Args:
            unified_signal: Unified signal

        Returns:
            Dictionary in legacy LLM signal format
        """
        return {
            'ticker': unified_signal.ticker,
            'action': unified_signal.action,
            'confidence': unified_signal.confidence,
            'reasoning': unified_signal.reasoning,
            'signal_timestamp': unified_signal.signal_timestamp.isoformat() if isinstance(unified_signal.signal_timestamp, datetime) else str(unified_signal.signal_timestamp),
            'signal_type': unified_signal.action,  # Legacy compatibility
            'entry_price': unified_signal.entry_price,
            'llm_model': unified_signal.llm_model or unified_signal.model_type or 'unknown',
            'fallback': unified_signal.fallback,
            # Add Time Series fields if available
            'target_price': unified_signal.target_price,
            'stop_loss': unified_signal.stop_loss,
            'expected_return': unified_signal.expected_return,
            'risk_score': unified_signal.risk_score,
            'volatility': unified_signal.volatility,
            'source': unified_signal.source,
            'provenance': unified_signal.provenance
        }

    @staticmethod
    def normalize_signal(signal: Any) -> UnifiedSignal:
        """
        Normalize signal from any source to UnifiedSignal.

        Args:
            signal: Signal from any source (TimeSeriesSignal, dict, etc.)

        Returns:
            UnifiedSignal
        """
        if isinstance(signal, TimeSeriesSignal):
            return SignalAdapter.from_time_series_signal(signal)
        elif isinstance(signal, dict):
            # Try to detect source
            if signal.get('signal_type') == 'TIME_SERIES' or 'model_type' in signal:
                # Convert Time Series dict to UnifiedSignal
                return UnifiedSignal(
                    ticker=signal.get('ticker', ''),
                    action=signal.get('action', 'HOLD'),
                    confidence=signal.get('confidence', 0.5),
                    entry_price=signal.get('entry_price', 0.0),
                    signal_timestamp=datetime.fromisoformat(signal.get('signal_timestamp', datetime.now().isoformat()).replace('Z', '+00:00')) if isinstance(signal.get('signal_timestamp'), str) else signal.get('signal_timestamp', datetime.now()),
                    source='TIME_SERIES',
                    model_type=signal.get('model_type'),
                    reasoning=signal.get('reasoning', ''),
                    target_price=signal.get('target_price'),
                    stop_loss=signal.get('stop_loss'),
                    expected_return=signal.get('expected_return'),
                    risk_score=signal.get('risk_score'),
                    volatility=signal.get('volatility'),
                    provenance=signal.get('provenance', {}),
                    signal_type='TIME_SERIES'
                )
            else:
                # Assume LLM signal
                return SignalAdapter.from_llm_signal(signal)
        elif isinstance(signal, UnifiedSignal):
            return signal
        else:
            logger.warning(f"Unknown signal type: {type(signal)}, creating default HOLD signal")
            return UnifiedSignal(
                ticker='',
                action='HOLD',
                confidence=0.0,
                entry_price=0.0,
                signal_timestamp=datetime.now(),
                source='UNKNOWN',
                reasoning='Unknown signal type'
            )

    @staticmethod
    def validate_signal(signal: UnifiedSignal) -> tuple[bool, Optional[str]]:
        """
        Validate unified signal.

        Args:
            signal: Unified signal to validate

        Returns:
            (is_valid, error_message)
        """
        if not signal.ticker:
            return False, "Missing ticker"

        if signal.action not in ('BUY', 'SELL', 'HOLD'):
            return False, f"Invalid action: {signal.action}"

        if not (0.0 <= signal.confidence <= 1.0):
            return False, f"Confidence out of range: {signal.confidence}"

        if signal.entry_price <= 0:
            return False, f"Invalid entry price: {signal.entry_price}"

        return True, None


# Validation
assert SignalAdapter.from_time_series_signal.__doc__ is not None
assert SignalAdapter.from_llm_signal.__doc__ is not None
assert SignalAdapter.normalize_signal.__doc__ is not None

logger.info("Signal Adapter module loaded successfully")
