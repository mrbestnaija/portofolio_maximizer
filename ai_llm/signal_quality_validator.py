"""
Signal Quality Validator
Validates LLM-generated signals for accuracy and reliability
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """LLM-generated trading signal"""
    ticker: str
    direction: SignalDirection
    confidence: float
    reasoning: str
    timestamp: datetime
    price_at_signal: float
    expected_return: Optional[float] = None
    risk_estimate: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of signal validation"""
    is_valid: bool
    confidence_score: float
    warnings: List[str]
    quality_metrics: Dict[str, float]
    recommendation: str


class SignalQualityValidator:
    """
    Validates LLM-generated signals using multiple quality checks
    """
    
    def __init__(self):
        self.min_confidence_threshold = 0.6
        self.max_risk_threshold = 0.15
        self.min_expected_return = 0.02
        
    def validate_signal(self, signal: Signal, market_data: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive signal validation with 5-layer checks
        
        Args:
            signal: LLM-generated signal to validate
            market_data: Historical market data for context
            
        Returns:
            ValidationResult with quality assessment
        """
        warnings = []
        quality_metrics = {}
        
        # Layer 1: Basic signal validation
        basic_valid, basic_warnings = self._validate_basic_signal(signal)
        warnings.extend(basic_warnings)
        quality_metrics['basic_validation'] = 1.0 if basic_valid else 0.0
        
        # Layer 2: Market context validation
        context_valid, context_warnings = self._validate_market_context(signal, market_data)
        warnings.extend(context_warnings)
        quality_metrics['context_validation'] = 1.0 if context_valid else 0.0
        
        # Layer 3: Risk-return validation
        risk_valid, risk_warnings = self._validate_risk_return(signal)
        warnings.extend(risk_warnings)
        quality_metrics['risk_validation'] = 1.0 if risk_valid else 0.0
        
        # Layer 4: Technical validation
        technical_valid, technical_warnings = self._validate_technical_signals(signal, market_data)
        warnings.extend(technical_warnings)
        quality_metrics['technical_validation'] = 1.0 if technical_valid else 0.0
        
        # Layer 5: Confidence calibration
        confidence_valid, confidence_warnings = self._validate_confidence_calibration(signal, market_data)
        warnings.extend(confidence_warnings)
        quality_metrics['confidence_validation'] = 1.0 if confidence_valid else 0.0
        
        # Overall assessment
        all_valid = all([basic_valid, context_valid, risk_valid, technical_valid, confidence_valid])
        overall_confidence = np.mean(list(quality_metrics.values()))
        
        # Determine recommendation
        if all_valid and overall_confidence >= 0.8:
            recommendation = "STRONG_BUY" if signal.direction == SignalDirection.BUY else "STRONG_SELL"
        elif all_valid and overall_confidence >= 0.6:
            recommendation = "BUY" if signal.direction == SignalDirection.BUY else "SELL"
        elif overall_confidence >= 0.4:
            recommendation = "WEAK_BUY" if signal.direction == SignalDirection.BUY else "WEAK_SELL"
        else:
            recommendation = "HOLD"
        
        return ValidationResult(
            is_valid=all_valid,
            confidence_score=overall_confidence,
            warnings=warnings,
            quality_metrics=quality_metrics,
            recommendation=recommendation
        )
    
    def _validate_basic_signal(self, signal: Signal) -> Tuple[bool, List[str]]:
        """Layer 1: Basic signal structure validation"""
        warnings = []
        
        # Check confidence range
        if not 0.0 <= signal.confidence <= 1.0:
            warnings.append(f"Invalid confidence score: {signal.confidence}")
            return False, warnings
        
        # Check minimum confidence
        if signal.confidence < self.min_confidence_threshold:
            warnings.append(f"Low confidence: {signal.confidence:.2f} < {self.min_confidence_threshold}")
            return False, warnings
        
        # Check reasoning quality
        if len(signal.reasoning.strip()) < 60:
            warnings.append("Insufficient reasoning provided")
            return False, warnings
        
        # Check price validity
        if signal.price_at_signal <= 0:
            warnings.append("Invalid price at signal")
            return False, warnings
        
        return True, warnings
    
    def _validate_market_context(self, signal: Signal, market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Layer 2: Market context validation"""
        warnings = []
        
        if market_data.empty:
            warnings.append("No market data available for context validation")
            return False, warnings
        
        # Check if signal is for recent data relative to signal timestamp
        latest_date = market_data.index.max()
        reference_time = signal.timestamp or datetime.now()
        if isinstance(reference_time, pd.Timestamp):
            reference_time = reference_time.to_pydatetime()
        days_old = (reference_time - latest_date).days
        if days_old < 0:
            days_old = 0
        
        if days_old > 30:
            warnings.append(f"Market data is {days_old} days old")
            # Treat as caution but not outright failure; downstream scoring captures confidence impact
        
        # Check for extreme volatility
        if len(market_data) >= 20:
            recent_volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            if recent_volatility > 0.05:  # 5% daily volatility
                warnings.append(f"High volatility detected: {recent_volatility:.2%}")
        
        return True, warnings
    
    def _validate_risk_return(self, signal: Signal) -> Tuple[bool, List[str]]:
        """Layer 3: Risk-return validation"""
        warnings = []
        
        # Check expected return
        if signal.expected_return is not None:
            if signal.expected_return < self.min_expected_return:
                warnings.append(f"Low expected return: {signal.expected_return:.2%}")
                return False, warnings
        
        # Check risk estimate
        if signal.risk_estimate is not None:
            if signal.risk_estimate > self.max_risk_threshold:
                warnings.append(f"High risk estimate: {signal.risk_estimate:.2%}")
                return False, warnings
        
        # Check risk-return ratio
        if signal.expected_return is not None and signal.risk_estimate is not None:
            risk_return_ratio = signal.expected_return / signal.risk_estimate
            if risk_return_ratio < 0.5:  # Less than 0.5 risk-adjusted return
                warnings.append(f"Poor risk-return ratio: {risk_return_ratio:.2f}")
                return False, warnings
        
        return True, warnings
    
    def _validate_technical_signals(self, signal: Signal, market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Layer 4: Technical analysis validation"""
        warnings = []
        
        if market_data.empty or len(market_data) < 20:
            warnings.append("Insufficient data for technical validation")
            return True, warnings  # Don't fail for insufficient data
        
        # Calculate technical indicators
        close_prices = market_data['close']
        
        # RSI validation
        rsi = self._calculate_rsi(close_prices, 14)
        if not rsi.empty:
            current_rsi = rsi.iloc[-1]
            if signal.direction == SignalDirection.BUY and current_rsi > 70:
                warnings.append(f"RSI overbought: {current_rsi:.1f}")
            elif signal.direction == SignalDirection.SELL and current_rsi < 30:
                warnings.append(f"RSI oversold: {current_rsi:.1f}")
        
        # Moving average trend
        ma_20 = close_prices.rolling(20).mean()
        ma_50 = close_prices.rolling(50).mean()
        
        if not ma_20.empty and not ma_50.empty:
            current_price = close_prices.iloc[-1]
            ma_20_current = ma_20.iloc[-1]
            ma_50_current = ma_50.iloc[-1]
            
            # Check trend alignment
            if signal.direction == SignalDirection.BUY:
                if current_price < ma_20_current:
                    warnings.append("Price below 20-day MA")
                if ma_20_current < ma_50_current:
                    warnings.append("20-day MA below 50-day MA (downtrend)")
            elif signal.direction == SignalDirection.SELL:
                if current_price > ma_20_current:
                    warnings.append("Price above 20-day MA")
                if ma_20_current > ma_50_current:
                    warnings.append("20-day MA above 50-day MA (uptrend)")
        
        return True, warnings  # Technical validation is advisory
    
    def _validate_confidence_calibration(self, signal: Signal, market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Layer 5: Confidence calibration validation"""
        warnings = []
        
        # Check if confidence is too high for uncertain conditions
        if market_data.empty or len(market_data) < 10:
            if signal.confidence > 0.8:
                warnings.append("High confidence with limited data")
                return False, warnings
        
        # Check for overconfidence in volatile conditions
        if len(market_data) >= 20:
            recent_volatility = market_data['close'].pct_change().rolling(10).std().iloc[-1]
            if recent_volatility > 0.03 and signal.confidence > 0.9:  # 3% volatility
                warnings.append("High confidence in volatile market")
                return False, warnings
        
        return True, warnings
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def backtest_signal_quality(self, signals: List[Signal], 
                              market_data: pd.DataFrame, 
                              lookback_days: int = 30) -> Dict:
        """
        Backtest signal quality over historical period
        
        Args:
            signals: List of historical signals
            market_data: Historical market data
            lookback_days: Number of days to analyze
            
        Returns:
            Backtest results with quality metrics
        """
        if not signals:
            return {"error": "No signals provided for backtesting"}
        
        # Filter signals within lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_signals = [s for s in signals if s.timestamp >= cutoff_date]
        
        if not recent_signals:
            return {"error": "No signals in lookback period"}
        
        # Calculate performance metrics
        results = {
            "total_signals": len(recent_signals),
            "buy_signals": len([s for s in recent_signals if s.direction == SignalDirection.BUY]),
            "sell_signals": len([s for s in recent_signals if s.direction == SignalDirection.SELL]),
            "avg_confidence": np.mean([s.confidence for s in recent_signals]),
            "signal_accuracy": self._calculate_signal_accuracy(recent_signals, market_data),
            "confidence_calibration": self._calculate_confidence_calibration(recent_signals, market_data)
        }
        
        return results
    
    def _calculate_signal_accuracy(self, signals: List[Signal], market_data: pd.DataFrame) -> float:
        """Calculate directional accuracy of signals"""
        if not signals or market_data.empty:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for signal in signals:
            # Find market data around signal time
            signal_date = signal.timestamp.date()
            
            # Get price movement after signal
            future_data = market_data[market_data.index.date >= signal_date]
            if len(future_data) < 2:
                continue
            
            # Calculate actual return over next 5 days
            initial_price = future_data['close'].iloc[0]
            final_price = future_data['close'].iloc[min(4, len(future_data)-1)]  # 5-day lookahead
            actual_return = (final_price - initial_price) / initial_price
            
            # Check if prediction was correct
            predicted_positive = signal.direction == SignalDirection.BUY
            actual_positive = actual_return > 0
            
            if predicted_positive == actual_positive:
                correct_predictions += 1
            
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_confidence_calibration(self, signals: List[Signal], market_data: pd.DataFrame) -> float:
        """Calculate how well confidence scores correlate with accuracy"""
        if not signals or market_data.empty:
            return 0.0
        
        # Group signals by confidence bins
        confidence_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
        bin_accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            bin_signals = [s for s in signals 
                          if confidence_bins[i] <= s.confidence < confidence_bins[i+1]]
            
            if bin_signals:
                bin_accuracy = self._calculate_signal_accuracy(bin_signals, market_data)
                bin_accuracies.append(bin_accuracy)
        
        # Calculate calibration score (how well confidence matches accuracy)
        if len(bin_accuracies) < 2:
            return 0.0
        
        # Simple correlation between confidence and accuracy
        confidence_centers = [0.25, 0.6, 0.75, 0.85, 0.95][:len(bin_accuracies)]
        correlation = np.corrcoef(confidence_centers, bin_accuracies)[0, 1]
        
        return max(0.0, correlation)  # Return 0 if negative correlation


# Global validator instance
signal_validator = SignalQualityValidator()


def validate_llm_signal(signal: Signal, market_data: pd.DataFrame) -> ValidationResult:
    """Convenience function to validate a signal"""
    return signal_validator.validate_signal(signal, market_data)


def backtest_signal_quality(signals: List[Signal], market_data: pd.DataFrame, 
                           lookback_days: int = 30) -> Dict:
    """Convenience function to backtest signal quality"""
    return signal_validator.backtest_signal_quality(signals, market_data, lookback_days)
