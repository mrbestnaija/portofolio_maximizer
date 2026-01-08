"""
LLM Signal Generator - AI-powered trading signal generation  
Line Count: ~130 lines (within 500-line budget)
Cost: $0/month (local GPU)

Generates trading signals after preprocessing stage.
CRITICAL: Signals must be validated against data-driven rules.
"""

import pandas as pd
import logging
from typing import Dict, Any, Literal, Optional, Tuple
from datetime import datetime
import json

import os

from .ollama_client import OllamaClient, OllamaConnectionError
from .performance_monitor import record_latency_fallback

logger = logging.getLogger(__name__)

SignalType = Literal['BUY', 'SELL', 'HOLD']


class LLMSignalGenerator:
    """
    LLM-based trading signal generator.
    
    WARNING: LLM signals are advisory only. 
    Must be validated with quantitative backtests before use.
    
    Per AGENT_INSTRUCTION.md:
    - No trading until profitable strategy proven (>10% annual return)
    - LLM signals require 30+ days validation
    - Must beat buy-and-hold baseline
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        system_prompt: Optional[str] = None,
        temperature: float = 0.05,
        validation_rules: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with validated Ollama client"""
        self.client = ollama_client
        self.force_fallback = os.getenv("LLM_FORCE_FALLBACK", "0") == "1"
        if self.force_fallback:
            logger.info("LLM_FORCE_FALLBACK enabled for signal generator")
        self.system_prompt = system_prompt or (
            "You are a quantitative trading strategist. "
            "Generate trading signals based on data analysis. "
            "Be conservative - only strong signals justified by data. "
            "Output valid JSON only."
        )
        self.temperature = temperature
        self.validation_rules = validation_rules or {}
        logger.info("LLM Signal Generator initialized")
    
    def generate_signal(self,
                       data: pd.DataFrame,
                       ticker: str,
                       market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal with LLM reasoning.
        
        Args:
            data: Preprocessed OHLCV data
            ticker: Stock ticker
            market_analysis: Previous LLM market analysis
            
        Returns:
            Signal dict with action, confidence, reasoning
            
        Raises:
            OllamaConnectionError: If LLM unavailable
        """
        # Compute technical indicators for context
        indicators = self._compute_indicators(data)
        period_info = {
            "start": data.index.min().isoformat(),
            "end": data.index.max().isoformat(),
            "days": len(data),
        }
        
        # Create signal generation prompt
        prompt = self._create_signal_prompt(ticker, market_analysis, indicators)
        
        if self.force_fallback:
            logger.info("LLM_FORCE_FALLBACK=1 - skipping LLM signal for %s", ticker)
            record_latency_fallback(
                stage="signal_generation",
                ticker=ticker,
                reason="forced_fallback_env",
                inference_stats=None,
            )
            return self._fallback_signal(ticker, indicators, market_analysis, period_info)

        try:
            llm_response = self.client.generate(
                prompt=prompt,
                system=self.system_prompt,
                temperature=self.temperature  # Very low for consistency
            )
            
            # Parse LLM signal
            signal = self._parse_signal_response(llm_response)

            # Align signal timestamp with the latest available market data point
            latest_index = data.index[-1]
            if isinstance(latest_index, datetime):
                timestamp_iso = latest_index.isoformat()
            else:
                timestamp_iso = pd.to_datetime(latest_index).isoformat()

            # Add metadata
            signal.update({
                'ticker': ticker,
                'signal_timestamp': timestamp_iso,
                'data_period': period_info,
                'indicators': indicators,
                'llm_model': self.client.model,
                'fallback': False,
                'signal_type': signal.get('action'),
            })

            should_fallback, reason = self._latency_guard()
            if should_fallback:
                logger.warning(
                    "LLM signal generation exceeded performance guard (%s); using deterministic fallback.",
                    reason,
                )
                record_latency_fallback(
                    stage="signal_generation",
                    ticker=ticker,
                    reason=reason or "latency_guard_triggered",
                    inference_stats=self.client.get_last_inference_stats(),
                )
                self.force_fallback = True
                return self._fallback_signal(ticker, indicators, market_analysis, period_info)
            
            signal = self._apply_generation_rules(signal)
            
            logger.info(f"Signal generated for {ticker}: {signal['action']}")
            return signal
            
        except OllamaConnectionError as exc:
            logger.warning(
                "LLM signal unavailable for %s; raising to fail fast (%s)",
                ticker,
                exc,
            )
            raise
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise OllamaConnectionError(f"Signal generation failed: {e}")
    
    def _compute_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute simple technical indicators"""
        close = data['Close']
        volume = data['Volume']
        
        # Simple Moving Averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20

        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        if pd.isna(vol_ma20) or vol_ma20 == 0:
            # Avoid divide-by-zero/NaN when volumes are missing or zero
            volume_ma_ratio = 0.0
        else:
            volume_ma_ratio = round(volume.iloc[-1] / vol_ma20, 2)
        
        # RSI (simplified)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'current_price': round(close.iloc[-1], 2),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'rsi_14': round(rsi, 2),
            'volume_ma_ratio': volume_ma_ratio,
        }
    
    def _create_signal_prompt(self, ticker: str, analysis: Dict, indicators: Dict) -> str:
        """Create signal generation prompt"""
        prompt = f"""Generate trading signal for {ticker}:

MARKET ANALYSIS:
- Trend: {analysis.get('trend', 'unknown')}
- Strength: {analysis.get('strength', 'N/A')}/10
- Regime: {analysis.get('regime', 'unknown')}

TECHNICAL INDICATORS:
- Price: ${indicators['current_price']}
- SMA(20): ${indicators['sma_20']}
- SMA(50): ${indicators['sma_50']}
- RSI(14): {indicators['rsi_14']}
- Volume Ratio: {indicators['volume_ma_ratio']}x

Generate signal in JSON format:
- action: "BUY", "SELL", or "HOLD"
- confidence: 0.0-1.0 scale
- reasoning: 1-2 sentence justification
- risk_level: "low", "medium", or "high"

BE CONSERVATIVE. Only strong signals with clear data support.
Output ONLY valid JSON."""
        
        return prompt
    
    def _parse_signal_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM signal response"""
        try:
            # Extract JSON
            json_str = response.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            
            signal = json.loads(json_str)
            
            # Validate and normalize
            action = signal.get('action', 'HOLD').upper()
            if action not in ['BUY', 'SELL', 'HOLD']:
                action = 'HOLD'
            
            confidence = float(signal.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': signal.get('reasoning', 'No reasoning provided'),
                'risk_level': signal.get('risk_level', 'medium')
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse signal: {e}")
            # Return safe default
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': f'Signal parsing failed: {e}',
                'risk_level': 'high',
                'error': str(e)
            }

    def _fallback_signal(
        self,
        ticker: str,
        indicators: Dict[str, float],
        analysis: Dict[str, Any],
        period_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Conservative heuristic signal when LLM is unavailable."""
        action = 'HOLD'
        confidence = 0.35
        risk_level = 'medium'
        reasoning_parts = []

        sma20 = indicators.get('sma_20')
        sma50 = indicators.get('sma_50')
        rsi = indicators.get('rsi_14')
        trend = (analysis or {}).get('trend', 'neutral')

        if sma20 is not None and sma50 is not None:
            if sma20 > sma50 and trend == 'bullish' and rsi is not None and rsi < 70:
                action = 'BUY'
                confidence = 0.55
                reasoning_parts.append('SMA20 above SMA50 with supportive bullish trend')
            elif sma20 < sma50 and trend == 'bearish' and rsi is not None and rsi > 30:
                action = 'SELL'
                confidence = 0.55
                reasoning_parts.append('SMA20 below SMA50 with bearish regime')

        if rsi is not None:
            if rsi > 75:
                action = 'SELL'
                confidence = max(confidence, 0.5)
                reasoning_parts.append('RSI overbought signal')
            elif rsi < 25:
                action = 'BUY'
                confidence = max(confidence, 0.5)
                reasoning_parts.append('RSI oversold signal')

        if action == 'HOLD' and not reasoning_parts:
            reasoning_parts.append('No high-conviction setup detected; defaulting to HOLD')

        reasoning = '; '.join(reasoning_parts)

        return {
            'action': action,
            'confidence': round(confidence, 2),
            'reasoning': reasoning,
            'risk_level': risk_level,
            'fallback': True,
            'ticker': ticker,
            'signal_timestamp': datetime.now().isoformat(),
            'data_period': period_info,
            'indicators': indicators,
            'llm_model': f"fallback:{self.client.model}",
        }

    def _apply_generation_rules(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conservative generation rules before validation."""
        min_conf_for_action = float(self.validation_rules.get('min_confidence_for_action', 0.0))
        conservative_bias = bool(self.validation_rules.get('conservative_bias', False))
        require_reasoning = bool(self.validation_rules.get('require_reasoning', False))

        action = signal.get('action', 'HOLD').upper()
        confidence = float(signal.get('confidence', 0.0))
        reasoning = signal.get('reasoning', '')

        # Enforce reasoning requirement
        if require_reasoning and len(reasoning.strip()) < 30:
            logger.debug("Signal reasoning too short; defaulting to HOLD.")
            signal['action'] = 'HOLD'
            signal['reasoning'] = f"{reasoning} [Adjusted: reasoning below minimum length]"
            return signal

        # Enforce minimum confidence for buy/sell
        if action in ('BUY', 'SELL') and min_conf_for_action > 0:
            if confidence < min_conf_for_action:
                logger.debug(
                    "Signal confidence %.2f below threshold %.2f; demoting to HOLD.",
                    confidence,
                    min_conf_for_action,
                )
                signal['action'] = 'HOLD'
                signal['reasoning'] = (
                    f"{reasoning} [Adjusted: confidence {confidence:.2f} < {min_conf_for_action:.2f}]"
                )
                return signal

        # Enforce conservative bias (default to HOLD unless strong evidence)
        if conservative_bias and action in ('BUY', 'SELL'):
            confidence_threshold = max(min_conf_for_action, 0.75)
            if confidence < confidence_threshold:
                logger.debug(
                    "Conservative bias active; confidence %.2f below %.2f. Holding.",
                    confidence,
                    confidence_threshold,
                )
                signal['action'] = 'HOLD'
                signal['reasoning'] = (
                    f"{reasoning} [Adjusted: conservative bias enforced at {confidence_threshold:.2f}]"
                )

        return signal

    def _latency_guard(self) -> Tuple[bool, Optional[str]]:
        """Return whether a latency-based fallback should trigger for signal generation."""
        # In diagnostic mode, effectively disable latency fallback so we see LLM behaviour.
        diag_mode = str(os.getenv("DIAGNOSTIC_MODE") or os.getenv("TS_DIAGNOSTIC_MODE") or "0") == "1"
        if diag_mode:
            return False, None

        max_latency_env = os.getenv("LLM_MAX_LATENCY_SECONDS", "5.0")
        min_tokens_env = os.getenv("LLM_MIN_TOKENS_PER_SEC", "5.0")
        try:
            max_latency_override = float(max_latency_env)
        except ValueError:
            max_latency_override = 5.0
        try:
            min_tokens = float(min_tokens_env)
        except ValueError:
            min_tokens = 5.0

        guard_fn = getattr(self.client, "should_use_latency_fallback", None)
        if not callable(guard_fn):
            return False, None

        should_fallback, reason = guard_fn(
            max_latency_override=max_latency_override,
            min_token_rate=min_tokens,
        )
        return bool(should_fallback), reason


# Validation
assert LLMSignalGenerator.generate_signal.__doc__ is not None

# Line count: ~160 lines (within budget)

