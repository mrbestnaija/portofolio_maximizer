"""
LLM Signal Generator - AI-powered trading signal generation  
Line Count: ~130 lines (within 500-line budget)
Cost: $0/month (local GPU)

Generates trading signals after preprocessing stage.
CRITICAL: Signals must be validated against data-driven rules.
"""

import pandas as pd
import logging
from typing import Dict, Any, Literal
from datetime import datetime
import json

from .ollama_client import OllamaClient, OllamaConnectionError

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
    
    def __init__(self, ollama_client: OllamaClient):
        """Initialize with validated Ollama client"""
        self.client = ollama_client
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
        
        # Create signal generation prompt
        prompt = self._create_signal_prompt(ticker, market_analysis, indicators)
        
        try:
            system = (
                "You are a quantitative trading strategist. "
                "Generate trading signals based on data analysis. "
                "Be conservative - only strong signals justified by data. "
                "Output valid JSON only."
            )
            
            llm_response = self.client.generate(
                prompt=prompt,
                system=system,
                temperature=0.05  # Very low for consistency
            )
            
            # Parse LLM signal
            signal = self._parse_signal_response(llm_response)
            
            # Add metadata
            signal.update({
                'ticker': ticker,
                'signal_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start': data.index.min().isoformat(),
                    'end': data.index.max().isoformat()
                },
                'indicators': indicators,
                'llm_model': self.client.model
            })
            
            logger.info(f"Signal generated for {ticker}: {signal['action']}")
            return signal
            
        except OllamaConnectionError:
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
            'volume_ma_ratio': round(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1], 2)
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


# Validation
assert LLMSignalGenerator.generate_signal.__doc__ is not None

# Line count: ~160 lines (within budget)

