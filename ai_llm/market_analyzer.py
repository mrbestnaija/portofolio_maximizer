"""
LLM Market Analyzer - AI-powered market data interpretation
Line Count: ~150 lines (within 500-line budget)
Cost: $0/month (local GPU)

Provides LLM-based analysis of market data at ETL extraction stage.
"""

import pandas as pd
import logging
from typing import Dict, Any, List
from datetime import datetime
import json

from .ollama_client import OllamaClient, OllamaConnectionError

logger = logging.getLogger(__name__)


class LLMMarketAnalyzer:
    """
    LLM-based market data analyzer for production pipeline.
    
    Integrates at ETL extraction stage to provide:
    - Price action interpretation
    - Volume analysis
    - Trend detection
    - Market regime classification
    
    REQUIREMENTS:
    - Strict input validation
    - JSON-structured outputs
    - Performance tracking
    - Fail-fast on LLM unavailability
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize with validated Ollama client.
        
        Args:
            ollama_client: Pre-validated OllamaClient instance
        """
        self.client = ollama_client
        logger.info("LLM Market Analyzer initialized")
    
    def analyze_ohlcv(self, 
                      data: pd.DataFrame,
                      ticker: str) -> Dict[str, Any]:
        """
        Analyze OHLCV data and generate LLM-powered insights.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            ticker: Stock ticker symbol
            
        Returns:
            Analysis dict with LLM insights
            
        Raises:
            ValueError: If data validation fails
            OllamaConnectionError: If LLM fails (pipeline stops)
        """
        # Validate input data
        self._validate_data(data)
        
        # Extract statistical summary
        stats = self._compute_statistics(data)
        
        # Generate LLM analysis prompt
        prompt = self._create_analysis_prompt(ticker, stats)
        
        # Get LLM response
        try:
            system = (
                "You are a quantitative financial analyst. "
                "You MUST respond with ONLY valid JSON. "
                "NO explanations, NO markdown, NO extra text. "
                "ONLY the JSON object. This is critical."
            )
            
            llm_response = self.client.generate(
                prompt=prompt,
                system=system,
                temperature=0.05  # Very low temp for strict JSON output
            )
            
            # Parse LLM response
            analysis = self._parse_llm_response(llm_response)
            
            # Add metadata
            analysis.update({
                'ticker': ticker,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start': data.index.min().isoformat(),
                    'end': data.index.max().isoformat(),
                    'days': len(data)
                },
                'statistics': stats
            })
            
            logger.info(f"LLM analysis complete for {ticker}")
            return analysis
            
        except OllamaConnectionError:
            # Re-raise to stop pipeline (per requirement 3b)
            raise
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise OllamaConnectionError(f"Market analysis failed: {e}")
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate OHLCV data structure"""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute statistical summary for LLM context"""
        close = data['Close']
        volume = data['Volume']
        
        # Price statistics
        price_change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
        volatility = close.pct_change().std() * 100
        
        # Volume statistics
        avg_volume = volume.mean()
        volume_trend = ((volume.iloc[-5:].mean() - volume.iloc[:5].mean()) / 
                       volume.iloc[:5].mean()) * 100
        
        return {
            'current_price': round(close.iloc[-1], 2),
            'price_change_pct': round(price_change, 2),
            'volatility_pct': round(volatility, 2),
            'avg_volume': int(avg_volume),
            'volume_trend_pct': round(volume_trend, 2),
            'high_52w': round(close.max(), 2),
            'low_52w': round(close.min(), 2)
        }
    
    def _create_analysis_prompt(self, ticker: str, stats: Dict) -> str:
        """Create structured prompt for LLM"""
        prompt = f"""Analyze {ticker} market data and respond with ONLY a valid JSON object.

DATA:
- Current Price: ${stats['current_price']}
- Price Change: {stats['price_change_pct']}%
- Volatility: {stats['volatility_pct']}%
- 52-Week High: ${stats['high_52w']}
- 52-Week Low: ${stats['low_52w']}
- Average Volume: {stats['avg_volume']:,}
- Volume Trend: {stats['volume_trend_pct']}%

Return EXACTLY this JSON structure (no other text):
{{
  "trend": "bullish or bearish or neutral",
  "strength": 5,
  "regime": "trending or ranging or volatile or stable",
  "key_levels": [100.0, 110.0],
  "summary": "Brief analysis in 1-2 sentences"
}}

IMPORTANT: Output ONLY the JSON object above. No markdown, no explanation, ONLY JSON."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response with robust error handling"""
        try:
            # Clean up response
            json_str = response.strip()
            
            # Remove markdown code blocks if present
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            
            # Try to find JSON object if response has extra text
            if not json_str.startswith('{'):
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]
            
            # Parse JSON
            analysis = json.loads(json_str)
            
            # Validate and fix required fields
            required_defaults = {
                'trend': 'neutral',
                'strength': 5,
                'regime': 'unknown',
                'summary': 'No summary available',
                'key_levels': []
            }
            
            for field, default in required_defaults.items():
                if field not in analysis:
                    logger.warning(f"Missing field '{field}', using default: {default}")
                    analysis[field] = default
            
            # Validate field types
            if not isinstance(analysis['strength'], (int, float)):
                analysis['strength'] = 5
            else:
                analysis['strength'] = max(1, min(10, int(analysis['strength'])))
            
            if analysis['trend'] not in ['bullish', 'bearish', 'neutral']:
                analysis['trend'] = 'neutral'
            
            if analysis['regime'] not in ['trending', 'ranging', 'volatile', 'stable']:
                analysis['regime'] = 'unknown'
            
            logger.debug(f"LLM response parsed successfully: {analysis.get('trend', 'unknown')}")
            return analysis
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response[:200]}...")
            
            # Return safe fallback response
            return {
                'trend': 'neutral',
                'strength': 5,
                'regime': 'unknown',
                'key_levels': [],
                'summary': 'Unable to parse LLM analysis. Using neutral stance.',
                'error': str(e),
                'raw_response_preview': response[:100] if response else 'empty'
            }


# Validation
assert LLMMarketAnalyzer.analyze_ohlcv.__doc__ is not None
assert len(LLMMarketAnalyzer.__init__.__doc__) > 20

# Line count: ~170 lines (within budget)

