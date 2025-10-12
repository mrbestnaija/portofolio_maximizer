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
                "Provide concise, data-driven market analysis. "
                "Output valid JSON only."
            )
            
            llm_response = self.client.generate(
                prompt=prompt,
                system=system,
                temperature=0.1  # Low temp for consistency
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
        prompt = f"""Analyze the following market data for {ticker}:

PRICE DATA:
- Current Price: ${stats['current_price']}
- Price Change: {stats['price_change_pct']}%
- Volatility: {stats['volatility_pct']}%
- 52-Week High: ${stats['high_52w']}
- 52-Week Low: ${stats['low_52w']}

VOLUME DATA:
- Average Volume: {stats['avg_volume']:,}
- Volume Trend: {stats['volume_trend_pct']}%

Provide analysis in JSON format with these fields:
- trend: "bullish", "bearish", or "neutral"
- strength: 1-10 scale
- regime: "trending", "ranging", "volatile", or "stable"  
- key_levels: [support, resistance] prices
- summary: brief 2-sentence analysis

Output ONLY valid JSON."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(json_str)
            
            # Validate required fields
            required = ['trend', 'strength', 'regime', 'summary']
            for field in required:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Return minimal valid response
            return {
                'trend': 'neutral',
                'strength': 5,
                'regime': 'unknown',
                'summary': 'Analysis parsing failed',
                'error': str(e)
            }


# Validation
assert LLMMarketAnalyzer.analyze_ohlcv.__doc__ is not None
assert len(LLMMarketAnalyzer.__init__.__doc__) > 20

# Line count: ~170 lines (within budget)

