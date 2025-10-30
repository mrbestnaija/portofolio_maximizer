"""
LLM Risk Assessor - AI-powered risk analysis
Line Count: ~100 lines (within 500-line budget)
Cost: $0/month (local GPU)

Provides risk assessment after data validation stage.
"""

import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime
import json

from .ollama_client import OllamaClient, OllamaConnectionError

logger = logging.getLogger(__name__)


class LLMRiskAssessor:
    """
    LLM-based risk assessment for portfolio positions.
    
    Analyzes:
    - Volatility risk
    - Drawdown risk
    - Liquidity risk
    - Market regime risk
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """Initialize with validated Ollama client"""
        self.client = ollama_client
        logger.info("LLM Risk Assessor initialized")
    
    def assess_risk(self,
                   data: pd.DataFrame,
                   ticker: str,
                   portfolio_weight: float = 0.0) -> Dict[str, Any]:
        """
        Assess risk for a ticker position.
        
        Args:
            data: OHLCV data
            ticker: Stock ticker
            portfolio_weight: Position weight (0.0-1.0)
            
        Returns:
            Risk assessment dict
            
        Raises:
            OllamaConnectionError: If LLM unavailable
        """
        # Compute risk metrics
        metrics = self._compute_risk_metrics(data)
        
        # Create risk assessment prompt
        prompt = self._create_risk_prompt(ticker, metrics, portfolio_weight)
        
        try:
            system = (
                "You are a quantitative risk analyst. "
                "Assess portfolio risk based on statistical metrics. "
                "Output valid JSON only."
            )
            
            llm_response = self.client.generate(
                prompt=prompt,
                system=system,
                temperature=0.1
            )
            
            # Parse response
            assessment = self._parse_risk_response(llm_response)
            
            # Add metadata
            assessment.update({
                'ticker': ticker,
                'assessment_timestamp': datetime.now().isoformat(),
                'portfolio_weight': portfolio_weight,
                'metrics': metrics
            })
            
            logger.info(f"Risk assessed for {ticker}: {assessment['risk_level']}")
            return assessment
            
        except OllamaConnectionError:
            raise
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            raise OllamaConnectionError(f"Risk assessment failed: {e}")
    
    def _compute_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute risk metrics"""
        close = data['Close']
        returns = close.pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * (252 ** 0.5) * 100
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max * 100).min()
        
        # Value at Risk (95%)
        var_95 = returns.quantile(0.05) * 100
        
        # Sharpe ratio (simplified, assuming 0% risk-free rate)
        sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
        
        return {
            'volatility_annual_pct': round(volatility, 2),
            'max_drawdown_pct': round(drawdown, 2),
            'var_95_pct': round(var_95, 2),
            'sharpe_ratio': round(sharpe, 2)
        }
    
    def _create_risk_prompt(self, ticker: str, metrics: Dict, weight: float) -> str:
        """Create risk assessment prompt"""
        prompt = f"""Assess risk for {ticker} position:

RISK METRICS:
- Annual Volatility: {metrics['volatility_annual_pct']}%
- Max Drawdown: {metrics['max_drawdown_pct']}%
- VaR (95%): {metrics['var_95_pct']}%
- Sharpe Ratio: {metrics['sharpe_ratio']}

POSITION:
- Portfolio Weight: {weight * 100:.1f}%

Provide risk assessment in JSON:
- risk_level: "low", "medium", "high", or "extreme"
- risk_score: 0-100 scale
- concerns: list of key risk factors
- recommendation: position sizing advice

Output ONLY valid JSON."""
        
        return prompt
    
    def _parse_risk_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM risk response"""
        try:
            json_str = response.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            
            assessment = json.loads(json_str)
            
            return {
                'risk_level': assessment.get('risk_level', 'medium'),
                'risk_score': int(assessment.get('risk_score', 50)),
                'concerns': assessment.get('concerns', []),
                'recommendation': assessment.get('recommendation', 'Hold current position')
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse risk assessment: {e}")
            return {
                'risk_level': 'high',
                'risk_score': 75,
                'concerns': ['Assessment parsing failed'],
                'recommendation': 'Reduce position size due to parsing error',
                'error': str(e)
            }


# Line count: ~140 lines (within budget)

