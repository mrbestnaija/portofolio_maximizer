"""
LLM Risk Assessor - AI-powered risk analysis
Line Count: ~100 lines (within 500-line budget)
Cost: $0/month (local GPU)

Provides risk assessment after data validation stage.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

import os

from .ollama_client import OllamaClient, OllamaConnectionError
from .performance_monitor import record_latency_fallback

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
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """Initialize with validated Ollama client"""
        self.client = ollama_client
        self.force_fallback = os.getenv("LLM_FORCE_FALLBACK", "0") == "1"
        if self.force_fallback:
            logger.info("LLM_FORCE_FALLBACK enabled for risk assessor")
        self.system_prompt = system_prompt or (
            "You are a quantitative risk analyst. "
            "Assess portfolio risk based on statistical metrics. "
            "Output valid JSON only."
        )
        self.temperature = temperature
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
        
        if self.force_fallback:
            logger.info("LLM_FORCE_FALLBACK=1 - skipping LLM risk assessment for %s", ticker)
            record_latency_fallback(
                stage="risk_assessment",
                ticker=ticker,
                reason="forced_fallback_env",
                inference_stats=None,
            )
            return self._fallback_assessment(ticker, metrics, portfolio_weight)

        try:
            llm_response = self.client.generate(
                prompt=prompt,
                system=self.system_prompt,
                temperature=self.temperature
            )
            
            # Parse response
            assessment = self._parse_risk_response(llm_response)
            
            # Add metadata
            assessment.update({
                'ticker': ticker,
                'assessment_timestamp': datetime.now().isoformat(),
                'portfolio_weight': portfolio_weight,
                'metrics': metrics,
                'fallback': False,
            })

            should_fallback, reason = self._latency_guard()
            if should_fallback:
                logger.warning(
                    "LLM risk assessment exceeded performance guard (%s); using deterministic fallback.",
                    reason,
                )
                record_latency_fallback(
                    stage="risk_assessment",
                    ticker=ticker,
                    reason=reason or "latency_guard_triggered",
                    inference_stats=self.client.get_last_inference_stats(),
                )
                self.force_fallback = True
                return self._fallback_assessment(ticker, metrics, portfolio_weight)
            
            logger.info(f"Risk assessed for {ticker}: {assessment['risk_level']}")
            return assessment
            
        except OllamaConnectionError as exc:
            logger.warning(
                "LLM risk assessor unavailable for %s; raising to fail fast (%s)",
                ticker,
                exc,
            )
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

    def _fallback_assessment(
        self,
        ticker: str,
        metrics: Dict[str, float],
        weight: float,
    ) -> Dict[str, Any]:
        volatility = metrics.get('volatility_annual_pct', 0.0)
        drawdown = metrics.get('max_drawdown_pct', 0.0)
        sharpe = metrics.get('sharpe_ratio', 0.0)

        risk_level = 'medium'
        risk_score = 60
        concerns = []

        if volatility > 40 or drawdown < -35:
            risk_level = 'high'
            risk_score = 80
            concerns.append('Elevated volatility/drawdown conditions')
        elif volatility < 20 and drawdown > -15 and sharpe > 0.5:
            risk_level = 'low'
            risk_score = 40
        else:
            concerns.append('Moderate volatility regime')

        recommendation = 'Maintain position with existing limits'
        if risk_level == 'high':
            recommendation = 'Reduce exposure until volatility normalises'
        elif risk_level == 'low':
            recommendation = 'Position size acceptable within risk budget'

        return {
            'risk_level': risk_level,
            'risk_score': int(risk_score),
            'concerns': concerns,
            'recommendation': recommendation,
            'ticker': ticker,
            'assessment_timestamp': datetime.now().isoformat(),
            'portfolio_weight': weight,
            'metrics': metrics,
            'fallback': True,
        }

    def _latency_guard(self) -> Tuple[bool, Optional[str]]:
        """Return whether to fall back due to latency constraints."""
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


# Line count: ~140 lines (within budget)

