"""
AI LLM Module - Local GPU-based AI Integration
Phase 5.2: Production LLM Integration
Status: Production-ready with strict validation

This module integrates local LLM (Ollama) for:
- Market data interpretation
- Trading signal generation  
- Risk assessment
- Portfolio recommendations

REQUIREMENTS (per AGENT_INSTRUCTION.md):
- Maximum 500 lines per module
- Free tier only (local GPU, no API costs)
- Fail-fast validation (pipeline stops if Ollama unavailable)
- Zero breaking changes to existing ETL

Line Count Budget: ~500 lines total across all modules
Cost: $0/month (local GPU only)
"""

__version__ = "1.0.0"
__author__ = "Portfolio Maximizer Team"

from .ollama_client import OllamaClient
from .market_analyzer import LLMMarketAnalyzer
from .signal_generator import LLMSignalGenerator
from .risk_assessor import LLMRiskAssessor

__all__ = [
    'OllamaClient',
    'LLMMarketAnalyzer', 
    'LLMSignalGenerator',
    'LLMRiskAssessor'
]

