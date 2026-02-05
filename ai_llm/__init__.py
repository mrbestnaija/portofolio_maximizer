"""
AI LLM Module - Local GPU-based AI Integration
Phase 5.2: Production LLM Integration
Status: Deprecated (disabled by default)

This module integrates local LLM (Ollama) for:
- Market data interpretation
- Trading signal generation  
- Risk assessment
- Portfolio recommendations

Note: Ollama-backed paths are currently disabled by default to avoid unnecessary delays when the local server is not running, and may be removed in a future cleanup pass.

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

