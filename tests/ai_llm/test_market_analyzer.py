"""
Unit tests for LLM Market Analyzer
Line Count: ~150 lines (within 500-line test budget)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_llm.market_analyzer import LLMMarketAnalyzer
from ai_llm.ollama_client import OllamaConnectionError


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    prices = 100 + np.cumsum(np.random.randn(100))

    data = pd.DataFrame({
        'Open': prices + np.random.rand(100),
        'High': prices + np.random.rand(100) + 1,
        'Low': prices - np.random.rand(100),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


@pytest.fixture
def mock_ollama_client():
    """Create mock Ollama client"""
    client = Mock()
    client.model = 'deepseek-coder:6.7b-instruct-q4_K_M'
    client.should_use_latency_fallback.return_value = (False, None)
    return client


class TestMarketAnalyzerValidation:
    """Test input data validation"""

    def test_validate_rejects_empty_dataframe(self, mock_ollama_client):
        """Test that empty DataFrame is rejected"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError) as exc_info:
            analyzer.analyze_ohlcv(empty_df, 'AAPL')

        assert 'empty' in str(exc_info.value).lower()

    def test_validate_checks_required_columns(self, mock_ollama_client):
        """Test that missing columns are detected"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        incomplete_df = pd.DataFrame({'Close': [100, 101, 102]})

        with pytest.raises(ValueError) as exc_info:
            analyzer.analyze_ohlcv(incomplete_df, 'AAPL')

        assert 'missing columns' in str(exc_info.value).lower()

    def test_validate_requires_datetime_index(self, mock_ollama_client):
        """Test that DatetimeIndex is required"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        data = pd.DataFrame({
            'Open': [100], 'High': [101], 'Low': [99],
            'Close': [100], 'Volume': [1000000]
        })

        with pytest.raises(ValueError) as exc_info:
            analyzer.analyze_ohlcv(data, 'AAPL')

        assert 'datetimeindex' in str(exc_info.value).lower()


class TestStatisticsComputation:
    """Test statistical calculations"""

    def test_compute_statistics_calculates_price_metrics(self, mock_ollama_client, sample_ohlcv_data):
        """Test price statistics computation"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        stats = analyzer._compute_statistics(sample_ohlcv_data)

        assert 'current_price' in stats
        assert 'price_change_pct' in stats
        assert 'volatility_pct' in stats
        assert isinstance(stats['current_price'], (int, float))

    def test_compute_statistics_calculates_volume_metrics(self, mock_ollama_client, sample_ohlcv_data):
        """Test volume statistics computation"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        stats = analyzer._compute_statistics(sample_ohlcv_data)

        assert 'avg_volume' in stats
        assert 'volume_trend_pct' in stats
        assert stats['avg_volume'] > 0


class TestLLMAnalysis:
    """Test LLM integration"""

    def test_analyze_returns_complete_dict(self, mock_ollama_client, sample_ohlcv_data):
        """Test that analysis returns all required fields"""
        mock_ollama_client.generate.return_value = '''```json
{
    "trend": "bullish",
    "strength": 7,
    "regime": "trending",
    "key_levels": [95, 105],
    "summary": "Strong upward trend with increasing volume."
}
```'''

        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        result = analyzer.analyze_ohlcv(sample_ohlcv_data, 'AAPL')

        assert 'ticker' in result
        assert 'trend' in result
        assert 'statistics' in result
        assert 'analysis_timestamp' in result
        assert result['ticker'] == 'AAPL'

    def test_analyze_handles_parsing_errors_gracefully(self, mock_ollama_client, sample_ohlcv_data):
        """Test graceful handling of invalid LLM responses"""
        mock_ollama_client.generate.return_value = 'Invalid JSON response'

        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        result = analyzer.analyze_ohlcv(sample_ohlcv_data, 'AAPL')

        # Should return valid structure even if parsing fails
        assert 'trend' in result
        assert result['trend'] == 'neutral'  # Default fallback
        assert 'error' in result

    def test_analyze_uses_latency_fallback(self, mock_ollama_client, sample_ohlcv_data):
        """Latency guard should trigger deterministic fallback when performance drops."""
        mock_ollama_client.generate.return_value = '{"trend": "bullish", "strength": 8, "regime": "trending", "summary": "LLM"}'
        mock_ollama_client.should_use_latency_fallback.return_value = (True, "latency 12.0s > 5.0s")

        with patch("ai_llm.market_analyzer.record_latency_fallback") as mock_latency_record:
            analyzer = LLMMarketAnalyzer(mock_ollama_client)
            result = analyzer.analyze_ohlcv(sample_ohlcv_data, 'AAPL')

        assert result['fallback'] is True
        assert analyzer.force_fallback is True
        assert result['trend'] in {'bullish', 'bearish', 'neutral'}
        mock_latency_record.assert_called_once()

    def test_analyze_propagates_ollama_errors(self, mock_ollama_client, sample_ohlcv_data):
        """Test that Ollama errors stop the pipeline"""
        mock_ollama_client.generate.side_effect = OllamaConnectionError("Server down")

        analyzer = LLMMarketAnalyzer(mock_ollama_client)

        with pytest.raises(OllamaConnectionError):
            analyzer.analyze_ohlcv(sample_ohlcv_data, 'AAPL')


class TestPromptCreation:
    """Test LLM prompt generation"""

    def test_create_prompt_includes_statistics(self, mock_ollama_client):
        """Test that prompt includes all statistics"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        stats = {
            'current_price': 100.0,
            'price_change_pct': 5.0,
            'volatility_pct': 15.0,
            'avg_volume': 1000000,
            'volume_trend_pct': 10.0,
            'high_52w': 110.0,
            'low_52w': 90.0
        }

        prompt = analyzer._create_analysis_prompt('AAPL', stats)

        assert 'AAPL' in prompt
        assert '100.0' in prompt or '100' in prompt
        assert 'Volume' in prompt
        assert 'JSON' in prompt  # Should request JSON output


# Line count: ~160 lines (within budget)
