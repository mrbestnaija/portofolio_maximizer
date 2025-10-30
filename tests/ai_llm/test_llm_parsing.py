"""
Test LLM Response Parsing Robustness
Tests that LLM modules handle malformed responses gracefully
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from ai_llm.market_analyzer import LLMMarketAnalyzer


class MockOllamaClient:
    """Mock client for testing parsing"""
    
    def __init__(self, response_type='valid'):
        self.response_type = response_type
        self.model = "test-model"
    
    def generate(self, prompt, system=None, temperature=0.1):
        """Return different response types for testing"""
        responses = {
            'valid': '{"trend": "bullish", "strength": 7, "regime": "trending", "key_levels": [100, 110], "summary": "Strong uptrend"}',
            'markdown': '```json\n{"trend": "bearish", "strength": 3, "regime": "ranging", "summary": "Weak trend"}\n```',
            'extra_text': 'Based on the analysis:\n{"trend": "neutral", "strength": 5, "regime": "stable", "summary": "Neutral market"}\nThis is my analysis.',
            'incomplete': '{"trend": "bullish", "strength": 7}',  # Missing fields
            'invalid_json': '{trend: bullish, strength: 7',  # Invalid JSON
            'empty': '',
            'no_json': 'The market is bullish with strong momentum'
        }
        return responses.get(self.response_type, responses['valid'])


def create_test_data(days=60):
    """Create test OHLCV data with DatetimeIndex"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    return pd.DataFrame({
        'Open': [99.0 + i*0.5 for i in range(days)],
        'High': [103.0 + i*0.5 for i in range(days)],
        'Low': [98.0 + i*0.5 for i in range(days)],
        'Close': [100.0 + i*0.5 for i in range(days)],
        'Volume': [1000000 + i*10000 for i in range(days)]
    }, index=dates)


def test_valid_json_response():
    """Test parsing of valid JSON response"""
    client = MockOllamaClient('valid')
    analyzer = LLMMarketAnalyzer(client)
    data = create_test_data()
    
    result = analyzer.analyze_ohlcv(data, ticker='TEST')
    
    assert result['trend'] == 'bullish'
    assert result['strength'] == 7
    assert result['regime'] == 'trending'
    assert 'ticker' in result
    assert 'analysis_timestamp' in result


def test_markdown_wrapped_json():
    """Test parsing of JSON wrapped in markdown"""
    client = MockOllamaClient('markdown')
    analyzer = LLMMarketAnalyzer(client)
    data = create_test_data()
    
    result = analyzer.analyze_ohlcv(data, ticker='TEST')
    
    assert result['trend'] == 'bearish'
    assert result['strength'] == 3
    assert 'error' not in result  # Should parse successfully


def test_json_with_extra_text():
    """Test parsing of JSON with surrounding text"""
    client = MockOllamaClient('extra_text')
    analyzer = LLMMarketAnalyzer(client)
    data = create_test_data()
    
    result = analyzer.analyze_ohlcv(data, ticker='TEST')
    
    assert result['trend'] == 'neutral'
    assert result['strength'] == 5
    assert 'error' not in result  # Should extract JSON successfully


def test_incomplete_json():
    """Test handling of incomplete JSON (missing required fields)"""
    client = MockOllamaClient('incomplete')
    analyzer = LLMMarketAnalyzer(client)
    data = create_test_data()
    
    result = analyzer.analyze_ohlcv(data, ticker='TEST')
    
    # Should have all required fields with defaults
    assert 'trend' in result
    assert 'strength' in result
    assert 'regime' in result
    assert 'summary' in result
    assert 'key_levels' in result


def test_invalid_json():
    """Test handling of invalid JSON"""
    client = MockOllamaClient('invalid_json')
    analyzer = LLMMarketAnalyzer(client)
    data = create_test_data()
    
    result = analyzer.analyze_ohlcv(data, ticker='TEST')
    
    # Should return fallback response
    assert result['trend'] == 'neutral'
    assert result['strength'] == 5
    assert result['regime'] == 'unknown'
    assert 'error' in result


def test_empty_response():
    """Test handling of empty response"""
    client = MockOllamaClient('empty')
    analyzer = LLMMarketAnalyzer(client)
    data = create_test_data()
    
    result = analyzer.analyze_ohlcv(data, ticker='TEST')
    
    # Should return safe fallback
    assert result['trend'] == 'neutral'
    assert result['strength'] == 5
    assert 'error' in result


def test_no_json_in_response():
    """Test handling of plain text response with no JSON"""
    client = MockOllamaClient('no_json')
    analyzer = LLMMarketAnalyzer(client)
    data = create_test_data()
    
    result = analyzer.analyze_ohlcv(data, ticker='TEST')
    
    # Should return safe fallback
    assert result['trend'] == 'neutral'
    assert 'error' in result


def test_field_validation():
    """Test that invalid field values are corrected"""
    # This would require a custom mock response
    # Placeholder for now
    pass


if __name__ == '__main__':
    print("Running LLM parsing robustness tests...")
    
    tests = [
        ("Valid JSON", test_valid_json_response),
        ("Markdown wrapped JSON", test_markdown_wrapped_json),
        ("JSON with extra text", test_json_with_extra_text),
        ("Incomplete JSON", test_incomplete_json),
        ("Invalid JSON", test_invalid_json),
        ("Empty response", test_empty_response),
        ("No JSON in response", test_no_json_in_response),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

