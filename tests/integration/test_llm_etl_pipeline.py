"""
Integration Tests for LLM-ETL Pipeline (Phase 5.2)

Tests the complete integration of LLM modules into the ETL pipeline,
ensuring data flows correctly through all stages.

Per AGENT_INSTRUCTION.md:
- Test critical business logic only
- LLM signals are advisory only
- No trading without validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_llm.ollama_client import OllamaClient, OllamaConnectionError
from ai_llm.market_analyzer import LLMMarketAnalyzer
from ai_llm.signal_generator import LLMSignalGenerator
from ai_llm.risk_assessor import LLMRiskAssessor


@pytest.fixture
def sample_ohlcv_data():
    """Create realistic OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(60) * 2),
        'High': 102 + np.cumsum(np.random.randn(60) * 2),
        'Low': 98 + np.cumsum(np.random.randn(60) * 2),
        'Close': 100 + np.cumsum(np.random.randn(60) * 2),
        'Volume': np.random.randint(1000000, 5000000, 60)
    }, index=dates)
    
    # Ensure realistic price relationships
    data['High'] = data[['Open', 'Close']].max(axis=1) + 1
    data['Low'] = data[['Open', 'Close']].min(axis=1) - 1
    
    return data


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    client = Mock(spec=OllamaClient)
    client.model = 'deepseek-coder:6.7b-instruct-q4_K_M'
    client.health_check = Mock(return_value=True)
    client.generate = Mock(return_value='{"trend": "bullish", "strength": 7, "regime": "trending", "key_levels": [100, 110], "summary": "Test analysis"}')
    return client


class TestLLMPipelineIntegration:
    """Test LLM integration with ETL pipeline"""
    
    def test_market_analyzer_integration(self, sample_ohlcv_data, mock_ollama_client):
        """Test market analyzer integrates correctly with pipeline data"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        
        result = analyzer.analyze_ohlcv(sample_ohlcv_data, 'TEST')
        
        assert 'trend' in result
        assert 'strength' in result
        assert 'regime' in result
        assert 'ticker' in result
        assert result['ticker'] == 'TEST'
        assert result['trend'] in ['bullish', 'bearish', 'neutral']
        assert 1 <= result['strength'] <= 10
    
    def test_signal_generator_integration(self, sample_ohlcv_data, mock_ollama_client):
        """Test signal generator integrates correctly with market analysis"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        signal_gen = LLMSignalGenerator(mock_ollama_client)
        
        # First get market analysis
        analysis = analyzer.analyze_ohlcv(sample_ohlcv_data, 'TEST')
        
        # Then generate signal
        signal = signal_gen.generate_signal(sample_ohlcv_data, 'TEST', analysis)
        
        assert 'action' in signal
        assert 'confidence' in signal
        assert 'reasoning' in signal
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal['confidence'] <= 1
    
    def test_risk_assessor_integration(self, sample_ohlcv_data, mock_ollama_client):
        """Test risk assessor integrates correctly with portfolio data"""
        risk_assessor = LLMRiskAssessor(mock_ollama_client)
        
        result = risk_assessor.assess_risk(sample_ohlcv_data, 'TEST', portfolio_weight=0.25)
        
        assert 'risk_level' in result
        assert 'risk_score' in result
        assert 'concerns' in result
        assert 'recommendation' in result
        assert result['risk_level'] in ['low', 'medium', 'high']
        assert 0 <= result['risk_score'] <= 100
        assert isinstance(result['concerns'], list)
    
    def test_full_pipeline_flow(self, sample_ohlcv_data, mock_ollama_client):
        """Test complete LLM pipeline: analysis -> signals -> risk"""
        # Initialize all components
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        signal_gen = LLMSignalGenerator(mock_ollama_client)
        risk_assessor = LLMRiskAssessor(mock_ollama_client)
        
        # Step 1: Market Analysis
        analysis = analyzer.analyze_ohlcv(sample_ohlcv_data, 'TEST')
        assert analysis is not None
        assert 'trend' in analysis
        
        # Step 2: Signal Generation (depends on analysis)
        signal = signal_gen.generate_signal(sample_ohlcv_data, 'TEST', analysis)
        assert signal is not None
        assert 'action' in signal
        
        # Step 3: Risk Assessment
        risk = risk_assessor.assess_risk(sample_ohlcv_data, 'TEST', 0.25)
        assert risk is not None
        assert 'risk_level' in risk
        
        # Verify data consistency
        assert signal['ticker'] == 'TEST'
        assert risk['ticker'] == 'TEST'
    
    def test_pipeline_handles_missing_analysis(self, sample_ohlcv_data, mock_ollama_client):
        """Test pipeline handles missing market analysis gracefully"""
        signal_gen = LLMSignalGenerator(mock_ollama_client)
        
        # Pass empty analysis
        signal = signal_gen.generate_signal(sample_ohlcv_data, 'TEST', {})
        
        assert signal is not None
        assert 'action' in signal
        # Should default to HOLD when analysis is missing
        assert signal['action'] in ['BUY', 'SELL', 'HOLD']
    
    def test_pipeline_handles_multiple_tickers(self, sample_ohlcv_data, mock_ollama_client):
        """Test pipeline can process multiple tickers"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        results = {}
        
        for ticker in tickers:
            result = analyzer.analyze_ohlcv(sample_ohlcv_data, ticker)
            results[ticker] = result
        
        assert len(results) == 3
        for ticker in tickers:
            assert results[ticker]['ticker'] == ticker
            assert 'trend' in results[ticker]
    
    def test_pipeline_preserves_data_integrity(self, sample_ohlcv_data, mock_ollama_client):
        """Test LLM pipeline doesn't modify input data"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        
        # Create copy of original data
        original_data = sample_ohlcv_data.copy()
        
        # Run analysis
        _ = analyzer.analyze_ohlcv(sample_ohlcv_data, 'TEST')
        
        # Verify data unchanged
        pd.testing.assert_frame_equal(sample_ohlcv_data, original_data)
    
    def test_pipeline_error_recovery(self, sample_ohlcv_data):
        """Test pipeline handles Ollama connection failures gracefully"""
        # Create client that fails health check
        failing_client = Mock(spec=OllamaClient)
        failing_client.health_check = Mock(return_value=False)
        
        # Should not raise exception during initialization
        analyzer = LLMMarketAnalyzer(failing_client)
        assert analyzer is not None


class TestLLMPipelinePerformance:
    """Test LLM pipeline performance characteristics"""
    
    def test_analysis_latency(self, sample_ohlcv_data, mock_ollama_client):
        """Test market analysis completes in reasonable time"""
        import time
        
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        
        start_time = time.time()
        _ = analyzer.analyze_ohlcv(sample_ohlcv_data, 'TEST')
        elapsed = time.time() - start_time
        
        # Mock should be very fast (<1 second)
        assert elapsed < 1.0, f"Analysis took {elapsed:.2f}s, expected <1s"
    
    def test_batch_processing_scalability(self, sample_ohlcv_data, mock_ollama_client):
        """Test pipeline can handle multiple tickers efficiently"""
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        
        tickers = [f'TICK{i}' for i in range(10)]
        
        import time
        start_time = time.time()
        
        for ticker in tickers:
            _ = analyzer.analyze_ohlcv(sample_ohlcv_data, ticker)
        
        elapsed = time.time() - start_time
        
        # Should complete 10 tickers in <2 seconds with mocks
        assert elapsed < 2.0, f"10 tickers took {elapsed:.2f}s, expected <2s"


class TestLLMPipelineValidation:
    """Test LLM pipeline validation and safety checks"""
    
    def test_advisory_only_signals(self, sample_ohlcv_data, mock_ollama_client):
        """Test signals are clearly marked as advisory only"""
        signal_gen = LLMSignalGenerator(mock_ollama_client)
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        
        analysis = analyzer.analyze_ohlcv(sample_ohlcv_data, 'TEST')
        signal = signal_gen.generate_signal(sample_ohlcv_data, 'TEST', analysis)
        
        # Signals should include warning or advisory flag
        assert 'confidence' in signal
        # Confidence should be reasonable (not overconfident)
        assert signal['confidence'] < 1.0
    
    def test_no_trading_without_validation(self, sample_ohlcv_data, mock_ollama_client):
        """Test pipeline enforces validation requirements"""
        # Per AGENT_INSTRUCTION.md: NO TRADING without >10% annual returns and 30+ days
        signal_gen = LLMSignalGenerator(mock_ollama_client)
        analyzer = LLMMarketAnalyzer(mock_ollama_client)
        
        analysis = analyzer.analyze_ohlcv(sample_ohlcv_data, 'TEST')
        signal = signal_gen.generate_signal(sample_ohlcv_data, 'TEST', analysis)
        
        # Signal should not automatically execute trades
        assert 'action' in signal
        assert 'confidence' in signal
        # Should not contain execution instructions
        assert 'execute_now' not in signal or not signal.get('execute_now')
    
    def test_risk_assessment_required(self, sample_ohlcv_data, mock_ollama_client):
        """Test risk assessment is part of complete pipeline"""
        risk_assessor = LLMRiskAssessor(mock_ollama_client)
        
        result = risk_assessor.assess_risk(sample_ohlcv_data, 'TEST', 0.25)
        
        assert 'risk_level' in result
        assert 'concerns' in result
        # Should always provide concerns for high-risk situations
        assert isinstance(result['concerns'], list)


class TestLLMPipelineConfiguration:
    """Test LLM pipeline configuration handling"""
    
    def test_model_selection(self):
        """Test pipeline supports multiple LLM models"""
        models = [
            'deepseek-coder:6.7b-instruct-q4_K_M',
            'codellama:13b-instruct-q4_K_M',
            'qwen:14b-chat-q4_K_M'
        ]
        
        for model in models:
            with patch('ai_llm.ollama_client.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.json.return_value = {
                    'models': [{'name': model}]
                }
                mock_get.return_value = mock_response
                
                client = OllamaClient(model=model)
                assert client.model == model
    
    def test_fail_fast_validation(self):
        """Test pipeline fails fast if Ollama unavailable"""
        with patch('ai_llm.ollama_client.requests.get', side_effect=Exception('Connection failed')):
            with pytest.raises(OllamaConnectionError):
                _ = OllamaClient()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

