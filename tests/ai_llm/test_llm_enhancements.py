"""
Test LLM Enhancements
Tests for performance monitoring, signal validation, database integration, and optimization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ai_llm.performance_monitor import LLMPerformanceMonitor, monitor_inference
from ai_llm.signal_quality_validator import SignalQualityValidator, Signal, SignalDirection
from ai_llm.llm_database_integration import LLMDatabaseManager, LLMSignal, LLMRiskAssessment
from ai_llm.performance_optimizer import LLMPerformanceOptimizer, optimize_model_selection


class TestPerformanceMonitor:
    """Test LLM performance monitoring"""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        monitor = LLMPerformanceMonitor(max_history=100)
        assert monitor.max_history == 100
        assert len(monitor.metrics_history) == 0
    
    def test_record_inference_success(self):
        """Test recording successful inference"""
        monitor = LLMPerformanceMonitor()
        
        metrics = monitor.record_inference(
            model_name="test-model",
            prompt="Test prompt",
            response="Test response",
            inference_time=5.0,
            success=True
        )
        
        assert metrics.model_name == "test-model"
        assert metrics.inference_time == 5.0
        assert metrics.success is True
        assert len(monitor.metrics_history) == 1
    
    def test_record_inference_failure(self):
        """Test recording failed inference"""
        monitor = LLMPerformanceMonitor()
        
        metrics = monitor.record_inference(
            model_name="test-model",
            prompt="Test prompt",
            response="",
            inference_time=30.0,
            success=False,
            error_message="Timeout"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Timeout"
        assert metrics.inference_time == 30.0
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        monitor = LLMPerformanceMonitor()
        
        # Add some test metrics
        for i in range(5):
            monitor.record_inference(
                model_name="test-model",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                inference_time=5.0 + i,
                success=True
            )
        
        summary = monitor.get_performance_summary(24)
        
        assert summary["total_inferences"] == 5
        assert summary["successful_inferences"] == 5
        assert summary["success_rate"] == 1.0
        assert "avg_inference_time" in summary
        assert "model_breakdown" in summary


class TestSignalQualityValidator:
    """Test signal quality validation"""
    
    def test_signal_validation_initialization(self):
        """Test signal validator initialization"""
        validator = SignalQualityValidator()
        assert validator.min_confidence_threshold == 0.6
        assert validator.max_risk_threshold == 0.15
    
    def test_validate_signal_success(self):
        """Test successful signal validation"""
        validator = SignalQualityValidator()
        
        signal = Signal(
            ticker="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.8,
            reasoning="Strong technical indicators and positive market sentiment",
            timestamp=datetime.now(),
            price_at_signal=150.0,
            expected_return=0.05,
            risk_estimate=0.10
        )
        
        # Create mock market data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        market_data = pd.DataFrame({
            'close': np.random.normal(150, 5, 30)
        }, index=dates)
        
        result = validator.validate_signal(signal, market_data)
        
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.warnings, list)
        assert isinstance(result.quality_metrics, dict)
        assert result.recommendation in ["STRONG_BUY", "BUY", "WEAK_BUY", "WEAK_SELL", "SELL", "STRONG_SELL", "HOLD"]
    
    def test_validate_signal_low_confidence(self):
        """Test signal validation with low confidence"""
        validator = SignalQualityValidator()
        
        signal = Signal(
            ticker="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.3,  # Below threshold
            reasoning="Uncertain",
            timestamp=datetime.now(),
            price_at_signal=150.0
        )
        
        market_data = pd.DataFrame({'close': [150.0]}, index=[datetime.now()])
        result = validator.validate_signal(signal, market_data)
        
        assert not result.is_valid
        assert any("Low confidence" in warning for warning in result.warnings)
    
    def test_backtest_signal_quality(self):
        """Test signal quality backtesting"""
        validator = SignalQualityValidator()
        
        # Create test signals
        signals = [
            Signal(
                ticker="AAPL",
                direction=SignalDirection.BUY,
                confidence=0.8,
                reasoning="Test",
                timestamp=datetime.now() - timedelta(days=1),
                price_at_signal=150.0
            )
        ]
        
        # Create test market data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        market_data = pd.DataFrame({
            'close': np.linspace(150, 160, 10)  # Upward trend
        }, index=dates)
        
        results = validator.backtest_signal_quality(signals, market_data, 30)
        
        assert "total_signals" in results
        assert "avg_confidence" in results
        assert "signal_accuracy" in results


class TestLLMDatabaseIntegration:
    """Test LLM database integration"""
    
    def test_database_manager_initialization(self):
        """Test database manager initialization"""
        with patch('sqlite3.connect'):
            db_manager = LLMDatabaseManager("test.db")
            assert db_manager.db_path == "test.db"
    
    def test_llm_signal_creation(self):
        """Test LLM signal creation"""
        signal = LLMSignal(
            id=None,
            ticker="AAPL",
            signal_type="BUY",
            confidence=0.8,
            reasoning="Strong technical indicators",
            expected_return=0.05,
            risk_estimate=0.10,
            model_used="qwen:14b-chat-q4_K_M",
            timestamp=datetime.now(),
            market_data_snapshot={"price": 150.0, "volume": 1000000}
        )
        
        assert signal.ticker == "AAPL"
        assert signal.signal_type == "BUY"
        assert signal.confidence == 0.8
        assert signal.model_used == "qwen:14b-chat-q4_K_M"
    
    def test_risk_assessment_creation(self):
        """Test risk assessment creation"""
        assessment = LLMRiskAssessment(
            id=None,
            portfolio_id="portfolio_001",
            risk_score=0.3,
            risk_factors=["High volatility", "Market uncertainty"],
            recommendations=["Reduce position size", "Add hedging"],
            model_used="qwen:14b-chat-q4_K_M",
            timestamp=datetime.now(),
            market_conditions={"volatility": 0.25, "trend": "bearish"},
            confidence=0.85
        )
        
        assert assessment.portfolio_id == "portfolio_001"
        assert assessment.risk_score == 0.3
        assert len(assessment.risk_factors) == 2
        assert len(assessment.recommendations) == 2


class TestPerformanceOptimizer:
    """Test LLM performance optimization"""
    
    def test_optimizer_initialization(self):
        """Test performance optimizer initialization"""
        optimizer = LLMPerformanceOptimizer()
        assert len(optimizer.model_characteristics) == 3
        assert "qwen:14b-chat-q4_K_M" in optimizer.model_characteristics
        assert "deepseek-coder:6.7b-instruct-q4_K_M" in optimizer.model_characteristics
    
    def test_update_model_performance(self):
        """Test updating model performance"""
        optimizer = LLMPerformanceOptimizer()
        
        optimizer.update_model_performance(
            model_name="test-model",
            inference_time=10.0,
            tokens_per_second=15.0,
            success=True,
            accuracy_score=0.8
        )
        
        assert "test-model" in optimizer.model_performance
        performance = optimizer.model_performance["test-model"]
        assert performance.avg_inference_time == 10.0
        assert performance.avg_tokens_per_second == 15.0
        assert performance.success_rate == 1.0
        assert performance.accuracy_score == 0.8
    
    def test_get_optimal_model_fast(self):
        """Test getting optimal model for fast use case"""
        optimizer = LLMPerformanceOptimizer()
        
        # Add some performance data
        optimizer.update_model_performance("model1", 5.0, 20.0, True, 0.7)
        optimizer.update_model_performance("model2", 15.0, 10.0, True, 0.9)
        
        result = optimizer.get_optimal_model("fast")
        
        assert result.recommended_model in ["model1", "model2"]
        assert result.expected_inference_time >= 0
        assert result.expected_accuracy >= 0
        assert isinstance(result.alternative_models, list)
    
    def test_optimize_for_task(self):
        """Test task-based optimization"""
        optimizer = LLMPerformanceOptimizer()
        
        # Add performance data
        optimizer.update_model_performance("fast-model", 5.0, 25.0, True, 0.6)
        optimizer.update_model_performance("accurate-model", 20.0, 8.0, True, 0.9)
        
        # Test real-time task
        result = optimizer.optimize_for_task("real-time trading signals")
        assert result.recommended_model in ["fast-model", "accurate-model"]
        
        # Test analysis task
        result = optimizer.optimize_for_task("comprehensive market analysis")
        assert result.recommended_model in ["fast-model", "accurate-model"]


class TestIntegration:
    """Test integration between components"""
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        monitor = LLMPerformanceMonitor()
        
        # Simulate multiple inferences
        for i in range(10):
            monitor.record_inference(
                model_name="qwen:14b-chat-q4_K_M",
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                inference_time=5.0 + i * 0.5,
                success=True
            )
        
        summary = monitor.get_performance_summary(24)
        assert summary["total_inferences"] == 10
        assert summary["success_rate"] == 1.0
    
    def test_signal_validation_integration(self):
        """Test signal validation integration"""
        validator = SignalQualityValidator()
        
        # Create a realistic signal
        signal = Signal(
            ticker="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.75,
            reasoning="Strong technical indicators showing bullish momentum with RSI at 45 and price above 20-day moving average",
            timestamp=datetime.now(),
            price_at_signal=150.0,
            expected_return=0.08,
            risk_estimate=0.12
        )
        
        # Create realistic market data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        np.random.seed(42)  # For reproducible results
        market_data = pd.DataFrame({
            'close': 150 + np.cumsum(np.random.normal(0, 2, 50))
        }, index=dates)
        
        result = validator.validate_signal(signal, market_data)
        
        # Should be valid with good confidence
        assert result.is_valid is True
        assert result.confidence_score > 0.5
        assert len(result.quality_metrics) == 5  # 5 validation layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
