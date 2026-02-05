#!/usr/bin/env python3
"""
Test LLM Implementations
Simple test script to verify all new LLM enhancements work
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_performance_monitor():
    """Test performance monitoring functionality."""
    from ai_llm.performance_monitor import LLMPerformanceMonitor

    monitor = LLMPerformanceMonitor()
    metrics = monitor.record_inference(
        model_name="test-model",
        prompt="Test prompt",
        response="Test response",
        inference_time=5.0,
        success=True,
    )

    assert metrics.model_name == "test-model"
    assert metrics.success is True

    summary = monitor.get_performance_summary(24)
    assert summary["total_inferences"] == 1
    assert summary["successful_inferences"] == 1


def test_signal_validator():
    """Test signal validation functionality."""
    from ai_llm.signal_quality_validator import SignalQualityValidator, Signal, SignalDirection
    import pandas as pd
    import numpy as np
    from datetime import datetime

    validator = SignalQualityValidator()

    signal = Signal(
        ticker="AAPL",
        direction=SignalDirection.BUY,
        confidence=0.8,
        reasoning="Strong technical indicators and positive market sentiment for testing",
        timestamp=datetime.now(),
        price_at_signal=150.0,
        expected_return=0.05,
        risk_estimate=0.10,
    )

    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    market_data = pd.DataFrame({"close": np.random.normal(150, 5, 30)}, index=dates)

    result = validator.validate_signal(signal, market_data)

    assert result.is_valid is True
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.recommendation.startswith(("STRONG", "BUY"))


def test_database_integration():
    """Test database integration functionality."""
    from ai_llm.llm_database_integration import LLMDatabaseManager, LLMSignal
    from datetime import datetime
    import tempfile

    db_path = Path(tempfile.mkdtemp()) / "llm_test.db"
    db_manager = LLMDatabaseManager(str(db_path))

    signal = LLMSignal(
        id=None,
        ticker="AAPL",
        signal_type="BUY",
        confidence=0.8,
        reasoning="Test signal",
        expected_return=0.05,
        risk_estimate=0.10,
        model_used="test-model",
        timestamp=datetime.now(),
        market_data_snapshot={"price": 150.0},
    )

    signal_id = db_manager.save_llm_signal(signal)
    assert signal_id is not None
    assert signal_id >= 0


def test_performance_optimizer():
    """Test performance optimization functionality."""
    from ai_llm.performance_optimizer import LLMPerformanceOptimizer

    optimizer = LLMPerformanceOptimizer()

    optimizer.update_model_performance(
        model_name="test-model",
        inference_time=10.0,
        tokens_per_second=15.0,
        success=True,
        accuracy_score=0.8,
    )

    result = optimizer.get_optimal_model("fast")
    assert result.recommended_model == "test-model"
    assert result.expected_inference_time <= 10.0


def test_ollama_client_integration():
    """Test Ollama client integration (expects connection failure)."""
    from ai_llm.ollama_client import OllamaClient, OllamaConnectionError

    with pytest.raises(OllamaConnectionError):
        OllamaClient(host="http://127.0.0.1:65535", timeout=1)

def main():
    """Run all tests"""
    print("🚀 Testing LLM System Enhancements")
    print("=" * 50)

    tests = [
        test_performance_monitor,
        test_signal_validator,
        test_database_integration,
        test_performance_optimizer,
        test_ollama_client_integration
    ]

    results = []
    for test in tests:
        try:
            test()
            results.append(True)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All LLM enhancements are working correctly!")
        return 0
    else:
        print("⚠️ Some tests failed, but core functionality is available")
        return 1

if __name__ == "__main__":
    sys.exit(main())
