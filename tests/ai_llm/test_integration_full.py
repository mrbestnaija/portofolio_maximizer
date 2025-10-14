#!/usr/bin/env python3
"""
Quick LLM Integration Test
Tests Ollama service, OllamaClient, and MarketAnalyzer
"""

import sys
import traceback
import pandas as pd
from datetime import datetime

def test_imports():
    """Test 1: Import all LLM modules"""
    print("\n" + "="*60)
    print("1️⃣  Testing Module Imports")
    print("="*60)
    
    try:
        from ai_llm.ollama_client import OllamaClient
        print("   ✅ ollama_client imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import ollama_client: {e}")
        return False
    
    try:
        from ai_llm.market_analyzer import LLMMarketAnalyzer
        print("   ✅ market_analyzer imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import market_analyzer: {e}")
        return False
    
    try:
        from ai_llm.signal_generator import LLMSignalGenerator
        print("   ✅ signal_generator imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import signal_generator: {e}")
        return False
    
    try:
        from ai_llm.risk_assessor import LLMRiskAssessor
        print("   ✅ risk_assessor imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import risk_assessor: {e}")
        return False
    
    return True


def test_ollama_client():
    """Test 2: OllamaClient basic functionality"""
    print("\n" + "="*60)
    print("2️⃣  Testing OllamaClient")
    print("="*60)
    
    try:
        from ai_llm.ollama_client import OllamaClient
        
        # Initialize client
        client = OllamaClient()
        print("   ✅ OllamaClient initialized")
        
        # Health check
        health = client.health_check()
        print(f"   ✅ Health check: {health}")
        
        if not health:
            print("   ⚠️  Warning: Health check returned False")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def test_basic_generation():
    """Test 3: Basic LLM generation"""
    print("\n" + "="*60)
    print("3️⃣  Testing Basic LLM Generation")
    print("="*60)
    
    try:
        from ai_llm.ollama_client import OllamaClient
        
        client = OllamaClient()
        
        # Simple test prompt
        prompt = "What is the capital of France? Answer in one word."
        print(f"   📝 Prompt: {prompt}")
        
        response = client.generate(prompt)
        print(f"   ✅ Response: {response[:100]}...")
        print(f"   📊 Response length: {len(response)} characters")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def test_market_analyzer():
    """Test 4: LLMMarketAnalyzer functionality"""
    print("\n" + "="*60)
    print("4️⃣  Testing LLMMarketAnalyzer")
    print("="*60)
    
    try:
        from ai_llm.ollama_client import OllamaClient
        from ai_llm.market_analyzer import LLMMarketAnalyzer
        
        # Initialize with client
        client = OllamaClient()
        analyzer = LLMMarketAnalyzer(client)
        print("   ✅ LLMMarketAnalyzer initialized")
        
        # Create realistic market data with DatetimeIndex (60 days for proper indicators)
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'Open': [99.0 + i*0.5 for i in range(60)],
            'High': [103.0 + i*0.5 for i in range(60)],
            'Low': [98.0 + i*0.5 for i in range(60)],
            'Close': [100.0 + i*0.5 for i in range(60)],
            'Volume': [1000000 + i*10000 for i in range(60)]
        }, index=dates)
        print("   ✅ Realistic market data created (60 days with DatetimeIndex)")
        
        # Analyze OHLCV
        analysis = analyzer.analyze_ohlcv(data, ticker='TEST')
        print(f"   ✅ OHLCV analysis completed")
        print(f"   📊 Analysis type: {type(analysis)}")
        print(f"   📝 Keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'Not a dict'}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def test_signal_generator():
    """Test 5: LLMSignalGenerator functionality"""
    print("\n" + "="*60)
    print("5️⃣  Testing LLMSignalGenerator")
    print("="*60)
    
    try:
        from ai_llm.ollama_client import OllamaClient
        from ai_llm.market_analyzer import LLMMarketAnalyzer
        from ai_llm.signal_generator import LLMSignalGenerator
        
        # Initialize
        client = OllamaClient()
        analyzer = LLMMarketAnalyzer(client)
        generator = LLMSignalGenerator(client)
        print("   ✅ LLMSignalGenerator initialized")
        
        # Create realistic data with DatetimeIndex
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'Open': [99.0 + i*0.5 for i in range(60)],
            'High': [103.0 + i*0.5 for i in range(60)],
            'Low': [98.0 + i*0.5 for i in range(60)],
            'Close': [100.0 + i*0.5 for i in range(60)],
            'Volume': [1000000 + i*10000 for i in range(60)]
        }, index=dates)
        print("   ✅ Realistic market data created (60 days)")
        
        # First get market analysis (required for signal generation)
        print("   ⏳ Getting market analysis first...")
        market_analysis = analyzer.analyze_ohlcv(data, ticker='AAPL')
        print("   ✅ Market analysis obtained")
        
        # Generate signal with market analysis
        signal = generator.generate_signal(data, ticker='AAPL', market_analysis=market_analysis)
        print(f"   ✅ Signal generated")
        print(f"   📊 Signal type: {type(signal)}")
        print(f"   📝 Action: {signal.get('action', 'unknown')}")
        print(f"   📝 Confidence: {signal.get('confidence', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def test_risk_assessor():
    """Test 6: LLMRiskAssessor functionality"""
    print("\n" + "="*60)
    print("6️⃣  Testing LLMRiskAssessor")
    print("="*60)
    
    try:
        from ai_llm.ollama_client import OllamaClient
        from ai_llm.risk_assessor import LLMRiskAssessor
        
        # Initialize with client
        client = OllamaClient()
        assessor = LLMRiskAssessor(client)
        print("   ✅ LLMRiskAssessor initialized")
        
        # Create realistic market data with DatetimeIndex (60 days for volatility calcs)
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'Open': [99.0 + i*0.5 for i in range(60)],
            'High': [103.0 + i*0.5 for i in range(60)],
            'Low': [98.0 + i*0.5 for i in range(60)],
            'Close': [100.0 + i*0.5 for i in range(60)],
            'Volume': [1000000 + i*10000 for i in range(60)]
        }, index=dates)
        print("   ✅ Realistic market data created (60 days with DatetimeIndex)")
        
        # Assess risk
        assessment = assessor.assess_risk(data, ticker='AAPL', portfolio_weight=0.40)
        print(f"   ✅ Risk assessment completed")
        print(f"   📊 Assessment type: {type(assessment)}")
        print(f"   📝 Keys: {list(assessment.keys()) if isinstance(assessment, dict) else str(assessment)[:150]}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 LOCAL LLM INTEGRATION TEST SUITE")
    print("="*60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Module Imports": test_imports(),
        "OllamaClient": test_ollama_client(),
        "Basic Generation": test_basic_generation(),
        "MarketAnalyzer": test_market_analyzer(),
        "SignalGenerator": test_signal_generator(),
        "RiskAssessor": test_risk_assessor()
    }
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print("\n" + "="*60)
    print(f"📈 Results: {passed_tests}/{total_tests} tests passed")
    print("="*60)
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - LLM Integration is fully operational!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure Ollama service is running: ollama list")
        print("   2. Check model is available: ollama run deepseek-coder:6.7b-instruct-q4_K_M")
        print("   3. Verify LLM config: cat config/llm_config.yml")
        return 1


if __name__ == "__main__":
    sys.exit(main())

