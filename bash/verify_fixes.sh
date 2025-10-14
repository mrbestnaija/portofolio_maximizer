#!/bin/bash
# Quick verification script for LLM fixes

set -e

echo "=========================================="
echo "🔍 Verifying LLM Integration Fixes"
echo "=========================================="
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi
echo "✅ Python3 available"

# Check pytest is available
if ! python3 -m pytest --version &> /dev/null; then
    echo "❌ pytest not found"
    exit 1
fi
echo "✅ pytest available"

# Run LLM unit tests
echo ""
echo "Running LLM unit tests..."
echo "------------------------------------------"
python3 -m pytest tests/ai_llm/test_ollama_client.py -v --tb=short

echo ""
echo "Running Market Analyzer tests..."
echo "------------------------------------------"
python3 -m pytest tests/ai_llm/test_market_analyzer.py -v --tb=short

echo ""
echo "Running LLM parsing robustness tests..."
echo "------------------------------------------"
python3 tests/ai_llm/test_llm_parsing.py

echo ""
echo "=========================================="
echo "✅ All fixes verified successfully!"
echo "=========================================="

