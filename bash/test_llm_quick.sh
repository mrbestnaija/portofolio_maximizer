#!/bin/bash
# Quick LLM Integration Test Runner
# Usage: bash bash/test_llm_quick.sh

echo "=========================================="
echo "🚀 Local LLM Integration Quick Test"
echo "=========================================="

# Check if Ollama is running
echo ""
echo "1️⃣  Checking Ollama service..."
if command -v ollama &> /dev/null; then
    echo "   ✅ Ollama CLI found"
    
    # Check if service is responding
    if ollama list &> /dev/null; then
        echo "   ✅ Ollama service is running"
    else
        echo "   ❌ Ollama service is not responding"
        echo "   💡 Start Ollama with: ollama serve"
        exit 1
    fi
else
    echo "   ❌ Ollama CLI not found"
    echo "   💡 Install from: https://ollama.ai"
    exit 1
fi

# Check for Python
echo ""
echo "2️⃣  Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "   ❌ Python not found"
    exit 1
fi

echo "   ✅ Python found: $($PYTHON_CMD --version)"

# Activate virtual environment if it exists
if [ -d "simpleTrader_env" ]; then
    echo ""
    echo "3️⃣  Activating virtual environment..."
    
    if [ -f "simpleTrader_env/bin/activate" ]; then
        source simpleTrader_env/bin/activate
        echo "   ✅ Virtual environment activated"
    elif [ -f "simpleTrader_env/Scripts/activate" ]; then
        source simpleTrader_env/Scripts/activate
        echo "   ✅ Virtual environment activated (Windows)"
    fi
fi

# Check required packages
echo ""
echo "4️⃣  Checking required packages..."
$PYTHON_CMD -c "import requests; import yaml; import pandas" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ Required packages installed"
else
    echo "   ⚠️  Some packages missing, installing..."
    pip install -q -r requirements-llm.txt
fi

# Run the test
echo ""
echo "5️⃣  Running LLM integration tests..."
echo "=========================================="
$PYTHON_CMD tests/ai_llm/test_integration_full.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
else
    echo "❌ Test failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

