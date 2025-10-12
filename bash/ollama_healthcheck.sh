#!/bin/bash

# Quick health check for Ollama
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "🔍 Quick Ollama Health Check"

# Check service
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null; then
    echo "✅ Ollama service is running"
    
    # Get model count
    models=$(curl -s "$OLLAMA_HOST/api/tags" | grep -o '"name":"[^"]*' | wc -l)
    echo "📊 Available models: $models"
    
    # Test basic functionality
    echo "🧪 Testing basic inference..."
    if curl -s -X POST "$OLLAMA_HOST/api/generate" \
        -H "Content-Type: application/json" \
        -d '{"model": "llama2", "prompt": "Hello", "stream": false}' | grep -q "response"; then
        echo "✅ Basic inference working"
    else
        echo "❌ Basic inference failed"
    fi
else
    echo "❌ Ollama service not available"
    echo "💡 Start with: ollama serve"
fi