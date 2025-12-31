#!/bin/bash

# Quick health check for Ollama
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OLLAMA_MODEL_ENV="${OLLAMA_MODEL:-}"

echo "🔍 Quick Ollama Health Check"

# Check service
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null; then
    echo "✅ Ollama service is running"

    # Get model count
    models=$(curl -s "$OLLAMA_HOST/api/tags" | grep -o '"name":"[^\"]*' | wc -l)
    echo "📊 Available models: $models"

    # Determine model to test
    if [ -n "$OLLAMA_MODEL_ENV" ]; then
        test_model="$OLLAMA_MODEL_ENV"
    else
        # Pick the first installed model name from /api/tags
        test_model=$(curl -s "$OLLAMA_HOST/api/tags" \
            | grep -o '"name":"[^\"]*"' \
            | head -n1 \
            | awk -F'"' '{print $4}')
    fi

    if [ -z "$test_model" ]; then
        echo "❌ No models installed. Pull one, e.g.:"
        echo "   ollama pull llama3"
        exit 1
    fi
    echo "🧭 Using model: $test_model"

    # Test basic functionality
    echo "🧪 Testing basic inference..."
    resp=$(curl -sS -X POST "$OLLAMA_HOST/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$test_model\", \"prompt\": \"Hello\", \"stream\": false}")

    if echo "$resp" | grep -q '"response"' && ! echo "$resp" | grep -q '"error"'; then
        echo "✅ Basic inference working"
    else
        echo "❌ Basic inference failed"
        # Show a short snippet of the response for debugging
        echo "↪ Response: $(echo "$resp" | head -c 300)"
        echo "💡 Tip: set OLLAMA_MODEL to a specific installed model, e.g.:"
        echo "   OLLAMA_MODEL=llama3:instruct ./bash/ollama_healthcheck.sh"
    fi
else
    echo "❌ Ollama service not available"
    echo "💡 Start with: ollama serve"
fi

