#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_MODELS=("llama2" "mistral")  # Add your preferred models

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Ollama is installed and running
check_ollama_service() {
    print_status "Checking Ollama service..."
    
    # Check if Ollama process is running
    if pgrep -x "ollama" > /dev/null; then
        print_success "Ollama process is running"
    else
        print_error "Ollama process not found"
        return 1
    fi

    # Check if Ollama API is accessible
    if curl -s "$OLLAMA_HOST/api/tags" > /dev/null; then
        print_success "Ollama API is accessible"
        return 0
    else
        print_error "Ollama API not accessible at $OLLAMA_HOST"
        return 1
    fi
}

# Check available models
check_models() {
    print_status "Checking available models..."
    
    local response
    response=$(curl -s "$OLLAMA_HOST/api/tags")
    
    if [ $? -ne 0 ]; then
        print_error "Failed to fetch models from Ollama"
        return 1
    fi

    local available_models
    available_models=$(echo "$response" | grep -o '"name":"[^"]*' | cut -d'"' -f4)
    
    if [ -z "$available_models" ]; then
        print_warning "No models found. Please pull models using: ollama pull llama2"
        return 1
    fi

    print_success "Available models:"
    echo "$available_models" | while read -r model; do
        echo "  - $model"
    done

    # Check if our test models are available
    for model in "${TEST_MODELS[@]}"; do
        if echo "$available_models" | grep -q "$model"; then
            print_success "Required model '$model' is available"
        else
            print_warning "Model '$model' not found"
        fi
    done
}

# Test basic model functionality
test_model_inference() {
    print_status "Testing model inference..."
    
    local available_models
    available_models=$(curl -s "$OLLAMA_HOST/api/tags" | grep -o '"name":"[^"]*' | cut -d'"' -f4 | head -1)
    
    if [ -z "$available_models" ]; then
        print_error "No models available for testing"
        return 1
    fi

    local test_model
    test_model=$(echo "$available_models" | head -1)
    
    print_status "Testing inference with model: $test_model"
    
    local response
    response=$(curl -s -X POST "$OLLAMA_HOST/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$test_model\",
            \"prompt\": \"Say 'Hello World' in a creative way.\",
            \"stream\": false
        }")
    
    if [ $? -eq 0 ] && echo "$response" | grep -q "response"; then
        local ai_response
        ai_response=$(echo "$response" | grep -o '"response":"[^"]*' | cut -d'"' -f4)
        print_success "Model inference test passed"
        echo -e "${GREEN}Model response:${NC} $ai_response"
        return 0
    else
        print_error "Model inference test failed"
        return 1
    fi
}

# Test Python modules
test_python_modules() {
    print_status "Testing Python module imports..."
    
    cd "$PROJECT_ROOT" || {
        print_error "Failed to change to project root"
        return 1
    }

    # Test core module imports
    local modules=(
        "ai_llm.ollama_client"
        "ai_llm.market_analyzer" 
        "ai_llm.signal_generator"
        "ai_llm.risk_assessor"
    )

    for module in "${modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            print_success "Module $module imported successfully"
        else
            print_error "Failed to import module $module"
            return 1
        fi
    done

    return 0
}

# Run unit tests
run_unit_tests() {
    print_status "Running LLM unit tests..."
    
    cd "$PROJECT_ROOT" || {
        print_error "Failed to change to project root"
        return 1
    }

    if [ -d "tests" ]; then
        if python3 -m pytest tests/ai_llm/ -v; then
            print_success "All LLM unit tests passed"
            return 0
        else
            print_error "Some LLM unit tests failed"
            return 1
        fi
    else
        print_warning "Tests directory not found"
        return 1
    fi
}

# Test configuration files
test_configuration() {
    print_status "Testing configuration files..."
    
    local config_files=(
        "config/llm_config.yml"
        "config/pipeline_config.yml" 
        "requirements-llm.txt"
    )

    for config_file in "${config_files[@]}"; do
        if [ -f "$PROJECT_ROOT/$config_file" ]; then
            print_success "Config file found: $config_file"
        else
            print_error "Config file missing: $config_file"
            return 1
        fi
    done

    # Test if requirements are installed
    if [ -f "$PROJECT_ROOT/requirements-llm.txt" ]; then
        print_status "Checking LLM requirements..."
        while IFS= read -r package; do
            [ -z "$package" ] && continue
            if python3 -c "import pkgutil; exit(0 if pkgutil.find_loader('${package%%=*}') else 1)" 2>/dev/null; then
                print_success "Package installed: ${package%%=*}"
            else
                print_error "Package not installed: ${package%%=*}"
                return 1
            fi
        done < "$PROJECT_ROOT/requirements-llm.txt"
    fi

    return 0
}

# Test documentation exists
test_documentation() {
    print_status "Checking documentation..."
    
    local docs=(
        "Documentation/LLM_INTEGRATION.md"
        "Documentation/PHASE_5_2_SUMMARY.md"
    )

    for doc in "${docs[@]}"; do
        if [ -f "$PROJECT_ROOT/$doc" ]; then
            print_success "Documentation found: $doc"
        else
            print_warning "Documentation missing: $doc"
        fi
    done
}

# Performance test
test_performance() {
    print_status "Running basic performance check..."
    
    local start_time
    start_time=$(date +%s)
    
    # Test a simple inference
    curl -s -X POST "$OLLAMA_HOST/api/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llama2",
            "prompt": "What is 2+2?",
            "stream": false
        }' > /dev/null
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $duration -lt 10 ]; then
        print_success "Basic inference completed in ${duration}s (acceptable)"
    else
        print_warning "Basic inference took ${duration}s (might be slow)"
    fi
}

# Main test function
main() {
    echo -e "${BLUE}=== Local LLM Integration Test ===${NC}"
    echo -e "${BLUE}Project Root: $PROJECT_ROOT${NC}"
    echo -e "${BLUE}Ollama Host: $OLLAMA_HOST${NC}"
    echo

    local tests_passed=0
    local tests_failed=0

    # Run tests
    if check_ollama_service; then ((tests_passed++)); else ((tests_failed++)); fi
    echo

    if check_models; then ((tests_passed++)); else ((tests_failed++)); fi
    echo

    if test_model_inference; then ((tests_passed++)); else ((tests_failed++)); fi
    echo

    if test_python_modules; then ((tests_passed++)); else ((tests_failed++)); fi
    echo

    if test_configuration; then ((tests_passed++)); else ((tests_failed++)); fi
    echo

    test_documentation
    echo

    test_performance
    echo

    # Run unit tests (optional - might take longer)
    read -p "Run unit tests? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if run_unit_tests; then ((tests_passed++)); else ((tests_failed++)); fi
    else
        print_warning "Skipping unit tests"
    fi

    # Summary
    echo
    echo -e "${BLUE}=== Test Summary ===${NC}"
    echo -e "${GREEN}Tests passed: $tests_passed${NC}"
    echo -e "${RED}Tests failed: $tests_failed${NC}"
    
    if [ $tests_failed -eq 0 ]; then
        echo -e "${GREEN}✅ All critical tests passed! LLM integration is working.${NC}"
    else
        echo -e "${YELLOW}⚠️  Some tests failed. Please check the issues above.${NC}"
    fi
}

# Handle script interruption
cleanup() {
    echo
    print_status "Test interrupted"
    exit 1
}

trap cleanup INT TERM

# Run main function
main "$@"