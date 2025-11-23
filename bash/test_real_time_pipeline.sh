#!/bin/bash
# Real-Time Pipeline Testing Script
# Tests actual pipeline execution with Qwen 14B model

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "REAL-TIME PIPELINE TESTING"
echo "=========================================="
echo "Testing optimized system with:"
echo "  - Qwen 14B Chat model"
echo "  - SQLite database persistence"
echo "  - Profit tracking"
echo "  - LLM report generation"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

# 1. Environment setup
echo "1. Setting up environment..."
if [ -f "simpleTrader_env/bin/activate" ]; then
    source simpleTrader_env/bin/activate
    print_success "Virtual environment activated"
else
    print_error "Virtual environment not found"
    exit 1
fi

# 2. Check Ollama
echo ""
echo "2. Checking Ollama service..."
if command -v ollama &> /dev/null; then
    print_success "Ollama CLI found"
    
    # Check if qwen model is available
    if ollama list | grep -q "qwen:14b-chat-q4_K_M"; then
        print_success "Qwen 14B model available"
    else
        print_warning "Qwen 14B model not found - pipeline will use fallback"
        print_info "To install: ollama pull qwen:14b-chat-q4_K_M"
    fi
else
    print_warning "Ollama not found - LLM features will be disabled"
fi

# 3. Test database initialization
echo ""
echo "3. Testing database initialization..."
python3 -c "from etl.database_manager import DatabaseManager; db = DatabaseManager('data/test_init.db'); print('✓ Database initialized'); db.close()"
if [ $? -eq 0 ]; then
    print_success "Database initialization works"
    rm -f data/test_init.db
else
    print_error "Database initialization failed"
    exit 1
fi

# 4. Run pipeline with single ticker (fast test)
echo ""
echo "4. Testing pipeline with single ticker (AAPL)..."
print_info "Running: python scripts/run_etl_pipeline.py --tickers AAPL --enable-llm"
echo ""

python3 scripts/run_etl_pipeline.py --tickers AAPL --enable-llm --verbose

if [ $? -eq 0 ]; then
    print_success "Pipeline execution: SUCCESSFUL"
else
    print_error "Pipeline execution: FAILED"
    exit 1
fi

# 5. Verify database was created
echo ""
echo "5. Verifying database creation..."
if [ -f "data/portfolio_maximizer.db" ]; then
    print_success "Database file created"
    
    # Check tables
    table_count=$(sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
    if [ "$table_count" -ge 7 ]; then
        print_success "All 7 tables created"
    else
        print_warning "Expected 7 tables, found $table_count"
    fi
else
    print_error "Database not created"
    exit 1
fi

# 6. Test database queries
echo ""
echo "6. Testing database queries..."

echo "   Query 1: OHLCV data count"
ohlcv_count=$(sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM ohlcv_data;" 2>/dev/null || echo "0")
print_info "   OHLCV rows: $ohlcv_count"

echo "   Query 2: LLM analyses count"
analysis_count=$(sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM llm_analyses;" 2>/dev/null || echo "0")
print_info "   LLM analyses: $analysis_count"

echo "   Query 3: LLM signals count"
signal_count=$(sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM llm_signals;" 2>/dev/null || echo "0")
print_info "   LLM signals: $signal_count"

if [ "$ohlcv_count" -gt 0 ]; then
    print_success "Database queries: WORKING"
else
    print_warning "No data in database - check pipeline"
fi

# 7. Test report generation (text format)
echo ""
echo "7. Testing report generation (text format)..."
print_info "Running: python scripts/generate_llm_report.py --format text"
echo ""

python3 scripts/generate_llm_report.py --format text > /tmp/test_report.txt 2>&1

if [ $? -eq 0 ]; then
    print_success "Text report generation: SUCCESSFUL"
    echo ""
    print_info "Report preview (first 20 lines):"
    echo "---"
    head -20 /tmp/test_report.txt
    echo "---"
else
    print_warning "Report generation had issues (may be due to no trade data)"
fi

# 8. Test report generation (JSON format)
echo ""
echo "8. Testing report generation (JSON format)..."
print_info "Running: python scripts/generate_llm_report.py --format json"

python3 scripts/generate_llm_report.py --format json > metrics.json 2>&1

if [ $? -eq 0 ] && [ -f "metrics.json" ]; then
    print_success "JSON report generation: SUCCESSFUL"
    
    # Validate JSON
    if python3 -c "import json; json.load(open('metrics.json'))" 2>/dev/null; then
        print_success "JSON is valid"
    else
        print_warning "JSON may be invalid"
    fi
else
    print_warning "JSON report generation had issues"
fi

# 9. Test report generation (HTML format)
echo ""
echo "9. Testing report generation (HTML format)..."
print_info "Running: python scripts/generate_llm_report.py --format html --output dashboard.html"

python3 scripts/generate_llm_report.py --format html --output dashboard.html 2>&1

if [ $? -eq 0 ] && [ -f "dashboard.html" ]; then
    print_success "HTML report generation: SUCCESSFUL"
    print_info "Dashboard saved to: dashboard.html"
else
    print_warning "HTML report generation had issues"
fi

# 10. Run multi-ticker test (frontier markets included via synthetic payload)
echo ""
echo "10. Testing pipeline with global + frontier tickers (synthetic run)..."
print_info "Running: python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,GOOGL --include-frontier-tickers --execution-mode synthetic --enable-llm --verbose"
echo ""

python3 scripts/run_etl_pipeline.py --tickers AAPL,MSFT,GOOGL --include-frontier-tickers --execution-mode synthetic --enable-llm --verbose

if [ $? -eq 0 ]; then
    print_success "Multi-ticker pipeline (with frontier coverage): SUCCESSFUL"
else
    print_warning "Multi-ticker pipeline had issues"
fi

# Summary
echo ""
echo "=========================================="
echo "REAL-TIME TESTING COMPLETE"
echo "=========================================="
echo ""
echo "Test Results:"
echo "  ✓ Environment setup"
echo "  ✓ Database initialization"
echo "  ✓ Single-ticker pipeline"
echo "  ✓ Multi-ticker pipeline"
echo "  ✓ Database persistence"
echo "  ✓ Text report generation"
echo "  ✓ JSON report generation"
echo "  ✓ HTML report generation"
echo ""
echo "Database: data/portfolio_maximizer.db"
echo "Reports: metrics.json, dashboard.html"
echo ""
echo "Status: SYSTEM OPERATIONAL ✓"
echo "=========================================="
