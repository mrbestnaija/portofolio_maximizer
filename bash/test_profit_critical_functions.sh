#!/bin/bash
# Test Profit-Critical Functions (Per AGENT_INSTRUCTION.md)
# Maximum test complexity per phase guidelines
# Focus: Business logic that affects money

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "PROFIT-CRITICAL FUNCTION TESTS"
echo "=========================================="
echo "Per AGENT_INSTRUCTION.md:"
echo "- Test ONLY profit-affecting functions"
echo "- Maximum 500 lines Phase 4-6"
echo "- Focus on money calculations"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. Activate virtual environment
echo "1. Activating virtual environment..."
if [ -f "simpleTrader_env/bin/activate" ]; then
    source simpleTrader_env/bin/activate
    print_success "Virtual environment activated"
else
    print_error "Virtual environment not found"
    exit 1
fi

# 2. Check Python and pytest
echo ""
echo "2. Checking dependencies..."
python3 --version || { print_error "Python3 not found"; exit 1; }
python3 -c "import pytest" 2>/dev/null || { print_error "pytest not installed"; exit 1; }
print_success "Dependencies OK"

# 3. Run profit-critical database tests
echo ""
echo "3. Testing profit-critical database functions..."
echo "   (Profit calculations, win rate, profit factor)"
python3 -m pytest tests/integration/test_profit_critical_functions.py::TestProfitCriticalDatabaseFunctions -v

if [ $? -eq 0 ]; then
    print_success "Database profit calculations: PASSED"
else
    print_error "Database profit calculations: FAILED"
    exit 1
fi

# 4. Run MVS criteria validation tests
echo ""
echo "4. Testing MVS (Minimum Viable System) criteria..."
python3 -m pytest tests/integration/test_profit_critical_functions.py::TestMVSCriteriaValidation -v

if [ $? -eq 0 ]; then
    print_success "MVS criteria validation: PASSED"
else
    print_error "MVS criteria validation: FAILED"
    exit 1
fi

# 5. Run report generation tests
echo ""
echo "5. Testing profit report generation accuracy..."
python3 -m pytest tests/integration/test_llm_report_generation.py::TestProfitReportAccuracy -v

if [ $? -eq 0 ]; then
    print_success "Profit report generation: PASSED"
else
    print_error "Profit report generation: FAILED"
    exit 1
fi

# 6. Run performance tests
echo ""
echo "6. Testing system performance requirements..."
python3 -m pytest tests/integration/test_profit_critical_functions.py::TestSystemPerformanceRequirements -v

if [ $? -eq 0 ]; then
    print_success "Performance requirements: PASSED"
else
    print_warning "Performance requirements: Review needed"
fi

# 7. Run full test suite
echo ""
echo "7. Running FULL profit-critical test suite..."
python3 -m pytest tests/integration/test_profit_critical_functions.py \
                  tests/integration/test_llm_report_generation.py \
                  -v --tb=short

if [ $? -eq 0 ]; then
    print_success "ALL PROFIT-CRITICAL TESTS PASSED ✓"
else
    print_error "SOME TESTS FAILED - Review output above"
    exit 1
fi

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "✓ Profit calculation accuracy: VERIFIED"
echo "✓ Win rate calculation: VERIFIED"
echo "✓ Profit factor: VERIFIED"
echo "✓ MVS criteria validation: VERIFIED"
echo "✓ Report generation: VERIFIED"
echo "✓ Database persistence: VERIFIED"
echo ""
echo "Status: READY FOR REAL-TIME TESTING"
echo "=========================================="

