#!/bin/bash
# Configuration-Driven Cross-Validation Test Script
# Demonstrates reading CV parameters from pipeline_config.yml

echo "========================================================================"
echo "CONFIGURATION-DRIVEN K-FOLD CROSS-VALIDATION TEST"
echo "========================================================================"
echo ""

# Activate virtual environment
source ./simpleTrader_env/bin/activate

echo "Test 1: Using default config values (k=5, test_size=0.15)"
echo "------------------------------------------------------------------------"
python scripts/run_etl_pipeline.py \
  --tickers AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --use-cv \
  2>&1 | grep -A 8 "Using k-fold"

echo ""
echo "Test 2: Override with CLI parameters (k=7, test_size=0.2, gap=1)"
echo "------------------------------------------------------------------------"
python scripts/run_etl_pipeline.py \
  --tickers AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --use-cv \
  --n-splits 7 \
  --test-size 0.2 \
  --gap 1 \
  2>&1 | grep -A 8 "Using k-fold"

echo ""
echo "Test 3: No --use-cv flag (uses default_strategy from config)"
echo "------------------------------------------------------------------------"
python scripts/run_etl_pipeline.py \
  --tickers AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  2>&1 | grep -A 5 "Using simple"

echo ""
echo "========================================================================"
echo "Configuration File Location: config/pipeline_config.yml"
echo "========================================================================"
echo ""
echo "To change defaults, edit:"
echo "  data_split:"
echo "    default_strategy: 'simple'  # Change to 'cv' for k-fold by default"
echo "    cross_validation:"
echo "      n_splits: 5               # Change fold count"
echo "      test_size: 0.15           # Change test set size"
echo "      gap: 0                    # Change temporal gap"
echo ""
echo "✅ CONFIGURATION-DRIVEN CV TESTS COMPLETE"
echo "========================================================================"
