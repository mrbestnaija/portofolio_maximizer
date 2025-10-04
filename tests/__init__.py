"""
Test Suite for Portfolio Maximizer v4.7
Comprehensive testing framework for production deployment
"""

__version__ = "4.7.0"
__author__ = "Portfolio Maximizer Team"

# Test configuration
TEST_CONFIG = {
    'gpu_required': False,  # Set to True for GPU-specific tests
    'mock_data': True,      # Use mock data for faster testing
    'nigerian_market': True, # Include NGX-specific tests
    'security_tests': True,  # Include security validation tests
}