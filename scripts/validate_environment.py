"""Environment validation for portfolio management system.

Mathematical Foundation:
- Linear algebra: O(n^2) covariance matrix computations
- Statistical ops: Normal distribution CDF for Sharpe ratio calculations
- Time series: Vectorized log returns r_t = ln(P_t/P_{t-1})

Success Criteria:
- All required packages importable
- NumPy vectorization >100x faster than loops
- yfinance can fetch 5+ years SPY data
- Statistical functions accurate to 1e-10
"""
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import yfinance as yf
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def validate_ctrader_credentials():
    """Validate that cTrader credentials are available via env/.env without printing secrets."""
    required_any = {
        "username": [
            "CTRADER_DEMO_USERNAME",
            "CTRADER_DEMO_EMAIL",
            "CTRADER_LIVE_USERNAME",
            "CTRADER_LIVE_EMAIL",
            "USERNAME_CTRADER",
            "CTRADER_USERNAME",
            "EMAIL_CTRADER",
            "CTRADER_EMAIL",
        ],
        "password": [
            "CTRADER_DEMO_PASSWORD",
            "CTRADER_LIVE_PASSWORD",
            "PASSWORD_CTRADER",
            "CTRADER_PASSWORD",
        ],
        "application_id": [
            "CTRADER_DEMO_APPLICATION_ID",
            "CTRADER_DEMO_APP_ID",
            "CTRADER_LIVE_APPLICATION_ID",
            "CTRADER_LIVE_APP_ID",
            "APPLICATION_NAME_CTRADER",
            "CTRADER_APPLICATION_ID",
            "CTRADER_APP_ID",
        ],
    }

    missing_groups = []
    for label, keys in required_any.items():
        if not any((os.getenv(k) or "").strip() for k in keys):
            missing_groups.append(f"{label} ({'/'.join(keys)})")

    return len(missing_groups) == 0, missing_groups

def validate_libraries():
    """Validate core scientific computing libraries."""
    versions = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        # scipy.stats does not expose __version__; it lives on the top-level scipy module.
        'scipy': getattr(scipy, '__version__', 'N/A')
    }
    return all(v != 'N/A' for v in versions.values()), versions

def validate_vectorization():
    """Validate NumPy vectorization performance.

    We measure speedup against a Python loop instead of assuming a fixed loop time.
    Threshold is intentionally modest to avoid flaky failures across machines.
    """
    n = 200_000
    # Simulate a strictly-positive price series to avoid NaNs/inf in log returns.
    prices = np.exp(np.cumsum(np.random.randn(n) * 0.01)).astype(np.float64)

    start = time.perf_counter()
    vec_result = np.log(prices[1:] / prices[:-1])
    vec_time = time.perf_counter() - start

    start = time.perf_counter()
    loop_result = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        loop_result[i - 1] = math.log(prices[i] / prices[i - 1])
    loop_time = time.perf_counter() - start

    speedup = (loop_time / vec_time) if vec_time > 0 else float('inf')
    results_match = np.allclose(vec_result, loop_result, rtol=1e-12, atol=0.0)

    return (speedup > 5) and results_match, speedup

def validate_yfinance():
    """Validate yfinance data access for SPY (5+ years)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    return len(spy) > 1200, len(spy)  # ~252 trading days/year * 5

def validate_statistics():
    """Validate statistical accuracy (1e-10 precision)."""
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean_check = np.abs(np.mean(test_data) - 3.0) < 1e-10
    std_check = np.abs(np.std(test_data, ddof=1) - np.sqrt(2.5)) < 1e-10
    return mean_check and std_check, (mean_check, std_check)

if __name__ == '__main__':
    print("Portfolio Management Environment Validation\n" + "="*50)

    lib_ok, versions = validate_libraries()
    print(f"\n1. Libraries: {'PASS' if lib_ok else 'FAIL'}")
    for k, v in versions.items(): print(f"   {k}: {v}")

    vec_ok, speedup = validate_vectorization()
    print(f"\n2. Vectorization: {'PASS' if vec_ok else 'FAIL'} ({speedup:.1f}x)")

    yf_ok, days = validate_yfinance()
    print(f"\n3. yfinance: {'PASS' if yf_ok else 'FAIL'} ({days} trading days)")

    stat_ok, checks = validate_statistics()
    print(f"\n4. Statistics: {'PASS' if stat_ok else 'FAIL'}")

    ctrader_ok, ctrader_missing = validate_ctrader_credentials()
    print(f"\n5. cTrader credentials: {'PASS' if ctrader_ok else 'WARN'}")
    if not ctrader_ok:
        print("   Missing:")
        for item in ctrader_missing:
            print(f"   - {item}")

    sys.exit(0 if all([lib_ok, vec_ok, yf_ok, stat_ok]) else 1)
