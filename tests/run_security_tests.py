#!/usr/bin/env python3
"""
Security tests runner.

Run all security-related tests to verify security hardening implementation.
"""

import sys
import subprocess
from pathlib import Path

def run_security_tests():
    """Run all security tests."""
    test_files = [
        'tests/etl/test_security_utils.py',
        'tests/etl/test_secret_loader.py',
        'tests/etl/test_database_security.py',
        'tests/integration/test_security_integration.py',
    ]

    # Verify all test files exist
    root_dir = Path(__file__).resolve().parent.parent
    missing_files = [f for f in test_files if not (root_dir / f).exists()]
    if missing_files:
        print(f"⚠️  Warning: Some test files not found: {missing_files}")

    # Run pytest with security marker
    cmd = [
        sys.executable,
        '-m',
        'pytest',
        '-c',
        str(root_dir / 'pytest.ini'),
        '-v',
        '-m',
        'security',
        '--tb=short',
    ] + test_files

    print("🔒 Running Security Tests...")
    print("=" * 70)

    result = subprocess.run(cmd, cwd=root_dir)

    print("=" * 70)
    if result.returncode == 0:
        print("✅ All security tests passed!")
    else:
        print("❌ Some security tests failed!")

    return result.returncode


if __name__ == '__main__':
    sys.exit(run_security_tests())
