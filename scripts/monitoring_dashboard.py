#!/usr/bin/env python3
"""
Portfolio Maximizer v45 - Monitoring Dashboard
Real-time monitoring dashboard for system health
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.error_monitor import ErrorMonitor
from scripts.cache_manager import CacheManager

def display_dashboard():
    """Display real-time monitoring dashboard"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("=" * 60)
    print("📊 Portfolio Maximizer v45 - Monitoring Dashboard")
    print("=" * 60)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Error monitoring status
    print("🚨 Error Monitoring Status:")
    print("-" * 30)
    error_monitor = ErrorMonitor()
    error_status = error_monitor.monitor_errors()
    
    if error_status['status'] == 'no_errors':
        print("✅ No errors detected")
    else:
        print(f"⚠️  Status: {error_status['status']}")
        if 'recent_errors' in error_status:
            recent = error_status['recent_errors']
            print(f"   Total errors (24h): {recent.get('total_errors', 0)}")
            print(f"   Error categories: {recent.get('error_categories', {})}")
    
    print()
    
    # Cache health status
    print("🧹 Cache Health Status:")
    print("-" * 30)
    cache_manager = CacheManager()
    cache_health = cache_manager.check_cache_health()
    
    if cache_health['status'] == 'healthy':
        print("✅ Cache is healthy")
    else:
        print(f"⚠️  Status: {cache_health['status']}")
        print(f"   Issues: {cache_health.get('total_issues', 0)}")
        print(f"   Warnings: {cache_health.get('total_warnings', 0)}")
    
    print()
    
    # System recommendations
    print("💡 Recommendations:")
    print("-" * 30)
    cache_stats = cache_manager._get_cache_statistics()
    if cache_stats.get('total_cache_size_mb', 0) > 50:
        print("   - Consider clearing large cache files")
    if cache_health.get('total_issues', 0) > 0:
        print("   - Clear stale cache files")
    if error_status.get('status') != 'no_errors':
        print("   - Investigate recent errors")
    
    print()
    print("Press Ctrl+C to exit")

def main():
    """Main dashboard function"""
    try:
        while True:
            display_dashboard()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed")

if __name__ == "__main__":
    main()
