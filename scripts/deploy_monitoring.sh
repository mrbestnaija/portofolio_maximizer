#!/bin/bash
# Deploy Monitoring Systems
# Portfolio Maximizer v45 - Monitoring Deployment Script

set -e

echo "🚀 Deploying Portfolio Maximizer v45 Monitoring Systems..."

# Check if running from correct directory
if [ ! -f "scripts/run_etl_pipeline.py" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating monitoring directories..."
mkdir -p logs/alerts
mkdir -p logs/archive/errors
mkdir -p logs/archive/cache
mkdir -p config/monitoring

# Set permissions
echo "🔐 Setting permissions..."
for target in     "scripts/error_monitor.py"     "scripts/cache_manager.py"     "scripts/cleanup_cache.sh"     "scripts/deploy_monitoring.sh"
do
    if [ -f "" ]; then
        chmod +x ""
    else
        echo "⚠️ Warning:  not found, skipping chmod"
    fi
done

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Run initial cache cleanup
echo "🧹 Running initial cache cleanup..."
python scripts/cache_manager.py

# Validate system health
echo "🔍 Validating system health..."
python scripts/error_monitor.py --check-only

# Run method signature tests
echo "🧪 Running method signature validation tests..."
python -m pytest tests/etl/test_method_signature_validation.py -v

# Create systemd service files (if on Linux)
if command -v systemctl &> /dev/null; then
    echo "⚙️ Creating systemd service files..."
    
    # Error monitoring service
    cat > /tmp/portfolio-maximizer-error-monitor.service << EOF
[Unit]
Description=Portfolio Maximizer Error Monitor
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/scripts/error_monitor.py
Restart=always
RestartSec=300

[Install]
WantedBy=multi-user.target
EOF

    # Cache management service
    cat > /tmp/portfolio-maximizer-cache-manager.service << EOF
[Unit]
Description=Portfolio Maximizer Cache Manager
After=network.target

[Service]
Type=oneshot
User=www-data
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/scripts/cache_manager.py

[Install]
WantedBy=multi-user.target
EOF

    echo "📋 Systemd service files created in /tmp/"
    echo "   - portfolio-maximizer-error-monitor.service"
    echo "   - portfolio-maximizer-cache-manager.service"
    echo "   Copy these to /etc/systemd/system/ and run 'systemctl daemon-reload'"
fi

# Create cron jobs
echo "⏰ Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "# Portfolio Maximizer v45 Monitoring") | crontab -
(crontab -l 2>/dev/null; echo "0 * * * * cd $(pwd) && python scripts/error_monitor.py") | crontab -
(crontab -l 2>/dev/null; echo "0 2 * * * cd $(pwd) && python scripts/cache_manager.py") | crontab -
(crontab -l 2>/dev/null; echo "0 3 * * 0 cd $(pwd) && python scripts/cleanup_cache.sh") | crontab -

echo "✅ Cron jobs configured:"
echo "   - Error monitoring: Every hour"
echo "   - Cache management: Daily at 2 AM"
echo "   - Cache cleanup: Weekly on Sunday at 3 AM"

# Create monitoring dashboard script
echo "📊 Creating monitoring dashboard..."
cat > scripts/monitoring_dashboard.py << 'EOF'
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
EOF

chmod +x scripts/monitoring_dashboard.py

# Final validation
echo "🔍 Running final system validation..."
python scripts/monitoring_dashboard.py --test

echo ""
echo "🎉 Monitoring systems deployed successfully!"
echo ""
echo "📋 Next Steps:"
echo "   1. Review configuration in config/error_monitoring_config.yml"
echo "   2. Start monitoring: python scripts/monitoring_dashboard.py"
echo "   3. Check logs: tail -f logs/errors/errors.log"
echo "   4. Run tests: python -m pytest tests/etl/test_method_signature_validation.py -v"
echo ""
echo "📚 Documentation:"
echo "   - Error Monitoring Guide: Documentation/SYSTEM_ERROR_MONITORING_GUIDE.md"
echo "   - Error Fixes Summary: Documentation/ERROR_FIXES_SUMMARY_2025-10-22.md"
echo ""
echo "✅ Deployment complete!"
