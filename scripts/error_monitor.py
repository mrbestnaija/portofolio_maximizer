#!/usr/bin/env python3
"""
Enhanced Error Monitoring System
Monitors system errors and provides automated alerting
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class ErrorMonitor:
    """
    Enhanced error monitoring system with automated alerting
    """
    
    def __init__(self, config_file: str = "config/error_monitoring_config.yml"):
        self.config_file = config_file
        self.error_thresholds = {
            'max_errors_per_hour': 5,
            'max_errors_per_day': 20,
            'critical_error_types': ['TypeError', 'ValueError', 'ConnectionError'],
            'alert_cooldown_minutes': 30
        }
        self.last_alert_time = None
        self.error_counts = {}
        
    def monitor_errors(self) -> Dict[str, any]:
        """
        Monitor current error status and generate alerts if needed
        """
        try:
            # Check error log
            error_log_path = "logs/errors/errors.log"
            if not os.path.exists(error_log_path):
                return {"status": "no_errors", "message": "No error log found"}
            
            # Analyze recent errors
            recent_errors = self._analyze_recent_errors(error_log_path)
            
            # Check error thresholds
            alert_needed = self._check_error_thresholds(recent_errors)
            
            # Generate alert if needed
            if alert_needed:
                self._send_alert(recent_errors)
            
            return {
                "status": "monitored",
                "recent_errors": recent_errors,
                "alert_sent": alert_needed,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_recent_errors(self, error_log_path: str) -> Dict[str, any]:
        """Analyze recent errors from log file"""
        try:
            with open(error_log_path, 'r') as f:
                lines = f.readlines()
            
            # Get errors from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_errors = []
            
            for line in lines:
                if line.strip():
                    try:
                        # Parse timestamp from log line
                        timestamp_str = line.split(' - ')[0]
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        if timestamp >= cutoff_time:
                            recent_errors.append({
                                'timestamp': timestamp.isoformat(),
                                'line': line.strip()
                            })
                    except:
                        # Skip lines that don't match expected format
                        continue
            
            # Categorize errors
            error_categories = {}
            for error in recent_errors:
                line = error['line']
                if 'ERROR' in line:
                    # Extract error type
                    if 'TypeError' in line:
                        error_categories['TypeError'] = error_categories.get('TypeError', 0) + 1
                    elif 'ValueError' in line:
                        error_categories['ValueError'] = error_categories.get('ValueError', 0) + 1
                    elif 'ConnectionError' in line:
                        error_categories['ConnectionError'] = error_categories.get('ConnectionError', 0) + 1
                    else:
                        error_categories['Other'] = error_categories.get('Other', 0) + 1
            
            return {
                'total_errors': len(recent_errors),
                'error_categories': error_categories,
                'errors': recent_errors[-10:]  # Last 10 errors
            }
            
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return {'total_errors': 0, 'error_categories': {}, 'errors': []}
    
    def _check_error_thresholds(self, recent_errors: Dict[str, any]) -> bool:
        """Check if error thresholds are exceeded"""
        total_errors = recent_errors.get('total_errors', 0)
        
        # Check hourly threshold
        if total_errors > self.error_thresholds['max_errors_per_hour']:
            return True
        
        # Check for critical error types
        error_categories = recent_errors.get('error_categories', {})
        for critical_type in self.error_thresholds['critical_error_types']:
            if error_categories.get(critical_type, 0) > 0:
                return True
        
        return False
    
    def _send_alert(self, recent_errors: Dict[str, any]):
        """Send alert notification"""
        try:
            # Check cooldown period
            if self.last_alert_time:
                time_since_last = datetime.now() - self.last_alert_time
                if time_since_last.total_seconds() < self.error_thresholds['alert_cooldown_minutes'] * 60:
                    return  # Still in cooldown period
            
            # Create alert message
            alert_message = self._create_alert_message(recent_errors)
            
            # Log alert
            logger.warning(f"ERROR ALERT: {alert_message}")
            
            # Save alert to file
            self._save_alert_to_file(alert_message, recent_errors)
            
            # Update last alert time
            self.last_alert_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _create_alert_message(self, recent_errors: Dict[str, any]) -> str:
        """Create alert message"""
        total_errors = recent_errors.get('total_errors', 0)
        error_categories = recent_errors.get('error_categories', {})
        
        message = f"""
🚨 ERROR ALERT - Portfolio Maximizer v45
=====================================

Timestamp: {datetime.now().isoformat()}
Total Errors (24h): {total_errors}

Error Breakdown:
"""
        
        for error_type, count in error_categories.items():
            message += f"  - {error_type}: {count}\n"
        
        if recent_errors.get('errors'):
            message += "\nRecent Errors:\n"
            for error in recent_errors['errors'][-3:]:  # Last 3 errors
                message += f"  - {error['timestamp']}: {error['line'][:100]}...\n"
        
        message += f"\nSystem Status: {'CRITICAL' if total_errors > 10 else 'WARNING'}"
        
        return message
    
    def _save_alert_to_file(self, alert_message: str, recent_errors: Dict[str, any]):
        """Save alert to file"""
        alert_file = f"logs/alerts/error_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)
        
        with open(alert_file, 'w') as f:
            f.write(alert_message)
            f.write(f"\n\nDetailed Error Data:\n")
            f.write(json.dumps(recent_errors, indent=2))
    
    def generate_error_report(self, days: int = 7) -> Dict[str, any]:
        """Generate comprehensive error report"""
        try:
            error_log_path = "logs/errors/errors.log"
            if not os.path.exists(error_log_path):
                return {"error": "No error log found"}
            
            with open(error_log_path, 'r') as f:
                lines = f.readlines()
            
            # Analyze errors over specified period
            cutoff_time = datetime.now() - timedelta(days=days)
            errors_by_day = {}
            error_types = {}
            
            for line in lines:
                if line.strip():
                    try:
                        timestamp_str = line.split(' - ')[0]
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        if timestamp >= cutoff_time:
                            day_key = timestamp.strftime('%Y-%m-%d')
                            errors_by_day[day_key] = errors_by_day.get(day_key, 0) + 1
                            
                            # Categorize error types
                            if 'TypeError' in line:
                                error_types['TypeError'] = error_types.get('TypeError', 0) + 1
                            elif 'ValueError' in line:
                                error_types['ValueError'] = error_types.get('ValueError', 0) + 1
                            elif 'ConnectionError' in line:
                                error_types['ConnectionError'] = error_types.get('ConnectionError', 0) + 1
                            else:
                                error_types['Other'] = error_types.get('Other', 0) + 1
                    except:
                        continue
            
            return {
                "period_days": days,
                "total_errors": sum(errors_by_day.values()),
                "errors_by_day": errors_by_day,
                "error_types": error_types,
                "average_errors_per_day": sum(errors_by_day.values()) / max(len(errors_by_day), 1),
                "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else "None",
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main error monitoring function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = ErrorMonitor()
    
    print("🔍 Starting Enhanced Error Monitoring...")
    
    # Monitor current errors
    result = monitor.monitor_errors()
    print(f"Monitoring Result: {result['status']}")
    
    if result.get('alert_sent'):
        print("🚨 Alert sent due to error threshold exceeded")
    
    # Generate error report
    print("\n📊 Generating Error Report...")
    report = monitor.generate_error_report(7)
    
    if 'error' in report:
        print(f"❌ Report generation failed: {report['error']}")
    else:
        print(f"✅ Error Report Generated:")
        print(f"  - Total Errors (7 days): {report['total_errors']}")
        print(f"  - Average per day: {report['average_errors_per_day']:.1f}")
        print(f"  - Most common error: {report['most_common_error']}")
    
    print("\n✅ Error monitoring complete")


if __name__ == "__main__":
    main()
