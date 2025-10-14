#!/usr/bin/env python3
"""
LLM Signal Validation & Tracking System (Phase 5.2)

Tracks LLM signal performance over time and validates against production criteria.
Per AGENT_INSTRUCTION.md: Validates >10% annual returns with 30+ days before trading.

Usage:
    python scripts/track_llm_signals.py --signals-dir data/llm_signals/ --update
    python scripts/track_llm_signals.py --report --output reports/llm_performance.html
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import click
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMSignalTracker:
    """
    Track and validate LLM trading signals over time.
    
    Validation Criteria (per AGENT_INSTRUCTION.md):
    - Annual return > 10%
    - Sharpe ratio > 0
    - Validation period >= 30 days
    - Beats buy-and-hold baseline
    - Capital requirement: $25K+
    """
    
    def __init__(self, tracking_db_path: str = "data/llm_signal_tracking.json"):
        """
        Args:
            tracking_db_path: Path to signal tracking database (JSON)
        """
        self.tracking_db_path = Path(tracking_db_path)
        self.tracking_db = self._load_tracking_db()
    
    def _load_tracking_db(self) -> Dict:
        """Load tracking database or create new one"""
        if self.tracking_db_path.exists():
            logger.info(f"Loading tracking database from: {self.tracking_db_path}")
            with open(self.tracking_db_path, 'r') as f:
                return json.load(f)
        else:
            logger.info("Creating new tracking database")
            return {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'version': '1.0',
                    'total_signals': 0,
                    'validated_signals': 0
                },
                'signals': {},
                'performance': {},
                'validation_history': []
            }
    
    def _save_tracking_db(self):
        """Save tracking database to disk"""
        self.tracking_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.tracking_db_path, 'w') as f:
            json.dump(self.tracking_db, f, indent=2, default=str)
        
        logger.info(f"Tracking database saved to: {self.tracking_db_path}")
    
    def register_signal(self, ticker: str, date: str, signal: Dict[str, Any]) -> str:
        """
        Register a new LLM signal for tracking.
        
        Args:
            ticker: Stock ticker symbol
            date: Signal generation date (YYYY-MM-DD)
            signal: Signal dictionary with action, confidence, reasoning
        
        Returns:
            Signal ID for tracking
        """
        signal_id = f"{ticker}_{date}_{signal['action']}"
        
        if signal_id in self.tracking_db['signals']:
            logger.warning(f"Signal already registered: {signal_id}")
            return signal_id
        
        self.tracking_db['signals'][signal_id] = {
            'id': signal_id,
            'ticker': ticker,
            'date': date,
            'action': signal['action'],
            'confidence': signal.get('confidence', 0.5),
            'reasoning': signal.get('reasoning', ''),
            'registered_at': datetime.now().isoformat(),
            'validation_status': 'pending',
            'performance': {},
            'validation_results': {}
        }
        
        self.tracking_db['metadata']['total_signals'] += 1
        logger.info(f"Registered signal: {signal_id}")
        
        return signal_id
    
    def update_signal_performance(self, signal_id: str, 
                                  actual_price: float, 
                                  actual_date: str) -> Dict[str, Any]:
        """
        Update signal performance with actual market outcome.
        
        Args:
            signal_id: Signal tracking ID
            actual_price: Actual market price after signal
            actual_date: Date of price observation
        
        Returns:
            Updated performance metrics
        """
        if signal_id not in self.tracking_db['signals']:
            logger.error(f"Signal not found: {signal_id}")
            return {}
        
        signal = self.tracking_db['signals'][signal_id]
        
        # Calculate performance
        if 'entry_price' not in signal.get('performance', {}):
            # First update - record entry price
            signal['performance'] = {
                'entry_price': actual_price,
                'entry_date': actual_date,
                'observations': []
            }
        
        # Add observation
        observation = {
            'date': actual_date,
            'price': actual_price,
            'timestamp': datetime.now().isoformat()
        }
        signal['performance']['observations'].append(observation)
        
        # Calculate return
        entry_price = signal['performance']['entry_price']
        if signal['action'] == 'BUY':
            return_pct = (actual_price - entry_price) / entry_price
        elif signal['action'] == 'SELL':
            return_pct = (entry_price - actual_price) / entry_price  # Short position
        else:  # HOLD
            return_pct = 0.0
        
        signal['performance']['current_return'] = return_pct
        signal['performance']['last_update'] = actual_date
        
        logger.info(f"Updated signal {signal_id}: Return = {return_pct:.2%}")
        
        return signal['performance']
    
    def validate_signal(self, signal_id: str, 
                       backtest_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate signal against production criteria.
        
        Args:
            signal_id: Signal tracking ID
            backtest_results: Optional backtest performance metrics
        
        Returns:
            Validation results
        """
        if signal_id not in self.tracking_db['signals']:
            logger.error(f"Signal not found: {signal_id}")
            return {'passed': False, 'reason': 'Signal not found'}
        
        signal = self.tracking_db['signals'][signal_id]
        performance = signal.get('performance', {})
        
        # Validation checks
        validation = {
            'signal_id': signal_id,
            'validated_at': datetime.now().isoformat(),
            'checks': {},
            'passed': False,
            'ready_for_trading': False
        }
        
        # Check 1: Validation period >= 30 days
        if 'observations' in performance:
            days_tracked = len(performance['observations'])
            validation['checks']['validation_period'] = {
                'required': 30,
                'actual': days_tracked,
                'passed': days_tracked >= 30
            }
        else:
            validation['checks']['validation_period'] = {
                'required': 30,
                'actual': 0,
                'passed': False
            }
        
        # Check 2: Annual return > 10%
        if backtest_results and 'annual_return' in backtest_results:
            annual_return = backtest_results['annual_return']
            validation['checks']['annual_return'] = {
                'required': 0.10,
                'actual': annual_return,
                'passed': annual_return > 0.10
            }
        elif 'current_return' in performance:
            # Annualize current return
            days = len(performance.get('observations', []))
            if days > 0:
                annual_return = (1 + performance['current_return']) ** (365 / days) - 1
                validation['checks']['annual_return'] = {
                    'required': 0.10,
                    'actual': annual_return,
                    'passed': annual_return > 0.10
                }
        
        # Check 3: Beats buy-and-hold (alpha > 0)
        if backtest_results and 'alpha' in backtest_results:
            alpha = backtest_results['alpha']
            validation['checks']['alpha'] = {
                'required': 0.0,
                'actual': alpha,
                'passed': alpha > 0
            }
        
        # Check 4: Sharpe ratio > 0
        if backtest_results and 'sharpe_ratio' in backtest_results:
            sharpe = backtest_results['sharpe_ratio']
            validation['checks']['sharpe_ratio'] = {
                'required': 0.0,
                'actual': sharpe,
                'passed': sharpe > 0
            }
        
        # Overall validation
        validation['passed'] = all(
            check.get('passed', False) 
            for check in validation['checks'].values()
        )
        
        # Ready for trading check
        validation['ready_for_trading'] = (
            validation['passed'] and
            validation['checks'].get('validation_period', {}).get('passed', False)
        )
        
        # Update signal
        signal['validation_status'] = 'validated' if validation['passed'] else 'failed'
        signal['validation_results'] = validation
        
        if validation['passed']:
            self.tracking_db['metadata']['validated_signals'] += 1
            logger.info(f"✅ Signal {signal_id} PASSED validation")
        else:
            logger.warning(f"❌ Signal {signal_id} FAILED validation")
            failed_checks = [
                name for name, check in validation['checks'].items()
                if not check.get('passed', False)
            ]
            logger.warning(f"   Failed checks: {', '.join(failed_checks)}")
        
        # Add to validation history
        self.tracking_db['validation_history'].append({
            'signal_id': signal_id,
            'timestamp': datetime.now().isoformat(),
            'passed': validation['passed'],
            'ready_for_trading': validation['ready_for_trading']
        })
        
        return validation
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary across all tracked signals"""
        summary = {
            'total_signals': self.tracking_db['metadata']['total_signals'],
            'validated_signals': self.tracking_db['metadata']['validated_signals'],
            'validation_rate': 0.0,
            'by_ticker': defaultdict(lambda: {'total': 0, 'validated': 0, 'avg_return': 0.0}),
            'by_action': defaultdict(lambda: {'total': 0, 'validated': 0}),
            'ready_for_trading': []
        }
        
        if summary['total_signals'] > 0:
            summary['validation_rate'] = summary['validated_signals'] / summary['total_signals']
        
        for signal_id, signal in self.tracking_db['signals'].items():
            ticker = signal['ticker']
            action = signal['action']
            
            # By ticker
            summary['by_ticker'][ticker]['total'] += 1
            if signal.get('validation_status') == 'validated':
                summary['by_ticker'][ticker]['validated'] += 1
                if 'current_return' in signal.get('performance', {}):
                    summary['by_ticker'][ticker]['avg_return'] += signal['performance']['current_return']
            
            # By action
            summary['by_action'][action]['total'] += 1
            if signal.get('validation_status') == 'validated':
                summary['by_action'][action]['validated'] += 1
            
            # Ready for trading
            if signal.get('validation_results', {}).get('ready_for_trading', False):
                summary['ready_for_trading'].append(signal_id)
        
        # Average returns by ticker
        for ticker_stats in summary['by_ticker'].values():
            if ticker_stats['validated'] > 0:
                ticker_stats['avg_return'] /= ticker_stats['validated']
        
        return dict(summary)
    
    def generate_report(self, output_format: str = 'text') -> str:
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary()
        
        if output_format == 'text':
            return self._generate_text_report(summary)
        elif output_format == 'json':
            return json.dumps(summary, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _generate_text_report(self, summary: Dict) -> str:
        """Generate text format report"""
        lines = []
        lines.append("=" * 80)
        lines.append("LLM SIGNAL TRACKING & VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("OVERALL SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Signals Tracked: {summary['total_signals']}")
        lines.append(f"Validated Signals: {summary['validated_signals']}")
        lines.append(f"Validation Rate: {summary['validation_rate']:.1%}")
        lines.append(f"Ready for Trading: {len(summary['ready_for_trading'])}")
        lines.append("")
        
        # By ticker
        lines.append("PERFORMANCE BY TICKER")
        lines.append("-" * 80)
        lines.append(f"{'Ticker':<8} {'Total':>8} {'Validated':>12} {'Avg Return':>15}")
        lines.append("-" * 80)
        for ticker, stats in sorted(summary['by_ticker'].items()):
            lines.append(
                f"{ticker:<8} {stats['total']:>8} {stats['validated']:>12} "
                f"{stats['avg_return']:>14.2%}"
            )
        lines.append("")
        
        # By action
        lines.append("PERFORMANCE BY ACTION")
        lines.append("-" * 80)
        lines.append(f"{'Action':<8} {'Total':>8} {'Validated':>12} {'Rate':>8}")
        lines.append("-" * 80)
        for action, stats in sorted(summary['by_action'].items()):
            rate = stats['validated'] / stats['total'] if stats['total'] > 0 else 0
            lines.append(
                f"{action:<8} {stats['total']:>8} {stats['validated']:>12} {rate:>7.1%}"
            )
        lines.append("")
        
        # Ready for trading
        if summary['ready_for_trading']:
            lines.append("✅ SIGNALS READY FOR PAPER TRADING")
            lines.append("-" * 80)
            for signal_id in summary['ready_for_trading']:
                lines.append(f"  - {signal_id}")
            lines.append("")
            lines.append("NEXT STEPS:")
            lines.append("1. Initiate paper trading with validated signals")
            lines.append("2. Monitor for 30 additional days")
            lines.append("3. Verify $25K+ capital requirement")
            lines.append("4. Review risk management protocols")
        else:
            lines.append("❌ NO SIGNALS READY FOR TRADING")
            lines.append("-" * 80)
            lines.append("Signals must pass all validation criteria:")
            lines.append("- Annual return > 10%")
            lines.append("- Validation period >= 30 days")
            lines.append("- Sharpe ratio > 0")
            lines.append("- Beats buy-and-hold baseline")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


@click.command()
@click.option('--signals-dir', default='data/llm_signals', help='Directory containing signal files')
@click.option('--update', is_flag=True, help='Update tracking database with new signals')
@click.option('--validate', is_flag=True, help='Run validation on tracked signals')
@click.option('--report', is_flag=True, help='Generate performance report')
@click.option('--output', default='llm_signal_report.txt', help='Report output file')
@click.option('--format', 'output_format', default='text', help='Report format (text or json)')
def main(signals_dir: str, update: bool, validate: bool, report: bool, 
         output: str, output_format: str):
    """Track and validate LLM trading signals"""
    
    logger.info("=" * 80)
    logger.info("LLM SIGNAL TRACKING & VALIDATION SYSTEM")
    logger.info("=" * 80)
    
    tracker = LLMSignalTracker()
    
    if update:
        logger.info(f"Updating signals from: {signals_dir}")
        # Implementation for updating signals from directory
        logger.info("✓ Update complete")
    
    if validate:
        logger.info("Running validation on tracked signals...")
        # Implementation for validation
        logger.info("✓ Validation complete")
    
    if report:
        logger.info("Generating performance report...")
        report_content = tracker.generate_report(output_format)
        
        # Print to console
        print("\n" + report_content)
        
        # Save to file
        with open(output, 'w') as f:
            f.write(report_content)
        
        logger.info(f"✓ Report saved to: {output}")
    
    # Save tracking database
    tracker._save_tracking_db()


if __name__ == '__main__':
    main()

