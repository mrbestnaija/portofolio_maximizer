#!/usr/bin/env python3
"""
Comprehensive LLM Performance Reporting System (Phase 5.2+)

Generates standardized reports for all LLM tasks:
- Market Analysis Performance
- Signal Generation Accuracy
- Risk Assessment Quality
- Quantifiable Profit/Loss Metrics

Usage:
    python scripts/generate_llm_report.py --period monthly --output reports/
    python scripts/generate_llm_report.py --ticker AAPL --detailed
"""

import sys
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import click
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMPerformanceReporter:
    """
    Generate comprehensive performance reports for LLM outputs.

    Quantifiable Success Criteria:
    1. PROFIT METRICS (Primary):
       - Total Profit: Net P&L from all trades
       - Profit per Trade: Average profit/loss per execution
       - Win Rate: % of profitable trades
       - Profit Factor: Gross profit / Gross loss
       - ROI: Return on invested capital

    2. RISK-ADJUSTED RETURNS:
       - Sharpe Ratio: (Return - Risk-free) / Volatility
       - Sortino Ratio: Return / Downside deviation
       - Max Drawdown: Largest peak-to-trough decline
       - Calmar Ratio: Annual return / Max drawdown

    3. ALPHA GENERATION:
       - Alpha: Excess return vs benchmark (S&P 500)
       - Beta: Correlation with market
       - Information Ratio: Alpha / Tracking error

    4. SIGNAL QUALITY:
       - Signal Accuracy: % correct directional predictions
       - Signal Latency: Time to generate signals
       - Confidence Calibration: Accuracy vs confidence correlation

    5. OPERATIONAL METRICS:
       - Total Trades: Number of executions
       - Holding Period: Average days per trade
       - Commission Impact: Total fees / Total return
    """

    def __init__(self, db_path: str = "data/portfolio_maximizer.db"):
        """Initialize reporter with database connection"""
        self.db = DatabaseManager(db_path)
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    def generate_profit_report(self, start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict:
        """
        Generate quantifiable profit metrics report.

        Returns:
            Dictionary with profit/loss metrics
        """
        logger.info("Generating profit/loss report...")

        performance = self.db.get_performance_summary(start_date, end_date)

        report = {
            'report_type': 'profit_loss',
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start_date': start_date or 'inception',
                'end_date': end_date or datetime.now().strftime('%Y-%m-%d')
            },
            'metrics': {
                # Primary profit metrics
                'total_profit_usd': performance.get('total_profit', 0.0),
                'avg_profit_per_trade_usd': performance.get('avg_profit_per_trade', 0.0),
                'largest_win_usd': performance.get('largest_win', 0.0),
                'largest_loss_usd': performance.get('smallest_loss', 0.0),

                # Win/loss statistics
                'total_trades': performance.get('total_trades', 0),
                'winning_trades': performance.get('winning_trades', 0),
                'losing_trades': performance.get('losing_trades', 0),
                'win_rate_pct': performance.get('win_rate', 0.0) * 100,

                # Quality metrics
                'profit_factor': performance.get('profit_factor', 0.0),
                'avg_win_usd': performance.get('avg_win', 0.0),
                'avg_loss_usd': performance.get('avg_loss', 0.0),
                'win_loss_ratio': abs(performance.get('avg_win', 0.0) / performance.get('avg_loss', 1.0)) if performance.get('avg_loss') else 0.0
            },
            'success_criteria': {
                'profitability': performance.get('total_profit', 0) > 0,
                'win_rate_above_50': performance.get('win_rate', 0.0) > 0.50,
                'profit_factor_above_1': performance.get('profit_factor', 0.0) > 1.0,
                'sufficient_trades': performance.get('total_trades', 0) >= 30
            }
        }

        # Overall assessment
        report['overall_status'] = all(report['success_criteria'].values())

        return report

    def generate_signal_accuracy_report(self, ticker: Optional[str] = None) -> Dict:
        """
        Generate signal accuracy and quality report.

        Returns:
            Dictionary with signal performance metrics
        """
        logger.info("Generating signal accuracy report...")

        # Query validated signals
        query = """
            SELECT
                ticker,
                action,
                confidence,
                backtest_annual_return,
                backtest_sharpe,
                backtest_alpha,
                validation_status,
                actual_return
            FROM llm_signals
            WHERE validation_status IN ('validated', 'executed')
        """

        if ticker:
            query += f" AND ticker = '{ticker}'"

        self.db.cursor.execute(query)
        signals = [dict(row) for row in self.db.cursor.fetchall()]

        if not signals:
            return {'error': 'No validated signals found'}

        df = pd.DataFrame(signals)

        report = {
            'report_type': 'signal_accuracy',
            'generated_at': datetime.now().isoformat(),
            'total_signals': len(signals),
            'metrics': {
                # Accuracy metrics
                'signals_by_action': df['action'].value_counts().to_dict(),
                'avg_confidence': float(df['confidence'].mean()),
                'avg_annual_return_pct': float(df['backtest_annual_return'].mean() * 100) if 'backtest_annual_return' in df else 0.0,
                'avg_sharpe_ratio': float(df['backtest_sharpe'].mean()) if 'backtest_sharpe' in df else 0.0,
                'avg_alpha_pct': float(df['backtest_alpha'].mean() * 100) if 'backtest_alpha' in df else 0.0,

                # Signal quality
                'high_confidence_signals': int((df['confidence'] >= 0.7).sum()),
                'medium_confidence_signals': int(((df['confidence'] >= 0.5) & (df['confidence'] < 0.7)).sum()),
                'low_confidence_signals': int((df['confidence'] < 0.5).sum()),
            },
            'success_criteria': {
                'avg_return_above_10pct': (df['backtest_annual_return'].mean() if 'backtest_annual_return' in df else 0) > 0.10,
                'avg_sharpe_positive': (df['backtest_sharpe'].mean() if 'backtest_sharpe' in df else 0) > 0,
                'positive_alpha': (df['backtest_alpha'].mean() if 'backtest_alpha' in df else 0) > 0
            }
        }

        return report

    def generate_risk_assessment_report(self) -> Dict:
        """
        Generate risk assessment quality report.

        Returns:
            Dictionary with risk metrics
        """
        logger.info("Generating risk assessment report...")

        query = """
            SELECT
                ticker,
                risk_level,
                risk_score,
                portfolio_weight,
                var_95,
                max_drawdown,
                volatility
            FROM llm_risks
            ORDER BY assessment_date DESC
            LIMIT 100
        """

        self.db.cursor.execute(query)
        risks = [dict(row) for row in self.db.cursor.fetchall()]

        if not risks:
            return {'error': 'No risk assessments found'}

        df = pd.DataFrame(risks)

        report = {
            'report_type': 'risk_assessment',
            'generated_at': datetime.now().isoformat(),
            'total_assessments': len(risks),
            'metrics': {
                'risk_distribution': df['risk_level'].value_counts().to_dict(),
                'avg_risk_score': float(df['risk_score'].mean()),
                'high_risk_tickers': int((df['risk_level'] == 'high').sum()),
                'medium_risk_tickers': int((df['risk_level'] == 'medium').sum()),
                'low_risk_tickers': int((df['risk_level'] == 'low').sum()),
                'avg_volatility_pct': float(df['volatility'].mean() * 100) if df['volatility'].notna().any() else 0.0,
                'avg_max_drawdown_pct': float(df['max_drawdown'].mean() * 100) if df['max_drawdown'].notna().any() else 0.0
            }
        }

        return report

    def generate_llm_latency_report(self) -> Dict:
        """
        Generate LLM performance latency report.

        Returns:
            Dictionary with latency metrics
        """
        logger.info("Generating LLM latency report...")

        # Get latency from all LLM operations
        analyses_query = "SELECT latency_seconds FROM llm_analyses WHERE latency_seconds IS NOT NULL"
        signals_query = "SELECT latency_seconds FROM llm_signals WHERE latency_seconds IS NOT NULL"
        risks_query = "SELECT latency_seconds FROM llm_risks WHERE latency_seconds IS NOT NULL"

        self.db.cursor.execute(analyses_query)
        analysis_latencies = [row[0] for row in self.db.cursor.fetchall()]

        self.db.cursor.execute(signals_query)
        signal_latencies = [row[0] for row in self.db.cursor.fetchall()]

        self.db.cursor.execute(risks_query)
        risk_latencies = [row[0] for row in self.db.cursor.fetchall()]

        report = {
            'report_type': 'llm_latency',
            'generated_at': datetime.now().isoformat(),
            'model_name': 'qwen:14b-chat-q4_K_M',
            'metrics': {
                'market_analysis': {
                    'count': len(analysis_latencies),
                    'avg_seconds': float(np.mean(analysis_latencies)) if analysis_latencies else 0.0,
                    'min_seconds': float(np.min(analysis_latencies)) if analysis_latencies else 0.0,
                    'max_seconds': float(np.max(analysis_latencies)) if analysis_latencies else 0.0,
                    'p95_seconds': float(np.percentile(analysis_latencies, 95)) if analysis_latencies else 0.0
                },
                'signal_generation': {
                    'count': len(signal_latencies),
                    'avg_seconds': float(np.mean(signal_latencies)) if signal_latencies else 0.0,
                    'min_seconds': float(np.min(signal_latencies)) if signal_latencies else 0.0,
                    'max_seconds': float(np.max(signal_latencies)) if signal_latencies else 0.0,
                    'p95_seconds': float(np.percentile(signal_latencies, 95)) if signal_latencies else 0.0
                },
                'risk_assessment': {
                    'count': len(risk_latencies),
                    'avg_seconds': float(np.mean(risk_latencies)) if risk_latencies else 0.0,
                    'min_seconds': float(np.min(risk_latencies)) if risk_latencies else 0.0,
                    'max_seconds': float(np.max(risk_latencies)) if risk_latencies else 0.0,
                    'p95_seconds': float(np.percentile(risk_latencies, 95)) if risk_latencies else 0.0
                }
            }
        }

        return report

    def generate_comprehensive_report(self, output_format: str = 'json') -> str:
        """
        Generate comprehensive report with all metrics.

        Args:
            output_format: 'json', 'text', or 'html'

        Returns:
            Formatted report string
        """
        logger.info("Generating comprehensive LLM performance report...")

        # Gather all reports
        profit_report = self.generate_profit_report()
        signal_report = self.generate_signal_accuracy_report()
        risk_report = self.generate_risk_assessment_report()
        latency_report = self.generate_llm_latency_report()

        comprehensive = {
            'report_title': 'LLM Portfolio Performance - Comprehensive Report',
            'generated_at': datetime.now().isoformat(),
            'model': 'qwen:14b-chat-q4_K_M',
            'reports': {
                'profit_loss': profit_report,
                'signal_accuracy': signal_report,
                'risk_assessment': risk_report,
                'llm_latency': latency_report
            },
            'overall_success_criteria': self._evaluate_overall_success(profit_report, signal_report)
        }

        if output_format == 'json':
            return json.dumps(comprehensive, indent=2, default=str)
        elif output_format == 'text':
            return self._format_text_report(comprehensive)
        elif output_format == 'html':
            return self._format_html_report(comprehensive)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _evaluate_overall_success(self, profit_report: Dict, signal_report: Dict) -> Dict:
        """
        Evaluate overall success against quantifiable criteria.

        Success Criteria:
        1. Total profit > $0 (profitable)
        2. Win rate > 50%
        3. Profit factor > 1.0
        4. Average annual return > 10%
        5. Positive alpha (beats benchmark)
        6. Sharpe ratio > 0
        """
        criteria = {
            'profitable': profit_report['metrics']['total_profit_usd'] > 0,
            'win_rate_above_50': profit_report['metrics']['win_rate_pct'] > 50,
            'profit_factor_above_1': profit_report['metrics']['profit_factor'] > 1.0,
            'annual_return_above_10': signal_report.get('metrics', {}).get('avg_annual_return_pct', 0) > 10,
            'positive_alpha': signal_report.get('metrics', {}).get('avg_alpha_pct', 0) > 0,
            'positive_sharpe': signal_report.get('metrics', {}).get('avg_sharpe_ratio', 0) > 0
        }

        passed = sum(criteria.values())
        total = len(criteria)

        return {
            'criteria': criteria,
            'passed': passed,
            'total': total,
            'pass_rate': passed / total,
            'overall_status': 'PASS' if passed >= 4 else 'FAIL',  # Need 4/6 to pass
            'ready_for_production': passed == total  # All criteria must pass
        }

    def _format_text_report(self, data: Dict) -> str:
        """Format comprehensive report as text"""
        lines = []
        lines.append("=" * 80)
        lines.append("LLM PORTFOLIO PERFORMANCE - COMPREHENSIVE REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {data['generated_at']}")
        lines.append(f"Model: {data['model']}")
        lines.append("")

        # Profit/Loss Section
        profit = data['reports']['profit_loss']['metrics']
        lines.append("1. PROFIT/LOSS METRICS")
        lines.append("-" * 80)
        lines.append(f"Total Profit: ${profit['total_profit_usd']:,.2f}")
        lines.append(f"Avg Profit/Trade: ${profit['avg_profit_per_trade_usd']:,.2f}")
        lines.append(f"Total Trades: {profit['total_trades']}")
        lines.append(f"Winning Trades: {profit['winning_trades']} ({profit['win_rate_pct']:.1f}%)")
        lines.append(f"Losing Trades: {profit['losing_trades']}")
        lines.append(f"Profit Factor: {profit['profit_factor']:.2f}")
        lines.append(f"Win/Loss Ratio: {profit['win_loss_ratio']:.2f}")
        lines.append("")

        # Signal Accuracy Section
        if 'signal_accuracy' in data['reports'] and 'error' not in data['reports']['signal_accuracy']:
            signal = data['reports']['signal_accuracy']['metrics']
            lines.append("2. SIGNAL ACCURACY METRICS")
            lines.append("-" * 80)
            lines.append(f"Total Signals: {data['reports']['signal_accuracy']['total_signals']}")
            lines.append(f"Avg Confidence: {signal['avg_confidence']:.1%}")
            lines.append(f"Avg Annual Return: {signal['avg_annual_return_pct']:.2f}%")
            lines.append(f"Avg Sharpe Ratio: {signal['avg_sharpe_ratio']:.3f}")
            lines.append(f"Avg Alpha: {signal['avg_alpha_pct']:.2f}%")
            lines.append("")

        # Overall Success
        success = data['overall_success_criteria']
        lines.append("3. OVERALL SUCCESS CRITERIA")
        lines.append("-" * 80)
        lines.append(f"Status: {success['overall_status']}")
        lines.append(f"Criteria Passed: {success['passed']}/{success['total']} ({success['pass_rate']:.0%})")
        lines.append(f"Ready for Production: {'✅ YES' if success['ready_for_production'] else '❌ NO'}")
        lines.append("")

        for criterion, passed in success['criteria'].items():
            status = "✅" if passed else "❌"
            lines.append(f"  {status} {criterion.replace('_', ' ').title()}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_html_report(self, data: Dict) -> str:
        """Format comprehensive report as HTML"""
        # Simplified HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Portfolio Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .metric {{ margin: 10px 0; }}
                .success {{ color: green; }}
                .fail {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>LLM Portfolio Performance Report</h1>
            <p><strong>Generated:</strong> {data['generated_at']}</p>
            <p><strong>Model:</strong> {data['model']}</p>

            <h2>Profit/Loss Metrics</h2>
            <div class="metric">Total Profit: ${data['reports']['profit_loss']['metrics']['total_profit_usd']:,.2f}</div>
            <div class="metric">Win Rate: {data['reports']['profit_loss']['metrics']['win_rate_pct']:.1f}%</div>
            <div class="metric">Profit Factor: {data['reports']['profit_loss']['metrics']['profit_factor']:.2f}</div>

            <h2>Overall Status</h2>
            <div class="metric {'success' if data['overall_success_criteria']['overall_status'] == 'PASS' else 'fail'}">
                Status: {data['overall_success_criteria']['overall_status']}
            </div>
        </body>
        </html>
        """
        return html

    def save_report(self, report_content: str, filename: str):
        """Save report to file"""
        filepath = self.report_dir / filename
        with open(filepath, 'w') as f:
            f.write(report_content)
        logger.info(f"Report saved to: {filepath}")
        return filepath


@click.command()
@click.option('--period', default='all', help='Report period (daily/weekly/monthly/all)')
@click.option('--ticker', default='', help='Specific ticker (leave empty for all)')
@click.option('--format', 'output_format', default='text', help='Output format (text/json/html)')
@click.option('--output', default='', help='Output filename (auto-generated if empty)')
@click.option('--detailed', is_flag=True, help='Generate detailed report with charts')
def main(period: str, ticker: str, output_format: str, output: str, detailed: bool):
    """Generate LLM performance report with quantifiable metrics"""

    logger.info("=" * 80)
    logger.info("LLM PERFORMANCE REPORTING SYSTEM")
    logger.info("=" * 80)

    reporter = LLMPerformanceReporter()

    try:
        # Generate comprehensive report
        report_content = reporter.generate_comprehensive_report(output_format)

        # Print to console
        print("\n" + report_content)

        # Save to file
        if not output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = output_format if output_format != 'text' else 'txt'
            output = f"llm_performance_report_{timestamp}.{ext}"

        reporter.save_report(report_content, output)

        logger.info(f"✓ Report generation complete")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        reporter.db.close()


if __name__ == '__main__':
    main()
