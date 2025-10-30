"""
LLM Report Generation Tests (Per AGENT_INSTRUCTION.md)

Tests report accuracy for profit tracking.
Critical: Reports must show correct profit metrics.

Maximum: 200 lines (profit-critical testing only)
"""

import pytest
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.database_manager import DatabaseManager
from scripts.generate_llm_report import LLMPerformanceReporter


class TestProfitReportAccuracy:
    """Test that profit reports show correct numbers"""
    
    @pytest.fixture
    def reporter_with_data(self, tmp_path):
        """Create reporter with realistic test data"""
        db_path = tmp_path / "test_report.db"
        db = DatabaseManager(str(db_path))
        
        # Insert test trades: 3 wins, 2 losses
        trades = [
            ('AAPL', '2025-01-05', 150.00),   # Win
            ('MSFT', '2025-01-10', -50.00),   # Loss
            ('GOOGL', '2025-01-15', 200.00),  # Win
            ('TSLA', '2025-01-20', -75.00),   # Loss
            ('NVDA', '2025-01-25', 100.00),   # Win
        ]
        
        for ticker, date, pnl in trades:
            db.cursor.execute("""
                INSERT INTO trade_executions 
                (ticker, trade_date, action, shares, price, total_value, realized_pnl)
                VALUES (?, ?, 'SELL', 1, 100, 100, ?)
            """, (ticker, date, pnl))
        
        db.conn.commit()
        db.close()
        
        # Create reporter
        reporter = LLMPerformanceReporter(str(db_path))
        yield reporter
        reporter.db.close()
    
    def test_profit_report_total_profit(self, reporter_with_data):
        """CRITICAL: Profit report must show correct total profit"""
        report = reporter_with_data.generate_profit_report()
        
        # Expected: 150 - 50 + 200 - 75 + 100 = 325.00
        expected_profit = 325.00
        actual_profit = report['metrics']['total_profit_usd']
        
        assert abs(actual_profit - expected_profit) < 0.01, \
            f"Profit report wrong: ${actual_profit:.2f} (expected ${expected_profit:.2f})"
    
    def test_profit_report_win_rate(self, reporter_with_data):
        """CRITICAL: Win rate must be calculated correctly in report"""
        report = reporter_with_data.generate_profit_report()
        
        # Expected: 3 wins / 5 trades = 60%
        expected_win_rate = 60.0
        actual_win_rate = report['metrics']['win_rate_pct']
        
        assert abs(actual_win_rate - expected_win_rate) < 0.1, \
            f"Win rate wrong: {actual_win_rate:.1f}% (expected {expected_win_rate:.1f}%)"
    
    def test_profit_report_profit_factor(self, reporter_with_data):
        """CRITICAL: Profit factor must be correct"""
        report = reporter_with_data.generate_profit_report()
        
        # Expected: (150 + 200 + 100) / (50 + 75) = 450 / 125 = 3.6
        expected_pf = 3.6
        actual_pf = report['metrics']['profit_factor']
        
        assert abs(actual_pf - expected_pf) < 0.1, \
            f"Profit factor wrong: {actual_pf:.2f} (expected {expected_pf:.2f})"
    
    def test_mvs_criteria_evaluation(self, reporter_with_data):
        """CRITICAL: Report must correctly evaluate MVS criteria"""
        report = reporter_with_data.generate_profit_report()
        
        # This system should FAIL MVS (only 5 trades, need 30)
        success_criteria = report['success_criteria']
        
        assert 'profitability' in success_criteria
        assert 'win_rate_above_50' in success_criteria
        assert 'sufficient_trades' in success_criteria
        
        # Should fail on trade count
        assert not success_criteria['sufficient_trades'], \
            "Should fail MVS due to insufficient trades"
    
    def test_comprehensive_report_json_format(self, reporter_with_data):
        """Test that comprehensive report generates valid JSON"""
        report_json = reporter_with_data.generate_comprehensive_report(output_format='json')
        
        # Must be valid JSON
        try:
            report_data = json.loads(report_json)
        except json.JSONDecodeError as e:
            pytest.fail(f"Report is not valid JSON: {e}")
        
        # Must have all required sections
        assert 'reports' in report_data
        assert 'profit_loss' in report_data['reports']
        assert 'overall_success_criteria' in report_data
    
    def test_text_report_format(self, reporter_with_data):
        """Test that text report is human-readable"""
        report_text = reporter_with_data.generate_comprehensive_report(output_format='text')
        
        # Must contain key profit metrics
        assert 'Total Profit' in report_text
        assert 'Win Rate' in report_text
        assert 'Profit Factor' in report_text
        assert 'OVERALL SUCCESS CRITERIA' in report_text
        
        # Must show profit value
        assert '$325.00' in report_text or '325.00' in report_text


class TestMVSCriteriaInReport:
    """Test MVS criteria evaluation in reports"""
    
    @pytest.fixture
    def mvs_passing_reporter(self, tmp_path):
        """Create reporter with MVS-passing data"""
        db_path = tmp_path / "mvs_pass.db"
        db = DatabaseManager(str(db_path))
        
        # 30 trades, 60% win rate, profitable
        for i in range(30):
            pnl = 100.00 if i < 18 else -60.00  # 18 wins, 12 losses
            db.cursor.execute("""
                INSERT INTO trade_executions 
                (ticker, trade_date, action, shares, price, total_value, realized_pnl)
                VALUES (?, ?, 'SELL', 1, 100, 100, ?)
            """, (f'TEST{i}', f'2025-01-{(i%28)+1:02d}', pnl))
        
        db.conn.commit()
        db.close()
        
        reporter = LLMPerformanceReporter(str(db_path))
        yield reporter
        reporter.db.close()
    
    def test_mvs_passing_criteria(self, mvs_passing_reporter):
        """Test that MVS-passing system is correctly identified"""
        report = mvs_passing_reporter.generate_profit_report()
        
        # Should pass all MVS criteria
        assert report['success_criteria']['profitability'], "Should be profitable"
        assert report['success_criteria']['win_rate_above_50'], "Should have >50% win rate"
        assert report['success_criteria']['profit_factor_above_1'], "Should have PF > 1.0"
        assert report['success_criteria']['sufficient_trades'], "Should have 30+ trades"
        
        # Overall should pass
        assert report['overall_status'], "MVS criteria should pass overall"

