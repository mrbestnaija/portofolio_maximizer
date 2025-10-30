"""
Profit-Critical Function Tests (Per AGENT_INSTRUCTION.md)

Tests ONLY functions that directly affect money:
1. Database profit calculations
2. LLM report generation accuracy
3. Pipeline data persistence
4. Performance metric calculations

Maximum: 500 lines (per Phase 4-6 guidelines)
Focus: Business logic that loses money if broken
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import json
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.database_manager import DatabaseManager


class TestProfitCriticalDatabaseFunctions:
    """Test database functions that track money - CRITICAL"""
    
    @pytest.fixture
    def test_db(self, tmp_path):
        """Create temporary test database"""
        db_path = tmp_path / "test_portfolio.db"
        db = DatabaseManager(str(db_path))
        yield db
        db.close()
    
    def test_profit_calculation_accuracy(self, test_db):
        """
        CRITICAL: Profit calculation must be exact to the penny.
        Wrong math = lost money = UNACCEPTABLE.
        
        Tests:
        1. Total profit = Sum of all realized P&L
        2. Winning/losing trade counts
        3. Average profit per trade
        4. Win rate percentage
        5. Gross profit/loss separation
        """
        # Insert test trades with realistic scenarios
        test_db.cursor.execute("""
            INSERT INTO trade_executions 
            (ticker, trade_date, action, shares, price, total_value, realized_pnl)
            VALUES 
            ('AAPL', '2025-01-01', 'BUY', 10, 100.00, 1000.00, NULL),
            ('AAPL', '2025-01-15', 'SELL', 10, 110.00, 1100.00, 100.00),   -- Win: +100
            ('MSFT', '2025-01-05', 'BUY', 5, 200.00, 1000.00, NULL),
            ('MSFT', '2025-01-20', 'SELL', 5, 190.00, 950.00, -50.00)      -- Loss: -50
        """)
        test_db.conn.commit()
        
        # Get performance summary
        perf = test_db.get_performance_summary()
        
        # Test 1: Total profit (EXACT to the penny)
        expected_total = 100.00 - 50.00  # = 50.00
        assert abs(perf['total_profit'] - expected_total) < 0.01, \
            f"Total profit wrong: ${perf['total_profit']:.2f} (expected ${expected_total:.2f})"
        
        # Test 2: Trade counts
        assert perf['total_trades'] == 2, \
            f"Total trades wrong: {perf['total_trades']} (expected 2)"
        assert perf['winning_trades'] == 1, \
            f"Winning trades wrong: {perf['winning_trades']} (expected 1)"
        assert perf['losing_trades'] == 1, \
            f"Losing trades wrong: {perf['losing_trades']} (expected 1)"
        
        # Test 3: Average profit per trade
        expected_avg = 50.00 / 2  # = 25.00
        assert abs(perf['avg_profit_per_trade'] - expected_avg) < 0.01, \
            f"Avg profit wrong: ${perf['avg_profit_per_trade']:.2f} (expected ${expected_avg:.2f})"
        
        # Test 4: Win rate (exact to 0.1%)
        expected_win_rate = 1 / 2  # 50%
        assert abs(perf['win_rate'] - expected_win_rate) < 0.001, \
            f"Win rate wrong: {perf['win_rate']:.1%} (expected {expected_win_rate:.1%})"
        
        # Test 5: Gross profit/loss separation
        assert perf['gross_profit'] == 100.00, \
            f"Gross profit wrong: ${perf['gross_profit']:.2f} (expected $100.00)"
        assert perf['gross_loss'] == 50.00, \
            f"Gross loss wrong: ${perf['gross_loss']:.2f} (expected $50.00)"
        
        # Test 6: Largest win/loss tracking
        assert perf['largest_win'] == 100.00, \
            f"Largest win wrong: ${perf['largest_win']:.2f}"
        assert perf['smallest_loss'] == -50.00, \
            f"Smallest loss wrong: ${perf['smallest_loss']:.2f}"
    
    def test_profit_factor_calculation(self, test_db):
        """
        CRITICAL: Profit factor = Total Gross Profit / Total Gross Loss
        This metric determines system profitability.
        
        Profit Factor Formula (CORRECT):
        PF = Sum(All Winning Trades) / Abs(Sum(All Losing Trades))
        
        NOT: avg_win / avg_loss (WRONG!)
        """
        # Insert trades with known profit factor
        test_db.cursor.execute("""
            INSERT INTO trade_executions 
            (ticker, trade_date, action, shares, price, total_value, realized_pnl)
            VALUES 
            ('TEST1', '2025-01-01', 'SELL', 1, 100, 100, 150.00),  -- Win: +150
            ('TEST2', '2025-01-02', 'SELL', 1, 100, 100, 100.00),  -- Win: +100
            ('TEST3', '2025-01-03', 'SELL', 1, 100, 100, -50.00)   -- Loss: -50
        """)
        test_db.conn.commit()
        
        perf = test_db.get_performance_summary()
        
        # Verify components
        assert perf['gross_profit'] == 250.0, \
            f"Gross profit wrong: {perf['gross_profit']} (expected 250.0)"
        assert perf['gross_loss'] == 50.0, \
            f"Gross loss wrong: {perf['gross_loss']} (expected 50.0)"
        
        # Profit factor = (150 + 100) / 50 = 5.0
        expected_profit_factor = 5.0
        assert abs(perf['profit_factor'] - expected_profit_factor) < 0.01, \
            f"Profit factor wrong: {perf['profit_factor']:.2f} (expected {expected_profit_factor:.2f})"
        
        # Additional validation
        assert perf['profit_factor'] > 1.0, \
            "System with more wins than losses MUST have profit factor > 1.0"
    
    def test_profit_factor_edge_cases(self, test_db):
        """
        CRITICAL: Test edge cases for profit factor calculation.
        """
        # Test Case 1: All wins (profit factor = infinity)
        test_db.cursor.execute("""
            INSERT INTO trade_executions 
            (ticker, trade_date, action, shares, price, total_value, realized_pnl)
            VALUES 
            ('WIN1', '2025-01-01', 'SELL', 1, 100, 100, 100.00),
            ('WIN2', '2025-01-02', 'SELL', 1, 100, 100, 50.00)
        """)
        test_db.conn.commit()
        
        perf = test_db.get_performance_summary()
        assert perf['profit_factor'] == float('inf'), \
            "All wins should result in infinite profit factor"
        
        # Clear for next test
        test_db.cursor.execute("DELETE FROM trade_executions")
        test_db.conn.commit()
        
        # Test Case 2: More losses than wins (profit factor < 1.0)
        test_db.cursor.execute("""
            INSERT INTO trade_executions 
            (ticker, trade_date, action, shares, price, total_value, realized_pnl)
            VALUES 
            ('LOSE1', '2025-01-01', 'SELL', 1, 100, 100, -200.00),
            ('WIN1', '2025-01-02', 'SELL', 1, 100, 100, 100.00)
        """)
        test_db.conn.commit()
        
        perf = test_db.get_performance_summary()
        # Profit factor = 100 / 200 = 0.5
        assert abs(perf['profit_factor'] - 0.5) < 0.01, \
            f"Losing system should have PF < 1.0 (got {perf['profit_factor']:.2f})"
    
    def test_negative_profit_tracking(self, test_db):
        """
        CRITICAL: System must correctly track losses.
        False positives = dangerous overconfidence.
        """
        # Insert losing trades
        test_db.cursor.execute("""
            INSERT INTO trade_executions 
            (ticker, trade_date, action, shares, price, total_value, realized_pnl)
            VALUES 
            ('LOSE1', '2025-01-01', 'SELL', 1, 100, 100, -200.00),
            ('LOSE2', '2025-01-02', 'SELL', 1, 100, 100, -150.00)
        """)
        test_db.conn.commit()
        
        perf = test_db.get_performance_summary()
        
        # Must show negative total profit
        assert perf['total_profit'] == -350.00, \
            f"System not tracking losses correctly: {perf['total_profit']}"
        assert perf['winning_trades'] == 0, "Should have no winning trades"
        assert perf['losing_trades'] == 2, "Should have 2 losing trades"
        assert perf['win_rate'] == 0.0, "Win rate should be 0%"
    
    def test_llm_analysis_persistence(self, test_db):
        """
        CRITICAL: LLM analysis must be saved correctly.
        Lost data = repeated API calls = wasted resources.
        """
        analysis = {
            'trend': 'bullish',
            'strength': 8,
            'regime': 'trending',
            'key_levels': [100.0, 110.0, 120.0],
            'summary': 'Strong uptrend',
            'confidence': 0.85
        }
        
        row_id = test_db.save_llm_analysis(
            ticker='AAPL',
            date='2025-01-01',
            analysis=analysis,
            model_name='qwen:14b-chat-q4_K_M',
            latency=25.3
        )
        
        assert row_id > 0, "Analysis not saved"
        
        # Verify saved data
        test_db.cursor.execute("""
            SELECT trend, strength, regime, latency_seconds, model_name
            FROM llm_analyses WHERE id = ?
        """, (row_id,))
        
        row = test_db.cursor.fetchone()
        assert row['trend'] == 'bullish', "Trend not saved correctly"
        assert row['strength'] == 8, "Strength not saved correctly"
        assert row['latency_seconds'] == 25.3, "Latency not tracked"
        assert row['model_name'] == 'qwen:14b-chat-q4_K_M', "Model name wrong"
    
    def test_signal_validation_status_tracking(self, test_db):
        """
        CRITICAL: Signal validation prevents untested strategies.
        Per AGENT_INSTRUCTION.md: NO TRADING until 30-day validation.
        """
        signal = {
            'action': 'BUY',
            'confidence': 0.75,
            'reasoning': 'Strong momentum',
            'entry_price': 150.00
        }
        
        row_id = test_db.save_llm_signal(
            ticker='AAPL',
            date='2025-01-01',
            signal=signal,
            model_name='qwen:14b-chat-q4_K_M',
            latency=18.5
        )
        
        # Verify default status is 'pending'
        test_db.cursor.execute("""
            SELECT validation_status FROM llm_signals WHERE id = ?
        """, (row_id,))
        
        status = test_db.cursor.fetchone()['validation_status']
        assert status == 'pending', \
            "New signals must default to 'pending' - no auto-trading!"


class TestReportGenerationAccuracy:
    """Test report calculations that show profit metrics"""
    
    @pytest.fixture
    def populated_db(self, tmp_path):
        """Create database with realistic test data"""
        db_path = tmp_path / "test_portfolio.db"
        db = DatabaseManager(str(db_path))
        
        # Insert realistic trading data
        trades = [
            ('AAPL', '2025-01-05', 'SELL', 10, 150, 1500, 120.00),   # Win
            ('AAPL', '2025-01-12', 'SELL', 10, 148, 1480, -20.00),   # Loss
            ('MSFT', '2025-01-08', 'SELL', 5, 300, 1500, 250.00),    # Win
            ('GOOGL', '2025-01-15', 'SELL', 3, 140, 420, 45.00),     # Win
            ('TSLA', '2025-01-20', 'SELL', 2, 200, 400, -80.00),     # Loss
        ]
        
        for ticker, date, action, shares, price, total, pnl in trades:
            db.cursor.execute("""
                INSERT INTO trade_executions 
                (ticker, trade_date, action, shares, price, total_value, realized_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, date, action, shares, price, total, pnl))
        
        db.conn.commit()
        yield db
        db.close()
    
    def test_performance_summary_completeness(self, populated_db):
        """
        CRITICAL: Performance summary must include all key metrics.
        Missing metrics = incomplete profit picture.
        """
        perf = populated_db.get_performance_summary()
        
        # Must have all profit-critical metrics
        required_fields = [
            'total_trades', 'winning_trades', 'losing_trades',
            'total_profit', 'avg_profit_per_trade',
            'win_rate', 'profit_factor',
            'avg_win', 'avg_loss',
            'largest_win', 'smallest_loss'
        ]
        
        for field in required_fields:
            assert field in perf, f"Missing critical metric: {field}"
            assert perf[field] is not None, f"Metric {field} is None"
    
    def test_win_rate_calculation_correctness(self, populated_db):
        """
        CRITICAL: Win rate = winning trades / total trades
        This is a PRIMARY success metric.
        """
        perf = populated_db.get_performance_summary()
        
        # Expected: 3 wins out of 5 trades = 60%
        assert perf['total_trades'] == 5, "Total trades wrong"
        assert perf['winning_trades'] == 3, "Winning trades wrong"
        assert perf['losing_trades'] == 2, "Losing trades wrong"
        
        expected_win_rate = 3 / 5  # 0.60
        assert abs(perf['win_rate'] - expected_win_rate) < 0.001, \
            f"Win rate wrong: {perf['win_rate']:.1%} (expected 60%)"
    
    def test_total_profit_accuracy(self, populated_db):
        """
        CRITICAL: Total profit is THE PRIMARY success metric.
        Must be exact to the penny.
        """
        perf = populated_db.get_performance_summary()
        
        # Expected: 120 - 20 + 250 + 45 - 80 = 315.00
        expected_total = 315.00
        
        assert abs(perf['total_profit'] - expected_total) < 0.01, \
            f"Total profit calculation wrong: ${perf['total_profit']:.2f} (expected ${expected_total:.2f})"


class TestOHLCVDataPersistence:
    """Test that price data is saved correctly - basis for all calculations"""
    
    @pytest.fixture
    def test_db(self, tmp_path):
        """Create temporary test database"""
        db_path = tmp_path / "test_portfolio.db"
        db = DatabaseManager(str(db_path))
        yield db
        db.close()
    
    def test_ohlcv_data_saves_correctly(self, test_db):
        """
        CRITICAL: Price data must be saved accurately.
        Wrong prices = wrong profit calculations.
        """
        # Create test OHLCV data
        dates = pd.date_range('2025-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'Open': [100.0, 102.0, 101.0],
            'High': [105.0, 106.0, 104.0],
            'Low': [99.0, 101.0, 100.0],
            'Close': [103.0, 104.0, 102.0],
            'Volume': [1000000, 1200000, 1100000],
            'Adj Close': [103.0, 104.0, 102.0]
        }, index=pd.MultiIndex.from_product([['AAPL'], dates], names=['Ticker', 'Date']))
        
        # Save to database
        rows_saved = test_db.save_ohlcv_data(data, source='test')
        
        assert rows_saved == 3, f"Should save 3 rows, saved {rows_saved}"
        
        # Verify saved data
        test_db.cursor.execute("""
            SELECT ticker, date, open, high, low, close, volume
            FROM ohlcv_data
            WHERE ticker = 'AAPL'
            ORDER BY date
        """)
        
        rows = test_db.cursor.fetchall()
        assert len(rows) == 3, "Should retrieve 3 rows"
        
        # Verify first row exact values
        assert rows[0]['open'] == 100.0, "Open price wrong"
        assert rows[0]['high'] == 105.0, "High price wrong"
        assert rows[0]['low'] == 99.0, "Low price wrong"
        assert rows[0]['close'] == 103.0, "Close price wrong"
        assert rows[0]['volume'] == 1000000, "Volume wrong"


class TestMVSCriteriaValidation:
    """
    Test Minimum Viable System (MVS) criteria validation.
    Per QUANTIFIABLE_SUCCESS_CRITERIA.md
    """
    
    @pytest.fixture
    def mvs_passing_db(self, tmp_path):
        """Create database that PASSES MVS criteria"""
        db_path = tmp_path / "mvs_pass.db"
        db = DatabaseManager(str(db_path))
        
        # Create 30 trades with >45% win rate, profit factor > 1.0
        for i in range(30):
            pnl = 100.00 if i < 14 else -80.00  # 14 wins, 16 losses = 46.7% win rate
            db.cursor.execute("""
                INSERT INTO trade_executions 
                (ticker, trade_date, action, shares, price, total_value, realized_pnl)
                VALUES (?, ?, 'SELL', 1, 100, 100, ?)
            """, (f'TEST{i}', f'2025-01-{i+1:02d}', pnl))
        
        db.conn.commit()
        yield db
        db.close()
    
    @pytest.fixture
    def mvs_failing_db(self, tmp_path):
        """Create database that FAILS MVS criteria"""
        db_path = tmp_path / "mvs_fail.db"
        db = DatabaseManager(str(db_path))
        
        # Create only 10 trades (< 30 minimum)
        for i in range(10):
            pnl = -50.00  # All losses
            db.cursor.execute("""
                INSERT INTO trade_executions 
                (ticker, trade_date, action, shares, price, total_value, realized_pnl)
                VALUES (?, ?, 'SELL', 1, 100, 100, ?)
            """, (f'TEST{i}', f'2025-01-{i+1:02d}', pnl))
        
        db.conn.commit()
        yield db
        db.close()
    
    def test_mvs_passing_system(self, mvs_passing_db):
        """
        CRITICAL: System must correctly identify when MVS criteria are met.
        """
        perf = mvs_passing_db.get_performance_summary()
        
        # MVS Criteria:
        # 1. Total profit > 0
        assert perf['total_profit'] > 0, "MVS requires profit > 0"
        
        # 2. Win rate > 45%
        assert perf['win_rate'] > 0.45, f"MVS requires win rate > 45% (got {perf['win_rate']:.1%})"
        
        # 3. Profit factor > 1.0
        assert perf['profit_factor'] > 1.0, f"MVS requires profit factor > 1.0 (got {perf['profit_factor']})"
        
        # 4. Total trades >= 30
        assert perf['total_trades'] >= 30, "MVS requires minimum 30 trades"
    
    def test_mvs_failing_system(self, mvs_failing_db):
        """
        CRITICAL: System must correctly identify when MVS criteria FAIL.
        False positives = dangerous overconfidence.
        """
        perf = mvs_failing_db.get_performance_summary()
        
        # Should fail on multiple criteria
        mvs_passed = (
            perf['total_profit'] > 0 and
            perf['win_rate'] > 0.45 and
            perf['profit_factor'] > 1.0 and
            perf['total_trades'] >= 30
        )
        
        assert not mvs_passed, "System should correctly identify MVS failure"


class TestDatabaseIntegrityConstraints:
    """Test database prevents invalid data that could corrupt profit tracking"""
    
    @pytest.fixture
    def test_db(self, tmp_path):
        """Create temporary test database"""
        db_path = tmp_path / "test_portfolio.db"
        db = DatabaseManager(str(db_path))
        yield db
        db.close()
    
    def test_duplicate_signal_prevention(self, test_db):
        """
        CRITICAL: Prevent duplicate signals for same ticker/date/model.
        Duplicates = double-counting = inflated performance metrics.
        """
        signal = {
            'action': 'BUY',
            'confidence': 0.75,
            'reasoning': 'Test',
            'entry_price': 150.00
        }
        
        # Save first signal
        id1 = test_db.save_llm_signal(
            ticker='AAPL',
            date='2025-01-01',
            signal=signal,
            model_name='qwen:14b-chat-q4_K_M'
        )
        
        # Try to save duplicate (should update, not create new)
        id2 = test_db.save_llm_signal(
            ticker='AAPL',
            date='2025-01-01',
            signal=signal,
            model_name='qwen:14b-chat-q4_K_M'
        )
        
        # Check total count
        test_db.cursor.execute("""
            SELECT COUNT(*) as count FROM llm_signals
            WHERE ticker = 'AAPL' AND signal_date = '2025-01-01'
        """)
        
        count = test_db.cursor.fetchone()['count']
        assert count == 1, f"Should have only 1 signal, found {count}"


# Performance test per AGENT_INSTRUCTION.md
class TestSystemPerformanceRequirements:
    """Ensure system meets performance requirements"""
    
    def test_database_query_performance(self, tmp_path):
        """
        Database queries must be fast enough for real-time trading.
        Target: < 0.5 seconds for performance summary.
        """
        import time
        
        db_path = tmp_path / "perf_test.db"
        db = DatabaseManager(str(db_path))
        
        # Insert 1000 trades
        for i in range(1000):
            db.cursor.execute("""
                INSERT INTO trade_executions 
                (ticker, trade_date, action, shares, price, total_value, realized_pnl)
                VALUES (?, ?, 'SELL', 1, 100, 100, ?)
            """, (f'TEST{i%10}', f'2025-01-{(i%28)+1:02d}', float(i % 100 - 50)))
        
        db.conn.commit()
        
        # Time performance summary query
        start = time.time()
        perf = db.get_performance_summary()
        elapsed = time.time() - start
        
        db.close()
        
        # Must complete in < 0.5 seconds
        assert elapsed < 0.5, \
            f"Performance summary too slow: {elapsed:.3f}s (target < 0.5s)"
        
        # Verify correctness
        assert perf['total_trades'] == 1000, "Query returned wrong count"

