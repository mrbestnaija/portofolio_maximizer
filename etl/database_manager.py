"""
Database Manager for Portfolio Maximizer (Phase 5.2+)

Persistent relational database for pipeline data storage and retrieval.
Uses SQLite for simplicity with option to upgrade to PostgreSQL.

Features:
- OHLCV data storage with indexing
- LLM analysis results persistence
- Signal tracking with performance metrics
- Risk assessment history
- Quantitative profit/loss tracking
- Portfolio performance analytics
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manage persistent storage for portfolio data and LLM outputs.
    
    Database Schema:
    - ohlcv_data: Historical price data with quality scores
    - llm_analyses: Market analysis results from LLM
    - llm_signals: Trading signals with confidence scores
    - llm_risks: Risk assessments per ticker
    - portfolio_positions: Current and historical positions
    - trade_executions: Executed trades with P&L
    - performance_metrics: Daily/weekly/monthly performance
    """
    
    def __init__(self, db_path: str = "data/portfolio_maximizer.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.cursor = None
        
        self._connect()
        self._initialize_schema()
        
        logger.info(f"Database initialized at: {self.db_path}")
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
    
    def _initialize_schema(self):
        """Create database schema if not exists"""
        
        # OHLCV data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                adj_close REAL,
                source TEXT DEFAULT 'yfinance',
                quality_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date, source)
            )
        """)
        
        # LLM market analyses
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_date DATE NOT NULL,
                trend TEXT NOT NULL CHECK(trend IN ('bullish', 'bearish', 'neutral')),
                strength INTEGER CHECK(strength BETWEEN 1 AND 10),
                regime TEXT CHECK(regime IN ('trending', 'ranging', 'volatile', 'stable', 'unknown')),
                key_levels TEXT,  -- JSON array
                summary TEXT,
                model_name TEXT NOT NULL,
                confidence REAL,
                latency_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, analysis_date, model_name)
            )
        """)
        
        # LLM trading signals
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_date DATE NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'HOLD')),
                confidence REAL CHECK(confidence BETWEEN 0 AND 1),
                reasoning TEXT,
                model_name TEXT NOT NULL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                position_size REAL,
                validation_status TEXT DEFAULT 'pending' CHECK(validation_status IN ('pending', 'validated', 'failed', 'executed')),
                actual_return REAL,
                backtest_annual_return REAL,
                backtest_sharpe REAL,
                backtest_alpha REAL,
                latency_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, signal_date, model_name)
            )
        """)
        
        # LLM risk assessments
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_risks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                assessment_date DATE NOT NULL,
                risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high')),
                risk_score INTEGER CHECK(risk_score BETWEEN 0 AND 100),
                portfolio_weight REAL,
                concerns TEXT,  -- JSON array
                recommendation TEXT,
                model_name TEXT NOT NULL,
                var_95 REAL,  -- Value at Risk 95%
                max_drawdown REAL,
                volatility REAL,
                latency_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, assessment_date, model_name)
            )
        """)
        
        # Portfolio positions
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                position_date DATE NOT NULL,
                shares REAL NOT NULL,
                average_cost REAL NOT NULL,
                current_price REAL NOT NULL,
                market_value REAL NOT NULL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                portfolio_weight REAL,
                days_held INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, position_date)
            )
        """)
        
        # Trade executions (for profit/loss tracking)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                trade_date DATE NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL')),
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                commission REAL DEFAULT 0,
                signal_id INTEGER,  -- Link to llm_signals
                realized_pnl REAL,
                realized_pnl_pct REAL,
                holding_period_days INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES llm_signals (id)
            )
        """)
        
        # Performance metrics (quantifiable success criteria)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date DATE NOT NULL,
                period TEXT NOT NULL CHECK(period IN ('daily', 'weekly', 'monthly', 'quarterly', 'annual')),
                total_value REAL NOT NULL,
                total_return REAL,
                total_return_pct REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                alpha REAL,  -- Excess return vs benchmark
                beta REAL,
                num_trades INTEGER,
                num_winning_trades INTEGER,
                num_losing_trades INTEGER,
                avg_win REAL,
                avg_loss REAL,
                largest_win REAL,
                largest_loss REAL,
                total_commission REAL,
                benchmark_return REAL,  -- Buy-and-hold S&P500
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(metric_date, period)
            )
        """)
        
        # Create indices for performance
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date ON ohlcv_data(ticker, date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_ticker_date ON llm_analyses(ticker, analysis_date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON llm_signals(ticker, signal_date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_risks_ticker_date ON llm_risks(ticker, assessment_date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker_date ON trade_executions(ticker, trade_date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(metric_date)")
        
        self.conn.commit()
        logger.info("Database schema initialized successfully")
    
    def save_ohlcv_data(self, df: pd.DataFrame, source: str = 'yfinance') -> int:
        """
        Save OHLCV data to database.
        
        Args:
            df: DataFrame with OHLCV data (MultiIndex: ticker, date)
            source: Data source name
        
        Returns:
            Number of rows inserted
        """
        rows_inserted = 0
        
        if isinstance(df.index, pd.MultiIndex):
            # MultiIndex DataFrame
            for (ticker, date), row in df.iterrows():
                try:
                    self.cursor.execute("""
                        INSERT OR REPLACE INTO ohlcv_data 
                        (ticker, date, open, high, low, close, volume, adj_close, source, quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, date.strftime('%Y-%m-%d'),
                        float(row.get('Open', 0)), float(row.get('High', 0)),
                        float(row.get('Low', 0)), float(row.get('Close', 0)),
                        int(row.get('Volume', 0)), float(row.get('Adj Close', row.get('Close', 0))),
                        source, 1.0
                    ))
                    rows_inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert {ticker} {date}: {e}")
        else:
            # Single ticker DataFrame
            ticker = df.attrs.get('ticker', 'UNKNOWN')
            for date, row in df.iterrows():
                try:
                    self.cursor.execute("""
                        INSERT OR REPLACE INTO ohlcv_data 
                        (ticker, date, open, high, low, close, volume, adj_close, source, quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, date.strftime('%Y-%m-%d'),
                        float(row.get('Open', 0)), float(row.get('High', 0)),
                        float(row.get('Low', 0)), float(row.get('Close', 0)),
                        int(row.get('Volume', 0)), float(row.get('Adj Close', row.get('Close', 0))),
                        source, 1.0
                    ))
                    rows_inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert {ticker} {date}: {e}")
        
        self.conn.commit()
        logger.info(f"Saved {rows_inserted} OHLCV rows to database")
        return rows_inserted
    
    def save_llm_analysis(self, ticker: str, date: str, analysis: Dict, 
                         model_name: str = 'qwen:14b-chat-q4_K_M', 
                         latency: float = 0.0) -> int:
        """
        Save LLM market analysis to database.
        
        Args:
            ticker: Stock ticker
            date: Analysis date (YYYY-MM-DD)
            analysis: Analysis dictionary from LLMMarketAnalyzer
            model_name: LLM model used
            latency: Analysis latency in seconds
        
        Returns:
            Row ID of inserted/updated record
        """
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO llm_analyses
                (ticker, analysis_date, trend, strength, regime, key_levels, summary, 
                 model_name, confidence, latency_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, date,
                analysis.get('trend', 'neutral'),
                int(analysis.get('strength', 5)),
                analysis.get('regime', 'unknown'),
                json.dumps(analysis.get('key_levels', [])),
                analysis.get('summary', ''),
                model_name,
                analysis.get('confidence', 0.5),
                latency
            ))
            
            self.conn.commit()
            row_id = self.cursor.lastrowid
            logger.info(f"Saved LLM analysis for {ticker} on {date} (ID: {row_id})")
            return row_id
        
        except Exception as e:
            logger.error(f"Failed to save LLM analysis: {e}")
            return -1
    
    def save_llm_signal(self, ticker: str, date: str, signal: Dict,
                       model_name: str = 'qwen:14b-chat-q4_K_M',
                       latency: float = 0.0) -> int:
        """Save LLM trading signal to database"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO llm_signals
                (ticker, signal_date, action, confidence, reasoning, model_name,
                 entry_price, latency_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, date,
                signal.get('action', 'HOLD'),
                float(signal.get('confidence', 0.5)),
                signal.get('reasoning', ''),
                model_name,
                float(signal.get('entry_price', 0.0)),
                latency
            ))
            
            self.conn.commit()
            row_id = self.cursor.lastrowid
            logger.info(f"Saved LLM signal for {ticker} on {date} (ID: {row_id})")
            return row_id
        
        except Exception as e:
            logger.error(f"Failed to save LLM signal: {e}")
            return -1
    
    def save_llm_risk(self, ticker: str, date: str, risk: Dict,
                     model_name: str = 'qwen:14b-chat-q4_K_M',
                     latency: float = 0.0) -> int:
        """Save LLM risk assessment to database"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO llm_risks
                (ticker, assessment_date, risk_level, risk_score, portfolio_weight,
                 concerns, recommendation, model_name, latency_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, date,
                risk.get('risk_level', 'medium'),
                int(risk.get('risk_score', 50)),
                float(risk.get('portfolio_weight', 0.0)),
                json.dumps(risk.get('concerns', [])),
                risk.get('recommendation', ''),
                model_name,
                latency
            ))
            
            self.conn.commit()
            row_id = self.cursor.lastrowid
            logger.info(f"Saved LLM risk assessment for {ticker} on {date} (ID: {row_id})")
            return row_id
        
        except Exception as e:
            logger.error(f"Failed to save LLM risk: {e}")
            return -1
    
    def get_latest_signals(self, ticker: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Retrieve latest trading signals"""
        query = """
            SELECT * FROM llm_signals
            WHERE 1=1
        """
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        query += " ORDER BY signal_date DESC LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(query, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_performance_summary(self, start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict:
        """
        Get quantifiable performance summary.
        
        Returns:
            Dictionary with key performance metrics
        """
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(realized_pnl) as total_profit,
                AVG(realized_pnl) as avg_profit_per_trade,
                AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
                AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss,
                SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl ELSE 0 END)) as gross_loss,
                MAX(realized_pnl) as largest_win,
                MIN(realized_pnl) as smallest_loss
            FROM trade_executions
            WHERE realized_pnl IS NOT NULL
        """
        
        params = []
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
        
        self.cursor.execute(query, params)
        result = dict(self.cursor.fetchone())
        
        # Calculate win rate and profit factor
        # Profit Factor = Total Gross Profit / Total Gross Loss (CORRECT formula)
        if result['total_trades'] > 0:
            result['win_rate'] = result['winning_trades'] / result['total_trades']
            
            # FIXED: Use gross_profit / gross_loss (not averages)
            if result['gross_loss'] and result['gross_loss'] > 0:
                result['profit_factor'] = result['gross_profit'] / result['gross_loss']
            else:
                # All wins, no losses
                result['profit_factor'] = float('inf') if result['gross_profit'] > 0 else 0.0
        else:
            result['win_rate'] = 0.0
            result['profit_factor'] = 0.0
        
        return result
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

