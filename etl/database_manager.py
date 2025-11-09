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
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import sys
import time
import os
import shutil

try:
    from etl.security_utils import sanitize_error
except ModuleNotFoundError:  # pragma: no cover - CLI fallback
    # Allow running this module directly (python etl/database_manager.py)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from etl.security_utils import sanitize_error

logger = logging.getLogger(__name__)

# Updated in Phase 5.x to support extreme risk classification
ALLOWED_RISK_LEVELS = {'low', 'medium', 'high', 'extreme'}


def _normalise_risk_level(level: Any) -> str:
    """Coerce risk levels into the allowed database taxonomy."""
    if isinstance(level, str):
        candidate = level.strip().lower()
        if candidate in ALLOWED_RISK_LEVELS:
            return candidate
    return 'high'


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
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SECURITY: Set secure file permissions (read/write for owner only)
        try:
            if self.db_path.exists():
                os.chmod(self.db_path, 0o600)  # Read/write for owner only
            else:
                self.db_path.touch()
                os.chmod(self.db_path, 0o600)
        except OSError as exc:  # pragma: no cover - best effort on Windows/WSL mounts
            logger.debug("Unable to adjust permissions for %s: %s", self.db_path, exc)
        
        self.conn = None
        self.cursor = None
        self._busy_timeout_ms = 10000  # Reduce disk I/O contention on Windows
        self._mirror_path: Optional[Path] = None
        self._active_db_path: Path = self.db_path

        self._connect()
        self._initialize_schema()
        
        logger.info(f"Database initialized at: {self.db_path}")
    
    def _connect(self):
        """Establish database connection"""
        connect_kwargs = {
            "detect_types": sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        }
        attempts = 0
        use_wal = True
        last_error: Optional[Exception] = None

        while attempts < 3:
            try:
                self._establish_connection(self._active_db_path, use_wal, connect_kwargs)
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                message = str(exc).lower()
                if "disk i/o error" in message:
                    logger.warning(
                        "SQLite disk I/O error while configuring journal (%s). "
                        "Attempt %s/%s falling back to DELETE mode.",
                        exc,
                        attempts + 1,
                        3,
                    )
                    self._cleanup_wal_artifacts()
                    self._close_safely()
                    use_wal = False
                    attempts += 1
                    time.sleep(min(0.2 * attempts, 1.0))
                    continue
                raise

        if last_error and self._should_use_posix_mirror(last_error):
            self._activate_posix_mirror(connect_kwargs)
            return

        if last_error:
            raise last_error

    def _establish_connection(self, target_path: Path, use_wal: bool, connect_kwargs: Dict[str, Any]) -> None:
        """Open a SQLite connection to the desired path."""
        self.conn = sqlite3.connect(str(target_path), **connect_kwargs)
        journal_mode = "WAL" if use_wal else "DELETE"
        self.conn.execute(f"PRAGMA journal_mode={journal_mode};")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms};")
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
        self._active_db_path = target_path

    def _cleanup_wal_artifacts(self, target: Optional[Path] = None) -> None:
        """Remove stale WAL/SHM files that can trigger disk I/O errors."""
        target_path = target or self._active_db_path
        for suffix in ("-wal", "-shm"):
            artifact = Path(f"{target_path}{suffix}")
            try:
                if artifact.exists():
                    artifact.unlink()
                    logger.info("Removed stale SQLite artifact: %s", artifact)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.debug("Unable to remove SQLite artifact %s: %s", artifact, exc)

    def _should_use_posix_mirror(self, error: Exception) -> bool:
        """Determine if we should fall back to a POSIX-local mirror (WSL scenario)."""
        if os.name != "posix":
            return False
        err = str(error).lower()
        path_str = str(self.db_path)
        return "disk i/o error" in err and path_str.startswith("/mnt/")

    def _activate_posix_mirror(self, connect_kwargs: Dict[str, Any]) -> None:
        """Copy the database to a POSIX-friendly temp path and operate on that copy."""
        tmp_root = Path(os.environ.get("WSL_SQLITE_TMP", "/tmp"))
        tmp_root.mkdir(parents=True, exist_ok=True)
        mirror_path = tmp_root / f"{self.db_path.name}.wsl"

        try:
            if self.db_path.exists():
                shutil.copy2(self.db_path, mirror_path)
            else:
                mirror_path.touch()
        except OSError as exc:
            logger.error("Failed to stage mirror database at %s: %s", mirror_path, exc)
            raise

        self._mirror_path = mirror_path
        logger.warning(
            "Operating on temporary SQLite mirror %s due to cross-filesystem locking issues with %s.",
            mirror_path,
            self.db_path,
        )
        # Mirrors stay in DELETE journal mode for compatibility.
        self._establish_connection(mirror_path, use_wal=False, connect_kwargs=connect_kwargs)

    def _close_safely(self) -> None:
        """Close SQLite connection without raising if already closed."""
        try:
            if self.conn:
                self.conn.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Error closing SQLite connection: %s", exc)
        finally:
            self.conn = None
            self.cursor = None

    def _sync_mirror_if_needed(self) -> None:
        """Copy mirror database back to the original Windows path when applicable."""
        if not self._mirror_path:
            return
        if not self._mirror_path.exists():
            self._mirror_path = None
            return
        try:
            shutil.copy2(self._mirror_path, self.db_path)
            logger.info("Synchronized mirror database %s back to %s", self._mirror_path, self.db_path)
        except OSError as exc:
            logger.error("Failed to synchronize mirror database %s: %s", self._mirror_path, exc)
        finally:
            try:
                self._mirror_path.unlink()
            except OSError:
                logger.debug("Unable to remove mirror database %s", self._mirror_path)
            self._mirror_path = None
        self._active_db_path = self.db_path

    def _reset_connection(self):
        """Reset SQLite connection (used after disk I/O errors)."""
        self._close_safely()
        self._connect()
    
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
                signal_timestamp TIMESTAMP,
                action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'HOLD')),
                signal_type TEXT CHECK(signal_type IN ('BUY', 'SELL', 'HOLD')),
                confidence REAL CHECK(confidence BETWEEN 0 AND 1),
                reasoning TEXT,
                model_name TEXT NOT NULL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                position_size REAL,
                validation_status TEXT DEFAULT 'pending' CHECK(validation_status IN ('pending', 'validated', 'failed', 'executed', 'archived')),
                actual_return REAL,
                backtest_annual_return REAL,
                backtest_sharpe REAL,
                backtest_alpha REAL,
                backtest_hit_rate REAL,
                backtest_profit_factor REAL,
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
                risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high', 'extreme')),
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
        
        # Time series forecasts (SARIMAX/GARCH/SAMOSSA)
        forecast_table_sql = """
            CREATE TABLE IF NOT EXISTS time_series_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                forecast_date DATE NOT NULL,
                model_type TEXT NOT NULL CHECK(model_type IN ('SARIMAX', 'GARCH', 'COMBINED', 'SAMOSSA', 'MSSA_RL')),
                forecast_horizon INTEGER NOT NULL,  -- Steps ahead
                forecast_value REAL NOT NULL,
                lower_ci REAL,
                upper_ci REAL,
                volatility REAL,  -- GARCH volatility forecast
                model_order TEXT,  -- JSON string for model parameters
                aic REAL,
                bic REAL,
                diagnostics TEXT,  -- JSON string for model diagnostics
                regression_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, forecast_date, model_type, forecast_horizon)
            )
        """
        self.cursor.execute(forecast_table_sql)

        # Migration: ensure existing installations accept SAMOSSA forecasts
        self.cursor.execute("""
            SELECT sql FROM sqlite_master
            WHERE type='table' AND name='time_series_forecasts'
        """)
        table_info = self.cursor.fetchone()
        table_sql = table_info['sql'] if table_info else None
        if table_sql and ('SAMOSSA' not in table_sql or 'MSSA_RL' not in table_sql or 'COMBINED' not in table_sql):
            logger.info("Upgrading time_series_forecasts schema to include SAMOSSA/MSSA_RL model types")
            self.cursor.execute("ALTER TABLE time_series_forecasts RENAME TO time_series_forecasts_old")
            self.cursor.execute(forecast_table_sql)
            self.cursor.execute("""
                INSERT INTO time_series_forecasts
                (id, ticker, forecast_date, model_type, forecast_horizon,
                 forecast_value, lower_ci, upper_ci, volatility,
                 model_order, aic, bic, diagnostics, regression_metrics, created_at)
                SELECT
                    id, ticker, forecast_date, model_type, forecast_horizon,
                    forecast_value, lower_ci, upper_ci, volatility,
                    model_order, aic, bic, diagnostics, NULL, created_at
                FROM time_series_forecasts_old
            """)
            self.cursor.execute("DROP TABLE time_series_forecasts_old")
            self.conn.commit()
        else:
            self.cursor.execute("PRAGMA table_info(time_series_forecasts)")
            columns = {row['name'] for row in self.cursor.fetchall()}
            if 'regression_metrics' not in columns:
                logger.info("Adding regression_metrics column to time_series_forecasts")
                self.cursor.execute("ALTER TABLE time_series_forecasts ADD COLUMN regression_metrics TEXT")
                self.conn.commit()
        
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

        # Signal validation audit trail
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_signal_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                validator_version TEXT,
                confidence_score REAL,
                recommendation TEXT,
                warnings TEXT,
                quality_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES llm_signals (id) ON DELETE CASCADE
            )
        """)

        # Backtest summaries for LLM signals
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_signal_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                lookback_days INTEGER NOT NULL,
                signals_analyzed INTEGER NOT NULL,
                hit_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                annual_return REAL,
                information_ratio REAL,
                information_coefficient REAL,
                p_value REAL,
                statistically_significant INTEGER,
                bootstrap TEXT,
                statistical_summary TEXT,
                autocorrelation TEXT
            )
        """)

        self._migrate_llm_risks_table()
        self._migrate_llm_signals_table()

    def _migrate_llm_signals_table(self) -> None:
        """Ensure the llm_signals table exposes the latest schema additions."""
        try:
            self.cursor.execute("PRAGMA table_info(llm_signals)")
            columns = {row[1] for row in self.cursor.fetchall()}

            def _add_column(column_sql: str) -> None:
                attempts = 0
                while attempts < 3:
                    try:
                        with self.conn:
                            self.cursor.execute(column_sql)
                        return
                    except sqlite3.OperationalError as exc:
                        message = str(exc).lower()
                        if "disk i/o error" in message or "database is locked" in message:
                            attempts += 1
                            logger.warning(
                                "SQLite reported '%s' while running migration; retry %s/3",
                                message,
                                attempts,
                            )
                            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                            time.sleep(0.25 * attempts)
                            self._reset_connection()
                            continue
                        raise
                logger.error(
                    "Failed to apply migration '%s' after retries due to persistent disk I/O errors",
                    column_sql,
                )

            # Add missing columns
            if 'signal_type' not in columns:
                logger.info("Adding signal_type column to llm_signals table")
                _add_column(
                    "ALTER TABLE llm_signals "
                    "ADD COLUMN signal_type TEXT CHECK(signal_type IN ('BUY', 'SELL', 'HOLD'))"
                )

            if 'signal_timestamp' not in columns:
                logger.info("Adding signal_timestamp column to llm_signals table")
                _add_column(
                    "ALTER TABLE llm_signals "
                    "ADD COLUMN signal_timestamp TIMESTAMP"
                )

            if 'backtest_hit_rate' not in columns:
                logger.info("Adding backtest_hit_rate column to llm_signals table")
                _add_column(
                    "ALTER TABLE llm_signals "
                    "ADD COLUMN backtest_hit_rate REAL"
                )

            if 'backtest_profit_factor' not in columns:
                logger.info("Adding backtest_profit_factor column to llm_signals table")
                _add_column(
                    "ALTER TABLE llm_signals "
                    "ADD COLUMN backtest_profit_factor REAL"
                )

            # Backfill defaults for newly added columns
            with self.conn:
                self.cursor.execute("""
                    UPDATE llm_signals
                    SET signal_type = action
                    WHERE signal_type IS NULL AND action IS NOT NULL
                """)

                self.cursor.execute("""
                    UPDATE llm_signals
                    SET signal_timestamp = 
                        CASE 
                            WHEN signal_timestamp IS NULL AND signal_date IS NOT NULL
                            THEN signal_date
                            ELSE signal_timestamp
                        END
                """)

        except Exception as migration_error:
            logger.warning("llm_signals migration skipped due to error: %s", migration_error)

    def _migrate_llm_risks_table(self):
        """Ensure llm_risks table supports 'extreme' risk level."""
        try:
            self.cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_risks'"
            )
            row = self.cursor.fetchone()
            if not row or row[0] is None:
                return

            schema_sql = row[0]
            if "risk_level IN ('low', 'medium', 'high', 'extreme')" in schema_sql:
                return  # Already up to date

            if "risk_level IN ('low', 'medium', 'high')" not in schema_sql:
                return  # Unexpected schema; skip migration

            logger.info("Upgrading llm_risks table to allow 'extreme' risk level")

            with self.conn:
                self.cursor.execute("ALTER TABLE llm_risks RENAME TO llm_risks_old")
                self.cursor.execute("""
                    CREATE TABLE llm_risks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        assessment_date DATE NOT NULL,
                        risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high', 'extreme')),
                        risk_score INTEGER CHECK(risk_score BETWEEN 0 AND 100),
                        portfolio_weight REAL,
                        concerns TEXT,
                        recommendation TEXT,
                        model_name TEXT NOT NULL,
                        var_95 REAL,
                        max_drawdown REAL,
                        volatility REAL,
                        latency_seconds REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, assessment_date, model_name)
                    )
                """)

                self.cursor.execute("PRAGMA table_info('llm_risks_old')")
                old_columns = [row[1] for row in self.cursor.fetchall()]

                desired_columns = [
                    "id",
                    "ticker",
                    "assessment_date",
                    "risk_level",
                    "risk_score",
                    "portfolio_weight",
                    "concerns",
                    "recommendation",
                    "model_name",
                    "var_95",
                    "max_drawdown",
                    "volatility",
                    "latency_seconds",
                    "created_at",
                ]

                common_columns = [col for col in desired_columns if col in old_columns]
                columns_sql = ", ".join(common_columns)
                self.cursor.execute(
                    f"INSERT INTO llm_risks ({columns_sql}) "
                    f"SELECT {columns_sql} FROM llm_risks_old"
                )
                self.cursor.execute("DROP TABLE llm_risks_old")

            logger.info("llm_risks table migration complete")
        except Exception as migration_error:  # pragma: no cover - defensive
            logger.warning(f"llm_risks migration skipped due to error: {migration_error}")
        
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

        def _perform_inserts(cursor: sqlite3.Cursor) -> int:
            rows = 0
            if isinstance(df.index, pd.MultiIndex):
                for (ticker, date), row in df.iterrows():
                    try:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO ohlcv_data 
                            (ticker, date, open, high, low, close, volume, adj_close, source, quality_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ticker,
                                date.strftime('%Y-%m-%d'),
                                float(row.get('Open', 0)),
                                float(row.get('High', 0)),
                                float(row.get('Low', 0)),
                                float(row.get('Close', 0)),
                                int(row.get('Volume', 0)),
                                float(row.get('Adj Close', row.get('Close', 0))),
                                source,
                                1.0,
                            ),
                        )
                        rows += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        safe_error = sanitize_error(exc)
                        logger.error("Failed to insert %s %s: %s", ticker, date, safe_error)
            else:
                default_ticker = df.attrs.get('ticker')
                if not default_ticker and 'ticker' in df.columns and not df['ticker'].dropna().empty:
                    default_ticker = str(df['ticker'].dropna().iloc[0])
                default_ticker = default_ticker or 'UNKNOWN'

                has_ticker_column = 'ticker' in df.columns
                for date, row in df.iterrows():
                    ticker_value = row.get('ticker') if has_ticker_column else None
                    ticker_value = str(ticker_value) if ticker_value not in (None, '') else default_ticker
                    try:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO ohlcv_data 
                            (ticker, date, open, high, low, close, volume, adj_close, source, quality_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ticker_value,
                                date.strftime('%Y-%m-%d'),
                                float(row.get('Open', 0)),
                                float(row.get('High', 0)),
                                float(row.get('Low', 0)),
                                float(row.get('Close', 0)),
                                int(row.get('Volume', 0)),
                                float(row.get('Adj Close', row.get('Close', 0))),
                                source,
                                1.0,
                            ),
                        )
                        rows += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        safe_error = sanitize_error(exc)
                        logger.error("Failed to insert %s %s: %s", ticker_value, date, safe_error)
            return rows

        try:
            rows_inserted = _perform_inserts(self.cursor)
            self.conn.commit()
            logger.info("Saved %s OHLCV rows to database", rows_inserted)
            return rows_inserted
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            message = str(exc).lower()
            if "disk i/o error" in message:
                logger.warning(
                    "Disk I/O error while saving OHLCV data; attempting connection reset."
                )
                self._reset_connection()
                try:
                    rows_inserted = _perform_inserts(self.cursor)
                    self.conn.commit()
                    logger.info(
                        "Saved %s OHLCV rows to database after connection reset",
                        rows_inserted,
                    )
                    return rows_inserted
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc2:
                    safe_error = sanitize_error(exc2)
                    logger.error("Retry failed to save OHLCV data after reset: %s", safe_error)
                    raise
            safe_error = sanitize_error(exc)
            logger.error("Failed to save OHLCV data: %s", safe_error)
            raise
    
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
            safe_error = sanitize_error(e)
            logger.error(f"Failed to save LLM analysis: {safe_error}")
            return -1
    
    def save_llm_signal(
        self,
        ticker: str,
        date: str,
        signal: Dict,
        model_name: str = 'qwen:14b-chat-q4_K_M',
        latency: float = 0.0,
        validation_status: str = 'pending',
    ) -> int:
        """Persist an LLM trading signal and keep schema metrics in sync."""
        allowed_statuses = {'pending', 'validated', 'failed', 'executed', 'archived'}
        status = validation_status.lower() if validation_status else 'pending'
        if status not in allowed_statuses:
            status = 'pending'

        try:
            action = str(signal.get('action', 'HOLD')).upper()
            signal_type = str(signal.get('signal_type', action)).upper()
            if action not in {'BUY', 'SELL', 'HOLD'}:
                action = 'HOLD'
            if signal_type not in {'BUY', 'SELL', 'HOLD'}:
                signal_type = action

            timestamp_raw = signal.get('signal_timestamp')
            if isinstance(timestamp_raw, datetime):
                signal_timestamp = timestamp_raw
            elif isinstance(timestamp_raw, str) and timestamp_raw.strip():
                try:
                    signal_timestamp = datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
                except ValueError:
                    signal_timestamp = datetime.strptime(date, "%Y-%m-%d")
            else:
                signal_timestamp = datetime.strptime(date, "%Y-%m-%d")

            actual_return = signal.get('actual_return')
            backtest_metrics = signal.get('backtest_metrics', {}) or {}

            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO llm_signals
                    (ticker, signal_date, signal_timestamp, action, signal_type, confidence,
                     reasoning, model_name, entry_price, validation_status, latency_seconds,
                     actual_return, backtest_annual_return, backtest_sharpe, backtest_alpha,
                     backtest_hit_rate, backtest_profit_factor)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, signal_date, model_name)
                    DO UPDATE SET
                        signal_timestamp = COALESCE(excluded.signal_timestamp, llm_signals.signal_timestamp),
                        action = excluded.action,
                        signal_type = excluded.signal_type,
                        confidence = excluded.confidence,
                        reasoning = excluded.reasoning,
                        entry_price = excluded.entry_price,
                        validation_status = excluded.validation_status,
                        latency_seconds = excluded.latency_seconds,
                        actual_return = COALESCE(excluded.actual_return, llm_signals.actual_return),
                        backtest_annual_return = COALESCE(excluded.backtest_annual_return, llm_signals.backtest_annual_return),
                        backtest_sharpe = COALESCE(excluded.backtest_sharpe, llm_signals.backtest_sharpe),
                        backtest_alpha = COALESCE(excluded.backtest_alpha, llm_signals.backtest_alpha),
                        backtest_hit_rate = COALESCE(excluded.backtest_hit_rate, llm_signals.backtest_hit_rate),
                        backtest_profit_factor = COALESCE(excluded.backtest_profit_factor, llm_signals.backtest_profit_factor)
                    """,
                    (
                        ticker,
                        date,
                        signal_timestamp,
                        action,
                        signal_type,
                        float(signal.get('confidence', 0.5)),
                        signal.get('reasoning', ''),
                        model_name,
                        float(signal.get('entry_price', 0.0)),
                        status,
                        latency,
                        actual_return,
                        backtest_metrics.get('annual_return'),
                        backtest_metrics.get('sharpe_ratio'),
                        backtest_metrics.get('information_ratio'),
                        backtest_metrics.get('hit_rate'),
                        backtest_metrics.get('profit_factor'),
                    ),
                )

            self.cursor.execute(
                """
                SELECT id FROM llm_signals
                WHERE ticker = ? AND signal_date = ? AND model_name = ?
                """,
                (ticker, date, model_name),
            )
            row = self.cursor.fetchone()
            row_id = row['id'] if row else -1
            logger.info("Saved LLM signal for %s on %s (ID: %s)", ticker, date, row_id)
            return row_id

        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save LLM signal: %s", safe_error)
            raise

    def save_signal_validation(self, signal_id: int, validation: Dict[str, Any]) -> int:
        """Persist signal validation results for auditability."""
        if signal_id <= 0:
            logger.warning("Skipping validation save; invalid signal_id=%s", signal_id)
            return -1

        warnings_text = json.dumps(validation.get('warnings', []))
        quality_metrics = json.dumps(validation.get('quality_metrics', {}))

        def _execute_insert() -> int:
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO llm_signal_validations
                    (signal_id, validator_version, confidence_score,
                     recommendation, warnings, quality_metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal_id,
                        validation.get("validator_version", "v1"),
                        float(validation.get("confidence_score", 0.0)),
                        validation.get("recommendation", "HOLD"),
                        warnings_text,
                        quality_metrics,
                    ),
                )
            validation_id = self.cursor.lastrowid
            logger.info(
                "Recorded signal validation for signal_id=%s (ID: %s)",
                signal_id,
                validation_id,
            )
            return validation_id

        try:
            return _execute_insert()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            message = str(exc).lower()
            if "disk i/o error" in message:
                logger.warning(
                    "Disk I/O error while saving signal validation; attempting connection reset."
                )
                self._reset_connection()
                try:
                    return _execute_insert()
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc2:
                    safe_error = sanitize_error(exc2)
                    logger.error(
                        "Retry failed to save signal validation after reset: %s", safe_error
                    )
                    return -1
            safe_error = sanitize_error(exc)
            logger.error(f"Failed to save signal validation: {safe_error}")
            return -1
        except Exception as exc:  # pragma: no cover - defensive
            safe_error = sanitize_error(exc)
            logger.error(f"Failed to save signal validation: {safe_error}")
            return -1

    def fetch_recent_signals(
        self,
        ticker: str,
        reference_timestamp: Optional[datetime] = None,
        lookback_days: int = 30,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent signals for a ticker ordered chronologically."""
        if reference_timestamp is None:
            reference_timestamp = datetime.utcnow()

        cutoff = reference_timestamp - timedelta(days=max(lookback_days, 1))
        cutoff_iso = cutoff.isoformat()

        try:
            self.cursor.execute(
                """
                SELECT *
                FROM llm_signals
                WHERE ticker = ?
                  AND COALESCE(signal_timestamp, signal_date || 'T00:00:00') >= ?
                ORDER BY COALESCE(signal_timestamp, signal_date || 'T00:00:00') ASC
                LIMIT ?
                """,
                (ticker, cutoff_iso, limit),
            )
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to fetch recent signals for %s: %s", ticker, safe_error)
            raise

    def update_signal_performance(self, signal_id: int, performance: Dict[str, Any]) -> None:
        """Update a persisted signal row with realised/backtest performance metrics."""
        if signal_id <= 0 or not performance:
            return

        column_map = {
            'actual_return': 'actual_return',
            'annual_return': 'backtest_annual_return',
            'sharpe_ratio': 'backtest_sharpe',
            'information_ratio': 'backtest_alpha',
            'hit_rate': 'backtest_hit_rate',
            'profit_factor': 'backtest_profit_factor',
        }

        assignments: List[str] = []
        values: List[Any] = []
        for key, column in column_map.items():
            value = performance.get(key)
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(float(value))

        if not assignments:
            return

        values.append(signal_id)
        query = f"UPDATE llm_signals SET {', '.join(assignments)} WHERE id = ?"

        with self.conn:
            self.cursor.execute(query, values)

    def save_signal_backtest_summary(
        self,
        ticker: str,
        lookback_days: int,
        report: Any,
    ) -> None:
        """Persist aggregated backtest diagnostics for monitoring/reporting."""
        if report is None:
            return

        payload = {
            'ticker': ticker,
            'lookback_days': lookback_days,
            'signals_analyzed': getattr(report, 'trades_analyzed', 0),
            'hit_rate': getattr(report, 'hit_rate', None),
            'profit_factor': getattr(report, 'profit_factor', None),
            'sharpe_ratio': getattr(report, 'sharpe_ratio', None),
            'annual_return': getattr(report, 'annual_return', None),
            'information_ratio': getattr(report, 'information_ratio', None),
            'information_coefficient': getattr(report, 'information_coefficient', None),
            'p_value': getattr(report, 'p_value', None),
            'statistically_significant': int(
                bool(getattr(report, 'statistically_significant', False))
            ),
            'bootstrap': json.dumps(getattr(report, 'bootstrap_intervals', {}) or {}),
            'statistical_summary': json.dumps(getattr(report, 'statistical_summary', {}) or {}),
            'autocorrelation': json.dumps(getattr(report, 'autocorrelation', {}) or {}),
        }

        with self.conn:
            self.cursor.execute(
                """
                INSERT INTO llm_signal_backtests
                (ticker, lookback_days, signals_analyzed, hit_rate, profit_factor, sharpe_ratio,
                 annual_return, information_ratio, information_coefficient, p_value,
                 statistically_significant, bootstrap, statistical_summary, autocorrelation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload['ticker'],
                    payload['lookback_days'],
                    payload['signals_analyzed'],
                    payload['hit_rate'],
                    payload['profit_factor'],
                    payload['sharpe_ratio'],
                    payload['annual_return'],
                    payload['information_ratio'],
                    payload['information_coefficient'],
                    payload['p_value'],
                    payload['statistically_significant'],
                    payload['bootstrap'],
                    payload['statistical_summary'],
                    payload['autocorrelation'],
                ),
            )
    
    def save_llm_risk(self, ticker: str, date: str, risk: Dict,
                     model_name: str = 'qwen:14b-chat-q4_K_M',
                     latency: float = 0.0) -> int:
        """Save LLM risk assessment to database"""
        try:
            risk_level = _normalise_risk_level(risk.get('risk_level', 'medium'))
            risk_score_raw = risk.get('risk_score', 50)
            try:
                risk_score = int(risk_score_raw)
            except (TypeError, ValueError):
                risk_score = 50
            risk_score = max(0, min(100, risk_score))

            concerns = risk.get('concerns', [])
            if not isinstance(concerns, list):
                concerns = [str(concerns)]

            self.cursor.execute("""
                INSERT OR REPLACE INTO llm_risks
                (ticker, assessment_date, risk_level, risk_score, portfolio_weight,
                 concerns, recommendation, model_name, latency_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, date,
                risk_level,
                risk_score,
                float(risk.get('portfolio_weight', 0.0)),
                json.dumps(concerns),
                risk.get('recommendation', ''),
                model_name,
                latency
            ))
            
            self.conn.commit()
            row_id = self.cursor.lastrowid
            logger.info(f"Saved LLM risk assessment for {ticker} on {date} (ID: {row_id})")
            return row_id
        
        except Exception as e:
            safe_error = sanitize_error(e)
            logger.error(f"Failed to save LLM risk: {safe_error}")
            return -1

    def save_trade_execution(
        self,
        ticker: str,
        trade_date: Any,
        action: str,
        shares: float,
        price: float,
        total_value: float,
        commission: float = 0.0,
        signal_id: Optional[int] = None,
        realized_pnl: Optional[float] = None,
        realized_pnl_pct: Optional[float] = None,
        holding_period_days: Optional[int] = None,
    ) -> int:
        """Persist trade execution details."""
        try:
            if isinstance(trade_date, datetime):
                trade_date = trade_date.date()
            if isinstance(trade_date, date):
                trade_date = trade_date.isoformat()

            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO trade_executions
                    (ticker, trade_date, action, shares, price, total_value,
                     commission, signal_id, realized_pnl, realized_pnl_pct,
                     holding_period_days)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        trade_date,
                        action,
                        float(shares),
                        float(price),
                        float(total_value),
                        float(commission),
                        signal_id,
                        realized_pnl,
                        realized_pnl_pct,
                        holding_period_days,
                    ),
                )
            trade_id = self.cursor.lastrowid
            logger.debug("Trade execution saved (id=%s) for %s", trade_id, ticker)
            return trade_id
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save trade execution: %s", safe_error)
            return -1
    
    def save_forecast(self, ticker: str, forecast_date: str, forecast_data: Dict) -> int:
        """
        Save time series forecast to database.
        
        Args:
            ticker: Stock ticker
            forecast_date: Forecast generation date
            forecast_data: Dictionary with forecast information:
                - model_type: 'SARIMAX', 'GARCH', 'SAMOSSA', or 'COMBINED'
                - forecast_horizon: Number of steps ahead
                - forecast_value: Forecasted value
                - lower_ci: Lower confidence interval (optional)
                - upper_ci: Upper confidence interval (optional)
                - volatility: Volatility forecast (optional, for GARCH)
                - model_order: Model parameters (dict)
                - aic: AIC value (optional)
                - bic: BIC value (optional)
                - diagnostics: Diagnostic metrics (dict, optional)
                - regression_metrics: RMSE/sMAPE/Tracking-error dict (optional)
        
        Returns:
            Row ID of inserted record
        """
        try:
            import json
            
            model_type = forecast_data.get('model_type', 'COMBINED')
            horizon = forecast_data.get('forecast_horizon', 1)
            forecast_value = forecast_data.get('forecast_value')
            
            if forecast_value is None:
                raise ValueError("forecast_value is required")
            
            # Convert model_order and diagnostics to JSON
            model_order_str = json.dumps(forecast_data.get('model_order', {}))
            diagnostics_data = forecast_data.get('diagnostics', {}) or {}
            regression_metrics = forecast_data.get('regression_metrics')
            diagnostics_str = json.dumps(diagnostics_data)
            regression_metrics_str = json.dumps(regression_metrics or {})
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO time_series_forecasts
                (ticker, forecast_date, model_type, forecast_horizon,
                 forecast_value, lower_ci, upper_ci, volatility,
                 model_order, aic, bic, diagnostics, regression_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                forecast_date,
                model_type,
                horizon,
                float(forecast_value),
                float(forecast_data.get('lower_ci')) if forecast_data.get('lower_ci') is not None else None,
                float(forecast_data.get('upper_ci')) if forecast_data.get('upper_ci') is not None else None,
                float(forecast_data.get('volatility')) if forecast_data.get('volatility') is not None else None,
                model_order_str,
                float(forecast_data.get('aic')) if forecast_data.get('aic') is not None else None,
                float(forecast_data.get('bic')) if forecast_data.get('bic') is not None else None,
                diagnostics_str,
                regression_metrics_str,
            ))
            
            self.conn.commit()
            row_id = self.cursor.lastrowid
            logger.info(f"Saved forecast for {ticker} on {forecast_date} (ID: {row_id})")
            return row_id
            
        except Exception as e:
            safe_error = sanitize_error(e)
            logger.error(f"Failed to save forecast: {safe_error}")
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
        self._close_safely()
        self._sync_mirror_if_needed()
        logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
