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
import datetime as dt
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

# SQLite error markers used to decide fallback strategies (docs: arch_tree/implementation)
SQLITE_TRANSIENT_ERRORS = ("disk i/o error",)
SQLITE_CORRUPTION_ERRORS = (
    "database disk image is malformed",
    "file is encrypted or is not a database",
    "unable to open database file",
)
SQLITE_RECOVERABLE_ERRORS = SQLITE_TRANSIENT_ERRORS + SQLITE_CORRUPTION_ERRORS
# Errors that should go through the disk I/O fallback + mirror branch before any rebuild.
SQLITE_CONNECT_DISK_IO_ERRORS = SQLITE_TRANSIENT_ERRORS + ("database disk image is malformed",)


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
        self.backend = "sqlite"
        self.paramstyle = "?"

        # Allow env override to bypass locked/corrupt defaults.
        import os
        db_path_str = str(db_path or os.getenv("PORTFOLIO_DB_PATH", "data/portfolio_maximizer.db")).strip()
        self._sqlite_in_memory = self.backend == "sqlite" and db_path_str == ":memory:"
        self._db_path_hint_is_dir = isinstance(db_path, str) and db_path_str.endswith(("/", "\\"))
        self.db_path = Path(db_path_str)
        
        self.conn = None
        self.cursor = None
        self._busy_timeout_ms = 10000  # Reduce disk I/O contention on Windows
        self._mirror_path: Optional[Path] = None
        self._active_db_path: Path = self.db_path
        if self.backend == "sqlite":
            self._ensure_sqlite_path_exists()
        self._connect()
        self._initialize_schema()
        
        logger.info(f"Database initialized at: {self.db_path}")

    def _ensure_sqlite_path_exists(self) -> None:
        """
        Ensure the SQLite database path is valid before connecting.
        
        Handles three cases documented in the architecture/implementation notes:
        1. Respect explicit in-memory connections (":memory:") without touching disk.
        2. Allow callers to pass a directory so we drop the default DB name inside it.
        3. Create/secure the SQLite file on disk with owner-only permissions.
        """
        if getattr(self, "_sqlite_in_memory", False):
            # Keep the special value untouched so sqlite3.connect(":memory:") works.
            self.db_path = Path(":memory:")
            self._active_db_path = self.db_path
            logger.debug("Configured DatabaseManager to use in-memory SQLite database.")
            return

        normalized_path = Path(self.db_path).expanduser()

        # Allow callers to pass a directory and automatically place the DB inside it.
        if self._db_path_hint_is_dir or normalized_path.is_dir():
            normalized_path = normalized_path / "portfolio_maximizer.db"

        normalized_path = normalized_path.resolve()

        try:
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Unable to create SQLite directory for {normalized_path}: {exc}") from exc

        if not normalized_path.exists():
            try:
                normalized_path.touch()
            except OSError as exc:
                raise RuntimeError(f"Unable to create SQLite file at {normalized_path}: {exc}") from exc

        # SECURITY: Set secure file permissions (read/write for owner only)
        try:
            os.chmod(normalized_path, 0o600)
        except OSError as exc:  # pragma: no cover - best effort on Windows/WSL mounts
            logger.debug("Unable to adjust permissions for %s: %s", normalized_path, exc)

        self.db_path = normalized_path
        self._active_db_path = normalized_path
    
    def _is_transient_sqlite_error(self, message: str) -> bool:
        """Return True if the sqlite error message is a transient I/O issue."""
        return any(marker in message for marker in SQLITE_TRANSIENT_ERRORS)

    def _is_corruption_sqlite_error(self, message: str) -> bool:
        """Return True if sqlite reported corruption/missing backing files."""
        return any(marker in message for marker in SQLITE_CORRUPTION_ERRORS)

    def _should_route_disk_io_recovery(self, message: str) -> bool:
        """
        Determine if connection setup should run through the disk I/O fallback path.
        
        Even known corruption markers (e.g., "database disk image is malformed") first
        flow through the disk I/O cleanup/mirror branch before we attempt a rebuild.
        """
        return any(marker in message for marker in SQLITE_CONNECT_DISK_IO_ERRORS)

    def _backup_corrupted_database(self) -> Optional[Path]:
        """Move the corrupted SQLite store aside so a clean file can be created."""
        if getattr(self, "_sqlite_in_memory", False):
            return None
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_name = f"{self.db_path.name}.corrupt.{timestamp}"
        backup_path = self.db_path.with_name(backup_name)

        try:
            if self.db_path.exists():
                shutil.move(str(self.db_path), str(backup_path))
        except OSError as exc:
            logger.error("Failed to back up corrupted SQLite database %s: %s", self.db_path, exc)
            raise

        # Preserve any lingering WAL/SHM files for forensic analysis.
        for suffix in ("-wal", "-shm"):
            artifact = Path(f"{self.db_path}{suffix}")
            if artifact.exists():
                artifact_backup = artifact.with_name(f"{artifact.name}.corrupt.{timestamp}")
                try:
                    shutil.move(str(artifact), str(artifact_backup))
                except OSError as exc:
                    logger.debug("Unable to move SQLite artifact %s: %s", artifact, exc)

        return backup_path if backup_path.exists() else None
    
    def _connect(self):
        """Establish database connection"""
        connect_kwargs = {
            "detect_types": sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            "timeout": self._busy_timeout_ms / 1000.0,
        }
        if self._should_skip_wal(self.db_path):
            # Defensive: stale WAL/SHM artifacts on Windows mounts can confuse
            # integrity checks and tooling that opens the canonical DB directly.
            # Since we operate against a POSIX mirror in DELETE mode, WAL is not
            # expected on the /mnt path.
            self._cleanup_wal_artifacts(target=self.db_path)
            if self._mirror_path and self._mirror_path.exists():
                logger.debug(
                    "Reusing existing SQLite mirror at %s for %s",
                    self._mirror_path,
                    self.db_path,
                )
                self._establish_connection(self._mirror_path, use_wal=False, connect_kwargs=connect_kwargs)
                return
            logger.debug(
                "Database path %s is on a Windows mount; operating on a POSIX mirror to avoid locking issues.",
                self.db_path,
            )
            self._activate_posix_mirror(connect_kwargs)
            return
        attempts = 0
        use_wal = True
        last_error: Optional[Exception] = None
        rebuild_attempted = False

        while attempts < 3:
            try:
                self._establish_connection(self._active_db_path, use_wal, connect_kwargs)
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                message = str(exc).lower()
                if self._should_route_disk_io_recovery(message):
                    if self._is_corruption_sqlite_error(message):
                        logger.warning(
                            "SQLite reported corruption during connection (%s). "
                            "Attempt %s/%s invoking disk I/O recovery before rebuild.",
                            exc,
                            attempts + 1,
                            3,
                        )
                    else:
                        logger.warning(
                            "SQLite storage error while configuring journal (%s). "
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
                    if not self._is_corruption_sqlite_error(message):
                        continue
                    logger.warning(
                        "Disk I/O branch completed for sqlite corruption marker; attempting rebuild fallback next."
                    )
                if self._is_corruption_sqlite_error(message) and not rebuild_attempted:
                    rebuild_attempted = True
                    self._close_safely()
                    backup_path = self._backup_corrupted_database()
                    logger.error(
                        "SQLite reported corruption (%s). Backed up broken store to %s and creating a clean database.",
                        exc,
                        backup_path or "N/A",
                    )
                    self._ensure_sqlite_path_exists()
                    attempts = 0
                    use_wal = True
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
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
        self._active_db_path = target_path
        self._configure_sqlite_connection(target_path, use_wal)

    def _configure_sqlite_connection(self, target_path: Path, use_wal: bool) -> None:
        """
        Apply SQLite tuning pragmas but gracefully handle environments where they fail.
        
        WSL/Windows drives often reject WAL/synchronous pragmas, so we treat them as optional.
        """
        if use_wal:
            if not self._should_skip_wal(target_path):
                if not self._safe_execute_pragma("journal_mode", "WAL"):
                    logger.warning(
                        "Unable to enable SQLite WAL mode on %s. Continuing with default journal mode.",
                        target_path,
                    )
            else:
                logger.debug("Skipping WAL configuration for %s due to cross-filesystem constraints.", target_path)

        self._safe_execute_pragma("synchronous", "NORMAL")
        self._safe_execute_pragma("foreign_keys", "ON")
        self._safe_execute_pragma("busy_timeout", str(self._busy_timeout_ms))

    def _safe_execute_pragma(self, pragma: str, value: str) -> bool:
        """Apply a SQLite PRAGMA and return True on success."""
        try:
            self.conn.execute(f"PRAGMA {pragma}={value}")
            return True
        except sqlite3.OperationalError as exc:
            logger.debug(
                "Unable to apply SQLite PRAGMA %s=%s on %s: %s",
                pragma,
                value,
                getattr(self, "_active_db_path", "unknown"),
                exc,
            )
            return False

    def _should_skip_wal(self, target_path: Path) -> bool:
        """WSL Windows mounts (/mnt/*) do not support SQLite WAL reliably."""
        if os.name != "posix":
            return False
        target = str(target_path)
        return target.startswith("/mnt/")

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
        return any(marker in err for marker in SQLITE_RECOVERABLE_ERRORS) and path_str.startswith("/mnt/")

    def _activate_posix_mirror(self, connect_kwargs: Dict[str, Any]) -> None:
        """Copy the database to a POSIX-friendly temp path and operate on that copy."""
        tmp_root = Path(os.environ.get("WSL_SQLITE_TMP", "/tmp"))
        tmp_root.mkdir(parents=True, exist_ok=True)
        mirror_path = self._mirror_path or (tmp_root / f"{self.db_path.name}.wsl")

        def _quick_check(candidate: Path) -> bool:
            """Return True when PRAGMA quick_check reports ok for the candidate DB."""
            try:
                conn = sqlite3.connect(str(candidate), timeout=1.0)
                cur = conn.cursor()
                cur.execute("PRAGMA quick_check(1)")
                row = cur.fetchone()
                conn.close()
                return bool(row and row[0] == "ok")
            except Exception:
                return False

        try:
            source_stat = self.db_path.stat() if self.db_path.exists() else None
        except OSError:
            source_stat = None

        if mirror_path.exists():
            mirror_ok = _quick_check(mirror_path)
            if not mirror_ok:
                logger.warning(
                    "Existing SQLite mirror %s failed quick_check; rebuilding from %s.",
                    mirror_path,
                    self.db_path,
                )
                try:
                    mirror_path.unlink()
                except OSError:
                    logger.debug("Unable to remove unhealthy SQLite mirror %s", mirror_path)

            refreshed = False
            if source_stat is not None and mirror_path.exists():
                try:
                    mirror_stat = mirror_path.stat()
                except OSError:
                    mirror_stat = None

                if mirror_stat is not None and mirror_stat.st_mtime < source_stat.st_mtime:
                    if _quick_check(self.db_path):
                        try:
                            shutil.copy2(self.db_path, mirror_path)
                            refreshed = True
                            logger.info(
                                "Refreshed SQLite mirror %s from %s (source newer).",
                                mirror_path,
                                self.db_path,
                            )
                        except OSError as exc:
                            logger.error(
                                "Failed to refresh SQLite mirror %s from %s: %s",
                                mirror_path,
                                self.db_path,
                                exc,
                            )
                    else:
                        # Keep the existing mirror as a recovery path when the
                        # Windows-mount DB has been corrupted.
                        logger.warning(
                            "SQLite database %s failed quick_check; continuing with existing mirror %s.",
                            self.db_path,
                            mirror_path,
                        )

            if not refreshed and mirror_path.exists():
                logger.debug(
                    "Continuing to use existing SQLite mirror %s for %s.",
                    mirror_path,
                    self.db_path,
                )

        if not mirror_path.exists():
            try:
                if self.db_path.exists():
                    shutil.copy2(self.db_path, mirror_path)
                else:
                    mirror_path.touch()
                logger.info(
                    "Operating on temporary SQLite mirror %s due to cross-filesystem locking issues with %s.",
                    mirror_path,
                    self.db_path,
                )
            except OSError as exc:
                logger.error("Failed to stage mirror database at %s: %s", mirror_path, exc)
                raise

        self._mirror_path = mirror_path
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
            # Remove any stale WAL/SHM artifacts on the canonical path; mirror
            # syncs a single consistent DB file.
            self._cleanup_wal_artifacts(target=self.db_path)
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
    
    def _rebuild_sqlite_store(self) -> None:
        """Backup the corrupted database, recreate the file, and reload schema."""
        self._close_safely()
        backup_path = self._backup_corrupted_database()
        logger.error(
            "SQLite reported corruption; backed up broken store to %s and rebuilding.",
            backup_path or "N/A",
        )
        self._ensure_sqlite_path_exists()
        self._connect()
        self._initialize_schema()

    def _recover_sqlite_failure(self, exc: Exception, context: str) -> bool:
        """Attempt recovery steps for SQLite write failures."""
        if not isinstance(exc, (sqlite3.DatabaseError, sqlite3.OperationalError)):
            return False
        message = str(exc).lower()
        if "database is locked" in message:
            logger.warning(
                "SQLite reported 'database is locked' during %s; resetting connection.",
                context,
            )
            self._reset_connection()
            return True
        if self._is_transient_sqlite_error(message):
            logger.warning(
                "SQLite transient error during %s (%s). Resetting connection.",
                context,
                exc,
            )
            self._reset_connection()
            return True
        if self._is_corruption_sqlite_error(message):
            logger.error(
                "SQLite corruption detected during %s: %s",
                context,
                exc,
            )
            self._rebuild_sqlite_store()
            return True
        return False
    
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
        fk_clause = ""
        if self.backend != "sqlite":
            fk_clause = ", FOREIGN KEY (signal_id) REFERENCES llm_signals (id)"
        trade_executions_sql = f"""
            CREATE TABLE IF NOT EXISTS trade_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                trade_date DATE NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL')),
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                commission REAL DEFAULT 0,
                mid_price REAL,
                mid_slippage_bps REAL,
                signal_id INTEGER,
                data_source TEXT,
                execution_mode TEXT,
                synthetic_dataset_id TEXT,
                synthetic_generator_version TEXT,
                run_id TEXT,
                realized_pnl REAL,
                realized_pnl_pct REAL,
                holding_period_days INTEGER,
                asset_class TEXT DEFAULT 'equity',
                instrument_type TEXT DEFAULT 'spot',
                underlying_ticker TEXT,
                strike REAL,
                expiry TEXT,
                multiplier REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                {fk_clause}
            )
        """
        self.cursor.execute(trade_executions_sql)

        # Migration: ensure extended trade metadata columns exist (SQLite only)
        self.cursor.execute("PRAGMA table_info(trade_executions)")
        trade_cols = {row["name"] for row in self.cursor.fetchall()}
        # Add new columns one by one to avoid destructive migrations.
        if "mid_price" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN mid_price REAL")
        if "mid_slippage_bps" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN mid_slippage_bps REAL")
        if "asset_class" not in trade_cols:
            self.cursor.execute(
                "ALTER TABLE trade_executions ADD COLUMN asset_class TEXT DEFAULT 'equity'"
            )
        if "instrument_type" not in trade_cols:
            self.cursor.execute(
                "ALTER TABLE trade_executions ADD COLUMN instrument_type TEXT DEFAULT 'spot'"
            )
        if "underlying_ticker" not in trade_cols:
            self.cursor.execute(
                "ALTER TABLE trade_executions ADD COLUMN underlying_ticker TEXT"
            )
        if "strike" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN strike REAL")
        if "expiry" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN expiry TEXT")
        if "multiplier" not in trade_cols:
            self.cursor.execute(
                "ALTER TABLE trade_executions ADD COLUMN multiplier REAL DEFAULT 1.0"
            )
        if "data_source" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN data_source TEXT")
        if "execution_mode" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN execution_mode TEXT")
        if "synthetic_dataset_id" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN synthetic_dataset_id TEXT")
        if "synthetic_generator_version" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN synthetic_generator_version TEXT")
        if "run_id" not in trade_cols:
            self.cursor.execute("ALTER TABLE trade_executions ADD COLUMN run_id TEXT")

        # Database metadata for provenance + governance flags.
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS db_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Synthetic legs for synthetic/structured trades (Phase 4 – MTM plan).
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS synthetic_legs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                synthetic_trade_id INTEGER NOT NULL,
                leg_type TEXT NOT NULL,
                ticker TEXT,
                underlying_ticker TEXT,
                direction INTEGER NOT NULL,
                quantity REAL NOT NULL,
                strike REAL,
                expiry TEXT,
                multiplier REAL DEFAULT 1.0
            )
            """
        )
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_synthetic_legs_trade
            ON synthetic_legs(synthetic_trade_id)
            """
        )

        # Data quality snapshots
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                window_start DATE NOT NULL,
                window_end DATE NOT NULL,
                length INTEGER NOT NULL,
                missing_pct REAL,
                coverage REAL,
                outlier_frac REAL,
                quality_score REAL,
                source TEXT,
                note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, window_start, window_end, source)
            )
        """)

        # Latency metrics per ticker/run
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS latency_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                run_id TEXT,
                stage TEXT,
                ts_ms REAL,
                llm_ms REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Split drift diagnostics for CV and holdout evaluation
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS split_drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                ticker TEXT,
                split_name TEXT NOT NULL,
                psi REAL,
                mean_delta REAL,
                std_delta REAL,
                vol_psi REAL,
                vol_delta REAL,
                volatility_delta REAL,
                volatility_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Strategy optimization cache (stochastic search results)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                regime TEXT NOT NULL,
                params TEXT NOT NULL,
                metrics TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Time-series model candidate cache (TS hyper-parameter search results)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ts_model_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                regime TEXT,
                candidate_name TEXT NOT NULL,
                params TEXT NOT NULL,
                metrics TEXT NOT NULL,
                stability REAL,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ts_model_candidates_ticker_regime
            ON ts_model_candidates(ticker, regime)
            """
        )

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

        # Unified trading signals table (Time Series + LLM)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_date DATE NOT NULL,
                signal_timestamp TIMESTAMP,
                action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'HOLD')),
                source TEXT NOT NULL CHECK(source IN ('TIME_SERIES', 'LLM', 'HYBRID')),
                model_type TEXT,
                confidence REAL CHECK(confidence BETWEEN 0 AND 1),
                entry_price REAL NOT NULL,
                target_price REAL,
                stop_loss REAL,
                expected_return REAL,
                risk_score REAL,
                volatility REAL,
                reasoning TEXT,
                provenance TEXT,  -- JSON string
                validation_status TEXT DEFAULT 'pending' CHECK(validation_status IN ('pending', 'validated', 'failed', 'executed', 'archived')),
                actual_return REAL,
                backtest_annual_return REAL,
                backtest_sharpe REAL,
                backtest_alpha REAL,
                backtest_hit_rate REAL,
                backtest_profit_factor REAL,
                latency_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, signal_date, source, model_type)
            )
        """)
        
        # Create index for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_ticker_date 
            ON trading_signals(ticker, signal_date DESC)
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
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_ticker_dates ON data_quality_snapshots(ticker, window_start, window_end)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_latency_ticker_date ON llm_signals(ticker, signal_date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_latency_metrics_ticker ON latency_metrics(ticker)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_split_drift_run_split ON split_drift_metrics(run_id, split_name)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_configs_regime_score ON strategy_configs(regime, score)")
        
        self.conn.commit()
        logger.info("Database schema initialized successfully")
    
    def save_ohlcv_data(self, df: pd.DataFrame, source: str = 'yfinance') -> int:
        """
        Save OHLCV data to database.
        """

        insert_sql = """
            INSERT OR REPLACE INTO ohlcv_data 
            (ticker, date, open, high, low, close, volume, adj_close, source, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        def _insert_row(payload: Tuple[Any, ...], ticker: str, date_label: Any) -> bool:
            attempts = 0
            while attempts < 3:
                try:
                    cursor = self.cursor
                    cursor.execute(insert_sql, payload)
                    self.conn.commit()
                    return True
                except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
                    attempts += 1
                    if self._recover_sqlite_failure(
                        exc, context=f"save_ohlcv_data ({ticker} {date_label})"
                    ):
                        continue
                    safe_error = sanitize_error(exc)
                    logger.error("Failed to insert %s %s: %s", ticker, date_label, safe_error)
                    return False
                except Exception as exc:  # pragma: no cover - defensive
                    safe_error = sanitize_error(exc)
                    logger.error("Failed to insert %s %s: %s", ticker, date_label, safe_error)
                    return False
            logger.error(
                "Failed to insert %s %s after %s retries (SQLite remained locked).",
                ticker,
                date_label,
                attempts,
            )
            return False

        rows_inserted = 0
        if isinstance(df.index, pd.MultiIndex):
            for (ticker, date), row in df.iterrows():
                payload = (
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
                )
                if _insert_row(payload, ticker, date):
                    rows_inserted += 1
        else:
            default_ticker = df.attrs.get('ticker')
            if not default_ticker and 'ticker' in df.columns and not df['ticker'].dropna().empty:
                default_ticker = str(df['ticker'].dropna().iloc[0])
            default_ticker = default_ticker or 'UNKNOWN'

            has_ticker_column = 'ticker' in df.columns
            for date, row in df.iterrows():
                ticker_value = row.get('ticker') if has_ticker_column else None
                ticker_value = str(ticker_value) if ticker_value not in (None, '') else default_ticker
                payload = (
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
                )
                if _insert_row(payload, ticker_value, date):
                    rows_inserted += 1

        logger.info("Saved %s OHLCV rows to database", rows_inserted)
        return rows_inserted

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

    def save_llm_analysis(
        self,
        ticker: str,
        date: Any,
        analysis: Dict[str, Any],
        model_name: str = 'qwen:14b-chat-q4_K_M',
        latency: float = 0.0,
    ) -> int:
        """Persist high-level LLM market analysis snapshots."""
        if not analysis:
            logger.error("Failed to save LLM analysis: analysis payload missing")
            return -1
        if not isinstance(analysis, dict):
            analysis = dict(getattr(analysis, "__dict__", {}))

        def _normalise_date(value: Any) -> str:
            if isinstance(value, dt.datetime):
                return value.strftime("%Y-%m-%d")
            if isinstance(value, dt.date):
                return value.strftime("%Y-%m-%d")
            return str(value)

        def _clamp_strength(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                strength_val = int(round(float(value)))
            except (TypeError, ValueError):
                return None
            return int(min(10, max(1, strength_val)))

        def _normalise_choice(value: Any, allowed: set, default: str) -> str:
            if isinstance(value, str):
                candidate = value.strip().lower()
                if candidate in allowed:
                    return candidate
            return default

        analysis_date = _normalise_date(date)
        trend = _normalise_choice(analysis.get('trend'), {'bullish', 'bearish', 'neutral'}, 'neutral')
        regime = _normalise_choice(
            analysis.get('regime'),
            {'trending', 'ranging', 'volatile', 'stable', 'unknown'},
            'unknown',
        )
        strength = _clamp_strength(analysis.get('strength')) or 5
        key_levels = analysis.get('key_levels') or []
        if isinstance(key_levels, (list, tuple, set)):
            serialisable_levels = list(key_levels)
        elif key_levels in (None, ''):
            serialisable_levels = []
        else:
            serialisable_levels = [key_levels]
        cleaned_levels: List[Any] = []
        for level in serialisable_levels:
            try:
                cleaned_levels.append(float(level))
            except (TypeError, ValueError):
                cleaned_levels.append(str(level))
        summary = str(analysis.get('summary', '') or '')
        try:
            confidence_raw = analysis.get('confidence')
            confidence = (
                float(confidence_raw)
                if confidence_raw is not None
                else None
            )
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
        payload = (
            ticker,
            analysis_date,
            trend,
            strength,
            regime,
            json.dumps(cleaned_levels),
            summary,
            model_name,
            confidence,
            float(latency),
        )

        def _execute_insert() -> int:
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO llm_analyses
                    (ticker, analysis_date, trend, strength, regime, key_levels,
                     summary, model_name, confidence, latency_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, analysis_date, model_name)
                    DO UPDATE SET
                        trend = excluded.trend,
                        strength = excluded.strength,
                        regime = excluded.regime,
                        key_levels = excluded.key_levels,
                        summary = excluded.summary,
                        confidence = excluded.confidence,
                        latency_seconds = excluded.latency_seconds
                    """,
                    payload,
                )
            self.cursor.execute(
                """
                SELECT id FROM llm_analyses
                WHERE ticker = ? AND analysis_date = ? AND model_name = ?
                """,
                (ticker, analysis_date, model_name),
            )
            row = self.cursor.fetchone()
            row_id = row['id'] if row else -1
            logger.info("Saved LLM analysis for %s on %s (ID: %s)", ticker, analysis_date, row_id)
            return row_id

        try:
            return _execute_insert()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            if self._recover_sqlite_failure(exc, context="save_llm_analysis"):
                try:
                    return _execute_insert()
                except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc2:
                    safe_error = sanitize_error(exc2)
                    logger.error("Retry failed to save LLM analysis: %s", safe_error)
                    return -1
            safe_error = sanitize_error(exc)
            logger.error("Failed to save LLM analysis: %s", safe_error)
            return -1
        except Exception as exc:  # pragma: no cover - defensive
            safe_error = sanitize_error(exc)
            logger.error("Failed to save LLM analysis: %s", safe_error)
            return -1

    def save_llm_signal(
        self,
        ticker: str,
        date: Any,
        signal: Dict[str, Any],
        model_name: str = 'qwen:14b-chat-q4_K_M',
        latency: float = 0.0,
        validation_status: str = 'pending',
    ) -> int:
        """Persist LLM signal outputs plus validation metadata."""
        if not signal:
            logger.error("Failed to save LLM signal: signal payload missing")
            return -1
        if not isinstance(signal, dict):
            signal = dict(getattr(signal, "__dict__", {}))

        def _normalise_date(value: Any) -> str:
            if isinstance(value, dt.datetime):
                return value.strftime("%Y-%m-%d")
            if isinstance(value, dt.date):
                return value.strftime("%Y-%m-%d")
            return str(value)

        def _parse_timestamp(raw: Any, fallback_date: str) -> datetime:
            if isinstance(raw, datetime):
                return raw
            if isinstance(raw, str) and raw.strip():
                try:
                    return datetime.fromisoformat(raw.replace('Z', '+00:00'))
                except ValueError:
                    pass
            try:
                return datetime.strptime(fallback_date, "%Y-%m-%d")
            except ValueError:
                return datetime.utcnow()

        def _safe_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        allowed_statuses = {'pending', 'validated', 'failed', 'executed', 'archived'}
        status = (validation_status or 'pending').lower()
        if status not in allowed_statuses:
            status = 'pending'

        action = str(signal.get('action', 'HOLD')).upper()
        if action not in {'BUY', 'SELL', 'HOLD'}:
            action = 'HOLD'
        signal_type = str(signal.get('signal_type', action)).upper()
        if signal_type not in {'BUY', 'SELL', 'HOLD'}:
            signal_type = action

        signal_date = _normalise_date(date)
        timestamp = _parse_timestamp(signal.get('signal_timestamp'), signal_date)
        confidence = signal.get('confidence', 0.0)
        try:
            confidence_val = float(confidence)
        except (TypeError, ValueError):
            confidence_val = 0.0
        confidence_val = max(0.0, min(1.0, confidence_val))
        entry_price = _safe_float(signal.get('entry_price'))
        target_price = _safe_float(signal.get('target_price'))
        stop_loss = _safe_float(signal.get('stop_loss'))
        position_size = _safe_float(signal.get('position_size'))
        actual_return = _safe_float(signal.get('actual_return'))

        backtest_metrics = signal.get('backtest_metrics', {}) or {}
        payload = (
            ticker,
            signal_date,
            timestamp,
            action,
            signal_type,
            confidence_val,
            str(signal.get('reasoning', '') or ''),
            model_name,
            entry_price,
            target_price,
            stop_loss,
            position_size,
            status,
            actual_return,
            _safe_float(backtest_metrics.get('annual_return')),
            _safe_float(backtest_metrics.get('sharpe_ratio')),
            _safe_float(backtest_metrics.get('information_ratio')),
            _safe_float(backtest_metrics.get('hit_rate')),
            _safe_float(backtest_metrics.get('profit_factor')),
            float(latency),
        )

        def _execute_insert() -> int:
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO llm_signals
                    (ticker, signal_date, signal_timestamp, action, signal_type, confidence,
                     reasoning, model_name, entry_price, target_price, stop_loss, position_size,
                     validation_status, actual_return, backtest_annual_return, backtest_sharpe,
                     backtest_alpha, backtest_hit_rate, backtest_profit_factor, latency_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, signal_date, model_name)
                    DO UPDATE SET
                        signal_timestamp = COALESCE(excluded.signal_timestamp, llm_signals.signal_timestamp),
                        action = excluded.action,
                        signal_type = excluded.signal_type,
                        confidence = excluded.confidence,
                        reasoning = excluded.reasoning,
                        entry_price = excluded.entry_price,
                        target_price = excluded.target_price,
                        stop_loss = excluded.stop_loss,
                        position_size = excluded.position_size,
                        validation_status = excluded.validation_status,
                        actual_return = COALESCE(excluded.actual_return, llm_signals.actual_return),
                        backtest_annual_return = COALESCE(excluded.backtest_annual_return, llm_signals.backtest_annual_return),
                        backtest_sharpe = COALESCE(excluded.backtest_sharpe, llm_signals.backtest_sharpe),
                        backtest_alpha = COALESCE(excluded.backtest_alpha, llm_signals.backtest_alpha),
                        backtest_hit_rate = COALESCE(excluded.backtest_hit_rate, llm_signals.backtest_hit_rate),
                        backtest_profit_factor = COALESCE(excluded.backtest_profit_factor, llm_signals.backtest_profit_factor),
                        latency_seconds = excluded.latency_seconds
                    """,
                    payload,
                )
            self.cursor.execute(
                """
                SELECT id FROM llm_signals
                WHERE ticker = ? AND signal_date = ? AND model_name = ?
                """,
                (ticker, signal_date, model_name),
            )
            row = self.cursor.fetchone()
            row_id = row['id'] if row else -1
            logger.info("Saved LLM signal for %s on %s (ID: %s)", ticker, signal_date, row_id)
            return row_id

        try:
            return _execute_insert()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            if self._recover_sqlite_failure(exc, context="save_llm_signal"):
                try:
                    return _execute_insert()
                except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc2:
                    safe_error = sanitize_error(exc2)
                    logger.error("Retry failed to save LLM signal: %s", safe_error)
                    return -1
            safe_error = sanitize_error(exc)
            logger.error("Failed to save LLM signal: %s", safe_error)
            return -1
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save LLM signal: %s", safe_error)
            return -1

    def save_signal_validation(self, signal_id: int, payload: Dict[str, Any]) -> int:
        """Persist validator audit trail entries for an LLM signal."""
        if signal_id <= 0 or not payload:
            return -1
        if not isinstance(payload, dict):
            payload = dict(getattr(payload, "__dict__", {}))

        def _safe_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        warnings = payload.get('warnings')
        if isinstance(warnings, (list, tuple, set)):
            warnings_blob = json.dumps(list(warnings))
        elif warnings is None:
            warnings_blob = json.dumps([])
        else:
            warnings_blob = str(warnings)

        def _json_default(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            return str(value)

        quality_metrics = payload.get('quality_metrics')
        if isinstance(quality_metrics, (dict, list)):
            quality_blob = json.dumps(quality_metrics, default=_json_default)
        elif quality_metrics is None:
            quality_blob = json.dumps({})
        else:
            quality_blob = str(quality_metrics)

        def _execute_insert() -> int:
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO llm_signal_validations
                    (signal_id, validator_version, confidence_score, recommendation,
                     warnings, quality_metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal_id,
                        payload.get('validator_version'),
                        _safe_float(payload.get('confidence_score')),
                        payload.get('recommendation'),
                        warnings_blob,
                        quality_blob,
                    ),
                )
            row_id = self.cursor.lastrowid
            logger.info("Saved signal validation for signal_id=%s (ID: %s)", signal_id, row_id)
            return row_id

        try:
            return _execute_insert()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            if self._recover_sqlite_failure(exc, context="save_signal_validation"):
                try:
                    return _execute_insert()
                except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc2:
                    safe_error = sanitize_error(exc2)
                    logger.error("Retry failed to save signal validation: %s", safe_error)
                    return -1
            safe_error = sanitize_error(exc)
            logger.error("Failed to save signal validation: %s", safe_error)
            return -1
        except Exception as exc:  # pragma: no cover - defensive
            safe_error = sanitize_error(exc)
            logger.error("Failed to save signal validation: %s", safe_error)
            return -1
    
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
        mid_price: Optional[float] = None,
        mid_slippage_bps: Optional[float] = None,
        signal_id: Optional[int] = None,
        data_source: Optional[str] = None,
        execution_mode: Optional[str] = None,
        synthetic_dataset_id: Optional[str] = None,
        synthetic_generator_version: Optional[str] = None,
        run_id: Optional[str] = None,
        realized_pnl: Optional[float] = None,
        realized_pnl_pct: Optional[float] = None,
        holding_period_days: Optional[int] = None,
        asset_class: str = "equity",
        instrument_type: str = "spot",
        underlying_ticker: Optional[str] = None,
        strike: Optional[float] = None,
        expiry: Optional[str] = None,
        multiplier: float = 1.0,
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
                     commission, mid_price, mid_slippage_bps, signal_id,
                     data_source, execution_mode, synthetic_dataset_id, synthetic_generator_version, run_id,
                     realized_pnl, realized_pnl_pct,
                     holding_period_days, asset_class, instrument_type,
                     underlying_ticker, strike, expiry, multiplier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        trade_date,
                        action,
                        float(shares),
                        float(price),
                        float(total_value),
                        float(commission),
                        mid_price,
                        mid_slippage_bps,
                        signal_id,
                        data_source,
                        execution_mode,
                        synthetic_dataset_id,
                        synthetic_generator_version,
                        run_id,
                        realized_pnl,
                        realized_pnl_pct,
                        holding_period_days,
                        asset_class,
                        instrument_type,
                        underlying_ticker,
                        strike,
                        expiry,
                        float(multiplier),
                    ),
                )
            trade_id = self.cursor.lastrowid
            logger.debug("Trade execution saved (id=%s) for %s", trade_id, ticker)
            return trade_id
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save trade execution: %s", safe_error)
            return -1

    # ------------------------------------------------------------------
    # DB provenance / governance helpers
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value: Any) -> bool:
        """Persist a small metadata key/value for the active database."""
        try:
            if not isinstance(key, str) or not key.strip():
                return False
            stored = value if isinstance(value, str) else json.dumps(value, default=str)
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO db_metadata (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE SET
                      value = excluded.value,
                      updated_at = CURRENT_TIMESTAMP
                    """,
                    (key.strip(), stored),
                )
            return True
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.debug("Failed to set db metadata %s: %s", key, safe_error)
            return False

    def get_metadata(self, key: str) -> Optional[str]:
        """Return a stored metadata value or None."""
        try:
            if not isinstance(key, str) or not key.strip():
                return None
            self.cursor.execute("SELECT value FROM db_metadata WHERE key = ?", (key.strip(),))
            row = self.cursor.fetchone()
            if not row:
                return None
            return row["value"]
        except Exception:
            return None

    def record_run_provenance(
        self,
        *,
        run_id: str,
        execution_mode: Optional[str] = None,
        data_source: Optional[str] = None,
        synthetic_dataset_id: Optional[str] = None,
        synthetic_generator_version: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        """Stamp the DB with the most recent pipeline/trading provenance."""
        payload = {
            "run_id": run_id,
            "execution_mode": execution_mode,
            "data_source": data_source,
            "synthetic_dataset_id": synthetic_dataset_id,
            "synthetic_generator_version": synthetic_generator_version,
            "note": note,
            "db_path": str(self.db_path),
            "recorded_at": datetime.utcnow().isoformat() + "Z",
        }
        self.set_metadata("last_run_provenance", payload)
        if (execution_mode or "").lower() == "synthetic" or (data_source or "").lower() == "synthetic":
            self.set_metadata("profitability_proof", "false")
            self.set_metadata("profitability_proof_reason", "synthetic_data")

    def get_data_provenance_summary(self) -> Dict[str, Any]:
        """Summarize whether the DB contains synthetic artifacts (for labeling)."""
        sources: Dict[str, int] = {}
        trade_sources: Dict[str, int] = {}
        synthetic_dataset_ids: List[str] = []

        try:
            self.cursor.execute("SELECT source, COUNT(*) AS n FROM ohlcv_data GROUP BY source")
            for row in self.cursor.fetchall():
                src = str(row["source"] or "")
                if src:
                    sources[src] = int(row["n"] or 0)
        except Exception:
            sources = {}

        try:
            self.cursor.execute("SELECT data_source, COUNT(*) AS n FROM trade_executions GROUP BY data_source")
            for row in self.cursor.fetchall():
                src = str(row["data_source"] or "")
                if src:
                    trade_sources[src] = int(row["n"] or 0)
        except Exception:
            trade_sources = {}

        try:
            self.cursor.execute(
                """
                SELECT DISTINCT synthetic_dataset_id
                FROM trade_executions
                WHERE synthetic_dataset_id IS NOT NULL AND synthetic_dataset_id != ''
                ORDER BY synthetic_dataset_id
                """
            )
            synthetic_dataset_ids = [str(r[0]) for r in self.cursor.fetchall() if r and r[0]]
        except Exception:
            synthetic_dataset_ids = []

        has_synthetic = bool(sources.get("synthetic") or trade_sources.get("synthetic") or synthetic_dataset_ids)
        origin = "synthetic" if has_synthetic else "live"
        if has_synthetic and len([s for s in sources if s and s != "synthetic"]) > 0:
            origin = "mixed"

        last_run_provenance_raw = self.get_metadata("last_run_provenance")
        last_run_provenance = None
        if last_run_provenance_raw:
            try:
                last_run_provenance = json.loads(last_run_provenance_raw)
            except Exception:
                last_run_provenance = last_run_provenance_raw

        return {
            "origin": origin,
            "ohlcv_sources": sources,
            "trade_sources": trade_sources,
            "synthetic_dataset_ids": synthetic_dataset_ids,
            "profitability_proof": self.get_metadata("profitability_proof"),
            "profitability_proof_reason": self.get_metadata("profitability_proof_reason"),
            "last_run_provenance": last_run_provenance,
            "db_path": str(self.db_path),
        }

    def save_latency_metrics(
        self,
        ticker: str,
        run_id: Optional[str],
        stage: str,
        ts_ms: Optional[float] = None,
        llm_ms: Optional[float] = None,
    ) -> int:
        """Persist per-ticker latency metrics."""
        try:
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO latency_metrics (ticker, run_id, stage, ts_ms, llm_ms)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ticker, run_id, stage, ts_ms, llm_ms),
                )
            return self.cursor.lastrowid
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save latency metrics: %s", safe_error)
            return -1

    def save_split_drift(
        self,
        run_id: Optional[str],
        ticker: Optional[str],
        split_name: str,
        metrics: Dict[str, float],
    ) -> int:
        """Persist drift diagnostics for a specific split."""
        try:
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO split_drift_metrics
                    (run_id, ticker, split_name, psi, mean_delta, std_delta,
                     vol_psi, vol_delta, volatility_delta, volatility_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        ticker,
                        split_name,
                        metrics.get("psi"),
                        metrics.get("mean_delta"),
                        metrics.get("std_delta"),
                        metrics.get("vol_psi"),
                        metrics.get("vol_delta"),
                        metrics.get("std_delta"),  # volatility_delta aligns to std_delta
                        metrics.get("volatility_ratio"),
                    ),
                )
            return self.cursor.lastrowid
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save split drift metrics: %s", safe_error)
            return -1

    def save_quality_snapshot(
        self,
        ticker: str,
        window_start: Any,
        window_end: Any,
        length: int,
        missing_pct: float,
        coverage: float,
        outlier_frac: float,
        quality_score: float,
        source: Optional[str] = None,
        note: Optional[str] = None,
    ) -> int:
        """Persist data quality metrics for a window."""
        try:
            def _to_iso(val: Any) -> str:
                if isinstance(val, datetime):
                    val = val.date()
                if isinstance(val, date):
                    return val.isoformat()
                return str(val)

            with self.conn:
                self.cursor.execute(
                    """
                    INSERT OR REPLACE INTO data_quality_snapshots
                    (ticker, window_start, window_end, length, missing_pct, coverage,
                     outlier_frac, quality_score, source, note, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        ticker,
                        _to_iso(window_start),
                        _to_iso(window_end),
                        int(length),
                        float(missing_pct),
                        float(coverage),
                        float(outlier_frac),
                        float(quality_score),
                        source,
                        note,
                    ),
                )
            return self.cursor.lastrowid
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save data quality snapshot: %s", safe_error)
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
        import json

        model_type = forecast_data.get('model_type', 'COMBINED')
        horizon = forecast_data.get('forecast_horizon', 1)
        forecast_value = forecast_data.get('forecast_value')

        if forecast_value is None:
            raise ValueError("forecast_value is required")

        model_order_str = json.dumps(forecast_data.get('model_order', {}))
        diagnostics_data = forecast_data.get('diagnostics', {}) or {}
        regression_metrics = forecast_data.get('regression_metrics')
        diagnostics_str = json.dumps(diagnostics_data)
        regression_metrics_str = json.dumps(regression_metrics or {})

        def _execute_insert() -> int:
            self.cursor.execute(
                """
                INSERT OR REPLACE INTO time_series_forecasts
                (ticker, forecast_date, model_type, forecast_horizon,
                 forecast_value, lower_ci, upper_ci, volatility,
                 model_order, aic, bic, diagnostics, regression_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    forecast_date,
                    model_type,
                    horizon,
                    float(forecast_value),
                    float(forecast_data.get('lower_ci'))
                    if forecast_data.get('lower_ci') is not None
                    else None,
                    float(forecast_data.get('upper_ci'))
                    if forecast_data.get('upper_ci') is not None
                    else None,
                    float(forecast_data.get('volatility'))
                    if forecast_data.get('volatility') is not None
                    else None,
                    model_order_str,
                    float(forecast_data.get('aic'))
                    if forecast_data.get('aic') is not None
                    else None,
                    float(forecast_data.get('bic'))
                    if forecast_data.get('bic') is not None
                    else None,
                    diagnostics_str,
                    regression_metrics_str,
                ),
            )
            self.conn.commit()
            row_id = self.cursor.lastrowid
            logger.info("Saved forecast for %s on %s (ID: %s)", ticker, forecast_date, row_id)
            return row_id

        try:
            return _execute_insert()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            if self._recover_sqlite_failure(exc, context="save_forecast"):
                try:
                    return _execute_insert()
                except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc2:
                    safe_error = sanitize_error(exc2)
                    logger.error("Retry failed to save forecast: %s", safe_error)
                    return -1
            safe_error = sanitize_error(exc)
            logger.error("Failed to save forecast: %s", safe_error)
            return -1

    def get_forecasts(
        self,
        ticker: str,
        *,
        model_types: Optional[list[str]] = None,
        limit: int = 200,
    ) -> list[Dict[str, Any]]:
        """Fetch stored forecast rows for a ticker (newest first by default)."""
        ticker = str(ticker or "")
        if not ticker:
            return []
        types = [str(t).upper() for t in (model_types or []) if t]
        params: list[Any] = [ticker]
        where = "ticker = ?"
        if types:
            placeholders = ", ".join(["?"] * len(types))
            where += f" AND model_type IN ({placeholders})"
            params.extend(types)
        params.append(int(limit))
        try:
            cursor = self.cursor.execute(
                f"""
                SELECT id, ticker, forecast_date, model_type, forecast_horizon,
                       forecast_value, regression_metrics
                FROM time_series_forecasts
                WHERE {where}
                ORDER BY forecast_date DESC
                LIMIT ?
                """,
                params,
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:  # pragma: no cover - defensive
            safe_error = sanitize_error(exc)
            logger.error("Failed to fetch forecasts for %s: %s", ticker, safe_error)
            return []

    def update_forecast_regression_metrics(
        self,
        forecast_id: int,
        regression_metrics: Dict[str, Any],
    ) -> bool:
        """Update regression_metrics for an existing forecast row."""
        import json

        try:
            forecast_id_int = int(forecast_id)
        except Exception:
            return False
        try:
            payload = json.dumps(regression_metrics or {})
        except Exception:
            payload = json.dumps({})

        def _execute_update() -> bool:
            self.cursor.execute(
                """
                UPDATE time_series_forecasts
                SET regression_metrics = ?
                WHERE id = ?
                """,
                (payload, forecast_id_int),
            )
            self.conn.commit()
            return True

        try:
            return _execute_update()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            if self._recover_sqlite_failure(exc, context="update_forecast_regression_metrics"):
                try:
                    return _execute_update()
                except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc2:
                    safe_error = sanitize_error(exc2)
                    logger.error("Retry failed to update forecast metrics: %s", safe_error)
                    return False
            safe_error = sanitize_error(exc)
            logger.error("Failed to update forecast metrics: %s", safe_error)
            return False

    # ------------------------------------------------------------------
    # Forecast monitoring helpers
    # ------------------------------------------------------------------

    def get_forecast_regression_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_type: str = "COMBINED",
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate regression metrics (RMSE/sMAPE/tracking_error) for a given model
        type from time_series_forecasts over an optional [start_date, end_date]
        window.

        Args:
            start_date: Optional ISO date string (inclusive lower bound).
            end_date: Optional ISO date string (inclusive upper bound).
            model_type: One of the model_type values stored in
                time_series_forecasts (e.g. 'COMBINED', 'SARIMAX', 'SAMOSSA',
                'MSSA_RL'). Defaults to 'COMBINED' for backward compatibility.

        Returns:
            Mapping alias -> metrics dict with mean rmse/smape/tracking_error.
            Alias is:
              - 'ensemble' when model_type == 'COMBINED'
              - model_type.lower() otherwise
            If no rows are available, metrics are None.
        """
        import json

        try:
            where_clauses = ["model_type = ?"]
            params: list[Any] = [model_type]

            if start_date:
                where_clauses.append("forecast_date >= ?")
                params.append(start_date)
            if end_date:
                where_clauses.append("forecast_date <= ?")
                params.append(end_date)

            where_sql = " AND ".join(where_clauses)

            query = f"""
            SELECT regression_metrics
            FROM time_series_forecasts
            WHERE {where_sql}
            """
            cursor = self.cursor.execute(query, params)

            rmse_vals: list[float] = []
            smape_vals: list[float] = []
            te_vals: list[float] = []

            for row in cursor.fetchall():
                raw = row["regression_metrics"]
                if not raw:
                    continue
                try:
                    metrics = json.loads(raw)
                except Exception:
                    continue
                rmse_val = metrics.get("rmse")
                smape_val = metrics.get("smape")
                te_val = metrics.get("tracking_error")
                if isinstance(rmse_val, (int, float)):
                    rmse_vals.append(float(rmse_val))
                if isinstance(smape_val, (int, float)):
                    smape_vals.append(float(smape_val))
                if isinstance(te_val, (int, float)):
                    te_vals.append(float(te_val))

            def _mean(xs: list[float]) -> Optional[float]:
                return float(sum(xs) / len(xs)) if xs else None

            alias = "ensemble" if model_type == "COMBINED" else model_type.lower()
            return {
                alias: {
                    "rmse": _mean(rmse_vals),
                    "smape": _mean(smape_vals),
                    "tracking_error": _mean(te_vals),
                }
            }
        except Exception as exc:  # pragma: no cover - defensive
            safe_error = sanitize_error(exc)
            logger.error(
                "Failed to aggregate forecast regression metrics: %s", safe_error
            )
            alias = "ensemble" if model_type == "COMBINED" else model_type.lower()
            return {alias: {"rmse": None, "smape": None, "tracking_error": None}}
    
    def save_trading_signal(
        self,
        ticker: str,
        date: str,
        signal: Dict,
        source: str = 'TIME_SERIES',  # 'TIME_SERIES', 'LLM', 'HYBRID'
        model_type: Optional[str] = None,
        validation_status: str = 'pending',
        latency: float = 0.0,
    ) -> int:
        """
        Save unified trading signal (Time Series or LLM) to trading_signals table.
        
        Args:
            ticker: Stock ticker symbol
            date: Signal date (YYYY-MM-DD)
            signal: Signal dictionary with action, confidence, etc.
            source: Signal source ('TIME_SERIES', 'LLM', 'HYBRID')
            model_type: Model type (e.g., 'ENSEMBLE', 'SARIMAX', 'qwen:14b-chat-q4_K_M')
            validation_status: Validation status
            latency: Signal generation latency in seconds
            
        Returns:
            Signal ID or -1 on error
        """
        allowed_statuses = {'pending', 'validated', 'failed', 'executed', 'archived'}
        status = validation_status.lower() if validation_status else 'pending'
        if status not in allowed_statuses:
            status = 'pending'
        
        if source not in ('TIME_SERIES', 'LLM', 'HYBRID'):
            logger.warning(f"Invalid source '{source}', defaulting to 'TIME_SERIES'")
            source = 'TIME_SERIES'
        
        try:
            action = str(signal.get('action', 'HOLD')).upper()
            if action not in {'BUY', 'SELL', 'HOLD'}:
                action = 'HOLD'
            
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
            
            # Extract provenance (convert dict to JSON string if needed)
            provenance = signal.get('provenance', {})
            if isinstance(provenance, dict):
                provenance = json.dumps(provenance)
            elif not isinstance(provenance, str):
                provenance = json.dumps({})
            
            actual_return = signal.get('actual_return')
            backtest_metrics = signal.get('backtest_metrics', {}) or {}
            
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO trading_signals
                    (ticker, signal_date, signal_timestamp, action, source, model_type,
                     confidence, entry_price, target_price, stop_loss, expected_return,
                     risk_score, volatility, reasoning, provenance, validation_status,
                     latency_seconds, actual_return, backtest_annual_return, backtest_sharpe,
                     backtest_alpha, backtest_hit_rate, backtest_profit_factor)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, signal_date, source, model_type)
                    DO UPDATE SET
                        signal_timestamp = COALESCE(excluded.signal_timestamp, trading_signals.signal_timestamp),
                        action = excluded.action,
                        confidence = excluded.confidence,
                        entry_price = excluded.entry_price,
                        target_price = excluded.target_price,
                        stop_loss = excluded.stop_loss,
                        expected_return = excluded.expected_return,
                        risk_score = excluded.risk_score,
                        volatility = excluded.volatility,
                        reasoning = excluded.reasoning,
                        provenance = excluded.provenance,
                        validation_status = excluded.validation_status,
                        latency_seconds = excluded.latency_seconds,
                        actual_return = COALESCE(excluded.actual_return, trading_signals.actual_return),
                        backtest_annual_return = COALESCE(excluded.backtest_annual_return, trading_signals.backtest_annual_return),
                        backtest_sharpe = COALESCE(excluded.backtest_sharpe, trading_signals.backtest_sharpe),
                        backtest_alpha = COALESCE(excluded.backtest_alpha, trading_signals.backtest_alpha),
                        backtest_hit_rate = COALESCE(excluded.backtest_hit_rate, trading_signals.backtest_hit_rate),
                        backtest_profit_factor = COALESCE(excluded.backtest_profit_factor, trading_signals.backtest_profit_factor)
                    """,
                    (
                        ticker,
                        date,
                        signal_timestamp,
                        action,
                        source,
                        model_type or signal.get('model_type'),
                        float(signal.get('confidence', 0.5)),
                        float(signal.get('entry_price', 0.0)),
                        signal.get('target_price'),
                        signal.get('stop_loss'),
                        signal.get('expected_return'),
                        signal.get('risk_score'),
                        signal.get('volatility'),
                        signal.get('reasoning', ''),
                        provenance,
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
                SELECT id FROM trading_signals
                WHERE ticker = ? AND signal_date = ? AND source = ? AND model_type = ?
                """,
                (ticker, date, source, model_type or signal.get('model_type')),
            )
            row = self.cursor.fetchone()
            row_id = row['id'] if row else -1
            logger.info("Saved trading signal for %s on %s (source=%s, ID: %s)", ticker, date, source, row_id)
            return row_id
            
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save trading signal: %s", safe_error)
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

    def get_realized_pnl_history(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Return realized PnL events for equity curve construction."""
        self.cursor.execute(
            """
            SELECT trade_date, realized_pnl, realized_pnl_pct, ticker, action
            FROM trade_executions
            WHERE realized_pnl IS NOT NULL
            ORDER BY trade_date ASC, id ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_equity_curve(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Build a simple equity curve from realized PnL events within a date window.
        Returns a list of {"date": iso_date, "equity": value} ordered by date.
        """
        query = """
            SELECT trade_date, realized_pnl
            FROM trade_executions
            WHERE realized_pnl IS NOT NULL
        """
        params: List[Any] = []
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
        query += " ORDER BY trade_date ASC, id ASC"

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        equity = []
        running = float(initial_capital)
        for row in rows:
            pnl = row["realized_pnl"] if row["realized_pnl"] is not None else 0.0
            running += float(pnl)
            equity.append({"date": row["trade_date"], "equity": running})
        return equity

    def load_ohlcv(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data for one or more tickers within an optional date range."""
        if not tickers:
            return pd.DataFrame()
        query = """
            SELECT ticker, date, open, high, low, close, volume
            FROM ohlcv_data
            WHERE ticker IN ({placeholders})
        """
        placeholders = ",".join(["?"] * len(tickers))
        params: List[Any] = list(tickers)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date ASC"
        query = query.format(placeholders=placeholders)

        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=["date"])
        if not df.empty:
            # Ensure date column parsed to datetime
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["ticker", "date"]).set_index("date")
        return df

    def get_distinct_tickers(self, limit: Optional[int] = None) -> List[str]:
        """Return distinct tickers from stored OHLCV data."""
        query = "SELECT DISTINCT ticker FROM ohlcv_data ORDER BY ticker"
        params: List[Any] = []
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return [row[0] for row in rows] if rows else []

    def save_strategy_config(
        self,
        regime: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        score: float,
    ) -> int:
        """Persist a single strategy optimization result for a given regime."""
        try:
            payload_params = json.dumps(params or {})
            payload_metrics = json.dumps(metrics or {})
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO strategy_configs (regime, params, metrics, score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (regime, payload_params, payload_metrics, float(score)),
                )
            return self.cursor.lastrowid
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save strategy config: %s", safe_error)
            return -1

    def save_ts_model_candidate(
        self,
        ticker: str,
        regime: Optional[str],
        candidate_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        stability: Optional[float] = None,
        score: Optional[float] = None,
    ) -> int:
        """Persist a single time-series model candidate (hyper-parameter search result)."""
        try:
            payload_params = json.dumps(params or {})
            payload_metrics = json.dumps(metrics or {})
            with self.conn:
                self.cursor.execute(
                    """
                    INSERT INTO ts_model_candidates
                    (ticker, regime, candidate_name, params, metrics, stability, score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        regime,
                        candidate_name,
                        payload_params,
                        payload_metrics,
                        stability,
                        None if score is None else float(score),
                    ),
                )
            return self.cursor.lastrowid
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to save TS model candidate: %s", safe_error)
            return -1

    def get_best_strategy_config(self, regime: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the highest-scoring cached strategy configuration for a regime."""
        try:
            if regime:
                self.cursor.execute(
                    """
                    SELECT regime, params, metrics, score, created_at
                    FROM strategy_configs
                    WHERE regime = ?
                    ORDER BY score DESC, created_at DESC
                    LIMIT 1
                    """,
                    (regime,),
                )
            else:
                self.cursor.execute(
                    """
                    SELECT regime, params, metrics, score, created_at
                    FROM strategy_configs
                    ORDER BY score DESC, created_at DESC
                    LIMIT 1
                    """
                )
            row = self.cursor.fetchone()
            if not row:
                return None
            return {
                "regime": row["regime"],
                "params": json.loads(row["params"] or "{}"),
                "metrics": json.loads(row["metrics"] or "{}"),
                "score": float(row["score"]),
                "created_at": row["created_at"],
            }
        except Exception as exc:
            safe_error = sanitize_error(exc)
            logger.error("Failed to load best strategy config: %s", safe_error)
            return None
    
    def get_performance_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict:
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
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
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
