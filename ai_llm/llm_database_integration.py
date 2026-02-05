"""
LLM Database Integration
Ensures LLM risk assessments and signals are properly saved to database
"""

import logging
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_RISK_LEVELS = ("low", "medium", "high", "extreme")
DEFAULT_RISK_LEVEL = "high"
DEFAULT_DB_PATH = os.environ.get("LLM_DB_PATH", "data/portfolio_maximizer.db")


def _normalise_risk_level(level: Any) -> str:
    """Coerce arbitrary risk level into the allowed taxonomy."""
    if isinstance(level, str):
        normalised = level.strip().lower()
        if normalised in ALLOWED_RISK_LEVELS:
            return normalised
    return DEFAULT_RISK_LEVEL


@dataclass
class LLMSignal:
    """LLM-generated signal for database storage"""
    id: Optional[int]
    ticker: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    expected_return: Optional[float]
    risk_estimate: Optional[float]
    model_used: str
    timestamp: datetime
    market_data_snapshot: Dict[str, Any]
    validation_result: Optional[Dict[str, Any]] = None


@dataclass
class LLMRiskAssessment:
    """LLM risk assessment for database storage"""
    id: Optional[int]
    portfolio_id: str
    risk_level: str
    risk_score: float
    risk_factors: List[str]
    recommendations: List[str]
    model_used: str
    timestamp: datetime
    market_conditions: Dict[str, Any]
    confidence: float


class LLMDatabaseManager:
    """
    Manages LLM data persistence in the database
    """
    
    def __init__(self, db_path: str = "data/portfolio_maximizer.db"):
        # Preserve original string path for backward compatibility/tests.
        self.db_path = db_path
        self._db_file: Optional[Path] = None
        self._db_path_str = db_path
        if db_path != ":memory:":
            resolved = Path(db_path).resolve()
            self._db_file = resolved
            self._db_path_str = str(resolved)
        self._prepare_db_path()
        self._ensure_tables_exist()

    def _prepare_db_path(self) -> None:
        """Ensure the SQLite file and parent directories exist with safe permissions."""
        if self.db_path == ":memory:" or self._db_file is None:
            return
        parent = self._db_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not self._db_file.exists():
            self._db_file.touch()
        try:
            os.chmod(self._db_file, 0o600)
        except OSError:  # pragma: no cover - best effort on Windows/WSL
            logger.debug("Unable to update permissions for %s", self._db_file)

    # ------------------------------------------------------------------ #
    # Utility helpers (backward compatibility for legacy schemas)
    # ------------------------------------------------------------------ #
    def _resolve_column(self, cursor: sqlite3.Cursor,
                        table: str,
                        preferred_order: List[str]) -> str:
        """Return the first column from preferred_order that exists in table."""
        cursor.execute(f"PRAGMA table_info({table})")
        available = {row[1] for row in cursor.fetchall()}
        for column in preferred_order:
            if column in available:
                return column
        return preferred_order[0]

    def _parse_timestamp(self, value: Any) -> datetime:
        """Robust timestamp parsing with sensible fallback to current time."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return datetime.now()
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%dT%H:%M:%S.%fZ",
                            "%Y-%m-%d %H:%M:%S.%f",
                            "%Y-%m-%dT%H:%M:%S"):
                    try:
                        return datetime.strptime(raw, fmt)
                    except ValueError:
                        continue
        return datetime.now()

    def _safe_json_load(self, raw_value: Any, default):
        """Safely decode JSON content with fallback default."""
        if raw_value in (None, "", b""):
            return default
        if isinstance(raw_value, (dict, list)):
            return raw_value
        try:
            return json.loads(raw_value)
        except Exception:
            logger.warning(f"Failed to decode JSON payload: {raw_value}")
            return default
    
    def _ensure_tables_exist(self):
        """Create LLM-specific tables if they don't exist"""
        with sqlite3.connect(self._db_path_str) as conn:
            cursor = conn.cursor()
            
            # LLM Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    expected_return REAL,
                    risk_estimate REAL,
                    model_used TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    market_data_snapshot TEXT,
                    validation_result TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # LLM Risk Assessments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_risk_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL,
                    risk_level TEXT NOT NULL CHECK(risk_level IN ('low', 'medium', 'high', 'extreme')),
                    risk_score REAL NOT NULL,
                    risk_factors TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    market_conditions TEXT,
                    confidence REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._migrate_risk_assessments_table(cursor)
            
            # LLM Performance Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    inference_time REAL NOT NULL,
                    tokens_per_second REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("LLM database tables ensured")

    def _migrate_risk_assessments_table(self, cursor: sqlite3.Cursor) -> None:
        """Upgrade llm_risk_assessments schema to support extreme risk levels."""
        try:
            cursor.execute("PRAGMA table_info(llm_risk_assessments)")
            rows = cursor.fetchall()
            if not rows:
                return

            columns = [row[1] for row in rows]
            has_risk_level = "risk_level" in columns

            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_risk_assessments'"
            )
            schema_row = cursor.fetchone()
            schema_sql = schema_row[0] if schema_row else ""
            constraint_missing_extreme = has_risk_level and "'extreme'" not in schema_sql

            if has_risk_level and not constraint_missing_extreme:
                return

            logger.info("Upgrading llm_risk_assessments table to support extreme risk levels")
            cursor.execute("ALTER TABLE llm_risk_assessments RENAME TO llm_risk_assessments_old")
            cursor.execute("""
                CREATE TABLE llm_risk_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL,
                    risk_level TEXT NOT NULL CHECK(risk_level IN ('low', 'medium', 'high', 'extreme')),
                    risk_score REAL NOT NULL,
                    risk_factors TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    market_conditions TEXT,
                    confidence REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            if has_risk_level:
                cursor.execute("""
                    INSERT INTO llm_risk_assessments (
                        id, portfolio_id, risk_level, risk_score, risk_factors,
                        recommendations, model_used, timestamp,
                        market_conditions, confidence, created_at
                    )
                    SELECT
                        id,
                        portfolio_id,
                        LOWER(
                            CASE
                                WHEN risk_level IN ('low', 'medium', 'high', 'extreme') THEN risk_level
                                WHEN LOWER(risk_level) IN ('low', 'medium', 'high', 'extreme') THEN LOWER(risk_level)
                                ELSE 'high'
                            END
                        ) AS risk_level,
                        risk_score,
                        risk_factors,
                        recommendations,
                        model_used,
                        timestamp,
                        market_conditions,
                        confidence,
                        created_at
                    FROM llm_risk_assessments_old
                """)
            else:
                cursor.execute("""
                    INSERT INTO llm_risk_assessments (
                        id, portfolio_id, risk_level, risk_score, risk_factors,
                        recommendations, model_used, timestamp,
                        market_conditions, confidence, created_at
                    )
                    SELECT
                        id,
                        portfolio_id,
                        'high' AS risk_level,
                        risk_score,
                        risk_factors,
                        recommendations,
                        model_used,
                        timestamp,
                        market_conditions,
                        confidence,
                        created_at
                    FROM llm_risk_assessments_old
                """)

            cursor.execute("DROP TABLE llm_risk_assessments_old")
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("Skipped llm_risk_assessments migration: %s", exc)

    def save_llm_signal(self, signal: LLMSignal) -> int:
        """Save LLM signal to database"""
        try:
            with sqlite3.connect(self._db_path_str) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO llm_signals (
                        ticker, signal_type, confidence, reasoning,
                        expected_return, risk_estimate, model_used,
                        timestamp, market_data_snapshot, validation_result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.ticker,
                    signal.signal_type,
                    signal.confidence,
                    signal.reasoning,
                    signal.expected_return,
                    signal.risk_estimate,
                    signal.model_used,
                    signal.timestamp,
                    json.dumps(signal.market_data_snapshot),
                    json.dumps(signal.validation_result) if signal.validation_result else None
                ))
                
                signal_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"LLM signal saved with ID: {signal_id}")
                return signal_id
                
        except Exception as e:
            logger.error(f"Failed to save LLM signal: {e}")
            raise
    
    def save_risk_assessment(self, assessment: LLMRiskAssessment) -> int:
        """Save LLM risk assessment to database"""
        try:
            with sqlite3.connect(self._db_path_str) as conn:
                cursor = conn.cursor()
                risk_level = _normalise_risk_level(assessment.risk_level)
                
                cursor.execute("""
                    INSERT INTO llm_risk_assessments (
                        portfolio_id, risk_level, risk_score, risk_factors, recommendations,
                        model_used, timestamp, market_conditions, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    assessment.portfolio_id,
                    risk_level,
                    assessment.risk_score,
                    json.dumps(assessment.risk_factors),
                    json.dumps(assessment.recommendations),
                    assessment.model_used,
                    assessment.timestamp,
                    json.dumps(assessment.market_conditions),
                    assessment.confidence
                ))
                
                assessment_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"LLM risk assessment saved with ID: {assessment_id}")
                return assessment_id
                
        except Exception as e:
            logger.error(f"Failed to save risk assessment: {e}")
            raise
    
    def save_performance_metrics(self, model_name: str, inference_time: float,
                               tokens_per_second: float, success: bool,
                               error_message: Optional[str] = None) -> int:
        """Save LLM performance metrics to database"""
        try:
            with sqlite3.connect(self._db_path_str) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO llm_performance_metrics (
                        model_name, inference_time, tokens_per_second,
                        success, error_message, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    inference_time,
                    tokens_per_second,
                    success,
                    error_message,
                    datetime.now()
                ))
                
                metrics_id = cursor.lastrowid
                conn.commit()
                
                return metrics_id
                
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
            raise
    
    def get_recent_signals(self, hours: int = 24) -> List[LLMSignal]:
        """Get recent LLM signals"""
        try:
            with sqlite3.connect(self._db_path_str) as conn:
                cursor = conn.cursor()

                cutoff_time = datetime.now() - timedelta(hours=hours)

                time_column = self._resolve_column(cursor, 'llm_signals', ['timestamp', 'created_at'])
                query = f"""
                    SELECT *, {time_column} AS event_time
                    FROM llm_signals
                    WHERE {time_column} >= ?
                    ORDER BY {time_column} DESC
                """
                cursor.execute(query, (cutoff_time,))

                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                signals: List[LLMSignal] = []
                for row in rows:
                    record = dict(zip(columns, row))
                    timestamp = self._parse_timestamp(record.get('event_time') or record.get(time_column))

                    signal = LLMSignal(
                        id=record.get('id'),
                        ticker=record.get('ticker', ''),
                        signal_type=record.get('signal_type', ''),
                        confidence=record.get('confidence', 0.0),
                        reasoning=record.get('reasoning', ''),
                        expected_return=record.get('expected_return'),
                        risk_estimate=record.get('risk_estimate'),
                        model_used=record.get('model_used', ''),
                        timestamp=timestamp,
                        market_data_snapshot=self._safe_json_load(record.get('market_data_snapshot'), {}),
                        validation_result=self._safe_json_load(record.get('validation_result'), None)
                    )
                    signals.append(signal)

                return signals
                
        except Exception as e:
            logger.error(f"Failed to get recent signals: {e}")
            return []
    
    def get_recent_risk_assessments(self, hours: int = 24) -> List[LLMRiskAssessment]:
        """Get recent LLM risk assessments"""
        try:
            with sqlite3.connect(self._db_path_str) as conn:
                cursor = conn.cursor()

                cutoff_time = datetime.now() - timedelta(hours=hours)

                time_column = self._resolve_column(cursor, 'llm_risk_assessments', ['timestamp', 'created_at'])
                query = f"""
                    SELECT *, {time_column} AS event_time
                    FROM llm_risk_assessments
                    WHERE {time_column} >= ?
                    ORDER BY {time_column} DESC
                """
                cursor.execute(query, (cutoff_time,))

                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                assessments: List[LLMRiskAssessment] = []
                for row in rows:
                    record = dict(zip(columns, row))
                    timestamp = self._parse_timestamp(record.get('event_time') or record.get(time_column))

                    assessment = LLMRiskAssessment(
                        id=record.get('id'),
                        portfolio_id=record.get('portfolio_id', ''),
                        risk_level=_normalise_risk_level(record.get('risk_level')),
                        risk_score=record.get('risk_score', 0.0),
                        risk_factors=self._safe_json_load(record.get('risk_factors'), []),
                        recommendations=self._safe_json_load(record.get('recommendations'), []),
                        model_used=record.get('model_used', ''),
                        timestamp=timestamp,
                        market_conditions=self._safe_json_load(record.get('market_conditions'), {}),
                        confidence=record.get('confidence', 0.0)
                    )
                    assessments.append(assessment)

                return assessments
                
        except Exception as e:
            logger.error(f"Failed to get recent risk assessments: {e}")
            return []
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get LLM performance summary"""
        try:
            with sqlite3.connect(self._db_path_str) as conn:
                cursor = conn.cursor()

                cutoff_time = datetime.now() - timedelta(hours=hours)

                perf_time_column = self._resolve_column(cursor, 'llm_performance_metrics', ['timestamp', 'created_at'])
                cursor.execute(f"""
                    SELECT 
                        model_name,
                        AVG(inference_time) as avg_inference_time,
                        AVG(tokens_per_second) as avg_tokens_per_second,
                        COUNT(*) as total_inferences,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_inferences
                    FROM llm_performance_metrics 
                    WHERE {perf_time_column} >= ?
                    GROUP BY model_name
                """, (cutoff_time,))

                performance_data = cursor.fetchall()

                signal_time_column = self._resolve_column(cursor, 'llm_signals', ['timestamp', 'created_at'])
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_signals,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN signal_type = 'BUY' THEN 1 END) as buy_signals,
                        COUNT(CASE WHEN signal_type = 'SELL' THEN 1 END) as sell_signals
                    FROM llm_signals 
                    WHERE {signal_time_column} >= ?
                """, (cutoff_time,))

                signal_stats = cursor.fetchone()
                
                return {
                    "time_period_hours": hours,
                    "performance_by_model": [
                        {
                            "model_name": row[0],
                            "avg_inference_time": row[1],
                            "avg_tokens_per_second": row[2],
                            "total_inferences": row[3],
                            "success_rate": row[4] / row[3] if row[3] > 0 else 0
                        }
                        for row in performance_data
                    ],
                    "signal_statistics": {
                        "total_signals": signal_stats[0] if signal_stats else 0,
                        "avg_confidence": signal_stats[1] if signal_stats else 0,
                        "buy_signals": signal_stats[2] if signal_stats else 0,
                        "sell_signals": signal_stats[3] if signal_stats else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old LLM data to prevent database bloat"""
        try:
            with sqlite3.connect(self._db_path_str) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now().replace(day=datetime.now().day - days_to_keep)
                
                # Clean up old signals
                cursor.execute("""
                    DELETE FROM llm_signals 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Clean up old risk assessments
                cursor.execute("""
                    DELETE FROM llm_risk_assessments 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Clean up old performance metrics
                cursor.execute("""
                    DELETE FROM llm_performance_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up LLM data older than {days_to_keep} days")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")


# Global database manager instance (lazy load to avoid import-time DB access)
_DEFAULT_LLM_DB_MANAGER: Optional[LLMDatabaseManager] = None


def get_llm_db_manager() -> LLMDatabaseManager:
    """Return a cached LLMDatabaseManager using DEFAULT_DB_PATH."""
    global _DEFAULT_LLM_DB_MANAGER
    if _DEFAULT_LLM_DB_MANAGER is None:
        _DEFAULT_LLM_DB_MANAGER = LLMDatabaseManager(DEFAULT_DB_PATH)
    return _DEFAULT_LLM_DB_MANAGER


def save_llm_signal(ticker: str, signal_type: str, confidence: float,
                   reasoning: str, model_used: str, 
                   expected_return: Optional[float] = None,
                   risk_estimate: Optional[float] = None,
                   market_data_snapshot: Optional[Dict] = None) -> int:
    """Convenience function to save LLM signal"""
    signal = LLMSignal(
        id=None,
        ticker=ticker,
        signal_type=signal_type,
        confidence=confidence,
        reasoning=reasoning,
        expected_return=expected_return,
        risk_estimate=risk_estimate,
        model_used=model_used,
        timestamp=datetime.now(),
        market_data_snapshot=market_data_snapshot or {}
    )
    return get_llm_db_manager().save_llm_signal(signal)


def save_risk_assessment(portfolio_id: str, risk_score: float,
                        risk_factors: List[str], recommendations: List[str],
                        model_used: str, confidence: float,
                        market_conditions: Optional[Dict] = None,
                        risk_level: str = DEFAULT_RISK_LEVEL) -> int:
    """Convenience function to save risk assessment"""
    assessment = LLMRiskAssessment(
        id=None,
        portfolio_id=portfolio_id,
        risk_level=risk_level,
        risk_score=risk_score,
        risk_factors=risk_factors,
        recommendations=recommendations,
        model_used=model_used,
        timestamp=datetime.now(),
        market_conditions=market_conditions or {},
        confidence=confidence
    )
    return get_llm_db_manager().save_risk_assessment(assessment)


def get_performance_summary(hours: int = 24) -> Dict[str, Any]:
    """Convenience function to retrieve aggregated LLM performance data."""
    return get_llm_db_manager().get_performance_summary(hours)
