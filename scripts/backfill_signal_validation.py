#!/usr/bin/env python3
"""
Backfill signal validation metadata for legacy LLM signals.

This maintenance script replays pending signals through the active validator,
persists validation results, and optionally marks irrecoverable signals as
archived. It can also recompute empirical accuracy metrics over a recent
window to keep monitoring dashboards in sync.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import warnings
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_llm.signal_validator import SignalValidator
from etl.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

UTC = timezone.utc


def _ensure_utc_datetime(value: Any) -> datetime:
    """Normalize assorted SQLite/built-in types into timezone-aware UTC datetimes."""
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return datetime.now(UTC)
        # Normalise simple Z suffix first.
        candidate = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw

        # First attempt: let datetime.fromisoformat handle common variants
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            # Heuristic repairs for legacy/broken formats.
            repaired = candidate
            # Case 1: legacy "YYYY-MM-DD HH:MM:SST00:00:00" -> strip trailing part
            if " " in repaired and "T" in repaired:
                base = repaired.split("T", 1)[0]
                try:
                    dt = datetime.fromisoformat(base)
                except ValueError:
                    # Fall back to replacing the space with 'T'
                    base = base.replace(" ", "T", 1)
                    dt = datetime.fromisoformat(base)
            else:
                # Case 2: "YYYY-MM-DD HH:MM:SS" without T – replace first space
                if "T" not in repaired and " " in repaired:
                    repaired = repaired.replace(" ", "T", 1)
                # Case 3: date-only "YYYY-MM-DD" – append midnight
                elif "T" not in repaired:
                    repaired = f"{repaired}T00:00:00"
                dt = datetime.fromisoformat(repaired)

        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported timestamp value: {value!r}")


sqlite3.register_adapter(datetime, lambda val: _ensure_utc_datetime(val).isoformat())
sqlite3.register_converter("TIMESTAMP", lambda raw: _ensure_utc_datetime(raw))
sqlite3.register_converter("DATETIME", lambda raw: _ensure_utc_datetime(raw))
sqlite3.register_converter("DATE", lambda raw: _ensure_utc_datetime(raw).date())


@dataclass
class BackfillStats:
    validated: int = 0
    failed: int = 0
    archived: int = 0
    errors: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "validated": self.validated,
            "failed": self.failed,
            "archived": self.archived,
            "errors": self.errors,
        }


def ensure_archived_status(conn: sqlite3.Connection) -> None:
    """Ensure the llm_signals table supports the 'archived' status."""
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_signals'"
    )
    row = cursor.fetchone()
    if not row:
        return
    table_sql = row[0] or ""
    if "archived" in table_sql.lower():
        return

    logger.info("Migrating llm_signals table to support 'archived' status...")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ALTER TABLE llm_signals RENAME TO llm_signals_old")
    conn.execute(
        """
        CREATE TABLE llm_signals (
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
            validation_status TEXT DEFAULT 'pending' CHECK(validation_status IN ('pending', 'validated', 'failed', 'executed', 'archived')),
            actual_return REAL,
            backtest_annual_return REAL,
            backtest_sharpe REAL,
            backtest_alpha REAL,
            latency_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, signal_date, model_name)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO llm_signals (
            id, ticker, signal_date, action, confidence, reasoning, model_name,
            entry_price, target_price, stop_loss, position_size, validation_status,
            actual_return, backtest_annual_return, backtest_sharpe, backtest_alpha,
            latency_seconds, created_at
        )
        SELECT
            id, ticker, signal_date, action, confidence, reasoning, model_name,
            entry_price, target_price, stop_loss, position_size, validation_status,
            actual_return, backtest_annual_return, backtest_sharpe, backtest_alpha,
            latency_seconds, created_at
        FROM llm_signals_old
        """
    )
    conn.execute("DROP TABLE llm_signals_old")
    conn.execute("PRAGMA foreign_keys = ON")
    logger.info("llm_signals migration complete.")


def load_market_data(
    conn: sqlite3.Connection,
    ticker: str,
    signal_date: datetime,
    lookback_days: int,
) -> pd.DataFrame:
    """Load OHLCV data for the specified ticker around the signal date."""
    signal_date = _ensure_utc_datetime(signal_date)
    start_date = (signal_date - timedelta(days=lookback_days)).date().isoformat()
    end_date = signal_date.date().isoformat()
    query = """
        SELECT CAST(date AS TEXT) AS date, open, high, low, close, volume
        FROM ohlcv_data
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """
    df = pd.read_sql_query(
        query,
        conn,
        params=(ticker, start_date, end_date),
        parse_dates=["date"],
    )
    if df.empty:
        return df

    df = df.set_index("date").rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return df


def build_signal_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a database row into the structure expected by the validator."""
    keys = {key: True for key in row.keys()}
    risk_level = row["risk_level"] if "risk_level" in keys and row["risk_level"] else "medium"
    timestamp_source = (
        row["signal_timestamp"]
        if "signal_timestamp" in keys and row["signal_timestamp"]
        else row["signal_date"]
    )
    timestamp_iso = _ensure_utc_datetime(timestamp_source).isoformat()
    return {
        "ticker": row["ticker"],
        "action": row["action"],
        "confidence": row["confidence"] if row["confidence"] is not None else 0.5,
        "reasoning": row["reasoning"] or "",
        "risk_level": risk_level,
        "signal_timestamp": timestamp_iso,
    }


def update_signal_status(
    conn: sqlite3.Connection,
    signal_id: int,
    status: str,
) -> None:
    conn.execute(
        "UPDATE llm_signals SET validation_status = ? WHERE id = ?", (status, signal_id)
    )


def insert_validation_record(
    db: DatabaseManager,
    signal_id: int,
    result: Any,
    validator_version: str,
) -> None:
    payload = {
        "validator_version": validator_version,
        "confidence_score": float(result.confidence_score),
        "recommendation": result.recommendation,
        "warnings": result.warnings,
        "quality_metrics": getattr(result, "layer_results", {}),
    }
    db.save_signal_validation(signal_id, payload)


def backfill_pending_signals(
    db_path: Path,
    lookback_days: int = 60,
    portfolio_value: float = 10_000.0,
    backtest_days: int = 30,
) -> Dict[str, Any]:
    """Backfill pending signals and return processing summary."""
    stats = BackfillStats()
    validator = SignalValidator()
    processed_signal_ids: List[int] = []

    with DatabaseManager(str(db_path)) as db:
        ensure_archived_status(db.conn)

        db.conn.row_factory = sqlite3.Row
        pending_rows = db.conn.execute(
            """
            SELECT
                id,
                ticker,
                CAST(signal_date AS TEXT) AS signal_date,
                action,
                confidence,
                reasoning,
                model_name,
                entry_price,
                validation_status,
                latency_seconds
            FROM llm_signals
            WHERE validation_status = 'pending'
            ORDER BY signal_date ASC
            """
        ).fetchall()

        logger.info("Found %s pending signals to backfill.", len(pending_rows))

        for row in pending_rows:
            signal_id = row["id"]
            ticker = row["ticker"]
            raw_date = row["signal_date"]
            if isinstance(raw_date, datetime):
                signal_date = raw_date
            else:
                raw_str = str(raw_date)
                try:
                    signal_date = datetime.fromisoformat(raw_str)
                except ValueError:
                    signal_date = datetime.strptime(raw_str, "%Y-%m-%d")

            market_data = load_market_data(db.conn, ticker, signal_date, lookback_days)
            if market_data.empty or len(market_data) < 5:
                logger.warning(
                    "Insufficient market data for %s on %s; marking archived.",
                    ticker,
                    row["signal_date"],
                )
                update_signal_status(db.conn, signal_id, "archived")
                stats.archived += 1
                processed_signal_ids.append(signal_id)
                continue

            signal = build_signal_dict(row)
            try:
                result = validator.validate_llm_signal(
                    signal, market_data, portfolio_value
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Validation error for %s on %s: %s",
                    ticker,
                    row["signal_date"],
                    exc,
                )
                stats.errors += 1
                continue

            recommendation = (result.recommendation or "").upper()
            is_valid = bool(result.is_valid)
            if not is_valid or recommendation in {"REJECT", "HOLD"}:
                update_signal_status(db.conn, signal_id, "failed")
                stats.failed += 1
            else:
                update_signal_status(db.conn, signal_id, "validated")
                stats.validated += 1

            insert_validation_record(db, signal_id, result, validator_version="v2-backfill")
            processed_signal_ids.append(signal_id)

        db.conn.commit()

        backtest_summary: Dict[str, Any] = {}
        if backtest_days > 0:
            now_utc = datetime.now(UTC)
            cutoff = (now_utc - timedelta(days=backtest_days)).date().isoformat()
            recent_signals = db.conn.execute(
                """
                SELECT *
                FROM llm_signals
                WHERE signal_date >= ?
                ORDER BY signal_date ASC
                """,
                (cutoff,),
            ).fetchall()

            signals_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
            for row in recent_signals:
                timestamp_source = row["signal_timestamp"] if "signal_timestamp" in row.keys() and row["signal_timestamp"] else row["signal_date"]
                timestamp_iso = _ensure_utc_datetime(timestamp_source).isoformat()
                signals_by_ticker.setdefault(row["ticker"], []).append(
                    {
                        "action": row["action"],
                        "confidence": row["confidence"] or 0.5,
                        "ticker": row["ticker"],
                        "signal_timestamp": timestamp_iso,
                        "risk_level": "medium",
                    }
                )

            metrics: Dict[str, Any] = {}
            for ticker, signals in signals_by_ticker.items():
                if not signals:
                    continue
                last_date = max(
                    _ensure_utc_datetime(sig["signal_timestamp"]) for sig in signals
                )
                price_df = load_market_data(db.conn, ticker, last_date, backtest_days * 2)
                if price_df.empty:
                    continue
                report = validator.backtest_signal_quality(
                    signals=signals,
                    actual_prices=price_df,
                    lookback_days=backtest_days,
                )
                metrics[ticker] = {
                    "hit_rate": report.hit_rate,
                    "profit_factor": report.profit_factor,
                    "sharpe_ratio": report.sharpe_ratio,
                    "trades": report.trades_analyzed,
                    "recommendation": report.recommendation,
                    "statistical": report.statistical_summary,
                    "autocorrelation": report.autocorrelation,
                    "bootstrap": report.bootstrap_intervals,
                }
            backtest_summary = metrics

    return {
        "stats": stats.as_dict(),
        "processed_signal_ids": processed_signal_ids,
        "backtest": backtest_summary,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill LLM signal validation metadata and recompute accuracy metrics."
    )
    parser.add_argument(
        "--db-path",
        default="data/portfolio_maximizer.db",
        help="Path to SQLite database (default: data/portfolio_maximizer.db)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=60,
        help="Number of historical days to load for validation (default: 60)",
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=10_000.0,
        help="Portfolio value used for position sizing during validation.",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=30,
        help="Window (in days) for recomputing backtest accuracy (0 disables).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=str(logs_dir / "backfill_signal_validation.log"),
        filemode="a",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)
    logging.captureWarnings(True)
    warnings.simplefilter("default")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    db_path = Path(args.db_path)

    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        raise SystemExit(1)

    summary = backfill_pending_signals(
        db_path=db_path,
        lookback_days=args.lookback_days,
        portfolio_value=args.portfolio_value,
        backtest_days=args.backtest_days,
    )

    logger.info("Backfill summary: %s", summary["stats"])
    if summary["backtest"]:
        logger.info("Backtest metrics:")
        for ticker, metrics in summary["backtest"].items():
            logger.info("  %s → %s", ticker, metrics)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
