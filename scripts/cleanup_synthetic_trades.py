#!/usr/bin/env python3
"""
cleanup_synthetic_trades.py
---------------------------

Tag contaminated test/synthetic data in the database to separate it from
production trades. This enables accurate profitability reporting.

Contaminated data includes:
- Trades with NULL/empty data_source or execution_mode
- Trades on synthetic test tickers (SYN0, SYN1, etc.)
- Trades without proper audit trail (missing pipeline_id/run_id)

Usage:
    python scripts/cleanup_synthetic_trades.py
    python scripts/cleanup_synthetic_trades.py --dry-run
    python scripts/cleanup_synthetic_trades.py --db-path data/portfolio_maximizer.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/cleanup_synthetic_trades.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "portfolio_maximizer.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseCleanup:
    """Clean up contaminated data in the database."""

    def __init__(self, db_path: Path, dry_run: bool = False):
        self.db_path = db_path
        self.dry_run = dry_run
        self.conn: sqlite3.Connection | None = None

    def __enter__(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None and not self.dry_run:
                self.conn.commit()
                logger.info("Changes committed to database")
            else:
                if self.dry_run:
                    logger.info("DRY RUN - No changes committed")
                else:
                    self.conn.rollback()
                    logger.warning("Transaction rolled back due to error")
            self.conn.close()

    def add_audit_columns(self) -> None:
        """Add is_test_data and audit_notes columns if they don't exist."""
        logger.info("Adding audit columns to trade_executions table")

        # Check if columns already exist
        cursor = self.conn.execute("PRAGMA table_info(trade_executions)")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        if "is_test_data" not in existing_cols:
            logger.info("Adding is_test_data column")
            self.conn.execute(
                "ALTER TABLE trade_executions ADD COLUMN is_test_data BOOLEAN DEFAULT FALSE"
            )

        if "audit_notes" not in existing_cols:
            logger.info("Adding audit_notes column")
            self.conn.execute(
                "ALTER TABLE trade_executions ADD COLUMN audit_notes TEXT"
            )

    def _column_exists(self, column_name: str) -> bool:
        """Check if a column exists in trade_executions table."""
        cursor = self.conn.execute("PRAGMA table_info(trade_executions)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        return column_name in existing_cols

    def analyze_contamination(self) -> Dict[str, Any]:
        """Analyze the extent of data contamination."""
        logger.info("Analyzing data contamination...")

        stats = {}

        # Total trades
        cursor = self.conn.execute("SELECT COUNT(*) as cnt FROM trade_executions")
        stats["total_trades"] = cursor.fetchone()["cnt"]

        # NULL data_source
        if self._column_exists("data_source"):
            cursor = self.conn.execute(
                """
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE data_source IS NULL OR data_source = ''
                """
            )
            stats["null_source"] = cursor.fetchone()["cnt"]
        else:
            stats["null_source"] = 0
            logger.warning("data_source column does not exist - assuming all trades lack provenance")

        # NULL execution_mode
        if self._column_exists("execution_mode"):
            cursor = self.conn.execute(
                """
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE execution_mode IS NULL OR execution_mode = ''
                """
            )
            stats["null_exec_mode"] = cursor.fetchone()["cnt"]
        else:
            stats["null_exec_mode"] = 0
            logger.warning("execution_mode column does not exist")

        # Synthetic tickers
        cursor = self.conn.execute(
            """
            SELECT COUNT(*) as cnt FROM trade_executions
            WHERE ticker LIKE 'SYN%'
            """
        )
        stats["synthetic_tickers"] = cursor.fetchone()["cnt"]

        # NULL pipeline_id (optional column)
        if self._column_exists("pipeline_id"):
            cursor = self.conn.execute(
                """
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE pipeline_id IS NULL OR pipeline_id = ''
                """
            )
            stats["null_pipeline_id"] = cursor.fetchone()["cnt"]
        else:
            stats["null_pipeline_id"] = 0

        # NULL run_id (optional column)
        if self._column_exists("run_id"):
            cursor = self.conn.execute(
                """
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE run_id IS NULL OR run_id = ''
                """
            )
            stats["null_run_id"] = cursor.fetchone()["cnt"]
        else:
            stats["null_run_id"] = 0

        # Already tagged
        if self._column_exists("is_test_data"):
            cursor = self.conn.execute(
                """
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE is_test_data = TRUE
                """
            )
            stats["already_tagged"] = cursor.fetchone()["cnt"]
        else:
            stats["already_tagged"] = 0

        return stats

    def tag_null_sources(self) -> int:
        """Tag trades with NULL or empty data_source."""
        logger.info("Tagging trades with NULL data_source...")

        # Check if required columns exist
        has_data_source = self._column_exists("data_source")
        has_exec_mode = self._column_exists("execution_mode")

        if not has_data_source and not has_exec_mode:
            logger.warning("Neither data_source nor execution_mode columns exist - skipping")
            return 0

        # Build WHERE clause based on available columns
        where_conditions = []
        if has_data_source:
            where_conditions.append("(data_source IS NULL OR data_source = '')")
        if has_exec_mode:
            where_conditions.append("(execution_mode IS NULL OR execution_mode = '')")

        where_clause = " OR ".join(where_conditions)

        query = f"""
        UPDATE trade_executions
        SET is_test_data = TRUE,
            audit_notes = CASE
                WHEN audit_notes IS NULL THEN 'Missing data_source/execution_mode - likely test data'
                ELSE audit_notes || '; Missing data_source/execution_mode'
            END
        WHERE ({where_clause})
          AND (is_test_data IS NULL OR is_test_data = FALSE)
        """

        if self.dry_run:
            # In dry-run, is_test_data might not exist yet
            if self._column_exists("is_test_data"):
                count_query = f"""
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE ({where_clause})
                  AND (is_test_data IS NULL OR is_test_data = FALSE)
                """
            else:
                count_query = f"""
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE ({where_clause})
                """
            cursor = self.conn.execute(count_query)
            count = cursor.fetchone()["cnt"]
            logger.info(f"DRY RUN: Would tag {count} trades with NULL source")
            return count

        cursor = self.conn.execute(query)
        count = cursor.rowcount
        logger.info(f"Tagged {count} trades with NULL data_source/execution_mode")
        return count

    def tag_synthetic_tickers(self) -> int:
        """Tag trades on synthetic test tickers."""
        logger.info("Tagging trades on synthetic test tickers...")

        query = """
        UPDATE trade_executions
        SET is_test_data = TRUE,
            audit_notes = CASE
                WHEN audit_notes IS NULL THEN 'Synthetic test ticker'
                ELSE audit_notes || '; Synthetic test ticker'
            END
        WHERE ticker LIKE 'SYN%'
          AND (is_test_data IS NULL OR is_test_data = FALSE)
        """

        if self.dry_run:
            # In dry-run, is_test_data might not exist yet
            if self._column_exists("is_test_data"):
                count_query = """
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE ticker LIKE 'SYN%'
                  AND (is_test_data IS NULL OR is_test_data = FALSE)
                """
            else:
                count_query = """
                SELECT COUNT(*) as cnt FROM trade_executions
                WHERE ticker LIKE 'SYN%'
                """
            cursor = self.conn.execute(count_query)
            count = cursor.fetchone()["cnt"]
            logger.info(f"DRY RUN: Would tag {count} synthetic ticker trades")
            return count

        cursor = self.conn.execute(query)
        count = cursor.rowcount
        logger.info(f"Tagged {count} synthetic ticker trades")
        return count

    def create_production_view(self) -> None:
        """Create or replace production_trades view."""
        logger.info("Creating production_trades view...")

        # Drop existing view if it exists
        self.conn.execute("DROP VIEW IF EXISTS production_trades")

        # Create view that excludes test data
        query = """
        CREATE VIEW production_trades AS
        SELECT * FROM trade_executions
        WHERE is_test_data = FALSE OR is_test_data IS NULL
        """

        self.conn.execute(query)
        logger.info("Created production_trades view")

    def get_synthetic_ticker_details(self) -> List[Dict[str, Any]]:
        """Get details of synthetic ticker trades for reporting."""
        cursor = self.conn.execute(
            """
            SELECT
                ticker,
                COUNT(*) as trade_count,
                SUM(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl ELSE 0 END) as total_pnl,
                MIN(created_at) as first_trade,
                MAX(created_at) as last_trade
            FROM trade_executions
            WHERE ticker LIKE 'SYN%'
            GROUP BY ticker
            ORDER BY ticker
            """
        )

        results = []
        for row in cursor.fetchall():
            results.append({
                "ticker": row["ticker"],
                "trade_count": row["trade_count"],
                "total_pnl": row["total_pnl"] or 0.0,
                "first_trade": row["first_trade"],
                "last_trade": row["last_trade"],
            })

        return results

    def print_cleanup_report(self, stats_before: Dict[str, Any], stats_after: Dict[str, Any]) -> None:
        """Print a detailed cleanup report."""
        print("\n" + "=" * 80)
        print("DATABASE CLEANUP REPORT")
        print("=" * 80)
        print(f"\nDatabase: {self.db_path}")
        print(f"Timestamp: {_utc_now_iso()}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")

        print("\n--- DATA CONTAMINATION ANALYSIS ---")
        print(f"Total trades:              {stats_before['total_trades']:>6}")
        print(f"NULL data_source:          {stats_before['null_source']:>6} ({stats_before['null_source'] / stats_before['total_trades'] * 100:.1f}%)")
        print(f"NULL execution_mode:       {stats_before['null_exec_mode']:>6} ({stats_before['null_exec_mode'] / stats_before['total_trades'] * 100:.1f}%)")
        print(f"Synthetic tickers:         {stats_before['synthetic_tickers']:>6} ({stats_before['synthetic_tickers'] / stats_before['total_trades'] * 100:.1f}%)")
        print(f"NULL pipeline_id:          {stats_before['null_pipeline_id']:>6}")
        print(f"NULL run_id:               {stats_before['null_run_id']:>6}")

        print("\n--- CLEANUP ACTIONS ---")
        print(f"Already tagged:            {stats_before['already_tagged']:>6}")
        print(f"Newly tagged:              {stats_after['already_tagged'] - stats_before['already_tagged']:>6}")
        print(f"Total tagged after:        {stats_after['already_tagged']:>6} ({stats_after['already_tagged'] / stats_after['total_trades'] * 100:.1f}%)")

        # Synthetic ticker details
        synthetic_details = self.get_synthetic_ticker_details()
        if synthetic_details:
            print("\n--- SYNTHETIC TICKER BREAKDOWN ---")
            total_synthetic_pnl = sum(d["total_pnl"] for d in synthetic_details)
            for detail in synthetic_details:
                print(f"{detail['ticker']:>6}: {detail['trade_count']:>3} trades, P&L: ${detail['total_pnl']:>8.2f}")
            print(f"{'TOTAL':>6}: {sum(d['trade_count'] for d in synthetic_details):>3} trades, P&L: ${total_synthetic_pnl:>8.2f}")

        print("\n--- PRODUCTION DATA METRICS ---")
        production_count = stats_after['total_trades'] - stats_after['already_tagged']
        print(f"Production trades:         {production_count:>6} ({production_count / stats_after['total_trades'] * 100:.1f}%)")
        print(f"Test/Synthetic trades:     {stats_after['already_tagged']:>6} ({stats_after['already_tagged'] / stats_after['total_trades'] * 100:.1f}%)")

        print("\n" + "=" * 80)

        if self.dry_run:
            print("\nNOTE: This was a DRY RUN. No changes were made to the database.")
            print("Run without --dry-run to apply these changes.")
        else:
            print("\nChanges have been committed to the database.")
            print("Production metrics will now exclude test/synthetic data.")

        print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag contaminated test/synthetic data in the database."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to portfolio database (default: data/portfolio_maximizer.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze contamination without making changes",
    )
    args = parser.parse_args()

    # Ensure logs directory exists
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting database cleanup (dry_run={args.dry_run})")
    logger.info(f"Database: {args.db_path}")

    try:
        with DatabaseCleanup(args.db_path, dry_run=args.dry_run) as cleanup:
            # Analyze before
            stats_before = cleanup.analyze_contamination()

            # Add audit columns if needed (only in live mode)
            if not args.dry_run:
                cleanup.add_audit_columns()

            # Tag contaminated data
            null_source_count = cleanup.tag_null_sources()
            synthetic_count = cleanup.tag_synthetic_tickers()

            # Create production view (only in live mode)
            if not args.dry_run:
                cleanup.create_production_view()

            # Analyze after
            stats_after = cleanup.analyze_contamination()

            # Print report
            cleanup.print_cleanup_report(stats_before, stats_after)

            logger.info("Database cleanup completed successfully")

    except Exception as exc:
        logger.error(f"Database cleanup failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
