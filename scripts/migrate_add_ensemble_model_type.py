#!/usr/bin/env python3
"""
Database migration: Add 'ENSEMBLE' to model_type CHECK constraint.

This script updates the time_series_forecasts table to allow 'ENSEMBLE' as a model_type.
Since SQLite doesn't support ALTER CONSTRAINT, we recreate the table with the new constraint.

Usage:
    python scripts/migrate_add_ensemble_model_type.py
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path("data/portfolio_maximizer.db")
BUSY_TIMEOUT_MS = 5000

def migrate_database():
    """Add 'ENSEMBLE' to model_type CHECK constraint."""

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return False

    conn = sqlite3.connect(str(DB_PATH), timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    cursor = conn.cursor()

    try:
        print("=" * 80)
        print("DATABASE MIGRATION: Add 'ENSEMBLE' to model_type")
        print("=" * 80)

        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='time_series_forecasts'
        """)
        if not cursor.fetchone():
            print("ERROR: time_series_forecasts table does not exist!")
            return False

        # Count existing records
        cursor.execute("SELECT COUNT(*) FROM time_series_forecasts")
        total_records = cursor.fetchone()[0]
        print(f"\n1. Current Records: {total_records}")

        # Check if ENSEMBLE records already exist (shouldn't, but check anyway)
        cursor.execute("""
            SELECT COUNT(*) FROM time_series_forecasts
            WHERE model_type = 'ENSEMBLE'
        """)
        ensemble_count = cursor.fetchone()[0]
        print(f"   ENSEMBLE records (before): {ensemble_count}")

        # SQLite doesn't support ALTER CONSTRAINT, so we need to:
        # 1. Create new table with updated constraint
        # 2. Copy all data
        # 3. Drop old table
        # 4. Rename new table

        print("\n2. Creating new table with updated constraint...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_series_forecasts_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                forecast_date DATE NOT NULL,
                model_type TEXT NOT NULL CHECK(model_type IN ('SARIMAX', 'GARCH', 'COMBINED', 'ENSEMBLE', 'SAMOSSA', 'MSSA_RL')),
                forecast_horizon INTEGER NOT NULL,
                forecast_value REAL NOT NULL,
                lower_ci REAL,
                upper_ci REAL,
                volatility REAL,
                model_order TEXT,
                aic REAL,
                bic REAL,
                diagnostics TEXT,
                regression_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("   [OK] New table created")

        print("\n3. Copying all data to new table...")
        cursor.execute("""
            INSERT INTO time_series_forecasts_new
                (id, ticker, forecast_date, model_type, forecast_horizon,
                 forecast_value, lower_ci, upper_ci, volatility, model_order,
                 aic, bic, diagnostics, regression_metrics, created_at)
            SELECT
                id, ticker, forecast_date, model_type, forecast_horizon,
                forecast_value, lower_ci, upper_ci, volatility, model_order,
                aic, bic, diagnostics, regression_metrics, created_at
            FROM time_series_forecasts
        """)
        copied_records = cursor.rowcount
        print(f"   [OK] Copied {copied_records} records")

        # Verify data integrity
        cursor.execute("SELECT COUNT(*) FROM time_series_forecasts_new")
        new_count = cursor.fetchone()[0]
        if new_count != total_records:
            print(f"   [ERROR] Record count mismatch ({new_count} vs {total_records})")
            conn.rollback()
            return False

        print("\n4. Dropping old table...")
        cursor.execute("DROP TABLE time_series_forecasts")
        print("   [OK] Old table dropped")

        print("\n5. Renaming new table...")
        cursor.execute("ALTER TABLE time_series_forecasts_new RENAME TO time_series_forecasts")
        print("   [OK] New table renamed")

        # Recreate indexes if they existed
        print("\n6. Recreating indexes...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_ticker_date
            ON time_series_forecasts(ticker, forecast_date, model_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_model_type
            ON time_series_forecasts(model_type)
        """)
        print("   [OK] Indexes recreated")

        # Commit transaction
        conn.commit()

        print("\n7. Verifying migration...")
        cursor.execute("SELECT COUNT(*) FROM time_series_forecasts")
        final_count = cursor.fetchone()[0]
        print(f"   Final record count: {final_count}")

        # Test that ENSEMBLE is now allowed
        print("\n8. Testing ENSEMBLE constraint...")
        try:
            cursor.execute("""
                INSERT INTO time_series_forecasts
                (ticker, forecast_date, model_type, forecast_horizon, forecast_value)
                VALUES ('TEST', '2026-01-20', 'ENSEMBLE', 1, 100.0)
            """)
            cursor.execute("DELETE FROM time_series_forecasts WHERE ticker = 'TEST'")
            conn.commit()
            print("   [OK] ENSEMBLE model_type is now allowed!")
        except sqlite3.IntegrityError as e:
            print(f"   [ERROR] ENSEMBLE still blocked: {e}")
            return False

        print("\n" + "=" * 80)
        print("[SUCCESS] MIGRATION SUCCESSFUL!")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  - Records migrated: {final_count}")
        print(f"  - ENSEMBLE model_type: ENABLED")
        print(f"  - Database: {DB_PATH}")
        print()

        return True

    except Exception as e:
        print(f"\n[ERROR] MIGRATION FAILED: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)
