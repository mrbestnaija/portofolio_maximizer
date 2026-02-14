"""
Database security tests.

Tests for database file permissions and secure database operations.
"""

import os
import stat
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from etl.database_manager import DatabaseManager


class TestDatabaseFilePermissions:
    """Test database file permission security."""

    def test_new_database_has_secure_permissions(self, tmp_path):
        """New database should be created with secure permissions (0o600)."""
        db_path = tmp_path / "test_db.db"
        manager = DatabaseManager(str(db_path))

        # Check file exists
        assert db_path.exists()

        # Check permissions (on Unix-like systems)
        if os.name != 'nt':  # Skip on Windows
            file_stat = os.stat(db_path)
            permissions = stat.filemode(file_stat.st_mode)
            # Should be -rw------- (600) - owner read/write only
            assert permissions == '-rw-------'

    def test_existing_database_permissions_set(self, tmp_path):
        """Existing database should have permissions set to secure."""
        db_path = tmp_path / "existing_db.db"

        # Create database file first
        conn = sqlite3.connect(str(db_path))
        conn.close()

        # Set permissive permissions initially
        if os.name != 'nt':
            os.chmod(db_path, 0o644)

        # Initialize DatabaseManager - should set secure permissions
        manager = DatabaseManager(str(db_path))

        # Verify permissions are now secure
        if os.name != 'nt':
            file_stat = os.stat(db_path)
            permissions = stat.filemode(file_stat.st_mode)
            assert permissions == '-rw-------'

    def test_database_permissions_on_windows(self, tmp_path):
        """Database creation should work on Windows (permissions handled by OS)."""
        db_path = tmp_path / "windows_test_db.db"
        manager = DatabaseManager(str(db_path))

        # Should work without errors
        assert db_path.exists()

        # Verify database is functional
        assert manager.conn is not None
        assert manager.cursor is not None


@pytest.mark.security
class TestDatabaseSecurityIntegration:
    """Integration tests for database security."""

    def test_database_operations_with_secure_permissions(self, tmp_path):
        """Database operations should work correctly with secure permissions."""
        db_path = tmp_path / "secure_db.db"
        manager = DatabaseManager(str(db_path))

        # Test database operations
        import pandas as pd
        from datetime import datetime

        test_data = pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Low': [99.0],
            'Close': [103.0],
            'Volume': [1000000],
            'Adj Close': [103.0]
        }, index=[datetime(2025, 1, 1)])
        test_data.attrs['ticker'] = 'TEST'

        # Save data
        rows_saved = manager.save_ohlcv_data(test_data, source='test')
        assert rows_saved == 1

        # Verify data saved
        assert db_path.exists()

        # Verify permissions remain secure
        if os.name != 'nt':
            file_stat = os.stat(db_path)
            permissions = stat.filemode(file_stat.st_mode)
            assert permissions == '-rw-------'

    def test_runtime_guardrails_block_dangerous_pragma(self, tmp_path, monkeypatch):
        """Operational DB connections should block dangerous PRAGMA toggles."""
        monkeypatch.setenv("SECURITY_SQLITE_GUARDRAILS", "1")
        monkeypatch.setenv("SECURITY_SQLITE_GUARDRAILS_HARD_FAIL", "1")
        db_path = tmp_path / "guardrails_pragma.db"
        manager = DatabaseManager(str(db_path))
        with pytest.raises(sqlite3.DatabaseError):
            manager.conn.execute("PRAGMA ignore_check_constraints=ON")

    def test_runtime_guardrails_block_schema_drop(self, tmp_path, monkeypatch):
        """Operational DB connections should deny destructive schema operations."""
        monkeypatch.setenv("SECURITY_SQLITE_GUARDRAILS", "1")
        monkeypatch.setenv("SECURITY_SQLITE_GUARDRAILS_HARD_FAIL", "1")
        db_path = tmp_path / "guardrails_drop.db"
        manager = DatabaseManager(str(db_path))
        with pytest.raises(sqlite3.DatabaseError):
            manager.conn.execute("DROP TABLE trade_executions")
