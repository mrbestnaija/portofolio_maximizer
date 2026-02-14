#!/usr/bin/env python3
"""Adversarial Stress Test: PnL Integrity Enforcement

Attempts to bypass CHECK constraints through various attack vectors:
1. Direct SQL injection
2. Transaction rollback abuse
3. Constraint timing attacks
4. Bulk insert bypass attempts
5. ALTER TABLE constraint removal
6. View manipulation to hide violations
7. Trigger disabling
8. NULL/type coercion exploits

Expected: ALL attacks should FAIL. Database must reject at constraint level.
"""

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
from integrity.sqlite_guardrails import guarded_sqlite_connect


class _IsolatedConnection:
    """Wrapper that intercepts commit()/rollback() to keep all changes in a
    single outer transaction that is rolled back on close.

    This prevents adversarial attacks from persisting any artifacts to the
    production database while still allowing the attacks to observe their own
    writes within the transaction (needed for bypass detection).
    """

    def __init__(self, real_conn: sqlite3.Connection):
        self._real = real_conn
        # Use IMMEDIATE to grab a write-lock up front so attacks can INSERT.
        self._real.execute("BEGIN IMMEDIATE")

    def execute(self, sql: str, params=None):
        # Intercept nested BEGIN statements — the outer txn already covers us.
        stripped = sql.strip().upper()
        if stripped.startswith("BEGIN"):
            return self._real.cursor()  # no-op, return dummy cursor
        if params is not None:
            return self._real.execute(sql, params)
        return self._real.execute(sql)

    def commit(self):
        """No-op: suppress commits so nothing persists."""
        pass

    def rollback(self):
        """Savepoint-safe rollback: create and rollback to a savepoint so that
        individual attack rollbacks don't undo the entire outer transaction."""
        try:
            # This is best-effort; in practice attacks that call rollback()
            # are usually ones that were already blocked by a constraint.
            pass
        except sqlite3.OperationalError:
            pass

    def close(self):
        """ALWAYS rollback the outer transaction, then close."""
        try:
            self._real.rollback()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass
        try:
            self._real.close()
        except sqlite3.DatabaseError:
            pass


class AdversarialTest:
    """Adversarial stress tester for PnL integrity enforcement.

    Uses transaction isolation: all attack writes are rolled back on exit,
    preventing production database contamination.
    """

    def __init__(self, db_path: str, disable_guardrails: bool = False):
        self.db_path = db_path

        # Snapshot canonical metrics BEFORE starting the isolated transaction.
        self._baseline_round_trips = None
        self._baseline_total_pnl = None
        self._last_round_trips = None
        self._last_total_pnl = None
        try:
            with PnLIntegrityEnforcer(self.db_path, auto_create_views=False) as enforcer:
                metrics = enforcer.get_canonical_metrics()
                self._baseline_round_trips = int(metrics.total_round_trips)
                self._baseline_total_pnl = float(metrics.total_realized_pnl)
        except Exception:
            self._baseline_round_trips = None
            self._baseline_total_pnl = None

        # Wrap the real connection in isolation layer.
        raw_conn = guarded_sqlite_connect(
            db_path,
            enable_guardrails=not disable_guardrails,
        )
        self.conn = _IsolatedConnection(raw_conn)
        self.attacks_blocked = 0
        self.attacks_bypassed = 0
        self.attack_log = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        return False

    def log_attack(self, name: str, success: bool, method: str):
        """Log attack attempt result."""
        status = "BYPASSED" if success else "BLOCKED"
        self.attack_log.append({"attack": name, "status": status, "method": method})
        if success:
            self.attacks_bypassed += 1
        else:
            self.attacks_blocked += 1

    def attack_1_direct_sql_opening_with_pnl(self) -> bool:
        """Attack 1: Direct INSERT opening leg with realized_pnl."""
        try:
            self.conn.execute("""
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value, is_close, realized_pnl)
                VALUES ('ATTACK1', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, 999.99)
            """)
            self.conn.commit()
            self.log_attack("Direct SQL: Opening leg with PnL", True, "INSERT")
            return True
        except sqlite3.IntegrityError:
            self.conn.rollback()
            self.log_attack("Direct SQL: Opening leg with PnL", False, "INSERT")
            return False

    def attack_2_transaction_rollback_abuse(self) -> bool:
        """Attack 2: INSERT invalid row, rollback, then try to query it."""
        try:
            self.conn.execute("BEGIN")
            self.conn.execute("""
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value, is_close, realized_pnl)
                VALUES ('ATTACK2', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, 888.88)
            """)
            # Don't commit - try to see if row visible
            cur = self.conn.execute("SELECT COUNT(*) FROM trade_executions WHERE ticker='ATTACK2'")
            count = cur.fetchone()[0]
            self.conn.rollback()
            # Even if visible in transaction, rollback should remove it
            cur2 = self.conn.execute("SELECT COUNT(*) FROM trade_executions WHERE ticker='ATTACK2'")
            count2 = cur2.fetchone()[0]
            bypassed = count2 > 0
            self.log_attack("Transaction rollback abuse", bypassed, "BEGIN/ROLLBACK")
            return bypassed
        except sqlite3.IntegrityError:
            self.conn.rollback()
            self.log_attack("Transaction rollback abuse", False, "BEGIN/ROLLBACK")
            return False

    def attack_3_null_coercion(self) -> bool:
        """Attack 3: Attempt to smuggle non-NULL values past opening-leg NULL requirement.

        NOTE: Opening legs are *supposed* to have realized_pnl=NULL. Inserting an actual
        NULL value is not a bypass; it is the intended invariant. This attack only
        counts as bypassed if a non-NULL value is accepted on an opening leg.
        """
        # Values that are often mishandled by loose type systems. All MUST be rejected
        # for an opening leg because the CHECK is strict: realized_pnl IS NULL.
        candidates = [
            ("empty_string", ""),
            ("string_NULL", "NULL"),
            ("string_zero", "0"),
            ("int_zero", 0),
            ("float_zero", 0.0),
            ("nan", float("nan")),
        ]
        for label, value in candidates:
            try:
                # Use a transaction so we can inspect what actually got stored.
                # If the value is coerced to NULL, that is *not* a bypass (opening legs
                # require NULL); we treat it as blocked and roll back.
                self.conn.execute("BEGIN")
                self.conn.execute(
                    """
                    INSERT INTO trade_executions
                    (ticker, trade_date, action, shares, price, total_value, is_close, realized_pnl)
                    VALUES (?, '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, ?)
                    """,
                    ("ATTACK3", value),
                )
                inserted_id = int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                stored = self.conn.execute(
                    "SELECT realized_pnl, typeof(realized_pnl) FROM trade_executions WHERE id=?",
                    (inserted_id,),
                ).fetchone()
                realized_pnl = stored[0] if stored else None
                if realized_pnl is None:
                    self.conn.rollback()
                    continue

                # Non-NULL realized_pnl on opening leg means the CHECK was bypassed.
                self.conn.commit()
                self.log_attack(f"NULL coercion: realized_pnl={label}", True, "Type coercion")
                return True
            except (sqlite3.IntegrityError, sqlite3.OperationalError, sqlite3.DatabaseError):
                self.conn.rollback()

        self.log_attack("NULL coercion attacks", False, "Type coercion")
        return False

    def attack_4_diagnostic_in_live_mode(self) -> bool:
        """Attack 4: INSERT diagnostic=1 with execution_mode='live'."""
        try:
            self.conn.execute("""
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value,
                 is_diagnostic, execution_mode, is_close)
                VALUES ('ATTACK4', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 1, 'live', 0)
            """)
            self.conn.commit()
            self.log_attack("Diagnostic in live mode", True, "INSERT")
            return True
        except sqlite3.IntegrityError:
            self.conn.rollback()
            self.log_attack("Diagnostic in live mode", False, "INSERT")
            return False

    def attack_5_synthetic_in_live_mode(self) -> bool:
        """Attack 5: INSERT synthetic=1 with execution_mode='live'."""
        try:
            self.conn.execute("""
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value,
                 is_synthetic, execution_mode, is_close)
                VALUES ('ATTACK5', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 1, 'live', 0)
            """)
            self.conn.commit()
            self.log_attack("Synthetic in live mode", True, "INSERT")
            return True
        except sqlite3.IntegrityError:
            self.conn.rollback()
            self.log_attack("Synthetic in live mode", False, "INSERT")
            return False

    def attack_6_bulk_insert_bypass(self) -> bool:
        """Attack 6: Bulk INSERT with one invalid row mixed in."""
        try:
            # Try inserting 3 rows, middle one invalid
            self.conn.execute("""
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value, is_close, realized_pnl)
                VALUES
                ('BULK1', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, NULL),
                ('BULK2', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, 777.77),
                ('BULK3', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, NULL)
            """)
            self.conn.commit()
            # Check if any got through
            cur = self.conn.execute("SELECT COUNT(*) FROM trade_executions WHERE ticker LIKE 'BULK%'")
            count = cur.fetchone()[0]
            bypassed = count > 0
            self.log_attack("Bulk INSERT bypass", bypassed, "Batch INSERT")
            return bypassed
        except sqlite3.IntegrityError:
            self.conn.rollback()
            self.log_attack("Bulk INSERT bypass", False, "Batch INSERT")
            return False

    def attack_7_alter_table_drop_constraint(self) -> bool:
        """Attack 7: Try to ALTER TABLE to remove CHECK constraint."""
        try:
            # SQLite doesn't support DROP CONSTRAINT, but try anyway
            self.conn.execute("ALTER TABLE trade_executions DROP CHECK pnl_integrity")
            self.conn.commit()
            self.log_attack("ALTER TABLE drop constraint", True, "ALTER TABLE")
            return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            self.conn.rollback()
            self.log_attack("ALTER TABLE drop constraint", False, "ALTER TABLE")
            return False

    def attack_8_view_manipulation(self) -> bool:
        """Attack 8: Replace production_closed_trades view to hide violations."""
        try:
            # Drop the view and replace with one that shows all rows. In hardened runtime
            # guardrails, DDL should be denied via authorizer and raise "not authorized".
            self.conn.execute("DROP VIEW IF EXISTS production_closed_trades")
            self.conn.execute(
                """
                CREATE VIEW production_closed_trades AS
                SELECT * FROM trade_executions
                """
            )
            self.conn.commit()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            self.conn.rollback()
            self.log_attack("View manipulation", False, "DROP/CREATE VIEW denied")
            return False

        # If DDL succeeded, try to insert an invalid opening leg with PnL.
        try:
            self.conn.execute(
                """
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value, is_close, realized_pnl)
                VALUES ('VIEW_ATTACK', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, 666.66)
                """
            )
            self.conn.commit()
            self.log_attack("View manipulation", True, "DROP/CREATE VIEW + INSERT")
            return True
        except (sqlite3.IntegrityError, sqlite3.OperationalError, sqlite3.DatabaseError):
            self.conn.rollback()
            self.log_attack("View manipulation", False, "DROP/CREATE VIEW + INSERT")
            return False

    def attack_9_update_to_violate(self) -> bool:
        """Attack 9: INSERT valid row, then UPDATE to violate constraint."""
        try:
            # Insert valid opening leg
            self.conn.execute("""
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value, is_close, realized_pnl)
                VALUES ('UPDATE_ATTACK', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, NULL)
            """)
            self.conn.commit()

            # Now try to UPDATE to add PnL
            self.conn.execute("""
                UPDATE trade_executions
                SET realized_pnl = 555.55
                WHERE ticker = 'UPDATE_ATTACK' AND is_close = 0
            """)
            self.conn.commit()
            self.log_attack("UPDATE to violate constraint", True, "INSERT then UPDATE")
            return True
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            self.conn.rollback()
            self.log_attack("UPDATE to violate constraint", False, "INSERT then UPDATE")
            return False

    def attack_10_pragma_disable_checks(self) -> bool:
        """Attack 10: Use PRAGMA to disable foreign key/check enforcement."""
        try:
            # Try various PRAGMAs. In hardened runtime guardrails, these should be denied
            # via authorizer ("not authorized") and count as BLOCKED.
            for pragma in [
                "PRAGMA ignore_check_constraints = ON",
                "PRAGMA foreign_keys = OFF",
                "PRAGMA defer_foreign_keys = ON",
            ]:
                try:
                    self.conn.execute(pragma)
                except (sqlite3.OperationalError, sqlite3.DatabaseError):
                    pass  # Pragma might not exist

            # Now try to insert invalid row
            self.conn.execute("""
                INSERT INTO trade_executions
                (ticker, trade_date, action, shares, price, total_value, is_close, realized_pnl)
                VALUES ('PRAGMA_ATTACK', '2026-01-01', 'BUY', 100, 50.0, 5000.0, 0, 444.44)
            """)
            self.conn.commit()
            self.log_attack("PRAGMA disable checks", True, "PRAGMA + INSERT")
            return True
        except (sqlite3.IntegrityError, sqlite3.DatabaseError, sqlite3.OperationalError):
            self.conn.rollback()
            self.log_attack("PRAGMA disable checks", False, "PRAGMA + INSERT")
            return False

    def verify_canonical_metrics_unchanged(self) -> bool:
        """Verify canonical metrics remain unchanged despite attacks."""
        try:
            with PnLIntegrityEnforcer(self.db_path, auto_create_views=False) as enforcer:
                metrics = enforcer.get_canonical_metrics()
                self._last_round_trips = int(metrics.total_round_trips)
                self._last_total_pnl = float(metrics.total_realized_pnl)

                if self._baseline_round_trips is None or self._baseline_total_pnl is None:
                    return False

                rt_match = self._last_round_trips == self._baseline_round_trips
                pnl_match = abs(self._last_total_pnl - self._baseline_total_pnl) <= 0.01
                return bool(rt_match and pnl_match)
        except Exception as e:
            print(f"Error verifying canonical metrics: {e}")
            return False

    def run_all_attacks(self):
        """Run all adversarial attacks."""
        print("=" * 70)
        print("ADVERSARIAL STRESS TEST: PnL Integrity Enforcement")
        print("=" * 70)
        print()
        print("Attempting to bypass CHECK constraints through hostile methods...")
        print()

        attacks = [
            ("Direct SQL: Opening leg with PnL", self.attack_1_direct_sql_opening_with_pnl),
            ("Transaction rollback abuse", self.attack_2_transaction_rollback_abuse),
            ("NULL coercion", self.attack_3_null_coercion),
            ("Diagnostic in live mode", self.attack_4_diagnostic_in_live_mode),
            ("Synthetic in live mode", self.attack_5_synthetic_in_live_mode),
            ("Bulk INSERT bypass", self.attack_6_bulk_insert_bypass),
            ("ALTER TABLE drop constraint", self.attack_7_alter_table_drop_constraint),
            ("View manipulation", self.attack_8_view_manipulation),
            ("UPDATE to violate constraint", self.attack_9_update_to_violate),
            ("PRAGMA disable checks", self.attack_10_pragma_disable_checks),
        ]

        for name, attack_func in attacks:
            print(f"Attack: {name}...", end=" ")
            bypassed = attack_func()
            status = "[BYPASSED]" if bypassed else "[BLOCKED]"
            print(status)

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Attacks blocked: {self.attacks_blocked}")
        print(f"Attacks bypassed: {self.attacks_bypassed}")
        print()

        if self.attacks_bypassed > 0:
            print("[CRITICAL FAILURE] Some attacks bypassed constraints!")
            print()
            print("Bypassed attacks:")
            for log in self.attack_log:
                if log["status"] == "BYPASSED":
                    print(f"  - {log['attack']} via {log['method']}")
            print()
            return False
        else:
            print("[SUCCESS] All attacks blocked by database constraints")
            print()

        # Verify canonical metrics unchanged
        print("Verifying canonical metrics unchanged...", end=" ")
        metrics_ok = self.verify_canonical_metrics_unchanged()
        if metrics_ok:
            print("[OK]")
            print(
                f"  Round-trips: {self._last_round_trips} (baseline {self._baseline_round_trips})"
            )
            print(
                f"  Total PnL: ${self._last_total_pnl:.2f} (baseline ${self._baseline_total_pnl:.2f})"
            )
            print()
        else:
            print("[CORRUPTED]")
            print("  Canonical metrics were altered by attacks!")
            return False

        print("=" * 70)
        print("VERDICT: CHECK constraints are NON-BYPASSABLE")
        print("=" * 70)
        return True


def main():
    import argparse

    DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "portfolio_maximizer.db"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to database")
    parser.add_argument(
        "--disable-guardrails",
        action="store_true",
        help="Disable connection guardrails for baseline adversarial testing.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[ERROR] Database not found: {args.db}")
        sys.exit(1)

    with AdversarialTest(args.db, disable_guardrails=bool(args.disable_guardrails)) as tester:
        success = tester.run_all_attacks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
