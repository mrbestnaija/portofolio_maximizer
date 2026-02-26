import sqlite3
from pathlib import Path

from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer


def _create_trade_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            action TEXT NOT NULL,
            shares REAL,
            price REAL,
            close_size REAL,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            entry_price REAL,
            exit_price REAL,
            holding_period_days REAL,
            exit_reason TEXT,
            execution_mode TEXT DEFAULT 'live',
            is_close INTEGER NOT NULL,
            entry_trade_id INTEGER,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            commission REAL
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_rows(db_path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, close_size,
            realized_pnl, realized_pnl_pct, entry_price, exit_price,
            holding_period_days, exit_reason, execution_mode, is_close,
            entry_trade_id, is_diagnostic, is_synthetic, commission
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_duplicate_close_check_allows_legit_partial_exits(tmp_path):
    db_path = tmp_path / "partial_ok.db"
    _create_trade_db(db_path)
    _insert_rows(
        db_path,
        [
            (1, "AAPL", "2026-01-01", "BUY", 4.0, 100.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (2, "AAPL", "2026-01-02", "SELL", 1.0, 101.0, 1.0, 1.0, 0.01, 100.0, 101.0, 1.0, "TIME_EXIT", "live", 1, 1, 0, 0, 0.0),
            (3, "AAPL", "2026-01-03", "SELL", 3.0, 102.0, 3.0, 6.0, 0.02, 100.0, 102.0, 2.0, "TIME_EXIT", "live", 1, 1, 0, 0, 0.0),
        ],
    )

    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        violations = enforcer._check_duplicate_close_for_same_entry()

    assert violations == []


def test_duplicate_close_check_flags_overclosed_entries(tmp_path):
    db_path = tmp_path / "partial_overclosed.db"
    _create_trade_db(db_path)
    _insert_rows(
        db_path,
        [
            (1, "AAPL", "2026-01-01", "BUY", 4.0, 100.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (2, "AAPL", "2026-01-02", "SELL", 3.0, 101.0, 3.0, 3.0, 0.01, 100.0, 101.0, 1.0, "TIME_EXIT", "live", 1, 1, 0, 0, 0.0),
            (3, "AAPL", "2026-01-03", "SELL", 2.0, 102.0, 2.0, 4.0, 0.02, 100.0, 102.0, 2.0, "TIME_EXIT", "live", 1, 1, 0, 0, 0.0),
        ],
    )

    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        violations = enforcer._check_duplicate_close_for_same_entry()

    assert len(violations) == 1
    assert violations[0].check_name == "DUPLICATE_CLOSE_FOR_ENTRY"
    assert violations[0].affected_ids == [1]


def test_backfill_entry_links_supports_partial_close_inventory(tmp_path):
    db_path = tmp_path / "backfill_partial.db"
    _create_trade_db(db_path)
    _insert_rows(
        db_path,
        [
            (10, "AAPL", "2026-01-01", "BUY", 4.0, 100.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (11, "AAPL", "2026-01-02", "SELL", 1.0, 101.0, 1.0, 1.0, 0.01, 100.0, 101.0, 1.0, "TIME_EXIT", "live", 1, 10, 0, 0, 0.0),
            (12, "AAPL", "2026-01-03", "SELL", 3.0, 102.0, 3.0, 6.0, 0.02, 100.0, 102.0, 2.0, "TIME_EXIT", "live", 1, None, 0, 0, 0.0),
        ],
    )

    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        dry_run_count = enforcer.backfill_entry_trade_ids(dry_run=True)
        assert dry_run_count == 1

        applied_count = enforcer.backfill_entry_trade_ids(dry_run=False)
        assert applied_count == 1

        linked_id = enforcer.conn.execute(
            "SELECT entry_trade_id FROM trade_executions WHERE id = 12"
        ).fetchone()[0]
        assert linked_id == 10


def test_pnl_arithmetic_allows_long_sell_close(tmp_path):
    db_path = tmp_path / "pnl_long_close.db"
    _create_trade_db(db_path)
    _insert_rows(
        db_path,
        [
            (1, "AAPL", "2026-01-01", "BUY", 2.0, 100.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (2, "AAPL", "2026-01-02", "SELL", 2.0, 110.0, 2.0, 20.0, 0.10, 100.0, 110.0, 1.0, "TIME_EXIT", "live", 1, 1, 0, 0, 0.0),
        ],
    )

    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        violations = enforcer._check_pnl_arithmetic()

    assert violations == []


def test_pnl_arithmetic_allows_short_buy_close(tmp_path):
    db_path = tmp_path / "pnl_short_cover.db"
    _create_trade_db(db_path)
    _insert_rows(
        db_path,
        [
            (10, "TSLA", "2026-01-01", "SELL", 2.0, 100.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (11, "TSLA", "2026-01-02", "BUY", 2.0, 90.0, 2.0, 20.0, 0.10, 100.0, 90.0, 1.0, "TIME_EXIT", "live", 1, 10, 0, 0, 0.0),
        ],
    )

    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        violations = enforcer._check_pnl_arithmetic()

    assert violations == []


def test_pnl_arithmetic_flags_wrong_sign_for_short_buy_close(tmp_path):
    db_path = tmp_path / "pnl_short_cover_bad.db"
    _create_trade_db(db_path)
    _insert_rows(
        db_path,
        [
            (20, "TSLA", "2026-01-01", "SELL", 2.0, 100.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (21, "TSLA", "2026-01-02", "BUY", 2.0, 90.0, 2.0, -20.0, -0.10, 100.0, 90.0, 1.0, "TIME_EXIT", "live", 1, 20, 0, 0, 0.0),
        ],
    )

    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        violations = enforcer._check_pnl_arithmetic()

    assert len(violations) == 1
    assert violations[0].check_name == "PNL_ARITHMETIC_MISMATCH"
    assert violations[0].affected_ids == [21]
