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
            is_contaminated INTEGER DEFAULT 0,
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


def test_orphan_whitelist_ids_249_250_251_253_are_not_flagged(tmp_path):
    """Anti-regression: AAPL batch-replay duplicate opens from 2026-03-05 must not
    trigger ORPHANED_POSITION even after max_open_age_days=3 has elapsed.
    These are confirmed orphans (no close, not in portfolio_positions) whitelisted
    by policy in _check_orphaned_positions known_historical set.
    """
    db_path = tmp_path / "aapl_orphan_whitelist.db"
    _create_trade_db(db_path)
    # Insert the 4 stale AAPL BUY opens using the production IDs that triggered the gate.
    _insert_rows(
        db_path,
        [
            (249, "AAPL", "2026-03-04", "BUY", 1.0, 262.67, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (250, "AAPL", "2026-03-04", "BUY", 1.0, 262.67, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (251, "AAPL", "2026-03-04", "BUY", 1.0, 262.67, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (253, "AAPL", "2026-03-04", "BUY", 1.0, 262.67, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
        ],
    )

    import os
    env_backup = os.environ.pop("INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS", None)
    os.environ["INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS"] = "0"  # force stale for all dates
    try:
        with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
            violations = enforcer._check_orphaned_positions()
    finally:
        if env_backup is None:
            os.environ.pop("INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS", None)
        else:
            os.environ["INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS"] = env_backup

    orphan_violations = [v for v in violations if v.check_name == "ORPHANED_POSITION"]
    assert orphan_violations == [], (
        f"IDs 249,250,251,253 must be in known_historical whitelist and not trigger ORPHANED_POSITION. "
        f"Got: {orphan_violations}"
    )


# ---------------------------------------------------------------------------
# INT-05: cross-mode contamination detection
# ---------------------------------------------------------------------------

def _create_contamination_db(db_path: Path) -> None:
    """Minimal schema for cross-mode contamination tests."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            action TEXT NOT NULL,
            shares REAL DEFAULT 1,
            price REAL DEFAULT 100,
            close_size REAL,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            entry_price REAL,
            exit_price REAL,
            holding_period_days REAL DEFAULT 0,
            exit_reason TEXT,
            execution_mode TEXT DEFAULT 'live',
            is_close INTEGER NOT NULL DEFAULT 0,
            entry_trade_id INTEGER,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            is_contaminated INTEGER DEFAULT 0,
            commission REAL DEFAULT 0
        );
        """
    )
    conn.commit()
    conn.close()


def test_cross_mode_contamination_tagged_closer_detected(tmp_path):
    """is_contaminated=1 on a closing leg is reported as CROSS_MODE_CONTAMINATION."""
    db_path = tmp_path / "contam.db"
    _create_contamination_db(db_path)
    conn = sqlite3.connect(db_path)
    # opener: synthetic SELL at synthetic price
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, is_synthetic, execution_mode) VALUES (1,'MSFT','2026-02-27','SELL',3,64.0,0,1,'synthetic')"
    )
    # closer: live BUY but is_contaminated=1 (entry_price was $64, real exit $405)
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "close_size, realized_pnl, entry_price, exit_price, is_close, is_contaminated, "
        "entry_trade_id, execution_mode) "
        "VALUES (2,'MSFT','2026-03-04','BUY',3,405.0,3,-1027.0,64.0,405.0,1,1,1,'live')"
    )
    conn.commit()
    conn.close()

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    contam = [v for v in violations if v.check_name == "CROSS_MODE_CONTAMINATION"]
    assert len(contam) == 1
    assert 2 in contam[0].affected_ids
    assert contam[0].severity == "HIGH"


def test_cross_mode_contamination_untagged_closer_detected_via_opener_join(tmp_path):
    """Closing leg with no is_contaminated flag but opener is_synthetic=1 is also detected."""
    db_path = tmp_path / "contam_untagged.db"
    _create_contamination_db(db_path)
    conn = sqlite3.connect(db_path)
    # opener: synthetic SELL
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, is_synthetic, execution_mode) VALUES (10,'AAPL','2026-01-01','SELL',2,80.0,0,1,'synthetic')"
    )
    # closer: live BUY, is_contaminated NOT set (untagged)
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "close_size, realized_pnl, entry_price, exit_price, is_close, is_contaminated, "
        "entry_trade_id, execution_mode) "
        "VALUES (11,'AAPL','2026-03-01','BUY',2,350.0,2,-540.0,80.0,350.0,1,0,10,'live')"
    )
    conn.commit()
    conn.close()

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    contam = [v for v in violations if v.check_name == "CROSS_MODE_CONTAMINATION"]
    assert len(contam) == 1
    assert 11 in contam[0].affected_ids
    # description should mention untagged
    assert "untagged" in contam[0].description


def test_cross_mode_contamination_clean_live_round_trip_no_violation(tmp_path):
    """A normal live-open / live-close round trip produces no contamination violation."""
    db_path = tmp_path / "clean.db"
    _create_contamination_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, is_synthetic, execution_mode) VALUES (20,'NVDA','2026-02-01','BUY',1,800.0,0,0,'live')"
    )
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "close_size, realized_pnl, entry_price, exit_price, is_close, is_contaminated, "
        "entry_trade_id, execution_mode) "
        "VALUES (21,'NVDA','2026-02-10','SELL',1,850.0,1,50.0,800.0,850.0,1,0,20,'live')"
    )
    conn.commit()
    conn.close()

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    contam = [v for v in violations if v.check_name == "CROSS_MODE_CONTAMINATION"]
    assert contam == [], f"Clean round-trip must not trigger contamination check: {contam}"


def test_cross_mode_contamination_excluded_from_canonical_metrics(tmp_path):
    """Contaminated closes are excluded from get_canonical_metrics()."""
    db_path = tmp_path / "metrics.db"
    _create_contamination_db(db_path)
    conn = sqlite3.connect(db_path)
    # clean winning round-trip: +$50
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, execution_mode) VALUES (30,'AAPL','2026-01-01','BUY',1,100.0,0,'live')"
    )
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "close_size, realized_pnl, entry_price, exit_price, holding_period_days, is_close, "
        "is_contaminated, entry_trade_id, execution_mode) "
        "VALUES (31,'AAPL','2026-01-10','SELL',1,150.0,1,50.0,100.0,150.0,9,1,0,30,'live')"
    )
    # contaminated losing round-trip: -$1000 (phantom)
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, is_synthetic, execution_mode) VALUES (32,'MSFT','2026-02-01','SELL',3,64.0,0,1,'synthetic')"
    )
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "close_size, realized_pnl, entry_price, exit_price, holding_period_days, is_close, "
        "is_contaminated, entry_trade_id, execution_mode) "
        "VALUES (33,'MSFT','2026-03-01','BUY',3,405.0,3,-1020.0,64.0,405.0,28,1,1,32,'live')"
    )
    conn.commit()
    conn.close()

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        metrics = enforcer.get_canonical_metrics()

    assert metrics.total_round_trips == 1
    assert abs(metrics.total_realized_pnl - 50.0) < 0.01
    assert metrics.win_rate == 1.0
    assert metrics.contaminated_trades_excluded >= 1


def test_cross_mode_contamination_whitelist_suppresses_known_ids(tmp_path):
    """Whitelisted contaminated IDs (252, 255) do not raise CROSS_MODE_CONTAMINATION."""
    db_path = tmp_path / "whitelist.db"
    _create_contamination_db(db_path)
    conn = sqlite3.connect(db_path)
    # Simulate trades 252 and 255: both tagged is_contaminated=1, already excluded from metrics
    for trade_id in (252, 255):
        conn.execute(
            "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
            "close_size, realized_pnl, entry_price, exit_price, is_close, is_contaminated, "
            "execution_mode) "
            "VALUES (?,'MSFT','2026-03-04','BUY',1,100.0,1,-50.0,50.0,100.0,1,1,'live')",
            (trade_id,),
        )
    conn.commit()
    conn.close()

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    contam = [v for v in violations if v.check_name == "CROSS_MODE_CONTAMINATION"]
    assert contam == [], (
        f"Whitelisted IDs 252/255 must not trigger CROSS_MODE_CONTAMINATION, got: {contam}"
    )


def test_canonical_metrics_reject_null_production_flags(tmp_path):
    db_path = tmp_path / "null_flags.db"
    _create_trade_db(db_path)
    _insert_rows(
        db_path,
        [
            (1, "AAPL", "2026-01-01", "BUY", 1.0, 100.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (2, "AAPL", "2026-01-02", "SELL", 1.0, 110.0, 1.0, 10.0, 0.10, 100.0, 110.0, 1.0, "TIME_EXIT", "live", 1, 1, None, None, 0.0),
            (3, "MSFT", "2026-01-03", "SELL", 1.0, 220.0, 1.0, 20.0, 0.10, 200.0, 220.0, 2.0, "TIME_EXIT", "live", 1, 1, 0, 0, 0.0),
        ],
    )

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE trade_executions SET is_contaminated = NULL WHERE id = 2")
    conn.execute("UPDATE trade_executions SET is_contaminated = 0 WHERE id = 3")
    conn.commit()
    conn.close()

    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        metrics = enforcer.get_canonical_metrics()
        violations = enforcer.run_full_integrity_audit()

    assert metrics.total_round_trips == 1
    assert metrics.total_realized_pnl == 20.0
    null_flag = [v for v in violations if v.check_name == "NULL_PRODUCTION_FLAGS"]
    assert len(null_flag) == 1
    assert null_flag[0].severity == "CRITICAL"
    assert null_flag[0].affected_ids == [2]


# ---------------------------------------------------------------------------
# INT-06: metrics drift detection
# ---------------------------------------------------------------------------

def _make_pnl_db(db_path: Path, pnls: list[float]) -> None:
    """Create DB with production_closed_trades populated from given PnL list."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL DEFAULT 'AAPL',
            trade_date TEXT NOT NULL DEFAULT '2026-01-01',
            action TEXT NOT NULL DEFAULT 'SELL',
            shares REAL DEFAULT 1,
            price REAL DEFAULT 100,
            close_size REAL DEFAULT 1,
            realized_pnl REAL,
            entry_price REAL DEFAULT 90,
            exit_price REAL DEFAULT 100,
            holding_period_days REAL DEFAULT 1,
            is_close INTEGER DEFAULT 1,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            is_contaminated INTEGER DEFAULT 0,
            entry_trade_id INTEGER,
            execution_mode TEXT DEFAULT 'live',
            commission REAL DEFAULT 0
        );
        """
    )
    for pnl in pnls:
        conn.execute("INSERT INTO trade_executions (realized_pnl) VALUES (?)", (pnl,))
    conn.commit()
    conn.close()


def test_metrics_drift_fires_when_rolling_wr_drops(tmp_path, monkeypatch):
    """Drift check fires HIGH when last-30 WR drops 20pp below historical."""
    monkeypatch.setenv("INTEGRITY_DRIFT_ROLLING_WINDOW", "30")
    monkeypatch.setenv("INTEGRITY_DRIFT_THRESHOLD", "0.15")
    monkeypatch.setenv("INTEGRITY_DRIFT_MIN_TRADES", "15")

    # 30 historical wins then 30 losses (all below $0)
    pnls = [50.0] * 30 + [-20.0] * 30
    db_path = tmp_path / "drift_down.db"
    _make_pnl_db(db_path, pnls)

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    drift = [v for v in violations if v.check_name == "METRICS_DRIFT"]
    assert len(drift) == 1
    assert drift[0].severity == "HIGH"
    assert "MODEL DRIFT WARNING" in drift[0].description


def test_metrics_drift_no_violation_when_stable(tmp_path, monkeypatch):
    """Drift check does not fire when WR is consistent across periods."""
    monkeypatch.setenv("INTEGRITY_DRIFT_ROLLING_WINDOW", "30")
    monkeypatch.setenv("INTEGRITY_DRIFT_THRESHOLD", "0.15")
    monkeypatch.setenv("INTEGRITY_DRIFT_MIN_TRADES", "15")

    # Consistent 50% WR across 60 trades
    pnls = [50.0, -20.0] * 30
    db_path = tmp_path / "drift_stable.db"
    _make_pnl_db(db_path, pnls)

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    drift = [v for v in violations if v.check_name == "METRICS_DRIFT"]
    assert drift == [], f"Stable WR must not trigger drift check: {drift}"


def test_metrics_drift_skipped_with_insufficient_history(tmp_path, monkeypatch):
    """Drift check is silently skipped when not enough trades exist."""
    monkeypatch.setenv("INTEGRITY_DRIFT_ROLLING_WINDOW", "30")
    monkeypatch.setenv("INTEGRITY_DRIFT_THRESHOLD", "0.15")
    monkeypatch.setenv("INTEGRITY_DRIFT_MIN_TRADES", "15")

    # Only 20 trades — below min_needed (15+30=45)
    pnls = [50.0] * 10 + [-20.0] * 10
    db_path = tmp_path / "drift_short.db"
    _make_pnl_db(db_path, pnls)

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    drift = [v for v in violations if v.check_name == "METRICS_DRIFT"]
    assert drift == [], "Insufficient history must not trigger drift check"
