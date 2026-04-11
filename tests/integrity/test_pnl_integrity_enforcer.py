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


def test_cross_mode_contamination_tagged_closer_suppressed(tmp_path):
    """is_contaminated=1 on a closing leg does NOT raise CROSS_MODE_CONTAMINATION.

    Tagged closes are already excluded from production_closed_trades — re-blocking
    on them here would require a manual whitelist entry for every new contaminated
    close.  The architectural fix is to only flag UNTAGGED contamination.
    """
    db_path = tmp_path / "contam.db"
    _create_contamination_db(db_path)
    conn = sqlite3.connect(db_path)
    # opener: synthetic SELL at synthetic price
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, is_synthetic, execution_mode) VALUES (1,'MSFT','2026-02-27','SELL',3,64.0,0,1,'synthetic')"
    )
    # closer: live BUY with is_contaminated=1 — already excluded from canonical metrics
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
    assert contam == [], (
        f"Tagged closes (is_contaminated=1) must not trigger violation — "
        f"they are already excluded from production_closed_trades. Got: {contam}"
    )


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


def test_cross_mode_contamination_tagged_count_in_description_when_untagged_fires(tmp_path):
    """When untagged contamination fires, the description reports the tagged-suppressed count."""
    db_path = tmp_path / "mixed.db"
    _create_contamination_db(db_path)
    conn = sqlite3.connect(db_path)
    # tagged close (is_contaminated=1): suppressed, opener is synthetic
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, is_synthetic, execution_mode) VALUES (50,'MSFT','2026-02-01','SELL',1,64.0,0,1,'synthetic')"
    )
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "close_size, realized_pnl, entry_price, exit_price, is_close, is_contaminated, "
        "entry_trade_id, execution_mode) "
        "VALUES (51,'MSFT','2026-03-01','BUY',1,100.0,1,-36.0,64.0,100.0,1,1,50,'live')"
    )
    # untagged close (is_contaminated=0 but opener is synthetic): fires violation
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "is_close, is_synthetic, execution_mode) VALUES (60,'AAPL','2026-02-01','SELL',1,80.0,0,1,'synthetic')"
    )
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "close_size, realized_pnl, entry_price, exit_price, is_close, is_contaminated, "
        "entry_trade_id, execution_mode) "
        "VALUES (61,'AAPL','2026-03-15','BUY',1,350.0,1,-270.0,80.0,350.0,1,0,60,'live')"
    )
    conn.commit()
    conn.close()

    with PnLIntegrityEnforcer(str(db_path)) as enforcer:
        violations = enforcer.run_full_integrity_audit()

    contam = [v for v in violations if v.check_name == "CROSS_MODE_CONTAMINATION"]
    assert len(contam) == 1, f"Only untagged contamination should fire, got: {contam}"
    assert 61 in contam[0].affected_ids
    assert 51 not in contam[0].affected_ids
    # tagged count must appear in the description for observability
    assert "1" in contam[0].description  # 1 tagged-and-suppressed close reported


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


def test_orphan_fifo_respects_entry_trade_id_linkage(tmp_path, monkeypatch):
    """Regression: FIFO must use entry_trade_id for direct consumption rather than
    blind date-order consumption. Without this fix, a close leg arriving after other
    close legs has already consumed earlier BUY legs from the queue, leaving the
    explicitly-linked BUY leg stranded with residual qty and falsely flagged ORPHANED.

    Scenario mirrors the production bug where ids 31/33 were flagged despite having
    matching close legs 32/35 with correct entry_trade_id linkage.
    """
    import os
    db_path = tmp_path / "fifo_entry_trade_id.db"
    _create_trade_db(db_path)
    # Two early BUY opens (ids 1, 2) will be consumed by their own closes (ids 3, 4).
    # Then a third BUY (id 5) is explicitly closed by id 6 via entry_trade_id=5.
    # Without entry_trade_id-aware FIFO, close id 6 might consume BUY id 5 correctly
    # only if it arrives after ids 3/4 have cleared ids 1/2. This test verifies that
    # even when a BUY is preceded by other BUY legs, the correct close leg reaches it.
    _insert_rows(
        db_path,
        [
            # BUY opens (is_close=0)
            (1, "AAPL", "2024-01-01", "BUY", 9.0, 150.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (2, "AAPL", "2024-01-02", "BUY", 8.0, 151.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            (3, "AAPL", "2024-01-03", "BUY", 5.0, 152.0, None, None, None, None, None, None, None, "live", 0, None, 0, 0, 0.0),
            # SELL closes — ids 10 and 11 close ids 1 and 2 via entry_trade_id
            (10, "AAPL", "2024-01-05", "SELL", 9.0, 155.0, 9.0, 27.0, 0.02, 150.0, 155.0, 4.0, "TIME_EXIT", "live", 1, 1, 0, 0, 0.0),
            (11, "AAPL", "2024-01-06", "SELL", 8.0, 156.0, 8.0, 40.0, 0.03, 151.0, 156.0, 4.0, "TIME_EXIT", "live", 1, 2, 0, 0, 0.0),
            # id 12 closes id 3 — arrives last but is explicitly linked
            (12, "AAPL", "2024-01-10", "SELL", 5.0, 160.0, 5.0, 40.0, 0.05, 152.0, 160.0, 7.0, "TIME_EXIT", "live", 1, 3, 0, 0, 0.0),
        ],
    )

    monkeypatch.setenv("INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS", "0")
    with PnLIntegrityEnforcer(str(db_path), allow_schema_changes=True) as enforcer:
        violations = enforcer._check_orphaned_positions()

    orphan = [v for v in violations if v.check_name == "ORPHANED_POSITION"]
    assert orphan == [], (
        f"All BUY legs have matching close legs via entry_trade_id; none should be orphaned. "
        f"Got: {orphan}"
    )


def test_is_synthetic_set_from_execution_mode(tmp_path):
    """Regression: PaperTradingEngine must set is_synthetic=1 when execution_mode='synthetic',
    even if data_source does not contain 'synthetic'. This ensures historical as-of-date runs
    (which use yfinance data_source but are tagged execution_mode='synthetic') are excluded from
    the orphan check's is_synthetic=0 filter.
    """
    import unittest.mock as mock
    from execution.paper_trading_engine import PaperTradingEngine
    import pandas as pd
    import numpy as np

    db_path = tmp_path / "pte_synthetic_mode.db"
    engine = PaperTradingEngine(db_path=str(db_path), initial_capital=100_000.0)

    market_data = pd.DataFrame(
        {
            "Open": [150.0, 151.0],
            "High": [152.0, 153.0],
            "Low": [149.0, 150.0],
            "Close": [151.0, 152.0],
            "Volume": [1_000_000, 1_000_000],
        },
        index=pd.date_range("2024-01-02", periods=2, freq="D"),
    )
    signal = {
        "ticker": "AAPL",
        "action": "BUY",
        "confidence": 0.70,
        "reasoning": "test",
        "execution_mode": "synthetic",   # execution_mode is synthetic
        "data_source": "yfinance",        # data_source is NOT synthetic
        "run_id": "test_run",
    }

    recorded_trade = None
    original_record = engine.db_manager.save_trade_execution

    def capture_trade(**kwargs):
        nonlocal recorded_trade
        recorded_trade = kwargs
        return 1

    with mock.patch.object(engine.db_manager, "save_trade_execution", side_effect=capture_trade):
        engine.execute_signal(signal, market_data)

    assert recorded_trade is not None, "save_trade_execution must have been called"
    assert recorded_trade.get("is_synthetic") == 1, (
        f"execution_mode='synthetic' must set is_synthetic=1 regardless of data_source. "
        f"Got is_synthetic={recorded_trade.get('is_synthetic')}"
    )


def test_resume_backfill_skips_synthetic_openers(tmp_path):
    """Backfill of entry_trade_ids at PTE resume must ignore synthetic openers.

    Scenario: a live open (id=1) exists for AAPL; a later synthetic open (id=2)
    also exists.  Without the fix, `ORDER BY id DESC LIMIT 1` would return id=2
    (the synthetic one) and the subsequent close would be tagged is_contaminated=1,
    preventing it from counting toward THIN_LINKAGE.

    With the fix (`AND COALESCE(is_synthetic, 0) = 0`), the backfill returns id=1
    (the live opener) so the close stays clean.
    """
    import sqlite3 as _sqlite3
    from etl.database_manager import DatabaseManager
    from execution.paper_trading_engine import PaperTradingEngine

    db_path = str(tmp_path / "backfill_test.db")

    # Seed the DB with a live opener (id=1) and a later synthetic opener (id=2)
    engine = PaperTradingEngine(db_path=db_path, initial_capital=100_000.0)
    conn = engine.db_manager.conn
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "total_value, is_close, is_synthetic, execution_mode) "
        "VALUES (1,'AAPL','2026-04-01','BUY',5,170.0,850.0,0,0,'live')"
    )
    conn.execute(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, "
        "total_value, is_close, is_synthetic, execution_mode) "
        "VALUES (2,'AAPL','2026-04-03','BUY',5,175.0,875.0,0,1,'synthetic')"
    )
    # portfolio_cash_state row is required for load_portfolio_state() to return non-None
    conn.execute(
        "INSERT OR REPLACE INTO portfolio_cash_state (id, cash, initial_capital) "
        "VALUES (1, 99150.0, 100000.0)"
    )
    # portfolio_state uses shares (not position); entry_timestamp is required
    conn.execute(
        "INSERT INTO portfolio_state (ticker, shares, entry_price, entry_timestamp) "
        "VALUES ('AAPL', 5, 170.0, '2026-04-01T00:00:00+00:00')"
    )
    conn.commit()

    # Re-create engine with resume_from_db=True to trigger the backfill
    engine2 = PaperTradingEngine(db_path=db_path, initial_capital=100_000.0, resume_from_db=True)

    # Backfill must have selected the LIVE opener (id=1), not the synthetic one (id=2)
    entry_id = engine2.portfolio.entry_trade_ids.get("AAPL")
    assert entry_id == 1, (
        f"Backfill must prefer live opener (id=1) over synthetic opener (id=2). "
        f"Got entry_trade_id={entry_id}"
    )
