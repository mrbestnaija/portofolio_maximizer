from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts import repair_unlinked_closes as repair


def _create_trade_executions_table(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            action TEXT NOT NULL,
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
            entry_price REAL,
            exit_price REAL,
            close_size REAL,
            position_before REAL,
            position_after REAL,
            is_close INTEGER,
            bar_timestamp TEXT,
            exit_reason TEXT,
            asset_class TEXT DEFAULT 'equity',
            instrument_type TEXT DEFAULT 'spot',
            underlying_ticker TEXT,
            strike REAL,
            expiry TEXT,
            multiplier REAL DEFAULT 1.0,
            barbell_bucket TEXT,
            barbell_multiplier REAL,
            base_confidence REAL,
            effective_confidence REAL,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            confidence_calibrated REAL,
            entry_trade_id INTEGER,
            bar_open REAL,
            bar_high REAL,
            bar_low REAL,
            bar_close REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_unlinked_close_row(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size, position_before, position_after,
            is_close, bar_timestamp, exit_reason, asset_class, instrument_type,
            multiplier, effective_confidence, entry_trade_id
        ) VALUES (
            66, 'GS', '2026-01-23', 'SELL', 1.0, 918.34705448, 918.34705448,
            0.137752058, 927.119995117, -94.625730039,
            'yfinance', 'live', '20260214_122848',
            -44.348657578, -0.045930642, 7,
            962.55796, 918.34705448, 1.0, 1.0, 0.0,
            1, '2026-01-23T00:00:00+00:00', 'STOP_LOSS', 'US_EQUITY', 'spot',
            1.0, 0.9, NULL
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_unlinked_short_cover_close_row(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size, position_before, position_after,
            is_close, bar_timestamp, exit_reason, asset_class, instrument_type,
            multiplier, effective_confidence, entry_trade_id, is_synthetic
        ) VALUES (
            88, 'AAPL', '2026-02-01', 'BUY', 2.0, 95.0, 190.0,
            0.0, 95.0, 0.0,
            'synthetic', 'synthetic', '20260214_130000',
            10.0, 0.05, 3,
            100.0, 95.0, 2.0, -2.0, 0.0,
            1, '2026-02-01T00:00:00+00:00', 'TAKE_PROFIT', 'US_EQUITY', 'spot',
            1.0, 0.75, NULL, 1
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_unlinked_close_without_realized_pnl(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size, position_before, position_after,
            is_close, bar_timestamp, exit_reason, asset_class, instrument_type,
            multiplier, effective_confidence, entry_trade_id
        ) VALUES (
            99, 'MSFT', '2026-02-15', 'SELL', 1.0, 300.0, 300.0,
            0.0, 300.0, 0.0,
            'synthetic', 'synthetic', '20260215_090000',
            NULL, NULL, 2,
            305.0, 300.0, 1.0, 1.0, 0.0,
            1, '2026-02-15T00:00:00+00:00', 'EXIT_RULE', 'US_EQUITY', 'spot',
            1.0, 0.8, NULL
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_orphan_short_entry(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size, position_before, position_after,
            is_close, bar_timestamp, exit_reason, asset_class, instrument_type,
            multiplier, effective_confidence, entry_trade_id, is_synthetic
        ) VALUES (
            77, 'AAPL', '2026-01-29', 'SELL', 2.0, 100.0, 200.0,
            0.0, 100.0, 0.0,
            'synthetic', 'synthetic', '20260214_129500',
            NULL, NULL, NULL,
            NULL, NULL, NULL, 0.0, -2.0,
            0, '2026-01-29T00:00:00+00:00', NULL, 'US_EQUITY', 'spot',
            1.0, 0.75, NULL, 1
        );
        """
    )
    conn.commit()
    conn.close()


def test_find_unlinked_closes_includes_rows_without_realized_pnl(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.db"
    _create_trade_executions_table(db_path)
    _insert_unlinked_close_without_realized_pnl(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = repair.find_unlinked_closes(conn)
    conn.close()

    close_ids = {int(r["id"]) for r in rows}
    assert 99 in close_ids


def test_repair_linkage_reconstruct_from_state_dry_run(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.db"
    report_path = tmp_path / "forensic.json"
    _create_trade_executions_table(db_path)
    _insert_unlinked_close_row(db_path)

    rc = repair.repair_linkage(
        db_path=db_path,
        dry_run=True,
        close_ids=[66],
        reconstruct_from_state=True,
        forensic_report_file=report_path,
        logs_root=tmp_path / "logs",
    )
    assert rc == 0
    assert report_path.exists()

    conn = sqlite3.connect(db_path)
    close_row = conn.execute(
        "SELECT entry_trade_id FROM trade_executions WHERE id = 66"
    ).fetchone()
    conn.close()
    assert close_row is not None
    assert close_row[0] is None


def test_repair_linkage_reconstruct_from_state_apply(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.db"
    report_path = tmp_path / "forensic.json"
    _create_trade_executions_table(db_path)
    _insert_unlinked_close_row(db_path)

    rc = repair.repair_linkage(
        db_path=db_path,
        dry_run=False,
        close_ids=[66],
        reconstruct_from_state=True,
        forensic_report_file=report_path,
        logs_root=tmp_path / "logs",
    )
    assert rc == 0
    assert report_path.exists()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    close_row = conn.execute(
        "SELECT entry_trade_id FROM trade_executions WHERE id = 66"
    ).fetchone()
    assert close_row is not None
    linked_entry_id = int(close_row["entry_trade_id"])
    assert linked_entry_id > 0

    entry_row = conn.execute(
        "SELECT id, action, is_close, ticker, trade_date, shares, price, run_id, execution_mode "
        "FROM trade_executions WHERE id = ?",
        (linked_entry_id,),
    ).fetchone()
    conn.close()

    assert entry_row is not None
    assert int(entry_row["id"]) == linked_entry_id
    assert str(entry_row["action"]) == "BUY"
    assert int(entry_row["is_close"]) == 0
    assert str(entry_row["ticker"]) == "GS"
    assert str(entry_row["trade_date"]) == "2026-01-16"
    assert abs(float(entry_row["shares"]) - 1.0) < 1e-9
    assert abs(float(entry_row["price"]) - 962.55796) < 1e-9
    assert str(entry_row["run_id"]).endswith("_recon_entry_66")
    assert str(entry_row["execution_mode"]).endswith("_entry_reconstructed")


def test_repair_linkage_short_cover_reconstructs_sell_entry(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.db"
    report_path = tmp_path / "forensic_short.json"
    _create_trade_executions_table(db_path)
    _insert_unlinked_short_cover_close_row(db_path)

    rc = repair.repair_linkage(
        db_path=db_path,
        dry_run=False,
        close_ids=[88],
        reconstruct_from_state=True,
        forensic_report_file=report_path,
        logs_root=tmp_path / "logs",
    )
    assert rc == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    close_row = conn.execute("SELECT entry_trade_id FROM trade_executions WHERE id = 88").fetchone()
    assert close_row is not None
    entry_id = int(close_row["entry_trade_id"])
    assert entry_id > 0

    entry = conn.execute(
        "SELECT action, is_close, shares, price, position_after FROM trade_executions WHERE id = ?",
        (entry_id,),
    ).fetchone()
    conn.close()

    assert entry is not None
    assert str(entry["action"]) == "SELL"
    assert int(entry["is_close"]) == 0
    assert abs(float(entry["shares"]) - 2.0) < 1e-9
    assert abs(float(entry["price"]) - 100.0) < 1e-9
    assert abs(float(entry["position_after"]) - (-2.0)) < 1e-9


def test_match_fifo_rejects_synthetic_to_live_provenance_mismatch(tmp_path: Path) -> None:
    """A live BUY-close must not be matched to a synthetic SELL-open (provenance guard)."""
    db_path = tmp_path / "portfolio.db"
    _create_trade_executions_table(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Synthetic SELL open (orphan)
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size, position_before, position_after,
            is_close, bar_timestamp, exit_reason, asset_class, instrument_type,
            multiplier, effective_confidence, entry_trade_id, is_synthetic
        ) VALUES (
            114, 'TSLA', '2021-06-14', 'SELL', 2.0, 600.0, 1200.0,
            0.0, 600.0, 0.0,
            'synthetic', 'synthetic', '20210614_000000',
            NULL, NULL, NULL,
            NULL, NULL, NULL, 0.0, -2.0,
            0, '2021-06-14T00:00:00+00:00', NULL, 'US_EQUITY', 'spot',
            1.0, 0.75, NULL, 1
        );
        """
    )
    # Live BUY close (the problematic trade 255 scenario)
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size, position_before, position_after,
            is_close, bar_timestamp, exit_reason, asset_class, instrument_type,
            multiplier, effective_confidence, entry_trade_id, is_synthetic
        ) VALUES (
            255, 'TSLA', '2026-03-06', 'BUY', 2.0, 396.96, 793.92,
            0.0, 396.96, 0.0,
            'yfinance', 'live', '20260309_063607',
            -629.77, -0.44, 0,
            710.0, 396.96, 2.0, -2.0, 0.0,
            1, '2026-03-06T00:00:00+00:00', 'STOP_LOSS', 'US_EQUITY', 'spot',
            1.0, 0.8, NULL, 0
        );
        """
    )
    conn.commit()

    orphans = repair.find_orphaned_entries(conn, "TSLA", "SELL")
    unlinked = repair.find_unlinked_closes(conn)
    conn.close()

    assert len(orphans) == 1, "Expected one orphan SELL"
    assert len(unlinked) == 1, "Expected one unlinked close"

    matched = repair.match_fifo(unlinked[0], orphans)
    # Provenance mismatch (live close vs synthetic open) must be rejected.
    assert matched is None, (
        f"match_fifo matched live close (id=255) to synthetic open (id=114) — "
        "provenance guard failed"
    )


def test_match_fifo_rejects_extreme_date_gap(tmp_path: Path) -> None:
    """An orphan open more than _MAX_MATCH_GAP_DAYS before close must be rejected."""
    db_path = tmp_path / "portfolio.db"
    _create_trade_executions_table(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Orphan BUY open from 3 years ago
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            data_source, execution_mode, run_id,
            position_before, position_after, is_close,
            effective_confidence, is_synthetic
        ) VALUES (
            10, 'AAPL', '2020-01-01', 'BUY', 1.0, 300.0, 300.0,
            'yfinance', 'live', '20200101_000000',
            0.0, 1.0, 0,
            0.8, 0
        );
        """
    )
    # Live SELL close from 2023 (3+ year gap)
    conn.execute(
        """
        INSERT INTO trade_executions (
            id, ticker, trade_date, action, shares, price, total_value,
            commission, mid_price, mid_slippage_bps,
            data_source, execution_mode, run_id,
            realized_pnl, realized_pnl_pct, holding_period_days,
            entry_price, exit_price, close_size, position_before, position_after,
            is_close, bar_timestamp, exit_reason, asset_class, instrument_type,
            multiplier, effective_confidence, entry_trade_id, is_synthetic
        ) VALUES (
            20, 'AAPL', '2023-06-01', 'SELL', 1.0, 180.0, 180.0,
            0.0, 180.0, 0.0,
            'yfinance', 'live', '20230601_000000',
            -120.0, -0.4, 0,
            300.0, 180.0, 1.0, 1.0, 0.0,
            1, '2023-06-01T00:00:00+00:00', 'STOP_LOSS', 'US_EQUITY', 'spot',
            1.0, 0.7, NULL, 0
        );
        """
    )
    conn.commit()

    orphans = repair.find_orphaned_entries(conn, "AAPL", "BUY")
    unlinked = repair.find_unlinked_closes(conn)
    conn.close()

    matched = repair.match_fifo(unlinked[0], orphans)
    # Gap of ~1248 days exceeds _MAX_MATCH_GAP_DAYS (365); must be rejected.
    assert matched is None, (
        "match_fifo matched close to open from 3+ years ago — date-gap guard failed"
    )


def test_repair_linkage_short_cover_matches_existing_orphan_sell(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.db"
    _create_trade_executions_table(db_path)
    _insert_orphan_short_entry(db_path)
    _insert_unlinked_short_cover_close_row(db_path)

    rc = repair.repair_linkage(
        db_path=db_path,
        dry_run=False,
        close_ids=[88],
        reconstruct_from_state=False,
        forensic_report_file=None,
        logs_root=tmp_path / "logs",
    )
    assert rc == 0

    conn = sqlite3.connect(db_path)
    close_row = conn.execute("SELECT entry_trade_id FROM trade_executions WHERE id = 88").fetchone()
    conn.close()
    assert close_row is not None
    assert int(close_row[0]) == 77
