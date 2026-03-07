"""Tests for scripts/exit_quality_audit.py"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.exit_quality_audit import (
    compute_exit_reason_breakdown,
    diagnose_direction_gap,
    load_production_trades,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_trades(**overrides) -> pd.DataFrame:
    """Return a minimal DataFrame that matches the schema returned by load_production_trades."""
    base = {
        "ticker": "AAPL",
        "trade_date": "2026-01-01",
        "action": "BUY",
        "exit_reason": "signal_exit",
        "realized_pnl": 100.0,
        "entry_price": 150.0,
        "exit_price": 155.0,
        "bar_high": 156.0,
        "bar_low": 149.0,
        "holding_period_days": 2,
        "is_winner": 1,
        "atr_proxy": 7.0,
        "r_multiple": 100.0 / (7.0 * 1.5),
        "correct_dir_neg_pnl": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _make_trade_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a multi-row trades DataFrame."""
    dfs = [_make_trades(**r) for r in rows]
    return pd.concat(dfs, ignore_index=True)


def _build_db(tmp_path: Path) -> Path:
    """Create a minimal in-memory-like SQLite DB with production_closed_trades view."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            trade_date TEXT,
            action TEXT,
            exit_reason TEXT,
            realized_pnl REAL,
            entry_price REAL,
            exit_price REAL,
            bar_high REAL,
            bar_low REAL,
            holding_period_days INTEGER,
            is_close INTEGER DEFAULT 0,
            is_diagnostic INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE VIEW production_closed_trades AS
        SELECT * FROM trade_executions
        WHERE is_close = 1
          AND COALESCE(is_diagnostic, 0) = 0
          AND COALESCE(is_synthetic, 0) = 0
    """)
    conn.commit()
    conn.close()
    return db_path


def _insert_trade(db_path: Path, **kwargs) -> None:
    defaults = {
        "ticker": "AAPL",
        "trade_date": "2026-01-01",
        "action": "BUY",
        "exit_reason": "signal_exit",
        "realized_pnl": 50.0,
        "entry_price": 100.0,
        "exit_price": 105.0,
        "bar_high": 106.0,
        "bar_low": 99.0,
        "holding_period_days": 1,
        "is_close": 1,
        "is_diagnostic": 0,
        "is_synthetic": 0,
    }
    defaults.update(kwargs)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trade_executions (
            ticker, trade_date, action, exit_reason, realized_pnl,
            entry_price, exit_price, bar_high, bar_low, holding_period_days,
            is_close, is_diagnostic, is_synthetic
        ) VALUES (
            :ticker, :trade_date, :action, :exit_reason, :realized_pnl,
            :entry_price, :exit_price, :bar_high, :bar_low, :holding_period_days,
            :is_close, :is_diagnostic, :is_synthetic
        )
    """, defaults)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# load_production_trades
# ---------------------------------------------------------------------------

class TestLoadProductionTrades:
    def test_returns_empty_when_db_missing(self, tmp_path):
        df = load_production_trades(tmp_path / "no.db")
        assert df.empty

    def test_returns_empty_when_no_production_trades(self, tmp_path):
        db = _build_db(tmp_path)
        # Insert diagnostic-only trade (should be excluded by view)
        _insert_trade(db, is_diagnostic=1)
        df = load_production_trades(db)
        assert df.empty

    def test_production_trades_excludes_diagnostic(self, tmp_path):
        db = _build_db(tmp_path)
        _insert_trade(db, is_diagnostic=0, realized_pnl=100.0)
        _insert_trade(db, is_diagnostic=1, realized_pnl=999.0)  # excluded
        df = load_production_trades(db)
        assert len(df) == 1
        assert df["realized_pnl"].iloc[0] == 100.0

    def test_production_trades_excludes_synthetic(self, tmp_path):
        db = _build_db(tmp_path)
        _insert_trade(db, is_synthetic=0, realized_pnl=50.0)
        _insert_trade(db, is_synthetic=1, realized_pnl=999.0)  # excluded
        df = load_production_trades(db)
        assert len(df) == 1
        assert df["realized_pnl"].iloc[0] == 50.0

    def test_r_multiple_column_present(self, tmp_path):
        db = _build_db(tmp_path)
        _insert_trade(db, realized_pnl=21.0, entry_price=100.0, bar_high=108.0, bar_low=100.0)
        df = load_production_trades(db)
        assert "r_multiple" in df.columns
        # ATR = 8.0, risk_unit = 8*1.5=12.0, R = 21/12 = 1.75
        assert abs(df["r_multiple"].iloc[0] - 21.0 / (8.0 * 1.5)) < 0.01

    def test_correct_dir_neg_pnl_buy(self, tmp_path):
        """BUY that went up (exit > entry) but realized negative PnL = correct direction loss."""
        db = _build_db(tmp_path)
        _insert_trade(db, action="BUY", entry_price=100.0, exit_price=105.0, realized_pnl=-10.0)
        df = load_production_trades(db)
        assert df["correct_dir_neg_pnl"].iloc[0] == 1

    def test_correct_dir_neg_pnl_sell(self, tmp_path):
        """SELL that went down (exit < entry) but realized negative PnL = correct direction loss."""
        db = _build_db(tmp_path)
        _insert_trade(db, action="SELL", entry_price=100.0, exit_price=95.0, realized_pnl=-5.0)
        df = load_production_trades(db)
        assert df["correct_dir_neg_pnl"].iloc[0] == 1

    def test_tail_n_limits_rows(self, tmp_path):
        db = _build_db(tmp_path)
        for i in range(5):
            _insert_trade(db, realized_pnl=float(i * 10))
        df_all = load_production_trades(db)
        df_tail = load_production_trades(db, tail_n=3)
        assert len(df_all) == 5
        assert len(df_tail) == 3

    def test_handles_all_null_atr_inputs_without_dtype_error(self, tmp_path):
        db = _build_db(tmp_path)
        _insert_trade(
            db,
            realized_pnl=-5.0,
            entry_price=100.0,
            exit_price=99.0,
            bar_high=None,
            bar_low=None,
        )
        df = load_production_trades(db)
        assert len(df) == 1
        assert np.isfinite(float(df["atr_proxy"].iloc[0]))
        assert np.isfinite(float(df["r_multiple"].iloc[0]))


# ---------------------------------------------------------------------------
# compute_exit_reason_breakdown
# ---------------------------------------------------------------------------

class TestExitReasonBreakdown:
    def test_returns_empty_on_empty_input(self):
        df = compute_exit_reason_breakdown(pd.DataFrame())
        assert df.empty

    def test_groups_by_exit_reason(self):
        trades = _make_trade_rows([
            {"exit_reason": "stop_loss", "realized_pnl": -50.0, "is_winner": 0},
            {"exit_reason": "stop_loss", "realized_pnl": -30.0, "is_winner": 0},
            {"exit_reason": "time_exit", "realized_pnl": 10.0, "is_winner": 1},
        ])
        breakdown = compute_exit_reason_breakdown(trades)
        reasons = set(breakdown["exit_reason"])
        assert "stop_loss" in reasons
        assert "time_exit" in reasons

    def test_win_rate_per_exit_reason(self):
        trades = _make_trade_rows([
            {"exit_reason": "stop_loss", "realized_pnl": -50.0, "is_winner": 0},
            {"exit_reason": "stop_loss", "realized_pnl": -30.0, "is_winner": 0},
            {"exit_reason": "time_exit", "realized_pnl": 10.0, "is_winner": 1},
        ])
        breakdown = compute_exit_reason_breakdown(trades)
        sl = breakdown[breakdown["exit_reason"] == "stop_loss"].iloc[0]
        te = breakdown[breakdown["exit_reason"] == "time_exit"].iloc[0]
        assert sl["win_rate"] == pytest.approx(0.0)
        assert te["win_rate"] == pytest.approx(1.0)

    def test_stop_loss_exits_have_negative_mean_pnl(self):
        trades = _make_trade_rows([
            {"exit_reason": "stop_loss", "realized_pnl": -50.0, "is_winner": 0},
            {"exit_reason": "stop_loss", "realized_pnl": -30.0, "is_winner": 0},
        ])
        breakdown = compute_exit_reason_breakdown(trades)
        sl = breakdown[breakdown["exit_reason"] == "stop_loss"].iloc[0]
        assert sl["mean_pnl"] < 0.0


# ---------------------------------------------------------------------------
# diagnose_direction_gap
# ---------------------------------------------------------------------------

class TestDiagnoseDirectionGap:
    def test_no_trades_returns_empty_gracefully(self):
        gap = diagnose_direction_gap(pd.DataFrame())
        assert gap["total_trades"] == 0
        assert gap["interpretation"] == "no_data"

    def test_direction_gap_returns_required_keys(self):
        trades = _make_trade_rows([
            {"exit_reason": "stop_loss", "realized_pnl": -30.0, "is_winner": 0, "correct_dir_neg_pnl": 0},
            {"exit_reason": "time_exit", "realized_pnl": 20.0, "is_winner": 1, "correct_dir_neg_pnl": 0},
        ])
        gap = diagnose_direction_gap(trades)
        required = [
            "total_trades", "overall_win_rate", "stop_loss_pct", "time_exit_pct",
            "signal_exit_pct", "stop_loss_win_rate", "time_exit_win_rate",
            "signal_exit_win_rate", "correct_direction_negative_pnl",
            "pct_correct_dir_neg_pnl", "mean_holding_days_winners",
            "mean_holding_days_losers", "median_r_multiple_by_reason", "interpretation",
        ]
        for k in required:
            assert k in gap, f"Missing key: {k}"

    def test_stop_too_tight_interpretation(self):
        """When stop_loss > 40% of exits, interpretation should be stop_too_tight."""
        rows = [{"exit_reason": "stop_loss", "realized_pnl": -20.0, "is_winner": 0,
                 "correct_dir_neg_pnl": 0, "holding_period_days": 1}] * 5
        rows += [{"exit_reason": "time_exit", "realized_pnl": 10.0, "is_winner": 1,
                  "correct_dir_neg_pnl": 0, "holding_period_days": 2}] * 4
        trades = _make_trade_rows(rows)
        gap = diagnose_direction_gap(trades)
        assert gap["stop_loss_pct"] > 0.40
        assert gap["interpretation"] == "stop_too_tight"

    def test_holding_too_short_interpretation(self):
        """time_exit > 40% AND time_exit win-rate < 45% → holding_too_short."""
        rows = [{"exit_reason": "time_exit", "realized_pnl": -10.0, "is_winner": 0,
                 "correct_dir_neg_pnl": 0, "holding_period_days": 1}] * 5
        rows += [{"exit_reason": "signal_exit", "realized_pnl": 15.0, "is_winner": 1,
                  "correct_dir_neg_pnl": 0, "holding_period_days": 5}] * 4
        trades = _make_trade_rows(rows)
        gap = diagnose_direction_gap(trades)
        assert gap["time_exit_pct"] > 0.40
        assert gap["interpretation"] == "holding_too_short"

    def test_correct_direction_negative_pnl_counted(self):
        trades = _make_trade_rows([
            # BUY, exit > entry but PnL negative → correct direction, exit killed it
            {"action": "BUY", "entry_price": 100.0, "exit_price": 102.0,
             "realized_pnl": -5.0, "is_winner": 0, "correct_dir_neg_pnl": 1,
             "exit_reason": "stop_loss"},
            # Normal winner
            {"action": "BUY", "entry_price": 100.0, "exit_price": 105.0,
             "realized_pnl": 15.0, "is_winner": 1, "correct_dir_neg_pnl": 0,
             "exit_reason": "signal_exit"},
        ])
        gap = diagnose_direction_gap(trades)
        assert gap["correct_direction_negative_pnl"] == 1
        assert gap["pct_correct_dir_neg_pnl"] == pytest.approx(0.5)

    def test_median_r_multiple_by_exit_reason(self):
        trades = _make_trade_rows([
            {"exit_reason": "stop_loss", "r_multiple": -1.0, "is_winner": 0,
             "correct_dir_neg_pnl": 0, "realized_pnl": -10.0},
            {"exit_reason": "stop_loss", "r_multiple": -0.8, "is_winner": 0,
             "correct_dir_neg_pnl": 0, "realized_pnl": -8.0},
            {"exit_reason": "signal_exit", "r_multiple": 2.0, "is_winner": 1,
             "correct_dir_neg_pnl": 0, "realized_pnl": 20.0},
        ])
        gap = diagnose_direction_gap(trades)
        assert "stop_loss" in gap["median_r_multiple_by_reason"]
        assert "signal_exit" in gap["median_r_multiple_by_reason"]
        sl_r = gap["median_r_multiple_by_reason"]["stop_loss"]
        assert sl_r is not None and sl_r < 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_exits_0_with_valid_db(self, tmp_path):
        db = _build_db(tmp_path)
        _insert_trade(db)
        rc = main(["--db", str(db)])
        assert rc == 0

    def test_cli_exits_0_with_empty_db(self, tmp_path):
        db = _build_db(tmp_path)
        rc = main(["--db", str(db)])
        assert rc == 0

    def test_cli_exits_0_missing_db(self, tmp_path):
        rc = main(["--db", str(tmp_path / "ghost.db")])
        assert rc == 0  # non-blocking diagnostic
