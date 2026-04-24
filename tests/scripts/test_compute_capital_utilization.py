"""Tests for compute_capital_utilization.py

Pins the exact formula and denominator so the KPI cannot silently change.

Formula under test:
    deployment_fraction = sum(notional_i * hold_days_i) / (capital * total_days)

Key regression invariants:
1. denominator is (capital × total_days), NOT sum_of_notionals or trade_count
2. avg_notional per trade is NOT the same as time-weighted capital
3. overstatement factor = avg_notional / twc_per_day
4. scenario projections scale PnL linearly with trades/day (same-edge assumption)
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from scripts.compute_capital_utilization import compute_utilization, _project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_db(tmp_path):
    """Minimal DB with known values for formula verification."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()

    # Schema subset required by the script
    cur.execute("""
        CREATE TABLE portfolio_cash_state (
            id INTEGER PRIMARY KEY,
            cash REAL,
            initial_capital REAL,
            updated_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            trade_date TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            total_value REAL,
            realized_pnl REAL,
            holding_period_days REAL,
            is_close INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            is_diagnostic INTEGER DEFAULT 0,
            entry_trade_id INTEGER,
            exit_reason TEXT
        )
    """)

    # Canonical production_closed_trades view (mirrors integrity enforcer definition)
    cur.execute("""
        CREATE VIEW production_closed_trades AS
        SELECT * FROM trade_executions
        WHERE is_close = 1
          AND COALESCE(is_synthetic, 0) = 0
          AND COALESCE(is_diagnostic, 0) = 0
    """)

    cur.execute(
        "INSERT INTO portfolio_cash_state VALUES (1, 24000.0, 25000.0, '2026-01-01')"
    )

    # Two known round-trips:
    # Trade A: NVDA open  id=1  price=200, shares=4, date=2026-01-10
    # Trade A: NVDA close id=2  holding=2d, pnl=+80, entry_trade_id=1
    # Trade B: AAPL open  id=3  price=250, shares=2, date=2026-01-15
    # Trade B: AAPL close id=4  holding=4d, pnl=-40, entry_trade_id=3
    cur.executemany(
        "INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "NVDA", "2026-01-10", "BUY",  4, 200.0, 800.0,  None,  None, 0, 0, 0, None,  None),
            (2, "NVDA", "2026-01-12", "SELL", 4, 204.0, 816.0,  80.0,  2.0,  1, 0, 0, 1,     "TAKE_PROFIT"),
            (3, "AAPL", "2026-01-15", "BUY",  2, 250.0, 500.0,  None,  None, 0, 0, 0, None,  None),
            (4, "AAPL", "2026-01-19", "SELL", 2, 230.0, 460.0, -40.0,  4.0,  1, 0, 0, 3,     "STOP_LOSS"),
        ],
    )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# Formula regression tests
# ---------------------------------------------------------------------------

class TestFormulaContract:
    """Pin the exact formula and denominator — must not change silently."""

    def test_denominator_is_capital_times_total_days(self, minimal_db):
        """deployment_fraction = notional_days / (capital * total_days)"""
        result = compute_utilization(minimal_db)

        # Trade A: notional=800, hold=2 → 1600 notional-days
        # Trade B: notional=500, hold=4 → 2000 notional-days
        # Total notional_days = 3600
        expected_nd = 800 * 2 + 500 * 4  # = 3600
        assert result["notional_days"] == pytest.approx(expected_nd, abs=1.0)

        # total_days = 2026-01-19 - 2026-01-10 = 9 days
        assert result["total_days"] == 9

        # twc_per_day = 3600 / 9 = 400
        assert result["twc_per_day"] == pytest.approx(400.0, abs=1.0)

        # deployment_fraction = 400 / 25000 = 0.016
        assert result["deployment_fraction"] == pytest.approx(400.0 / 25000.0, rel=0.01)

    def test_avg_notional_is_NOT_the_denominator(self, minimal_db):
        """avg_notional / capital must NOT equal deployment_fraction."""
        result = compute_utilization(minimal_db)
        avg_notional_frac = result["avg_notional_per_trade"] / result["capital"]
        # avg_notional = (800+500)/2 = 650 → 650/25000 = 2.6%
        # twc = 400/25000 = 1.6%
        # They are not equal
        assert avg_notional_frac != pytest.approx(result["deployment_fraction"], rel=0.01), (
            "avg_notional/capital == deployment_fraction — denominator may have regressed "
            "to the wrong proxy"
        )

    def test_overstatement_factor_is_avg_notional_over_twc(self, minimal_db):
        """overstatement = avg_notional / twc_per_day — pinned formula."""
        result = compute_utilization(minimal_db)
        expected = result["avg_notional_per_trade"] / result["twc_per_day"]
        assert result["avg_notional_overstatement_factor"] == pytest.approx(expected, rel=0.01)

    def test_capital_comes_from_portfolio_cash_state(self, minimal_db):
        result = compute_utilization(minimal_db)
        assert result["capital"] == 25000.0

    def test_capital_override_is_respected(self, minimal_db):
        result = compute_utilization(minimal_db, capital=10000.0)
        assert result["capital"] == 10000.0
        assert result["deployment_fraction"] == pytest.approx(
            result["twc_per_day"] / 10000.0, rel=0.01
        )


class TestScenarioProjection:
    """Scenario projections scale PnL linearly with trades/day."""

    def test_projection_scales_linearly(self, minimal_db):
        result = compute_utilization(minimal_db)
        base_trades_per_day = result["trades_per_day"]
        base_pnl = result["total_pnl"]

        for name, s in result["scenarios"].items():
            expected_scale = s["trades_per_day"] / base_trades_per_day
            expected_pnl = base_pnl * expected_scale
            assert s["proj_pnl"] == pytest.approx(expected_pnl, rel=0.01), (
                f"Scenario {name}: expected proj_pnl {expected_pnl:.2f}, got {s['proj_pnl']:.2f}"
            )

    def test_current_scenario_matches_actual_pnl(self, minimal_db):
        result = compute_utilization(minimal_db)
        current = result["scenarios"]["current"]
        assert current["proj_pnl"] == pytest.approx(result["total_pnl"], rel=0.01)

    def test_project_helper_is_consistent(self):
        s = _project(base_pnl=100.0, base_trips=10, total_days=20, capital=5000.0,
                     target_trades_per_day=1.0)
        # base trades/day = 10/20 = 0.5; scale = 1.0/0.5 = 2.0
        assert s["scale_factor"] == pytest.approx(2.0, rel=0.01)
        assert s["proj_pnl"] == pytest.approx(200.0, rel=0.01)


class TestEdgeCases:

    def test_raises_on_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE portfolio_cash_state (id INTEGER, cash REAL, initial_capital REAL, updated_at TEXT)")
        conn.execute("CREATE TABLE trade_executions (id INTEGER, ticker TEXT, trade_date TEXT, action TEXT, shares REAL, price REAL, total_value REAL, realized_pnl REAL, holding_period_days REAL, is_close INTEGER DEFAULT 0, is_synthetic INTEGER DEFAULT 0, is_diagnostic INTEGER DEFAULT 0, entry_trade_id INTEGER, exit_reason TEXT)")
        conn.execute("CREATE VIEW production_closed_trades AS SELECT * FROM trade_executions WHERE is_close=1 AND COALESCE(is_synthetic,0)=0 AND COALESCE(is_diagnostic,0)=0")
        conn.execute("INSERT INTO portfolio_cash_state VALUES (1, 25000, 25000, '2026-01-01')")
        conn.commit()
        conn.close()
        with pytest.raises(ValueError, match="No closed round-trips"):
            compute_utilization(db)

    def test_synthetic_trades_excluded(self, tmp_path):
        """Synthetic trades must not contribute to notional_days."""
        db = tmp_path / "synth.db"
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cur.execute("CREATE TABLE portfolio_cash_state (id INTEGER, cash REAL, initial_capital REAL, updated_at TEXT)")
        cur.execute("CREATE TABLE trade_executions (id INTEGER, ticker TEXT, trade_date TEXT, action TEXT, shares REAL, price REAL, total_value REAL, realized_pnl REAL, holding_period_days REAL, is_close INTEGER DEFAULT 0, is_synthetic INTEGER DEFAULT 0, is_diagnostic INTEGER DEFAULT 0, entry_trade_id INTEGER, exit_reason TEXT)")
        cur.execute("CREATE VIEW production_closed_trades AS SELECT * FROM trade_executions WHERE is_close=1 AND COALESCE(is_synthetic,0)=0 AND COALESCE(is_diagnostic,0)=0")
        cur.execute("INSERT INTO portfolio_cash_state VALUES (1, 24000, 25000, '2026-01-01')")
        # One real close + one synthetic close
        cur.executemany("INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, "NVDA", "2026-01-10", "BUY",  2, 100.0, 200.0, None, None, 0, 0, 0, None, None),
            (2, "NVDA", "2026-01-12", "SELL", 2, 110.0, 220.0, 20.0, 2.0,  1, 0, 0, 1,    "TAKE_PROFIT"),
            (3, "AAPL", "2026-01-10", "BUY",  3, 200.0, 600.0, None, None, 0, 1, 0, None, None),
            (4, "AAPL", "2026-01-14", "SELL", 3, 190.0, 570.0,-30.0, 4.0,  1, 1, 0, 3,    "STOP_LOSS"),
        ])
        conn.commit()
        conn.close()
        result = compute_utilization(db)
        assert result["n_trips"] == 1  # only the real close
        # notional_days = 200 * 2 = 400
        assert result["notional_days"] == pytest.approx(400.0, abs=1.0)

    def test_json_output_contains_formula_field(self, minimal_db):
        result = compute_utilization(minimal_db)
        assert "formula" in result
        assert result["formula"] == "notional_days / (capital * total_days)"
