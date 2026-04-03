"""
tests/scripts/test_validate_profitability_proof_hardening.py
-------------------------------------------------------------
Phase 7.21 tests: validate_profitability_proof.py must read all PnL metrics
exclusively from production_closed_trades view (INT-03 fix).

Key contracts:
  1. Synthetic trades (is_synthetic=1) never inflate win_rate or total_pnl.
  2. Diagnostic trades (is_diagnostic=1) are excluded.
  3. Opening legs (is_close=0) are excluded even when they carry realized_pnl.
  4. The returned metrics dict declares data_source='production_closed_trades'.
  5. get_trade_stats() / calculate_win_rate() work with the fallback filter
     when the view doesn't exist (pre-migration DB).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_profitability_proof import (
    calculate_win_rate,
    get_trade_stats,
    validate_profitability_proof,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE trade_executions (
    id INTEGER PRIMARY KEY,
    ticker TEXT,
    trade_date TEXT,
    action TEXT,
    realized_pnl REAL,
    entry_price REAL,
    exit_price REAL,
    is_close INTEGER DEFAULT 0,
    is_diagnostic INTEGER DEFAULT 0,
    is_synthetic INTEGER DEFAULT 0,
    is_contaminated INTEGER DEFAULT 0,
    entry_trade_id INTEGER,
    data_source TEXT
);
"""

_VIEW = """
CREATE VIEW production_closed_trades AS
    SELECT t.* FROM trade_executions t
    WHERE t.is_close = 1
      AND t.is_diagnostic = 0
      AND t.is_synthetic = 0
      AND t.is_contaminated = 0
      AND NOT EXISTS (
          SELECT 1 FROM trade_executions o
          WHERE o.id = t.entry_trade_id
            AND o.is_synthetic = 1
      );
"""


def _make_db(tmp_path: Path, with_view: bool = True) -> Path:
    db = tmp_path / "test.db"
    con = sqlite3.connect(db)
    con.executescript(_SCHEMA)
    if with_view:
        con.executescript(_VIEW)
    con.commit()
    con.close()
    return db


def _insert(db: Path, rows: list[tuple]) -> None:
    """rows: (id, ticker, trade_date, action, realized_pnl, entry_price, exit_price,
              is_close, is_diagnostic, is_synthetic, is_contaminated, entry_trade_id,
              data_source)"""
    con = sqlite3.connect(db)
    con.executemany(
        "INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def _cursor(db: Path) -> sqlite3.Cursor:
    con = sqlite3.connect(db)
    return con.cursor()


# ---------------------------------------------------------------------------
# Phase 7.21: get_trade_stats isolation
# ---------------------------------------------------------------------------

class TestGetTradeStatsFromCanonicalView:
    def test_excludes_synthetic_trades(self, tmp_path):
        db = _make_db(tmp_path)
        _insert(db, [
            # production winner
            (1, "AAPL", "2026-01-10", "SELL", 100.0, 100.0, 110.0, 1, 0, 0, 0, None, "live"),
            # synthetic with huge profit -- must be excluded
            (2, "SYN1", "2026-01-11", "SELL", 9999.0, 1.0, 9999.0, 1, 0, 1, 0, None, "synthetic"),
        ])
        cur = _cursor(db)
        stats = get_trade_stats(cur)
        assert stats["total_trades"] == 1, "Synthetic close must be excluded"
        assert abs(stats["total_pnl"] - 100.0) < 1e-6, "Synthetic PnL must not inflate total"

    def test_excludes_diagnostic_trades(self, tmp_path):
        db = _make_db(tmp_path)
        _insert(db, [
            (1, "AAPL", "2026-01-10", "SELL", 50.0, 100.0, 105.0, 1, 0, 0, 0, None, "live"),
            (2, "AAPL", "2026-01-11", "SELL", 500.0, 100.0, 150.0, 1, 1, 0, 0, None, "diagnostic"),
        ])
        cur = _cursor(db)
        stats = get_trade_stats(cur)
        assert stats["total_trades"] == 1
        assert abs(stats["total_pnl"] - 50.0) < 1e-6

    def test_excludes_opening_legs(self, tmp_path):
        db = _make_db(tmp_path)
        _insert(db, [
            # opening leg with stray PnL (integrity violation, but we must exclude it)
            (1, "AAPL", "2026-01-09", "BUY",  200.0, 100.0, None, 0, 0, 0, 0, None, "live"),
            # proper closing leg
            (2, "AAPL", "2026-01-10", "SELL",  50.0, 100.0, 105.0, 1, 0, 0, 0, 1, "live"),
        ])
        cur = _cursor(db)
        stats = get_trade_stats(cur)
        assert stats["total_trades"] == 1, "Opening leg must not count as a closed trade"
        assert abs(stats["total_pnl"] - 50.0) < 1e-6

    def test_profit_factor_computed_from_canonical_only(self, tmp_path):
        db = _make_db(tmp_path)
        _insert(db, [
            (1, "AAPL", "2026-01-10", "SELL",  100.0, 100.0, 110.0, 1, 0, 0, 0, None, "live"),
            (2, "AAPL", "2026-01-11", "SELL",  -40.0, 110.0, 106.0, 1, 0, 0, 0, None, "live"),
            # synthetic loss -- must not reduce production profit factor
            (3, "SYN0", "2026-01-12", "SELL", -500.0, 50.0, 1.0, 1, 0, 1, 0, None, "synthetic"),
        ])
        cur = _cursor(db)
        stats = get_trade_stats(cur)
        assert stats["winning_trades"] == 1
        assert stats["losing_trades"] == 1
        assert stats["profit_factor"] == pytest.approx(100.0 / 40.0, rel=1e-3)

    def test_fallback_filter_when_view_absent(self, tmp_path):
        db = _make_db(tmp_path, with_view=False)  # no view
        _insert(db, [
            (1, "AAPL", "2026-01-10", "SELL",  80.0, 100.0, 108.0, 1, 0, 0, 0, None, "live"),
            (2, "SYN1", "2026-01-11", "SELL", 999.0, 1.0, 999.0, 1, 0, 1, 0, None, "synthetic"),
        ])
        cur = _cursor(db)
        # Should not raise; fallback filter applies is_close + diagnostic + synthetic
        stats = get_trade_stats(cur)
        assert stats["total_trades"] == 1
        assert abs(stats["total_pnl"] - 80.0) < 1e-6


class TestCalculateWinRateFromCanonicalView:
    def test_synthetic_wins_do_not_inflate_win_rate(self, tmp_path):
        db = _make_db(tmp_path)
        _insert(db, [
            # 1 real win
            (1, "AAPL", "2026-01-10", "SELL",  30.0, 100.0, 103.0, 1, 0, 0, 0, None, "live"),
            # 1 real loss
            (2, "AAPL", "2026-01-11", "SELL", -10.0, 103.0, 102.0, 1, 0, 0, 0, None, "live"),
            # synthetic wins (100%) -- must be excluded
            (3, "SYN0", "2026-01-12", "SELL", 999.0, 1.0, 999.0, 1, 0, 1, 0, None, "synthetic"),
            (4, "SYN1", "2026-01-13", "SELL", 888.0, 1.0, 888.0, 1, 0, 1, 0, None, "synthetic"),
        ])
        cur = _cursor(db)
        win_rate = calculate_win_rate(cur)
        assert win_rate == pytest.approx(0.50, abs=0.01), (
            "Win rate must be 50% from 1W/1L production trades, not inflated by synthetic wins"
        )

    def test_returns_none_when_no_production_closed_trades(self, tmp_path):
        db = _make_db(tmp_path)
        _insert(db, [
            (1, "AAPL", "2026-01-10", "BUY", None, 100.0, None, 0, 0, 0, 0, None, "live"),
        ])
        cur = _cursor(db)
        win_rate = calculate_win_rate(cur)
        assert win_rate is None


class TestValidateProfitabilityProofIntegration:
    """End-to-end: validate_profitability_proof() must not be fooled by contaminated data."""

    def _proof_db(self, tmp_path: Path) -> Path:
        """Create a DB usable by validate_profitability_proof (needs integrity module)."""
        from integrity.sqlite_guardrails import guarded_sqlite_connect

        db = tmp_path / "pmx.db"
        con = guarded_sqlite_connect(str(db), allow_schema_changes=True)
        con.execute(_SCHEMA)
        con.execute(_VIEW)
        con.commit()
        con.close()
        return db

    def test_proof_declares_canonical_data_source(self, tmp_path):
        try:
            db = self._proof_db(tmp_path)
        except Exception:
            pytest.skip("guarded_sqlite_connect unavailable in this environment")

        _insert(db, [
            (1, "AAPL", "2026-01-10", "BUY",  None,  100.0, None, 0, 0, 0, 0, None, "live"),
            (2, "AAPL", "2026-01-11", "SELL",  50.0, 100.0, 105.0, 1, 0, 0, 0, 1, "live"),
        ])
        result = validate_profitability_proof(str(db))
        assert result["metrics"].get("data_source") == "production_closed_trades", (
            "Phase 7.21 fix: metrics must declare data_source=production_closed_trades"
        )

    def test_synthetic_only_profits_do_not_make_proof_valid(self, tmp_path):
        """If all profitable trades are synthetic, the proof must not pass."""
        try:
            db = self._proof_db(tmp_path)
        except Exception:
            pytest.skip("guarded_sqlite_connect unavailable in this environment")

        # Insert 35 synthetic wins (above min_closed_trades=30 threshold)
        rows = []
        for i in range(35):
            rows.append((
                i + 1, f"SYN{i}", f"2026-01-{(i % 28) + 1:02d}", "SELL",
                100.0, 10.0, 20.0, 1, 0, 1, 0, None, "synthetic"
            ))
        _insert(db, rows)

        result = validate_profitability_proof(str(db))
        # production_closed_trades excludes is_synthetic=1, so total_trades=0 -> FAIL
        assert result["is_proof_valid"] is False, (
            "Proof must be INVALID when only synthetic trades exist"
        )
        assert result["is_profitable"] is False

    def test_diagnostic_trades_excluded_from_proof(self, tmp_path):
        try:
            db = self._proof_db(tmp_path)
        except Exception:
            pytest.skip("guarded_sqlite_connect unavailable in this environment")

        # 35 diagnostic wins only
        rows = []
        for i in range(35):
            rows.append((
                i + 1, "AAPL", f"2026-01-{(i % 28) + 1:02d}", "SELL",
                200.0, 100.0, 120.0, 1, 1, 0, 0, None, "diagnostic"
            ))
        _insert(db, rows)

        result = validate_profitability_proof(str(db))
        assert result["is_proof_valid"] is False, (
            "Proof must be INVALID when only diagnostic trades exist"
        )


# ---------------------------------------------------------------------------
# Phase 7.40: import-path collision tests
# ---------------------------------------------------------------------------

class TestImportPathCollision:
    """validate_profitability_proof must function without the integrity package."""

    def test_gate_runs_without_guarded_connect(self, tmp_path, monkeypatch):
        """If integrity.sqlite_guardrails is absent, the gate falls back to sqlite3
        and still returns a valid result dict — no ImportError propagates."""
        import builtins

        db = _make_db(tmp_path)
        _insert(db, [
            (1, "AAPL", "2026-01-10", "SELL", 80.0, 100.0, 108.0, 1, 0, 0, 0, None, "live"),
        ])

        real_import = builtins.__import__

        def _block_guarded(name, *args, **kwargs):
            if "sqlite_guardrails" in name or (
                name == "integrity" and args and "sqlite_guardrails" in (args[2] or [])
            ):
                raise ImportError("integrity.sqlite_guardrails blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_guarded)
        # Should not raise; falls back to sqlite3.connect
        result = validate_profitability_proof(str(db))
        assert isinstance(result, dict)
        assert "is_proof_valid" in result
        assert "violations" in result

    def test_wiring_error_does_not_count_as_evidence_violation(self, tmp_path, monkeypatch):
        """An ImportError in the connect path must not add a violation entry.
        Only evidence issues (missing trades, bad stats) should trigger violations."""
        import builtins

        db = _make_db(tmp_path)
        # Insert enough production closed trades to pass statistical thresholds
        rows = []
        for i in range(35):
            rows.append((
                i + 1, "AAPL", f"2026-01-{(i % 28) + 1:02d}", "SELL",
                50.0 if i % 2 == 0 else -20.0,
                100.0, 110.0 if i % 2 == 0 else 98.0,
                1, 0, 0, 0, None, "live",
            ))
        _insert(db, rows)

        real_import = builtins.__import__

        def _block_guarded(name, *args, **kwargs):
            if "sqlite_guardrails" in name:
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_guarded)
        result = validate_profitability_proof(str(db))
        # The violations list must not contain any import/wiring error string
        for v in result["violations"]:
            assert "import" not in v.lower(), f"Wiring error leaked into violations: {v}"
            assert "sqlite_guardrails" not in v.lower()

    def test_fallback_rejects_null_flags_and_contaminated_rows(self, tmp_path):
        db = _make_db(tmp_path, with_view=False)
        _insert(db, [
            (1, "AAPL", "2026-01-10", "SELL", 40.0, 100.0, 104.0, 1, None, None, 0, None, "live"),
            (2, "MSFT", "2026-01-11", "SELL", 60.0, 200.0, 206.0, 1, 0, 0, 1, None, "live"),
            (3, "NVDA", "2026-01-12", "SELL", 30.0, 300.0, 303.0, 1, 0, 0, 0, None, "live"),
        ])
        cur = _cursor(db)
        stats = get_trade_stats(cur)
        assert stats["total_trades"] == 1
        assert stats["total_pnl"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Adversarial check: INT-03 must now be CLEARED
# ---------------------------------------------------------------------------

def test_int03_now_cleared_in_adversarial_runner():
    """INT-03 adversarial check must read 'production_closed_trades' in the fixed source."""
    from scripts.adversarial_diagnostic_runner import chk_proof_raw_table
    result = chk_proof_raw_table(ROOT / "data" / "portfolio_maximizer.db")
    assert result.id == "INT-03"
    assert result.passed is True, (
        "INT-03 must be CLEARED after Phase 7.21 fix. "
        "validate_profitability_proof.py now uses production_closed_trades view."
    )
