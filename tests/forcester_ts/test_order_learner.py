"""
tests/forcester_ts/test_order_learner.py
-----------------------------------------
Unit tests for OrderLearner — AIC-weighted order cache with regime conditioning.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from forcester_ts.order_learner import OrderLearner, _NO_REGIME, _canonical_json, _regime_key


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS model_order_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL,
    model_type   TEXT    NOT NULL,
    regime       TEXT,
    order_params TEXT    NOT NULL,
    n_fits       INTEGER DEFAULT 0,
    aic_sum      REAL    DEFAULT 0.0,
    bic_sum      REAL    DEFAULT 0.0,
    best_aic     REAL,
    last_used    DATE,
    first_seen   DATE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, model_type, regime, order_params)
)
"""


def _make_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test_order_learner.db")
    conn = sqlite3.connect(db_path)
    conn.execute(DDL)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def db_path(tmp_path):
    return _make_db(tmp_path)


@pytest.fixture
def learner(db_path):
    return OrderLearner(db_path=db_path, config={
        "min_fits_to_suggest": 3,
        "max_order_age_days": 90,
        "skip_grid_threshold": 5,
    })


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _REGIME_KEY_HELPER(regime):
    return _regime_key(regime)


def _canonical(d):
    return _canonical_json(d)


# ---------------------------------------------------------------------------
# _canonical_json + _regime_key helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_canonical_json_stable(self):
        d1 = {"q": 1, "p": 2, "dist": "normal"}
        d2 = {"dist": "normal", "p": 2, "q": 1}
        assert _canonical(d1) == _canonical(d2)

    def test_regime_key_none_maps_to_sentinel(self):
        assert _regime_key(None) == _NO_REGIME

    def test_regime_key_passthrough(self):
        assert _regime_key("CRISIS") == "CRISIS"


# ---------------------------------------------------------------------------
# record_fit + suggest
# ---------------------------------------------------------------------------

class TestRecordAndSuggest:
    def test_suggest_returns_none_below_min_fits(self, learner):
        learner.record_fit("AAPL", "GARCH", "TRENDING",
                           {"p": 1, "q": 1, "dist": "skewt", "mean": "AR"},
                           aic=300.0, bic=320.0, n_obs=200)
        # Only 1 fit — below min_fits=3
        result = learner.suggest("AAPL", "GARCH", "TRENDING")
        assert result is None

    def test_suggest_returns_best_aic_order(self, learner):
        params_a = {"p": 1, "q": 1, "dist": "normal", "mean": "Zero"}
        params_b = {"p": 2, "q": 1, "dist": "skewt", "mean": "AR"}

        # Record params_a 3 times with good AIC
        for _ in range(3):
            learner.record_fit("AAPL", "GARCH", None, params_a,
                               aic=250.0, bic=270.0, n_obs=200)
        # Record params_b 3 times with worse AIC
        for _ in range(3):
            learner.record_fit("AAPL", "GARCH", None, params_b,
                               aic=300.0, bic=320.0, n_obs=200)

        suggestion = learner.suggest("AAPL", "GARCH", None)
        assert suggestion == params_a

    def test_n_fits_accumulates(self, learner):
        params = {"p": 1, "q": 1}
        for _ in range(5):
            learner.record_fit("AAPL", "GARCH", None, params,
                               aic=280.0, bic=300.0, n_obs=100)
        conn = sqlite3.connect(learner._db_path)
        row = conn.execute(
            "SELECT n_fits FROM model_order_stats WHERE ticker='AAPL'"
        ).fetchone()
        conn.close()
        assert row[0] == 5

    def test_aic_sum_accumulates(self, learner):
        params = {"p": 1, "q": 1}
        for aic in [280.0, 290.0, 300.0]:
            learner.record_fit("AAPL", "GARCH", None, params,
                               aic=aic, bic=0.0, n_obs=100)
        conn = sqlite3.connect(learner._db_path)
        row = conn.execute(
            "SELECT aic_sum FROM model_order_stats WHERE ticker='AAPL'"
        ).fetchone()
        conn.close()
        assert row[0] == pytest.approx(870.0)

    def test_best_aic_tracks_minimum(self, learner):
        params = {"p": 1, "q": 1}
        for aic in [300.0, 250.0, 275.0]:
            learner.record_fit("AAPL", "GARCH", None, params,
                               aic=aic, bic=0.0, n_obs=100)
        conn = sqlite3.connect(learner._db_path)
        row = conn.execute(
            "SELECT best_aic FROM model_order_stats WHERE ticker='AAPL'"
        ).fetchone()
        conn.close()
        assert row[0] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Regime fallback
# ---------------------------------------------------------------------------

class TestRegimeFallback:
    def test_suggest_falls_back_to_no_regime(self, learner):
        params_generic = {"p": 1, "q": 1, "dist": "normal"}
        # Record 3 fits with regime=None (stored as _NO_REGIME)
        for _ in range(3):
            learner.record_fit("AAPL", "GARCH", None, params_generic,
                               aic=260.0, bic=280.0, n_obs=150)

        # Suggest for specific regime — no CRISIS entries exist
        result = learner.suggest("AAPL", "GARCH", "CRISIS")
        assert result == params_generic

    def test_regime_specific_takes_priority(self, learner):
        generic_params = {"p": 1, "q": 1, "dist": "normal"}
        crisis_params = {"p": 2, "q": 2, "dist": "skewt"}

        for _ in range(3):
            learner.record_fit("AAPL", "GARCH", None, generic_params,
                               aic=260.0, bic=0.0, n_obs=100)
        for _ in range(3):
            learner.record_fit("AAPL", "GARCH", "CRISIS", crisis_params,
                               aic=240.0, bic=0.0, n_obs=100)

        result = learner.suggest("AAPL", "GARCH", "CRISIS")
        assert result == crisis_params

    def test_suggest_returns_none_no_entries(self, learner):
        result = learner.suggest("NVDA", "GARCH", "TRENDING")
        assert result is None

    def test_suggest_falls_back_to_legacy_unknown_rows(self, db_path):
        learner = OrderLearner(
            db_path=db_path,
            config={"min_fits_to_suggest": 3, "max_order_age_days": 90, "skip_grid_threshold": 5},
        )
        params = {"p": 1, "q": 2}
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO model_order_stats
                (ticker, model_type, regime, order_params, n_fits, aic_sum, bic_sum, best_aic, last_used, first_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, DATE('now'), DATE('now'))
            """,
            ("AAPL", "SARIMAX", "UNKNOWN", json.dumps(params), 3, 330.0, 360.0, 110.0),
        )
        conn.commit()
        conn.close()

        assert learner.suggest("AAPL", "SARIMAX", "CRISIS") == params


# ---------------------------------------------------------------------------
# should_skip_grid
# ---------------------------------------------------------------------------

class TestSkipGrid:
    def test_skip_grid_false_below_threshold(self, learner):
        params = {"p": 1, "q": 1}
        for _ in range(4):  # threshold=5
            learner.record_fit("AAPL", "GARCH", None, params,
                               aic=280.0, bic=300.0, n_obs=100)
        assert learner.should_skip_grid("AAPL", "GARCH", None) is False

    def test_skip_grid_true_at_threshold(self, learner):
        params = {"p": 1, "q": 1}
        for _ in range(5):
            learner.record_fit("AAPL", "GARCH", None, params,
                               aic=280.0, bic=300.0, n_obs=100)
        assert learner.should_skip_grid("AAPL", "GARCH", None) is True

    def test_skip_grid_false_no_entries(self, learner):
        assert learner.should_skip_grid("UNKNOWN", "GARCH", None) is False

    def test_threshold_config_cannot_lower_baseline(self, db_path):
        learner = OrderLearner(
            db_path=db_path,
            config={
                "min_fits_to_suggest": 1,
                "max_order_age_days": 0,
                "skip_grid_threshold": 2,
            },
        )
        params = {"p": 1, "q": 1}
        for _ in range(2):
            learner.record_fit("AAPL", "GARCH", None, params, aic=280.0, bic=300.0, n_obs=100)
        assert learner.suggest("AAPL", "GARCH", None) is None
        assert learner.should_skip_grid("AAPL", "GARCH", None) is False
        learner.record_fit("AAPL", "GARCH", None, params, aic=279.0, bic=299.0, n_obs=100)
        assert learner.suggest("AAPL", "GARCH", None) == params
        assert learner.should_skip_grid("AAPL", "GARCH", None) is False
        for _ in range(2):
            learner.record_fit("AAPL", "GARCH", None, params, aic=278.0, bic=298.0, n_obs=100)
        assert learner.should_skip_grid("AAPL", "GARCH", None) is True


# ---------------------------------------------------------------------------
# prune_stale
# ---------------------------------------------------------------------------

class TestPruneStale:
    def test_prune_removes_old_entries(self, learner):
        params = {"p": 1, "q": 1}
        learner.record_fit("AAPL", "GARCH", None, params,
                           aic=280.0, bic=300.0, n_obs=100)

        # Backdate the entry
        conn = sqlite3.connect(learner._db_path)
        conn.execute(
            "UPDATE model_order_stats SET last_used = '2020-01-01'"
        )
        conn.commit()
        conn.close()

        deleted = learner.prune_stale()
        assert deleted >= 1

    def test_prune_keeps_recent_entries(self, learner):
        params = {"p": 1, "q": 1}
        learner.record_fit("AAPL", "GARCH", None, params,
                           aic=280.0, bic=300.0, n_obs=100)
        deleted = learner.prune_stale()
        assert deleted == 0


# ---------------------------------------------------------------------------
# coverage_stats
# ---------------------------------------------------------------------------

class TestCoverageStats:
    def test_coverage_stats_empty(self, learner):
        stats = learner.coverage_stats()
        assert stats["total_entries"] == 0
        assert stats["qualified_entries"] == 0

    def test_coverage_stats_after_fits(self, learner):
        params = {"p": 1, "q": 1}
        for _ in range(3):
            learner.record_fit("AAPL", "GARCH", None, params,
                               aic=280.0, bic=300.0, n_obs=100)
        stats = learner.coverage_stats()
        assert stats["total_entries"] == 1
        assert stats["qualified_entries"] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nan_aic_handled(self, learner):
        """Non-finite AIC should be ignored so invalid fits never short-circuit search."""
        params = {"p": 1, "q": 1}
        learner.record_fit("AAPL", "GARCH", None, params,
                           aic=float("nan"), bic=float("nan"), n_obs=100)
        conn = sqlite3.connect(learner._db_path)
        count = conn.execute("SELECT COUNT(*) FROM model_order_stats").fetchone()[0]
        conn.close()
        assert count == 0

    def test_empty_ticker_ignored(self, learner):
        """Empty ticker should not insert a row."""
        learner.record_fit("", "GARCH", None, {"p": 1},
                           aic=100.0, bic=110.0, n_obs=50)
        conn = sqlite3.connect(learner._db_path)
        count = conn.execute("SELECT COUNT(*) FROM model_order_stats").fetchone()[0]
        conn.close()
        assert count == 0

    def test_samossa_arima_model_type(self, learner):
        params = {"ar_lag": 2}
        for _ in range(3):
            learner.record_fit("AAPL", "SAMOSSA_ARIMA", None, params,
                               aic=50.0, bic=55.0, n_obs=80)
        result = learner.suggest("AAPL", "SAMOSSA_ARIMA", None)
        assert result == params
