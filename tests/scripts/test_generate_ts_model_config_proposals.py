import json
import sqlite3
from pathlib import Path

import pytest

from scripts.generate_ts_model_config_proposals import _load_best_candidates


def _init_ts_candidates_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE ts_model_candidates (
                ticker TEXT,
                regime TEXT,
                candidate_name TEXT,
                score REAL,
                stability REAL,
                metrics TEXT,
                created_at TEXT
            )
            """
        )

        # Baseline candidate with lower score and higher p-value.
        baseline_metrics = json.dumps(
            {"dm_vs_baseline": {"better_model": "baseline", "p_value": 0.50}}
        )
        cur.execute(
            """
            INSERT INTO ts_model_candidates
                (ticker, regime, candidate_name, score, stability, metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", "high_vol", "baseline", 0.5, 0.9, baseline_metrics, "2025-01-01T00:00:00Z"),
        )

        # Better candidate with higher score and lower p-value.
        better_metrics = json.dumps(
            {"dm_vs_baseline": {"better_model": "better_model", "p_value": 0.01}}
        )
        cur.execute(
            """
            INSERT INTO ts_model_candidates
                (ticker, regime, candidate_name, score, stability, metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", "high_vol", "sarimax_samossa", 1.0, 0.8, better_metrics, "2025-01-02T00:00:00Z"),
        )

        conn.commit()
    finally:
        conn.close()


def test_load_best_candidates_picks_highest_score_and_parses_dm(tmp_path: Path) -> None:
    """_load_best_candidates should pick the highest-score candidate per key and parse DM stats."""
    db_path = tmp_path / "ts_model_candidates.db"
    _init_ts_candidates_db(db_path)

    candidates = _load_best_candidates(db_path)

    # Only one (ticker, regime) pair, so we should get a single best candidate.
    assert len(candidates) == 1
    cand = candidates[0]

    assert cand.ticker == "AAPL"
    assert cand.regime == "high_vol"
    assert cand.candidate_name == "sarimax_samossa"
    assert cand.score == pytest.approx(1.0)
    assert cand.stability == pytest.approx(0.8)
    # DM block should be parsed from metrics JSON.
    assert cand.dm_better_model == "better_model"
    assert cand.dm_p_value == pytest.approx(0.01)

