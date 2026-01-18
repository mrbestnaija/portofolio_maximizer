from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from etl.regime_detector import RegimeState
from scripts import run_ts_model_search as ts_search


class DummyProfile:
    """Minimal profile stub with the attributes used by the candidate builder."""

    def __init__(self, name: str) -> None:
        self.name = name
        # Keep kwargs simple but valid for SARIMAX path.
        self.payload: Dict[str, Any] = {
            "sarimax": {"enabled": True, "max_p": 1, "max_d": 1, "max_q": 1},
            "samossa": {"enabled": True, "window_length": 20, "n_components": 4},
        }


class DummyRollingValidator:
    """
    Lightweight stand-in for RollingWindowValidator.

    Returns a fixed aggregate_metrics/folds structure so that scoring,
    stability, and DM tests can run without touching the real forecaster.
    """

    def __init__(self, forecaster_config: Any, cv_config: Any) -> None:  # noqa: D401
        self.forecaster_config = forecaster_config
        self.cv_config = cv_config

    def run(self, price_series: pd.Series, returns_series: pd.Series) -> Dict[str, Any]:
        # Two folds with slightly different RMSE values to exercise stability logic.
        aggregate_metrics = {"combined": {"rmse": 1.0}}
        folds: List[Dict[str, Any]] = [
            {"metrics": {"combined": {"rmse": 1.0}}},
            {"metrics": {"combined": {"rmse": 1.1}}},
        ]
        return {"aggregate_metrics": aggregate_metrics, "folds": folds}


def test_run_ts_model_search_uses_overrides_and_persists_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    """
    Smoke-test: run_ts_model_search.main uses select_profile_with_overrides and
    save_ts_model_candidate when profiles are enabled.
    """

    # Use a temporary DB path; DatabaseManager will create schema as needed.
    db_path = tmp_path / "ts_search_test.db"

    # Provide a simple synthetic price series via the internal loader to avoid
    # relying on actual OHLCV storage.
    def fake_load_price_series(db: Any, ticker: str, lookback_days: int) -> pd.Series:
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        prices = np.linspace(100.0, 110.0, num=len(dates))
        series = pd.Series(prices, index=dates, name="close")
        return series

    monkeypatch.setattr(
        ts_search,
        "_load_price_series",
        fake_load_price_series,
        raising=False,
    )

    # Replace RollingWindowValidator with a lightweight deterministic stub.
    monkeypatch.setattr(
        ts_search,
        "RollingWindowValidator",
        DummyRollingValidator,
        raising=False,
    )

    # Capture calls to select_profile_with_overrides to ensure it is used.
    calls: Dict[str, Any] = {}

    def fake_select_profile_with_overrides(
        ticker: str,
        sleeve: str,
        returns,
        **kwargs,
    ):
        calls["ticker"] = ticker
        calls["sleeve"] = sleeve
        calls["returns_len"] = len(returns)
        profile = DummyProfile("override_profile")
        regime_state = RegimeState(
            regime_type="high_vol",
            confidence=0.9,
            duration=10,
            transition_probability=0.1,
        )
        override = None
        return profile, regime_state, override

    monkeypatch.setattr(
        ts_search,
        "select_profile_with_overrides",
        fake_select_profile_with_overrides,
        raising=False,
    )

    # Capture persisted TS model candidates without depending on the real DB insert.
    saved_candidates: List[Dict[str, Any]] = []

    def fake_save_ts_model_candidate(
        self,
        ticker: str,
        regime: str,
        candidate_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        stability=None,
        score=None,
    ) -> int:
        saved_candidates.append(
            {
                "ticker": ticker,
                "regime": regime,
                "candidate_name": candidate_name,
                "params": params,
                "metrics": metrics,
                "stability": stability,
                "score": score,
            }
        )
        return 1

    monkeypatch.setattr(
        "etl.database_manager.DatabaseManager.save_ts_model_candidate",
        fake_save_ts_model_candidate,
        raising=False,
    )

    # Call the Click command's underlying callback directly to bypass CLI parsing.
    ts_search.main.callback(
        tickers="AAPL",
        db_path=str(db_path),
        lookback_days=180,
        regime="default",
        min_train_size=20,
        horizon=5,
        step_size=10,
        max_folds=2,
        use_profiles=True,
        verbose=False,
    )

    # Ensure the override-aware selector was called with our ticker and a sleeve label.
    assert calls.get("ticker") == "AAPL"
    assert "sleeve" in calls and isinstance(calls["sleeve"], str)
    assert calls.get("returns_len", 0) > 0

    # At least one TS model candidate should have been "saved" via the fake saver.
    assert saved_candidates, "No TS model candidates were persisted in the smoke test."
    # All saved entries should be for the requested ticker.
    assert all(cand["ticker"] == "AAPL" for cand in saved_candidates)
