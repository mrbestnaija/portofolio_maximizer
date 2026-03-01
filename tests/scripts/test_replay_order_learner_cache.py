from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import replay_order_learner_cache as mod


def _write_forecasting_config(path: Path, *, order_learning_enabled: bool = True) -> Path:
    path.write_text(
        """
forecasting:
  order_learning:
    enabled: {enabled}
  garch:
    enabled: true
  samossa:
    enabled: true
  sarimax:
    enabled: false
  mssa_rl:
    enabled: false
  ensemble:
    enabled: true
  regime_detection:
    enabled: true
  monte_carlo:
    enabled: false
""".strip().format(enabled=str(order_learning_enabled).lower()),
        encoding="utf-8",
    )
    return path


def _empty_db(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def _synthetic_frame(tickers: list[str], points: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=points, freq="B")
    frames: list[pd.DataFrame] = []
    for idx, ticker in enumerate(tickers):
        base = 100.0 + idx
        close = pd.Series(range(points), index=dates, dtype=float) * 0.1 + base
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1000,
                "ticker": ticker,
            },
            index=dates,
        )
        frames.append(frame)
    return pd.concat(frames).sort_index()


def test_iter_train_lengths_stays_bounded_and_end_anchored() -> None:
    lengths = mod._iter_train_lengths(261, min_train_size=180, train_step=30, max_windows=3)

    assert lengths == [201, 231, 261]


def test_replay_order_learner_cache_reports_actual_vs_restored(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_forecasting_config(tmp_path / "forecasting.yml")
    db_path = _empty_db(tmp_path / "orders.db")
    market_data = _synthetic_frame(["AAPL"])

    class FakeExtractor:
        def extract_ohlcv(self, tickers, start_date, end_date):
            assert tickers == ["AAPL"]
            assert start_date <= end_date
            return market_data

    class FakeLearner:
        calls = 0

        def __init__(self, db_path: str, config: dict | None = None) -> None:
            self.db_path = db_path
            self.config = config or {}
            self._min_fits = 3

        def coverage_stats(self) -> dict[str, int]:
            FakeLearner.calls += 1
            if FakeLearner.calls == 1:
                return {"total_entries": 1, "qualified_entries": 0}
            return {"total_entries": 3, "qualified_entries": 1}

    class FakeForecaster:
        fit_calls = 0

        def __init__(self, config=None) -> None:
            self.config = config
            self._events: list[dict[str, object]] = []

        def fit(self, price_series, returns_series=None, ticker="", macro_context=None):
            FakeForecaster.fit_calls += 1
            assert ticker == "AAPL"
            if FakeForecaster.fit_calls == 1:
                self._events = [
                    {"model": "GARCH", "phase": "fit_complete", "restored": False},
                    {"model": "SAMOSSA", "phase": "fit_complete", "restored": False},
                ]
            else:
                self._events = [
                    {"model": "GARCH", "phase": "fit_complete", "restored": True},
                ]
            return self

        def get_component_summaries(self):
            return {"events": list(self._events)}

    snapshots = [
        {
            ("AAPL", "GARCH", "__none__", '{"p":1,"q":1}'): {
                "ticker": "AAPL",
                "model_type": "GARCH",
                "regime": "__none__",
                "order_params": '{"p":1,"q":1}',
                "n_fits": 2,
                "best_aic": 90.0,
                "last_used": "2026-02-28",
            }
        },
        {
            ("AAPL", "GARCH", "__none__", '{"p":1,"q":1}'): {
                "ticker": "AAPL",
                "model_type": "GARCH",
                "regime": "__none__",
                "order_params": '{"p":1,"q":1}',
                "n_fits": 3,
                "best_aic": 90.0,
                "last_used": "2026-02-28",
            }
        },
    ]

    def fake_snapshot(db_path: Path, tickers: list[str]):
        assert tickers == ["AAPL"]
        return snapshots.pop(0)

    monkeypatch.setattr(mod, "SyntheticExtractor", lambda: FakeExtractor())
    monkeypatch.setattr(mod, "OrderLearner", FakeLearner)
    monkeypatch.setattr(mod, "TimeSeriesForecaster", FakeForecaster)
    monkeypatch.setattr(mod, "_read_cache_snapshot", fake_snapshot)

    result = mod.replay_order_learner_cache(
        tickers=["AAPL"],
        db_path=db_path,
        forecasting_config_path=cfg_path,
        replays=1,
        lookback_days=365,
        min_train_size=180,
        train_step=30,
        max_train_windows=2,
    )

    assert result["status"] == "PASS"
    assert result["actual_fit_count"] == 2
    assert result["restored_fit_count"] == 1
    assert result["coverage_delta"]["total_entries"] == 2
    assert result["coverage_delta"]["qualified_entries"] == 1
    assert result["cache_evidence"]["n_fits_delta"] == 1
    assert result["cache_evidence"]["by_model_type"]["GARCH"]["actual_without_cache_write"] == 0
    assert result["cache_evidence"]["by_model_type"]["SAMOSSA_ARIMA"]["actual_without_cache_write"] == 1
    assert result["warnings"] == ["SAMOSSA_ARIMA: 1 actual fit(s) produced no cache write"]
    assert len(result["windows"]) == 2


def test_main_json_emits_error_when_order_learning_disabled(tmp_path: Path, capsys) -> None:
    cfg_path = _write_forecasting_config(tmp_path / "forecasting.yml", order_learning_enabled=False)
    db_path = _empty_db(tmp_path / "orders.db")

    exit_code = mod.main(
        [
            "--tickers",
            "AAPL",
            "--db",
            str(db_path),
            "--forecasting-config",
            str(cfg_path),
            "--json",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "Order learning is disabled" in payload["error"]
