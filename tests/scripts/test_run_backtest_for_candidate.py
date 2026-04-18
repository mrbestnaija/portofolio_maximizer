from __future__ import annotations

import json
from types import SimpleNamespace

import scripts.run_backtest_for_candidate as backtest_script


def test_run_backtest_for_candidate_emits_alpha_report(monkeypatch, capsys):
    captured = {}

    class FakeDB:
        def __init__(self, db_path: str):
            self.db_path = db_path

        def get_distinct_tickers(self, limit=None):
            return ["AAPL"]

        def close(self):
            captured["closed"] = True

    def fake_backtest_candidate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            total_profit=12.0,
            win_rate=0.6,
            profit_factor=1.5,
            max_drawdown=0.1,
            total_trades=4,
            total_return=0.12,
            alpha=0.03,
            information_ratio=1.7,
            beta=0.95,
            tracking_error=0.08,
            r_squared=0.6,
            benchmark_proxy="equal_weight_universe",
            benchmark_metrics_status="aligned",
            benchmark_observations=7,
            anti_barbell_ok=True,
            anti_barbell_reason="all anti-barbell checks passed",
            anti_barbell_evidence={"anti_barbell_ok": True},
            strategy_returns=None,
        )

    monkeypatch.setattr(backtest_script, "DatabaseManager", FakeDB)
    monkeypatch.setattr(backtest_script, "backtest_candidate", fake_backtest_candidate)

    backtest_script.main.callback(
        db_path="data/portfolio_maximizer.db",
        regime="default",
        lookback_days=14,
        candidate_json='{"ensemble_weight_sarimax": 0.7, "sizing_kelly_fraction_cap": 0.25}',
        quant_config_path="config/quant_success_config.yml",
        forecasting_config_path="config/forecasting_config.yml",
        verbose=False,
        prefer_gpu=False,
    )

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["benchmark_proxy"] == "equal_weight_universe"
    assert payload["benchmark_metrics_status"] == "aligned"
    assert payload["candidate_params"]["ensemble_weight_sarimax"] == 0.7
    assert payload["metrics"]["alpha"] == 0.03
    assert payload["metrics"]["information_ratio"] == 1.7
    assert payload["metrics"]["benchmark_observations"] == 7
    assert payload["metrics"]["anti_barbell_ok"] is True
    assert payload["metrics"]["anti_barbell_reason"] == "all anti-barbell checks passed"
    assert payload["metrics"]["strategy_returns_count"] == 0
    assert captured["tickers"] == ["AAPL"]
    assert captured["forecasting_config_path"] == str(backtest_script.ROOT_PATH / "config/forecasting_config.yml")
    assert captured["candidate_params"]["sizing_kelly_fraction_cap"] == 0.25
    assert captured["closed"] is True
