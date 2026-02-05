from __future__ import annotations

import json

import pytest

from scripts import run_barbell_pnl_evaluation as mod


def test_apply_barbell_confidence_scales_by_bucket() -> None:
    cfg = mod.BarbellConfig(
        enable_barbell_allocation=False,
        enable_barbell_validation=False,
        enable_antifragility_tests=False,
        safe_min=0.75,
        safe_max=0.95,
        risk_max=0.25,
        safe_symbols=["SHY"],
        core_symbols=["MSFT"],
        speculative_symbols=["BTC-USD"],
        core_max=0.20,
        core_max_per=0.10,
        spec_max=0.10,
        spec_max_per=0.05,
        risk_symbols=["MSFT", "BTC-USD"],
    )
    multipliers = {"safe": 1.0, "core": 0.2, "spec": 0.1}

    assert mod._apply_barbell_confidence(ticker="SHY", confidence=0.9, cfg=cfg, multipliers=multipliers) == pytest.approx(
        0.9
    )
    assert mod._apply_barbell_confidence(
        ticker="MSFT", confidence=0.9, cfg=cfg, multipliers=multipliers
    ) == pytest.approx(0.18)
    assert mod._apply_barbell_confidence(
        ticker="BTC-USD", confidence=0.9, cfg=cfg, multipliers=multipliers
    ) == pytest.approx(0.09)


def test_run_barbell_eval_report_shape(monkeypatch) -> None:
    # Avoid running the full walk-forward simulation in unit tests.
    def fake_sim(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("enable_barbell_sizing"):
            return {"total_return": 2.0, "profit_factor": 1.2, "win_rate": 0.55, "max_drawdown": 0.1, "total_trades": 40}
        return {"total_return": 1.0, "profit_factor": 1.1, "win_rate": 0.52, "max_drawdown": 0.12, "total_trades": 40}

    monkeypatch.setattr(mod, "_simulate_walk_forward", fake_sim)

    # Also avoid touching the filesystem for barbell.yml inside BarbellConfig.from_yaml.
    cfg = mod.BarbellConfig(
        enable_barbell_allocation=False,
        enable_barbell_validation=False,
        enable_antifragility_tests=False,
        safe_min=0.75,
        safe_max=0.95,
        risk_max=0.25,
        safe_symbols=["SHY"],
        core_symbols=["MSFT"],
        speculative_symbols=["BTC-USD"],
        core_max=0.20,
        core_max_per=0.10,
        spec_max=0.10,
        spec_max_per=0.05,
        risk_symbols=["MSFT", "BTC-USD"],
    )
    monkeypatch.setattr(mod.BarbellConfig, "from_yaml", classmethod(lambda cls: cfg))

    payload = mod.run_barbell_eval(
        db_path=":memory:",
        tickers=["SHY", "MSFT"],
        window=mod.Window(start_date="2024-01-01", end_date="2024-06-01"),
        initial_capital=10_000.0,
        forecast_horizon=5,
        history_bars=120,
        min_bars=60,
        step_days=5,
    )

    assert payload["metrics"]["ts_only"]["total_return"] == 1.0
    assert payload["metrics"]["barbell_sized"]["total_return"] == 2.0
    assert payload["metrics"]["delta"]["total_return"] == 1.0

    json.dumps(payload)
