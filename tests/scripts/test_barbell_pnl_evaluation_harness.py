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
    calls = []

    def fake_sim(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("bootstrap_seed"))
        regime_summary = mod.summarize_regime_realism(["RANGE"] * 5 + ["TREND"] * 4 + ["CRISIS"] * 3)
        if kwargs.get("enable_barbell_sizing"):
            metrics = {
                "total_return": 2.0,
                "profit_factor": 1.2,
                "win_rate": 0.55,
                "max_drawdown": 0.1,
                "total_trades": 40,
                "losing_trades": 10,
                "expected_shortfall": -0.02,
                "omega_ratio": 1.7,
                "omega_robustness_score": 0.62,
                "omega_monotonicity_ok": True,
                "omega_cliff_drop_ratio": 0.11,
                "omega_cliff_ok": True,
                "omega_ci_lower": 1.04,
                "omega_ci_upper": 1.58,
                "omega_ci_width": 0.54,
                "omega_right_tail_ok": True,
                "expected_shortfall_to_edge": 0.9,
                "es_to_edge_bounded": True,
                "payoff_asymmetry": 1.65,
                "payoff_asymmetry_support_ok": True,
                "payoff_asymmetry_effective": 1.35,
                "winner_concentration_ratio": 0.45,
                "path_risk_trade_count": 30,
                "path_risk_ok_rate": 0.9,
                "barbell_path_risk_ok": True,
            }
            metrics.update(regime_summary)
            return metrics

        return {
            "total_return": 1.0,
            "profit_factor": 1.1,
            "win_rate": 0.52,
            "max_drawdown": 0.12,
            "total_trades": 40,
            "losing_trades": 10,
            "expected_shortfall": -0.03,
        }

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
    assert payload["metrics"]["barbell_sized"]["omega_ratio"] == pytest.approx(1.7)
    assert payload["metrics"]["barbell_sized"]["omega_cliff_ok"] is True
    assert payload["metrics"]["barbell_sized"]["omega_right_tail_ok"] is True
    assert payload["metrics"]["barbell_sized"]["es_to_edge_bounded"] is True
    assert payload["metrics"]["barbell_sized"]["regime_realism_ok"] is True
    assert len(calls) == 2
    assert calls[0] is not None
    assert calls[1] is not None
    assert calls[0] != calls[1]
    assert payload["promotion_decision"]["passed"] is True

    json.dumps(payload)


def test_augment_distribution_metrics_is_deterministic_with_seed() -> None:
    pnl_events = [
        ("2024-01-01", "AAPL", 120.0),
        ("2024-01-02", "MSFT", -80.0),
        ("2024-01-03", "AAPL", 90.0),
        ("2024-01-04", "MSFT", -60.0),
        ("2024-01-05", "AAPL", 140.0),
        ("2024-01-06", "MSFT", -110.0),
        ("2024-01-07", "AAPL", 100.0),
        ("2024-01-08", "MSFT", -50.0),
        ("2024-01-09", "AAPL", 130.0),
        ("2024-01-10", "MSFT", -70.0),
        ("2024-01-11", "AAPL", 110.0),
        ("2024-01-12", "MSFT", -40.0),
    ]
    regime_labels = ["RANGE"] * 5 + ["TREND"] * 4 + ["CRISIS"] * 3
    path_risk_records = [
        {
            "barbell_path_risk_ok": True,
            "roundtrip_cost_to_edge": 0.20,
            "gap_risk_to_edge": 0.15,
            "funding_to_edge": 0.05,
            "liquidity_to_depth": 0.40,
        },
        {
            "barbell_path_risk_ok": True,
            "roundtrip_cost_to_edge": 0.10,
            "gap_risk_to_edge": 0.12,
            "funding_to_edge": 0.04,
            "liquidity_to_depth": 0.35,
        },
    ]
    summary = {
        "total_return": 0.0,
        "total_profit": 0.0,
        "total_return_pct": 0.0,
        "profit_factor": 1.2,
        "win_rate": 0.5,
        "max_drawdown": 0.12,
        "total_trades": len(pnl_events),
        "losing_trades": 6,
        "expected_shortfall": -0.02,
    }

    first = mod._augment_distribution_metrics(
        summary=dict(summary),
        pnl_events=pnl_events,
        initial_capital=10_000.0,
        path_risk_records=path_risk_records,
        execution_drag_fractions=[0.001, 0.0015],
        regime_labels=regime_labels,
        bootstrap_seed=12345,
    )
    second = mod._augment_distribution_metrics(
        summary=dict(summary),
        pnl_events=pnl_events,
        initial_capital=10_000.0,
        path_risk_records=path_risk_records,
        execution_drag_fractions=[0.001, 0.0015],
        regime_labels=regime_labels,
        bootstrap_seed=12345,
    )

    assert first["omega_ci_lower"] == second["omega_ci_lower"]
    assert first["omega_ci_upper"] == second["omega_ci_upper"]
    assert first["omega_ci_width"] == second["omega_ci_width"]
    assert first["omega_right_tail_ok"] == second["omega_right_tail_ok"]
    assert first["omega_cliff_drop_ratio"] == second["omega_cliff_drop_ratio"]
    assert first["omega_cliff_ok"] == second["omega_cliff_ok"]
    assert first["expected_shortfall_to_edge"] == second["expected_shortfall_to_edge"]
    assert first["es_to_edge_bounded"] == second["es_to_edge_bounded"]
    assert first["regime_realism_ok"] is True
    assert first["regime_realism_labeled_trade_count"] == 12
    assert first["regime_realism_coverage_rate"] == pytest.approx(1.0)
    assert first["path_risk_trade_count"] == 2
    assert first["path_risk_ok_rate"] == pytest.approx(1.0)
    assert first["roundtrip_cost_to_edge_mean"] == pytest.approx(0.15)
