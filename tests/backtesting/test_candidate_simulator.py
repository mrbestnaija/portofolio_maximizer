from __future__ import annotations

from backtesting.candidate_simulator import (
    _build_candidate_forecaster_config,
    _extract_candidate_ensemble_weights,
    _summarize_candidate_anti_barbell_evidence,
)


def test_extract_candidate_ensemble_weights_normalizes_and_rejects_empty_vectors():
    weights, reason = _extract_candidate_ensemble_weights(
        {
            "ensemble_weight_sarimax": 2.0,
            "ensemble_weight_garch": 1.0,
            "ensemble_weight_samossa": 1.0,
            "ensemble_weight_mssa_rl": 0.0,
        }
    )

    assert reason is None
    assert weights is not None
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["sarimax"] == 0.5
    assert weights["garch"] == 0.25
    assert weights["samossa"] == 0.25
    assert "mssa_rl" not in weights

    empty_weights, empty_reason = _extract_candidate_ensemble_weights(
        {
            "ensemble_weight_sarimax": 0.0,
            "ensemble_weight_garch": 0.0,
            "ensemble_weight_samossa": 0.0,
            "ensemble_weight_mssa_rl": 0.0,
        }
    )
    assert empty_weights is None
    assert empty_reason == "invalid_ensemble_weight_vector"


def test_build_candidate_forecaster_config_disables_regime_overrides_for_candidate_vector():
    forecasting_cfg = {
        "sarimax": {"enabled": True, "max_p": 1},
        "garch": {"enabled": True, "p": 1, "q": 1},
        "samossa": {"enabled": True, "window_length": 30},
        "mssa_rl": {"enabled": True, "window_length": 30},
        "ensemble": {"enabled": True, "confidence_scaling": True, "candidate_weights": [{"sarimax": 1.0}]},
        "regime_detection": {"enabled": True, "lookback_window": 60},
        "order_learning": {"enabled": True},
        "monte_carlo": {"enabled": False},
    }

    config = _build_candidate_forecaster_config(
        forecasting_cfg=forecasting_cfg,
        forecast_horizon=5,
        candidate_weights={"sarimax": 0.7, "garch": 0.3},
    )

    assert config.forecast_horizon == 5
    assert config.ensemble_enabled is True
    assert config.regime_detection_enabled is False
    assert config.ensemble_kwargs["candidate_weights"] == [{"sarimax": 0.7, "garch": 0.3}]
    assert config.ensemble_kwargs["adaptive_candidate_weights"] == []
    assert config.sarimax_enabled is True
    assert config.garch_enabled is True


def test_summarize_candidate_anti_barbell_evidence_maps_specific_arguments():
    metrics = {
        "alpha": 0.03,
        "information_ratio": 1.40,
        "benchmark_proxy": "equal_weight_universe",
        "benchmark_metrics_status": "aligned",
        "benchmark_observations": 7,
        "omega_robustness_score": 0.62,
        "omega_monotonicity_ok": True,
        "omega_cliff_drop_ratio": 0.11,
        "omega_cliff_ok": True,
        "omega_ci_lower": 1.04,
        "omega_ci_upper": 1.58,
        "omega_ci_width": 0.54,
        "omega_right_tail_ok": True,
        "expected_shortfall_raw": -0.02,
        "expected_shortfall_to_edge": 0.9,
        "es_to_edge_bounded": True,
    }
    path_risk_records = [
        {
            "barbell_path_risk_ok": True,
            "path_risk_evidence": True,
            "path_risk_checks": {
                "roundtrip_cost_to_edge": True,
                "gap_risk_to_edge": True,
                "funding_to_edge": True,
                "liquidity_to_depth": None,
                "leverage": True,
            },
            "path_risk_reason": "path risk passed",
        }
    ]
    regime_labels = ["RANGE"] * 5 + ["TREND"] * 4 + ["CRISIS"] * 3

    summary = _summarize_candidate_anti_barbell_evidence(
        metrics=metrics,
        path_risk_records=path_risk_records,
        regime_labels=regime_labels,
        promotion_thresholds={
            "min_regime_realism_labeled_trades": 10,
            "min_regime_coverage_rate": 0.80,
            "max_regime_dominance_rate": 0.70,
            "min_unique_regimes": 2,
        },
    )

    assert summary["anti_barbell_ok"] is True
    assert summary["threshold_sensitivity"]["omega_monotonicity_ok"] is True
    assert summary["threshold_sensitivity"]["omega_cliff_ok"] is True
    assert summary["right_tail_confidence"]["omega_right_tail_ok"] is True
    assert summary["left_tail_containment"]["es_to_edge_bounded"] is True
    assert summary["path_risk"]["barbell_path_risk_ok"] is True
    assert summary["regime_realism"]["regime_realism_ok"] is True


def test_summarize_candidate_anti_barbell_evidence_fails_closed_when_path_risk_missing():
    metrics = {
        "alpha": 0.03,
        "information_ratio": 1.40,
        "benchmark_proxy": "equal_weight_universe",
        "benchmark_metrics_status": "aligned",
        "benchmark_observations": 7,
        "omega_robustness_score": 0.62,
        "omega_monotonicity_ok": True,
        "omega_cliff_drop_ratio": 0.11,
        "omega_cliff_ok": True,
        "omega_ci_lower": 1.04,
        "omega_ci_upper": 1.58,
        "omega_ci_width": 0.54,
        "omega_right_tail_ok": True,
        "expected_shortfall_raw": -0.02,
        "expected_shortfall_to_edge": 0.9,
        "es_to_edge_bounded": True,
    }
    summary = _summarize_candidate_anti_barbell_evidence(
        metrics=metrics,
        path_risk_records=[],
        regime_labels=["RANGE"] * 5 + ["TREND"] * 4 + ["CRISIS"] * 3,
        promotion_thresholds={
            "min_regime_realism_labeled_trades": 10,
            "min_regime_coverage_rate": 0.80,
            "max_regime_dominance_rate": 0.70,
            "min_unique_regimes": 2,
        },
    )

    assert summary["anti_barbell_ok"] is False
    assert "path_risk" in summary["anti_barbell_reason"]
