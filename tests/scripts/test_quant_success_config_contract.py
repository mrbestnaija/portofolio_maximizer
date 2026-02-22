from pathlib import Path

import yaml


def test_quant_success_config_weight_schema_contract() -> None:
    config_path = Path("config/quant_success_config.yml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    qv = payload.get("quant_validation") or {}

    assert qv.get("scoring_mode") == "weighted"
    assert bool(qv.get("strict_weight_coverage")) is True

    pass_threshold = float(qv.get("pass_threshold", 0.0))
    assert 0.0 <= pass_threshold <= 1.0

    weights = qv.get("criterion_weights") or {}
    assert isinstance(weights, dict)

    required = {
        "expected_profit",
        "annual_return",
        "max_drawdown",
        "rmse_ratio_vs_baseline",
        "directional_accuracy",
    }
    assert required.issubset(set(weights.keys()))
    assert "rmse_ratio" not in weights

    total = sum(float(v) for v in weights.values())
    assert abs(total - 1.0) < 1e-6
