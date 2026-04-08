from pathlib import Path

import yaml


def test_quant_success_config_weight_schema_contract() -> None:
    config_path = Path("config/quant_success_config.yml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    qv = payload.get("quant_validation") or {}

    assert qv.get("objective_mode") == "domain_utility"
    assert qv.get("scoring_mode") == "domain_utility"
    assert bool(qv.get("strict_weight_coverage")) is True

    pass_threshold = float(qv.get("pass_threshold", 0.0))
    assert 0.0 <= pass_threshold <= 1.0

    weights = qv.get("utility_weights") or {}
    assert isinstance(weights, dict)

    required = {
        "expected_profit",
        "omega_ratio",
        "profit_factor",
        "terminal_directional_accuracy",
        "max_drawdown",
        "expected_shortfall",
    }
    assert required.issubset(set(weights.keys()))
    assert "criterion_weights" not in qv
    assert "directional_accuracy" not in weights

    total = sum(float(v) for v in weights.values())
    assert abs(total - 1.0) < 1e-6

    success = qv.get("success_criteria") or {}
    assert "min_terminal_directional_accuracy" in success
    assert "min_expected_shortfall" in success


def test_barbell_objective_docs_contract() -> None:
    policy_text = (
        "Barbell asymmetry is the primary economic objective. "
        "The system optimizes for asymmetric upside with bounded downside, "
        "not for symmetric textbook efficiency metrics."
    )

    canonical_doc = Path("Documentation/REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md")
    assert canonical_doc.exists()
    canonical_text = canonical_doc.read_text(encoding="utf-8")
    assert policy_text in canonical_text
    for token in (
        "objective_mode: domain_utility",
        "GENUINE_PASS",
        "WARMUP_COVERED_PASS",
        "forecast_horizon_bars",
        'forecast_horizon_units="bars"',
        "expected_close_source",
        "utility_breakdown",
        "matrix_health",
    ):
        assert token in canonical_text

    link_only_docs = (
        Path("Documentation/DOCS_INDEX.md"),
        Path("Documentation/DOCUMENTATION_INDEX.md"),
        Path("Documentation/CORE_PROJECT_DOCUMENTATION.md"),
        Path("README.md"),
        Path("Documentation/README.md"),
        Path("Documentation/PROJECT_STATUS.md"),
        Path("Documentation/NEXT_TO_DO_SEQUENCED.md"),
        Path("Documentation/METRICS_AND_EVALUATION.md"),
        Path("Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md"),
        Path("Documentation/QUANT_VALIDATION_MONITORING_POLICY.md"),
        Path("Documentation/QUANT_TIME_SERIES_STACK.md"),
        Path("Documentation/TIME_SERIES_FEATURE_AND_AUDIT_CONTRACT.md"),
        Path("Documentation/BARBELL_INTEGRATION_TODO.md"),
    )
    for path in link_only_docs:
        text = path.read_text(encoding="utf-8")
        assert "REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md" in text
