from __future__ import annotations

from pathlib import Path


def test_thresholds_match_capital_readiness_contract() -> None:
    from scripts.capital_readiness_check import (
        R3_MIN_PROFIT_FACTOR,
        R3_MIN_TRADES,
        R3_MIN_WIN_RATE,
        R4_MAX_BRIER,
    )
    from scripts.robustness_thresholds import threshold_map

    thresholds = threshold_map()
    assert thresholds["r3_min_trades"] == R3_MIN_TRADES
    assert thresholds["r3_min_win_rate"] == R3_MIN_WIN_RATE
    assert thresholds["r3_min_profit_factor"] == R3_MIN_PROFIT_FACTOR
    assert thresholds["r4_max_brier"] == R4_MAX_BRIER


def test_thresholds_include_source_paths_and_hashes() -> None:
    from scripts.robustness_thresholds import threshold_map

    thresholds = threshold_map()
    assert "source_paths" in thresholds
    assert "source_hashes" in thresholds
    assert thresholds["source_paths"]["capital_readiness_check"]
    assert thresholds["source_paths"]["forecaster_monitoring"]


def test_thresholds_include_pending_calibration_marker() -> None:
    from scripts.robustness_thresholds import (
        DOMAIN_OBJECTIVE_VERSION,
        PENDING_CALIBRATION,
        locate_pending_calibration_target,
        threshold_map,
    )

    thresholds = threshold_map()
    assert thresholds["pending_calibration"] == PENDING_CALIBRATION
    assert thresholds["domain_objective_version"] == DOMAIN_OBJECTIVE_VERSION == "v1.0.0"
    assert locate_pending_calibration_target() == {"min_signal_to_noise": PENDING_CALIBRATION}
    source = (Path(__file__).resolve().parents[2] / "scripts" / "robustness_thresholds.py").read_text(encoding="utf-8")
    assert "PENDING_CALIBRATION" in source
    assert PENDING_CALIBRATION in source


def test_take_profit_policy_floors_are_enforced(tmp_path: Path, monkeypatch) -> None:
    from scripts.robustness_thresholds import load_floored_thresholds, threshold_map

    monkeypatch.setenv("PMX_DISABLE_GATE_FLOORS", "1")
    cfg_path = tmp_path / "take_profit_thresholds.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "signal_routing:",
                "  min_omega_ratio: 0.2",
                "  min_payoff_asymmetry: 1.0",
                "  min_take_profit_frequency_live: 0.01",
                "  min_target_amplitude_hit_rate: 0.02",
            ]
        ),
        encoding="utf-8",
    )

    floored = load_floored_thresholds(cfg_path)
    canonical = threshold_map()

    assert floored["min_omega_ratio"] == canonical["min_omega_ratio"] == 1.0
    assert floored["min_payoff_asymmetry"] == canonical["min_payoff_asymmetry"] == 2.0
    assert floored["min_take_profit_frequency_live"] == canonical["min_take_profit_frequency_live"] == 0.05
    assert floored["min_target_amplitude_hit_rate"] == canonical["min_target_amplitude_hit_rate"] == 0.10
    assert floored["system_objective"] == "TAKE_PROFIT_CAPTURE"
    assert floored["domain_objective_version"] == canonical["domain_objective_version"] == "v1.0.0"
    assert floored["gate_floor_bypass_active"] is False
    assert floored["floor_warnings"] == [
        "min_omega_ratio_raised_to_floor(0.2->1.0)",
        "min_payoff_asymmetry_raised_to_floor(1.0->2.0)",
        "min_take_profit_frequency_live_raised_to_floor(0.01->0.05)",
        "min_target_amplitude_hit_rate_raised_to_floor(0.02->0.1)",
    ]
