from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from forcester_ts.mssa_rl import MSSARLConfig, MSSARLForecaster


def _series(seed: int = 11, n: int = 220) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = 100.0 + np.linspace(0.0, 8.0, n)
    seasonal = 1.5 * np.sin(np.arange(n) / 9.0)
    noise = rng.normal(0.0, 0.45, n)
    return pd.Series(trend + seasonal + noise, index=idx, name="Close")


def _fit_default_forecaster(series: pd.Series | None = None) -> MSSARLForecaster:
    forecaster = MSSARLForecaster(
        MSSARLConfig(
            window_length=30,
            rank_policy="action_cutoffs",
            action_rank_cutoffs={0: 0.25, 1: 0.90, 2: 1.0},
            policy_seed=7,
        )
    )
    forecaster.fit(series if series is not None else _series())
    return forecaster


def test_mssa_policy_forecast_exposes_offline_contract_fields(
    mssa_ready_policy_env: Path,
) -> None:
    forecaster = _fit_default_forecaster()

    result = forecaster.forecast(steps=5)
    diagnostics = forecaster.get_diagnostics()

    assert result["policy_version"] == "offline_policy_v1"
    assert result["policy_status"] == "ready"
    assert result["policy_source"] == str(mssa_ready_policy_env)
    assert result["policy_support"] >= 5
    assert result["action_value_margin"] is not None
    assert result["active_action"] in {0, 1, 2}
    assert result["active_rank"] == diagnostics["rank_by_action"][result["active_action"]]
    assert result["q_state"] in {0, 1, 2, 3}
    assert result["q_table_size"] == len(diagnostics["q_table"])
    assert diagnostics["policy_status"] == "ready"
    assert diagnostics["policy_support"] == result["policy_support"]
    assert diagnostics["residual_diagnostics"]["white_noise"] is True


def test_mssa_policy_is_deterministic_for_same_artifact_and_series(
    mssa_ready_policy_env: Path,
) -> None:
    series = _series(seed=19)
    first = _fit_default_forecaster(series)
    second = _fit_default_forecaster(series)

    first_result = first.forecast(steps=5)
    second_result = second.forecast(steps=5)

    assert first_result["active_action"] == second_result["active_action"]
    assert first_result["active_rank"] == second_result["active_rank"]
    assert first_result["q_state"] == second_result["q_state"]
    assert first_result["policy_source"] == str(mssa_ready_policy_env)
    pd.testing.assert_series_equal(first_result["forecast"], second_result["forecast"])


def test_missing_policy_artifact_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMX_MSSA_POLICY_ARTIFACT_PATH", "missing/mssa_rl_policy.v1.json")
    forecaster = _fit_default_forecaster()

    diagnostics = forecaster.get_diagnostics()
    assert diagnostics["policy_status"] == "missing_artifact"
    with pytest.raises(ValueError, match="missing_artifact"):
        forecaster.forecast(steps=5)


def test_invalid_policy_artifact_fails_closed(
    mssa_policy_writer: Callable[..., tuple[Path, dict[str, Any]]],
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> None:
    force_mssa_ready_residuals()
    mssa_policy_writer(artifact={"schema_version": 1})

    forecaster = _fit_default_forecaster()

    assert forecaster.get_diagnostics()["policy_status"] == "invalid_artifact"
    with pytest.raises(ValueError, match="invalid_artifact"):
        forecaster.forecast(steps=5)


def test_stale_policy_artifact_fails_closed(
    mssa_policy_writer: Callable[..., tuple[Path, dict[str, Any]]],
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> None:
    force_mssa_ready_residuals()
    mssa_policy_writer(change_point_threshold=99.0)  # deliberately wrong — forces stale_artifact

    forecaster = _fit_default_forecaster()

    assert forecaster.get_diagnostics()["policy_status"] == "stale_artifact"
    with pytest.raises(ValueError, match="stale_artifact"):
        forecaster.forecast(steps=5)


def test_unsupported_state_policy_fails_closed(
    mssa_policy_writer: Callable[..., tuple[Path, dict[str, Any]]],
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> None:
    force_mssa_ready_residuals()
    mssa_policy_writer(
        states={
            9: {
                "action_values": {0: 0.1, 1: 0.2, 2: 0.0},
                "support": {0: 7, 1: 8, 2: 6},
                "best_action": 1,
                "action_value_margin": 0.1,
            }
        }
    )

    forecaster = _fit_default_forecaster()

    assert forecaster.get_diagnostics()["policy_status"] == "unsupported_state"
    with pytest.raises(ValueError, match="unsupported_state"):
        forecaster.forecast(steps=5)


def test_insufficient_support_policy_fails_closed(
    mssa_policy_writer: Callable[..., tuple[Path, dict[str, Any]]],
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> None:
    force_mssa_ready_residuals()
    mssa_policy_writer(
        states={
            state: {
                "action_values": {0: 0.05, 1: 0.20, 2: 0.02},
                "support": {0: 1, 1: 2, 2: 1},
                "best_action": 1,
                "action_value_margin": 0.15,
            }
            for state in range(4)
        }
    )

    forecaster = _fit_default_forecaster()

    assert forecaster.get_diagnostics()["policy_status"] == "insufficient_support"
    with pytest.raises(ValueError, match="insufficient_support"):
        forecaster.forecast(steps=5)


def test_degraded_residual_diagnostics_fail_closed(
    mssa_policy_writer: Callable[..., tuple[Path, dict[str, Any]]],
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> None:
    mssa_policy_writer()
    force_mssa_ready_residuals(white_noise=False, lb_pvalue=0.01, jb_pvalue=0.02)

    forecaster = _fit_default_forecaster()

    diagnostics = forecaster.get_diagnostics()
    assert diagnostics["policy_status"] == "degraded_residual_diagnostics"
    assert diagnostics["residual_diagnostics"]["white_noise"] is False
    with pytest.raises(ValueError, match="degraded_residual_diagnostics"):
        forecaster.forecast(steps=5)


def test_disabled_policy_selection_blocks_live_forecast() -> None:
    forecaster = MSSARLForecaster(
        MSSARLConfig(
            window_length=30,
            rank_policy="action_cutoffs",
            action_rank_cutoffs={0: 0.25, 1: 0.90, 2: 1.0},
            use_q_strategy_selection=False,
        )
    )
    forecaster.fit(_series())

    diagnostics = forecaster.get_diagnostics()
    assert diagnostics["policy_status"] == "disabled"
    assert diagnostics["policy_source"] == "disabled_by_config"
    with pytest.raises(ValueError, match="disabled_by_config"):
        forecaster.forecast(steps=5)


# ---------------------------------------------------------------------------
# Trained-artifact tests — exercise models/mssa_rl_policy.v1.json directly.
# These complement the fixture-based tests above which always use best_action=1
# for all states with thin support (7-9). The committed artifact has real
# action diversity (states 0-2: best_action=0, state 3: best_action=1).
# ---------------------------------------------------------------------------


def test_trained_artifact_loads_and_reaches_ready_status(
    mssa_real_artifact_env: Path,
) -> None:
    """Committed trained artifact passes all 7 validation gates and reaches ready."""
    forecaster = _fit_default_forecaster()
    result = forecaster.forecast(steps=5)
    diagnostics = forecaster.get_diagnostics()

    assert diagnostics["policy_status"] == "ready"
    assert result["policy_version"] == "offline_policy_v1"
    assert result["policy_source"] == str(mssa_real_artifact_env)
    # q_table is keyed by (state, action) tuples; count unique states.
    assert len({s for s, _ in diagnostics["q_table"]}) == 4
    # Current-state support must meet min_policy_state_support floor.
    assert result["policy_support"] >= 5


def test_trained_artifact_action_not_uniform_across_states(
    mssa_real_artifact_env: Path,
) -> None:
    """Trained artifact has action diversity; fixture artifact always returns best_action=1."""
    import json

    artifact = json.loads(mssa_real_artifact_env.read_text(encoding="utf-8"))
    best_actions = {
        int(state): int(payload["best_action"])
        for state, payload in artifact["states"].items()
    }
    action_set = set(best_actions.values())
    assert 0 in action_set, (
        f"trained artifact must have >=1 state with best_action=0; got {best_actions}"
    )
    assert 1 in action_set, (
        f"trained artifact must have >=1 state with best_action=1; got {best_actions}"
    )


def test_trained_artifact_stale_when_threshold_mismatched(
    monkeypatch: pytest.MonkeyPatch,
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> None:
    """Trained artifact (change_point_threshold=4.0) is stale when forecaster uses 99.0."""
    artifact_path = Path(__file__).parent.parent.parent / "models" / "mssa_rl_policy.v1.json"
    if not artifact_path.exists():
        pytest.skip("models/mssa_rl_policy.v1.json not present")
    monkeypatch.setenv("PMX_MSSA_POLICY_ARTIFACT_PATH", str(artifact_path))
    force_mssa_ready_residuals()

    forecaster = MSSARLForecaster(
        MSSARLConfig(
            window_length=30,
            rank_policy="action_cutoffs",
            action_rank_cutoffs={0: 0.25, 1: 0.90, 2: 1.0},
            change_point_threshold=99.0,
        )
    )
    forecaster.fit(_series())

    assert forecaster.get_diagnostics()["policy_status"] == "stale_artifact"
