from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forcester_ts.mssa_rl import MSSARLConfig, MSSARLForecaster


def _series(seed: int = 11, n: int = 180) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    signal = 100 + np.linspace(0.0, 6.0, n) + 2.0 * np.sin(np.arange(n) / 8.0)
    noise = rng.normal(0.0, 0.5, n)
    return pd.Series(signal + noise, index=idx)


def test_mssa_policy_forecast_exposes_contract_fields() -> None:
    forecaster = MSSARLForecaster(
        MSSARLConfig(
            window_length=30,
            rank_policy="action_cutoffs",
            action_rank_cutoffs={0: 0.25, 1: 0.9, 2: 1.0},
            policy_seed=7,
        )
    )
    forecaster.fit(_series())

    result = forecaster.forecast(steps=5)
    diagnostics = forecaster.get_diagnostics()

    assert result["policy_version"] == "bounded_rank_v2"
    assert result["active_action"] in {0, 1, 2}
    assert result["active_rank"] == diagnostics["rank_by_action"][result["active_action"]]
    assert result["q_state"] == diagnostics["q_state"]
    assert diagnostics["policy_seed"] == 7


def test_mssa_policy_is_deterministic_for_same_seed_and_series() -> None:
    config = MSSARLConfig(
        window_length=30,
        rank_policy="action_cutoffs",
        action_rank_cutoffs={0: 0.25, 1: 0.9, 2: 1.0},
        policy_seed=7,
    )
    series = _series(seed=19)

    first = MSSARLForecaster(config)
    second = MSSARLForecaster(
        MSSARLConfig(
            window_length=30,
            rank_policy="action_cutoffs",
            action_rank_cutoffs={0: 0.25, 1: 0.9, 2: 1.0},
            policy_seed=7,
        )
    )

    first.fit(series)
    second.fit(series)
    first_result = first.forecast(steps=5)
    second_result = second.forecast(steps=5)

    assert first_result["active_action"] == second_result["active_action"]
    assert first_result["active_rank"] == second_result["active_rank"]
    assert first_result["q_state"] == second_result["q_state"]
    assert first.get_diagnostics()["rank_by_action"][0] <= first.get_diagnostics()["rank_by_action"][1]
    assert first.get_diagnostics()["rank_by_action"][1] <= first.get_diagnostics()["rank_by_action"][2]


def test_mssa_q_learning_bootstraps_from_next_state_not_action() -> None:
    forecaster = MSSARLForecaster(
        MSSARLConfig(
            q_learning_alpha=0.5,
            q_learning_gamma=0.8,
            q_learning_epsilon=0.0,
        )
    )
    forecaster._q_table = {
        (0, 2): 5.0,
        (2, 0): 0.1,
        (2, 1): 0.4,
        (2, 2): 0.9,
    }

    forecaster._update_q_table(
        variance_ratio=1.2,
        state=1,
        action=0,
        next_state=2,
        realized_return=0.02,
    )

    assert forecaster._q_table[(1, 0)] == pytest.approx(0.35)


def test_mssa_fit_resets_q_table_between_runs() -> None:
    forecaster = MSSARLForecaster(
        MSSARLConfig(
            window_length=30,
            rank_policy="action_cutoffs",
            action_rank_cutoffs={0: 0.25, 1: 0.9, 2: 1.0},
            policy_seed=7,
        )
    )
    series = _series(seed=23)

    forecaster.fit(series)
    forecaster._q_table[(99, 99)] = 123.0
    forecaster.fit(series)

    assert (99, 99) not in forecaster.get_diagnostics()["q_table"]
