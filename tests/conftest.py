"""
Shared pytest fixtures for repository-wide test hygiene.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # Python 3.10 fallback
    UTC = timezone.utc

# Guard import-time side effects from test modules that instantiate forecasters
# before fixtures execute.
os.environ["TS_FORECAST_AUDIT_DIR"] = ""


@pytest.fixture(autouse=True)
def _disable_forecast_audit_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevent tests from writing into logs/forecast_audits by default.

    Production gate evidence should come from runtime pipelines, not unit/integration
    test executions. Tests that need a specific audit dir can still override this.
    """
    monkeypatch.setenv("TS_FORECAST_AUDIT_DIR", "")


def _default_mssa_policy_artifact(
    *,
    window_length: int = 30,
    change_point_threshold: float = 4.0,
    reward_horizon: int = 5,
    action_rank_cutoffs: dict[int, float] | None = None,
    states: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cutoffs = action_rank_cutoffs or {0: 0.25, 1: 0.90, 2: 1.00}
    if states is None:
        states = {
            state: {
                "action_values": {0: 0.05, 1: 0.20 + state * 0.01, 2: 0.02},
                "support": {0: 7, 1: 9, 2: 6},
                "best_action": 1,
                "action_value_margin": 0.15,
            }
            for state in range(4)
        }

    return {
        "schema_version": 1,
        "policy_version": "offline_policy_v1",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "policy_source": "pytest_fixture",
        "config": {
            "window_length": int(window_length),
            "change_point_threshold": float(change_point_threshold),
            "reward_horizon": int(reward_horizon),
            "state_bins": [0.8, 1.0, 1.2],
            "action_rank_cutoffs": {
                str(action): float(cutoff) for action, cutoff in cutoffs.items()
            },
        },
        "states": {
            str(state): {
                "action_values": {
                    str(action): float(score)
                    for action, score in payload["action_values"].items()
                },
                "support": {
                    str(action): int(count)
                    for action, count in payload["support"].items()
                },
                "best_action": int(payload["best_action"]),
                "action_value_margin": float(payload["action_value_margin"]),
            }
            for state, payload in states.items()
        },
        "training_metadata": {
            "baseline_model": "random_walk",
            "reward_definition": "clipped_relative_rmse_improvement_vs_random_walk",
            "aggregation": "mean_reward_per_state_action",
            "series_count": 4,
            "window_count": 32,
            "min_train_size": 150,
            "step_size": 5,
        },
        "validation_metrics": {
            "overall": {
                "mean_reward": 0.12,
                "mean_baseline_rmse": 1.0,
            }
        },
    }


@pytest.fixture
def mssa_policy_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[Path, dict[str, Any]]]:
    def _write_policy(
        *,
        filename: str = "mssa_rl_policy.v1.json",
        set_env: bool = True,
        artifact: dict[str, Any] | None = None,
        window_length: int = 30,
        change_point_threshold: float = 4.0,
        reward_horizon: int = 5,
        action_rank_cutoffs: dict[int, float] | None = None,
        states: dict[int, dict[str, Any]] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        payload = artifact or _default_mssa_policy_artifact(
            window_length=window_length,
            change_point_threshold=change_point_threshold,
            reward_horizon=reward_horizon,
            action_rank_cutoffs=action_rank_cutoffs,
            states=states,
        )
        path = tmp_path / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if set_env:
            monkeypatch.setenv("PMX_MSSA_POLICY_ARTIFACT_PATH", str(path))
        return path, payload

    return _write_policy


@pytest.fixture
def force_mssa_ready_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., dict[str, Any]]:
    def _apply(
        *,
        white_noise: bool = True,
        lb_pvalue: float | None = 0.45,
        jb_pvalue: float | None = 0.55,
        n: int = 180,
    ) -> dict[str, Any]:
        payload = {
            "white_noise": bool(white_noise),
            "lb_pvalue": lb_pvalue,
            "jb_pvalue": jb_pvalue,
            "n": int(n),
        }

        from forcester_ts import residual_diagnostics

        def _fake_run_residual_diagnostics(_residuals: Any) -> dict[str, Any]:
            return dict(payload)

        monkeypatch.setattr(
            residual_diagnostics,
            "run_residual_diagnostics",
            _fake_run_residual_diagnostics,
        )
        return payload

    return _apply


@pytest.fixture
def mssa_ready_policy_env(
    mssa_policy_writer: Callable[..., tuple[Path, dict[str, Any]]],
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> Path:
    path, _artifact = mssa_policy_writer()
    force_mssa_ready_residuals()
    return path


@pytest.fixture
def mssa_real_artifact_env(
    monkeypatch: pytest.MonkeyPatch,
    force_mssa_ready_residuals: Callable[..., dict[str, Any]],
) -> Path:
    """Sets env var to the committed trained artifact. Skips if artifact absent."""
    artifact_path = Path(__file__).parent.parent / "models" / "mssa_rl_policy.v1.json"
    if not artifact_path.exists():
        pytest.skip("models/mssa_rl_policy.v1.json not present")
    monkeypatch.setenv("PMX_MSSA_POLICY_ARTIFACT_PATH", str(artifact_path))
    force_mssa_ready_residuals()
    return artifact_path
