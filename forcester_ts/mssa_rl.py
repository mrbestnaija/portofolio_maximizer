"""
Hybrid Multi-Singular Spectrum Analysis (mSSA) with reinforcement learning
style change-point detection.

The implementation is intentionally lightweight: it focuses on producing
diagnostics that can replace LLM-driven analytics in monitoring dashboards.
"""

from __future__ import annotations

import json
import logging
import importlib.util
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import svd
from pandas.tseries.frequencies import to_offset

from ._freq_compat import normalize_freq
from .metrics import compute_regression_metrics

try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # Python 3.10 fallback
    UTC = timezone.utc

cp = None
_CUPY_AVAILABLE = False
_CUPY_TRIED = False


def _preload_cuda_libraries() -> None:
    """Preload CUDA shared libraries shipped via pip (best-effort).

    In WSL, CuPy may import successfully but later fail to dlopen CUDA libs
    (e.g. `libnvrtc.so.12`) because the dynamic loader cannot see the packaged
    `site-packages/nvidia/**/lib/` directories. Preloading by absolute path via
    `ctypes.CDLL(..., RTLD_GLOBAL)` makes them available for CuPy's subsequent
    loads.
    """
    if not sys.platform.startswith("linux"):
        return

    try:  # pragma: no cover - optional platform behaviour
        import ctypes
    except Exception:
        return

    rtld_global = getattr(ctypes, "RTLD_GLOBAL", None)
    if rtld_global is None:
        return

    # Load order matters: nvrtc -> cudart -> cusolver.
    candidates = (
        ("nvidia.cuda_nvrtc", "libnvrtc.so.12"),
        ("nvidia.cuda_runtime", "libcudart.so.12"),
        ("nvidia.cusolver", "libcusolver.so.11"),
    )

    for module_name, library_name in candidates:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                continue

            base_path = None
            if spec.submodule_search_locations:
                base_path = Path(list(spec.submodule_search_locations)[0])
            elif spec.origin:
                base_path = Path(spec.origin).parent

            if base_path is None:
                continue

            lib_path = base_path / "lib" / library_name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=rtld_global)
        except Exception:
            continue


def _load_cupy() -> bool:
    """Best-effort CuPy import (optional); caches the result for the process."""
    global cp, _CUPY_AVAILABLE, _CUPY_TRIED
    if _CUPY_TRIED:
        return _CUPY_AVAILABLE
    _CUPY_TRIED = True
    try:  # Optional GPU acceleration via CuPy
        _preload_cuda_libraries()

        import cupy as _cp  # type: ignore

        try:
            if int(_cp.cuda.runtime.getDeviceCount()) < 1:
                raise RuntimeError("No CUDA devices detected by CuPy")

            # Minimal self-test to ensure linalg backends can be loaded.
            test = _cp.eye(2, dtype=_cp.float32)
            _cp.linalg.svd(test, full_matrices=False)
        except Exception as exc:
            cp = None
            _CUPY_AVAILABLE = False
            logger.debug("MSSARL: CuPy unavailable; falling back to CPU (%s).", exc)
            return _CUPY_AVAILABLE

        cp = _cp
        _CUPY_AVAILABLE = True
    except Exception:  # pragma: no cover - optional dependency
        cp = None
        _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE

logger = logging.getLogger(__name__)
DEFAULT_MSSA_POLICY_PATH = "models/mssa_rl_policy.v1.json"
MSSA_POLICY_SCHEMA_VERSION = 1
MSSA_POLICY_VERSION = "offline_policy_v1"


@dataclass
class MSSARLConfig:
    window_length: int = 30
    rank: Optional[int] = None
    change_point_threshold: float = 4.0   # Phase 7.10b value: 4.0 σ; reverted from 10.0 (2026-04-04)
    forecast_horizon: int = 10
    use_gpu: bool = False
    # PHASE 7.3 FIX: Accept additional params from YAML config
    min_series_length: int = 150
    max_forecast_steps: int = 30
    # Phase 8.3: action selection is driven by a frozen offline policy artifact.
    use_q_strategy_selection: bool = True
    rank_policy: str = "action_cutoffs"
    action_rank_cutoffs: Dict[int, float] = None  # type: ignore[assignment]
    policy_seed: int = 7
    policy_artifact_path: str = DEFAULT_MSSA_POLICY_PATH
    # MSSA-RL is fail-closed for most error states: forecast() raises ValueError when
    # policy_status is not "ready" or "insufficient_support".  The one exception is
    # insufficient_support (P3-B design): states with too few training samples return
    # neutral action 1 (HOLD) rather than raising, to prevent cascading forecast failures
    # on cold-start states.  See _select_action() lines ~786–796 for the neutral path.
    min_policy_state_support: int = 5
    reward_horizon: int = 5
    # Deprecated compatibility placeholders kept so older config/scripts do not crash.
    reward_mode: Optional[str] = None
    q_learning_alpha: float = 0.3
    q_learning_gamma: float = 0.85
    q_learning_epsilon: float = 0.1
    n_training_epochs: int = 15
    epsilon_start: float = 0.5


@dataclass(frozen=True)
class MSSAOfflinePolicyTrainingConfig:
    reward_horizon: int = 5
    min_train_size: int = 150
    step_size: int = 5
    max_windows_per_series: Optional[int] = None
    reward_clip: float = 1.0
    policy_source: str = "offline_trainer"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_mssa_policy_path(path_value: Optional[str]) -> Path:
    env_override = os.environ.get("PMX_MSSA_POLICY_ARTIFACT_PATH", "").strip()
    raw = (env_override or path_value or DEFAULT_MSSA_POLICY_PATH).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _state_key(state: int) -> str:
    return str(int(state))


def _action_key(action: int) -> str:
    return str(int(action))


def _flatten_policy_action_values(
    action_values_by_state: Dict[int, Dict[int, float]],
) -> Dict[Tuple[int, int], float]:
    flattened: Dict[Tuple[int, int], float] = {}
    for state, values in action_values_by_state.items():
        for action, score in values.items():
            flattened[(int(state), int(action))] = float(score)
    return flattened


def _coerce_policy_artifact(raw: Dict[str, Any]) -> Dict[str, Any]:
    required_top_level = {
        "schema_version",
        "policy_version",
        "trained_at_utc",
        "policy_source",
        "config",
        "states",
        "training_metadata",
        "validation_metrics",
    }
    missing = sorted(required_top_level.difference(raw))
    if missing:
        raise ValueError(f"missing policy artifact keys: {missing}")

    config = raw.get("config")
    states = raw.get("states")
    if not isinstance(config, dict) or not isinstance(states, dict):
        raise ValueError("policy artifact config/states must be dicts")

    parsed_states: Dict[int, Dict[str, Any]] = {}
    for state_key, state_payload in states.items():
        try:
            state_int = int(state_key)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid state key {state_key!r}") from exc
        if not isinstance(state_payload, dict):
            raise ValueError(f"state payload for {state_key!r} must be a dict")

        action_values = state_payload.get("action_values")
        support = state_payload.get("support")
        best_action = state_payload.get("best_action")
        margin = state_payload.get("action_value_margin")
        if not isinstance(action_values, dict) or not isinstance(support, dict):
            raise ValueError(f"state {state_key!r} missing action_values/support dicts")

        parsed_action_values = {
            int(action): float(score)
            for action, score in action_values.items()
        }
        parsed_support = {
            int(action): int(count)
            for action, count in support.items()
        }
        parsed_states[state_int] = {
            "action_values": parsed_action_values,
            "support": parsed_support,
            "best_action": int(best_action),
            "action_value_margin": None if margin is None else float(margin),
        }

    return {
        "schema_version": int(raw["schema_version"]),
        "policy_version": str(raw["policy_version"]),
        "trained_at_utc": str(raw["trained_at_utc"]),
        "policy_source": str(raw["policy_source"]),
        "config": {
            "window_length": int(config["window_length"]),
            "change_point_threshold": float(config["change_point_threshold"]),
            "reward_horizon": int(config["reward_horizon"]),
            "state_bins": [float(v) for v in config["state_bins"]],
            "action_rank_cutoffs": {
                int(action): float(cutoff)
                for action, cutoff in (config.get("action_rank_cutoffs") or {}).items()
            },
        },
        "states": parsed_states,
        "training_metadata": dict(raw["training_metadata"]),
        "validation_metrics": dict(raw["validation_metrics"]),
    }


def build_mssa_offline_policy_artifact(
    series_collection: Iterable[pd.Series],
    *,
    model_config: Optional[MSSARLConfig] = None,
    training_config: Optional[MSSAOfflinePolicyTrainingConfig] = None,
) -> Dict[str, Any]:
    cfg = model_config or MSSARLConfig()
    train_cfg = training_config or MSSAOfflinePolicyTrainingConfig(
        reward_horizon=int(cfg.reward_horizon or 5)
    )
    reward_horizon = max(1, int(train_cfg.reward_horizon or cfg.reward_horizon or 5))

    rewards: Dict[int, Dict[int, List[float]]] = {
        state: {action: [] for action in range(3)}
        for state in range(len(np.asarray([0, *MSSARLForecaster(cfg)._state_bins], dtype=float)))
    }
    directional_hits: Dict[int, Dict[int, List[float]]] = {
        state: {action: [] for action in range(3)}
        for state in rewards
    }
    rmse_rows: Dict[int, Dict[int, List[float]]] = {
        state: {action: [] for action in range(3)}
        for state in rewards
    }
    baseline_rows: List[float] = []
    used_windows = 0
    used_series = 0

    for series in series_collection:
        cleaned = series.dropna()
        if len(cleaned) < max(int(train_cfg.min_train_size), reward_horizon + cfg.window_length):
            continue
        used_series += 1
        windows_seen = 0
        start = int(train_cfg.min_train_size)
        stop = len(cleaned) - reward_horizon + 1
        for train_end in range(start, stop, max(1, int(train_cfg.step_size))):
            if train_cfg.max_windows_per_series is not None and windows_seen >= train_cfg.max_windows_per_series:
                break
            train = cleaned.iloc[:train_end]
            holdout = cleaned.iloc[train_end : train_end + reward_horizon]
            if len(holdout) != reward_horizon:
                continue

            trainer_model = MSSARLForecaster(
                MSSARLConfig(
                    window_length=int(cfg.window_length),
                    rank=cfg.rank,
                    change_point_threshold=float(cfg.change_point_threshold),
                    forecast_horizon=int(cfg.forecast_horizon),
                    use_gpu=False,
                    min_series_length=int(cfg.min_series_length),
                    max_forecast_steps=int(cfg.max_forecast_steps),
                    use_q_strategy_selection=False,
                    rank_policy=str(cfg.rank_policy),
                    action_rank_cutoffs=dict(cfg.action_rank_cutoffs or {}),
                    policy_seed=int(cfg.policy_seed),
                    policy_artifact_path="",
                    min_policy_state_support=int(cfg.min_policy_state_support),
                    reward_horizon=reward_horizon,
                )
            )
            trainer_model.fit(train)
            state = trainer_model._current_state
            if state is None:
                continue

            baseline = pd.Series(float(train.iloc[-1]), index=holdout.index, name="rw_baseline")
            baseline_metrics = compute_regression_metrics(holdout, baseline) or {}
            baseline_rmse = baseline_metrics.get("rmse")
            if baseline_rmse is None or not np.isfinite(float(baseline_rmse)) or float(baseline_rmse) <= 0.0:
                continue
            baseline_rmse = float(baseline_rmse)
            baseline_rows.append(baseline_rmse)

            windows_seen += 1
            used_windows += 1
            for action in range(3):
                forecast_payload = trainer_model._build_action_forecast(action, reward_horizon)
                metrics = compute_regression_metrics(holdout, forecast_payload["forecast"]) or {}
                rmse = metrics.get("rmse")
                if rmse is None or not np.isfinite(float(rmse)):
                    continue
                reward = (baseline_rmse - float(rmse)) / baseline_rmse
                reward = float(np.clip(reward, -abs(train_cfg.reward_clip), abs(train_cfg.reward_clip)))
                rewards[state][action].append(reward)
                rmse_rows[state][action].append(float(rmse))
                da = metrics.get("directional_accuracy")
                if da is not None and np.isfinite(float(da)):
                    directional_hits[state][action].append(float(da))

    state_payload: Dict[str, Any] = {}
    all_rewards: List[float] = []
    for state in sorted(rewards):
        action_values = {
            action: float(np.mean(values)) if values else -1.0
            for action, values in rewards[state].items()
        }
        support = {
            action: int(len(values))
            for action, values in rewards[state].items()
        }
        ranked_actions = sorted(action_values.items(), key=lambda item: item[1], reverse=True)
        best_action = int(ranked_actions[0][0])
        second_best = float(ranked_actions[1][1]) if len(ranked_actions) > 1 else float(ranked_actions[0][1])
        margin = float(ranked_actions[0][1] - second_best)
        all_rewards.extend(value for value in rewards[state][best_action] if np.isfinite(value))
        state_payload[_state_key(state)] = {
            "action_values": {
                _action_key(action): float(score)
                for action, score in action_values.items()
            },
            "support": {
                _action_key(action): int(count)
                for action, count in support.items()
            },
            "best_action": best_action,
            "action_value_margin": margin,
            "mean_rmse": {
                _action_key(action): (
                    float(np.mean(rmse_rows[state][action]))
                    if rmse_rows[state][action]
                    else None
                )
                for action in range(3)
            },
            "mean_directional_accuracy": {
                _action_key(action): (
                    float(np.mean(directional_hits[state][action]))
                    if directional_hits[state][action]
                    else None
                )
                for action in range(3)
            },
        }

    return {
        "schema_version": MSSA_POLICY_SCHEMA_VERSION,
        "policy_version": MSSA_POLICY_VERSION,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "policy_source": train_cfg.policy_source,
        "config": {
            "window_length": int(cfg.window_length),
            "change_point_threshold": float(cfg.change_point_threshold),
            "reward_horizon": reward_horizon,
            "state_bins": [float(v) for v in MSSARLForecaster(cfg)._state_bins],
            "action_rank_cutoffs": {
                _action_key(action): float(cutoff)
                for action, cutoff in (cfg.action_rank_cutoffs or {}).items()
            },
        },
        "states": state_payload,
        "training_metadata": {
            "baseline_model": "random_walk",
            "reward_definition": "clipped_relative_rmse_improvement_vs_random_walk",
            "aggregation": "mean_reward_per_state_action",
            "series_count": used_series,
            "window_count": used_windows,
            "min_train_size": int(train_cfg.min_train_size),
            "step_size": int(train_cfg.step_size),
        },
        "validation_metrics": {
            "overall": {
                "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "mean_baseline_rmse": float(np.mean(baseline_rows)) if baseline_rows else 0.0,
            }
        },
    }


def generate_mssa_policy_synthetic_curriculum() -> List[pd.Series]:
    """Deterministic curriculum for the bundled MSSA offline policy artifact."""
    curriculum: List[pd.Series] = []

    rng = np.random.default_rng(20260113)
    periods = 260
    idx = pd.date_range("2023-01-01", periods=periods, freq="D")
    t = np.arange(periods, dtype=float)
    structured = 100.0 + 0.6 * t + 3.0 * np.sin(2.0 * np.pi * t / 14.0) + rng.normal(0.0, 0.35, size=periods)
    curriculum.append(pd.Series(structured, index=idx, name="synthetic_structured"))

    for seed in (20260120, 20260133):
        walk_rng = np.random.default_rng(seed)
        returns = walk_rng.normal(0.0, 0.01, size=periods)
        prices = 100.0 * np.exp(np.cumsum(returns))
        curriculum.append(pd.Series(prices, index=idx, name=f"synthetic_random_walk_{seed}"))

    regime_rng = np.random.default_rng(7)
    n = 400
    regime_idx = pd.date_range("2020-01-01", periods=n, freq="D")
    q = n // 4
    seg1 = 100.0 + np.linspace(0.0, 8.0, q) + regime_rng.normal(0.0, 0.25, q)
    seg2 = seg1[-1] + regime_rng.normal(0.0, 4.5, q)
    seg3 = seg2[-1] + np.linspace(0.0, -6.0, q) + regime_rng.normal(0.0, 0.20, q)
    seg4 = seg3[-1] + regime_rng.normal(0.0, 5.0, n - 3 * q)
    curriculum.append(
        pd.Series(
            np.concatenate([seg1, seg2, seg3, seg4]),
            index=regime_idx,
            name="synthetic_regime_curriculum",
        )
    )

    return curriculum


class MSSARLForecaster:
    """
    Applies mSSA decomposition, change-point detection, and an offline action
    policy artifact.

    Live inference does not learn online. `fit()` computes reconstructions,
    diagnostics, and the current state. `forecast()` is only allowed when a
    valid offline policy artifact is loaded and the policy is ready.
    """

    def __init__(self, config: Optional[MSSARLConfig] = None) -> None:
        self.config = config or MSSARLConfig()
        if self.config.action_rank_cutoffs is None:
            self.config.action_rank_cutoffs = {0: 0.25, 1: 0.90, 2: 1.00}
        self._fitted = False
        self._action_values_by_state: Dict[int, Dict[int, float]] = {}
        self._support_by_state: Dict[int, Dict[int, int]] = {}
        self._best_action_by_state: Dict[int, int] = {}
        self._action_margin_by_state: Dict[int, float] = {}
        self._q_table: Dict[Tuple[int, int], float] = {}
        self._baseline_variance: float = 0.0
        self._last_index: Optional[pd.Timestamp] = None
        self._change_points: Optional[pd.DatetimeIndex] = None
        self._reconstruction: Optional[pd.Series] = None
        self._change_point_density: float = 0.0
        self._recent_change_point_days: Optional[int] = None
        self._freq_hint: Optional[str] = None
        self._last_observed_value: Optional[float] = None
        self._last_reconstruction_error: Optional[float] = None
        self._residual_diagnostics: Dict[str, Any] = {}
        self._use_gpu = bool(self.config.use_gpu and _load_cupy())
        if self.config.use_gpu and not self._use_gpu:
            # CuPy is optional; when unavailable we fall back to CPU silently.
            logger.debug("MSSARL: CuPy unavailable; falling back to CPU.")
        # Phase 8.1: per-action reconstruction matrices (set by _truncate_svd / fit)
        self._recon_matrix_by_action: Dict[int, np.ndarray] = {}
        self._reconstructions_by_action: Dict[int, pd.Series] = {}
        self._rank_by_action: Dict[int, int] = {}
        self._policy_version = MSSA_POLICY_VERSION
        self._policy_rng = np.random.default_rng(int(self.config.policy_seed))
        self._state_bins = np.array([0.8, 1.0, 1.2], dtype=float)
        self._current_state: Optional[int] = None
        self._last_q_state: Optional[int] = None
        self._last_active_action: Optional[int] = None
        self._last_active_rank: Optional[int] = None
        self._policy_status: str = "uninitialized"
        self._policy_source: Optional[str] = None
        self._policy_support: int = 0
        self._action_value_margin: Optional[float] = None
        self._policy_validation_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _construct_page_matrix(self, series: pd.Series) -> np.ndarray:
        L = self.config.window_length
        T = len(series)
        K = T - L + 1
        if K <= 0:
            raise ValueError("Window length longer than the series length")

        return np.column_stack([series.values[i : i + L] for i in range(K)])

    def _truncate_svd(self, trajectory: np.ndarray) -> np.ndarray:
        """Run SVD and return the 90%-variance reconstruction (standard action=1).

        Also stores per-action reconstructions in self._recon_matrix_by_action for
        Phase 8.1 component selection:
          action=0 (mean_revert)  -> top 25% cumulative variance (low-frequency, smooth)
          action=1 (hold)        -> top 90% cumulative variance (current behaviour)
          action=2 (trend_follow)-> all components (highest fidelity, retains noise)
        """
        if self._use_gpu:
            xp = cp
            traj = cp.asarray(trajectory)
            U, s, Vt = xp.linalg.svd(traj, full_matrices=False)
            _s_total_gpu = cp.sum(s)
            if float(_s_total_gpu) <= 0 or not np.isfinite(float(_s_total_gpu)):
                cumulative = np.ones(int(s.shape[0]))
            else:
                cumulative = cp.asnumpy(cp.cumsum(s) / _s_total_gpu)
            U_np = cp.asnumpy(U)
            s_np = cp.asnumpy(s)
            Vt_np = cp.asnumpy(Vt)
        else:
            xp = np
            U, s, Vt = svd(trajectory, full_matrices=False)
            _s_total = np.sum(s)
            if _s_total <= 0 or not np.isfinite(_s_total):
                # Degenerate trajectory (constant or near-zero series): use rank-1.
                cumulative = np.ones(len(s))
            else:
                cumulative = np.cumsum(s) / _s_total
            U_np, s_np, Vt_np = U, s, Vt

        max_rank = trajectory.shape[0]

        rank = self.config.rank
        if rank is None:
            rank = int(np.searchsorted(cumulative, 0.9)) + 1
        rank = max(1, min(int(rank), max_rank))
        self.config.rank = rank

        cutoffs = {
            int(action): float(cutoff)
            for action, cutoff in (self.config.action_rank_cutoffs or {}).items()
        }
        rank_policy = str(getattr(self.config, "rank_policy", "action_cutoffs")).lower()

        def _rank_for_cutoff(cutoff: float) -> int:
            bounded = float(np.clip(cutoff, 0.05, 1.0))
            resolved = int(np.searchsorted(cumulative, bounded)) + 1
            return max(1, min(resolved, max_rank))

        rank_25 = _rank_for_cutoff(cutoffs.get(0, 0.25))
        rank_90 = rank if rank_policy == "fixed_primary" else _rank_for_cutoff(cutoffs.get(1, 0.90))
        rank_all = _rank_for_cutoff(cutoffs.get(2, 1.00))
        # Enforce strict separation so actions 0 and 1 produce materially different
        # reconstructions (audit finding: rank_by_action={0:1,1:1,2:30} degeneracy).
        rank_90 = min(max(rank_25 + 1, rank_90), rank_all)
        self.config.rank = rank_90

        def _recon_for_rank(r: int) -> np.ndarray:
            mat = U_np[:, :r] @ np.diag(s_np[:r]) @ Vt_np[:r, :]
            return mat

        self._rank_by_action = {
            0: rank_25,
            1: rank_90,
            2: rank_all,
        }
        self._recon_matrix_by_action: Dict[int, np.ndarray] = {
            0: _recon_for_rank(rank_25),
            1: _recon_for_rank(rank_90),
            2: _recon_for_rank(rank_all),
        }

        return self._recon_matrix_by_action[1]

    def _cusum_change_points(self, residuals: pd.Series) -> pd.DatetimeIndex:
        cleaned = residuals.dropna()
        # Drop Inf/-Inf in addition to NaN so mean/std never propagates non-finite values.
        cleaned = cleaned[np.isfinite(cleaned)]
        if cleaned.empty:
            return pd.DatetimeIndex([])

        std = float(cleaned.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            return pd.DatetimeIndex([])

        # Guardrail: for ultra-low residual volatility, standardization can make
        # small noise look like frequent regime shifts. Floor the scale so
        # stable series do not over-trigger change points.
        std = max(std, 1.0)

        # Standardized one-sided CUSUM on residual mean shifts.
        # Reference: Page (1954) CUSUM tests for parameter changes.
        threshold = float(self.config.change_point_threshold)
        centered = (cleaned - float(cleaned.mean())) / (std + 1e-12)

        pos_sum = 0.0
        neg_sum = 0.0
        change_points = []
        for ts, value in zip(centered.index, centered.to_numpy()):
            pos_sum = max(0.0, pos_sum + float(value))
            neg_sum = min(0.0, neg_sum + float(value))
            if pos_sum > threshold or neg_sum < -threshold:
                change_points.append(ts)
                pos_sum = 0.0
                neg_sum = 0.0

        return pd.DatetimeIndex(change_points)

    def _build_q_table_alias(self) -> Dict[Tuple[int, int], float]:
        return _flatten_policy_action_values(self._action_values_by_state)

    def _compute_state_series(self, residuals: pd.Series) -> np.ndarray:
        baseline_variance = max(float(self._baseline_variance), 1e-12)
        variance_ratio = (
            residuals.rolling(window=self.config.window_length // 2, min_periods=5)
            .var()
            .fillna(baseline_variance)
            / baseline_variance
        )
        variance_ratio = variance_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        state_series = np.digitize(variance_ratio.to_numpy(dtype=float), bins=self._state_bins)
        self._current_state = int(state_series[-1]) if len(state_series) else None
        return state_series

    def _load_policy_artifact(self) -> None:
        self._action_values_by_state = {}
        self._support_by_state = {}
        self._best_action_by_state = {}
        self._action_margin_by_state = {}
        self._policy_support = 0
        self._action_value_margin = None
        self._policy_validation_metrics = {}
        self._q_table = {}

        if not getattr(self.config, "use_q_strategy_selection", True):
            self._policy_status = "disabled"
            self._policy_source = "disabled_by_config"
            return

        policy_path = resolve_mssa_policy_path(self.config.policy_artifact_path)
        self._policy_source = str(policy_path)
        if not policy_path.exists():
            self._policy_status = "missing_artifact"
            return

        try:
            raw = json.loads(policy_path.read_text(encoding="utf-8"))
            artifact = _coerce_policy_artifact(raw)
        except Exception as exc:
            logger.warning("MSSA-RL policy artifact invalid at %s: %s", policy_path, exc)
            self._policy_status = "invalid_artifact"
            return

        cfg = artifact["config"]
        expected_cutoffs = {
            int(action): float(cutoff)
            for action, cutoff in (self.config.action_rank_cutoffs or {}).items()
        }
        stale_reasons: List[str] = []
        if int(artifact["schema_version"]) != MSSA_POLICY_SCHEMA_VERSION:
            stale_reasons.append("schema_version")
        if str(artifact["policy_version"]) != MSSA_POLICY_VERSION:
            stale_reasons.append("policy_version")
        if int(cfg["window_length"]) != int(self.config.window_length):
            stale_reasons.append("window_length")
        if abs(float(cfg["change_point_threshold"]) - float(self.config.change_point_threshold)) > 1e-9:
            stale_reasons.append("change_point_threshold")
        if int(cfg["reward_horizon"]) != int(self.config.reward_horizon or 5):
            stale_reasons.append("reward_horizon")
        if [float(v) for v in cfg["state_bins"]] != [float(v) for v in self._state_bins]:
            stale_reasons.append("state_bins")
        if {
            int(action): float(cutoff)
            for action, cutoff in cfg["action_rank_cutoffs"].items()
        } != expected_cutoffs:
            stale_reasons.append("action_rank_cutoffs")
        if stale_reasons:
            logger.warning(
                "MSSA-RL policy artifact at %s is stale for current config (%s).",
                policy_path,
                ", ".join(stale_reasons),
            )
            self._policy_status = "stale_artifact"
            return

        self._action_values_by_state = {
            int(state): {
                int(action): float(score)
                for action, score in payload["action_values"].items()
            }
            for state, payload in artifact["states"].items()
        }
        self._support_by_state = {
            int(state): {
                int(action): int(count)
                for action, count in payload["support"].items()
            }
            for state, payload in artifact["states"].items()
        }
        self._best_action_by_state = {
            int(state): int(payload["best_action"])
            for state, payload in artifact["states"].items()
        }
        self._action_margin_by_state = {
            int(state): (
                None if payload["action_value_margin"] is None else float(payload["action_value_margin"])
            )
            for state, payload in artifact["states"].items()
        }
        self._policy_validation_metrics = dict(artifact.get("validation_metrics") or {})
        self._q_table = self._build_q_table_alias()

        if self._current_state is None:
            self._policy_status = "missing_state"
            return
        if self._current_state not in self._best_action_by_state:
            self._policy_status = "unsupported_state"
            return

        selected_action = int(self._best_action_by_state[self._current_state])
        selected_support = int(
            self._support_by_state.get(self._current_state, {}).get(selected_action, 0)
        )
        self._policy_support = selected_support
        self._action_value_margin = self._action_margin_by_state.get(self._current_state)
        if selected_support < int(self.config.min_policy_state_support):
            self._policy_status = "insufficient_support"
            return

        if not isinstance(getattr(self, "_residual_diagnostics", None), dict):
            self._policy_status = "degraded_residual_diagnostics"
            return
        for required_key in ("white_noise", "lb_pvalue", "jb_pvalue", "n"):
            if required_key not in self._residual_diagnostics:
                self._policy_status = "degraded_residual_diagnostics"
                return
        # SSA-based decomposition structurally produces autocorrelated residuals because
        # the SSA signal/noise separation leaves the un-decomposed component (noise) which
        # retains autocorrelation structure. A failing white-noise check is expected for
        # SSA models and is warn-only here (consistent with residual_diagnostics_rate_warn_only
        # in forecaster_monitoring.yml for SAMoSSA). The RL policy quality is gated by
        # policy_support, not by residual structure of the SSA decomposition.
        if self._residual_diagnostics.get("white_noise") is not True:
            logger.warning(
                "MSSA-RL residuals fail white-noise check (lb_p=%.4f, n=%d); "
                "SSA residuals are structurally autocorrelated — policy proceeds (warn-only).",
                self._residual_diagnostics.get("lb_pvalue") or 0.0,
                self._residual_diagnostics.get("n") or 0,
            )

        self._policy_status = "ready"

    def _resolve_active_action(self) -> int:
        if not getattr(self.config, "use_q_strategy_selection", True):
            self._policy_status = "disabled"
            self._policy_source = "disabled_by_config"
            self._policy_support = 0
            self._action_value_margin = None
            raise ValueError(
                "MSSA-RL offline policy not ready (disabled_by_config)"
            )

        if self._policy_status == "insufficient_support":
            # P3-B: low-support states return neutral action (1 = hold) rather than
            # failing entirely. A state with too few training samples should not drive
            # directional decisions — neutral is safer than random or no forecast.
            logger.debug(
                "MSSA-RL state %s support=%d < min=%d; returning neutral action",
                self._current_state,
                self._policy_support,
                int(self.config.min_policy_state_support),
            )
            return 1  # neutral

        if self._policy_status != "ready":
            # Fail-closed for all non-ready, non-insufficient_support states.
            raise ValueError(
                f"MSSA-RL offline policy not ready ({self._policy_status})"
            )

        assert self._current_state is not None  # guarded by _load_policy_artifact
        return int(self._best_action_by_state[self._current_state])

    def _build_action_forecast(self, action: int, steps: int) -> Dict[str, Any]:
        if self._reconstruction is None:
            raise ValueError("Model must be fitted before forecasting")

        action = int(action)
        active_recon = self._reconstructions_by_action.get(action)
        if active_recon is None or active_recon.empty:
            raise ValueError(f"No reconstruction available for action={action}")

        last_recon = float(active_recon.iloc[-1])
        last_obs = self._last_observed_value
        base_value = last_obs if last_obs is not None else last_recon
        recon_arr = active_recon.to_numpy(dtype=float)

        slope = 0.0
        if len(recon_arr) >= 3:
            k = min(self.config.window_length, len(recon_arr))
            slope_arr = recon_arr[-k:]
            if k >= 2:
                slope = float(np.polyfit(np.arange(k), slope_arr, deg=1)[0])

        effective_slope = slope
        if base_value != 0 and steps > 0:
            max_total_drift = abs(base_value) * 0.05
            max_slope_abs = max_total_drift / steps
            effective_slope = float(np.clip(effective_slope, -max_slope_abs, max_slope_abs))

        baseline_forecast = base_value + effective_slope * np.arange(1, steps + 1)

        if self._change_points is not None and len(self._change_points) > 0:
            apply_decay = False
            if self._recent_change_point_days is not None:
                if self._recent_change_point_days <= max(1, self.config.window_length // 4):
                    apply_decay = True
            if self._change_point_density < 0.1:
                apply_decay = False
            if apply_decay:
                decay = np.linspace(0.998, 0.99, num=steps)
                baseline_forecast = baseline_forecast * decay

        freq = (
            self._freq_hint
            or getattr(self._reconstruction.index, "freqstr", None)
            or getattr(self._reconstruction.index, "inferred_freq", None)
            or "D"
        )
        if self._last_index is not None:
            try:
                offset = to_offset(freq)
                start = self._last_index + offset
                future_index = pd.date_range(start=start, periods=steps, freq=freq)
            except Exception:
                future_index = pd.RangeIndex(start=0, stop=steps, step=1)
        else:
            future_index = pd.RangeIndex(start=0, stop=steps, step=1)

        forecast_series = pd.Series(baseline_forecast, index=future_index)
        noise = np.sqrt(self._baseline_variance)
        max_scale = np.sqrt(max(steps / 2, 1.0))
        horizon_scale = np.minimum(
            np.sqrt(np.arange(1, steps + 1, dtype=float)),
            max_scale,
        )
        ci_band = pd.Series(noise * horizon_scale, index=future_index)
        active_rank = self._rank_by_action.get(action, self.config.rank or len(recon_arr))
        q_state = self._current_state
        self._last_q_state = q_state
        self._last_active_action = action
        self._last_active_rank = active_rank

        return {
            "forecast": forecast_series,
            "lower_ci": forecast_series - ci_band,
            "upper_ci": forecast_series + ci_band,
            "change_points": self._change_points,
            "q_table_size": len(self._q_table),
            "baseline_variance": self._baseline_variance,
            "active_action": action,
            "active_rank": active_rank,
            "q_state": q_state,
            "policy_version": self._policy_version,
            "policy_status": self._policy_status,
            "policy_source": self._policy_source,
            "policy_support": self._policy_support,
            "action_value_margin": self._action_value_margin,
            "residual_diagnostics": self._residual_diagnostics,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "MSSARLForecaster":
        self._policy_rng = np.random.default_rng(int(self.config.policy_seed))
        self._action_values_by_state = {}
        self._support_by_state = {}
        self._best_action_by_state = {}
        self._action_margin_by_state = {}
        self._q_table = {}
        self._policy_status = "fitting"
        self._policy_source = None
        self._policy_support = 0
        self._action_value_margin = None
        self._current_state = None
        cleaned = series.dropna()
        if len(cleaned) < self.config.window_length + 5:
            raise ValueError("Series length insufficient for mSSA analysis")

        try:
            self._last_observed_value = float(cleaned.iloc[-1])
        except Exception:
            self._last_observed_value = None

        freq_hint = None
        try:
            freq_hint = series.attrs.get("_pm_freq_hint")
        except Exception:
            freq_hint = None
        self._freq_hint = normalize_freq(
            cleaned.index.freqstr or cleaned.index.inferred_freq or freq_hint
        )

        self._last_index = cleaned.index[-1]
        trajectory = self._construct_page_matrix(cleaned)
        # _truncate_svd populates self._recon_matrix_by_action as a side-effect (Phase 8.1).
        recon_matrix = self._truncate_svd(trajectory)

        def _diagonal_average(mat: np.ndarray, n: int) -> np.ndarray:
            """Anti-diagonal averaging of an SSA trajectory/reconstruction matrix."""
            out = np.zeros(n)
            cnt = np.zeros(n)
            rows, cols = mat.shape
            for i in range(rows):
                for j in range(cols):
                    out[i + j] += mat[i, j]
                    cnt[i + j] += 1
            cnt = np.where(cnt == 0, 1, cnt)
            return out / cnt

        n = len(cleaned)
        recon = _diagonal_average(recon_matrix, n)
        self._reconstruction = pd.Series(recon, index=cleaned.index)
        residuals = cleaned - self._reconstruction
        self._baseline_variance = float(residuals.var(ddof=1))
        try:
            self._last_reconstruction_error = float(abs(residuals.iloc[-1]))
        except Exception:
            self._last_reconstruction_error = None

        # Phase 8.2: run residual diagnostics and warn if not white noise.
        try:
            from .residual_diagnostics import run_residual_diagnostics  # pylint: disable=import-outside-toplevel
            self._residual_diagnostics = run_residual_diagnostics(residuals)
            if not self._residual_diagnostics.get("white_noise", True):
                logger.warning(
                    "MSSA-RL residuals fail white-noise check "
                    "(lb_p=%.4f, jb_p=%.4f, n=%d) — model may be mis-specified.",
                    self._residual_diagnostics.get("lb_pvalue") or 0.0,
                    self._residual_diagnostics.get("jb_pvalue") or 0.0,
                    self._residual_diagnostics.get("n", 0),
                )
        except Exception as _rd_exc:
            logger.debug("MSSARL residual_diagnostics failed: %s", _rd_exc)

        # Phase 8.1: precompute per-action reconstructed series for component selection.
        self._reconstructions_by_action: Dict[int, pd.Series] = {}
        for action_key, rmat in getattr(self, "_recon_matrix_by_action", {}).items():
            self._reconstructions_by_action[action_key] = pd.Series(
                _diagonal_average(rmat, n), index=cleaned.index
            )

        change_points = self._cusum_change_points(residuals)
        self._change_points = change_points
        self._change_point_density = len(change_points) / max(len(cleaned), 1)
        if len(change_points) > 0:
            last_cp = pd.to_datetime(change_points[-1])
            self._recent_change_point_days = abs(
                int((cleaned.index[-1] - last_cp).days)
            )
        else:
            self._recent_change_point_days = None

        self._compute_state_series(residuals)
        self._load_policy_artifact()

        logger.info(
            "MSSARL fit complete (window=%s, rank=%s, change_points=%s, "
            "policy_status=%s, q_state=%s, policy_support=%s)",
            self.config.window_length,
            self.config.rank,
            len(change_points),
            self._policy_status,
            self._current_state,
            self._policy_support,
        )
        self._fitted = True
        return self

    def forecast(self, steps: int) -> Dict[str, Any]:
        if not self._fitted or self._reconstruction is None:
            raise ValueError("Model must be fitted before forecasting")
        best_action = self._resolve_active_action()
        return self._build_action_forecast(best_action, steps)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "window_length": self.config.window_length,
            "rank": self.config.rank,
            "change_points": (
                self._change_points.to_pydatetime().tolist()
                if self._change_points is not None
                else []
            ),
            "q_table": dict(self._q_table),
            "baseline_variance": self._baseline_variance,
            "change_point_density": self._change_point_density,
            "recent_change_point_days": self._recent_change_point_days,
            "use_gpu": self._use_gpu,
            "rank_policy": self.config.rank_policy,
            "action_rank_cutoffs": dict(self.config.action_rank_cutoffs or {}),
            "rank_by_action": dict(self._rank_by_action),
            "policy_seed": int(self.config.policy_seed),
            "policy_version": self._policy_version,
            "policy_status": self._policy_status,
            "policy_source": self._policy_source,
            "policy_support": self._policy_support,
            "action_value_margin": self._action_value_margin,
            "action_values_by_state": {
                int(state): {
                    int(action): float(score)
                    for action, score in values.items()
                }
                for state, values in self._action_values_by_state.items()
            },
            "support_by_state": {
                int(state): {
                    int(action): int(count)
                    for action, count in values.items()
                }
                for state, values in self._support_by_state.items()
            },
            "q_state": self._last_q_state,
            "active_action": self._last_active_action,
            "active_rank": self._last_active_rank,
            "residual_diagnostics": dict(getattr(self, "_residual_diagnostics", {})),
        }
