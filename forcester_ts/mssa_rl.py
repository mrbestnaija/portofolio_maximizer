"""
Hybrid Multi-Singular Spectrum Analysis (mSSA) with reinforcement learning
style change-point detection.

The implementation is intentionally lightweight: it focuses on producing
diagnostics that can replace LLM-driven analytics in monitoring dashboards.
"""

from __future__ import annotations

import logging
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import svd
from pandas.tseries.frequencies import to_offset

from ._freq_compat import normalize_freq

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


@dataclass
class MSSARLConfig:
    window_length: int = 30
    rank: Optional[int] = None
    change_point_threshold: float = 4.0   # Phase 7.10b: raised from 3.5 (was 2.5); reduces false change-points
    q_learning_alpha: float = 0.3
    q_learning_gamma: float = 0.85
    q_learning_epsilon: float = 0.1
    forecast_horizon: int = 10
    use_gpu: bool = False
    # PHASE 7.3 FIX: Accept additional params from YAML config
    min_series_length: int = 150
    max_forecast_steps: int = 30
    # Phase 7.10b: Q-strategy selection wires Q-values into forecast direction
    use_q_strategy_selection: bool = True
    reward_mode: str = "directional_pnl"  # 'variance_reduction' (legacy) or 'directional_pnl'
    rank_policy: str = "action_cutoffs"
    action_rank_cutoffs: Dict[int, float] = None  # type: ignore[assignment]
    policy_seed: int = 7


class MSSARLForecaster:
    """
    Applies mSSA decomposition, simple CUSUM change-point detection, and a
    tabular Q-learning loop that rewards variance reduction. The reward signal
    is deliberately simple but provides a deterministic alternative to LLM
    reasoning for regime detection.
    """

    def __init__(self, config: Optional[MSSARLConfig] = None) -> None:
        self.config = config or MSSARLConfig()
        if self.config.action_rank_cutoffs is None:
            self.config.action_rank_cutoffs = {0: 0.25, 1: 0.90, 2: 1.00}
        self._fitted = False
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
        self._use_gpu = bool(self.config.use_gpu and _load_cupy())
        if self.config.use_gpu and not self._use_gpu:
            # CuPy is optional; when unavailable we fall back to CPU silently.
            logger.debug("MSSARL: CuPy unavailable; falling back to CPU.")
        # Phase 8.1: per-action reconstruction matrices (set by _truncate_svd / fit)
        self._recon_matrix_by_action: Dict[int, np.ndarray] = {}
        self._reconstructions_by_action: Dict[int, pd.Series] = {}
        self._rank_by_action: Dict[int, int] = {}
        self._policy_version = "bounded_rank_v2"
        self._policy_rng = np.random.default_rng(int(self.config.policy_seed))
        self._state_bins = np.array([0.8, 1.0, 1.2], dtype=float)
        self._last_q_state: Optional[int] = None
        self._last_active_action: Optional[int] = None
        self._last_active_rank: Optional[int] = None

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
        rank_90 = max(rank_25, min(rank_90, rank_all))
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

    def _update_q_table(
        self,
        variance_ratio: float,
        state: int,
        action: int,
        next_state: Optional[int] = None,
        realized_return: Optional[float] = None,
    ) -> None:
        key = (state, action)
        current_q = self._q_table.get(key, 0.0)

        reward_mode = str(getattr(self.config, "reward_mode", "directional_pnl")).lower()
        if reward_mode == "directional_pnl" and realized_return is not None:
            # Phase 7.10b: reward = sign(forecast_direction) * realized_return.
            # action 0=mean_revert, 1=hold, 2=trend_follow
            action_to_sign = {0: -1.0, 1: 0.0, 2: 1.0}
            forecast_sign = action_to_sign.get(action, 0.0)
            reward = forecast_sign * float(realized_return)
        else:
            # Legacy: variance reduction reward
            reward = 1.0 - variance_ratio

        best_future = 0.0
        if next_state is not None:
            best_future = max(
                (self._q_table.get((int(next_state), next_action), 0.0) for next_action in range(3)),
                default=0.0,
            )

        new_q = current_q + self.config.q_learning_alpha * (
            reward + self.config.q_learning_gamma * best_future - current_q
        )
        self._q_table[key] = new_q

    @staticmethod
    def _default_action_for_state(state: int) -> int:
        return min(2, max(0, int(state)))

    def _select_policy_action(self, state: int) -> int:
        if not getattr(self.config, "use_q_strategy_selection", True):
            return 1
        epsilon = float(np.clip(self.config.q_learning_epsilon, 0.0, 1.0))
        if epsilon > 0.0 and float(self._policy_rng.random()) < epsilon:
            return int(self._policy_rng.integers(0, 3))
        q_vals = {a: self._q_table.get((state, a), 0.0) for a in range(3)}
        if len(set(q_vals.values())) == 1:
            return self._default_action_for_state(state)
        return max(q_vals, key=q_vals.__getitem__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "MSSARLForecaster":
        self._policy_rng = np.random.default_rng(int(self.config.policy_seed))
        self._q_table = {}
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

        # Tabular Q-learning over pseudo states (variance buckets)
        baseline_variance = max(float(self._baseline_variance), 1e-12)
        variance_ratio = (
            residuals.rolling(window=self.config.window_length // 2, min_periods=5)
            .var()
            .fillna(baseline_variance)
            / baseline_variance
        )
        variance_ratio = variance_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        state_series = np.digitize(variance_ratio.to_numpy(dtype=float), bins=self._state_bins)

        # Phase 7.10b: realized returns for directional PnL reward signal.
        # Derived from the input price series (1-period pct change).
        realized_returns_arr = (
            cleaned.pct_change().fillna(0.0).reindex(variance_ratio.index).fillna(0.0).to_numpy(dtype=float)
        )

        for idx, ((state, ratio), realized_ret) in enumerate(
            zip(zip(state_series, variance_ratio.to_numpy(dtype=float)), realized_returns_arr)
        ):
            action = self._select_policy_action(int(state))
            next_state = int(state_series[idx + 1]) if idx + 1 < len(state_series) else None
            self._update_q_table(
                float(ratio),
                int(state),
                action,
                next_state=next_state,
                realized_return=float(realized_ret),
            )

        logger.info(
            "MSSARL fit complete (window=%s, rank=%s, change_points=%s)",
            self.config.window_length,
            self.config.rank,
            len(change_points),
        )
        self._fitted = True
        return self

    def forecast(self, steps: int) -> Dict[str, Any]:
        if not self._fitted or self._reconstruction is None:
            raise ValueError("Model must be fitted before forecasting")

        last_recon = float(self._reconstruction.iloc[-1])
        last_obs = self._last_observed_value
        base_value = last_obs if last_obs is not None else last_recon

        if (
            self._last_reconstruction_error is not None
            and self._baseline_variance is not None
            and np.isfinite(self._baseline_variance)
            and self._change_point_density < 0.005
        ):
            scale = float(np.sqrt(max(self._baseline_variance, 0.0)))
            if scale > 0 and self._last_reconstruction_error <= 1.5 * scale:
                base_value = 0.85 * base_value + 0.15 * last_recon

        # Phase 7.10b: Replace naive constant forecast with trend-adjusted forecast.
        # Compute slope from last window_length bars of reconstructed series.
        recon_arr = self._reconstruction.values if self._reconstruction is not None else np.array([base_value])

        # Phase 8.1: Q-strategy selection now controls component set (not just slope sign).
        # States: {0=low_vol, 1=normal_vol, 2=high_vol}
        # Actions: {0=mean_revert, 1=hold, 2=trend_follow}
        best_action = 1  # default: standard 90%-variance components
        q_direction_weight = 0.0
        if getattr(self.config, "use_q_strategy_selection", True) and self._q_table:
            try:
                recent_var = float(np.var(recon_arr[-min(10, len(recon_arr)):]))
                var_ratio = recent_var / max(self._baseline_variance, 1e-12)
                current_state = int(np.digitize([var_ratio], bins=self._state_bins)[0])
                best_action = self._select_policy_action(current_state)
                # Legacy slope-direction signal (retained; blended at 0.5 weight below).
                action_to_sign = {0: -1.0, 1: 0.0, 2: 1.0}
                q_direction_weight = action_to_sign[best_action]
            except Exception as qe:
                logger.debug("Q-strategy selection error: %s", qe)
                current_state = None
        else:
            current_state = None

        # Phase 8.1: select the reconstruction that matches the Q-table action.
        # action=0 -> low-frequency components (25% variance) -> smoothed mean-revert signal
        # action=1 -> standard components (90% variance)      -> current behaviour
        # action=2 -> all components (100% variance)          -> high-fidelity trend signal
        active_recon = getattr(self, "_reconstructions_by_action", {}).get(best_action)
        if active_recon is not None and len(active_recon) > 0:
            recon_arr = active_recon.values
            logger.debug(
                "MSSARL forecast: action=%d selected reconstruction "
                "(n_components via variance cutoff), len=%d",
                best_action, len(recon_arr),
            )

        slope = 0.0
        if len(recon_arr) >= 3:
            k = min(self.config.window_length, len(recon_arr))
            slope_arr = recon_arr[-k:]
            if k >= 2:
                slope = float(np.polyfit(np.arange(k), slope_arr, deg=1)[0])

        # Blend slope from reconstruction with Q-direction signal (legacy 0.5-weight).
        # When Q-table has no data (new model), q_direction_weight == 0 -> pure slope.
        effective_slope = slope + q_direction_weight * abs(slope) * 0.5

        # Slope magnitude cap: cumulative drift over `steps` bars capped at 5% of base_value.
        # Prevents divergent long-horizon forecasts when slope is large (e.g. sharp trends).
        if base_value != 0 and steps > 0:
            max_total_drift = abs(base_value) * 0.05  # 5% max cumulative drift
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
        active_rank = self._rank_by_action.get(best_action, self.config.rank or len(recon_arr))
        q_state = current_state
        self._last_q_state = q_state
        self._last_active_action = best_action
        self._last_active_rank = active_rank

        # Scale CI to grow with sqrt(step+1): uncertainty accumulates over the horizon.
        # A flat ±noise CI based on in-sample baseline variance is too narrow at step N
        # and systematically inflates SNR for multi-step trades.
        # Cap at sqrt(horizon/2) to prevent runaway width for range-bound, low-volatility
        # periods where uncapped growth would push SNR below any practical threshold.
        # Use float division (steps / 2) not integer (steps // 2): integer division
        # rounds down, making the cap 1.0 for steps=3 instead of sqrt(1.5)=1.225.
        max_scale = np.sqrt(max(steps / 2, 1.0))
        horizon_scale = np.minimum(
            np.sqrt(np.arange(1, steps + 1, dtype=float)),
            max_scale,
        )
        ci_band = pd.Series(noise * horizon_scale, index=future_index)

        return {
            "forecast": forecast_series,
            "lower_ci": forecast_series - ci_band,
            "upper_ci": forecast_series + ci_band,
            "change_points": self._change_points,
            "q_table_size": len(self._q_table),
            "baseline_variance": self._baseline_variance,
            # Phase 8.1: which component set was used (0=mean_revert, 1=hold, 2=trend_follow)
            "active_action": best_action,
            "active_rank": active_rank,
            "q_state": q_state,
            "policy_version": self._policy_version,
            # Phase 8.2: residual diagnostics (Ljung-Box + Jarque-Bera)
            "residual_diagnostics": self._residual_diagnostics,
        }

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
            "q_state": self._last_q_state,
            "active_action": self._last_active_action,
            "active_rank": self._last_active_rank,
        }
